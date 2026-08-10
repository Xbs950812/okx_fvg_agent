# -*- coding: utf-8 -*-
"""
纸面交易引擎 (Paper Trading Engine) — 模拟建仓，实时盘验证策略盈亏。

背景
----
现有 dry-run 模式只伪造订单号（绝不触碰真实下单端点），但不跟踪虚拟持仓与盈亏。
本引擎补上这一环：用用户设定的虚拟余额 + 实时行情，完整模拟交易生命周期：

  1. 限价挂单等待回补成交 —— 与实盘"不追涨杀跌"哲学一致：
     做多限价单在 candle.low <= 入场价 时成交；做空在 candle.high >= 入场价 时成交。
     超时（limit_timeout_min，默认 15 分钟，与实盘限价单超时口径一致）未成交则取消。
  2. 持仓退出（优先级从高到低，同根 K 线 TP/SL 同时触发时按 SL 保守处理）：
     - 止损（含 0.5% 滑点，与 executor.execute_signal 的 _sl_slippage 一致）
     - 止盈
     - 动态 ROI（freqtrade minimal_roi 模式，与 risk.dynamic_roi 一致）
     - 最大持仓时长（risk.max_hold_hours）
  3. 手续费（maker/taker 可配置，默认 0.02%/0.05%）在成交/平仓时扣减。
  4. 状态持久化（原子写 JSON），重启不丢虚拟持仓与历史交易。

设计约束
--------
  - 零新第三方依赖（仅标准库）。
  - 不 import agent.py / strategy.py（无循环依赖）；复用 executor.calculate_position_size，
    保证仓位口径（风险比例→仓位价值→张数）与实盘 100% 一致。
  - 行情通过注入的 market_data_fn(inst_id) 获取，由调用方（agent）适配 OKX SDK / WS 缓存，
    本模块不直接依赖任何行情源，便于单测用假数据驱动。

用法（由 agent.py 集成）:
    engine = PaperTradingEngine(config)
    engine.set_market_data_provider(lambda inst: {...candles, mark...})
    engine.load()
    engine.update()            # 每轮主循环调用：成交/退出/更新UPL
    engine.open_position(signal, instrument_info, risk_cfg, equity)
    engine.close_position(inst_id, reason="reverse")
    positions = engine.to_positions_dict()   # 与 monitor_positions 输出结构一致
    equity = engine.get_equity()
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# 复用实盘仓位计算，确保张数与实盘口径一致（仅在支持时导入，失败则本地回退）
try:
    from executor import calculate_position_size
except Exception:  # pragma: no cover - 依赖缺失时本地兜底
    def calculate_position_size(  # type: ignore[misc]
        equity: float, entry_price: float, stop_loss: float, risk_pct: float,
        leverage: int, margin_pct: float, direction: str = "long",
        contract_value: float = 1.0, min_sz: float = 1.0, sz_precision: int = 0,
        sizing: str = "risk", enforce_risk_cap: bool = False,
    ) -> Tuple[float, float]:
        if equity <= 0 or entry_price <= 0 or stop_loss <= 0:
            return 0.0, 0.0
        if direction == "long" and stop_loss >= entry_price:
            return 0.0, 0.0
        if direction == "short" and stop_loss <= entry_price:
            return 0.0, 0.0
        stop_dist = abs(entry_price - stop_loss) / entry_price
        if stop_dist < 1e-10:
            return 0.0, 0.0
        if sizing == "margin":
            # 保证金驱动 (满仓模拟): 用满 margin_pct% 权益做保证金
            margin = equity * margin_pct / 100.0
            if margin <= 0:
                return 0.0, 0.0
            # 以损定量硬上限: 与 executor.calculate_position_size 口径一致
            if enforce_risk_cap and risk_pct > 0:
                max_margin_by_risk = equity * risk_pct / 100.0 / stop_dist
                if margin > max_margin_by_risk:
                    margin = max_margin_by_risk
            position_value = margin * leverage
        else:
            # 风险倒推 (默认)
            risk_amount = equity * risk_pct / 100.0
            position_value = risk_amount / stop_dist
            margin = position_value / leverage
            max_margin = equity * margin_pct / 100.0
            if margin > max_margin:
                margin = max_margin
                position_value = margin * leverage
        sz = position_value / (entry_price * contract_value)
        sz = math.floor(sz * (10 ** sz_precision)) / (10 ** sz_precision)
        if sz < min_sz:
            return 0.0, 0.0
        return sz, margin


logger = logging.getLogger(__name__)

_DEFAULT_MAKER_FEE = 0.0002   # OKX 永续 maker 常见费率 0.02%
_DEFAULT_TAKER_FEE = 0.0005   # OKX 永续 taker 常见费率 0.05%
_SL_SLIPPAGE = 0.005          # 止损滑点，与 executor.execute_signal 的 _sl_slippage 一致
_ATR_PERIOD = 14              # 纸面移动止损 ATR 周期，与实盘 _compute_atr_from_cache 一致


def _candle_attr(c, key: str):
    """兼容 dict 与 Candle 对象的 K 线字段访问。

    2026-08-09 修复: 行情源 _paper_market_data 返回 Candle dataclass 对象
    (strategy.candles_from_raw)，而 _atr14/_process_pending/_check_exit 用
    c["high"] 字典键访问 → TypeError 被静默捕获 → ATR 恒为 0 → 移动止损
    退化为"TP 距离 50% 才激活"的超保守阈值 (RAVE +3.4% 仍未激活 TS 实测)。
    """
    if isinstance(c, dict):
        return c.get(key)
    return getattr(c, key, None)


def _atr14(candles: List[dict]) -> float:
    """Wilder 平滑 ATR(14) — 与实盘 ATR 计算口径一致。

    数据不足时返回 0.0，调用方回退固定百分比追踪。
    """
    try:
        if not isinstance(candles, list) or len(candles) < _ATR_PERIOD + 1:
            return 0.0
        highs = [float(_candle_attr(c, "high")) for c in candles[-(_ATR_PERIOD + 1):]]
        lows = [float(_candle_attr(c, "low")) for c in candles[-(_ATR_PERIOD + 1):]]
        closes = [float(_candle_attr(c, "close")) for c in candles[-(_ATR_PERIOD + 1):]]
        trs = []
        for i in range(1, len(closes)):
            trs.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
        if len(trs) < _ATR_PERIOD:
            return 0.0
        atr = sum(trs[:_ATR_PERIOD]) / float(_ATR_PERIOD)
        for t in trs[_ATR_PERIOD:]:
            atr = (atr * (_ATR_PERIOD - 1) + t) / float(_ATR_PERIOD)
        return float(atr)
    except (TypeError, ValueError, KeyError, IndexError):
        return 0.0


@dataclass
class PaperPosition:
    """纸面持仓（含未成交的限价挂单，filled=False 表示挂单中）。"""
    inst_id: str
    side: str                 # "long" | "short"
    entry_px: float
    size: float               # 合约张数
    leverage: float
    margin: float             # 占用保证金（USDT）
    tp_px: float
    sl_px: float
    open_time: float
    ct_val: float = 0.01
    filled: bool = False      # False = 限价挂单等待回补
    entry_fee: float = 0.0
    signal_id: str = ""
    last_mark: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)
    trigger_px: float = 0.0   # 2026-08-10: conditional 触发单触发价 (0=普通限价单);
    # 价格触及触发价前不成交(不激活), 激活后按普通限价单逻辑等回补成交


class PaperTradingEngine:
    """虚拟余额 + 实时行情驱动的纸面交易引擎。

    与实盘最大一致性保证：
      - 开仓张数复用 executor.calculate_position_size（含方向校验/保证金上限/精度截断）
      - 限价成交等待价格回补（不追涨杀跌）
      - 止损含 0.5% 滑点、退出收 taker 费、入场收 maker 费
      - 退出优先级：SL > TP > 动态ROI > 时间
    """

    def __init__(self, config: dict):
        pcfg = config.get("paper", {}) if isinstance(config, dict) else {}
        rcfg = config.get("risk", {}) if isinstance(config, dict) else {}
        self.initial_balance: float = float(pcfg.get("balance", 1000.0))
        self.balance: float = self.initial_balance
        # 修复 2026-08-10: state_file 缺失时默认 None(不落盘) —
        # 此前默认 "paper_state.json" 指向 cwd, 测试/误用未传 state_file 时
        # 会把假仓位直接覆盖真实纸面账户(曾致 BTC 幻影仓污染 27.19 余额账户)。
        self.state_file: Optional[str] = pcfg.get("state_file") or None
        self.maker_fee: float = float(pcfg.get("maker_fee", _DEFAULT_MAKER_FEE))
        self.taker_fee: float = float(pcfg.get("taker_fee", _DEFAULT_TAKER_FEE))
        self.limit_timeout_s: float = float(pcfg.get("limit_timeout_min", 15)) * 60.0
        # 成交加速兜底(纸面测试用): 限价单挂单超过 fill_assist_seconds 仍未成交时，
        # 按当前标记价模拟市价成交，保证能测到完整交易闭环(开仓→保护→追踪→平仓)。
        # 0 表示关闭该兜底，仅靠自然回补成交。
        self.fill_assist_s: float = float(pcfg.get("fill_assist_seconds", 0))
        # 纸面移动止损(2026-08-08 补齐): 与实盘 optimization.TrailingStop 同参 —
        # 激活阈值/追踪距离基于 ATR 动态计算, 无 ATR 时回退固定百分比。
        ocfg = config.get("optimization", {}) if isinstance(config, dict) else {}
        self.ts_activation_pct: float = float(
            ocfg.get("trailing_stop_activation_pct", 0.5))
        self.ts_trail_pct: float = float(ocfg.get("trailing_stop_trail_pct", 0.03))
        self.ts_atr_activation_mult: float = 0.5    # 0.5x ATR ≈ 1% 价格移动
        # 2026-08-10 用户要求: 追踪距离 0.75x→1.5x ATR (≈3% 价格移动) —
        # 实测 PUMP 峰值 +1.38% 时 0.75xATR 追踪仅 1.87% 空间, 刚激活就被回撤扫掉(锁在成本下方)。
        self.ts_atr_trail_mult: float = 1.5         # 1.5x ATR ≈ 3% 追踪距离
        self.max_hold_hours: float = float(rcfg.get("max_hold_hours", 48))
        self.dynamic_roi: Dict[str, float] = dict(
            rcfg.get("dynamic_roi", {"240": 0.015, "120": 0.025, "60": 0.035, "0": 0.05})
        )
        # 修复: ROI 不抢信号 TP 的利润。生效 ROI 目标 = max(配置 ROI, 信号TP距离×该比例)。
        # 实测 HOME: 信号 TP=+9.4%(RR2.5) 但 ROI 配置 5% 提前落袋 → 利润被砍半。
        # TP 优先、ROI 兜底 (价格冲到 TP×85% 附近回撤时保底，不影响 TP 正常触发)。
        self.dynamic_roi_tp_floor_pct: float = float(
            rcfg.get("dynamic_roi_tp_floor_pct", 0.85)
        )
        self.max_positions: int = int(rcfg.get("max_positions", 1))

        self._positions: Dict[str, PaperPosition] = {}
        self._trades: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._market_data_fn: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None
        self._cached_md: Dict[str, Dict[str, Any]] = {}
        self._last_close_pnl: Optional[float] = None
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # 行情注入
    # ------------------------------------------------------------------

    def set_market_data_provider(
        self, fn: Callable[[str], Optional[Dict[str, Any]]]
    ) -> None:
        """注入行情获取函数。

        fn(inst_id) -> {"candles": [{"high","low","close","ts"}...], "mark": float} | None
        """
        self._market_data_fn = fn

    # ------------------------------------------------------------------
    # 开仓
    # ------------------------------------------------------------------

    def open_position(
        self,
        signal: Any,
        instrument_info: Optional[Dict[str, Any]],
        risk_cfg: Dict[str, Any],
        equity: float,
    ) -> Optional[str]:
        """纸面开仓：按实盘口径计算张数，挂限价单等待回补成交。

        Returns:
            纸面订单号（挂单即返回，成交由 update() 判定）；参数非法返回 None。
        """
        if signal is None or getattr(signal, "inst_id", None) is None:
            return None
        inst_id = signal.inst_id
        with self._lock:
            if inst_id in self._positions:
                logger.info(f"[Paper] {inst_id} 已有纸面挂单/持仓，跳过重复开仓")
                return f"paper_already_{int(time.time() * 1000)}"

            if instrument_info is None:
                logger.error(f"[Paper] {inst_id} 无合约信息，纸面开仓失败")
                return None
            try:
                ct_val = float(instrument_info.get("ctVal", "0.01"))
                min_sz = float(instrument_info.get("minSz", "1"))
                lot = str(instrument_info.get("lotSz", "1"))
                sz_precision = len(lot.split(".")[1]) if "." in lot else 0
            except (TypeError, ValueError):
                logger.warning(f"[Paper] {inst_id} 合约信息解析失败，使用默认值")
                ct_val, min_sz, sz_precision = 0.01, 1.0, 0

            # 修复: 纸面杠杆口径与实盘一致 — executor.execute_signal 在计算仓位前
            # 会用 max_position_leverage 封顶 _eff_leverage，纸面此前直接用原始
            # signal.leverage，导致虚拟名义仓位偏离实盘（满杠杆 3-4x 时明显）。
            _paper_leverage = int(signal.leverage or 1)
            _lev_cap = int(risk_cfg.get("max_position_leverage", 0) or 0)
            if _lev_cap > 0 and _paper_leverage > _lev_cap:
                _paper_leverage = _lev_cap

            sz, margin = calculate_position_size(
                equity=equity,
                entry_price=float(signal.entry_price),
                stop_loss=float(signal.stop_loss),
                risk_pct=float(risk_cfg.get("risk_per_trade_pct", 1.0)),
                leverage=_paper_leverage,
                margin_pct=float(risk_cfg.get("margin_pct", 30.0)),
                direction=signal.position_side,
                contract_value=ct_val,
                min_sz=min_sz,
                sz_precision=sz_precision,
                sizing=str(risk_cfg.get("position_sizing", "risk")),
                enforce_risk_cap=bool(risk_cfg.get("enforce_risk_cap", True)),
            )
            if sz <= 0:
                logger.warning(
                    f"[Paper] {inst_id} 纸面仓位计算为 0，跳过 "
                    f"(entry={signal.entry_price}, sl={signal.stop_loss})")
                return None

            pos = PaperPosition(
                inst_id=inst_id,
                side=signal.position_side,
                entry_px=float(signal.entry_price),
                size=float(sz),
                leverage=float(_paper_leverage),
                margin=float(margin),
                tp_px=float(signal.take_profit),
                sl_px=float(signal.stop_loss),
                open_time=time.time(),
                ct_val=ct_val,
                filled=False,
                signal_id=str(getattr(signal, "signal_id", "") or ""),
                # 深挂 conditional 触发单 (2026-08-10): 价格先走到触发位
                # (距回补位一个阈值窗口), 触发后才挂限价等回补成交
                trigger_px=float(getattr(signal, "entry_trigger_px", 0) or 0),
            )
            self._positions[inst_id] = pos
            # 诊断日志: 挂单价与现价偏差（验证挂单距离限制是否生效）
            _md_now = self._cached_md.get(inst_id) or {}
            _mark_now = _md_now.get("mark")
            if _mark_now:
                try:
                    _dev_now = abs(float(pos.entry_px) - float(_mark_now)) / float(_mark_now) * 100.0
                    logger.info(
                        f"[Paper] {inst_id} 挂单距现价 {_dev_now:.2f}% "
                        f"(entry={pos.entry_px:.6g} mark={_mark_now:.6g})")
                except (TypeError, ValueError):
                    pass
            if pos.trigger_px > 0:
                logger.info(
                    f"[Paper] {inst_id} 深挂 conditional 触发单 "
                    f"{signal.position_side.upper()} trigger={pos.trigger_px:.6g} "
                    f"entry={signal.entry_price:.6g} size={sz}")
            logger.info(
                f"[Paper] {inst_id} 纸面限价挂单 {signal.position_side.upper()} "
                f"@{signal.entry_price:.6g} size={sz} "
                f"tp={signal.take_profit:.6g} sl={signal.stop_loss:.6g} "
                f"equity={equity:.2f} leverage={pos.leverage:.0f}x "
                f"margin={margin:.2f} notional={pos.size * pos.ct_val * pos.entry_px:.2f}")
            self.save()
            return f"paper_open_{int(time.time() * 1000)}"

    # ------------------------------------------------------------------
    # 每轮更新：成交判定 + 退出判定
    # ------------------------------------------------------------------

    def update(self) -> None:
        """推进纸面状态：限价成交判定、TP/SL/动态ROI/时间退出、更新 UPL。

        每轮主循环调用一次。所有行情获取失败时保守跳过（不误平不误成交）。
        """
        if self._market_data_fn is None:
            return
        with self._lock:
            insts = list(self._positions.keys())
            fresh_md: Dict[str, Dict[str, Any]] = {}
            for inst in insts:
                try:
                    md = self._market_data_fn(inst) or {}
                except Exception as _e:
                    logger.debug(f"[Paper] {inst} 行情获取失败: {_e}")
                    md = {}
                if not isinstance(md, dict):
                    md = {}
                fresh_md[inst] = md
            self._cached_md = fresh_md

            now = time.time()
            for inst in list(self._positions.keys()):
                pos = self._positions.get(inst)
                if pos is None:
                    continue
                md = self._cached_md.get(inst, {})
                if not pos.filled:
                    self._process_pending(pos, md, now)
                    continue
                exit_info = self._check_exit(pos, md)
                if exit_info is not None:
                    exit_px, reason = exit_info
                    self._close_locked(pos, exit_px, reason)
                    continue
                # 移动止损 (2026-08-08): 纸面持仓 SL 与实盘 trailing 同步抬升,
                # 验证追踪止损行为; 只在未触发退出时更新。
                self._update_trailing(pos, md)
                if md.get("mark") is not None:
                    pos.last_mark = float(md["mark"])
            self.save()

    def _process_pending(self, pos: PaperPosition, md: Dict[str, Any], now: float) -> None:
        """限价挂单成交/超时判定。

        修复: OKX REST candles 仅返回已收盘 K 线，candles[-1] 的 hi/lo
        可能严重滞后于实时价（曾出现实时价已越过挂单价却判定未成交）。
        成交判定必须同时纳入实时 mark 价。
        """
        candles = md.get("candles") or []
        mark = md.get("mark")
        if mark is not None:
            try:
                mark = float(mark)
            except (TypeError, ValueError):
                mark = None
        hi = lo = None
        if candles:
            try:
                hi, lo = (float(_candle_attr(candles[-1], "high")),
                          float(_candle_attr(candles[-1], "low")))
            except (TypeError, ValueError):
                hi = lo = None

        # conditional 触发单 (2026-08-10): 价格触及 trigger_px 前不激活/不成交,
        # 避免提前深挂空转。未触发期间不计超时(触发单等待期更长, 与实盘 algo 单一致)。
        if pos.trigger_px > 0 and not pos.extra.get("trigger_activated", False):
            if mark is None:
                return  # 无实时价无法判定触发, 保持挂单
            if pos.side == "long":
                if mark > pos.trigger_px:
                    return
            else:
                if mark < pos.trigger_px:
                    return
            pos.extra["trigger_activated"] = True
            logger.info(
                f"[Paper] {pos.inst_id} conditional 触发单已触发 "
                f"(mark={mark:.6g} trigger={pos.trigger_px:.6g}), "
                f"激活限价 {pos.entry_px:.6g} 等回补成交")

        if hi is not None or mark is not None:
            if pos.side == "long":
                if (lo is not None and lo <= pos.entry_px) or \
                        (mark is not None and mark <= pos.entry_px):
                    self._fill_locked(pos, md)
                    return
            elif pos.side == "short":
                if (hi is not None and hi >= pos.entry_px) or \
                        (mark is not None and mark >= pos.entry_px):
                    self._fill_locked(pos, md)
                    return
        # 成交加速兜底: 超过 fill_assist_s 仍未成交且能取到标记价时，
        # 按标记价模拟市价成交（纸面测试专用，实盘不受影响）。
        if self.fill_assist_s > 0 and now - pos.open_time > self.fill_assist_s:
            if mark is not None:
                self._fill_locked(pos, md, assist=True)
                return
        # 超时取消（与实盘限价单超时口径一致）
        # 修复: 用 >= 而非 > — 当 limit_timeout_s=0(测试/即时超时) 且
        # open 与 update 发生在同一时钟 tick 时，now-open_time==0.0，
        # 严格大于判定会静默跳过取消，导致"立即超时"配置失效。
        if now - pos.open_time >= self.limit_timeout_s:
            del self._positions[pos.inst_id]
            logger.info(f"[Paper] {pos.inst_id} 纸面限价单 {pos.entry_px:.6g} "
                        f"超时未成交，取消")

    def has_position(self, inst_id: str) -> bool:
        """是否已有该币种的纸面挂单/持仓（含未成交限价单）。

        2026-08-08: 供主循环在 execute_signal 前做源头去重，避免纸面模式下
        dry-run 假单路径每轮重复执行同一信号（positions_opened 虚增噪音）。
        """
        with self._lock:
            return inst_id in self._positions

    def _update_trailing(self, pos: PaperPosition, md: Dict[str, Any]) -> None:
        """纸面移动止损 — 与实盘 optimization.TrailingStop 同逻辑。

        激活: 价格朝有利方向移动 ≥ ATR×0.5 (无 ATR 时按 TP 距离×activation_pct);
        追踪: 止损 = 新高/新低 − ATR×0.75 (无 ATR 时 −entry×trail_pct)。
        只允许向有利方向收紧 sl_px，状态持久化于 pos.extra。
        """
        mark = md.get("mark")
        if mark is None:
            return
        try:
            mark = float(mark)
        except (TypeError, ValueError):
            return
        if mark <= 0 or pos.entry_px <= 0:
            return
        atr = _atr14(md.get("candles") or [])
        best = float(pos.extra.get("ts_best", 0.0) or 0.0)
        activated = bool(pos.extra.get("ts_activated", False))
        if pos.side == "long":
            if not activated:
                if atr > 0:
                    ok = (mark - pos.entry_px) >= atr * self.ts_atr_activation_mult
                else:
                    tp_dist = pos.tp_px - pos.entry_px
                    ok = (tp_dist > 0 and
                          (mark - pos.entry_px) / tp_dist >= self.ts_activation_pct)
                if ok:
                    activated = True
                    best = mark
            if activated:
                if mark > best:
                    best = mark
                trail = (atr * self.ts_atr_trail_mult if atr > 0
                         else pos.entry_px * self.ts_trail_pct)
                new_sl = best - trail
                if new_sl > pos.sl_px:
                    logger.info(
                        f"[Paper-TS] {pos.inst_id} long 追踪止损 "
                        f"{pos.sl_px:.6g} → {new_sl:.6g} "
                        f"(best={best:.6g} atr={atr:.4g})")
                    pos.sl_px = new_sl
        else:  # short
            if not activated:
                if atr > 0:
                    ok = (pos.entry_px - mark) >= atr * self.ts_atr_activation_mult
                else:
                    tp_dist = pos.entry_px - pos.tp_px
                    ok = (tp_dist > 0 and
                          (pos.entry_px - mark) / tp_dist >= self.ts_activation_pct)
                if ok:
                    activated = True
                    best = mark
            if activated:
                if mark < best or best <= 0:
                    best = mark
                trail = (atr * self.ts_atr_trail_mult if atr > 0
                         else pos.entry_px * self.ts_trail_pct)
                new_sl = best + trail
                if new_sl < pos.sl_px:
                    logger.info(
                        f"[Paper-TS] {pos.inst_id} short 追踪止损 "
                        f"{pos.sl_px:.6g} → {new_sl:.6g} "
                        f"(best={best:.6g} atr={atr:.4g})")
                    pos.sl_px = new_sl
        pos.extra["ts_best"] = best
        pos.extra["ts_activated"] = activated

    def update_sl(self, inst_id: str, new_sl: float) -> None:
        """纸面止损同步 (2026-08-10): 主循环 CE 抬止损(0R)/锁定保本 等外部保护
        决策只作用于实盘路径, 纸面内部 sl_px 不更新会导致纸面 PnL 与实盘口径
        不一致 — 同跌至成本价时实盘已 0R 保本出场, 纸面仍死等原止损(-1R)。

        只向有利方向收紧, 不放松:
          - long: 新 SL 高于当前 → 上移
          - short: 新 SL 低于当前 → 下移
        """
        with self._lock:
            pos = self._positions.get(inst_id)
            if pos is None or not pos.filled:
                return
            if pos.side == "long":
                if new_sl > pos.sl_px:
                    pos.sl_px = new_sl
                    logger.info(
                        f"[Paper] {inst_id} 止损同步上移 → {new_sl:.6g}")
            else:
                if new_sl < pos.sl_px:
                    pos.sl_px = new_sl
                    logger.info(
                        f"[Paper] {inst_id} 止损同步下移 → {new_sl:.6g}")
            self.save()

    def _fill_locked(self, pos: PaperPosition, md: Dict[str, Any],
                     assist: bool = False) -> None:
        """限价单成交；assist=True 为成交加速兜底(按标记价模拟市价成交)。

        修复: assist 成交价偏离挂单价时，TP/SL 必须按新成交价等比重算，
        否则 RR 严重缩水（实测 HOME: 挂单 RR=2.50 → 兜底成交 @+2.5%
        TP/SL 未动 → RR=1.09，等于把高质量信号开成了低质量仓位）。
        """
        pos.filled = True
        fill_px = pos.entry_px
        if assist and md.get("mark") is not None:
            fill_px = float(md["mark"])
            _orig_entry = pos.entry_px
            # 等比重算 TP/SL: 保持 |TP-entry|/entry 与 |SL-entry|/entry 不变
            if _orig_entry > 0 and abs(fill_px - _orig_entry) / _orig_entry > 1e-6:
                _tp_dist = abs(pos.tp_px - _orig_entry) / _orig_entry
                _sl_dist = abs(pos.sl_px - _orig_entry) / _orig_entry
                if pos.side == "long":
                    pos.tp_px = fill_px * (1 + _tp_dist)
                    pos.sl_px = fill_px * (1 - _sl_dist)
                else:
                    pos.tp_px = fill_px * (1 - _tp_dist)
                    pos.sl_px = fill_px * (1 + _sl_dist)
                logger.info(
                    f"[Paper] {pos.inst_id} 兜底成交偏离 "
                    f"{abs(fill_px - _orig_entry) / _orig_entry:.2%}，"
                    f"TP/SL 等比重算: tp={pos.tp_px:.6g} sl={pos.sl_px:.6g}")
            pos.entry_px = fill_px
        notional = pos.size * pos.ct_val * fill_px
        pos.entry_fee = notional * self.maker_fee
        self.balance -= pos.entry_fee
        if md.get("mark") is not None:
            pos.last_mark = float(md["mark"])
        logger.info(
            f"[Paper] {pos.inst_id} "
            f"{'成交加速兜底' if assist else '限价单成交'} @ {fill_px:.6g} "
            f"size={pos.size} 手续费-{pos.entry_fee:.4f}")

    # ------------------------------------------------------------------
    # 退出判定
    # ------------------------------------------------------------------

    def _check_exit(self, pos: PaperPosition, md: Dict[str, Any]) -> Optional[Tuple[float, str]]:
        """按优先级返回 (平仓价, 原因)；无退出信号返回 None。"""
        candles = md.get("candles") or []
        mark = md.get("mark")
        if mark is None:
            mark = pos.last_mark or pos.entry_px
        try:
            mark = float(mark)
        except (TypeError, ValueError):
            mark = pos.last_mark or pos.entry_px

        hi = lo = None
        if candles:
            try:
                hi, lo = (float(_candle_attr(candles[-1], "high")),
                          float(_candle_attr(candles[-1], "low")))
            except (TypeError, ValueError):
                hi = lo = None

        # 1. 止损（含滑点）+ 强平模拟 (2026-08-10 穿仓事故修复)
        # 修复 1: 判定必须结合实时 mark 价——REST candles 仅含已收盘 K 线，
        # 实时价可能已触及止损却因 K 线滞后未触发。
        # 修复 2: 高杠杆(50x)持仓价格跌 2% 保证金即耗尽, 真实交易所会在
        # 理论爆仓点强平, 只损失保证金。原实现无强平模拟, 让 SL(-5%) 在
        # 爆仓点之后才结算, 单笔亏损超过保证金导致账户穿仓
        # (实测 PUMP 换仓 PnL=-28.85, 保证金仅 11.40, 余额 25.54 → -3.31)。
        # 取止损触发价与理论爆仓价中更近 entry 者优先触发(跳空时止损优先,
        # 正常下跌时爆仓点先到则强平), 保证任何路径亏损 ≤ 保证金。
        # 注意: 滑点只影响成交价, 不提前触发判定 (与旧止损语义一致)。
        _liq_px = None
        try:
            _lev = max(1.0, float(pos.leverage or 1))
        except (TypeError, ValueError):
            _lev = 1.0
        if pos.side == "long":
            if _lev > 1.0 and pos.entry_px > 0:
                _liq_px = pos.entry_px * (1.0 - 1.0 / _lev)
                _trigger = max(pos.sl_px, _liq_px)
                _is_liq = _trigger == _liq_px
            else:
                _trigger = pos.sl_px
                _is_liq = False
            if (lo is not None and lo <= _trigger) or \
                    (mark is not None and mark <= _trigger):
                if _is_liq:
                    return _trigger, "liquidation"
                return pos.sl_px * (1 - _SL_SLIPPAGE), "stop_loss"
            # 2. 止盈（与止损同根 K 线同时触发时已按止损处理，保守）
            if (hi is not None and hi >= pos.tp_px) or \
                    (mark is not None and mark >= pos.tp_px):
                return pos.tp_px, "take_profit"
        elif pos.side == "short":
            if _lev > 1.0 and pos.entry_px > 0:
                _liq_px = pos.entry_px * (1.0 + 1.0 / _lev)
                _trigger = min(pos.sl_px, _liq_px)
                _is_liq = _trigger == _liq_px
            else:
                _trigger = pos.sl_px
                _is_liq = False
            if (hi is not None and hi >= _trigger) or \
                    (mark is not None and mark >= _trigger):
                if _is_liq:
                    return _trigger, "liquidation"
                return pos.sl_px * (1 + _SL_SLIPPAGE), "stop_loss"
            # 2. 止盈（与止损同根 K 线同时触发时已按止损处理，保守）
            if (lo is not None and lo <= pos.tp_px) or \
                    (mark is not None and mark <= pos.tp_px):
                return pos.tp_px, "take_profit"

        # 3. 动态 ROI（freqtrade minimal_roi 模式）
        elapsed_min = (time.time() - pos.open_time) / 60.0
        roi_target = self._dynamic_roi_target(elapsed_min)
        if roi_target is not None:
            # 修复: ROI 不抢信号 TP 的利润 — 生效目标 = max(配置ROI, 信号TP距离×杠杆×floor)。
            # 注意: ROI 目标量纲是"保证金收益率"(已含杠杆), 而 TP 距离是价格变动,
            # 换算: 价格到 TP 时保证金收益率 ≈ tp_dist × leverage。
            # 配置 {0:0.05} 在 2x 杠杆下价格+2.5% 就落袋, 而信号 TP 可能 +9.4%,
            # 会在 TP 之前截胡砍掉一半利润。TP 优先、ROI 只在冲高回撤时保底。
            _tp_dist_pct = self._tp_distance_pct(pos)
            if _tp_dist_pct is not None and _tp_dist_pct > 0:
                _tp_margin_roi = _tp_dist_pct * max(1.0, float(pos.leverage or 1))
                roi_target = max(roi_target, _tp_margin_roi * self.dynamic_roi_tp_floor_pct)
            upl_pct = self._upl_pct(pos, mark)
            if upl_pct >= roi_target:
                return mark, "dynamic_roi"

        # 4. 最大持仓时长
        if (time.time() - pos.open_time) / 3600.0 >= self.max_hold_hours:
            return mark, "time_exit"

        return None

    def _dynamic_roi_target(self, elapsed_min: float) -> Optional[float]:
        """取满足 elapsed >= 阈值分钟数 的最大分钟数对应的 ROI 目标。"""
        best: Optional[Tuple[float, float]] = None
        for mins_str, roi in self.dynamic_roi.items():
            try:
                mins = float(mins_str)
            except (TypeError, ValueError):
                continue
            if elapsed_min >= mins and (best is None or mins > best[0]):
                best = (mins, float(roi))
        return best[1] if best else None

    def _tp_distance_pct(self, pos: PaperPosition) -> Optional[float]:
        """持仓止盈距离百分比 (|TP-entry|/entry)，TP 无效返回 None。"""
        if pos is None or pos.entry_px <= 0 or pos.tp_px <= 0:
            return None
        if pos.side == "long":
            if pos.tp_px <= pos.entry_px:
                return None
            return (pos.tp_px - pos.entry_px) / pos.entry_px
        else:
            if pos.tp_px >= pos.entry_px:
                return None
            return (pos.entry_px - pos.tp_px) / pos.entry_px

    def close_position(self, inst_id: str, reason: str = "manual") -> Optional[str]:
        """手动平仓（反手/换仓时由主循环调用），按最新价成交。"""
        with self._lock:
            pos = self._positions.get(inst_id)
            if pos is None:
                return None
            md = self._cached_md.get(inst_id) or {}
            mark = md.get("mark") or pos.last_mark or pos.entry_px
            try:
                mark = float(mark)
            except (TypeError, ValueError):
                mark = pos.entry_px
            self._close_locked(pos, mark, reason)
            self.save()
            return f"paper_close_{int(time.time() * 1000)}"

    def scale_out(self, inst_id: str, pct: float,
                  reason: str = "scale_out") -> Optional[str]:
        """分批止盈: 平掉当前仓位 pct% 锁定利润, 剩余仓位继续持有。

        pct ∈ (0, 100)。100% 请走 close_position。减仓张数向下取整对齐张数，
        已实现盈亏记入 balance 与 _trades(reason=scale_out), 剩余仓位按比例
        缩减 margin。与实盘 close_position_limit 减仓口径一致。

        Returns:
            纸面部分平仓单号; 参数非法/无持仓返回 None
        """
        with self._lock:
            pos = self._positions.get(inst_id)
            if pos is None or not pos.filled or pos.size <= 0:
                return None
            try:
                pct = float(pct)
            except (TypeError, ValueError):
                return None
            if pct <= 0.0 or pct >= 100.0:
                return None
            md = self._cached_md.get(inst_id) or {}
            mark = md.get("mark") or pos.last_mark or pos.entry_px
            try:
                mark = float(mark)
            except (TypeError, ValueError):
                mark = pos.entry_px
            if mark <= 0:
                return None
            close_size = math.floor(pos.size * pct / 100.0)
            if close_size <= 0:
                logger.info(f"[Paper] {inst_id} 减仓比例 {pct:.0f}% 折算张数 < 1，跳过")
                return None
            remain_size = pos.size - close_size
            # 平掉部分的盈亏（含退出手续费）
            if pos.side == "long":
                gross = close_size * pos.ct_val * (mark - pos.entry_px)
            else:
                gross = close_size * pos.ct_val * (pos.entry_px - mark)
            exit_fee = close_size * pos.ct_val * mark * self.taker_fee
            net = gross - exit_fee
            self.balance += net
            self._last_close_pnl = net
            notional_close = close_size * pos.ct_val * pos.entry_px
            trade = {
                "inst_id": pos.inst_id,
                "side": pos.side,
                "entry_px": pos.entry_px,
                "exit_px": round(mark, 8),
                "size": close_size,
                "leverage": pos.leverage,
                "reason": reason,
                "pnl": round(net, 6),
                "pnl_pct": round(net / notional_close * 100, 4) if notional_close > 0 else 0.0,
                "fees": round(exit_fee, 6),
                "hold_hours": round((time.time() - pos.open_time) / 3600.0, 2),
                "open_time": pos.open_time,
                "closed_at": time.time(),
                "signal_id": pos.signal_id,
                "partial": True,
            }
            self._trades.append(trade)
            # 剩余仓位按比例缩减 margin（size 与 margin 同步缩小）
            if remain_size > 0:
                pos.size = remain_size
                pos.margin = pos.margin * (remain_size / (remain_size + close_size))
            else:
                del self._positions[pos.inst_id]
            logger.info(
                f"[Paper] {inst_id} 分批止盈: 平 {close_size} 张 @ {mark:.6g} "
                f"pnl={net:+.4f}，剩余 {remain_size} 张继续持有"
            )
            self.save()
            return f"paper_scale_{int(time.time() * 1000)}"

    def _close_locked(self, pos: PaperPosition, exit_px: float, reason: str) -> None:
        notional = pos.size * pos.ct_val * pos.entry_px
        notional_exit = pos.size * pos.ct_val * exit_px
        if pos.side == "long":
            gross = pos.size * pos.ct_val * (exit_px - pos.entry_px)
        else:
            gross = pos.size * pos.ct_val * (pos.entry_px - exit_px)
        exit_fee = notional_exit * self.taker_fee
        # 强平兜底 (2026-08-10 穿仓事故修复): 理论爆仓点结算 gross ≈ -margin,
        # 因滑点/跳空可能略超, 封顶在保证金(真实强平最多损失全部保证金),
        # 手续费仍照扣, 防止余额穿仓变负。
        if reason == "liquidation":
            gross = max(gross, -pos.margin)
        net = gross - exit_fee
        self.balance += net
        self._last_close_pnl = net
        hold_h = (time.time() - pos.open_time) / 3600.0
        trade = {
            "inst_id": pos.inst_id,
            "side": pos.side,
            "entry_px": pos.entry_px,
            "exit_px": round(exit_px, 8),
            "size": pos.size,
            "leverage": pos.leverage,
            "reason": reason,
            "pnl": round(net, 6),
            "pnl_pct": round(net / notional * 100, 4) if notional > 0 else 0.0,
            "fees": round(pos.entry_fee + exit_fee, 6),
            "hold_hours": round(hold_h, 2),
            "open_time": pos.open_time,
            "closed_at": time.time(),
            "signal_id": pos.signal_id,
        }
        self._trades.append(trade)
        del self._positions[pos.inst_id]
        logger.info(
            f"[Paper] 平仓 {pos.inst_id} {reason} @ {exit_px:.6g} "
            f"PnL={net:+.2f} USDT ({trade['pnl_pct']:+.2f}%) "
            f"手续费={trade['fees']:.4f} 持仓{hold_h:.1f}h")

    def consume_last_close_pnl(self) -> Optional[float]:
        """取走最近一次平仓的已实现盈亏（供 pending_close 确认用）。"""
        with self._lock:
            v = self._last_close_pnl
            self._last_close_pnl = None
            return v

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def _upl(self, pos: PaperPosition, mark: float) -> float:
        notional = pos.size * pos.ct_val * pos.entry_px
        if pos.side == "long":
            return notional * (mark - pos.entry_px) / pos.entry_px
        return notional * (pos.entry_px - mark) / pos.entry_px

    def _upl_pct(self, pos: PaperPosition, mark: float) -> float:
        """UPL 相对保证金（杠杆后）的百分比。"""
        if pos.margin <= 0:
            return 0.0
        return self._upl(pos, mark) / pos.margin

    def _equity_locked(self) -> float:
        """持锁前提款计算权益（内部方法，供 get_equity/summary 复用，避免嵌套锁死锁）。"""
        eq = self.balance
        for pos in self._positions.values():
            if pos.filled and pos.size > 0:
                md = self._cached_md.get(pos.inst_id) or {}
                mark = md.get("mark") or pos.last_mark or pos.entry_px
                try:
                    mark = float(mark)
                except (TypeError, ValueError):
                    mark = pos.last_mark or pos.entry_px
                eq += self._upl(pos, mark)
        return eq

    def get_equity(self) -> float:
        """当前权益 = 现金 + 全部已成交持仓的未实现盈亏。"""
        with self._lock:
            return self._equity_locked()

    def to_positions_dict(self) -> Dict[str, Dict[str, Any]]:
        """输出与 executor.monitor_positions 结构一致的持仓字典（供主循环决策）。

        仅含已成交（filled）持仓；未成交的限价挂单不出现在持仓中，
        由引擎内部去重（同币种不重复挂单）与超时取消管理。
        """
        with self._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for pos in self._positions.values():
                if not pos.filled or pos.size <= 0:
                    continue
                md = self._cached_md.get(pos.inst_id) or {}
                mark = md.get("mark") or pos.last_mark or pos.entry_px
                try:
                    mark = float(mark)
                except (TypeError, ValueError):
                    mark = pos.entry_px
                upl = self._upl(pos, mark)
                upl_pct = (upl / pos.margin) if pos.margin > 0 else 0.0
                out[pos.inst_id] = {
                    "pos_side": pos.side,
                    "size": pos.size,
                    "avg_px": pos.entry_px,
                    "mark_px": mark,
                    "upl": upl,
                    "upl_ratio_pct": upl_pct * 100.0,
                    "margin": pos.margin,
                    "leverage": pos.leverage,
                    "c_time": pos.open_time,
                    "signal_id": pos.signal_id,
                }
            return out

    def summary(self) -> str:
        """纸面交易状态报告。"""
        eq = self._equity_locked()
        with self._lock:
            wins = [t for t in self._trades if t["pnl"] > 0]
            losses = [t for t in self._trades if t["pnl"] <= 0]
            gross_pnl = sum(t["pnl"] for t in self._trades)
            fees = sum(t["fees"] for t in self._trades)
            lines = [
                "=" * 56,
                "  PAPER TRADING 纸面交易报告",
                "=" * 56,
                f"  初始余额:     {self.initial_balance:.2f} USDT",
                f"  当前权益:     {eq:.2f} USDT",
                f"  已实现盈亏:   {gross_pnl:+.2f} USDT",
                f"  累计手续费:   {fees:.2f} USDT",
                f"  平仓交易:     {len(self._trades)} 笔 "
                f"(胜 {len(wins)} / 负 {len(losses)})",
            ]
            if self._trades:
                wr = len(wins) / len(self._trades)
                pf = (sum(t['pnl'] for t in wins) /
                      abs(sum(t['pnl'] for t in losses))) if losses and sum(t['pnl'] for t in losses) != 0 else float('inf')
                lines.append(f"  胜率:         {wr:.1%}")
                lines.append(f"  利润因子:     {pf:.2f}")
            if self._positions:
                lines.append(f"  当前持仓/挂单: {len(self._positions)} 个")
            lines.append("=" * 56)
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self) -> None:
        """原子写状态文件（先写 .tmp + fsync 再 os.replace），防断电/异常损坏。

        修复 P2-7: 补 fsync（原实现无 fsync，断电可能落半截内容）。
        """
        try:
            if not self.state_file:
                return
            payload = {
                "initial_balance": self.initial_balance,
                "balance": self.balance,
                "positions": [asdict(p) for p in self._positions.values()],
                "trades": self._trades,
                "start_time": self._start_time,
            }
            tmp_path = f"{self.state_file}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.state_file)
        except Exception as _e:
            logger.warning(f"[Paper] 状态保存失败: {_e}")

    def load(self) -> None:
        """从状态文件恢复虚拟余额/持仓/历史交易。文件不存在则用初始余额。

        修复 P2-7/P2-8: 主文件损坏时回退 .bak；数值字段做 float 强转，
        脏数据不再静默丢仓。
        """
        if not self.state_file or not os.path.exists(self.state_file):
            return
        payload = None
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as _e:
            logger.warning(f"[Paper] 状态文件损坏，尝试 .bak 恢复: {_e}")
            bak_path = f"{self.state_file}.bak"
            if os.path.exists(bak_path):
                try:
                    with open(bak_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    logger.warning(f"[Paper] 从 {bak_path} 恢复成功")
                except Exception as _e2:
                    logger.warning(f"[Paper] .bak 也损坏，使用初始余额: {_e2}")
                    return
            else:
                return
        try:
            self.initial_balance = float(payload.get("initial_balance", self.initial_balance))
            self.balance = float(payload.get("balance", self.initial_balance))
            self._start_time = float(payload.get("start_time", self._start_time))
            self._positions = {}
            # 修复 P2-8: 数值字段 float 强转，类型校验不通过则丢弃该仓并告警
            _float_fields = ("entry_px", "size", "leverage", "margin",
                             "tp_px", "sl_px", "open_time", "ct_val", "entry_fee")
            for raw in payload.get("positions", []):
                try:
                    _clean = {k: v for k, v in raw.items()
                              if k in PaperPosition.__dataclass_fields__}
                    for _fk in _float_fields:
                        if _fk in _clean and _clean[_fk] is not None:
                            _clean[_fk] = float(_clean[_fk])
                    pos = PaperPosition(**_clean)
                    self._positions[pos.inst_id] = pos
                except (TypeError, ValueError) as _pe:
                    logger.warning(
                        f"[Paper] 持仓数据异常被跳过: {raw.get('inst_id', '?')}: {_pe}"
                    )
                    continue
            self._trades = list(payload.get("trades", []))
            logger.info(
                f"[Paper] 状态已加载: 余额 {self.balance:.2f} USDT, "
                f"持仓 {len(self._positions)} 个, 历史 {len(self._trades)} 笔")
        except Exception as _e:
            logger.warning(f"[Paper] 状态加载失败（使用初始余额）: {_e}")
