"""
执行模块 — 仓位计算、下单执行、持仓监控、提现检查。

职责：
  - 根据信号和风控参数计算精确仓位
  - 通过 OKXClient 执行限价单（含止盈止损）
  - 监控已开仓位状态，检测止盈止损触发
  - 钱包翻倍时触发提现提醒
"""

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple

from okx_client import OKXClient, OKXQueryError
from strategy import Signal


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 状态管理
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Agent 运行状态，持久化到 JSON 文件。"""
    initial_equity: float = 0.0          # 本轮初始权益
    highest_equity: float = 0.0          # 本轮最高权益
    total_pnl: float = 0.0               # 累计盈亏（含未实现盈亏，用于展示和回撤计算）
    realized_pnl: float = 0.0            # 累计已实现盈亏（仅平仓时更新，用于风控和统计）
    last_equity: float = 0.0             # 上一轮权益（用于计算轮间已实现盈亏）
    positions_opened: int = 0            # 本轮已开仓数
    positions_closed: int = 0            # 本轮已平仓数
    winning_trades: int = 0              # 盈利笔数
    losing_trades: int = 0               # 亏损笔数
    break_even_trades: int = 0           # 保本平仓笔数（pnl == 0）
    last_withdrawal_equity: float = 0.0  # 上次提现时的权益
    withdrawal_count: int = 0            # 提现次数
    daily_loss: float = 0.0              # 当日累计亏损（ equity - daily_start_equity，≤0）
    daily_date: str = ""                 # 当日日期 (YYYY-MM-DD)
    daily_start_equity: float = 0.0      # 当日开盘权益（用于计算实际日内盈亏）
    active_signals: Dict[str, dict] = field(default_factory=dict)  # 活跃信号 {inst_id: position_info}
    _pending_close: Optional[dict] = None  # 非阻塞平仓待确认状态 {inst_id, ord_id, ...}
    pending_close_meta: Optional[dict] = None  # 修复 P0-D: _pending_close 的可序列化元数据，
    # 持久化平仓确认所需的全部字段（不含 Signal/MasterAnalysis 等瞬态对象）。
    # 重启后据此重建 _pending_close，续跑平仓确认 → 已实现盈亏/日亏限额不丢。
    trailing_stop_state: Dict[str, float] = field(default_factory=dict)  # 追踪止损高水位线持久化 {inst_id: highest_price}
    recent_pnl: List[float] = field(default_factory=list)  # 近 N 笔已实现盈亏(期望值门禁用, 2026-08-07)
    ev_degrade_until: float = 0.0  # 期望值降频冷却截止时间戳(负期望时降频而非硬暂停, 2026-08-07)


class StateManager:
    """状态持久化管理器。

    修复: 添加 threading.Lock 防止多线程并发写入 JSON 文件损坏，
    使用原子写入（先写临时文件再 rename）确保写入过程不产生半截文件。
    """

    def __init__(self, state_path: str):
        self.state_path = state_path
        self._lock = threading.Lock()
        self.state = self._load()

    def lock(self):
        """返回线程锁上下文管理器，供外部安全地修改 active_signals。

        修复 M-1: 提供公开接口替代直接访问 _lock 私有属性。
        """
        return self._lock

    def _load(self) -> AgentState:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 清理 transient 字段：_pending_close 不应跨会话持久化
                # （其含 Signal/MasterAnalysis 等不可序列化对象）
                data.pop("_pending_close", None)
                state = AgentState(**data)
                # 修复 P0-D: 从持久化的可序列化元数据重建平仓确认状态，
                # 重启后续跑确认 → 已实现盈亏/日亏限额不丢
                if state.pending_close_meta:
                    state._pending_close = dict(state.pending_close_meta)
                return state
            except Exception as e:
                logger.warning(f"Failed to load state: {e}, trying backup...")
                # 修复: 主文件损坏时尝试从 .bak 备份恢复
                bak_path = self.state_path + ".bak"
                if os.path.exists(bak_path):
                    try:
                        with open(bak_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data.pop("_pending_close", None)
                        logger.warning(f"从备份文件 {bak_path} 恢复状态成功")
                        # 恢复后立即写回主文件
                        state = AgentState(**data)
                        # 修复 P0-D: 备份恢复同样重建平仓确认元数据
                        if state.pending_close_meta:
                            state._pending_close = dict(state.pending_close_meta)
                        self.state = state
                        self._write_atomic(data)
                        return state
                    except Exception as e2:
                        logger.warning(f"备份文件也损坏: {e2}, starting fresh")
        return AgentState()

    def _write_atomic(self, data: dict):
        """原子写入 + 备份保护。先写临时文件 → fsync → rename → 更新备份。"""
        tmp_path = self.state_path + ".tmp"
        bak_path = self.state_path + ".bak"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            # 同步写入磁盘，防止 OS 缓存导致断电丢失
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.state_path)
        # 写入成功后更新备份（在主文件原子替换之后）
        try:
            with open(bak_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass  # 备份失败不影响主文件

    def save(self):
        with self._lock:
            try:
                data = dict(self.state.__dict__)
                # 修复 P0-D: _pending_close 含 Signal/MasterAnalysis/Candle 等
                # 不可序列化对象，不能整体持久化（此前导致写入失败）。现在拆出
                # 可序列化元数据持久化（供重启续跑平仓确认 → 已实现盈亏与
                # 日亏限额不丢），瞬态对象本身仅存内存。
                _pc = self.state._pending_close
                if _pc:
                    data["pending_close_meta"] = {
                        k: v for k, v in _pc.items()
                        if k not in ("best_signal", "best_analysis", "best_coin",
                                     "best_regime", "best_funding_rate", "candles_4h")
                    }
                else:
                    data.pop("pending_close_meta", None)
                data.pop("_pending_close", None)
                self._write_atomic(data)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                # 修复: 写入失败时记录警告但不崩溃，保留旧文件不变

    def update_equity(self, equity: float):
        """更新权益追踪。"""
        if self.state.initial_equity == 0.0:
            self.state.initial_equity = equity
            self.state.highest_equity = equity
            self.state.last_withdrawal_equity = equity
        if equity > self.state.highest_equity:
            self.state.highest_equity = equity
        self.state.total_pnl = equity - self.state.initial_equity

    def reset_daily_if_new_day(self, equity: float = 0.0):
        """跨日重置当日亏损，记录当日开盘权益。

        Args:
            equity: 当前权益，用于作为当日开盘权益基准。
        """
        # 修复: 使用 UTC 时间对齐交易所日线结算基准 (UTC 00:00)
        # 加密货币 7x24h 交易，本地时区 (如 UTC+8) 会导致日切点与交易所不一致
        # 在极端单边行情中，本地时区的"一天"可能跨两次交易所结算，导致日亏损限额被重置两次
        # 修复: datetime.utcnow() 在 Python 3.12+ 已弃用，改用 datetime.now(timezone.utc)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.state.daily_date:
            self.state.daily_loss = 0.0
            self.state.daily_date = today
            self.state.daily_start_equity = equity
        elif self.state.daily_start_equity <= 0 and equity > 0:
            # 首次运行或旧状态文件无此字段时补设
            self.state.daily_start_equity = equity

    def check_withdrawal(self, equity: float, withdrawal_pct: float) -> bool:
        """检查是否触发提现条件。

        触发条件: 权益 >= 上次提现权益 * (1 + withdrawal_pct/100)
        例如 withdrawal_pct=25 → 权益增长 25% 触发
        """
        if self.state.last_withdrawal_equity <= 0:
            return False
        if withdrawal_pct <= 0:
            return False
        return equity >= self.state.last_withdrawal_equity * (1.0 + withdrawal_pct / 100.0)

    def record_withdrawal(self, equity: float, withdrawal_pct: float = 25.0):
        """记录提现建议事件。

        修复: 不再在本地状态中虚拟扣除资金。
        脚本不具备 API 资金划转权限，强行修改本地权益计数器
        会导致下一轮拉取的真实权益与本地记录脱节，扰乱 Kelly/自适应杠杆计算。
        仅记录提现提醒次数和最后提醒权益，不修改任何资金数字。
        """
        self.state.last_withdrawal_equity = equity
        self.state.withdrawal_count += 1
        self.save()
        withdraw_amount = equity * withdrawal_pct / 100.0
        logger.warning(
            f"=== 提现提醒 #{self.state.withdrawal_count} === "
            f"当前权益: {equity:.2f} USDT, "
            f"建议提现: {withdraw_amount:.2f} USDT ({withdrawal_pct:.0f}%) "
            f"—— 请手动通过 OKX App/Web 执行资金划转"
        )

    def record_realized_pnl(self, pnl: float):
        """记录已实现盈亏（平仓时调用），同步更新 daily_loss 和 realized_pnl。

        Args:
            pnl: 本次平仓的实现盈亏（USDT），可为负
        """
        # 修复 C3+H6: daily_loss 改为追踪实际累计已实现 PnL（不再被未实现盈亏覆盖）
        # 原先: 盈利时 min(0, daily_loss + pnl) 导致 profit+loss 序列后 daily_loss 显示亏损
        # 现在: 直接累加，正负均可，最终检查 daily_loss 是否低于 -max_daily_loss
        self.state.daily_loss += pnl
        self.state.realized_pnl += pnl
        # 滚动已实现盈亏列表(期望值门禁用): 保留最近 100 笔
        self.state.recent_pnl.append(pnl)
        if len(self.state.recent_pnl) > 100:
            self.state.recent_pnl = self.state.recent_pnl[-100:]
        if pnl < 0:
            self.state.losing_trades += 1
        elif pnl > 0:
            self.state.winning_trades += 1
        else:
            self.state.break_even_trades += 1
        self.state.positions_closed += 1

    def get_win_rate(self) -> float:
        """计算胜率（剔除保本交易，避免失真）。

        Returns:
            胜率 (0-1)，无交易时返回 0
        """
        decisive = self.state.winning_trades + self.state.losing_trades
        if decisive <= 0:
            return 0.0
        return self.state.winning_trades / decisive


# ---------------------------------------------------------------------------
# 仓位计算
# ---------------------------------------------------------------------------

def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: float,
    leverage: int,
    margin_pct: float,
    direction: str = "long",
    contract_value: float = 1.0,
    min_sz: float = 1.0,
    sz_precision: int = 0,
    sizing: str = "risk",
    enforce_risk_cap: bool = False,
) -> Tuple[float, float]:
    """计算仓位大小。

    两种仓位模式 (由 config risk.position_sizing 控制, 纸面/实盘统一口径):
      - "risk" (默认): 风险倒推 — 单笔风险金额固定 (equity×risk_pct%)，
        止损越近仓位越大。公式:
            risk_amount = equity * risk_pct / 100
            stop_distance_pct = |entry_price - stop_loss| / entry_price
            position_value = risk_amount / stop_distance_pct
            margin = position_value / leverage
            sz = position_value / (entry_price * contract_value)
      - "margin" (满仓模拟): 保证金驱动 — 用满 margin_pct% 权益做保证金，
        名义仓位 = 保证金 × 杠杆 (模拟真实满杠杆交易)。公式:
            margin = equity * margin_pct / 100
            position_value = margin * leverage
            sz = position_value / (entry_price * contract_value)

    Args:
        equity: 账户权益
        entry_price: 入场价
        stop_loss: 止损价
        risk_pct: 单笔风险比例 (1.0 = 1%)，仅 "risk" 模式使用
        leverage: 杠杆倍数
        margin_pct: 最大保证金比例 (30 = 30%)
        direction: 持仓方向 "long" | "short"
        contract_value: 每张合约面值 (USDT 本位默认为 1)
        min_sz: 最小下单量
        sz_precision: 数量精度（小数位数）
        sizing: 仓位模式 "risk" | "margin"

    Returns:
        (sz, margin_used) 合约张数和占用保证金
    """
    if equity <= 0 or entry_price <= 0 or stop_loss <= 0:
        return 0.0, 0.0
    # 方向校验：long 的止损必须低于入场价，short 的止损必须高于入场价
    if direction == "long" and stop_loss >= entry_price:
        return 0.0, 0.0
    if direction == "short" and stop_loss <= entry_price:
        return 0.0, 0.0

    stop_distance_pct = abs(entry_price - stop_loss) / entry_price
    if stop_distance_pct < 1e-10:
        return 0.0, 0.0

    if sizing == "margin":
        # 保证金驱动 (满仓模拟): 用满 margin_pct% 权益做保证金，名义 = 保证金 × 杠杆
        margin = equity * margin_pct / 100.0
        if margin <= 0:
            return 0.0, 0.0
        # 以损定量硬上限(2026-08-07 调研): 单笔风险(保证金×止损距离%)不得超过
        # 权益×risk_pct%(社区铁律单笔风险≤1-2%)。实测 ACT SL 6.96%×9U=2.1% 超 1%,
        # margin 模式若不禁则绕过 risk_per_trade_pct 风控。
        if enforce_risk_cap and risk_pct > 0:
            max_margin_by_risk = equity * risk_pct / 100.0 / stop_distance_pct
            if margin > max_margin_by_risk:
                logger.debug(
                    f"Margin capped by risk cap: {margin:.2f} → {max_margin_by_risk:.2f} "
                    f"(risk={risk_pct}%, stop={stop_distance_pct:.2%})")
                margin = max_margin_by_risk
        position_value = margin * leverage
    else:
        # 风险倒推 (默认, 风控优先)
        risk_amount = equity * risk_pct / 100.0
        # 仓位价值 = 风险金额 / 止损距离
        position_value = risk_amount / stop_distance_pct
        # 保证金 = 仓位价值 / 杠杆
        margin = position_value / leverage
        # 保证金上限检查
        max_margin = equity * margin_pct / 100.0
        if margin > max_margin:
            margin = max_margin
            position_value = margin * leverage
            logger.debug(f"Position capped by margin limit: "
                         f"margin={margin:.2f} (max={max_margin:.2f})")

    # 合约张数
    sz = position_value / (entry_price * contract_value)
    sz = math.floor(sz * (10 ** sz_precision)) / (10 ** sz_precision)

    if sz < min_sz:
        logger.debug(f"Position size {sz} < min {min_sz}, skipping")
        return 0.0, 0.0

    return sz, margin


# ---------------------------------------------------------------------------
# 订单执行
# ---------------------------------------------------------------------------

def resolve_full_leverage(
    client: OKXClient,
    inst_id: str,
    signal_leverage: int,
    risk_cfg: dict,
) -> int:
    """按币种最大杠杆解析执行杠杆 (2026-08-09 用户要求: 满倍率不分币种)。

    优先取 OKX position-tiers 档位 maxLever（该币种逐仓允许的最大杠杆），
    再受 max_position_leverage 封顶(>0 时)；获取失败回退信号建议杠杆。
    默认(max_position_leverage=0)时 = 币种最大杠杆。

    Returns:
        执行杠杆（≥1）
    """
    _cap = int(risk_cfg.get("max_position_leverage", 0) or 0)
    try:
        _tiers = client.get_position_tiers(inst_id)
        if _tiers:
            _max_lev = float(_tiers.get("maxLever", 0) or 0)
            if _max_lev > 1:
                _lev = int(_max_lev)
                if _cap > 0:
                    return max(1, min(_lev, _cap))
                return _lev
    except Exception:
        pass
    if _cap > 0:
        return max(1, min(int(signal_leverage or 1), _cap))
    return max(1, int(signal_leverage or 1))


def execute_signal(
    client: OKXClient,
    signal: Signal,
    equity: float,
    config: dict,
    instrument_info: Optional[dict] = None,
) -> Optional[str]:
    """执行交易信号 — 下单并设置止盈止损。

    Args:
        client: OKX 客户端
        signal: 交易信号
        equity: 当前权益
        config: 完整配置
        instrument_info: 合约信息 (ctVal, lotSz, minSz, tickSz 等)

    Returns:
        ord_id 或 None
    """
    risk_cfg = config["risk"]

    # 满倍率模式 (2026-08-09 用户要求): 执行杠杆 = 币种最大杠杆 (tiers.maxLever)，
    # 不再受 leverage_stop_budget 反推限制。max_position_leverage=0 时不封顶。
    # 信号层 leverage 若已由主循环 resolve_full_leverage 覆盖，此处幂等。
    _eff_leverage = resolve_full_leverage(
        client, signal.inst_id, int(signal.leverage or 1), risk_cfg)

    # ---- P0-A 修复: 止损距离 vs 爆仓距离 硬校验 ----
    # 此前全系统无强平价计算，高杠杆窄止损在跳空下会先爆仓后止损
    # （代码注释自认"强平引擎会比止损单先到"）。
    # 逐仓 isolated 近似: 强平距离 ≈ 1/杠杆 − 维持保证金率(MMR)（未含手续费，
    # 取安全侧余量）。MMR 从 position-tiers 档位获取，失败用保守默认 0.5%。
    # 硬校验: |entry − SL| < 强平距离 × 安全系数，否则拒单/降杠杆。
    _mmr = float(risk_cfg.get("default_mmr", 0.005) or 0.005)
    try:
        _tiers = client.get_position_tiers(signal.inst_id)
        if _tiers:
            _t_mmr = float(_tiers.get("mmr", 0) or 0)
            if _t_mmr > 0:
                _mmr = _t_mmr
    except Exception:
        pass
    _liq_dist = 1.0 / max(_eff_leverage, 1) - _mmr
    _entry_px = float(signal.entry_price or 0)
    _sl_px = float(signal.stop_loss or 0)
    _stop_dist = (abs(_entry_px - _sl_px) / _entry_px) if _entry_px > 0 else 1.0
    _safety = float(risk_cfg.get("liq_safety_factor", 0.5) or 0.5)
    # 满倍率模式 (2026-08-09): 满杠杆下爆仓距离(1/杠杆)通常远小于 FVG 止损距离，
    # 若保持 fail-closed 将拒掉全部信号。liq_check_fail_closed=false(默认) 时
    # 不再放行: 2026-08-10 起降杠杆使止损先于爆仓(用户要求), 仅杠杆异常
    # (liq_dist<=0) 或显式配置 true 时拒单。
    _fail_closed = bool(risk_cfg.get("liq_check_fail_closed", False))
    if _liq_dist <= 0:
        logger.error(
            f"[LiqCheck] {signal.inst_id} 拒单: 杠杆 {_eff_leverage}x 爆仓距离 "
            f"{_liq_dist:.2%} 非法 (MMR {_mmr:.3%})")
        return None
    if _stop_dist >= _liq_dist * _safety:
        _msg = (
            f"[LiqCheck] {signal.inst_id} 止损距离 {_stop_dist:.2%} >= "
            f"爆仓距离 {_liq_dist:.2%} × {_safety:.0%} "
            f"(杠杆 {_eff_leverage}x, MMR {_mmr:.3%}) — 止损先于爆仓"
        )
        if _fail_closed:
            logger.error(_msg + "，拒单(fail-closed)")
            return None
        # 修复 2026-08-10 (用户要求"先止损再爆仓"): 满杠杆下止损距离 ≥ 爆仓
        # 安全距离 = 价格必先爆仓后止损。paper 引擎曾缺强平模拟按全额亏损
        # 结算(实测 PUMP 50x SL=-5.09% vs 爆仓-2% → 单笔 -28.85 穿仓)。
        # 不再警告放行: 降杠杆使爆仓距离扩至覆盖止损距离, 止损必然先触发,
        # 单笔亏损 ≤ 保证金。新杠杆 = floor(1/(止损距离/安全系数 + MMR))。
        # 数学: liq_dist_new = 1/new_lev - mmr ≥ stop_dist/safety → 覆盖。
        _need_lev = 1.0 / (_stop_dist / _safety + _mmr)
        _new_lev = max(1, int(math.floor(_need_lev)))
        if _new_lev >= _eff_leverage:
            logger.error(
                _msg + f"，降杠杆无效(需 ≤{_new_lev}x 仍 ≥ 当前 {_eff_leverage}x)，拒单")
            return None
        logger.warning(
            f"[LiqCheck] {signal.inst_id} 降杠杆止损优先: {_eff_leverage}x → {_new_lev}x "
            f"(止损 {_stop_dist:.2%} ≥ 爆仓安全距离 {_liq_dist:.2%}×{_safety:.0%}, "
            f"降杠杆后爆仓距离 {1.0 / _new_lev - _mmr:.2%}×{_safety:.0%} 覆盖止损)"
        )
        signal.leverage = _new_lev
        _eff_leverage = _new_lev
        _liq_dist = 1.0 / max(_new_lev, 1) - _mmr
        if _stop_dist >= _liq_dist * _safety:
            # 冗余保护: 数学上 floor 后不应发生, 防御浮点/参数异常
            logger.error(
                f"[LiqCheck] {signal.inst_id} 降杠杆后仍不满足止损优先，拒单")
            return None
    else:
        logger.debug(
            f"[LiqCheck] {signal.inst_id} 通过: 止损距离 {_stop_dist:.2%} "
            f"< 爆仓距离 {_liq_dist:.2%} × {_safety:.0%} (杠杆 {_eff_leverage}x)"
        )

    # ---- 确定合约参数 ----
    # USDT 本位永续合约面值因币种而异，默认 0.01（如 BTC ctVal=0.01）
    # 当 instrument_info 获取失败时，用 0.01 比 1.0 更安全
    ct_val = 0.01
    if instrument_info:
        ct_val = float(instrument_info.get("ctVal", "0.01"))
    else:
        # 修复 H8: instrument_info 为 None 时无法获取合约面值，跳过交易
        # 默认 ct_val=0.01 仅 BTC 正确，altcoin 会严重超估仓位
        logger.error(f"Cannot get instrument info for {signal.inst_id}, aborting signal")
        return None

    min_sz = float(instrument_info.get("minSz", "1"))
    # 从 lotSz 推断精度
    lot_sz = instrument_info.get("lotSz", "1")
    if "." in lot_sz:
        sz_precision = len(lot_sz.split(".")[1])
    else:
        sz_precision = 0

    # ---- 计算仓位 ----
    sz, margin = calculate_position_size(
        equity=equity,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        risk_pct=risk_cfg["risk_per_trade_pct"],
        leverage=_eff_leverage,
        margin_pct=risk_cfg["margin_pct"],
        direction=signal.position_side,
        contract_value=ct_val,
        min_sz=min_sz,
        sz_precision=sz_precision,
        sizing=risk_cfg.get("position_sizing", "risk"),
        enforce_risk_cap=risk_cfg.get("enforce_risk_cap", True),
    )

    if sz <= 0:
        logger.warning(f"Calculated sz=0 for {signal.inst_id}, skip")
        return None

    # 修复 P2-5: 按 lotSz 整数倍对齐（floor）。此前仅按小数位截断，
    # 对 lotSz 非 10 的负幂的币种（如 lotSz=10/3）可能被 OKX 拒单，
    # 且拒单不报错仅 warning → 漏执行。
    try:
        lot_float = float(lot_sz)
        if lot_float > 0:
            sz = math.floor(sz / lot_float) * lot_float
            if sz < min_sz:
                logger.warning(
                    f"Aligned sz={sz} < min {min_sz} for {signal.inst_id}, skip"
                )
                return None
    except (TypeError, ValueError):
        pass

    # ---- 确定 side 和 posSide ----
    if signal.position_side == "long":
        side = "buy"
        pos_side = "long"
    else:
        side = "sell"
        pos_side = "short"

    # ---- 格式化价格精度 ----
    # OKX 要求价格和数量为字符串
    tick_sz = "0.1"
    if instrument_info:
        tick_sz = instrument_info.get("tickSz", "0.1")

    px_precision = 0
    if "." in tick_sz:
        px_precision = len(tick_sz.split(".")[1])

    entry_px = f"{signal.entry_price:.{px_precision}f}"
    tp_px = f"{signal.take_profit:.{px_precision}f}"
    sl_px = f"{signal.stop_loss:.{px_precision}f}"
    sz_str = f"{sz:.{sz_precision}f}"

    logger.info(
        f"\n{'='*60}\n"
        f"  SIGNAL: {signal.inst_id} {signal.position_side.upper()}\n"
        f"  Entry:  {entry_px}  |  TP: {tp_px}  |  SL: {sl_px}\n"
        f"  Size:   {sz_str}  |  Leverage: {signal.leverage}x  |  Margin: {margin:.2f}\n"
        f"  Score:  {signal.score:.2f}  |  {signal.reason}\n"
        f"{'='*60}"
    )

    # ---- 设置杠杆 ----
    # OKX 合约交易必须先设置杠杆，否则使用默认 1x
    # 修复 P1-1: 必须用封顶后的 _eff_leverage —— 此前仓位计算用封顶值、
    # set_leverage 用未封顶原始值，导致账户真实杠杆高于风控假设
    # （强平价更近 + ADL 顺位更前），max_position_leverage 形同虚设。
    leverage_ok = client.set_leverage(
        inst_id=signal.inst_id,
        lever=_eff_leverage,
        mgn_mode=risk_cfg["margin_mode"],
        pos_side=pos_side,  # 修复: 双向持仓模式下必须传入 posSide
    )
    if not leverage_ok:
        logger.error(f"Cannot set leverage for {signal.inst_id}, aborting signal")
        return None

    # ---- 下单 ----
    # 原则: 不追涨杀跌 — 只限价挂单等待 FVG 回补价位成交，绝不转市价追单。
    # 限价单创建失败（如价格保护/网络）则放弃本轮，等待下一轮信号。
    _sl_slippage = 0.005  # 0.5% 最大滑点
    _sl_val = signal.stop_loss  # 浮点止损价
    if pos_side == "long":
        sl_px_safe = f"{_sl_val * (1 - _sl_slippage):.{px_precision}f}"
    else:
        sl_px_safe = f"{_sl_val * (1 + _sl_slippage):.{px_precision}f}"

    # 限价单：只在 FVG 理想入场价挂单，等待回补成交
    # 深挂 conditional 触发单 (2026-08-10 用户要求): 回补位距现价超过有效
    # 阈值(ATR 挂钩)时, 不直接挂深限价(空转/逆向选择), 改挂触发单 — 价格
    # 先走到距回补位一个阈值窗口的触发位, 触发后才以回补位限价进场。
    if (getattr(signal, "use_conditional_entry", False)
            and float(getattr(signal, "entry_trigger_px", 0) or 0) > 0):
        ord_id = client.place_plan_order(
            inst_id=signal.inst_id,
            td_mode=risk_cfg["margin_mode"],
            side=side,
            pos_side=pos_side,
            sz=sz_str,
            trigger_px=float(signal.entry_trigger_px),
            ord_px=float(signal.entry_price),
        )
        if ord_id:
            logger.info(
                f"[PlanEntry] {signal.inst_id} 深挂改 conditional 触发单 "
                f"trigger={signal.entry_trigger_px:.6g} entry={entry_px}")
    else:
        ord_id = client.place_order(
            inst_id=signal.inst_id,
            side=side,
            pos_side=pos_side,
            sz=sz_str,
            px=entry_px,
            ord_type="limit",
            td_mode=risk_cfg["margin_mode"],
        )
    if not ord_id:
        logger.warning(
            f"[Limit] {signal.inst_id} 挂单创建失败，放弃本轮（不追市价）"
        )
        return None

    # 主单成功后独立挂止盈止损（限价单未成交时挂单可能失败，
    # 交由后续轮次的 trailing 逻辑补挂）
    algo_id = client.place_algo_order(
        inst_id=signal.inst_id,
        td_mode=risk_cfg["margin_mode"],
        side="sell" if pos_side == "long" else "buy",
        pos_side=pos_side,
        sz=sz_str,
        ord_type="conditional",
        tp_trigger_px=tp_px,
        tp_ord_px="-1",
        sl_trigger_px=sl_px,
        sl_ord_px=sl_px_safe,
        reduce_only=True,
    )
    if not algo_id:
        logger.warning(
            f"[Algo] {signal.inst_id} 止盈止损单未挂上（限价主单可能未成交），"
            f"将由后续监控补挂；主单 ord_id={ord_id}"
        )

    return ord_id


# ---------------------------------------------------------------------------
# 持仓监控
# ---------------------------------------------------------------------------

def monitor_positions(
    client: OKXClient,
    state_manager: StateManager,
    config: dict,
) -> Dict[str, dict]:
    """监控当前持仓状态。

    检查：
      - 持仓盈亏
      - 止盈止损是否触发
      - 更新状态追踪

    Returns:
        {inst_id: position_info}
    """
    positions = client.get_positions()
    # 修复 P0-B (fail-closed): get_positions 返回 None（查询失败）时抛异常，
    # 由主循环捕获后跳过本轮。绝不能把"API 故障"当"无持仓"——否则
    # active_count 误判 0、风控放行、已满仓时超限开仓。
    if positions is None:
        raise OKXQueryError("get_positions failed (fail-closed)")
    risk_cfg = config["risk"]
    result = {}

    for pos in positions:
        inst_id = pos.get("instId", "")
        pos_side = pos.get("posSide", "")
        pos_sz = float(pos.get("pos", "0"))
        avg_px = float(pos.get("avgPx", "0"))
        mark_px = float(pos.get("markPx", "0"))
        upl = float(pos.get("upl", "0"))          # 未实现盈亏
        upl_ratio = float(pos.get("uplRatio", "0"))
        margin = float(pos.get("margin", "0"))
        lever = float(pos.get("lever", "0"))

        result[inst_id] = {
            "pos_side": pos_side,
            # 修复 P1-4: size 归一化为 abs()。OKX 文档约定 isolated 下 pos 恒正、
            # 方向看 posSide；但 cross 模式下空头 pos 为负。全链路统一 abs，
            # 杜绝 `size > 0` 判空头永远为假 → 空头被风控忽略/裸奔。
            "size": abs(pos_sz),
            "avg_px": avg_px,
            "mark_px": mark_px,
            "upl": upl,
            "upl_ratio_pct": upl_ratio * 100,
            "margin": margin,
            "leverage": lever,
            "c_time": int(pos.get("cTime", "0")) / 1000.0 if pos.get("cTime") else 0.0,
        }

        if abs(pos_sz) > 0:
            logger.info(
                f"[POS] {inst_id} {pos_side} | "
                f"Sz={pos_sz} | Avg={avg_px:.2f} | Mark={mark_px:.2f} | "
                f"PnL={upl:.2f} ({upl_ratio*100:.2f}%) | "
                f"Margin={margin:.2f} | Lever={lever}x"
            )

    # 获取当前权益
    equity = client.get_total_equity() or 0.0

    # 检查平仓变动
    prev_active = len(state_manager.state.active_signals)
    current_active = len([p for p in positions if abs(float(p.get("pos", "0"))) > 0])

    # 修复 H5: 同步 active_signals 与交易所实际持仓
    # 清理已在交易所不存在的失效信号条目
    exchange_pos_ids = {p.get("instId", "") for p in positions if abs(float(p.get("pos", "0"))) > 0}
    stale_keys = []
    for key, sig in list(state_manager.state.active_signals.items()):
        inst_id = sig.get("inst_id") or sig.get("instId", "")
        if inst_id not in exchange_pos_ids:
            stale_keys.append(key)
            logger.warning(f"[Sync] 清理失效信号: {inst_id} (key={key})，交易所已无此持仓")
    for key in stale_keys:
        del state_manager.state.active_signals[key]

    # active_signals: {inst_id: position_info} — indexed by instrument ID, not order ID
    # 修复: 存储 normalized 格式（result 中的字段名），而非原始 OKX API 响应
    # 原始响应用 instId/pos/avgPx 等驼峰命名，而读取方期望 inst_id/size/avg_px 等下划线命名
    # 不一致导致 .get("inst_id") 返回 None，stale key 清理逻辑静默失效
    # 修复 QE-1: 合并保留自定义字段（如 signal_id / confidence / master_score 等），
    # 避免完整覆盖导致 SignalPerformanceTracker 无法关联持仓与信号。
    with state_manager._lock:
        new_active: Dict[str, dict] = {}
        for inst_id, info in result.items():
            if info.get("size", 0) <= 0:
                continue
            prev = state_manager.state.active_signals.get(inst_id, {})
            merged = dict(prev)
            merged.update(info)
            merged["inst_id"] = inst_id
            new_active[inst_id] = merged
        state_manager.state.active_signals = new_active

    state_manager.state.last_equity = equity

    # 修复 H7: 使用 daily_start_equity 作为基准（与 risk_gate 一致）
    max_daily_loss = state_manager.state.daily_start_equity * risk_cfg["max_daily_loss_pct"] / 100.0 if state_manager.state.daily_start_equity else 0
    # 修复 C3+H6: daily_loss 现在追踪累计已实现 PnL，检查是否低于负阈值
    if state_manager.state.daily_loss <= -max_daily_loss and max_daily_loss > 0:
        logger.warning(
            f"!!! DAILY LOSS LIMIT REACHED !!! "
            f"累计已实现 PnL: {state_manager.state.daily_loss:.2f} <= -{max_daily_loss:.2f}"
        )

    return result


# ---------------------------------------------------------------------------
# 挂单管理
# ---------------------------------------------------------------------------

def manage_pending_orders(
    client: OKXClient,
    current_price: float,
    signal: Optional[Signal] = None,
    stale_minutes: int = 30,
    limit_order_timeout_minutes: int = 15,  # 修复 P0-1: 限价单短超时，防止永远不成交
) -> Tuple[int, bool]:
    """管理挂单：取消过期订单，避免重复下单。

    Args:
        client: OKX 客户端
        current_price: 当前市场价格（用于价格偏离检测）
        signal: 如果有新信号，检查是否已存在相同方向的挂单
        stale_minutes: 超过此时间的挂单视为过期（通用）
        limit_order_timeout_minutes: 限价单超时时间（比 stale_minutes 更短，防止错过行情）

    Returns:
        (cancelled_count, should_skip_signal) — should_skip_signal=True 表示应跳过新信号
    """
    orders = client.get_pending_orders()
    # 修复 P0-B: 挂单查询失败返回 None 时，本轮不做挂单管理
    # （不撤单、不判冲突），避免把"查询失败"当"无挂单"而重复挂单。
    if orders is None:
        logger.warning("get_pending_orders failed (fail-closed)，本轮跳过挂单管理")
        return 0, False
    cancelled = 0
    should_skip = False
    now = time.time() * 1000  # ms

    def _cancel_orphan_protection(inst_id: str) -> int:
        """撤销该币种失效的保护单（oco/conditional）。

        场景: OKX 允许在无持仓时挂 reduceOnly 保护单，因此开仓限价单
        尚未成交时保护单已挂上。若限价单超时被撤且该币种无持仓，
        保护单会成为孤儿单残留交易所（此前 FLOW/SATS 残留的根因）。
        仅在该币种无持仓时联动撤销，避免误撤真实持仓的保护。
        """
        _n = 0
        try:
            _has_pos = any(
                float(p.get("pos", 0)) != 0
                for p in client.get_positions(inst_id=inst_id)
            )
        except Exception:
            _has_pos = True  # 查询失败时保守处理：不撤保护单
        if _has_pos:
            return 0
        for _ot in ("oco", "conditional"):
            try:
                for _a in client.get_algo_orders(
                    inst_id=inst_id, inst_type="SWAP", ord_type=_ot
                ):
                    if _a.get("state") in ("live", "effective"):
                        _aid = _a.get("algoId", "")
                        if client.cancel_algo_order(_aid, inst_id):
                            _n += 1
                            logger.info(f"[Pending] 联动撤销孤儿保护单 {_aid} ({inst_id})")
            except Exception as e:
                logger.debug(f"[Pending] 查询保护单失败 {inst_id}: {e}")
        return _n

    for order in orders:
        ord_id = order.get("ordId", "")
        inst_id = order.get("instId", "")
        pos_side = order.get("posSide", "")
        ord_type = order.get("ordType", "")
        px = float(order.get("px", "0"))
        create_time = int(order.get("cTime", "0"))

        age_minutes = (now - create_time) / 60000.0 if create_time > 0 else 0

        # 修复 P0-1: 限价单短超时 — 价格不回踩就取消，改追价或放弃
        # 顶级交易员做法: 挂单 N 根 K 线不成交 → 撤单，不等 60 分钟
        if ord_type == "limit" and age_minutes > limit_order_timeout_minutes:
            logger.info(f"Cancelling stale limit order {ord_id} ({inst_id}, age={age_minutes:.0f}m > {limit_order_timeout_minutes}m)")
            if client.cancel_order(inst_id, ord_id):
                cancelled += 1
                # 修复: 联动撤销孤儿保护单，防止 algo 单残留交易所
                cancelled += _cancel_orphan_protection(inst_id)
            continue

        # 检查是否过期
        if age_minutes > stale_minutes:
            logger.info(f"Cancelling stale order {ord_id} ({inst_id}, age={age_minutes:.0f}m)")
            if client.cancel_order(inst_id, ord_id):
                cancelled += 1
            continue

        # 如果有新信号，检查冲突
        if signal and inst_id == signal.inst_id and pos_side == signal.position_side:
            logger.info(f"Similar pending order exists for {inst_id}, skip new signal")
            should_skip = True
            continue

        # 如果价格远离挂单价超过阈值，取消
        if px > 0 and current_price > 0:
            deviation = abs(current_price - px) / px
            if deviation > 0.05:  # 5% 偏离
                logger.info(f"Cancelling deviated order {ord_id} ({inst_id}, "
                            f"deviation={deviation*100:.2f}%)")
                if client.cancel_order(inst_id, ord_id):
                    cancelled += 1
                    # 修复: 联动撤销孤儿保护单，防止 algo 单残留交易所
                    cancelled += _cancel_orphan_protection(inst_id)

    return cancelled, should_skip


# ---------------------------------------------------------------------------
# 获取合约标的列表
# ---------------------------------------------------------------------------

def compute_movers(tickers: list, config: dict) -> List[dict]:
    """从全量 SWAP tickers 计算 24h 涨跌幅榜/跌幅榜 (2026-08-09 用户要求)。

    OKX 涨幅榜/跌幅榜里动辄 ±10%+ 的极端波动币种，正是 FVG Hunter 硬门禁
    (ADX≥25 + ATR≥2%) 的理想猎场。原系统只按成交量排序选币，涨跌幅榜币种
    排到 100 名之后根本进不了扫描队列 —— 导致"总是找不到好行情"。

    使用同一份 tickers 数据（不额外调用 API），与 get_tradable_coins 共享。

    Args:
        tickers: OKXClient.get_tickers() 原始返回 (全量 SWAP)
        config: 完整配置 (读取 market_movers 段)

    Returns:
        按 |24h 涨跌幅| 降序的榜单币种 (结构同 get_tradable_coins 返回，
        另含 move_pct 字段)。market_movers.enabled=false 时返回 []。
    """
    mv_cfg = config.get("market_movers", {}) or {}
    if not mv_cfg.get("enabled", True):
        return []
    count = int(mv_cfg.get("count", 20) or 20)
    min_move = float(mv_cfg.get("min_move_pct", 8.0) or 8.0)
    min_vol = float(mv_cfg.get("min_volume_24h_usd", 1_000_000) or 1_000_000)

    rows = []
    for t in tickers:
        inst_id = t.get("instId", "")
        if not inst_id.endswith("-USDT-SWAP"):
            continue
        last = float(t.get("last", "0") or 0)
        open24 = float(t.get("open24h", "0") or 0)
        vol24h = float(t.get("volCcy24h", "0") or 0)
        if last <= 0 or open24 <= 0 or vol24h < min_vol:
            continue
        pct = (last - open24) / open24 * 100.0
        if abs(pct) < min_move:
            continue
        rows.append({
            "instId": inst_id,
            "last": last,
            "vol24h": vol24h,
            "bidPx": float(t.get("bidPx", "0") or 0),
            "askPx": float(t.get("askPx", "0") or 0),
            "high24h": float(t.get("high24h", "0") or 0),
            "low24h": float(t.get("low24h", "0") or 0),
            "move_pct": pct,   # 24h 涨跌幅 (%), 供优先级排序/日志
        })
    rows.sort(key=lambda x: abs(x["move_pct"]), reverse=True)
    return rows[:count]


def get_tradable_coins(
    client: OKXClient,
    config: dict,
) -> List[dict]:
    """获取可交易的 USDT 本位永续合约列表，按成交量排序。

    优先级 (2026-08-09 用户要求):
        1. 24h 涨跌幅榜/跌幅榜币种 (极端波动, FVG Hunter 理想猎场) 排最前
        2. 其余按成交量降序

    Returns:
        [{"instId": "BTC-USDT-SWAP", "last": 50000, "vol24h": 1e9, ...}, ...]
        榜单币种额外带 move_pct 字段。
    """
    tickers = client.get_tickers(inst_type="SWAP")
    limit = config["agent"].get("coin_scan_limit", 100)

    # 24h 涨跌幅榜/跌幅榜优先 (同一份 tickers 数据, 零额外 API 调用)
    movers = compute_movers(tickers, config)
    mover_ids = {m["instId"] for m in movers}

    # 只保留 USDT 本位永续合约
    usdt_swaps = []
    for t in tickers:
        inst_id = t.get("instId", "")
        if not inst_id.endswith("-USDT-SWAP"):
            continue
        if inst_id in mover_ids:
            continue  # 涨跌幅榜币种已单独收集

        vol24h = float(t.get("volCcy24h", "0"))
        if vol24h < config["strategy"]["min_volume_24h_usd"]:
            continue

        usdt_swaps.append({
            "instId": inst_id,
            "last": float(t.get("last", "0")),
            "vol24h": vol24h,
            "bidPx": float(t.get("bidPx", "0")),
            "askPx": float(t.get("askPx", "0")),
            "high24h": float(t.get("high24h", "0")),
            "low24h": float(t.get("low24h", "0")),
        })

    # 按成交量降序
    usdt_swaps.sort(key=lambda x: x["vol24h"], reverse=True)
    if movers:
        logger.info(
            f"[Movers] {len(movers)} 个 24h 涨跌幅榜币种优先进入扫描队列: "
            + ", ".join(f"{m['instId']}({m['move_pct']:+.1f}%)" for m in movers[:5])
        )
    return (movers + usdt_swaps)[:limit]


# ---------------------------------------------------------------------------
# 价差计算
# ---------------------------------------------------------------------------

def calculate_spread(bid: float, ask: float) -> float:
    """计算买卖价差百分比。"""
    if ask <= 0 or bid <= 0:
        return 0.0
    return (ask - bid) / ask * 100.0


# ---------------------------------------------------------------------------
# 打印统计摘要
# ---------------------------------------------------------------------------

def print_summary(state: AgentState, equity: float):
    """打印运行统计。"""
    if state.initial_equity <= 0:
        return

    # 修复 Bug 49: print_summary 使用 break_even 单独统计，
    # 胜率分母剔除保本交易，避免失真
    # 修复：微小初始权益时百分比溢出 — 初始权益 < 1 USDT 时不显示百分比
    if state.initial_equity >= 1.0:
        total_pnl_pct = (equity - state.initial_equity) / state.initial_equity * 100
        pnl_pct_str = f" ({total_pnl_pct:+.2f}%)"
    else:
        pnl_pct_str = ""
    decisive_trades = state.winning_trades + state.losing_trades
    all_trades = decisive_trades + state.break_even_trades
    win_rate = state.winning_trades / decisive_trades * 100 \
        if decisive_trades > 0 else 0

    logger.info(
        f"\n{'='*50}\n"
        f"  SUMMARY\n"
        f"  Equity:     {equity:.2f} USDT\n"
        f"  PnL:        {equity - state.initial_equity:+.2f}{pnl_pct_str}\n"
        f"  Trades:     {all_trades} "
        f"(W:{state.winning_trades} L:{state.losing_trades} BE:{state.break_even_trades})\n"
        f"  Win Rate:   {win_rate:.1f}%\n"
        f"  Withdrawals:{state.withdrawal_count}\n"
        f"{'='*50}"
    )