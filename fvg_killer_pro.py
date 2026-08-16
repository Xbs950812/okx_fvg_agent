# -*- coding: utf-8 -*-
"""
FVG KILLER PRO — 高级模块（私有，授权分发）。

包含 v3.3 增强版的核心增值逻辑；开源核心版(fvg_killer)不含本文件，
各调用点通过 try-import 守卫优雅降级到 v3.2 行为：

  - _TokenBucket / create_rate_limiter   全局 API 限流令牌桶
  - _ewma_window_stats / _kelly_f_from_stats / rolling_kelly_risk_pct
                                         滚动分数 Kelly 风险上限（探索→利用,
                                         EWMA 输入端平滑）
  - check_order_book_liquidity           开仓前订单簿流动性检查
  - vol_targeting_scale                  波动率目标仓位缩放
  - reconcile_funding_fees               资金费率实际对账（bills）
  - startup_reconciliation               启动三方对账（持仓↔状态↔保护单）

授权：随 FVG KILLER PRO 许可证分发，禁止公开再分发本文件。
核心版 README 的 PRO 介绍段落指向获取渠道。
"""

import logging
import math
import threading
import time
from typing import Optional, List, Tuple, Dict

import numpy as np

logger = logging.getLogger(__name__)

PRO_VERSION = "1.0.0"


# ===========================================================================
# 全局 API 限流令牌桶
# ===========================================================================

class _TokenBucket:
    """全局 API 限流令牌桶（经典算法，capacity 突发容量 / rate 每秒补充）。

    acquire() 无令牌时阻塞等待，保证 QPS 不超过配置值。与被动重试互补：
    令牌桶在源头削峰，重试兜底偶发失败。
    """

    def __init__(self, rate: float, capacity: float):
        self.rate = max(0.1, float(rate))
        self.capacity = max(1.0, float(capacity))
        self._tokens = self.capacity
        self._last = time.time()
        self._lock = threading.Lock()

    def acquire(self):
        """获取一个令牌（无令牌时阻塞至补充）。"""
        while True:
            with self._lock:
                now = time.time()
                self._tokens = min(
                    self.capacity,
                    self._tokens + (now - self._last) * self.rate,
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            time.sleep(min(wait, 0.2))


def create_rate_limiter(
    rate_limit_cfg: Optional[dict],
    dry_run: bool,
) -> Optional[_TokenBucket]:
    """按配置创建全局令牌桶（由 okx_client.OKXClient.__init__ 调用）。

    dry_run 纸面模式不启用（行情端点已被 coin_tracker 批间隔节流）。
    返回 None 表示不启用。
    """
    rl = rate_limit_cfg if isinstance(rate_limit_cfg, dict) else {}
    if dry_run or not rl.get("enabled", True):
        return None
    try:
        qps = float(rl.get("max_qps", 10) or 10)
        burst = float(rl.get("burst_capacity", 20) or 20)
        bucket = _TokenBucket(qps, burst)
        logger.info(f"[RateLimit] 全局令牌桶已启用: {qps:.1f} QPS, "
                    f"突发容量 {burst:.0f}")
        return bucket
    except (TypeError, ValueError):
        return None


# ===========================================================================
# 滚动分数 Kelly（探索 → 利用，EWMA 平滑）
# ===========================================================================

def _ewma_window_stats(
    pnls: List[float],
    lam: float,
) -> Tuple[float, float, float]:
    """指数加权窗口统计（输入端平滑，共享时钟衰减）。

    每笔 decisive 交易（pnl≠0）推进一次衰减，胜/负累加器同步衰减后累加
    新观测；保本交易跳过（不推进时钟）。pnls 必须按时间升序（旧→新）。
    """
    sw = 0.0
    nw = 0.0
    sl = 0.0
    nl = 0.0
    for pnl in pnls:
        if pnl == 0.0:
            continue
        sw *= lam
        nw *= lam
        sl *= lam
        nl *= lam
        if pnl > 0:
            sw += pnl
            nw += 1.0
        else:
            sl += -pnl
            nl += 1.0
    d = nw + nl
    p = (nw / d) if d > 0 else 0.0
    avg_win = (sw / nw) if nw > 0 else 0.0
    avg_loss = (sl / nl) if nl > 0 else avg_win * 0.5
    return p, avg_win, avg_loss


def _kelly_f_from_stats(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """从 (p, avg_win, avg_loss) 计算 f*，裁剪链与 compute_kelly 完全一致。"""
    if avg_loss == 0:
        avg_loss = 1.0
    b = (avg_win / avg_loss) if avg_loss > 0 else 1.0
    b = max(0.01, min(b, 100.0))
    q = 1 - win_rate
    if win_rate >= 0.999:
        kelly_f = 0.25
    elif win_rate <= 0.001:
        kelly_f = 0.0
    else:
        kelly_f = (win_rate * b - q) / b if b > 0 else 0.0
    return max(0.0, min(kelly_f, 0.5))


def rolling_kelly_risk_pct(
    recent_pnl: List[float],
    base_risk_pct: float,
    risk_cfg: Optional[dict] = None,
) -> Tuple[Optional[float], dict]:
    """滚动 Kelly 风险上限（小资金翻倍协议: 探索→利用）。

    样本量规则:
      - decisive < min_samples(10): 返回 None（不约束）
      - < sample_full_kelly(50): 1/4 Kelly 探索档
      - >= 50: 1/2 Kelly 利用档
      - 裁剪 [min_risk_pct(1.0), max_risk_pct(默认=base)]

    EWMA 平滑: ewma_lambda ∈ (0,1) 指数加权窗口统计，默认 0.97
    （有效记忆≈33 笔，消除窗口边界驱逐跳变+加快体制切换响应）；
    非法值回退平权窗口。
    """
    rk = (risk_cfg or {}).get("rolling_kelly", {}) or {}
    if not rk.get("enabled", True):
        return None, {"disabled": True}
    try:
        window = max(1, int(rk.get("window", 100) or 100))
    except (TypeError, ValueError):
        window = 100
    try:
        min_samples = max(1, int(rk.get("min_samples", 10) or 10))
    except (TypeError, ValueError):
        min_samples = 10
    try:
        sample_full = max(1, int(rk.get("sample_full_kelly", 50) or 50))
    except (TypeError, ValueError):
        sample_full = 50
    try:
        min_risk = float(rk.get("min_risk_pct", 1.0) or 0)
    except (TypeError, ValueError):
        min_risk = 1.0
    try:
        _mr = rk.get("max_risk_pct")
        max_risk = float(_mr) if (_mr is not None and float(_mr) > 0) \
            else float(base_risk_pct or 0)
    except (TypeError, ValueError):
        max_risk = float(base_risk_pct or 0)
    if max_risk <= 0:
        return None, {"disabled": True, "reason": "base_risk_pct<=0"}

    pnls = [float(p) for p in (recent_pnl or [])[-window:]]
    decisive = sum(1 for p in pnls if p != 0.0)
    if decisive < min_samples:
        return None, {"samples": decisive, "reason": "insufficient_samples"}

    try:
        _lam_raw = rk.get("ewma_lambda", 0.97)
        lam = float(_lam_raw) if _lam_raw is not None else 0.0
    except (TypeError, ValueError):
        lam = 0.0
    if not (0.0 < lam < 1.0):
        lam = 0.0

    if lam > 0.0:
        p, avg_win, avg_loss = _ewma_window_stats(pnls, lam)
        kelly_f = round(_kelly_f_from_stats(p, avg_win, avg_loss), 4)
        win_rate_r = round(p, 3)
        avg_win_r = round(avg_win, 3)
        avg_loss_r = round(avg_loss, 3)
    else:
        # 平权口径（等价于对窗口全量调用 compute_kelly）
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        w, l = len(wins), len(losses)
        p = (w / decisive) if decisive > 0 else 0.0
        avg_win = (sum(wins) / w) if w else 0.0
        avg_loss = (sum(-x for x in losses) / l) if l else avg_win * 0.5
        kelly_f = round(_kelly_f_from_stats(p, avg_win, avg_loss), 4)
        win_rate_r = round(p, 3)
        avg_win_r = round(avg_win, 3)
        avg_loss_r = round(avg_loss, 3)

    tier = "half" if decisive >= sample_full else "quarter"
    frac = 0.5 if tier == "half" else 0.25
    risk = kelly_f * frac * 100.0
    risk = max(min(min_risk, max_risk), min(risk, max_risk))
    return risk, {
        "samples": decisive,
        "tier": tier,
        "kelly_f": kelly_f,
        "win_rate": win_rate_r,
        "avg_win": avg_win_r,
        "avg_loss": avg_loss_r,
        "ewma_lambda": lam,
    }


# ===========================================================================
# 订单簿流动性检查
# ===========================================================================

def check_order_book_liquidity(
    client,
    inst_id: str,
    pos_side: str,
    notional_usd: float,
    ct_val: float,
    config: dict,
) -> Tuple[bool, str]:
    """开仓前订单簿深度/流动性检查。

    名义仓位相对前 N 档对侧深度过大时拒绝开仓（崩盘薄书滑点保护）。
    做多查 bids（平仓卖单吃买盘），做空查 asks。查询失败 fail-open。
    """
    _ob_cfg = (config.get("risk", {}) or {}).get("order_book_depth", {}) or {}
    if not _ob_cfg.get("enabled", True):
        return True, ""
    _max_ratio = float(_ob_cfg.get("max_notional_depth_ratio", 0.05) or 0)
    _levels = int(_ob_cfg.get("depth_levels", 10) or 10)
    if _max_ratio <= 0 or notional_usd <= 0:
        return True, ""
    try:
        _depth = client.get_order_book_depth_usd(
            inst_id, levels=_levels, ct_val=ct_val)
    except Exception as _e:
        logger.warning(f"[Liquidity] {inst_id} 订单簿深度查询失败，放行: {_e}")
        return True, ""
    if not _depth:
        return True, ""
    _side_depth = (_depth.get("bids_usd", 0.0) if pos_side == "long"
                   else _depth.get("asks_usd", 0.0))
    if _side_depth <= 0:
        return True, ""
    _ratio = notional_usd / _side_depth
    if _ratio > _max_ratio:
        return False, (
            f"[Liquidity] {inst_id} 名义 {notional_usd:.2f} USDT / 对侧前 "
            f"{_levels} 档深度 {_side_depth:.2f} USDT = {_ratio:.1%} "
            f"> {_max_ratio:.0%}，订单簿过薄，拒绝开仓")
    return True, ""


# ===========================================================================
# 波动率目标仓位
# ===========================================================================

def vol_targeting_scale(
    candles_1h: List,
    entry_price: float,
    config: dict,
) -> float:
    """波动率目标仓位缩放因子: scale = target_vol_pct / ATR%, 裁剪 [min,max]。

    数据不足返回 1.0（不缩放，fail-open）。
    """
    _vt_cfg = (config.get("risk", {}) or {}).get("vol_targeting", {}) or {}
    if not _vt_cfg.get("enabled", True):
        return 1.0
    _target = float(_vt_cfg.get("target_vol_pct", 2.0) or 0)
    _min_s = float(_vt_cfg.get("min_scale", 0.5) or 0.5)
    _max_s = float(_vt_cfg.get("max_scale", 1.5) or 1.5)
    if _target <= 0 or entry_price <= 0:
        return 1.0
    _period = int(config.get("strategy", {}).get("atr_period", 14) or 14)
    if not candles_1h or len(candles_1h) < _period + 1:
        return 1.0
    try:
        _trs = []
        for i in range(1, len(candles_1h)):
            h = float(candles_1h[i].high)
            l = float(candles_1h[i].low)
            pc = float(candles_1h[i - 1].close)
            _trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        if not _trs:
            return 1.0
        _atr = float(np.mean(_trs[:_period]))
        for i in range(_period, len(_trs)):
            _atr = (_atr * (_period - 1) + _trs[i]) / _period
        if _atr <= 0:
            return 1.0
        _atr_pct = _atr / entry_price * 100.0
        if _atr_pct <= 0:
            return 1.0
        return max(_min_s, min(_max_s, _target / _atr_pct))
    except (ValueError, TypeError, ZeroDivisionError, IndexError):
        return 1.0


# ===========================================================================
# 对账（资金费率 / 启动三方）
# ===========================================================================

def reconcile_funding_fees(
    client,
    positions: Dict[str, dict],
    config: dict,
    estimate_funding_cost_fn=None,
) -> None:
    """资金费率实际对账: 拉 bills(type=8) 与估算值对账（read-only，仅实盘）。"""
    if config.get("agent", {}).get("dry_run", False):
        return
    _rc_cfg = (config.get("agent", {}) or {}).get("reconciliation", {}) or {}
    if not _rc_cfg.get("funding_fee_enabled", True):
        return
    if not positions:
        return
    _now_ms = int(time.time() * 1000)
    for _inst, _pos in positions.items():
        try:
            _size = float(_pos.get("size", 0) or 0)
            _avg_px = float(_pos.get("avg_px", 0) or 0)
            _side = str(_pos.get("pos_side", "long") or "long")
            _ctime_s = float(_pos.get("c_time", 0) or 0)
            if _size <= 0 or _avg_px <= 0 or _ctime_s <= 0:
                continue
            _begin = str(int(_ctime_s * 1000))
            _bills = client.get_bills(
                inst_type="SWAP", limit=100, begin=_begin,
                end=str(_now_ms), bill_type="8")
            _actual = 0.0
            _n = 0
            for _b in (_bills or []):
                if str(_b.get("instId", "")) != _inst:
                    continue
                try:
                    _actual += float(_b.get("pnl", 0) or 0)
                except (TypeError, ValueError):
                    continue
                _n += 1
            _hold_hours = (_now_ms / 1000.0 - _ctime_s) / 3600.0
            if _hold_hours <= 0:
                continue
            _est = 0.0
            if estimate_funding_cost_fn is not None:
                _est = estimate_funding_cost_fn(
                    client, _inst, _size, _avg_px, _hold_hours, pos_side=_side)
            _diff = _actual - _est
            _msg = (f"[FundingReconcile] {_inst} {_side} 持仓 {_hold_hours:.1f}h: "
                    f"实际资金费 {_actual:+.4f} vs 估算 {_est:+.4f} "
                    f"偏差 {_diff:+.4f} USDT ({_n} 笔结算)")
            if abs(_diff) < max(0.5, abs(_est) * 0.5):
                logger.info(_msg)
            else:
                logger.warning(_msg + " — 估算失真，时间止损成本判断需复核")
        except Exception as _e:
            logger.debug(f"[FundingReconcile] {_inst} 对账失败(忽略): {_e}")
            continue


def startup_reconciliation(
    client,
    state_manager,
    config: dict,
    trailing_algo_ids: Dict[str, str],
    last_submitted_sl: Dict[str, float],
) -> None:
    """启动三方对账: 交易所持仓 ↔ 本地 active_signals ↔ 保护单。

    read-only + 登记跟踪表；破坏性动作留给运行时 trailing 自愈。
    仅实盘执行。
    """
    if config.get("agent", {}).get("dry_run", False):
        return
    _rc_cfg = (config.get("agent", {}) or {}).get("reconciliation", {}) or {}
    if not _rc_cfg.get("startup_enabled", True):
        return

    try:
        positions = client.get_positions()
    except Exception as _e:
        logger.warning(f"[Reconcile] 启动对账 get_positions 失败: {_e}")
        return
    if positions is None:
        logger.warning("[Reconcile] 启动对账 get_positions 返回 None (fail-closed)，跳过")
        return

    live_pos = {
        str(p.get("instId", "")): p
        for p in positions
        if float(p.get("pos", "0") or 0) != 0
    }

    # 1) 本地 active_signals 对齐
    for _key in list(state_manager.state.active_signals.keys()):
        _sig = state_manager.state.active_signals.get(_key, {}) or {}
        _inst = str(_sig.get("inst_id") or _sig.get("instId", ""))
        if _inst and _inst not in live_pos:
            logger.warning(f"[Reconcile] 清理本地残留信号 {_inst} (交易所已无持仓)")
            del state_manager.state.active_signals[_key]

    # 2) 拉取生效保护单
    _prot_by_inst: Dict[str, dict] = {}
    for _ot in ("oco", "conditional"):
        try:
            _orders = client.get_algo_orders(inst_type="SWAP", ord_type=_ot) or []
        except Exception as _e:
            logger.debug(f"[Reconcile] 查询保护单 {_ot} 失败: {_e}")
            _orders = []
        for _a in _orders:
            if _a.get("state") not in ("live", "effective"):
                continue
            _inst = str(_a.get("instId", ""))
            if _inst:
                _prot_by_inst.setdefault(_inst, _a)

    # 3) 持仓 ↔ 保护单核对
    for _inst, _p in live_pos.items():
        _side = str(_p.get("posSide", "") or "")
        _prot = _prot_by_inst.get(_inst)
        if not _prot:
            logger.warning(
                f"[Reconcile] {_inst} {_side} 有持仓但无生效保护单 — "
                f"首轮 trailing 将补挂，请确认")
            continue
        _a_side = str(_prot.get("posSide", "") or "")
        if _a_side in ("", _side):
            _aid = _prot.get("algoId", "")
            trailing_algo_ids[_inst] = _aid
            try:
                last_submitted_sl[_inst] = float(
                    _prot.get("slTriggerPx", 0) or 0)
            except (TypeError, ValueError):
                last_submitted_sl[_inst] = 0.0
            logger.info(
                f"[Reconcile] {_inst} {_side} 保护单 {_aid} 已登记 "
                f"(SL={_prot.get('slTriggerPx')} TP={_prot.get('tpTriggerPx')})")
        else:
            logger.warning(
                f"[Reconcile] {_inst} 保护单 posSide={_a_side} ≠ 持仓 {_side}，"
                f"未登记（交由运行时自愈）")

    # 4) 孤儿保护单告警
    for _inst, _prot in _prot_by_inst.items():
        if _inst not in live_pos:
            logger.warning(
                f"[Reconcile] {_inst} 存在生效保护单 {_prot.get('algoId', '')} "
                f"但无持仓 — 疑似孤儿保护单，运行时 trailing 将处理")
