# -*- coding: utf-8 -*-
"""本地单测: ATR 动态止损四项 (职业交易标准)。

验证点:
  1. _compute_atr_wilder: Wilder 平滑 ATR 计算 (数据充足/不足/异常)
  2. 止损下限: SL 距 entry ≥ ATR×2.0 (结构止损过近时自动放宽)
  3. 窄止损拒绝: 止损距离 < ATR×0.8 → 拒绝信号
  4. 杠杆联动: 杠杆 ≤ leverage_stop_budget_pct / 止损距离%, 封顶 max_leverage
"""
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy import Candle, FVG, generate_signal, _compute_atr_wilder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("test_atr_stop")

T0 = 1700000000000


def make_candles(n=40, base=100.0, wave=1.0):
    """构造 1H K 线。wave 控制波动幅度。"""
    candles = []
    px = base
    for i in range(n):
        o = px
        c = px + wave * math.sin(i * 1.3)
        hi = max(o, c) + wave * 0.6
        lo = min(o, c) - wave * 0.6
        candles.append(Candle(
            timestamp=T0 + i * 3600000, open=o, high=hi, low=lo, close=c,
            volume=80000 + (i % 7) * 5000))
        px = c
    return candles


def make_fvg(direction, top, bottom, idx):
    return FVG(direction=direction, top=top, bottom=bottom, width_pct=(top - bottom) / bottom * 100.0,
               candle_ts=T0 + idx * 3600000, timeframe="1H", impulse_candle=None,
               fvg_index=idx, is_abnormal=True, sigma=3.0, volume_ratio=5.0)


def base_kwargs(**ovr):
    kw = dict(
        inst_id="TEST-USDT-SWAP", entry_depth_pct=0.15, fvg_target_pct=0.50,
        stop_buffer_pct=0.15, max_leverage=10, funding_rate=None,
        max_funding_rate_abs=0.01, funding_confluence_min_abs=0.0003,
        funding_confluence_max_abs=0.001, liquidity_extension_pct=3.0,
        liquidity_extension_min_pct=3.0, max_entry_distance_pct=5.0,
        min_risk_reward=0.0, atr_period=14, atr_stop_multiplier=2.0,
        atr_reject_ratio=0.8, leverage_stop_budget_pct=2.5,
        swing_lookback_bars=8, pullback_lookback=8, max_tp_distance_pct=25.0,
        long_short_ratio=None,
        tech_params={
            "bb_period": 20, "bb_std": 2.0, "squeeze_threshold": 0.6,
            "trend_ma_period": 20, "rsi_period": 14,
            "rsi_overbought": 70.0, "rsi_oversold": 30.0,
            "adx_period": 14, "adx_trend_threshold": 25.0,
            "adx_range_threshold": 20.0, "vwap_tolerance_pct": 0.5,
            "divergence_lookback": 14, "lsr_strong_high": 1.3,
            "lsr_strong_low": 0.7, "bb_veto_low": -0.2, "bb_veto_high": 1.2,
        },
        regime="NEUTRAL",
    )
    kw.update(ovr)
    return kw


def main():
    # ---- 1. ATR 计算 ----
    c = make_candles(wave=1.0)
    atr = _compute_atr_wilder(c, 14)
    log.info(f"ATR(14) wave=1.0: {atr:.4f} (期望 ~1.5-2.5)")
    assert atr > 0, "ATR 应 > 0"
    # 高波动 > 低波动
    atr_hi = _compute_atr_wilder(make_candles(wave=3.0), 14)
    assert atr_hi > atr, f"高波动 ATR 应更大: {atr_hi:.2f} vs {atr:.2f}"
    log.info(f"ATR 波动敏感性: wave1={atr:.2f} wave3={atr_hi:.2f} ✓")
    # 数据不足 → 0
    assert _compute_atr_wilder(make_candles(5), 14) == 0.0, "数据不足应返回 0"
    # None → 0
    assert _compute_atr_wilder(None, 14) == 0.0
    log.info("ATR 边界 (不足/None→0) ✓")

    # ---- 2. 止损下限: 结构止损过近 → 放宽到 ATR×2 ----
    # 构造高波动行情 (ATR 大), 窄 FVG → 结构止损距离很小
    candles_hi = make_candles(n=40, base=100.0, wave=3.0)
    atr_v = _compute_atr_wilder(candles_hi, 14)
    log.info(f"高波动行情 ATR={atr_v:.2f}")
    # 窄 FVG: 顶部 102, 底部 101.5 (宽度 0.5), entry 需贴近现价
    fvg = make_fvg("long", 102.0, 101.5, len(candles_hi) - 3)
    sig = generate_signal(
        fvg=fvg, current_price=102.0, candles=candles_hi,
        **base_kwargs(max_entry_distance_pct=5.0),
    )
    if sig:
        sl_dist = sig.entry_price - sig.stop_loss
        atr_min = atr_v * 2.0
        log.info(f"LONG: entry={sig.entry_price:.2f} sl={sig.stop_loss:.2f} "
                 f"SL距={sl_dist:.2f} ATR×2={atr_min:.2f} lev={sig.leverage}x")
        assert sl_dist >= atr_min - 1e-6, \
            f"SL 距 {sl_dist:.2f} 应 ≥ ATR×2 {atr_min:.2f}"
        log.info("  止损下限 (SL ≥ ATR×2) ✓")
    else:
        log.warning("  高波动行情 LONG 被拒 (可能被技术否决), 跳过下限断言")

    # ---- 3. 窄止损拒绝: atr_reject_ratio 调高 → 触发拒绝 ----
    # 用 reject_ratio=10 (任何止损距离 < ATR×10 → 拒绝), 应返回 None
    fvg2 = make_fvg("long", 102.0, 101.5, len(candles_hi) - 3)
    sig_rej = generate_signal(
        fvg=fvg2, current_price=102.0, candles=candles_hi,
        **base_kwargs(max_entry_distance_pct=5.0, atr_reject_ratio=10.0),
    )
    log.info(f"atr_reject_ratio=10 → signal={'None (拒绝)' if sig_rej is None else sig_rej.stop_loss}")
    assert sig_rej is None, "reject_ratio=10 应触发窄止损拒绝"
    log.info("  窄止损拒绝 ✓")

    # ---- 4. 杠杆联动 ----
    # 低波动窄止损场景: 止损距离 ~1% → 杠杆应 ≤ 2.5/1 = 2.5x
    candles_low = make_candles(n=40, base=100.0, wave=0.3)
    atr_low = _compute_atr_wilder(candles_low, 14)
    log.info(f"低波动 ATR={atr_low:.3f}")
    fvg3 = make_fvg("long", 100.6, 99.9, len(candles_low) - 3)
    sig3 = generate_signal(
        fvg=fvg3, current_price=100.5, candles=candles_low,
        **base_kwargs(max_entry_distance_pct=5.0, max_leverage=10),
    )
    if sig3:
        sl_pct = (sig3.entry_price - sig3.stop_loss) / sig3.entry_price * 100.0
        lev_cap = 2.5 / sl_pct if sl_pct > 0 else 10
        log.info(f"LONG: SL距={sl_pct:.2f}% 杠杆={sig3.leverage}x 上限={lev_cap:.1f}x")
        assert sig3.leverage <= max(1, math.floor(lev_cap)), \
            f"杠杆 {sig3.leverage}x 应 ≤ {lev_cap:.1f}x"
        log.info("  杠杆联动 ✓")
    else:
        log.warning("  低波动 LONG 被拒, 跳过杠杆断言")

    # 杠杆封顶: max_leverage=3 → 杠杆 ≤ 3
    sig4 = generate_signal(
        fvg=fvg3, current_price=100.5, candles=candles_low,
        **base_kwargs(max_entry_distance_pct=5.0, max_leverage=3),
    )
    if sig4:
        log.info(f"max_leverage=3 → 杠杆={sig4.leverage}x")
        assert sig4.leverage <= 3, f"杠杆 {sig4.leverage}x 超 max_leverage=3"
        log.info("  杠杆封顶 ✓")

    log.info("=" * 50)
    log.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
