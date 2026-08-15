# -*- coding: utf-8 -*-
"""
本地测试: 模拟下跌行情下"新挂单逻辑"(挂单距离限制 + RR≥2.5 止盈) 与纸面成交链路。

验证点:
  1. generate_signal 在下跌行情中产出的 SHORT 信号:
     - entry 与现价偏差 ≤ max_entry_distance_pct(1.5%)
     - 止盈方向正确 (short: tp < entry < sl)
     - 盈亏比 RR ≥ 2.5
  2. 深挂(流动性猎手)偏离过大时回退 FVG 回补位 ([EntryLimit] 日志)
  3. 纸面引擎成交链路: 价格回补到挂单价 → 成交 → TP 触发 → 平仓入账
"""
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy import Candle, FVG, generate_signal
from paper_trading import PaperTradingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("test_entry_logic")

MAX_ENTRY_DEV = 1.5    # 与 config strategy.max_entry_distance_pct 一致
MIN_RR = 2.5           # 与 config strategy.min_risk_reward 一致
LIQ_EXT = 0.8          # 纸面专用外扩%(与 paper.liquidity_extension_pct 一致)
T0 = 1700000000000


def build_downtrend_candles(base=100.0, n=32):
    """构造下跌行情 1H K线(带锯齿回调避免布林否决), 最新一根在末尾。

    结构: 前 26 根锯齿下跌(总体 -0.4%/bar), 后 6 根窄幅整理(带反弹)。
    最后 3 根被替换为显式构造的看跌 FVG(大阴线冲动)。
    """
    candles = []
    px = base
    for i in range(n - 3):
        if i < n - 9:
            drift, noise_amp = -0.004, 0.002
        else:
            drift, noise_amp = 0.001, 0.0015
        o = px
        c = px * (1 + drift + noise_amp * math.sin(i * 1.7))
        hi = max(o, c) * 1.003
        lo = min(o, c) * 0.997
        candles.append(Candle(
            timestamp=T0 + i * 3600000, open=o, high=hi, low=lo, close=c,
            volume=80000 + (i % 7) * 5000))
        px = c
    # 显式构造看跌 FVG 三根: C1(前) / C2(大阴线冲动) / C3(反弹)
    c3 = Candle(T0 + (n - 1) * 3600000, open=px, high=px * 1.006,
                low=px * 0.995, close=px * 1.001, volume=90000)
    c2 = Candle(T0 + (n - 2) * 3600000, open=px * 1.012, high=px * 1.016,
                low=px * 0.998, close=px * 1.000, volume=200000)
    c1 = Candle(T0 + (n - 3) * 3600000, open=px * 1.020, high=px * 1.024,
                low=px * 1.010, close=px * 1.016, volume=70000)
    candles += [c1, c2, c3]
    return candles


def make_bearish_fvg(candles):
    """由最后 3 根构造看跌 FVG: top=C1.low, bottom=C3.high。"""
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    top, bottom = c1.low, c3.high
    assert top > bottom, f"FVG 结构无效 top={top:.4f} <= bottom={bottom:.4f}"
    width_pct = (top - bottom) / bottom * 100.0
    return FVG(
        direction="short", top=top, bottom=bottom, width_pct=width_pct,
        candle_ts=c2.timestamp, timeframe="1H", impulse_candle=c2,
        fvg_index=len(candles) - 2, is_abnormal=True, sigma=3.2,
        volume_ratio=6.0)


def make_bearish_fvg_wide(candles, current_price):
    """构造宽幅看跌 FVG(top 远高于现价, bottom 略高于现价):
    回补位距现价 > 1.5% → 应被最终距离检查拒绝 (返回 None)。"""
    c2 = candles[-2]
    top = current_price * 1.25          # 大缺口上沿
    bottom = current_price * 1.001      # 下沿略高于现价(deep 路径成立)
    width_pct = (top - bottom) / bottom * 100.0
    return FVG(
        direction="short", top=top, bottom=bottom, width_pct=width_pct,
        candle_ts=c2.timestamp, timeframe="1H", impulse_candle=c2,
        fvg_index=len(candles) - 2, is_abnormal=True, sigma=3.2,
        volume_ratio=6.0)


def build_uptrend_candles(base=100.0, n=32):
    """构造快速拉升行情 1H K线(带小幅回调), 最新一根在末尾。

    与下跌行情镜像: 锯齿上涨(总体 +0.4%/bar), 最后 3 根为看涨 FVG。
    """
    candles = []
    px = base
    for i in range(n - 3):
        if i < n - 9:
            drift, noise_amp = 0.004, 0.002
        else:
            drift, noise_amp = -0.001, 0.0015
        o = px
        c = px * (1 + drift + noise_amp * math.sin(i * 1.7))
        hi = max(o, c) * 1.003
        lo = min(o, c) * 0.997
        candles.append(Candle(
            timestamp=T0 + i * 3600000, open=o, high=hi, low=lo, close=c,
            volume=80000 + (i % 7) * 5000))
        px = c
    # 看涨 FVG 三根: C1(前) / C2(大阳线冲动) / C3(小幅回调)
    c3 = Candle(T0 + (n - 1) * 3600000, open=px, high=px * 1.004,
                low=px * 0.994, close=px * 0.999, volume=90000)
    c2 = Candle(T0 + (n - 2) * 3600000, open=px * 0.988, high=px * 1.002,
                low=px * 0.984, close=px * 1.000, volume=200000)
    c1 = Candle(T0 + (n - 3) * 3600000, open=px * 0.980, high=px * 0.990,
                low=px * 0.976, close=px * 0.984, volume=70000)
    candles += [c1, c2, c3]
    return candles


def make_bullish_fvg(candles):
    """由最后 3 根构造看涨 FVG: bottom=C1.high, top=C3.low。"""
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    bottom, top = c1.high, c3.low
    assert top > bottom, f"看涨FVG 结构无效 top={top:.4f} <= bottom={bottom:.4f}"
    width_pct = (top - bottom) / bottom * 100.0
    return FVG(
        direction="long", top=top, bottom=bottom, width_pct=width_pct,
        candle_ts=c2.timestamp, timeframe="1H", impulse_candle=c2,
        fvg_index=len(candles) - 2, is_abnormal=True, sigma=3.2,
        volume_ratio=6.0)


def _signal_kwargs(fvg, current_price, candles):
    """统一的 generate_signal 参数。

    entry_distance_atr_mult=0: 显式关闭 ATR 挂钩深度阈值 (2026-08-10 特性)，
    保持本测试的原始契约 = 固定 max_entry_distance_pct(1.5%) 距离检查。
    ATR 挂钩路径由 test_production_fixes.py::TestEntryDistanceAtrHook 覆盖。
    """
    return dict(
        inst_id="TEST-USDT-SWAP", fvg=fvg, current_price=current_price,
        entry_depth_pct=0.15, fvg_target_pct=0.50, stop_buffer_pct=0.15,
        max_leverage=10, funding_rate=None, max_funding_rate_abs=0.01,
        funding_confluence_min_abs=0.0003, funding_confluence_max_abs=0.001,
        liquidity_extension_pct=LIQ_EXT, liquidity_extension_min_pct=0.5,
        max_entry_distance_pct=MAX_ENTRY_DEV, min_risk_reward=MIN_RR,
        entry_distance_atr_mult=0.0,
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
        regime="NEUTRAL", candles=candles,
    )


def run_signal_case(name, fvg, candles, current_price):
    """跑 generate_signal，断言 距离/RR/方向(支持 long/short)，返回 Signal。"""
    log.info("=" * 70)
    log.info(f"用例: {name}")
    sig = generate_signal(**_signal_kwargs(fvg, current_price, candles))
    if sig is None:
        log.warning("!!! generate_signal 返回 None（被过滤器否决）— 需检查上方过滤日志")
        return None

    dev = abs(sig.entry_price - current_price) / current_price * 100.0
    if sig.position_side == "short":
        rr = (sig.entry_price - sig.take_profit) / (sig.stop_loss - sig.entry_price)
        dir_ok = sig.take_profit < sig.entry_price < sig.stop_loss
    else:
        rr = (sig.take_profit - sig.entry_price) / (sig.entry_price - sig.stop_loss)
        dir_ok = sig.take_profit > sig.entry_price > sig.stop_loss
    ok_dist = dev <= MAX_ENTRY_DEV + 1e-9
    ok_rr = rr >= MIN_RR - 1e-9

    log.info(f"信号: {sig.position_side} entry={sig.entry_price:.6g} "
             f"cur={current_price:.6g} dev={dev:.2f}%")
    log.info(f"     tp={sig.take_profit:.6g} sl={sig.stop_loss:.6g} "
             f"RR={rr:.2f} score={sig.score:.3f}")
    log.info(f"断言: dev<=1.5% [{ok_dist}] | RR>=2.5 [{ok_rr}] | "
             f"TP/SL方向 [{dir_ok}]")
    assert ok_dist, f"{name}: entry 距现价 {dev:.2f}% 超限"
    assert ok_rr, f"{name}: RR={rr:.2f} < {MIN_RR}"
    assert dir_ok, f"{name}: TP/SL 方向错误"
    log.info(f"✅ {name} 全部断言通过")
    return sig


def paper_fill_check(sig, candles):
    """纸面引擎成交链路(方向通用): 挂单 → 价格回补成交 → TP 触发平仓。

    注: 普通辅助函数（main() 以实参调用），非 pytest 测试 —
    原名 test_paper_fill 会被 pytest 收集并把 sig/candles 当 fixture
    解析，导致 'fixture not found' 收集错误（2026-08-15 修复）。
    """
    log.info("=" * 70)
    log.info(f"纸面成交链路模拟 ({sig.position_side})")
    cfg = {
        "paper": {
            "balance": 1000.0, "maker_fee": 0.0002, "taker_fee": 0.0005,
            "limit_timeout_min": 30, "fill_assist_seconds": 120,
            "state_file": "test_paper_state.json",   # 独立状态文件，避免覆盖实跑纸面状态
        },
        "risk": {"max_hold_hours": 48,
                 "dynamic_roi": {"240": 0.015, "120": 0.025, "60": 0.035, "0": 0.05}},
    }
    inst = sig.inst_id
    side = sig.position_side
    mark = [0.0]

    def market_data(_inst):
        return {"candles": candles, "mark": mark[0]}

    engine = PaperTradingEngine(cfg)
    engine.set_market_data_provider(market_data)

    # 1. 开仓(挂限价单), 现价停在挂单外侧(空单下方/多单上方, 不成交)
    mark[0] = sig.entry_price * (0.995 if side == "short" else 1.005)
    engine.update()                     # 预热缓存
    ord_id = engine.open_position(sig, {"ctVal": 1.0, "minSz": 1, "lotSz": "1"},
                                  cfg["risk"], 1000.0)
    assert ord_id, "纸面开仓失败"
    pos = engine.to_positions_dict()
    assert inst not in pos, "未成交限价单不应出现在持仓"
    log.info(f"开仓挂单成功: {ord_id}，当前标记价 {mark[0]:.6g}（未成交）")

    # 2. 模拟价格回补到挂单价 → 成交
    mark[0] = sig.entry_price * (1.002 if side == "short" else 0.998)
    engine.update()
    pos = engine.to_positions_dict()
    assert inst in pos, "价格扫到挂单价后应成交"
    log.info(f"✅ 限价单成交: entry={pos[inst]['avg_px']:.6g} "
             f"mark={pos[inst]['mark_px']:.6g}")

    # 3. 模拟价格朝盈利方向走到 TP → 平仓
    mark[0] = sig.take_profit * (0.998 if side == "short" else 1.002)
    engine.update()
    pos = engine.to_positions_dict()
    assert inst not in pos, "TP 触发后持仓应已平仓"
    log.info("✅ TP 触发平仓完成")
    trades = engine._trades
    if trades:
        t = trades[-1]
        log.info(f"   交易记录: {t['inst_id']} {t['side']} "
                 f"pnl={t['pnl']:+.4f} reason={t['reason']}")
        assert t["reason"] == "take_profit", \
            f"平仓原因应为 take_profit, 实为 {t['reason']}"
        assert t["pnl"] > 0, f"TP 平仓应盈利, 实为 {t['pnl']}"
    else:
        log.warning("   未找到交易记录")
    log.info(f"✅ 纸面成交链路全部通过 ({side})")


def main():
    # ===== 用例组1: 下跌行情 SHORT =====
    candles = build_downtrend_candles()
    current = candles[-1].close
    log.info(f"下跌行情构造完成: {len(candles)} 根 1H K线, 现价={current:.4f}")

    fvg_short = make_bearish_fvg(candles)
    log.info(f"看跌 FVG: top={fvg_short.top:.4f} bottom={fvg_short.bottom:.4f} "
             f"width={fvg_short.width_pct:.2f}%")

    sig_short = run_signal_case("下跌行情 SHORT (深挂+距离限制)", fvg_short,
                                candles, current)

    # ===== 用例组2: 宽幅 FVG 回补位过远 → 拒绝 =====
    log.info("=" * 70)
    log.info("用例: 宽幅 FVG 回补位过远 → 拒绝")
    fvg_wide = make_bearish_fvg_wide(candles, current)
    kw = _signal_kwargs(fvg_wide, current, candles)
    kw["inst_id"] = "TEST-WIDE-USDT-SWAP"
    sig_wide = generate_signal(**kw)
    assert sig_wide is None, "宽幅 FVG 回补位距现价过远应被 [EntryLimit] 拒绝"
    log.info("✅ 宽幅 FVG 拒绝用例通过（信号被 [EntryLimit] 正确拒绝）")

    # ===== 用例组3: 快速拉升行情 LONG + 止盈 =====
    ucandles = build_uptrend_candles()
    ucur = ucandles[-1].close
    log.info("=" * 70)
    log.info(f"快速拉升行情构造完成: {len(ucandles)} 根 1H K线, 现价={ucur:.4f}")
    fvg_long = make_bullish_fvg(ucandles)
    log.info(f"看涨 FVG: top={fvg_long.top:.4f} bottom={fvg_long.bottom:.4f} "
             f"width={fvg_long.width_pct:.2f}%")
    sig_long = run_signal_case("拉升行情 LONG (回补位+RR)", fvg_long,
                               ucandles, ucur)

    # ===== 纸面成交链路: SHORT + LONG 各跑一遍 =====
    if sig_short is not None:
        paper_fill_check(sig_short, candles)
    if sig_long is not None:
        paper_fill_check(sig_long, ucandles)

    log.info("=" * 70)
    log.info("🎉 全部测试通过")


def test_entry_logic_main():
    """pytest 入口 (2026-08-15): 跑完整 main() 流程。

    覆盖: SHORT 深挂信号 → 宽幅 FVG 拒绝 → LONG 回补信号 →
    纸面引擎 挂单/回补成交/TP 平仓 双向链路。独立运行方式不变:
    python test_entry_logic.py
    """
    main()


if __name__ == "__main__":
    main()
