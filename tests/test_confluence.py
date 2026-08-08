# -*- coding: utf-8 -*-
"""Confluence 汇流确认引擎单元测试。

覆盖: HTFBiasDetector / LiquidityPoolDetector / StructureBreaker /
      OrderFlowFilter / ConfluenceChecker + agent.py 端到端集成 + 性能预算。

运行: python -m unittest tests.test_confluence -v
"""

import math
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confluence import (  # noqa: E402
    HTFBiasDetector,
    LiquidityPoolDetector,
    StructureBreaker,
    OrderFlowFilter,
    ConfluenceChecker,
)
from fvg_detector import FVGDetector, FVGDetected  # noqa: E402

H = 3_600_000
TS0 = 1_700_000_000_000
N = 250


class C:
    """K 线桩（鸭子类型，与 strategy.Candle 同构）。"""

    def __init__(self, ts, o, h, l, c, v=100.0):
        self.timestamp, self.open, self.high, self.low = ts, o, h, l
        self.close, self.volume = c, v


def mk(ts, o, h, l, c, v=100.0):
    return C(ts, o, h, l, c, v)


def series(n: int = N, trend: float = 0.4, end_hour: int = 9,
           start: float = 100.0, body_dir: int = 0):
    """趋势 + 正弦锯齿序列（产生清晰 Swing）。

    Args:
        trend: 每根 K 线趋势增量 (+, - , 0)
        end_hour: 最后一根 K 线的 UTC 小时 (用于固定 time_window 判定)
        body_dir: 0=close 跟随 mid, >0 固定阳线实体, <0 固定阴线实体
    """
    out = []
    h0 = (end_hour - (n - 1)) % 24
    for i in range(n):
        mid = start + trend * i + 3.0 * math.sin(i * 0.5)
        if body_dir > 0:
            c = mid + 0.3
        elif body_dir < 0:
            c = mid - 0.3
        else:
            c = mid + (0.1 if trend >= 0 else -0.1)
        out.append(C((h0 + i) * H, mid, mid + 0.4, mid - 0.4, c))
    return out


def flat_range_series() -> list:
    """对称三角波 (Swing 高低点水平 99.5↔100.5) + 末端 5 根平价 K 线。"""
    out = []
    for i in range(120):
        ph = i % 12
        if ph < 6:
            mid = 100 - 0.5 + (1.0 * ph / 5)
        else:
            mid = 100 + 0.5 - (1.0 * (ph - 6) / 5)
        out.append(C(TS0 + i * H, mid, mid + 0.3, mid - 0.3, mid + 0.1))
    t0 = out[-1].timestamp
    for k in range(5):
        out.append(C(t0 + (k + 1) * H, 100.0, 100.0, 100.0, 100.0))
    return out


def sweep_series() -> list:
    """BSL 猎杀序列: swing low@24 (98.0) → 猎杀@30 (低 97.3 收回 98.6)。"""
    out = []
    for i in range(20):
        mid = 101.0 + 0.2 * math.sin(i * 0.7)
        out.append(C(TS0 + i * H, mid, mid + 0.3, mid - 0.3, mid + 0.1))
    for i, m in enumerate([100.0, 99.7, 99.4, 99.1]):
        out.append(C(TS0 + (20 + i) * H, m, m + 0.3, m - 0.3, m + 0.05))
    out.append(C(TS0 + 24 * H, 98.5, 98.8, 98.0, 98.3))   # swing low 98.0
    out.append(C(TS0 + 25 * H, 98.3, 98.7, 98.1, 98.6))
    out.append(C(TS0 + 26 * H, 98.7, 99.0, 98.5, 98.9))
    out.append(C(TS0 + 27 * H, 99.0, 99.3, 98.9, 99.2))   # FVG c0 (high 99.3)
    out.append(C(TS0 + 28 * H, 99.4, 99.9, 99.3, 99.7))   # FVG impulse c1
    out.append(C(TS0 + 29 * H, 99.7, 100.1, 99.6, 100.0))  # FVG c2 (low 99.6)
    out.append(C(TS0 + 30 * H, 99.5, 99.8, 97.3, 98.6))   # BSL 猎杀: 插针 97.3 收回 98.6
    out.append(C(TS0 + 31 * H, 98.8, 99.3, 98.6, 99.1))
    for i in range(20):
        m = 99.3 + 0.3 * i
        out.append(C(TS0 + (32 + i) * H, m, m + 0.3, m - 0.3, m + 0.1))
    return out


def detect_bullish(candles: list) -> FVGDetected:
    """检测序列中最后一个看涨 FVG。"""
    det = FVGDetector({"min_fvg_width_pct": {"1H": 1.5}})
    bulls = [f for f in det.detect({"1H": candles}) if f.direction == "bullish"]
    if not bulls:
        raise AssertionError("应检测到看涨 FVG")
    return bulls[-1]


DETECTOR_CFG = {"min_fvg_width_pct": {"1H": 1.5, "4H": 3.0}}


# ---------------------------------------------------------------------------
# 1.1 HTFBiasDetector
# ---------------------------------------------------------------------------

class TestHTFBiasDetector(unittest.TestCase):
    """大周期方向检测。"""

    def test_bullish_trend(self):
        up = series(N, trend=0.4)
        r = HTFBiasDetector({}).detect(up)
        self.assertEqual(r["bias"], "bullish")
        self.assertGreaterEqual(r["confidence"], 0.5)
        self.assertEqual(r["breakdown"]["swing_highs_lows"], "HH_HL")
        self.assertGreaterEqual(r["vwap_position"], 0)

    def test_bearish_trend(self):
        dn = series(200, trend=-0.3)
        r = HTFBiasDetector({}).detect(dn)
        self.assertEqual(r["bias"], "bearish")
        self.assertGreaterEqual(r["confidence"], 0.5)

    def test_flat_range_neutral(self):
        r = HTFBiasDetector({}).detect(flat_range_series())
        self.assertEqual(r["bias"], "neutral")
        self.assertEqual(r["confidence"], 0.0)

    def test_insufficient_data_neutral(self):
        r = HTFBiasDetector({}).detect([])
        self.assertEqual(r["bias"], "neutral")
        self.assertIn("reasons", r)
        self.assertIn("breakdown", r)
        # 空数据与短数据均不抛异常
        HTFBiasDetector({}).detect([mk(1, 100, 101, 99, 100)])


# ---------------------------------------------------------------------------
# 1.2 LiquidityPoolDetector
# ---------------------------------------------------------------------------

class TestLiquidityPoolDetector(unittest.TestCase):
    """流动性池识别。"""

    def test_swing_points(self):
        sw = LiquidityPoolDetector({}).find_swing_points(series(200, trend=0.2))
        self.assertTrue(sw["swing_highs"])
        self.assertTrue(sw["swing_lows"])
        # 上升趋势中最高高点应高于最高低点
        self.assertGreater(max(sw["swing_highs"]), max(sw["swing_lows"]))

    def test_identify_pools(self):
        lp = LiquidityPoolDetector({})
        pools = lp.identify_pools(series(120, trend=0.2))
        self.assertIn("nearest_bsl", pools)
        self.assertIn("nearest_ssl", pools)
        for p in pools["pools"]:
            self.assertIn(p["type"], ("bsl", "ssl"))
            self.assertIsInstance(p["is_swept"], bool)
        self.assertEqual(lp.identify_pools([])["pools"], [])

    def test_liquidity_sweep_detection(self):
        lp = LiquidityPoolDetector({})
        sw = sweep_series()
        self.assertTrue(lp.is_liquidity_sweep(sw, 30))
        self.assertFalse(lp.is_liquidity_sweep(sw, 10))
        d = lp.detect_sweep(sw, 30)
        self.assertEqual(d["pool"], "bsl")
        self.assertAlmostEqual(d["swept_level"], 98.0, places=6)
        # 边界: 越界不崩溃
        self.assertFalse(lp.is_liquidity_sweep(sw, 0))
        self.assertFalse(lp.is_liquidity_sweep(sw, len(sw) - 1))


# ---------------------------------------------------------------------------
# 1.3 StructureBreaker
# ---------------------------------------------------------------------------

class TestStructureBreaker(unittest.TestCase):
    """市场结构破坏 (ChoCH)。"""

    def test_uptrend_not_broken(self):
        r = StructureBreaker({}).detect(series(N, trend=0.4))
        self.assertEqual(r["regime"], "bullish")
        self.assertFalse(r["is_broken"])

    def test_bearish_break(self):
        brk = series(120, trend=0.4)
        levels = StructureBreaker({}).detect(brk)["structure_levels"]
        self.assertIsNotNone(levels["last_higher_low"])
        hl = levels["last_higher_low"]
        t = brk[-1].timestamp
        brk.append(C(t + H, hl + 1.0, hl + 1.2, hl - 2.0, hl - 0.8))   # 跌破 higher low
        brk.append(C(t + 2 * H, hl - 1.0, hl + 0.2, hl - 2.5, hl - 0.5))  # 确认
        r = StructureBreaker({}).detect(brk)
        self.assertTrue(r["is_broken"])
        self.assertEqual(r["last_break_type"], "bearish_break")
        self.assertEqual(r["regime"], "bearish")
        self.assertGreaterEqual(r["break_bar_index"], 0)


# ---------------------------------------------------------------------------
# 1.4 OrderFlowFilter
# ---------------------------------------------------------------------------

class TestOrderFlowFilter(unittest.TestCase):
    """订单流过滤 (模拟)。"""

    def test_absorption(self):
        candles = series(40, trend=0.0, body_dir=1)
        candles.append(C(TS0 + 40 * H, 100.0, 103.5, 99.5, 100.2, v=800.0))
        r = OrderFlowFilter({}).detect_absorption(candles, 40)
        self.assertTrue(r["is_absorption"])
        self.assertEqual(r["side"], "sell")
        self.assertGreaterEqual(r["volume_ratio"], 1.5)

    def test_delta_sign(self):
        of = OrderFlowFilter({})
        up_d = of.detect_delta_divergence(series(60, trend=0.5, body_dir=1))
        dn_d = of.detect_delta_divergence(series(60, trend=-0.5, body_dir=-1))
        self.assertGreater(up_d, 0.0)
        self.assertLess(dn_d, 0.0)
        self.assertLessEqual(abs(up_d), 1.0)
        self.assertEqual(of.detect_delta_divergence([]), 0.0)


# ---------------------------------------------------------------------------
# 1.5 ConfluenceChecker + agent 集成
# ---------------------------------------------------------------------------

class TestConfluence(unittest.TestCase):
    """汇流确认总控 + 端到端集成。"""

    def setUp(self):
        self.cc = ConfluenceChecker({})
        self.up = series(N, trend=0.4)
        self.fvg_up = detect_bullish(self.up)

    # ---- 汇流得分计算 ----
    def test_confluence_score_calculation(self):
        cr = self.cc.check(self.fvg_up, self.up, self.up,
                           {"current_price": float(self.up[-1].close)})
        self.assertGreaterEqual(cr["confluence_score"], 0.5)
        self.assertIn(cr["recommendation"], ("buy", "strong_buy"))
        self.assertIn("bias_alignment", cr["conditions_met"])
        self.assertEqual(cr["entry_quality"], "good")

        # 逆向: 看跌 FVG + 上升趋势 → 低分否决
        bearish_fvg = FVGDetected(
            inst_id="", timeframe="1H", start_idx=self.fvg_up.start_idx,
            end_idx=self.fvg_up.end_idx, gap_high=self.fvg_up.gap_high,
            gap_low=self.fvg_up.gap_low, width_pct=self.fvg_up.width_pct,
            volume_at_formation=100, formation_price=self.fvg_up.formation_price,
            direction="bearish", is_abnormal=False, quality_score=0.5,
            formation_ts=self.fvg_up.formation_ts)
        cr2 = self.cc.check(bearish_fvg, self.up, self.up,
                            {"current_price": float(self.up[-1].close)})
        self.assertLess(cr2["confluence_score"], 0.5)
        self.assertIn(cr2["recommendation"], ("neutral", "reject"))
        self.assertFalse(cr2["details"]["bias_alignment"]["met"])

    # ---- 汇流特征生成 ----
    def test_confluence_features(self):
        cr = self.cc.check(self.fvg_up, self.up, self.up, {})
        feats = self.cc.get_confluence_features(cr)
        self.assertEqual(set(feats.keys()), {
            "confluence_score", "bias_aligned", "liquidity_swept",
            "structure_broken", "orderflow_positive", "htf_nested",
            "in_premium_zone", "in_good_time", "num_conditions_met",
            "entry_quality_score"})
        self.assertEqual(feats["bias_aligned"], 1)
        self.assertEqual(feats["confluence_score"], cr["confluence_score"])
        self.assertIn(feats["num_conditions_met"], range(0, 8))

    # ---- 与 agent.py 的端到端集成（复刻主循环过滤块）----
    def test_end_to_end_with_agent(self):
        from strategy import Candle, FVG, Signal  # noqa: PLC0415
        from fvg_detector import from_legacy_fvg  # noqa: PLC0415

        d = self.fvg_up
        gap = d.gap_high - d.gap_low
        imp = Candle(timestamp=self.up[d.end_idx].timestamp,
                     open=self.up[d.end_idx].open,
                     high=self.up[d.end_idx].high,
                     low=self.up[d.end_idx].low,
                     close=self.up[d.end_idx].close, volume=100)
        legacy_fvg = FVG(direction="long", top=d.gap_high, bottom=d.gap_low,
                         width_pct=d.width_pct, candle_ts=d.formation_ts,
                         timeframe="1H", impulse_candle=imp,
                         fvg_index=d.end_idx, is_abnormal=d.is_abnormal,
                         sigma=d.sigma, volume_ratio=d.volume_ratio)
        sig = Signal(inst_id="BTC-USDT-SWAP", fvg=legacy_fvg,
                     entry_price=d.gap_low, stop_loss=d.gap_low - 1.5 * gap,
                     take_profit=d.gap_high, leverage=3,
                     position_side="long", score=0.8)
        # signal_analysis_map 结构 (agent 主循环)
        sig_map = {"BTC-USDT-SWAP": {"candles_1h": self.up,
                                     "candles_4h": self.up,
                                     "funding_rate": 0.0001}}

        # ---- 复刻 agent.py 汇流过滤块 ----
        threshold = 0.5
        cr = self.cc.check(
            from_legacy_fvg(sig.fvg),
            sig_map[sig.inst_id]["candles_1h"],
            sig_map[sig.inst_id]["candles_4h"],
            {"current_price": sig.entry_price, "funding_rate": 0.0001,
             "spread_pct": getattr(sig, "spread_pct", 0.0)},
        )
        sig.confluence_score = float(cr["confluence_score"])
        sig.confluence_details = cr
        sig.entry_quality = str(cr["entry_quality"])
        kept = sig if sig.confluence_score >= threshold else None

        self.assertIsNotNone(kept, "顺向信号应通过汇流过滤")
        self.assertGreaterEqual(sig.confluence_score, threshold)
        self.assertEqual(sig.confluence_details["confluence_score"],
                         sig.confluence_score)
        self.assertIn(sig.entry_quality, ("excellent", "good", "poor"))

        # ---- detect_with_confluence 端到端 ----
        det = FVGDetector(DETECTOR_CFG)
        pairs = det.detect_with_confluence(
            {"1H": self.up, "4H": self.up},
            {"current_price": float(self.up[-1].close)})
        self.assertGreaterEqual(len(pairs), 1)
        for f, r in pairs:
            self.assertIsInstance(f, FVGDetected)
            self.assertIn("confluence_score", r)
            self.assertIn("details", r)

        # ---- compute_features 含汇流特征 (25 维) ----
        feats = det.compute_features(pairs[0][0], self.up)
        self.assertEqual(len(feats), 25)
        self.assertIn("entry_quality_score", feats)

    # ---- 性能预算: 单次汇流检查 < 100ms ----
    def test_performance_within_budget(self):
        candles = series(200, trend=0.4)
        fvg = detect_bullish(candles)
        cc = ConfluenceChecker({})
        cc.check(fvg, candles, candles, {"current_price": 100.0})  # 预热
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            cc.check(fvg, candles, candles, {"current_price": 100.0})
        avg_ms = (time.perf_counter() - t0) / n * 1000.0
        self.assertLess(avg_ms, 100.0,
                        f"单次汇流检查耗时 {avg_ms:.1f}ms 超过 100ms 预算")

    # ---- 空数据中性兜底 ----
    def test_empty_data_neutral(self):
        fvg = FVGDetected(inst_id="", timeframe="1H", start_idx=0, end_idx=1,
                          gap_high=101.0, gap_low=100.0, width_pct=1.0,
                          volume_at_formation=100, formation_price=100.5,
                          direction="bullish", is_abnormal=False,
                          quality_score=0.5, formation_ts=0)
        cr = self.cc.check(fvg, [], [], {})
        self.assertEqual(cr["confluence_score"], 0.0)
        self.assertEqual(cr["recommendation"], "neutral")


if __name__ == "__main__":
    unittest.main(verbosity=2)
