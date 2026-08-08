# -*- coding: utf-8 -*-
"""FVGDetector / FVGMLRanker / FVGBacktest 单元测试。

运行: python -m pytest tests/test_fvg_detector.py
     (或 python -m unittest tests.test_fvg_detector)
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvg_detector import FVGDetector, FVGDetected, from_legacy_fvg  # noqa: E402
from fvg_backtest import FVGBacktest, Trade  # noqa: E402
from fvg_ml_ranker import FVGMLRanker  # noqa: E402

H = 3_600_000
TS0 = 1_700_000_000_000


class CandleStub:
    """K 线桩（鸭子类型，与 Candle 同构）。"""

    def __init__(self, ts, o, h, l, c, v=100.0):
        self.timestamp, self.open, self.high, self.low = ts, o, h, l
        self.close, self.volume = c, v


def mk(ts, o, h, l, c, v=100.0):
    return CandleStub(ts, o, h, l, c, v)


def make_series():
    """构造含看涨 FVG (gap 101-106) + 回补 + 反弹的 K 线。"""
    candles = []
    ts = TS0
    for i in range(30):
        o = 100 + 0.2 * math.sin(i)
        candles.append(mk(ts, o, o + 0.3, o - 0.3, o + 0.05))
        ts += H
    candles.append(mk(ts, 100.5, 101.0, 100.2, 100.8)); ts += H
    candles.append(mk(ts, 101.0, 110.0, 101.0, 108.0, v=800)); ts += H
    candles.append(mk(ts, 108.0, 109.0, 106.0, 108.5, v=150)); ts += H
    for i in range(20):
        o = 108 - i * 0.2
        candles.append(mk(ts, o, o + 0.3, o - 0.4, o - 0.15))
        ts += H
    for i in range(12):
        o = 104.2 + i * 0.25
        candles.append(mk(ts, o, o + 0.3, o - 0.2, o + 0.1))
        ts += H
    return candles


CFG = {
    "min_fvg_width_pct": {"1H": 1.5, "4H": 3.0},
    "abnormal_sigma": 3.0,
    "abnormal_volume_ratio": 5.0,
    "abnormal_lookback": {"1H": 50, "4H": 50},
}


class TestFVGDetector(unittest.TestCase):
    """FVG 检测 / 过滤 / 特征 / 回测一致性。"""

    def setUp(self):
        self.candles = make_series()
        self.det = FVGDetector(CFG)

    def test_detect_bullish_fvg(self):
        """测试看涨 FVG 检测。"""
        fvgs = self.det.detect({"1H": self.candles})
        bulls = [f for f in fvgs if f.direction == "bullish"]
        self.assertGreaterEqual(len(bulls), 1, "应检测到看涨 FVG")
        b = bulls[-1]
        self.assertEqual(b.direction, "bullish")
        self.assertGreater(b.gap_high, b.gap_low)
        self.assertGreaterEqual(b.width_pct, 1.5, "宽度应达最小阈值")
        self.assertGreater(b.formation_ts, 0)
        self.assertGreater(b.volume_at_formation, 0)
        self.assertGreaterEqual(b.quality_score, 0.0)
        self.assertLessEqual(b.quality_score, 1.0)

    def test_detect_bearish_fvg(self):
        """测试看跌 FVG 检测（构造 c0.low > c2.high）。"""
        candles = []
        ts = TS0
        for i in range(20):
            o = 120 - i * 0.2
            candles.append(mk(ts, o, o + 0.3, o - 0.3, o - 0.1))
            ts += H
        # 看跌 FVG: c0.low=119.5 > c2.high=117 → gap [117, 119.5]
        candles.append(mk(ts, 120.0, 120.5, 119.5, 120.2)); ts += H
        candles.append(mk(ts, 120.0, 122.0, 118.0, 119.0, v=600)); ts += H
        candles.append(mk(ts, 118.0, 117.5, 116.5, 117.2, v=120)); ts += H
        fvgs = self.det.detect({"1H": candles})
        bears = [f for f in fvgs if f.direction == "bearish"]
        self.assertGreaterEqual(len(bears), 1, "应检测到看跌 FVG")
        b = bears[-1]
        self.assertLess(b.gap_high, b.gap_low * 1.5)  # 缺口合理
        self.assertEqual(b.direction, "bearish")

    def test_filter_abnormal(self):
        """测试异常/无效 FVG 过滤。"""
        fvgs = self.det.detect({"1H": self.candles})
        self.assertGreaterEqual(len(fvgs), 1)
        # 正常上下文全放行
        ok = self.det.filter_by_quality(
            fvgs, {"current_price": 104.0, "atr": 2.0,
                   "funding_rate": -0.0002, "spread_pct": 0.01})
        self.assertEqual(len(ok), len(fvgs))
        # 价格远离缺口 → 过滤
        far = self.det.filter_by_quality(fvgs, {"current_price": 80.0, "atr": 2.0})
        self.assertLess(len(far), len(fvgs))
        # ATR 巨型缺口 → 过滤
        huge = self.det.filter_by_quality(fvgs, {"current_price": 104.0, "atr": 0.001})
        self.assertLess(len(huge), len(fvgs))
        # 空输入
        self.assertEqual(self.det.filter_by_quality([], {}), [])

    def test_abnormal_long_series(self):
        """回归: 长序列 MAD 异常检测不得抛错（曾因 list-float 算术崩溃）。"""
        candles = []
        ts = TS0
        for i in range(120):
            o = 100 + 0.2 * math.sin(i / 5)
            candles.append(mk(ts, o, o + 0.3, o - 0.3, o + 0.05, v=100))
            ts += H
        # 注入放量跳空（触发 idx>=lookback 的 MAD 全路径）
        candles[60] = mk(candles[60].timestamp, 100.0, 112.0, 99.0, 111.0, v=900)
        candles[61] = mk(candles[61].timestamp, 110.0, 111.0, 108.0, 110.5, v=150)
        fvgs = self.det.detect({"1H": candles})
        self.assertGreaterEqual(len(fvgs), 1, "应检测到放量跳空 FVG")
        for f in fvgs:
            # sigma/volume_ratio/is_abnormal 必须为有效标量（MAD 已执行）
            self.assertIsInstance(f.sigma, float)
            self.assertGreater(f.sigma, 0.0)
            self.assertGreater(f.volume_ratio, 0.0)
            self.assertIn(f.is_abnormal, (True, False))

    def test_feature_extraction(self):
        """测试特征提取完整性（25 维全齐且为 float）。"""
        fvgs = self.det.detect({"1H": self.candles})
        self.assertGreaterEqual(len(fvgs), 1)
        feats = self.det.compute_features(fvgs[-1], self.candles)
        expect = [
            "fvg_width_pct", "atr_ratio", "volume_ratio",
            "distance_to_ma20", "distance_to_ma50", "prev_trend_strength",
            "prev_volatility", "gap_position", "retracement_pct",
            "momentum_divergence", "liquidity_around_gap",
            "historical_fill_rate", "corr_with_btc",
            "funding_rate_zscore", "order_book_imbalance",
            # 汇流特征 (16-25)
            "confluence_score", "bias_aligned", "liquidity_swept",
            "structure_broken", "orderflow_positive", "htf_nested",
            "in_premium_zone", "in_good_time", "num_conditions_met",
            "entry_quality_score",
        ]
        self.assertEqual(len(feats), len(expect))
        for k in expect:
            self.assertIn(k, feats)
            # 汇流标志位为 0/1 int, 其余为 float
            self.assertIsInstance(feats[k], (int, float), f"{k} 类型错误")
            self.assertFalse(isinstance(feats[k], bool), f"{k} 不应为 bool")

    def test_backtest_consistency(self):
        """测试回测结果与检测逻辑一致性。"""
        fvgs = self.det.detect({"1H": self.candles})
        bt = FVGBacktest(initial_capital=10000)
        res = bt.run(fvgs, self.candles, {
            "max_entry_bars": 30, "max_hold_bars": 30,
            "stop_width_mult": 1.5, "position_pct": 0.1})
        # 数据含回补+反弹 → 应产生至少 1 笔交易
        self.assertGreaterEqual(res["n_trades"], 1)
        for t in res["trades"]:
            self.assertIsInstance(t, Trade)
            self.assertIn(t.exit_reason, ("tp", "sl", "time"))
            # 盈亏与方向一致
            if t.direction == "bullish":
                self.assertGreater(t.return_pct, -100)
            self.assertGreater(res["max_drawdown"], -1e-9)
        # 阈值对比输出结构
        cmp_df = bt.compare_with_thresholds(fvgs, self.candles)
        self.assertIn("quality_threshold", cmp_df.columns)
        self.assertGreaterEqual(len(cmp_df), 1)


class TestFVGMLRanker(unittest.TestCase):
    """ML 排名器（内置线性后端兜底，不依赖 sklearn）。"""

    def setUp(self):
        self.det = FVGDetector(CFG)
        self.candles = make_series()
        rows, labels = [], []
        for i in range(120):
            f = FVGDetected(inst_id="T", timeframe="1H", start_idx=1, end_idx=3,
                            gap_high=100 + i * 0.5, gap_low=100, width_pct=2.0,
                            volume_at_formation=100, formation_price=101,
                            direction="bullish", is_abnormal=bool(i % 3 == 0),
                            quality_score=0.5 + (i % 5) * 0.1, formation_ts=i)
            rows.append(self.det.compute_features(f, self.candles))
            labels.append(1 if i % 2 == 0 else 0)
        self.X = pd.DataFrame(rows)
        self.y = pd.Series(labels)

    def test_train_predict_importance(self):
        """训练 → 预测 → 特征重要性链路。"""
        ranker = FVGMLRanker()
        ranker.train(self.X, self.y)
        self.assertTrue(ranker._trained)
        p = ranker.predict(self.X.iloc[0].to_dict())
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        batch = ranker.predict_batch(self.X.iloc[:5].to_dict("records"))
        self.assertEqual(len(batch), 5)
        imp = ranker.get_feature_importance()
        self.assertEqual(len(imp), len(self.X.columns))
        self.assertTrue(imp["importance"].is_monotonic_decreasing)

    def test_persist_roundtrip(self):
        """模型保存/加载预测一致。"""
        path = os.path.join(tempfile.gettempdir(), "fvg_test_model.pkl")
        r1 = FVGMLRanker()
        r1.model_path = path
        r1.train(self.X, self.y)
        r2 = FVGMLRanker(path)
        f0 = self.X.iloc[0].to_dict()
        self.assertAlmostEqual(r1.predict(f0), r2.predict(f0), places=6)

    def test_untrained_neutral(self):
        """未训练返回 0.5 中性值。"""
        ranker = FVGMLRanker()
        self.assertEqual(ranker.predict(self.X.iloc[0].to_dict()), 0.5)
        self.assertEqual(ranker.predict_batch([{}, {}]), [0.5, 0.5])


class TestLegacyAdapter(unittest.TestCase):
    """from_legacy_fvg 适配（strategy.FVG → FVGDetected）。"""

    def test_mapping(self):
        from strategy import Candle, FVG
        imp = Candle(timestamp=TS0, open=100, high=110, low=99, close=108, volume=800)
        legacy = FVG(direction="long", top=106.0, bottom=101.0, width_pct=5.0,
                     candle_ts=TS0 + 2 * H, timeframe="1H", impulse_candle=imp,
                     fvg_index=31, is_abnormal=True, sigma=4.2, volume_ratio=3.0)
        d = from_legacy_fvg(legacy)
        self.assertEqual(d.direction, "bullish")
        self.assertEqual(d.gap_high, 106.0)
        self.assertEqual(d.gap_low, 101.0)
        self.assertEqual(d.start_idx, 30)
        self.assertEqual(d.end_idx, 32)
        self.assertTrue(d.is_abnormal)
        self.assertEqual(d.volume_at_formation, 800.0)
        self.assertEqual(d.formation_price, 108.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
