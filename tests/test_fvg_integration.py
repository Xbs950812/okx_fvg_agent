# -*- coding: utf-8 -*-
"""FVG 集成测试 — from_legacy_fvg 兼容性 / BTC 注入 / 训练脚本 dry-run。

运行: python -m unittest tests.test_fvg_integration -v
"""

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import Candle, FVG, Signal  # noqa: E402
from fvg_detector import FVGDetector, FVGDetected, from_legacy_fvg  # noqa: E402

H = 3_600_000
TS0 = 1_700_000_000_000


class CandleStub:
    def __init__(self, ts, o, h, l, c, v=100.0):
        self.timestamp, self.open, self.high, self.low = ts, o, h, l
        self.close, self.volume = c, v


def make_series():
    candles = []
    ts = TS0
    for i in range(40):
        o = 100 + 0.2 * math.sin(i)
        candles.append(CandleStub(ts, o, o + 0.3, o - 0.3, o + 0.05))
        ts += H
    candles.append(CandleStub(ts, 100.5, 101.0, 100.2, 100.8)); ts += H
    candles.append(CandleStub(ts, 101.0, 110.0, 101.0, 108.0, v=800)); ts += H
    candles.append(CandleStub(ts, 108.0, 109.0, 106.0, 108.5, v=150)); ts += H
    for i in range(20):
        o = 108 - i * 0.2
        candles.append(CandleStub(ts, o, o + 0.3, o - 0.4, o - 0.15))
        ts += H
    return candles


def make_legacy_signal() -> Signal:
    """构造真实 strategy.Signal（与 agent.py 主循环同构）。"""
    impulse = Candle(timestamp=TS0 + 2 * H, open=101.0, high=110.0,
                     low=101.0, close=108.0, volume=800)
    fvg = FVG(direction="long", top=106.0, bottom=101.0, width_pct=5.0,
              candle_ts=TS0 + 2 * H, timeframe="1H", impulse_candle=impulse,
              fvg_index=31, is_abnormal=True, sigma=4.2, volume_ratio=3.0)
    return Signal(inst_id="BTC-USDT-SWAP", fvg=fvg, entry_price=104.0,
                  stop_loss=99.0, take_profit=112.0, leverage=3,
                  position_side="long", score=0.82, reason="test")


class TestLegacySignalCompatibility(unittest.TestCase):
    """from_legacy_fvg 对真实 strategy.Signal.fvg 的兼容性。"""

    def setUp(self):
        self.signal = make_legacy_signal()
        self.det = FVGDetector({})

    def test_legacy_fvg_required_attrs(self):
        """Signal.fvg 必须含 from_legacy_fvg 所需全部属性（无 AttributeError）。"""
        f = self.signal.fvg
        for attr in ("top", "bottom", "direction", "timeframe", "width_pct",
                     "fvg_index", "impulse_candle", "is_abnormal", "sigma",
                     "volume_ratio", "candle_ts"):
            self.assertTrue(hasattr(f, attr), f"缺少属性 {attr}")
        d = from_legacy_fvg(f)
        self.assertEqual(d.direction, "bullish")
        self.assertEqual((d.gap_high, d.gap_low), (106.0, 101.0))
        self.assertEqual((d.start_idx, d.end_idx), (30, 32))
        self.assertTrue(d.is_abnormal)
        self.assertEqual(d.volume_at_formation, 800.0)

    def test_full_pipeline_with_signal(self):
        """真实信号全链路: 适配 → 特征 → ML 预测。"""
        from fvg_ml_ranker import FVGMLRanker  # noqa: E402
        d = from_legacy_fvg(self.signal.fvg)
        feats = self.det.compute_features(d, make_series())
        self.assertIn("fvg_width_pct", feats)
        self.assertEqual(feats["fvg_width_pct"], 5.0)
        ranker = FVGMLRanker()
        p = ranker.predict(feats)  # 未训练 → 中性 0.5
        self.assertEqual(p, 0.5)

    def test_signal_extra_json_serializable(self):
        """_record_signal_quant 落库的 FVG 字段必须可 JSON 序列化。"""
        f = self.signal.fvg
        imp = f.impulse_candle
        extra = {
            "fvg_timeframe": f.timeframe,
            "fvg_width_pct": f.width_pct,
            "fvg_top": f.top,
            "fvg_bottom": f.bottom,
            "fvg_direction": f.direction,
            "fvg_index": f.fvg_index,
            "fvg_is_abnormal": bool(f.is_abnormal),
            "fvg_sigma": float(f.sigma),
            "fvg_volume_ratio": float(f.volume_ratio),
            "fvg_candle_ts": int(f.candle_ts),
            "fvg_impulse": {"ts": imp.timestamp, "o": imp.open, "h": imp.high,
                            "l": imp.low, "c": imp.close, "v": imp.volume},
        }
        blob = json.dumps(extra)  # 不得抛异常
        restored = json.loads(blob)
        self.assertEqual(restored["fvg_direction"], "long")
        self.assertEqual(restored["fvg_impulse"]["h"], 110.0)


class TestBTCInjection(unittest.TestCase):
    """_btc_candles 注入 → corr_with_btc 特征生效。"""

    def setUp(self):
        self.det = FVGDetector({})
        self.candles = make_series()
        self.fvg = from_legacy_fvg(make_legacy_signal().fvg)

    def test_no_injection_neutral(self):
        feats = self.det.compute_features(self.fvg, self.candles)
        self.assertEqual(feats["corr_with_btc"], 0.0)

    def test_injected_correlated_btc(self):
        # BTC 与标的 1:1 同走势 → 对数收益完全相关 → corr ≈ 1
        btc = [CandleStub(c.timestamp, c.open, c.high, c.low, c.close, c.volume)
               for c in self.candles]
        self.fvg._btc_candles = btc
        feats = self.det.compute_features(self.fvg, self.candles)
        self.assertGreater(abs(feats["corr_with_btc"]), 0.9)

    def test_injected_uncorrelated_btc(self):
        # 独立频率正弦 → 对数收益与标的近乎不相关 → corr ≈ 0
        btc = [CandleStub(c.timestamp,
                          500 + 2 * math.sin(i * 0.37),
                          500 + 2 * math.sin(i * 0.37) + 0.1,
                          500 + 2 * math.sin(i * 0.37) - 0.1,
                          500 + 2 * math.sin(i * 0.37))
               for i, c in enumerate(self.candles)]
        self.fvg._btc_candles = btc
        feats = self.det.compute_features(self.fvg, self.candles)
        self.assertLess(abs(feats["corr_with_btc"]), 0.5)


class TestTrainScript(unittest.TestCase):
    """train_fvg_model --dry-run 端到端。"""

    def test_dry_run(self):
        import train_fvg_model  # noqa: E402
        workdir = tempfile.mkdtemp(prefix="fvg_int_")
        report = train_fvg_model.run_dry_run(model_type="auto", workdir=workdir)
        self.assertIn("n_samples", report)
        self.assertGreaterEqual(report["n_samples"], 20)
        self.assertGreaterEqual(report["n_positive"], 1)
        self.assertGreaterEqual(report["n_negative"], 1)
        self.assertIn("holdout", report)
        self.assertIn("auc", report["holdout"])
        self.assertIn("feature_importance_top10", report)
        self.assertTrue(os.path.exists(report["model_path"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
