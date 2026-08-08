# -*- coding: utf-8 -*-
"""
纸面交易引擎单元测试 — 覆盖虚拟余额模拟建仓全生命周期。

覆盖：
  - 开仓（限价挂单、张数口径与实盘一致、同币种去重）
  - 限价回补成交 / 超时取消
  - 止盈/止损/动态ROI/时间退出（含 SL 滑点）
  - 盈亏与手续费结算
  - 状态持久化往返
  - 持仓字典结构与 monitor_positions 兼容
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paper_trading import PaperTradingEngine


def _tmp_state_file():
    """为每个测试实例生成唯一临时 state 文件。

    修复: 若不隔离，测试会把 TEST-USDT-SWAP 等假仓位/盈亏写入真实
    paper_state.json，污染纸面模拟余额与交易记录（曾致真实运行中
    出现 TEST-USDT-SWAP 平仓记录、余额从 1000 被扣至 994.65）。
    """
    fd, path = tempfile.mkstemp(prefix="paper_test_", suffix=".json")
    os.close(fd)
    return path


class _Sig:
    """Signal 鸭子对象（仅含纸面引擎所需字段）。"""
    def __init__(self, inst_id="TEST-USDT-SWAP", side="long",
                 entry=100.0, sl=98.0, tp=104.0, lev=3):
        self.inst_id = inst_id
        self.position_side = side
        self.entry_price = entry
        self.stop_loss = sl
        self.take_profit = tp
        self.leverage = lev
        self.signal_id = "sig_1"


_INFO = {
    "ctVal": "0.01", "minSz": "1", "lotSz": "1", "tickSz": "0.1",
}


def _cfg(balance=1000.0, **over):
    cfg = {
        "paper": {"enabled": True, "balance": balance, "limit_timeout_min": 15,
                  "maker_fee": 0.0002, "taker_fee": 0.0005,
                  "state_file": _tmp_state_file()},
        "risk": {"risk_per_trade_pct": 1.0, "margin_pct": 30.0,
                 "max_hold_hours": 48,
                 "dynamic_roi": {"240": 0.015, "120": 0.025, "60": 0.035, "0": 0.05}},
    }
    for k, v in over.items():
        cfg["risk"][k] = v
    return cfg


def _candle(high, low, close, ts=0):
    return {"high": high, "low": low, "close": close, "ts": ts}


class TestPaperOpen(unittest.TestCase):
    def test_open_creates_pending_limit(self):
        eng = PaperTradingEngine(_cfg())
        sig = _Sig()
        ord_id = eng.open_position(sig, _INFO, _cfg()["risk"], 1000.0)
        self.assertTrue(ord_id and str(ord_id).startswith("paper_open_"))
        pos = eng._positions["TEST-USDT-SWAP"]
        self.assertFalse(pos.filled, "限价挂单初始未成交")
        self.assertEqual(pos.entry_px, 100.0)
        self.assertGreater(pos.size, 0)

    def test_open_dedupe_same_inst(self):
        eng = PaperTradingEngine(_cfg())
        sig = _Sig()
        eng.open_position(sig, _INFO, _cfg()["risk"], 1000.0)
        # 同币种重复开仓应被拒绝
        ord_id = eng.open_position(sig, _INFO, _cfg()["risk"], 1000.0)
        self.assertTrue(str(ord_id).startswith("paper_already_"))
        self.assertEqual(len(eng._positions), 1)

    def test_open_zero_size_returns_none(self):
        # 止损方向错误（long 的 SL >= entry）→ 张数为 0 → 拒绝
        eng = PaperTradingEngine(_cfg())
        bad = _Sig(sl=101.0, tp=105.0)  # SL 高于入场价，非法
        self.assertIsNone(eng.open_position(bad, _INFO, _cfg()["risk"], 1000.0))

    def test_size_matches_live_formula(self):
        # equity=1000, risk=1% → risk_amount=10; SL 距离 2% → 仓位价值 500;
        # margin=500/3=166.67 < 300 上限; sz=500/(100*0.01)=500 张
        from executor import calculate_position_size
        sz, margin = calculate_position_size(
            equity=1000.0, entry_price=100.0, stop_loss=98.0,
            risk_pct=1.0, leverage=3, margin_pct=30.0, direction="long",
            contract_value=0.01, min_sz=1.0, sz_precision=0)
        eng = PaperTradingEngine(_cfg())
        eng.open_position(_Sig(), _INFO, _cfg()["risk"], 1000.0)
        pos = eng._positions["TEST-USDT-SWAP"]
        self.assertEqual(pos.size, sz)
        self.assertAlmostEqual(pos.margin, margin, places=6)


class TestPaperFillAndExit(unittest.TestCase):
    def _engine_with_long_position(self, filled=True, **risk_over):
        eng = PaperTradingEngine(_cfg(**risk_over))
        eng.open_position(_Sig(), _INFO, _cfg(**risk_over)["risk"], 1000.0)
        pos = eng._positions["TEST-USDT-SWAP"]
        if filled:
            # 触发限价回补成交
            eng.set_market_data_provider(
                lambda inst: {"candles": [_candle(99.0, 99.0, 99.0)], "mark": 99.0})
            eng.update()
            self.assertTrue(pos.filled, "做多限价单应回补成交")
        return eng, pos

    def test_limit_fill_on_retrace(self):
        eng, pos = self._engine_with_long_position()
        self.assertTrue(pos.filled)
        self.assertEqual(pos.entry_px, 100.0)
        # 入场手续费已扣
        self.assertAlmostEqual(eng.balance, 1000.0 - 500.0 * 0.0002, places=6)

    def test_pending_timeout_cancel(self):
        # 注意: limit_timeout_min 属于 paper 配置段，需显式放入 paper
        pcfg = {**_cfg()["paper"], "limit_timeout_min": 0}  # 立即超时
        eng = PaperTradingEngine({"paper": pcfg, "risk": _cfg()["risk"]})
        eng.open_position(_Sig(), _INFO, _cfg()["risk"], 1000.0)
        eng.set_market_data_provider(
            lambda inst: {"candles": [_candle(101.0, 101.0, 101.0)], "mark": 101.0})
        eng.update()
        self.assertEqual(eng._positions, {}, "超时未成交应取消")

    def test_stop_loss_exit_with_slippage(self):
        eng, pos = self._engine_with_long_position()
        # 做多 SL=98, 滑点 0.5% → 平仓价 98*0.995=97.51
        eng.set_market_data_provider(
            lambda inst: {"candles": [_candle(99.5, 97.9, 98.5)], "mark": 97.9})
        eng.update()
        self.assertEqual(eng._positions, {})
        trade = eng._trades[-1]
        self.assertEqual(trade["reason"], "stop_loss")
        self.assertAlmostEqual(trade["exit_px"], 98.0 * 0.995, places=6)
        self.assertLess(trade["pnl"], 0)
        # 盈亏 = 500*(97.51-100) = -1245, 减手续费
        self.assertLess(eng.get_equity(), 1000.0)

    def test_take_profit_exit(self):
        eng = PaperTradingEngine(_cfg())
        # 做空: entry=100, TP=96
        eng.open_position(_Sig(side="short", entry=100.0, sl=103.0, tp=96.0),
                          _INFO, _cfg()["risk"], 1000.0)
        eng.set_market_data_provider(
            lambda inst: {"candles": [_candle(101.0, 101.0, 101.0)], "mark": 101.0})
        eng.update()  # 做空在 high>=entry 时成交
        self.assertTrue(eng._positions["TEST-USDT-SWAP"].filled)
        # 价格跌到 TP 下方 → 止盈
        eng.set_market_data_provider(
            lambda inst: {"candles": [_candle(97.0, 95.8, 96.0)], "mark": 95.8})
        eng.update()
        self.assertEqual(eng._positions, {})
        trade = eng._trades[-1]
        self.assertEqual(trade["reason"], "take_profit")
        self.assertAlmostEqual(trade["exit_px"], 96.0, places=6)
        self.assertGreater(trade["pnl"], 0)

    def test_dynamic_roi_exit(self):
        # 动态 ROI: 持仓 <60min 目标 5%。价格涨到 103.9（未触及 TP=104）→
        # UPL=500*(103.9-100)=+1950, 保证金 166.67 → +1170% >> 5% → 动态ROI落袋
        eng, pos = self._engine_with_long_position()
        eng.set_market_data_provider(
            lambda inst: {"candles": [_candle(103.9, 103.0, 103.9)], "mark": 103.9})
        eng.update()
        self.assertEqual(eng._positions, {}, "动态 ROI 应触发落袋")
        self.assertEqual(eng._trades[-1]["reason"], "dynamic_roi")
        self.assertGreater(eng._trades[-1]["pnl"], 0)

    def test_time_exit(self):
        eng, pos = self._engine_with_long_position(max_hold_hours=0)  # 立即超时
        eng.set_market_data_provider(
            lambda inst: {"candles": [_candle(99.5, 99.5, 99.5)], "mark": 99.5})
        eng.update()
        self.assertEqual(eng._positions, {})
        self.assertEqual(eng._trades[-1]["reason"], "time_exit")

    def test_sl_priority_over_tp_same_candle(self):
        # 同根 K 线同时穿越 SL 与 TP → 按 SL 保守处理
        eng, pos = self._engine_with_long_position()
        eng.set_market_data_provider(
            lambda inst: {"candles": [_candle(105.0, 97.0, 100.0)], "mark": 100.0})
        eng.update()
        self.assertEqual(eng._positions, {})
        self.assertEqual(eng._trades[-1]["reason"], "stop_loss")


class TestPaperState(unittest.TestCase):
    def test_to_positions_dict_shape(self):
        eng, _ = TestPaperFillAndExit()._engine_with_long_position()
        pd = eng.to_positions_dict()
        self.assertIn("TEST-USDT-SWAP", pd)
        pos = pd["TEST-USDT-SWAP"]
        for key in ("pos_side", "size", "avg_px", "mark_px", "upl",
                    "upl_ratio_pct", "margin", "leverage", "c_time"):
            self.assertIn(key, pos, f"缺少 {key}")

    def test_persistence_roundtrip(self, tmp_file="test_paper_state.json"):
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        try:
            eng = PaperTradingEngine({**_cfg(), "paper": {
                **_cfg()["paper"], "state_file": tmp_file}})
            eng.open_position(_Sig(), _INFO, _cfg()["risk"], 1000.0)
            eng.set_market_data_provider(
                lambda inst: {"candles": [_candle(99.0, 99.0, 99.0)], "mark": 99.0})
            eng.update()  # 成交
            eng.save()
            eng2 = PaperTradingEngine({**_cfg(), "paper": {
                **_cfg()["paper"], "state_file": tmp_file}})
            eng2.load()
            self.assertIn("TEST-USDT-SWAP", eng2._positions)
            self.assertTrue(eng2._positions["TEST-USDT-SWAP"].filled)
            self.assertAlmostEqual(eng2.balance, eng.balance, places=6)
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def test_equity_reflects_upl(self):
        eng, pos = TestPaperFillAndExit()._engine_with_long_position()
        eng.set_market_data_provider(
            lambda inst: {"candles": [_candle(102.0, 101.0, 101.5)], "mark": 101.5})
        eng.update()
        # 未平仓 → 权益 = 现金 + 浮盈(500*(101.5-100)=750)
        self.assertGreater(eng.get_equity(), 1000.0)

    def test_summary_contains_key_metrics(self):
        eng = PaperTradingEngine(_cfg())
        eng.open_position(_Sig(), _INFO, _cfg()["risk"], 1000.0)
        eng.set_market_data_provider(
            lambda inst: {"candles": [_candle(105.0, 97.0, 100.0)], "mark": 100.0})
        eng.update()  # SL 触发 → 产生一笔已实现交易
        text = eng.summary()
        for key in ("初始余额", "当前权益", "已实现盈亏", "平仓交易"):
            self.assertIn(key, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
