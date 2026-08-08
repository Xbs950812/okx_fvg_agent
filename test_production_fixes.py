# -*- coding: utf-8 -*-
"""生产审计修复的回归测试（2026-08-07）。

覆盖:
  - P0-A: 开仓前"止损距离 < 爆仓距离×安全系数"硬校验
  - P0-D: _pending_close 元数据持久化往返（重启续跑平仓确认）
  - P0-B: monitor_positions 对 get_positions 失败的 fail-closed 行为
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from okx_client import OKXQueryError
from executor import StateManager, AgentState, monitor_positions
from strategy import Signal, FVG, Candle


def _make_signal(inst_id="BTC-USDT-SWAP", side="long",
                 entry=100.0, sl=98.5, tp=104.0, leverage=10):
    candle = Candle(timestamp=1, open=100, high=101, low=99, close=100, volume=10)
    fvg = FVG(direction=side, top=101, bottom=99, width_pct=2.0,
              candle_ts=1, timeframe="1H", impulse_candle=candle)
    return Signal(inst_id=inst_id, fvg=fvg, entry_price=entry,
                  stop_loss=sl, take_profit=tp, leverage=leverage,
                  position_side=side)


class _FakeClient:
    """execute_signal 所需的最小假客户端。"""

    def __init__(self, tiers=None):
        self._tiers = tiers
        self.placed = 0
        self.algo_placed = 0

    def get_position_tiers(self, inst_id, td_mode="isolated"):
        return self._tiers

    def get_instrument_info(self, inst_id):
        return {"ctVal": "0.01", "minSz": "1", "lotSz": "1", "tickSz": "0.1"}

    def set_leverage(self, **kwargs):
        return True

    def place_order(self, **kwargs):
        self.placed += 1
        return f"ord{self.placed}"

    def place_algo_order(self, **kwargs):
        self.algo_placed += 1
        return "algo1"


def _run_execute(inst_id, side, entry, sl, tp, leverage, client=None,
                 risk_cfg=None):
    from executor import execute_signal
    from okx_client import OKXClient
    # execute_signal 只使用 client 的部分方法，用假客户端即可；
    # 类型仅用于 isinstance 检查（无），故直接传入假客户端
    _client = client or _FakeClient()
    _config = {"risk": {"risk_per_trade_pct": 1.0, "margin_pct": 30.0,
                        "margin_mode": "isolated",
                        "position_sizing": "risk",
                        "enforce_risk_cap": True,
                        "max_position_leverage": 0}}
    if risk_cfg:
        _config["risk"].update(risk_cfg)
    _signal = _make_signal(inst_id=inst_id, side=side,
                           entry=entry, sl=sl, tp=tp, leverage=leverage)
    return execute_signal(_client, _signal, equity=100.0,
                          config=_config, instrument_info=_client.get_instrument_info(inst_id))


def test_liq_check_rejects_high_leverage():
    """P0-A: 50x + 3% 止损 → 爆仓距离 1.5%，安全距离 0.75% → 必须拒单。"""
    client = _FakeClient()
    ord_id = _run_execute("HIGH-SWAP", "long", 100.0, 97.0, 104.0, 50, client=client)
    assert ord_id is None, "50x/3%止损 应被强平距离校验拒绝"
    assert client.placed == 0, "拒单路径不得下单"


def test_liq_check_allows_safe():
    """P0-A: 10x + 1.5% 止损 → 爆仓距离 9.5%，安全距离 4.75% → 放行。"""
    client = _FakeClient()
    ord_id = _run_execute("SAFE-SWAP", "long", 100.0, 98.5, 104.0, 10, client=client)
    assert ord_id is not None, "10x/1.5%止损 应通过强平距离校验"
    assert client.placed == 1


def test_liq_check_uses_tier_mmr():
    """P0-A: 档位 MMR 更高 → 爆仓距离更短 → 边界更严。"""
    client = _FakeClient(tiers={"mmr": "0.01", "maxLever": "10"})
    # 10x, MMR 1% → liq_dist = 10% - 1% = 9%; 止损 5% ≥ 9%×0.5=4.5% → 拒
    ord_id = _run_execute("MMR-SWAP", "long", 100.0, 95.0, 110.0, 10, client=client)
    assert ord_id is None, "高 MMR 档位下宽止损应被拒绝"


def test_pending_close_meta_roundtrip():
    """P0-D: _pending_close 含不可序列化对象时 save/load 往返不丢元数据。"""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "agent_state.json")
        sm = StateManager(path)
        sm.state.daily_loss = -5.0
        sm.state._pending_close = {
            "inst_id": "BTC-USDT-SWAP",
            "ord_id": "12345",
            "pos_side": "long",
            "avg_px": 100.0,
            "size": 3.0,
            "mark_px": 101.0,
            "upl": 3.0,
            "upl_ratio_pct": 1.0,
            "c_time": 1700000000.0,
            "leverage": 10,
            "timestamp": 1700000100.0,
            "signal_id": "sig1",
            "funding_rate": 0.0001,
            "exit_reason": "signal_switch",
            "best_signal": object(),   # 不可序列化对象（真实场景是 Signal）
            "best_analysis": object(),
            "candles_4h": [],
        }
        sm.save()

        # 文件里不允许出现瞬态字段，但必须含 pending_close_meta
        with open(path, "r", encoding="utf-8") as f:
            import json
            raw = json.load(f)
        assert "_pending_close" not in raw
        assert raw["pending_close_meta"]["ord_id"] == "12345"
        assert raw["pending_close_meta"]["inst_id"] == "BTC-USDT-SWAP"

        # 重启后 _pending_close 从元数据重建
        sm2 = StateManager(path)
        pc = sm2.state._pending_close
        assert pc is not None, "重启后必须恢复平仓确认状态"
        assert pc["ord_id"] == "12345"
        assert pc["pos_side"] == "long"
        assert "best_signal" not in pc
        assert sm2.state.daily_loss == -5.0


def test_monitor_positions_fail_closed():
    """P0-B: get_positions 失败(None) → monitor_positions 抛 OKXQueryError，
    绝不当"无持仓"清空 active_signals。"""
    class _BrokenClient:
        def get_positions(self, *a, **k):
            return None
        def get_total_equity(self):
            return 100.0

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "agent_state.json")
        sm = StateManager(path)
        sm.state.active_signals["BTC-USDT-SWAP"] = {"inst_id": "BTC-USDT-SWAP", "size": 3}
        try:
            monitor_positions(_BrokenClient(), sm, {"risk": {"max_daily_loss_pct": 10}})
            raise AssertionError("get_positions=None 应抛 OKXQueryError")
        except OKXQueryError:
            pass
        # 失败时不得清空 active_signals
        assert "BTC-USDT-SWAP" in sm.state.active_signals


def test_paper_leverage_cap_consistency():
    """纸面开仓必须应用 max_position_leverage 封顶（与实盘 execute_signal 口径一致）。

    回归: 2026-08-08 纸面监控发现纸面直接用原始 signal.leverage（3-4x），
    实盘 execute_signal 已封顶 _eff_leverage=1x → 虚拟名义仓位偏离实盘。
    """
    from paper_trading import PaperTradingEngine
    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "paper": {"enabled": True, "balance": 100.0,
                      "state_file": os.path.join(td, "paper_state.json")},
            "risk": {"position_sizing": "margin", "margin_pct": 30.0,
                     "risk_per_trade_pct": 1.0, "enforce_risk_cap": True,
                     "max_position_leverage": 1, "max_positions": 1,
                     "max_hold_hours": 48},
        }
        eng = PaperTradingEngine(cfg)
        # signal.leverage=10, 但 max_position_leverage=1 → 纸面必须按 1x 计算
        signal = _make_signal(leverage=10)
        instrument_info = {"ctVal": "0.01", "minSz": "1", "lotSz": "1"}
        ord_id = eng.open_position(signal, instrument_info, cfg["risk"], equity=100.0)
        assert ord_id is not None, "纸面开仓应成功"
        pos = eng._positions["BTC-USDT-SWAP"]
        assert pos.leverage == 1.0, f"纸面杠杆必须被 max_position_leverage 封顶, 实际 {pos.leverage}x"
        # margin 模式: margin=100×30%=30, 1x 下名义仓位=保证金=30
        notional = pos.size * pos.ct_val * pos.entry_px
        assert abs(notional - 30.0) < 1e-6, f"1x 下名义仓位应=保证金 30, 实际 {notional:.2f}"


def test_paper_trailing_moves_sl():
    """纸面移动止损 (2026-08-08): ATR 激活后 SL 只向有利方向收紧。

    回归: 纸面模式此前不模拟移动止损, trailing 只走 dry-run 假单,
    纸面 SL 恒为信号固定值 → 纸面无法验证追踪止损行为。
    """
    from paper_trading import PaperTradingEngine
    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "paper": {"enabled": True, "balance": 100.0,
                      "state_file": os.path.join(td, "paper_state.json")},
            "risk": {"position_sizing": "margin", "margin_pct": 30.0,
                     "risk_per_trade_pct": 1.0, "enforce_risk_cap": True,
                     "max_position_leverage": 1, "max_positions": 1,
                     "max_hold_hours": 48},
            "optimization": {"trailing_stop_activation_pct": 0.5,
                             "trailing_stop_trail_pct": 0.03},
        }
        eng = PaperTradingEngine(cfg)
        signal = _make_signal(entry=100.0, sl=98.5, tp=104.0, leverage=1)
        instrument_info = {"ctVal": "0.01", "minSz": "1", "lotSz": "1"}
        eng.open_position(signal, instrument_info, cfg["risk"], equity=100.0)
        pos = eng._positions["BTC-USDT-SWAP"]
        pos.filled = True  # 直接置为已成交，跳过挂单流程
        # K线: high=100.5 low=100.0 → TR=0.5, ATR(14)=0.5
        candles = [{"high": 100.5, "low": 100.0, "close": 100.3, "ts": i}
                   for i in range(20)]
        # mark=100.4, 涨幅 0.4 ≥ ATR×0.5=0.25 → 激活
        eng._update_trailing(pos, {"mark": 100.4, "candles": candles})
        assert pos.extra.get("ts_activated") is True, "价格越 0.5xATR 应激活"
        # best=100.4, trail=0.75×ATR=0.375 → new_sl=100.025 > 98.5
        assert pos.sl_px > 98.5, f"SL 应上移, 实际 {pos.sl_px:.4f}"
        # 价格回落到 100.1 → 高水位不变, SL 不得放松
        old_sl = pos.sl_px
        eng._update_trailing(pos, {"mark": 100.1, "candles": candles})
        assert pos.sl_px == old_sl, "SL 不得向不利方向放松"


def _make_choppy_candles(n=60):
    """交替 ±1% 无趋势震荡 → ADX 低 (Wilder 方向性指标近零)。"""
    out, px = [], 100.0
    for i in range(n):
        px = px * (1.01 if i % 2 == 0 else 0.99)
        out.append(Candle(timestamp=i, open=px * 0.995,
                          high=px * 1.015, low=px * 0.985, close=px, volume=10))
    return out


def _make_trend_candles(n=60):
    """单边 +2%/根 强趋势 → ADX 高 + ATR%≈3% (极端行情)。"""
    out, px = [], 100.0
    for i in range(n):
        px = px * 1.02
        out.append(Candle(timestamp=i, open=px * 0.99,
                          high=px * 1.01, low=px * 0.99, close=px, volume=10))
    return out


def test_extreme_move_gate():
    """FVG Hunter 硬门禁 (2026-08-08): 横盘拒绝 / 极端放行 / 数据不足 fail-open。"""
    from strategy import _extreme_move_reject_reason
    # 横盘震荡 (ADX 低) → 拒绝 (ATR%≈3% 达标, ADX 不达标)
    choppy = _make_choppy_candles()
    r = _extreme_move_reject_reason(choppy, 100.0, 14, 25.0, 2.0)
    assert r is not None and "ADX" in r, f"横盘应被 ADX 否决, 实际 {r!r}"
    # 强趋势 + 高波动 → 放行
    trend = _make_trend_candles()
    last_px = 100.0 * (1.02 ** 59)
    r2 = _extreme_move_reject_reason(trend, last_px, 14, 25.0, 2.0)
    assert r2 is None, f"强趋势极端行情应放行, 实际 {r2!r}"
    # 数据不足 → fail-open 放行 (防新币阻塞)
    r3 = _extreme_move_reject_reason(_make_choppy_candles(10), 100.0, 14, 25.0, 2.0)
    assert r3 is None, "K线不足应 fail-open 放行"
    # 门禁关闭 (0) → 恒放行
    r4 = _extreme_move_reject_reason(choppy, 100.0, 14, 0.0, 0.0)
    assert r4 is None, "门禁关闭时应放行"


class _FakeAnalysis:
    """最小分析对象：final_score + final_confidence + channel_agreement。"""
    def __init__(self, final_score, final_confidence=0.5, channel_agreement=0.6):
        self.final_score = final_score
        self.final_confidence = final_confidence
        self.channel_agreement = channel_agreement


class _FakeCacheEntry:
    """最小缓存条目：inst_id + signals + analysis + funding_rate。"""
    def __init__(self, inst_id, signals, analysis, funding_rate=0.0):
        self.inst_id = inst_id
        self.signals = signals
        self.analysis = analysis
        self.funding_rate = funding_rate


def test_switch_candidate_skips_gate_rejected():
    """预换仓候选 (2026-08-08): 无有效 FVG 信号的缓存条目不得参与换仓。

    回归: PIPPIN 实测 — 4H ADX=13 横盘，全部信号被 ExtremeMove 门禁拒绝
    (signals=[])，但研究分 final_score=+1.00 仍混入换仓候选，产生噪音
    决策。修复后候选选择与主扫描路径口径一致：只认真实 FVG 信号。
    """
    from agent import _pick_switch_candidate
    # 横盘币: 研究分虚高 +1.00 但信号已被门禁拒绝 (signals=[])
    rejected = _FakeCacheEntry("PIPPIN-USDT-SWAP", [], _FakeAnalysis(1.00))
    # 有效信号币: 分 0.80，有真实 FVG 信号
    valid = _FakeCacheEntry("WIF-USDT-SWAP", [_make_signal("WIF-USDT-SWAP")],
                            _FakeAnalysis(0.80))
    best = _pick_switch_candidate([(rejected, {}), (valid, {})], {})
    assert best is not None, "有效信号候选应被选中"
    assert best[0].inst_id == "WIF-USDT-SWAP", \
        "横盘无信号条目必须被排除, 不得抢占换仓候选"
    # 全部无信号 → 无候选 (不触发换仓)
    assert _pick_switch_candidate([(rejected, {})], {}) is None
    # 持仓中的币不参与候选
    assert _pick_switch_candidate(
        [(valid, {})], {"WIF-USDT-SWAP": {"size": 1}}) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
