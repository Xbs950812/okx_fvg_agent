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
        self.plan_placed = 0
        self.plan_args = None

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

    def place_plan_order(self, inst_id, td_mode, side, pos_side, sz,
                         trigger_px, ord_px, trigger_px_type="last"):
        self.plan_placed += 1
        self.plan_args = (inst_id, side, trigger_px, ord_px)
        return "plan1"


def _run_execute_sig(inst_id, side, entry, sl, tp, leverage, client=None,
                     risk_cfg=None):
    from executor import execute_signal
    # execute_signal 只使用 client 的部分方法，用假客户端即可；
    # 类型仅用于 isinstance 检查（无），故直接传入假客户端
    _client = client or _FakeClient()
    _config = {"risk": {"risk_per_trade_pct": 1.0, "margin_pct": 30.0,
                        "margin_mode": "isolated",
                        "position_sizing": "risk",
                        "enforce_risk_cap": True,
                        "max_position_leverage": 0,
                        # 旧 liq 拒单测试语义: 显式开启 fail-closed
                        "liq_check_fail_closed": True}}
    if risk_cfg:
        _config["risk"].update(risk_cfg)
    _signal = _make_signal(inst_id=inst_id, side=side,
                           entry=entry, sl=sl, tp=tp, leverage=leverage)
    ord_id = execute_signal(_client, _signal, equity=100.0,
                            config=_config, instrument_info=_client.get_instrument_info(inst_id))
    return ord_id, _signal


def _run_execute(inst_id, side, entry, sl, tp, leverage, client=None,
                 risk_cfg=None):
    return _run_execute_sig(inst_id, side, entry, sl, tp, leverage,
                            client=client, risk_cfg=risk_cfg)[0]


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


def test_liq_check_full_leverage_default_allow():
    """满倍率模式 (2026-08-10): liq_check_fail_closed 默认 false —
    满杠杆下止损距离 ≥ 爆仓安全距离时**降杠杆**使止损先于爆仓(用户要求
    "先止损再爆仓"), 不再警告放行; 仅杠杆非法时拒单。

    用户模型: 30% 余额做保证金 × 币种最大杠杆, 剩余 70% 余额当爆仓缓冲;
    2026-08-10 起强制止损先于爆仓, 单笔亏损 ≤ 保证金。
    """
    # 50x + 3% 止损: liq_dist=1.5%, 止损 3% ≥ 1.5%×0.5 → 降杠杆止损优先
    client = _FakeClient(tiers={"mmr": "0.005", "maxLever": "50"})
    ord_id, sig = _run_execute_sig("FULLLEV-SWAP", "long", 100.0, 97.0, 104.0, 50,
                                   client=client,
                                   risk_cfg={"liq_check_fail_closed": False})
    assert ord_id is not None, "满倍率默认(不fail-closed)应降杠杆放行"
    # 验证降杠杆: 新杠杆 = floor(1/(3%/0.5 + 0.5%)) = floor(15.38) = 15x
    assert sig.leverage == 15, f"应降杠杆到 15x, 实际 {sig.leverage}x"
    # 验证止损先于爆仓: 15x 爆仓距离 = 6.67%-0.5% = 6.17%, 安全距离 3.08% > 3%
    assert 3.0 < (1.0 / sig.leverage - 0.005) * 0.5 * 100 + 1e-9, \
        "降杠杆后爆仓安全距离必须覆盖止损距离"
    # 杠杆非法 (爆仓距离≤0) 时即使 fail-closed=false 也必须拒单
    client2 = _FakeClient(tiers={"mmr": "0.005", "maxLever": "1000"})
    # 1000x: 1/1000=0.1% - 0.5% < 0 → liq_dist<=0 拒单
    ord_id2 = _run_execute("BADLEV-SWAP", "long", 100.0, 97.0, 104.0, 1000,
                           client=client2)
    assert ord_id2 is None, "杠杆非法(爆仓距离≤0)必须拒单"


def test_liq_check_derates_leverage_stop_first():
    """降杠杆止损优先 (2026-08-10 用户要求): SL 先于爆仓触发, 亏损 ≤ 保证金。

    回归: 满倍率警告放行 → 50x + SL=-5.09% vs 爆仓 -2% → 价格先爆仓,
    paper 缺强平模拟按全额亏损结算, 单笔 -28.85 超过保证金 11.40 穿仓。
    修复后: 50x+5% 止损 → 降杠杆到 9x (floor(1/(5%/0.5+0.5%))=floor(9.52)=9),
    爆仓距离 11.1%-0.5%=10.6%, 安全距离 5.3% > 5% → 止损必然先触发。
    """
    client = _FakeClient(tiers={"mmr": "0.005", "maxLever": "50"})
    ord_id, sig = _run_execute_sig("DERATE-SWAP", "long", 100.0, 95.0, 110.0, 50,
                                   client=client,
                                   risk_cfg={"liq_check_fail_closed": False})
    assert ord_id is not None, "应降杠杆放行(非拒单)"
    assert sig.leverage == 9, f"应降杠杆到 9x, 实际 {sig.leverage}x"
    _liq = 1.0 / sig.leverage - 0.005
    assert 5.0 / 100.0 < _liq * 0.5 + 1e-9, \
        f"止损距离必须小于爆仓安全距离, liq={_liq:.4f}"
    # SL 距离保留原值(降杠杆不牺牲止损宽度)
    assert abs(sig.stop_loss - 95.0) < 1e-9, "降杠杆不得改动止损价"


def test_liq_check_derate_short_side():
    """做空方向降杠杆 (2026-08-10): 止损方向对称, 爆仓点在 entry 上方。
    3% 止损 → floor(1/(3%/0.5+0.5%)) = floor(15.38) = 15x。
    """
    client = _FakeClient(tiers={"mmr": "0.005", "maxLever": "50"})
    ord_id, sig = _run_execute_sig("SHORT-DERATE", "short", 100.0, 103.0, 96.0, 50,
                                   client=client,
                                   risk_cfg={"liq_check_fail_closed": False})
    assert ord_id is not None
    assert sig.leverage == 15, f"做空应降杠杆到 15x, 实际 {sig.leverage}x"
    _liq = 1.0 / sig.leverage - 0.005
    assert 3.0 / 100.0 < _liq * 0.5 + 1e-9


def test_resolve_full_leverage_uses_tier_max():
    """满倍率模式 (2026-08-09): 执行杠杆 = 币种 OKX position-tiers maxLever。"""
    from executor import resolve_full_leverage
    # tiers 返回 maxLever=50 → 满杠杆 50x (覆盖信号建议 3x)
    client = _FakeClient(tiers={"mmr": "0.005", "maxLever": "50"})
    lev = resolve_full_leverage(client, "BTC-USDT-SWAP", 3, {"max_position_leverage": 0})
    assert lev == 50, f"应取币种最大杠杆 50x, 实际 {lev}x"
    # tiers 获取失败 → 回退信号杠杆 (max_position_leverage=0 不封顶)
    client_none = _FakeClient(tiers=None)
    lev2 = resolve_full_leverage(client_none, "BTC-USDT-SWAP", 3,
                                 {"max_position_leverage": 0})
    assert lev2 == 3, f"tiers 失败应回退信号杠杆 3x, 实际 {lev2}x"
    # max_position_leverage>0 时仍封顶
    lev3 = resolve_full_leverage(client, "BTC-USDT-SWAP", 3,
                                 {"max_position_leverage": 5})
    assert lev3 == 5, f"配置封顶 5x 应生效, 实际 {lev3}x"


def _mk_ticker(inst_id, last, open24, vol, bid=None, ask=None, high=None, low=None):
    return {
        "instId": inst_id,
        "last": str(last),
        "open24h": str(open24),
        "volCcy24h": str(vol),
        "bidPx": str(bid if bid is not None else last),
        "askPx": str(ask if ask is not None else last),
        "high24h": str(high if high is not None else last),
        "low24h": str(low if low is not None else last),
    }


def test_compute_movers_priority_and_thresholds():
    """涨跌幅榜 (2026-08-09 用户要求): 极端波动币种优先进扫描队列。"""
    from executor import compute_movers
    tickers = [
        # inst, last, open24h, vol24h
        _mk_ticker("BTC-USDT-SWAP", 60000, 60000, 5_000_000_000),   # 0% 不入选
        _mk_ticker("MUBARAK-USDT-SWAP", 1.43, 1.00, 600_000_000),   # +43% 涨幅榜
        _mk_ticker("BICO-USDT-SWAP", 0.554, 1.00, 300_000_000),     # -44.6% 跌幅榜
        _mk_ticker("SMALL-USDT-SWAP", 1.20, 1.00, 500_000),         # +20% 但量不足 1M → 剔除
        _mk_ticker("LOWPCT-USDT-SWAP", 1.05, 1.00, 50_000_000),     # +5% < 8% → 剔除
        _mk_ticker("BOME-USDT-SWAP", 1.24, 1.00, 10_000_000_000),   # +24% 涨幅榜
        _mk_ticker("ETH-USDT-SWAP", 3500, 3450, 8_000_000_000),     # +1.4% 不入选
    ]
    cfg = {"market_movers": {"enabled": True, "count": 20,
                             "min_move_pct": 8.0,
                             "min_volume_24h_usd": 1_000_000}}
    movers = compute_movers(tickers, cfg)
    ids = [m["instId"] for m in movers]
    # 涨幅榜+跌幅榜都入选, 且按 |涨跌幅| 降序 (BICO -44.6% 排最前)
    assert "MUBARAK-USDT-SWAP" in ids
    assert "BICO-USDT-SWAP" in ids
    assert "BOME-USDT-SWAP" in ids
    assert ids[0] == "BICO-USDT-SWAP", f"应按 |涨跌幅| 降序, 实际 {ids}"
    assert ids[1] == "MUBARAK-USDT-SWAP", f"应按 |涨跌幅| 降序, 实际 {ids}"
    # 成交量不足 1M 与 |涨跌|<8% 被剔除
    assert "SMALL-USDT-SWAP" not in ids
    assert "LOWPCT-USDT-SWAP" not in ids
    assert "BTC-USDT-SWAP" not in ids
    # move_pct 字段带 24h 涨跌幅
    by_id = {m["instId"]: m for m in movers}
    assert abs(by_id["BICO-USDT-SWAP"]["move_pct"] - (-44.6)) < 0.01
    # enabled=false 时返回空
    assert compute_movers(tickers, {"market_movers": {"enabled": False}}) == []


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


def test_paper_trailing_atr_candle_objects():
    """纸面移动止损 (2026-08-09 回归): Candle 对象格式 K 线 ATR 必须算得出。

    Bug: 行情源返回 Candle dataclass (strategy.candles_from_raw)，而 _atr14
    用 c["high"] 字典键访问 → TypeError 被静默捕获 → ATR=0 → TS 退化为
    "TP 距离 50% 才激活" (RAVE +3.4% 仍未激活 TS 实测)。修复后 ATR 路径
    激活阈值 = 0.5×ATR，涨幅 +0.4 即激活。
    """
    from strategy import Candle
    from paper_trading import PaperTradingEngine, _atr14
    # Candle 对象格式必须算出 ATR
    clist = [Candle(timestamp=i, open=100.0, high=100.5, low=100.0,
                    close=100.3, volume=10) for i in range(20)]
    atr = _atr14(clist)
    assert atr > 0.0, f"Candle 对象格式必须算出 ATR, 实际 {atr}"
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
        pos.filled = True
        # ATR=0.5, 激活阈值=0.25: mark=100.3 (+0.3) 即应激活
        eng._update_trailing(pos, {"mark": 100.3, "candles": clist})
        assert pos.extra.get("ts_activated") is True, \
            "Candle 对象格式下 ATR 激活路径必须生效 (RAVE 回归)"
        assert pos.sl_px > 98.5, f"SL 应上移至 ATR 追踪位, 实际 {pos.sl_px:.4f}"


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
    """FVG Hunter 硬门禁 (2026-08-08): 横盘拒绝 / 极端放行 / 数据不足 fail-closed。"""
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
    # 数据不足 → fail-closed 拒绝 (2026-08-13 加固: 防次新币绕过门禁)
    r3 = _extreme_move_reject_reason(_make_choppy_candles(10), 100.0, 14, 25.0, 2.0)
    assert r3 is not None and "数据不足" in r3, f"K线不足应拒绝开仓, 实际 {r3!r}"
    # 门禁关闭 (0) → 恒放行
    r4 = _extreme_move_reject_reason(choppy, 100.0, 14, 0.0, 0.0)
    assert r4 is None, "门禁关闭时应放行"


def test_mover_direction_gate():
    """方向-涨跌幅一致性门禁 (2026-08-13): 超跌禁追空 / 暴涨禁追多。"""
    from strategy import _calc_24h_move_pct, _mover_direction_blocked
    # 超跌币(24h -20%) 做空 → 拒绝
    r1 = _mover_direction_blocked("short", -20.0, 8.0)
    assert r1 is not None and "禁追空" in r1, f"超跌追空应被拒, 实际 {r1!r}"
    # 超跌币做多 → 放行 (抄底方向正确)
    r2 = _mover_direction_blocked("long", -20.0, 8.0)
    assert r2 is None, f"超跌做多应放行, 实际 {r2!r}"
    # 暴涨币(24h +20%) 做多 → 拒绝
    r3 = _mover_direction_blocked("long", 20.0, 8.0)
    assert r3 is not None and "禁追多" in r3, f"暴涨追多应被拒, 实际 {r3!r}"
    # 暴涨币做空 → 放行
    r4 = _mover_direction_blocked("short", 20.0, 8.0)
    assert r4 is None, f"暴涨做空应放行, 实际 {r4!r}"
    # 阈值内(±8%内) 双向放行
    assert _mover_direction_blocked("short", -7.9, 8.0) is None
    assert _mover_direction_blocked("long", 7.9, 8.0) is None
    # 数据不足 → fail-open 放行
    assert _mover_direction_blocked("short", None, 8.0) is None
    # 门禁关闭 (阈值<=0) → 恒放行
    assert _mover_direction_blocked("short", -20.0, 0.0) is None
    # _calc_24h_move_pct: 由 1H 收盘价回推 24h
    candles = _make_choppy_candles(30)  # 30 根, 最后一根 close≈100
    candles[-1] = Candle(timestamp=29, open=80, high=81, low=79, close=80, volume=10)
    mv = _calc_24h_move_pct({"1H": candles}, 80.0)
    # 24h 前 ≈ candles[-25].close, 现价 80 → 涨跌幅应接近 0 (基数也是 100)
    assert mv is not None, "1H 数据充足应能算 24h 涨跌幅"
    # 数据不足 (<25 根) → None
    assert _calc_24h_move_pct({"1H": candles[:10]}, 80.0) is None
    print("PASS test_mover_direction_gate")


def test_mover_direction_gate_deep_drop():
    """深度跌幅(>30%) 门禁端到端拦截 (2026-08-13 用户要求)。

    构造真实 1H K 线场景: 24h 前(-25 根) close=100, 现价 68 → 跌幅 -32%。
    验证 _calc_24h_move_pct 回推出 -32% 后, _mover_direction_blocked 正确
    拦截做空信号(且放行做多抄底)。
    """
    from strategy import _calc_24h_move_pct, _mover_direction_blocked
    candles = []
    # 前 6 根(索引0~5) close=100, 保证 candles[-25](索引5) close=100 为 24h 基准
    for i in range(6):
        candles.append(Candle(timestamp=i, open=100.0, high=101.0,
                              low=99.0, close=100.0, volume=10.0))
    # 后 24 根: 99 → 68 单边暴跌 (模拟跌幅榜深度超跌币)
    for i in range(24):
        px = 99.0 - 31.0 * i / 23.0
        candles.append(Candle(timestamp=6 + i, open=px * 1.01, high=px * 1.02,
                              low=px * 0.98, close=px, volume=10.0))
    assert len(candles) == 30

    mv = _calc_24h_move_pct({"1H": candles}, 68.0)
    assert mv is not None, "1H 数据充足应能算 24h 涨跌幅"
    assert mv < -30.0, f"跌幅应 >30%, 实际 {mv:.2f}%"

    # 深度超跌 → 做空被拦截 (核心断言)
    r = _mover_direction_blocked("short", mv, 8.0)
    assert r is not None and "禁追空" in r and "-3" in r, \
        f"深度超跌做空应被拦截, 实际 {r!r}"

    # 深度超跌 → 做多放行 (抄底方向正确)
    r2 = _mover_direction_blocked("long", mv, 8.0)
    assert r2 is None, f"深度超跌做多应放行, 实际 {r2!r}"
    print("PASS test_mover_direction_gate_deep_drop")


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


def _make_volatile_candles(n=30, move_pct=2.0, start=57.0):
    """每根 +move_pct% 高波动趋势 → ATR%≈2-3% (测试 ATR 挂钩深度阈值用)。"""
    out, px = [], start
    for i in range(n):
        px = px * (1 + move_pct / 100.0)
        out.append(Candle(timestamp=i, open=px * 0.99,
                          high=px * (1 + move_pct / 100.0),
                          low=px * (1 - move_pct / 100.0),
                          close=px, volume=10))
    return out


def _make_fvg_long(top=96.0, bottom=95.0, idx=26):
    return FVG(direction="long", top=top, bottom=bottom, width_pct=1.0,
               candle_ts=idx, timeframe="1H",
               impulse_candle=Candle(timestamp=idx, open=95.5, high=96,
                                     low=95, close=95.5, volume=10),
               fvg_index=idx)


def _gen_signal(current_price, atr_mult=4.0, conditional_max=15.0,
                depth_conflict=0.0):
    """生成信号 (conditional 回归测试默认关闭 DepthGate, 验证 conditional 路径独立完好;
    生产 config 默认 direction_depth_conflict_pct=5.0 时 conditional 窗口被 DepthGate 前置覆盖)。"""
    from strategy import generate_signal
    candles = _make_volatile_candles()
    return generate_signal(
        inst_id="MOVER-USDT-SWAP",
        fvg=_make_fvg_long(top=85.0, bottom=84.0, idx=len(candles) - 2),
        current_price=current_price,
        candles=candles,
        liquidity_extension_pct=1.5,
        liquidity_extension_min_pct=1.5,
        max_entry_distance_pct=1.5,
        entry_distance_atr_mult=atr_mult,
        max_conditional_distance_pct=conditional_max,
        direction_depth_conflict_pct=depth_conflict,
        max_leverage=10,
    )


def test_atr_hooked_entry_distance():
    """ATR 挂钩深度阈值 (2026-08-10): 有效挂单距离 = max(1.5%, mult×ATR%)。

    极端波动币(涨跌幅榜) ATR 大 → 阈值自动放大, 允许更深挂;
    关闭挂钩(mult=0)时同一深挂退化为 conditional。
    """
    # current_price=90, 流动性猎手 entry≈85 → 偏离≈5.6%
    # ATR≈3%×current → mult=4 有效阈值≈13% → 5.6% 在阈值内 → 正常限价(非 conditional)
    sig = _gen_signal(90.0, atr_mult=4.0)
    assert sig is not None, "ATR 挂钩放行的深挂不应被拒绝"
    assert not sig.use_conditional_entry, \
        "偏离 5.6% < ATR挂钩阈值(~13%) 应为正常限价单"
    # 关闭挂钩(mult=0): 阈值退回 1.5% → 同一深挂超阈值 → conditional
    sig0 = _gen_signal(90.0, atr_mult=0.0)
    assert sig0 is not None and sig0.use_conditional_entry, \
        "关闭挂钩后同一深挂应标记 conditional 触发单"


def test_deep_fvg_conditional_entry():
    """深挂 conditional 触发单 (2026-08-10 用户要求):
    偏离 > 有效阈值但 ≤ 上限 → 标记触发单(不拒绝); > 上限 → 拒绝。"""
    # current_price=99 → 偏离≈14% (> ATR挂钩阈值≈13%, ≤ 15% 上限)
    sig = _gen_signal(99.0, atr_mult=4.0, conditional_max=15.0)
    assert sig is not None, "阈值内的深挂应生成 conditional 触发单"
    assert sig.use_conditional_entry, "深挂应标记 use_conditional_entry"
    assert sig.entry_trigger_px > 0, "conditional 单应有触发价"
    # 触发价介于 entry 与现价之间 (价格先回落到触发位, 触发后挂限价等回补)
    assert sig.entry_price < sig.entry_trigger_px < 99.0, (
        f"触发价应介于 entry={sig.entry_price:.4f} 与现价 99 之间, "
        f"实际 trigger={sig.entry_trigger_px:.4f}")
    # 超深: current_price=105 → 偏离≈19% > 15% 上限 → 拒绝
    sig_deep = _gen_signal(105.0, atr_mult=4.0, conditional_max=15.0)
    assert sig_deep is None, "超过 conditional 上限的深挂应拒绝"
    # conditional 关闭(0)时深挂直接拒绝
    sig_off = _gen_signal(99.0, atr_mult=4.0, conditional_max=0.0)
    assert sig_off is None, "conditional 关闭时超阈值深挂应拒绝"


def test_executor_conditional_uses_plan_order():
    """execute_signal (2026-08-10): use_conditional_entry 信号走 place_plan_order
    触发单, 而非普通限价单 place_order。"""
    client = _FakeClient(tiers={"mmr": "0.005", "maxLever": "50"})
    sig = _make_signal("BTC-USDT-SWAP", "long", entry=100.0, sl=98.5, tp=104.0,
                       leverage=10)
    sig.use_conditional_entry = True
    sig.entry_trigger_px = 102.0
    _config = {"risk": {"risk_per_trade_pct": 1.0, "margin_pct": 30.0,
                        "margin_mode": "isolated",
                        "position_sizing": "risk",
                        "enforce_risk_cap": True,
                        "max_position_leverage": 0,
                        # 生产默认: 满杠杆下止损=爆仓警告放行(逐仓模型)
                        "liq_check_fail_closed": False}}
    from executor import execute_signal
    ord_id = execute_signal(client, sig, equity=1000.0, config=_config,
                            instrument_info=client.get_instrument_info("BTC-USDT-SWAP"))
    assert ord_id == "plan1", f"conditional 信号应返回 plan 单号, 实际 {ord_id}"
    assert client.plan_placed == 1, "应调用 place_plan_order"
    assert client.placed == 0, "conditional 信号不得走普通限价 place_order"
    assert client.plan_args is not None and client.plan_args[2] == 102.0, \
        "触发价应传给 place_plan_order"
    # 普通信号仍走 place_order
    sig2 = _make_signal("BTC-USDT-SWAP", "long", entry=100.0, sl=98.5, tp=104.0,
                        leverage=10)
    ord_id2 = execute_signal(client, sig2, equity=1000.0, config=_config,
                             instrument_info=client.get_instrument_info("BTC-USDT-SWAP"))
    assert client.plan_placed == 1 and client.placed == 1, \
        "普通信号应走 place_order 而非 plan_order"


def test_paper_conditional_trigger():
    """纸面 conditional 触发单 (2026-08-10): 价格触及 trigger_px 前不成交,
    触发后才按普通限价单逻辑等回补成交。"""
    from paper_trading import PaperTradingEngine
    with tempfile.TemporaryDirectory() as td:
        engine = PaperTradingEngine({"paper": {"balance": 1000.0,
                                               "limit_timeout_min": 0,
                                               "state_file": os.path.join(
                                                   td, "paper_state.json")},
                                     "risk": {"max_hold_hours": 48},
                                     "optimization": {}})
        sig = _make_signal("BTC-USDT-SWAP", "long", entry=100.0, sl=98.5, tp=104.0,
                           leverage=10)
        sig.entry_trigger_px = 99.0  # 触发价在现价之下 (做多需价格回落触发)
        engine.open_position(sig,
                             instrument_info={"ctVal": "0.01", "minSz": "1",
                                              "lotSz": "1"},
                             risk_cfg={"risk_per_trade_pct": 1.0, "margin_pct": 30.0,
                                       "margin_mode": "isolated",
                                       "position_sizing": "risk",
                                       "enforce_risk_cap": True,
                                       "max_position_leverage": 0},
                             equity=1000.0)
        pos = engine._positions["BTC-USDT-SWAP"]
        assert pos.trigger_px == 99.0, "纸面持仓应记录触发价"
        assert not pos.filled, "触发前不应成交"
        # update() 需通过 market_data_fn 取行情; 用 provider 桥接 _cached_md
        engine.set_market_data_provider(lambda inst: engine._cached_md.get(inst))
        # mark=99.5 (> trigger 99): 未触发 → 不成交
        engine._cached_md["BTC-USDT-SWAP"] = {"mark": 99.5, "candles": []}
        engine.update()
        assert not pos.filled, "mark 未触及触发价前不得成交"
        # mark=98.9 (≤ trigger 99): 触发 → 且 ≤ entry 100 → 成交
        engine._cached_md["BTC-USDT-SWAP"] = {"mark": 98.9, "candles": []}
        engine.update()
        assert pos.filled, "触发且回补到 entry 后应成交"
        assert pos.extra.get("trigger_activated", False), "应记录触发状态"


def test_paper_update_sl_tighten_only():
    """纸面止损同步 (2026-08-10): 主循环 CE 抬止损(0R) 必须同步到纸面 sl_px,
    且只向有利方向收紧不放松。

    回归: CE 抬止损只作用于实盘路径, 纸面死等原止损 → 同跌至成本价时实盘
    已 0R 保本出场, 纸面仍亏 -1R, 纸面 PnL 与实盘口径不一致。
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
        signal = _make_signal(entry=100.0, sl=98.5, tp=104.0, leverage=1)
        instrument_info = {"ctVal": "0.01", "minSz": "1", "lotSz": "1"}
        eng.open_position(signal, instrument_info, cfg["risk"], equity=100.0)
        pos = eng._positions["BTC-USDT-SWAP"]
        pos.filled = True
        assert pos.sl_px == 98.5
        eng.update_sl("BTC-USDT-SWAP", 100.0)
        assert pos.sl_px == 100.0, f"CE 抬止损应同步到纸面, 实际 {pos.sl_px}"
        # 只收紧不放松: 新 SL 低于当前 → 忽略
        eng.update_sl("BTC-USDT-SWAP", 99.0)
        assert pos.sl_px == 100.0, "止损不得向不利方向放松"
        # 再收紧仍生效
        eng.update_sl("BTC-USDT-SWAP", 100.5)
        assert pos.sl_px == 100.5
        # 未成交挂单不生效
        eng.open_position(_make_signal("ETH-USDT-SWAP", "long", entry=200.0,
                                       sl=195.0, tp=220.0, leverage=1),
                          instrument_info, cfg["risk"], equity=100.0)
        eng.update_sl("ETH-USDT-SWAP", 199.0)
        assert eng._positions["ETH-USDT-SWAP"].sl_px == 195.0, \
            "未成交挂单不得被 CE 抬止损影响"


def test_paper_liquidation_caps_loss_at_margin():
    """纸面强平模拟 (2026-08-10 穿仓事故修复): 50x 持仓价格触及理论爆仓点
    (亏损=保证金) 即强平, 亏损封顶在保证金, 余额不为负。

    回归: 无强平模拟时 SL(-5%) 在爆仓点(-2%)之后才结算, 单笔亏损 -28.85
    超过保证金 11.40, 余额 25.54 → -3.31 穿仓。
    """
    from paper_trading import PaperTradingEngine
    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "paper": {"enabled": True, "balance": 100.0,
                      "maker_fee": 0.0002, "taker_fee": 0.0005,
                      "state_file": os.path.join(td, "paper_state.json")},
            "risk": {"position_sizing": "margin", "margin_pct": 30.0,
                     "risk_per_trade_pct": 1.0, "enforce_risk_cap": True,
                     "max_position_leverage": 0, "max_positions": 1,
                     "max_hold_hours": 48},
            "optimization": {},
        }
        eng = PaperTradingEngine(cfg)
        signal = _make_signal(entry=100.0, sl=95.0, tp=110.0, leverage=50)
        instrument_info = {"ctVal": "0.01", "minSz": "1", "lotSz": "1"}
        eng.open_position(signal, instrument_info, cfg["risk"], equity=100.0)
        pos = eng._positions["BTC-USDT-SWAP"]
        pos.filled = True
        assert pos.leverage == 50
        # 理论爆仓点 = entry × (1-1/50) = 98.0; SL=95 在爆仓点之下(更远)
        # mark=97.5 已破爆仓点 → 应强平而非等 SL
        eng.set_market_data_provider(lambda inst: eng._cached_md.get(inst))
        eng._cached_md["BTC-USDT-SWAP"] = {"mark": 97.5, "candles": []}
        eng.update()
        assert "BTC-USDT-SWAP" not in eng._positions, "破爆仓点应强平"
        trade = eng._trades[-1]
        assert trade["reason"] == "liquidation", f"应为强平, 实际 {trade['reason']}"
        assert trade["exit_px"] == 98.0, f"强平价应为理论爆仓点 98.0, 实际 {trade['exit_px']}"
        # 亏损封顶: 本金亏损 ≤ margin, 另扣手续费
        assert trade["pnl"] >= -pos.margin - 1.0, \
            f"亏损应封顶在保证金附近, pnl={trade['pnl']} margin={pos.margin}"
        assert eng.balance >= 0, f"余额不得穿仓变负, 实际 {eng.balance}"


def test_paper_stop_loss_precedes_liquidation():
    """SL 比爆仓点更近时正常止损, 不强平 (2026-08-10):
    entry=100, lev=50 → 爆仓点 98(-2%); SL=99(-1%) 更近 →
    价格跌到 98.5 触发 stop_loss, 损失 1% 远小于保证金。
    """
    from paper_trading import PaperTradingEngine
    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "paper": {"enabled": True, "balance": 100.0,
                      "maker_fee": 0.0002, "taker_fee": 0.0005,
                      "state_file": os.path.join(td, "paper_state.json")},
            "risk": {"position_sizing": "margin", "margin_pct": 30.0,
                     "risk_per_trade_pct": 1.0, "enforce_risk_cap": True,
                     "max_position_leverage": 0, "max_positions": 1,
                     "max_hold_hours": 48},
            "optimization": {},
        }
        eng = PaperTradingEngine(cfg)
        signal = _make_signal(entry=100.0, sl=99.0, tp=110.0, leverage=50)
        instrument_info = {"ctVal": "0.01", "minSz": "1", "lotSz": "1"}
        eng.open_position(signal, instrument_info, cfg["risk"], equity=100.0)
        pos = eng._positions["BTC-USDT-SWAP"]
        pos.filled = True
        eng.set_market_data_provider(lambda inst: eng._cached_md.get(inst))
        # mark=98.5: 低于 SL 99 但高于爆仓点 98 → 止损优先
        eng._cached_md["BTC-USDT-SWAP"] = {"mark": 98.5, "candles": []}
        eng.update()
        assert "BTC-USDT-SWAP" not in eng._positions, "触及 SL 应平仓"
        trade = eng._trades[-1]
        assert trade["reason"] == "stop_loss", f"应为止损, 实际 {trade['reason']}"
        # 止损价含滑点: 99 × 0.995 = 98.505
        assert abs(trade["exit_px"] - 99.0 * 0.995) < 1e-6, \
            f"止损执行价错误: {trade['exit_px']}"
        # 损失 = 1% × 名义, 远小于保证金, 不应触发强平封顶
        assert trade["pnl"] > -pos.margin, \
            f"止损损失应远小于保证金, pnl={trade['pnl']} margin={pos.margin}"


def test_paper_liquidation_gap_prefers_stop_loss():
    """跳空场景: 一根 K 线同时穿过 SL 与爆仓点时, 取更近 entry 的 SL 优先
    (2026-08-10 强平模拟): 避免跳空时误判为强平多亏。
    entry=100, lev=50 → 爆仓 98; SL=99(-1%); lo=97 同时穿破两者 → 止损优先。
    """
    from paper_trading import PaperTradingEngine
    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "paper": {"enabled": True, "balance": 100.0,
                      "maker_fee": 0.0002, "taker_fee": 0.0005,
                      "state_file": os.path.join(td, "paper_state.json")},
            "risk": {"position_sizing": "margin", "margin_pct": 30.0,
                     "risk_per_trade_pct": 1.0, "enforce_risk_cap": True,
                     "max_position_leverage": 0, "max_positions": 1,
                     "max_hold_hours": 48},
            "optimization": {},
        }
        eng = PaperTradingEngine(cfg)
        signal = _make_signal(entry=100.0, sl=99.0, tp=110.0, leverage=50)
        instrument_info = {"ctVal": "0.01", "minSz": "1", "lotSz": "1"}
        eng.open_position(signal, instrument_info, cfg["risk"], equity=100.0)
        pos = eng._positions["BTC-USDT-SWAP"]
        pos.filled = True
        eng.set_market_data_provider(lambda inst: eng._cached_md.get(inst))
        # 用 Candle 构造低点 97 (同根 K 线穿破 SL 99 和爆仓 98)
        candle = Candle(timestamp=1, open=99.5, high=99.8, low=97.0,
                        close=97.5, volume=10)
        eng._cached_md["BTC-USDT-SWAP"] = {"mark": 97.5, "candles": [candle]}
        eng.update()
        assert "BTC-USDT-SWAP" not in eng._positions
        trade = eng._trades[-1]
        assert trade["reason"] == "stop_loss", \
            f"跳空应优先止损(更近 entry), 实际 {trade['reason']}"


def _make_momentum_candles(direction="down", n=30, step=0.008, start=110.0):
    """单调趋势 K 线: direction=down 每根 -step%, up 每根 +step%。"""
    out, px = [], start
    for i in range(n):
        if direction == "down":
            px = px * (1 - step)
        else:
            px = px * (1 + step)
        out.append(Candle(timestamp=i, open=px * 1.005,
                          high=px * 1.01, low=px * 0.995,
                          close=px, volume=10))
    return out


def _make_range_candles(n=30, base=100.0, swing=2.0, deep_low=None,
                        deep_idx=(24, 25)):
    """区间震荡 K 线: ±swing% 波动; deep_low 指定某几根的低点(构造前低深挂)。"""
    out = []
    for i in range(n):
        hi = base * (1 + swing / 100.0)
        lo = base * (1 - swing / 100.0)
        if deep_low and i in deep_idx:
            lo = deep_low
        out.append(Candle(timestamp=i, open=base, high=hi,
                          low=lo, close=base, volume=10))
    return out


def test_direction_momentum_gate_rejects_long_in_downtrend():
    """方向动量一致性门 (2026-08-10 方案A): 1H 短期趋势明确向下时做多被否决。

    回归: HTF(4H)滞后 — 1H 已转跌时 4H 仍向上, 系统逆 1H 趋势开多
    (PUMP score=0.91 做多后 -4.6%)。
    """
    from agent import _direction_momentum_gate
    candles = _make_momentum_candles("down")   # 110 → ~86, SMA20 下行
    cfg = {"strategy": {"direction_momentum_gate": {"enabled": True,
                                                    "ma_period": 20}}}
    sig = _make_signal("TREND-SWAP", "long", entry=100.0, sl=95.0, tp=110.0,
                       leverage=10)
    ok, reason = _direction_momentum_gate(sig, candles, cfg)
    assert not ok, "1H 下行趋势中做多应被否决"
    assert "MomentumGate" in reason
    # 做空在同一下行趋势中应放行(顺 1H 趋势)
    sig_short = _make_signal("TREND-SWAP", "short", entry=100.0, sl=105.0,
                             tp=90.0, leverage=10)
    ok2, _ = _direction_momentum_gate(sig_short, candles, cfg)
    assert ok2, "下行趋势做空应与趋势一致, 放行"


def test_direction_momentum_gate_allows_long_in_uptrend():
    """方向动量一致性门: 1H 上行趋势做多放行, 做空被否决。"""
    from agent import _direction_momentum_gate
    candles = _make_momentum_candles("up", start=90.0)   # 90 → ~114, SMA20 上行
    cfg = {"strategy": {"direction_momentum_gate": {"enabled": True,
                                                    "ma_period": 20}}}
    sig = _make_signal("UPTREND-SWAP", "long", entry=100.0, sl=95.0, tp=115.0,
                       leverage=10)
    ok, _ = _direction_momentum_gate(sig, candles, cfg)
    assert ok, "1H 上行趋势做多应与趋势一致, 放行"
    sig_short = _make_signal("UPTREND-SWAP", "short", entry=100.0, sl=105.0,
                             tp=90.0, leverage=10)
    ok2, reason2 = _direction_momentum_gate(sig_short, candles, cfg)
    assert not ok2, "上行趋势做空应被否决"
    assert "MomentumGate" in reason2


def test_direction_momentum_gate_allows_when_flat():
    """方向动量一致性门: 横盘(SMA 走平)不构成明确冲突, 放行(不误杀)。"""
    from agent import _direction_momentum_gate
    candles = [Candle(timestamp=i, open=100.0, high=101.0, low=99.0,
                      close=100.0, volume=10) for i in range(30)]
    cfg = {"strategy": {"direction_momentum_gate": {"enabled": True,
                                                    "ma_period": 20}}}
    sig = _make_signal("FLAT-SWAP", "long", entry=100.0, sl=95.0, tp=110.0,
                       leverage=10)
    ok, _ = _direction_momentum_gate(sig, candles, cfg)
    assert ok, "横盘不构成明确方向冲突, 应放行"


def test_depth_gate_falls_back_to_fvg_reentry():
    """深挂深度-方向校验 (2026-08-10 方案B): 做多挂单价低于现价 >5%
    (预期深跌=接飞刀) → 回退 FVG 回补位, 回退后不深则保留信号。

    回归: NEIRO 深挂 11.5% (HTF 向上 vs 等深跌自相矛盾)。
    """
    from strategy import generate_signal
    # ±2% 波动区间 → ATR%≈2%, _eff_entry_dist = max(1.5, 4×2) = 8%
    # 前低 98 → 深挂 98×0.96 = 94.08, dev=(100-94.08)/100 = 5.92% < 8%
    # (ATR 挂钩第一步不回退) → DepthGate 5.92% > 5% → 回退 FVG 回补位
    candles = _make_range_candles(deep_low=98.0)
    fvg = _make_fvg_long(top=99.0, bottom=98.5, idx=len(candles) - 2)
    sig = generate_signal(
        inst_id="DEPTH-SWAP", fvg=fvg, current_price=100.0,
        candles=candles, liquidity_extension_pct=4.0,
        liquidity_extension_min_pct=3.0,
        max_entry_distance_pct=1.5,
        entry_distance_atr_mult=4.0,
        max_conditional_distance_pct=15.0,
        direction_depth_conflict_pct=5.0,
        swing_lookback_bars=8,
        max_leverage=10,
    )
    assert sig is not None, "回退到 FVG 回补位后应保留信号"
    # 回补位 = max(99 - 0.5×0.15, 98.5) = 98.925 (dev 1.08% < 5%)
    assert abs(sig.entry_price - 98.925) < 0.01, \
        f"应回退到 FVG 回补位 98.925, 实际 {sig.entry_price}"


def test_depth_gate_rejects_when_reentry_still_deep():
    """深挂深度校验: 回退到 FVG 回补位后仍偏离 >5% (缺口本身在深位)
    = 深跌预期未消除 → 否决信号(不接飞刀)。"""
    from strategy import generate_signal
    candles = _make_range_candles(deep_low=98.0)
    # 缺口深: top=94 bottom=93 → 回补位 max(94-0.15, 93) = 93.85
    # dev=(100-93.85)/100 = 6.15% > 5% → 深挂回退后仍超 → 否决
    fvg = _make_fvg_long(top=94.0, bottom=93.0, idx=len(candles) - 2)
    sig = generate_signal(
        inst_id="DEEP-SWAP", fvg=fvg, current_price=100.0,
        candles=candles, liquidity_extension_pct=4.0,
        liquidity_extension_min_pct=3.0,
        max_entry_distance_pct=1.5,
        entry_distance_atr_mult=4.0,
        max_conditional_distance_pct=15.0,
        direction_depth_conflict_pct=5.0,
        swing_lookback_bars=8,
        max_leverage=10,
    )
    assert sig is None, "FVG 回补位仍深偏离(接飞刀)应否决信号"


def test_depth_gate_allows_normal_deep():
    """深挂深度校验: 正常深挂(偏离 ≤5%)不受影响, 保留深挂逻辑。"""
    from strategy import generate_signal
    # ±1% 波动区间(前低 99, swing=1) → 深挂 99×0.96 = 95.04
    # dev=(100-95.04)/100 = 4.96% ≤ 5% → DepthGate 不触发, 保留深挂
    candles = _make_range_candles(swing=1.0)
    fvg = _make_fvg_long(top=99.5, bottom=99.0, idx=len(candles) - 2)
    sig = generate_signal(
        inst_id="NORMAL-SWAP", fvg=fvg, current_price=100.0,
        candles=candles, liquidity_extension_pct=4.0,
        liquidity_extension_min_pct=3.0,
        max_entry_distance_pct=1.5,
        entry_distance_atr_mult=4.0,
        max_conditional_distance_pct=15.0,
        direction_depth_conflict_pct=5.0,
        swing_lookback_bars=8,
        max_leverage=10,
    )
    assert sig is not None, "正常深挂应保留"
    assert abs(sig.entry_price - 95.04) < 0.01, \
        f"正常深挂不应被改动, 实际 entry={sig.entry_price}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASS")
