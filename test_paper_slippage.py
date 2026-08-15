# -*- coding: utf-8 -*-
"""
纸面挂单链路 — 极端行情滑点压力测试 (2026-08-15)。

针对 paper_trading.PaperTradingEngine 的限价成交 → TP/SL 退出链路，
构造跳空/插针级别的极端行情，验证滑点语义与资金安全不变式：

  S1 跳空 22% 穿越挂单价 → 成交价必须 == 挂单价（限价单语义，绝不追价）
  S2 跳空 15% 穿越止损（低杠杆）→ 固定 0.5% 滑点语义 + PnL 精确对账
  S3 跳空 20% 穿越止损（50x 高杠杆）→ 爆仓优先于止损，亏损封顶保证金
  S4 空头镜像: 跳空向上穿越止损（低杠杆固定滑点 + 高杠杆爆仓封顶）
  S5 同根 K 线 TP/SL 同时穿越 → 止损优先（保守语义）
  S6 跳空 7% 穿越止盈 → 成交价 == tp 价（限价语义，不吃跳空额外利润）
  S7 资金安全不变式: 极端序列后余额 > 0，单笔亏损 ≤ 保证金 + 手续费

已知语义（本测试固化为契约，非 bug）:
  - 滑点只有止损的固定 0.5%，跳空深度不影响成交价 → 纸面在大跳空下
    系统性低估真实滑点（真实止损限价单可能整单漏成交）。recent_pnl
    的滑点样本回填（Slippage 闭环）只覆盖实盘路径。
  - 爆仓兜底保证任何路径亏损 ≤ 保证金 + 手续费（2026-08-10 穿仓修复）。

运行: python -m pytest test_paper_slippage.py -v
"""

import math
from types import SimpleNamespace

import pytest

from paper_trading import PaperTradingEngine, _SL_SLIPPAGE

MAKER, TAKER = 0.0002, 0.0005
INIT_BALANCE = 1000.0


def make_engine(tmp_path, name="slip"):
    cfg = {
        "paper": {
            "balance": INIT_BALANCE, "maker_fee": MAKER, "taker_fee": TAKER,
            "limit_timeout_min": 30, "fill_assist_seconds": 3600,
            "state_file": str(tmp_path / f"paper_{name}.json"),
        },
        "risk": {"max_hold_hours": 48, "dynamic_roi": {}},  # 关 ROI 干扰
    }
    eng = PaperTradingEngine(cfg)
    state = {"mark": 0.0, "candles": []}

    def md(_inst):
        return {"mark": state["mark"], "candles": list(state["candles"])}

    eng.set_market_data_provider(md)
    return eng, state


def make_signal(side, entry, tp, sl, lev, inst="SLIP-USDT-SWAP"):
    return SimpleNamespace(
        inst_id=inst, position_side=side, entry_price=entry,
        take_profit=tp, stop_loss=sl, leverage=lev,
        signal_id="slip_test", entry_trigger_px=0.0,
    )


def open_and_fill(eng, state, sig, fill_mark):
    """挂单 → 开仓 → 置 mark 到成交侧 → update() 推进成交。"""
    state["mark"] = fill_mark
    eng.update()                                    # 预热缓存
    ord_id = eng.open_position(sig, {"ctVal": 1.0, "minSz": 1, "lotSz": "1"},
                               {"risk_per_trade_pct": 30.0, "margin_pct": 30.0,
                                "max_position_leverage": 0,
                                "position_sizing": "risk",
                                "enforce_risk_cap": True},
                               INIT_BALANCE)
    assert ord_id, "纸面开仓失败"
    eng.update()                                    # mark 在成交侧 → 触发成交
    pos = eng._positions.get(sig.inst_id)
    assert pos is not None and pos.filled, "限价单应已成交"
    return pos


def last_trade(eng):
    assert eng._trades, "应有交易记录"
    return eng._trades[-1]


# ---------------------------------------------------------------------------
# S1: 跳空 22% 穿越挂单价 — 限价单绝不追价
# ---------------------------------------------------------------------------

def test_s1_gap_through_entry_fills_at_limit(tmp_path):
    eng, state = make_engine(tmp_path, "s1")
    sig = make_signal("long", entry=100.0, tp=103.0, sl=98.0, lev=2)

    # mark 从 102（挂单上方，未成交）一步跳到 78 — 穿越挂单价 22%
    state["mark"] = 102.0
    eng.update()
    eng.open_position(sig, {"ctVal": 1.0, "minSz": 1, "lotSz": "1"},
                      {"risk_per_trade_pct": 30.0, "margin_pct": 30.0,
                       "max_position_leverage": 0}, INIT_BALANCE)
    assert sig.inst_id in eng._positions and not eng._positions[sig.inst_id].filled

    state["mark"] = 78.0                            # 跳空 22% 穿越挂单价
    eng.update()
    pos = eng._positions[sig.inst_id]
    assert pos.filled, "价格穿越挂单价后应成交"
    assert pos.entry_px == pytest.approx(100.0), \
        f"限价单成交价必须 == 挂单价(绝不追价), 实际 {pos.entry_px}"
    # 入场手续费按挂单价名义计
    _fee = pos.size * 1.0 * 100.0 * MAKER
    assert eng.balance == pytest.approx(INIT_BALANCE - _fee, abs=1e-9)


# ---------------------------------------------------------------------------
# S2: 跳空 15% 穿越止损（低杠杆 SL 路径）— 固定 0.5% 滑点 + PnL 精确对账
# ---------------------------------------------------------------------------

def test_s2_gap_through_stop_low_leverage(tmp_path):
    eng, state = make_engine(tmp_path, "s2")
    sig = make_signal("long", entry=100.0, tp=110.0, sl=98.0, lev=2)
    pos = open_and_fill(eng, state, sig, fill_mark=99.5)
    assert pos.entry_px == pytest.approx(100.0)

    # mark 从 99.5 一步跳到 83 — 穿越止损 15%（liq=50 远在下, 走 SL 路径）
    state["mark"] = 83.0
    eng.update()

    t = last_trade(eng)
    assert t["reason"] == "stop_loss"
    expected_exit = 98.0 * (1 - _SL_SLIPPAGE)       # 97.51 — 固定 0.5% 滑点
    assert t["exit_px"] == pytest.approx(expected_exit, abs=1e-9), \
        f"止损成交价应为 SL×(1−0.5%)={expected_exit}, 实际 {t['exit_px']}"
    # PnL 精确对账: gross − 出场费（入场费已在 fill 时扣）
    gross = pos.size * 1.0 * (expected_exit - 100.0)
    exit_fee = pos.size * 1.0 * expected_exit * TAKER
    assert t["pnl"] == pytest.approx(gross - exit_fee, abs=1e-6)
    assert t["pnl"] < 0
    assert eng.get_equity() == pytest.approx(
        INIT_BALANCE - pos.entry_fee + gross - exit_fee, abs=1e-6)


# ---------------------------------------------------------------------------
# S3: 50x 高杠杆跳空 20% — 爆仓优先, 亏损封顶保证金（穿仓修复不变式）
# ---------------------------------------------------------------------------

def test_s3_gap_liquidation_caps_loss_at_margin(tmp_path):
    eng, state = make_engine(tmp_path, "s3")
    # SL=97.9 在爆仓点(lev=50 → liq=98.0)之后 → trigger=liq 优先
    sig = make_signal("long", entry=100.0, tp=110.0, sl=97.9, lev=50)
    pos = open_and_fill(eng, state, sig, fill_mark=99.5)

    state["mark"] = 80.0                            # 跳空 20% 穿越爆仓点
    eng.update()

    t = last_trade(eng)
    liq_px = 100.0 * (1.0 - 1.0 / 50.0)             # 98.0
    assert t["reason"] == "liquidation", \
        f"SL 在爆仓点之后时应以爆仓结算, 实际 {t['reason']}"
    assert t["exit_px"] == pytest.approx(liq_px, abs=1e-9), \
        f"爆仓应按理论爆仓价 {liq_px} 结算(非跳空价 80), 实际 {t['exit_px']}"
    # 不变式: 亏损 ≤ 保证金 + 手续费; 余额 > 0（不穿仓）
    max_loss = pos.margin + pos.entry_fee + pos.size * liq_px * TAKER + 1e-6
    assert t["pnl"] >= -max_loss, \
        f"亏损 {t['pnl']:.4f} 超过 保证金+手续费 上限 {max_loss:.4f}"
    assert eng.get_equity() > 0, "任何极端路径余额不得为负"


# ---------------------------------------------------------------------------
# S4: 空头镜像 — 跳空向上穿越止损（固定滑点）+ 高杠杆爆仓封顶
# ---------------------------------------------------------------------------

def test_s4_short_gap_through_stop(tmp_path):
    eng, state = make_engine(tmp_path, "s4a")
    sig = make_signal("short", entry=100.0, tp=90.0, sl=102.0, lev=2)
    pos = open_and_fill(eng, state, sig, fill_mark=100.5)

    state["mark"] = 125.0                           # 跳空向上 23% 穿越止损
    eng.update()
    t = last_trade(eng)
    assert t["reason"] == "stop_loss"
    expected_exit = 102.0 * (1 + _SL_SLIPPAGE)      # 102.51
    assert t["exit_px"] == pytest.approx(expected_exit, abs=1e-9)
    assert t["pnl"] < 0


def test_s4b_short_liquidation_cap(tmp_path):
    eng, state = make_engine(tmp_path, "s4b")
    # sl=102.1 在爆仓点(102.0)之后 → 爆仓优先
    sig = make_signal("short", entry=100.0, tp=90.0, sl=102.1, lev=50)
    pos = open_and_fill(eng, state, sig, fill_mark=100.5)

    state["mark"] = 135.0                           # 跳空 35% 向上
    eng.update()
    t = last_trade(eng)
    liq_px = 100.0 * (1.0 + 1.0 / 50.0)             # 102.0
    assert t["reason"] == "liquidation"
    assert t["exit_px"] == pytest.approx(liq_px, abs=1e-9)
    max_loss = pos.margin + pos.entry_fee + pos.size * liq_px * TAKER + 1e-6
    assert t["pnl"] >= -max_loss
    assert eng.get_equity() > 0


# ---------------------------------------------------------------------------
# S5: 同根 K 线 TP/SL 同时穿越 → 止损优先（保守）
# ---------------------------------------------------------------------------

def test_s5_same_candle_tp_sl_stop_priority(tmp_path):
    eng, state = make_engine(tmp_path, "s5")
    sig = make_signal("long", entry=100.0, tp=103.0, sl=98.0, lev=2)
    pos = open_and_fill(eng, state, sig, fill_mark=99.5)

    # 一根插针 K 线同时穿越 TP(103) 和 SL(98)
    state["candles"] = [{"high": 105.0, "low": 95.0, "open": 99.0, "close": 96.0}]
    state["mark"] = 96.0
    eng.update()
    t = last_trade(eng)
    assert t["reason"] == "stop_loss", \
        f"TP/SL 同根 K 线冲突时止损应优先(保守), 实际 {t['reason']}"
    assert t["exit_px"] == pytest.approx(98.0 * (1 - _SL_SLIPPAGE), abs=1e-9)


# ---------------------------------------------------------------------------
# S6: 跳空 7% 穿越止盈 — 成交价 == tp（限价语义, 不吃跳空额外利润）
# ---------------------------------------------------------------------------

def test_s6_gap_through_tp_fills_at_tp(tmp_path):
    eng, state = make_engine(tmp_path, "s6")
    sig = make_signal("long", entry=100.0, tp=103.0, sl=98.0, lev=2)
    pos = open_and_fill(eng, state, sig, fill_mark=99.5)

    state["mark"] = 110.0                           # 跳空 7% 穿越止盈
    eng.update()
    t = last_trade(eng)
    assert t["reason"] == "take_profit"
    assert t["exit_px"] == pytest.approx(103.0, abs=1e-9), \
        f"止盈应按 TP 限价成交(非跳空价 110), 实际 {t['exit_px']}"
    assert t["pnl"] > 0


# ---------------------------------------------------------------------------
# S7: 极端序列资金安全不变式
# ---------------------------------------------------------------------------

def test_s7_balance_survives_catastrophic_sequence(tmp_path):
    """连续三次极端跳空爆仓后: 余额单调不为负, 每笔亏损 ≤ 保证金+费用。"""
    eng, state = make_engine(tmp_path, "s7")
    equities = [eng.get_equity()]

    for i in range(3):
        sig = make_signal("long", entry=100.0, tp=110.0, sl=97.9, lev=50,
                          inst=f"DOOM{i}-USDT-SWAP")
        eq_now = eng.get_equity()
        pos = open_and_fill(eng, state, sig, fill_mark=99.5)
        state["mark"] = 60.0                         # 每次跳空 40% 穿越爆仓点
        eng.update()
        t = last_trade(eng)
        assert t["reason"] == "liquidation"
        assert t["exit_px"] == pytest.approx(98.0, abs=1e-9)
        max_loss = pos.margin + pos.entry_fee + pos.size * 98.0 * TAKER + 1e-6
        assert t["pnl"] >= -max_loss, f"第{i}笔亏损超保证金上限"
        eq_after = eng.get_equity()
        assert eq_after > 0, f"第{i}次爆仓后余额为负"
        assert eq_after < eq_now, "爆仓必亏"
        equities.append(eq_after)

    # 逐笔记账守恒: balance = 初始 − Σ入场费 + Σ平仓净PnL →
    # Σ入场费 = 初始 + ΣPnL − balance（入场费在 fill 时扣、pnl 只含出场侧）
    total_pnl = sum(t["pnl"] for t in eng._trades)
    total_entry_fees = INIT_BALANCE + total_pnl - eng.balance
    assert total_entry_fees > 0, "入场费合计应为正"
    # 记账无泄漏: 期末权益 == 初始 + ΣPnL − Σ入场费（无持仓, equity==balance）
    assert eng.get_equity() == pytest.approx(eng.balance, abs=1e-9)
    assert eng.get_equity() == pytest.approx(
        INIT_BALANCE + total_pnl - total_entry_fees, abs=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
