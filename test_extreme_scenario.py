# -*- coding: utf-8 -*-
"""极端行情模拟测试: 止损 + 强平封顶 + 回撤断路器 + 连亏暂停 触发逻辑验证。

纯本地运行, 不依赖网络/交易所。模拟 Agent 主循环行为:
  1. paper_engine.update()  推进行情(极端插针)
  2. consume_close_events() → EdgeAnalyzer.add_trade (2026-08-11 风控修复链路)
  3. AdaptiveParameterTuner.adapt()  检查回撤断路器/连亏暂停

边界场景覆盖(2026-08-13 扩展):
  场景1-4  基础风控: 做多止损/强平/回撤断路器/连亏暂停
  场景5    做空方向: 止损滑点 + 强平封顶(与做多对称)
  场景6    移动止损: 做多抬升 + 做空下移, 回落按新 SL 止损
  场景7    动态 ROI TP floor: ROI 不抢信号 TP, 只在 TP 前保底
  场景8    回补无望撤单: fill_assist 偏离>3% 撤单不追价
  场景9    绝对回撤精确边界: 严格 >15% 才触发
  场景10   连亏降杠杆: derate 不暂停(分级风控)
  场景11   保本交易: pnl=0 不中断也不计入连亏

运行:
  C:\\Users\\casey\\AppData\\Local\\Programs\\Python\\Python310\\python.exe test_extreme_scenario.py

铁律: 全部使用独立临时 state_file, 不触碰真实 paper_state.json。
"""
import logging
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_trading import PaperTradingEngine
from optimization import EdgeAnalyzer, EdgeStats, TradeRecord, AdaptiveParameterTuner

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

fd, state_path = tempfile.mkstemp(prefix="extreme_test_", suffix=".json")
os.close(fd)

_INFO = {"ctVal": "0.01", "minSz": "1", "lotSz": "1", "tickSz": "0.1"}


def make_cfg(balance=30.0):
    return {
        "strategy": {},
        "paper": {"enabled": True, "balance": balance, "limit_timeout_min": 30,
                  "maker_fee": 0.0002, "taker_fee": 0.0005,
                  "state_file": state_path, "fill_assist_seconds": 120,
                  "fill_assist_deviation_pct": 3.0},
        "risk": {"risk_per_trade_pct": 30.0, "margin_pct": 30.0,
                 "max_hold_hours": 48, "max_leverage": 50,
                 "dynamic_roi": {"0": 0.5}},
        "optimization": {"adaptive_enabled": True, "consecutive_loss_pause": 2,
                         "consecutive_loss_derate": 2, "loss_pause_hours": 24,
                         "edge_lookback": 100},
    }


class _Sig:
    def __init__(self, inst, side, entry, sl, tp, lev):
        self.inst_id = inst
        self.position_side = side
        self.entry_price = entry
        self.stop_loss = sl
        self.take_profit = tp
        self.leverage = lev
        self.signal_id = "sig_" + inst


class Sim:
    """模拟 Agent 主循环: 开仓 → 行情推进 → 平仓事件喂 edge_analyzer → 自适应调参。"""

    def __init__(self):
        self.cfg = make_cfg()
        self.eng = PaperTradingEngine(self.cfg)
        self.ea = EdgeAnalyzer()
        self.tuner = AdaptiveParameterTuner(self.cfg)

    def open(self, sig, equity):
        self.eng.open_position(sig, _INFO, self.cfg["risk"], equity)

    def tick(self, hi, lo, close, mark):
        """推进一轮行情(极端插针通过 hi/lo/mark 表达)。"""
        self.eng.set_market_data_provider(
            lambda i, _h=hi, _l=lo, _c=close, _m=mark:
                {"candles": [{"high": _h, "low": _l, "close": _c}], "mark": _m})
        self.eng.update()
        self._feed()

    def _feed(self):
        """模拟 agent._feed_edge_analyzer_from_paper: 平仓事件 → edge_analyzer → adapt。"""
        for t in self.eng.consume_close_events():
            pnl = float(t.get("pnl", 0) or 0)
            self.ea.add_trade(TradeRecord(
                symbol=t["inst_id"], direction=t["side"],
                entry_time=t["open_time"], exit_time=t["closed_at"],
                entry_price=t["entry_px"], exit_price=t["exit_px"],
                quantity=t["size"], leverage=int(t["leverage"] or 1),
                pnl=pnl, pnl_pct=float(t.get("pnl_pct", 0) or 0),
                is_win=pnl > 0, exit_reason=t.get("reason", "paper_exit"),
            ))
        if self.ea.trades:
            stats = self.ea.analyze(100)
            self.tuner.adapt(stats, current_equity=self.eng.get_equity())

    def trade(self):
        return self.eng._trades[-1] if self.eng._trades else None

    def pos(self):
        return list(self.eng._positions.values())


failures = []
passed = []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
        print(f"  [PASS] {name} {detail}")
    else:
        failures.append(name)
        print(f"  [FAIL] {name} {detail}")


def scenario_stop_loss():
    """场景1: 常规止损 + 0.5% 滑点 (10x 多头, 价格温和回落触发 SL)。"""
    print("\n===== 场景1: 止损触发(含 0.5% 滑点) =====")
    sim = Sim()
    sim.open(_Sig("A", "long", 100.0, 98.0, 105.0, 10), sim.eng.get_equity())
    sim.tick(99.5, 99.0, 99.2, 99.0)   # mark<=entry → 限价成交 @100
    assert sim.pos() and sim.pos()[0].filled, "开仓应成交"
    sim.tick(99.0, 96.0, 97.0, 96.5)   # 极端回落: low 96 穿透 SL 98
    t = sim.trade()
    check("SL 触发 reason=stop_loss", t and t["reason"] == "stop_loss",
          f"reason={t['reason'] if t else None}")
    check("SL 滑点成交价 = 98×0.995", t and abs(t["exit_px"] - 98.0 * 0.995) < 1e-9,
          f"exit={t['exit_px'] if t else None}")
    check("亏损未穿仓(balance>=0)", sim.eng.balance >= 0,
          f"balance={sim.eng.balance:.4f}")
    return sim


def scenario_liquidation():
    """场景2: 极端插针强平 — 高杠杆爆仓点先于 SL, 亏损封顶 -margin 不穿仓。"""
    print("\n===== 场景2: 极端插针 → 强平封顶 =====")
    sim = Sim()
    # 50x 多头: 爆仓点 = 100×(1-1/50) = 98, 与 SL 98 重合 → 爆仓点先到判定强平
    sim.open(_Sig("B", "long", 100.0, 98.0, 110.0, 50), sim.eng.get_equity())
    pos = sim.pos()[0]
    sim.tick(99.5, 99.0, 99.2, 99.0)   # 成交 @100
    sim.tick(99.0, 90.0, 92.0, 91.0)   # 插针暴跌: low 90 深穿爆仓点 98
    t = sim.trade()
    check("强平 reason=liquidation", t and t["reason"] == "liquidation",
          f"reason={t['reason'] if t else None}")
    # 强平结算价=爆仓点(50x→98), gross=-margin; 封顶后亏损 = -(margin+退出费)
    check("强平亏损 ≈ -margin(封顶,不穿仓)",
          t and t["pnl"] >= -(pos.margin + 0.5) and t["pnl"] < 0,
          f"pnl={t['pnl'] if t else None} margin={pos.margin:.2f}")
    check("强平后 balance>=0 不穿仓", sim.eng.balance >= 0,
          f"balance={sim.eng.balance:.4f}")
    return sim


def scenario_drawdown_breaker():
    """场景3: 连续极端亏损 → 绝对回撤 >15% → 回撤断路器触发(暂停24h)。"""
    print("\n===== 场景3: 回撤断路器(峰值回撤>15% → 暂停24h) =====")
    sim = Sim()
    sim.tuner.peak_equity = sim.eng.get_equity()  # 峰值 = 30
    print(f"  [info] 峰值权益 = {sim.tuner.peak_equity:.2f}")

    # 连续 2 笔极端亏损(每笔封顶 -margin≈9)
    for i, inst in enumerate(["C1", "C2"]):
        eq = sim.eng.get_equity()
        sim.open(_Sig(inst, "long", 100.0, 98.0, 110.0, 50), eq)
        sim.tick(99.5, 99.0, 99.2, 99.0)
        sim.tick(99.0, 90.0, 92.0, 91.0)  # 插针强平
        t = sim.trade()
        print(f"  [info] {inst} 强平 pnl={t['pnl']:+.2f} | 权益={sim.eng.get_equity():.2f}")

    dd = (sim.tuner.peak_equity - sim.eng.get_equity()) / sim.tuner.peak_equity * 100
    check("峰值回撤 > 15%", dd > 15.0, f"回撤={dd:.1f}%")
    check("回撤断路器 trading_paused=True", sim.tuner.trading_paused is True,
          f"trading_paused={sim.tuner.trading_paused}")
    check("pause_until 已设置(24h)", sim.tuner.pause_until > time.time(),
          f"pause_until={sim.tuner.pause_until}")
    return sim


def scenario_consecutive_loss_pause():
    """场景4: 连亏 2 笔 → 连亏暂停(在未触发回撤断路器时独立生效)。"""
    print("\n===== 场景4: 连亏暂停(consecutive_loss_pause=2) =====")
    # 用小额亏损避免回撤断路器抢先(回撤阈值内)
    sim = Sim()
    sim.tuner.peak_equity = sim.eng.get_equity()
    for i, inst in enumerate(["D1", "D2"]):
        eq = sim.eng.get_equity()
        sim.open(_Sig(inst, "long", 100.0, 99.0, 102.0, 3), eq)  # 3x, 止损1%→小亏
        sim.tick(99.5, 99.4, 99.5, 99.4)   # 成交 @100
        sim.tick(99.0, 98.5, 98.7, 98.6)   # 跌破 SL 99 → 止损
        t = sim.trade()
        print(f"  [info] {inst} 止损 pnl={t['pnl']:+.4f} | 权益={sim.eng.get_equity():.4f}")
    dd = (sim.tuner.peak_equity - sim.eng.get_equity()) / sim.tuner.peak_equity * 100
    stats = sim.ea.analyze(100)
    check("连亏计数 = 2", stats.consecutive_losses == 2,
          f"consecutive_losses={stats.consecutive_losses}")
    check("回撤未超 15%(纯连亏场景)", dd < 15.0, f"回撤={dd:.2f}%")
    check("连亏暂停 trading_paused=True", sim.tuner.trading_paused is True)
    return sim


def scenario_short_stop_and_liquidation():
    """场景5: 做空方向 — 止损滑点(向上) + 强平封顶(与做多对称)。"""
    print("\n===== 场景5: 做空方向 止损 + 强平 =====")
    # 5a 做空止损: entry=100, sl=101, tp=95, lev=10 → 爆仓点110, SL 先到
    sim = Sim()
    sim.open(_Sig("E1", "short", 100.0, 101.0, 95.0, 10), sim.eng.get_equity())
    sim.tick(99.5, 100.5, 100.2, 100.5)   # 做空 hi>=entry → 成交 @100
    assert sim.pos() and sim.pos()[0].filled, "做空应成交"
    sim.tick(101.5, 100.0, 101.0, 101.5)  # 上穿 SL 101 → 止损(滑点向上)
    t = sim.trade()
    check("做空止损 reason=stop_loss", t and t["reason"] == "stop_loss",
          f"reason={t['reason'] if t else None}")
    check("做空止损滑点 = 101×1.005", t and abs(t["exit_px"] - 101.0 * 1.005) < 1e-9,
          f"exit={t['exit_px'] if t else None}")
    check("做空止损未穿仓", sim.eng.balance >= 0, f"balance={sim.eng.balance:.4f}")

    # 5b 做空强平: entry=100, sl=102, lev=50 → 爆仓点102=SL → 强平
    sim2 = Sim()
    sim2.open(_Sig("E2", "short", 100.0, 102.0, 90.0, 50), sim2.eng.get_equity())
    pos = sim2.pos()[0]
    sim2.tick(99.5, 100.5, 100.2, 100.5)   # 成交 @100
    sim2.tick(103.0, 101.0, 102.0, 103.0)  # 深插针上穿 → 强平
    t2 = sim2.trade()
    check("做空强平 reason=liquidation", t2 and t2["reason"] == "liquidation",
          f"reason={t2['reason'] if t2 else None}")
    check("做空强平亏损≈-margin(封顶不穿仓)",
          t2 and t2["pnl"] >= -(pos.margin + 0.5) and t2["pnl"] < 0,
          f"pnl={t2['pnl'] if t2 else None} margin={pos.margin:.2f}")
    check("做空强平 balance>=0", sim2.eng.balance >= 0,
          f"balance={sim2.eng.balance:.4f}")
    return sim2


def scenario_trailing_stop():
    """场景6: 移动止损 — 做多只上移 / 做空只下移, 回落按新 SL 止损。"""
    print("\n===== 场景6: 移动止损(ATR 回退路径) =====")
    # 6a 做多: entry=100, sl=98, tp=110, lev=5 (无ATR → fallback 激活)
    sim = Sim()
    sim.open(_Sig("F1", "long", 100.0, 98.0, 110.0, 5), sim.eng.get_equity())
    sim.tick(99.5, 99.0, 99.2, 99.0)   # 成交 @100
    sim.tick(105.5, 104.0, 105.0, 105.0)  # 涨至105 激活 trailing → SL 98→102
    check("做多 trailing 抬升 SL 98→102", abs(sim.pos()[0].sl_px - 102.0) < 1e-9,
          f"sl={sim.pos()[0].sl_px}")
    sim.tick(107.5, 106.0, 107.0, 107.0)  # 续涨 → SL 102→104
    check("做多 trailing 续抬 SL→104", abs(sim.pos()[0].sl_px - 104.0) < 1e-9,
          f"sl={sim.pos()[0].sl_px}")
    sim.tick(105.0, 103.0, 104.0, 103.5)  # 回落跌破新 SL 104 → 止损
    t = sim.trade()
    check("做多按新 SL 止损 = 104×0.995", t and t["reason"] == "stop_loss"
          and abs(t["exit_px"] - 104.0 * 0.995) < 1e-9,
          f"exit={t['exit_px'] if t else None}")

    # 6b 做空: entry=100, sl=102, tp=90, lev=5 (无ATR → fallback 激活)
    sim2 = Sim()
    sim2.open(_Sig("F2", "short", 100.0, 102.0, 90.0, 5), sim2.eng.get_equity())
    sim2.tick(99.5, 100.5, 100.2, 100.5)  # 成交 @100
    sim2.tick(94.5, 93.5, 94.0, 94.0)     # 跌至94 激活 trailing → SL 102→97
    check("做空 trailing 下移 SL 102→97", abs(sim2.pos()[0].sl_px - 97.0) < 1e-9,
          f"sl={sim2.pos()[0].sl_px}")
    sim2.tick(92.5, 91.5, 92.0, 92.0)     # 续跌 → SL 97→95
    check("做空 trailing 续移 SL→95", abs(sim2.pos()[0].sl_px - 95.0) < 1e-9,
          f"sl={sim2.pos()[0].sl_px}")
    sim2.tick(96.5, 94.5, 95.5, 95.5)     # 反弹上穿新 SL 95 → 止损
    t2 = sim2.trade()
    check("做空按新 SL 止损 = 95×1.005", t2 and t2["reason"] == "stop_loss"
          and abs(t2["exit_px"] - 95.0 * 1.005) < 1e-9,
          f"exit={t2['exit_px'] if t2 else None}")
    return sim2


def scenario_roi_tp_floor():
    """场景7: 动态 ROI TP floor — 配置 ROI(0.5) 不抢信号 TP, 价格先到 TP 才触发。"""
    print("\n===== 场景7: 动态 ROI TP floor 不抢 TP =====")
    # entry=100, tp=110, lev=10 → tp_margin_roi=0.10×10=1.0, floor 0.85 → roi_target=0.85
    # 配置 ROI {"0":0.5}。价格 +5%(upl=50%) 时若无 floor 会截胡, 有 floor(0.85) 不触发。
    sim = Sim()
    sim.open(_Sig("G1", "long", 100.0, 98.0, 110.0, 10), sim.eng.get_equity())
    sim.tick(99.5, 99.0, 99.2, 99.0)   # 成交 @100
    sim.tick(105.5, 104.5, 105.0, 105.0)  # +5%, upl=50% < roi_target 85% → 不触发 ROI
    check("+5% 时 ROI 被 TP floor 拦住(不截胡)",
          sim.pos() and sim.pos()[0].filled and sim.pos()[0].inst_id == "G1",
          f"positions={[p.inst_id for p in sim.pos()]}")
    sim.tick(110.5, 109.5, 110.0, 110.0)  # 涨至 TP → 触发 TP(优先于 ROI)
    t = sim.trade()
    check("TP 优先触发 reason=take_profit", t and t["reason"] == "take_profit",
          f"reason={t['reason'] if t else None}")
    return sim


def scenario_fill_assist_cancel():
    """场景8: 回补无望撤单 — 挂单超观察期且偏离>3% → 撤单不追价。"""
    print("\n===== 场景8: 回补无望撤单(fill_assist 偏离>3%) =====")
    sim = Sim()
    sim.open(_Sig("H1", "long", 100.0, 98.0, 105.0, 5), sim.eng.get_equity())
    # 回拨挂单时间 130s 前(观察期 120s), 再推 mark 偏离 4% > 3%
    sim.eng._positions["H1"].open_time = time.time() - 130.0
    sim.tick(104.5, 103.5, 104.0, 104.0)   # mark=104 偏离 4%, 做多不成交 → 撤单
    check("回补无望撤单(不追价)", "H1" not in sim.eng._positions,
          f"positions={list(sim.eng._positions.keys())}")
    check("撤单不产生成交记录", sim.trade() is None)
    return sim


def scenario_drawdown_exact_boundary():
    """场景9: 绝对回撤精确边界 — 严格 >15% 才触发(15.0% 不触发)。"""
    print("\n===== 场景9: 绝对回撤精确边界(严格 >15%) =====")
    tuner = AdaptiveParameterTuner(make_cfg())
    tuner.peak_equity = 100.0
    tuner.adapt(EdgeStats(), current_equity=85.0)   # 回撤恰好 15.0%
    check("回撤恰好 15.0% 不触发(严格>)", tuner.trading_paused is False,
          f"trading_paused={tuner.trading_paused}")

    tuner2 = AdaptiveParameterTuner(make_cfg())
    tuner2.peak_equity = 100.0
    tuner2.adapt(EdgeStats(), current_equity=84.99)  # 回撤 15.01%
    check("回撤 15.01% 触发断路器", tuner2.trading_paused is True,
          f"trading_paused={tuner2.trading_paused}")
    return tuner2


def scenario_consecutive_derate():
    """场景10: 连亏降杠杆 — derate(3) < pause(5) 时降杠杆但不暂停。"""
    print("\n===== 场景10: 连亏降杠杆(derate 不暂停) =====")
    cfg = make_cfg()
    cfg["optimization"]["consecutive_loss_pause"] = 5
    cfg["optimization"]["consecutive_loss_derate"] = 3
    tuner = AdaptiveParameterTuner(cfg)
    tuner.peak_equity = 100.0
    stats = EdgeStats()
    stats.consecutive_losses = 4
    tuner.adapt(stats, current_equity=95.0)
    check("连亏4笔不暂停(pause=5)", tuner.trading_paused is False,
          f"trading_paused={tuner.trading_paused}")
    check("降杠杆生效(current<base)",
          tuner.current_leverage < tuner.base_leverage,
          f"lev={tuner.current_leverage} base={tuner.base_leverage}")
    check("风险比例同步降低",
          tuner.current_risk_pct < tuner.base_risk_pct,
          f"risk={tuner.current_risk_pct} base={tuner.base_risk_pct}")
    return tuner


def scenario_breakeven_not_break_streak():
    """场景11: 保本交易(pnl=0) 不中断也不计入连亏。"""
    print("\n===== 场景11: 保本交易不中断连亏 =====")
    ea = EdgeAnalyzer()
    def _rec(pnl, i):
        return TradeRecord(
            symbol=f"X{i}", direction="long", entry_time=1000.0 + i,
            exit_time=2000.0 + i, entry_price=100.0, exit_price=100.0,
            quantity=1.0, leverage=1, pnl=pnl, pnl_pct=0.0,
            is_win=pnl > 0, exit_reason="sl",
        )
    ea.add_trade(_rec(-1.0, 0))
    ea.add_trade(_rec(0.0, 1))   # 保本
    ea.add_trade(_rec(-1.0, 2))
    stats = ea.analyze(100)
    check("保本不中断连亏: consecutive_losses=2", stats.consecutive_losses == 2,
          f"consecutive_losses={stats.consecutive_losses}")
    check("保本单独统计 break_even=1", stats.break_even_trades == 1,
          f"break_even={stats.break_even_trades}")
    return ea


if __name__ == "__main__":
    s1 = scenario_stop_loss()
    s2 = scenario_liquidation()
    s3 = scenario_drawdown_breaker()
    s4 = scenario_consecutive_loss_pause()
    s5 = scenario_short_stop_and_liquidation()
    s6 = scenario_trailing_stop()
    s7 = scenario_roi_tp_floor()
    s8 = scenario_fill_assist_cancel()
    s9 = scenario_drawdown_exact_boundary()
    s10 = scenario_consecutive_derate()
    s11 = scenario_breakeven_not_break_streak()

    print("\n" + "=" * 56)
    print(f"结果: PASS {len(passed)} | FAIL {len(failures)}")
    if failures:
        print("失败项:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL PASS")
    os.unlink(state_path)
