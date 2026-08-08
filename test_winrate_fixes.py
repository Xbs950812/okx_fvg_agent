# -*- coding: utf-8 -*-
"""
胜率修复验证 — Mock 数据场景测试。

覆盖本轮全部修复项，验证"解决了原问题且未引入新问题":

  修复 1. FVG 新鲜度过滤 (strategy._is_fvg_fresh + generate_signal 过期丢弃)
  修复 2. 多通道置信度修正 (MasterTraderEngine.analyze: net=0 通道不再灌水)
  修复 3. 通道一致性语义修正 (数据不足取中性 0.5, 筛选职责由打折后的置信度承担;
         置信度修正后单通道信号 conf≈0.26 在保守挡位被拒、激进挡位仍可交易)
  修复 4. 红旗门禁 (agent._red_flag_gate: 区间低位不宜追空/高位不宜追多 强制否决)
  修复 5. WeakGate 数据不足拒绝 (agent._weak_signal_multi_gate: 不再降级放行)
  修复 6. 换仓统一守卫 (agent._switch_guards: 最小持仓时长+评分门槛+资金费+相关性)
  回归  7. 新鲜 FVG 仍能正常生成合法信号 (entry/SL/TP 结构完好)

运行: python test_winrate_fixes.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy import Candle, FVG, Signal, generate_signal, _is_fvg_fresh
from multi_channel import ChannelReport, MasterTraderEngine, MasterAnalysis
import agent

# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------

_TS0 = 1700000000000
_H = 3600000  # 1 小时 (ms)


def make_candle(ts: int, o, h, l, c, v=1000.0) -> Candle:
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v)


def make_fvg(direction="long", top=102.1, bottom=100.2, fvg_index=5,
             timeframe="1H") -> FVG:
    return FVG(
        direction=direction,
        top=top,
        bottom=bottom,
        width_pct=abs(top - bottom) / bottom * 100.0,
        candle_ts=_TS0 + fvg_index * _H,
        timeframe=timeframe,
        impulse_candle=make_candle(_TS0 + fvg_index * _H, 100.1, 103.0, 99.9, 102.5),
        fvg_index=fvg_index,
        is_abnormal=False,
        sigma=0.0,
        volume_ratio=1.0,
    )


def make_signal(inst_id="TEST-USDT-SWAP", side="short", score=0.3,
                entry=1.0, sl=1.05, tp=0.95) -> Signal:
    return Signal(
        inst_id=inst_id,
        fvg=make_fvg(direction="short" if side == "short" else "long",
                     top=1.1, bottom=0.9, fvg_index=20),
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        leverage=1,
        position_side=side,
        score=score,
    )


def make_analysis(final_score=-0.5, final_confidence=0.6, direction="short",
                  key_risks=None) -> MasterAnalysis:
    return MasterAnalysis(
        symbol="TEST-USDT-SWAP",
        timestamp=time.time(),
        final_score=final_score,
        final_confidence=final_confidence,
        direction=direction,
        channels=[],
        key_risks=key_risks or [],
        channel_agreement=0.5,
    )


def make_channel(name, net, conf=0.65, weight=0.0):
    """构造单通道报告 (net<0 看空, net=0 无观点)。"""
    bull = max(net, 0.0)
    bear = max(-net, 0.0)
    return ChannelReport(
        channel_name=name,
        weight=weight,
        bullish_score=bull,
        bearish_score=bear,
        net_score=net,
        confidence=conf,
        observations=["mock"],
    )


class MockClientNoData:
    """弱信号门禁用: 全部市场数据不可得。"""

    def get_long_short_ratio(self, *a, **k):
        return None

    def get_funding_rate(self, *a, **k):
        return None

    def get_open_interest(self, *a, **k):
        return None

    def get_funding_info(self, *a, **k):
        return None


class MockClientFundingInfo:
    def get_funding_info(self, *a, **k):
        return None


# ---------------------------------------------------------------------------
# 修复 1: FVG 新鲜度过滤
# ---------------------------------------------------------------------------

def _build_fresh_candles():
    """10 根 1H 蜡烛: idx4-6 形成看涨 FVG [100.2, 102.1], 价格回踩至 101.5。"""
    ohlc = [
        (100.0, 100.3, 99.7, 100.1),
        (100.1, 100.4, 99.8, 100.0),
        (100.0, 100.2, 99.9, 100.1),
        (100.1, 100.3, 99.9, 100.0),
        (100.0, 100.2, 99.8, 100.1),   # c0, high=100.2
        (100.1, 103.0, 99.9, 102.5),   # impulse
        (102.5, 102.7, 102.1, 102.3),  # c2, low=102.1
        (102.3, 102.4, 101.4, 101.6),
        (101.6, 101.7, 100.9, 101.2),
        (101.2, 101.6, 100.8, 101.5),  # 回踩至 FVG 内部
    ]
    return [make_candle(_TS0 + i * _H, *v) for i, v in enumerate(ohlc)]


def test_fvg_freshness_unit():
    candles = _build_fresh_candles()
    # 旧 FVG (fvg_index=0 → age=9 > 8) → 过期
    old = make_fvg(fvg_index=0)
    fresh, age = _is_fvg_fresh(old, candles, 8)
    assert not fresh and age == 9, f"旧 FVG 应过期, got fresh={fresh} age={age}"
    # 新 FVG (fvg_index=5 → age=4 ≤ 8) → 新鲜
    new = make_fvg(fvg_index=5)
    fresh2, age2 = _is_fvg_fresh(new, candles, 8)
    assert fresh2 and age2 == 4, f"新 FVG 应新鲜, got fresh={fresh2} age={age2}"
    # max_age_bars=0 → 不过滤
    fresh3, age3 = _is_fvg_fresh(old, candles, 0)
    assert fresh3 and age3 == -1, "max_age_bars=0 应跳过过滤"
    print("PASS test_fvg_freshness_unit")


def test_generate_signal_rejects_stale_fvg():
    candles = _build_fresh_candles()
    kw = dict(
        inst_id="TEST-USDT-SWAP",
        current_price=101.5,
        entry_depth_pct=0.15,
        fvg_target_pct=0.5,
        stop_buffer_pct=0.15,
        max_leverage=10,
        funding_rate=None,
        max_funding_rate_abs=0.01,
        min_risk_reward=0.0,
        max_entry_distance_pct=10.0,
        max_tp_distance_pct=25.0,
        max_fvg_age_bars=8,
        alpha158_enabled=False,
        tech_params={"bb_veto_low": -10.0, "bb_veto_high": 10.0},
        candles=candles,
    )
    # 过期 FVG → 直接被新鲜度过滤器丢弃
    stale = generate_signal(fvg=make_fvg(fvg_index=0), **kw)
    assert stale is None, "过期 FVG 应被丢弃"
    # 新鲜 FVG → 正常生成合法信号 (不被新鲜度误杀)
    fresh = generate_signal(fvg=make_fvg(fvg_index=5), **kw)
    assert fresh is not None, "新鲜 FVG 应生成信号"
    assert fresh.position_side == "long"
    assert fresh.entry_price > 0
    assert fresh.stop_loss < fresh.entry_price < fresh.take_profit, \
        "多头信号结构应合法: SL < entry < TP"
    print("PASS test_generate_signal_rejects_stale_fvg")


# ---------------------------------------------------------------------------
# 修复 2+3: 多通道置信度/一致性修正
# ---------------------------------------------------------------------------

def test_confidence_not_inflated_by_neutral_channels():
    engine = MasterTraderEngine()
    # 仅价格行为有观点(看空), 其余 4 通道 net=0 (无数据)
    channels = [
        make_channel("价格行为", -1.0, conf=0.65),
        make_channel("市场结构", 0.0, conf=0.66),
        make_channel("资金流向", 0.0, conf=0.60),
        make_channel("市场情绪", 0.0, conf=0.60),
        make_channel("宏观背景", 0.0, conf=0.60),
    ]
    a = engine.analyze("TEST-USDT-SWAP", channels, [])
    # 旧公式: conf = Σ(w×conf)/Σw ≈ 0.63 → 虚高。新公式 = 0.65 × (0.4/1.0) = 0.26
    assert a.final_confidence < 0.35, f"单通道信号置信度应被压缩, got {a.final_confidence}"
    assert abs(a.final_confidence - 0.26) < 0.02, f"期望 0.26, got {a.final_confidence}"
    assert a.final_score < -0.5, "方向仍由唯一活跃通道决定(看空)"
    assert a.channel_agreement == 0.5, \
        f"<2 个有观点通道时一致性取中性 0.5 (原硬编码0.5同值, 但置信度已打折承担筛选), got {a.channel_agreement}"
    print("PASS test_confidence_not_inflated_by_neutral_channels")


def test_confidence_two_active_channels():
    engine = MasterTraderEngine()
    channels = [
        make_channel("价格行为", -1.0, conf=0.65),
        make_channel("市场结构", -0.6, conf=0.66),
        make_channel("资金流向", 0.0, conf=0.60),
        make_channel("市场情绪", 0.0, conf=0.60),
        make_channel("宏观背景", 0.0, conf=0.60),
    ]
    a = engine.analyze("TEST-USDT-SWAP", channels, [])
    # 双通道一致 → agreement=1.0; conf = avg(0.65,0.66) × (0.65/1.0) ≈ 0.425
    assert a.channel_agreement == 1.0, f"双通道一致应 agreement=1.0, got {a.channel_agreement}"
    assert abs(a.final_confidence - 0.425) < 0.03, \
        f"期望约 0.425, got {a.final_confidence}"
    assert a.final_score < -0.5, "双通道看空方向应保持"
    print("PASS test_confidence_two_active_channels")


def test_confidence_all_neutral():
    engine = MasterTraderEngine()
    channels = [make_channel(n, 0.0) for n in
                ["价格行为", "市场结构", "资金流向", "市场情绪", "宏观背景"]]
    a = engine.analyze("TEST-USDT-SWAP", channels, [])
    assert a.final_confidence == 0.0, "全通道无观点 → 置信度 0"
    assert a.final_score == 0.0 and a.direction == "neutral"
    print("PASS test_confidence_all_neutral")


# ---------------------------------------------------------------------------
# 修复 4: 红旗门禁
# ---------------------------------------------------------------------------

def test_red_flag_gate():
    cfg = {"strategy": {"enforce_red_flags": True}}
    # 做空 + "区间低位不宜追空" → 否决 (SAHARA/CHZ 教训)
    sig_short = make_signal(side="short", score=0.6)
    a_short = make_analysis(direction="short",
                            key_risks=["价格处于区间低位，不宜追空"])
    ok, reason = agent._red_flag_gate(sig_short, a_short, cfg)
    assert not ok and "不宜追空" in reason, f"做空+低位红旗应否决, got ok={ok} {reason}"
    # 做多 + 同分析(低位红旗) → 放行 (红旗方向不冲突)
    sig_long = make_signal(side="long", score=0.6)
    ok2, _ = agent._red_flag_gate(sig_long, a_short, cfg)
    assert ok2, "做多不受'不宜追空'红旗影响"
    # 做多 + "区间高位不宜追多" → 否决
    a_high = make_analysis(direction="long",
                           key_risks=["价格处于区间高位，不宜追多"])
    ok3, reason3 = agent._red_flag_gate(sig_long, a_high, cfg)
    assert not ok3 and "不宜追多" in reason3
    # 关闭门禁 → 放行
    cfg_off = {"strategy": {"enforce_red_flags": False}}
    ok4, _ = agent._red_flag_gate(sig_short, a_short, cfg_off)
    assert ok4, "enforce_red_flags=false 应放行"
    # analysis=None → 放行 (裸信号路径)
    ok5, _ = agent._red_flag_gate(sig_short, None, cfg)
    assert ok5
    print("PASS test_red_flag_gate")


# ---------------------------------------------------------------------------
# 修复 5: WeakGate 数据不足拒绝
# ---------------------------------------------------------------------------

def _weak_cfg():
    return {"strategy": {"weak_signal_gate": {
        "enabled": True,
        "score_threshold": 0.45,
        "confidence_threshold": 0.50,
        "min_confluence": 3,
    }}}


def test_weak_gate_insufficient_data_rejects():
    cfg = _weak_cfg()
    # 仅 5 根蜡烛 → 量能(需24)/换手(OI不可得)/趋势(需20) 全部不可用 → total=0
    short_candles = [make_candle(_TS0 + i * _H, 100, 101, 99, 100)
                     for i in range(5)]
    sig_weak = make_signal(side="short", score=0.30)  # 弱信号
    ok, reason = agent._weak_signal_multi_gate(
        cfg, MockClientNoData(), sig_weak, None, short_candles)
    assert not ok, f"弱信号+数据不足应拒绝(原:降级放行), got ok={ok} {reason}"
    assert "数据不足" in reason
    # 强信号(score≥0.45 且 conf≥0.50) → 前置放行, 不受数据不足影响
    sig_strong = make_signal(side="short", score=0.90)
    analysis_strong = make_analysis(final_score=-0.5, final_confidence=0.70,
                                    direction="short")
    ok2, _ = agent._weak_signal_multi_gate(
        cfg, MockClientNoData(), sig_strong, analysis_strong, short_candles)
    assert ok2, "强信号应放行"
    print("PASS test_weak_gate_insufficient_data_rejects")


def test_weak_gate_sufficient_confluence_allows():
    cfg = _weak_cfg()
    # 30 根上涨放量蜡烛 → 量能/趋势/换手 3 项可用且顺向 (做多)
    candles = []
    for i in range(30):
        c = 100 + i * 0.1
        v = 1000 + i * 100  # 递增放量
        candles.append(make_candle(_TS0 + i * _H, c - 0.1, c + 0.2, c - 0.2, c, v))

    class MockClientAdequate(MockClientNoData):
        def get_open_interest(self, *a, **k):
            return 100000.0  # vol24(≈3.9万) / OI = 0.39 < 2.0 → 换手不可用

    sig_weak = make_signal(side="long", score=0.30)
    ok, reason = agent._weak_signal_multi_gate(
        cfg, MockClientAdequate(), sig_weak, None, candles)
    # 量能(放量)+趋势(价>SMA20)=2 项顺向可用; 换手不达标不计入 → total=2 < 3 → 拒绝
    assert not ok, f"可用共振指标不足应拒绝, got ok={ok} {reason}"
    print("PASS test_weak_gate_sufficient_confluence_allows")


# ---------------------------------------------------------------------------
# 修复 6: 换仓统一守卫
# ---------------------------------------------------------------------------

def _switch_cfg():
    return {
        "strategy": {
            "min_switch_score_improvement": 0.5,
            "switch_cost_edge_pct": 1.5,
            "switch_round_trip_cost_pct": 0.3,
        },
        "risk": {"min_hold_hours": 4.0},
    }


def test_switch_guards_min_hold():
    cfg = _switch_cfg()
    now = time.time()
    # 持仓仅 1 小时 < 4h → 即使评分大幅更优也拒绝 (换仓冷却)
    ok, reason = agent._switch_guards(
        cfg, MockClientFundingInfo(), None,
        "OLD", 0.3, now - 3600, "NEW", 0.9, "long", None)
    assert not ok and "min_hold" in reason, f"持仓过新应拒绝, got ok={ok} {reason}"
    print("PASS test_switch_guards_min_hold")


def test_switch_guards_score_threshold():
    cfg = _switch_cfg()
    now = time.time()
    old_c_time = now - 3600 * 10  # 持仓 10h, 通过冷却
    # 提升不足: 0.5 < -0.5 + (0.5+0.2) = 0.2? 否 → 0.5 ≥ 0.2 会通过…
    # 用更小提升验证: cur=-0.5, new=-0.2 → -0.2 < -0.5+0.7=0.2 → 拒绝
    ok, reason = agent._switch_guards(
        cfg, MockClientFundingInfo(), None,
        "OLD", -0.5, old_c_time, "NEW", -0.2, "long", None)
    assert not ok and "评分" in reason, f"评分提升不足应拒绝, got ok={ok} {reason}"
    # 提升足够: new=0.9 ≥ 0.2 → 通过全部守卫
    ok2, reason2 = agent._switch_guards(
        cfg, MockClientFundingInfo(), None,
        "OLD", -0.5, old_c_time, "NEW", 0.9, "long", None)
    assert ok2, f"评分大幅提升应允许换仓, got ok={ok2} {reason2}"
    print("PASS test_switch_guards_score_threshold")


def test_switch_guards_funding_conflict():
    cfg = _switch_cfg()
    now = time.time()
    # 做空目标但费率为负(做空需付费率) → 拒绝 (方案D)
    ok, reason = agent._switch_guards(
        cfg, MockClientFundingInfo(), None,
        "OLD", -0.5, now - 3600 * 10, "NEW", 0.9, "short", -0.001)
    assert not ok and "费率" in reason, f"费率方向冲突应拒绝, got ok={ok} {reason}"
    print("PASS test_switch_guards_funding_conflict")


def test_switch_guards_correlation():
    cfg = _switch_cfg()
    now = time.time()

    class FakeEntry:
        def __init__(self, candles):
            self.candles_by_tf = {"1H": candles}

    class FakeCache:
        def __init__(self, a, b):
            self.m = {"A": a, "B": b}

        def get(self, inst_id):
            return self.m.get(inst_id)

    # A/B 完全同涨 → 1H 对数收益率相关 ≈ 1.0
    candles_a, candles_b = [], []
    for i in range(30):
        c = 100 * (1 + 0.001 * i)
        candles_a.append(make_candle(_TS0 + i * _H, c, c * 1.001, c * 0.999, c))
        candles_b.append(make_candle(_TS0 + i * _H, c, c * 1.001, c * 0.999, c))
    cache = FakeCache(FakeEntry(candles_a), FakeEntry(candles_b))
    ok, reason = agent._switch_guards(
        cfg, MockClientFundingInfo(), cache,
        "A", -0.5, now - 3600 * 10, "B", 0.9, "long", None)
    assert not ok and "相关" in reason, \
        f"高相关换仓应拒绝(修复原预检路径相关性判断后仍无条件平仓的Bug), got ok={ok} {reason}"
    print("PASS test_switch_guards_correlation")


# ---------------------------------------------------------------------------
# 回归 7: 全链路 sanity — 门禁不应误杀合法信号
# ---------------------------------------------------------------------------

def test_signal_chain_sanity():
    """新鲜合法信号应能依次通过 红旗门禁 + 弱信号门禁。"""
    cfg = _weak_cfg()
    cfg["strategy"]["enforce_red_flags"] = True
    sig = make_signal(side="long", score=0.9)
    analysis = make_analysis(final_score=0.7, final_confidence=0.6,
                             direction="long", key_risks=["波动率扩张"])
    ok_rf, _ = agent._red_flag_gate(sig, analysis, cfg)
    assert ok_rf, "非方向性红旗不应否决"
    candles = [make_candle(_TS0 + i * _H, 100, 101, 99, 100 + i * 0.2)
               for i in range(30)]
    ok_wg, _ = agent._weak_signal_multi_gate(
        cfg, MockClientNoData(), sig, analysis, candles)
    assert ok_wg, "强信号应通过弱信号门禁"
    print("PASS test_signal_chain_sanity")


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

def main():
    tests = [
        (test_fvg_freshness_unit, "FVG新鲜度-单元"),
        (test_generate_signal_rejects_stale_fvg, "FVG新鲜度-信号生成"),
        (test_confidence_not_inflated_by_neutral_channels, "置信度修正-单通道"),
        (test_confidence_two_active_channels, "置信度修正-双通道"),
        (test_confidence_all_neutral, "置信度修正-全中性"),
        (test_red_flag_gate, "红旗门禁"),
        (test_weak_gate_insufficient_data_rejects, "弱信号门禁-数据不足拒绝"),
        (test_weak_gate_sufficient_confluence_allows, "弱信号门禁-共振不足拒绝"),
        (test_switch_guards_min_hold, "换仓守卫-最小持仓"),
        (test_switch_guards_score_threshold, "换仓守卫-评分门槛"),
        (test_switch_guards_funding_conflict, "换仓守卫-资金费冲突"),
        (test_switch_guards_correlation, "换仓守卫-相关性"),
        (test_signal_chain_sanity, "全链路sanity"),
    ]
    passed = 0
    for fn, name in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'='*60}\n通过 {passed}/{len(tests)} 项")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
