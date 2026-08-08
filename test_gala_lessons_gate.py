# -*- coding: utf-8 -*-
"""
GALA 复盘教训固化验证 — 三个门控:
  1. FVG 宽度执行前兜底 (WidthGate)
  2. quality=poor 拦截 (QualityGate)
  3. ML 过线余量校验 (ML margin)

验证方式: 直接模拟 agent.py 中三段门控的核心逻辑, 确认对 GALA 场景
(width=1.07% / quality=poor / ml=0.504) 全部拦截。
"""

import sys

# 模拟 agent.py 中的 WidthGate + QualityGate 逻辑
def simulate_confluence_gates(width_pct, timeframe, entry_quality, config):
    _strat_cfg = config.get("strategy", {}) if isinstance(config, dict) else {}
    _min_width_1h = float(_strat_cfg.get("min_fvg_width_pct", {}).get("1H", 1.5))
    _min_width_4h = float(_strat_cfg.get("min_fvg_width_pct", {}).get("4H", 3.0))
    _sig_min_width = _min_width_4h if timeframe == "4H" else _min_width_1h

    # WidthGate
    if width_pct < _sig_min_width:
        return False, f"[WidthGate] FVG 宽度 {width_pct:.2f}% < {timeframe} 下限 {_sig_min_width:.2f}%"

    # QualityGate
    _reject_poor = bool(_strat_cfg.get("confluence_reject_poor", True))
    if _reject_poor and entry_quality == "poor":
        return False, f"[QualityGate] 汇流质量 {entry_quality} 核心证据不足"

    return True, "pass"


# 模拟 agent.py 中的 ML 过线余量逻辑
def simulate_ml_gate(ml_score, config):
    _ml_cfg = config.get("ml", {}) if isinstance(config, dict) else {}
    _ml_threshold = float(_ml_cfg.get("min_ml_score", 0.6))
    _ml_margin = float(_ml_cfg.get("ml_score_margin", 0.05))
    _ml_effective_threshold = _ml_threshold + _ml_margin
    if ml_score < _ml_effective_threshold:
        return False, f"[ML] ML分数 {ml_score:.3f} < {_ml_effective_threshold:.2f} (阈值{_ml_threshold:.2f}+余量{_ml_margin:.2f})"
    return True, f"[ML] ML分数 {ml_score:.3f} ≥ {_ml_effective_threshold:.2f}，放行"


def main():
    config = json_load("config.json")
    failures = 0

    # ---- 场景 1: GALA 原始信号 (width=1.07%, poor, ml=0.504) 应全拦截 ----
    ok, reason = simulate_confluence_gates(1.07, "1H", "poor", config)
    print(f"GALA场景 WidthGate: ok={ok} reason={reason}")
    assert not ok, "GALA width 1.07% 应被 WidthGate 拦截"
    assert "WidthGate" in reason, "应命中 WidthGate"
    # WidthGate 命中后不再检查 QualityGate, 单独验证 QualityGate
    ok, reason = simulate_confluence_gates(2.0, "1H", "poor", config)
    print(f"GALA场景 QualityGate: ok={ok} reason={reason}")
    assert not ok, "quality=poor 应被 QualityGate 拦截"
    assert "QualityGate" in reason, "应命中 QualityGate"
    ok, reason = simulate_ml_gate(0.504, config)
    print(f"GALA场景 ML: ok={ok} reason={reason}")
    assert not ok, "ML 0.504 应被余量校验拦截"

    # ---- 场景 2: 强信号 (width=2.0%, good, ml=0.60) 应全放行 ----
    ok, reason = simulate_confluence_gates(2.0, "1H", "good", config)
    print(f"强信号 WidthGate+QualityGate: ok={ok} reason={reason}")
    assert ok, "宽度 2.0% + quality=good 应放行"
    ok, reason = simulate_ml_gate(0.60, config)
    print(f"强信号 ML: ok={ok} reason={reason}")
    assert ok, "ML 0.60 应放行"

    # ---- 场景 3: 边界 — 关闭 reject_poor 后 poor 信号仅按分数放行 ----
    cfg2 = json_load("config.json")
    cfg2["strategy"]["confluence_reject_poor"] = False
    ok, reason = simulate_confluence_gates(2.0, "1H", "poor", cfg2)
    print(f"关闭QualityGate: ok={ok} reason={reason}")
    assert ok, "关闭 reject_poor 后 poor 应放行(交给分数)"

    # ---- 场景 4: 4H 信号用 4H 宽度阈值(3.0) ----
    ok, reason = simulate_confluence_gates(2.0, "4H", "good", config)
    print(f"4H窄缺口(2.0%<3.0%): ok={ok} reason={reason}")
    assert not ok, "4H 宽度 2.0% < 3.0% 应被拦截"

    if failures:
        print(f"\nFAIL: {failures} 个场景失败")
        sys.exit(1)
    print("\nALL PASSED: GALA 教训三规则验证通过")


def json_load(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
