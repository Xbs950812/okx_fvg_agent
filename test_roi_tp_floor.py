# -*- coding: utf-8 -*-
"""本地单测: 动态ROI不得抢在信号TP之前落袋 (利润保护)。

场景还原: HOME 持仓 entry=0.008414, 信号 TP=0.009208 (+9.4%, RR2.5),
ROI 配置 {0:0.05} (持仓<60min 目标5%)。修复前价格 +5% 即落袋砍半;
修复后 ROI 生效目标 = max(5%, 9.4%×85%=8.0%) = 8.0% → TP 优先。
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_trading import PaperPosition, PaperTradingEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("test_roi_tp_floor")


def make_engine(**risk_ovr):
    risk = {
        "risk_per_trade_pct": 1.0,
        "margin_pct": 30.0,
        "dynamic_roi": {"240": 0.015, "120": 0.025, "60": 0.035, "0": 0.05},
        "max_hold_hours": 48,
    }
    risk.update(risk_ovr)
    return PaperTradingEngine({
        "paper": {
            "enabled": True,
            "balance": 1000.0,
            "state_file": "test_paper_state.json",
            "maker_fee": 0.0002,
            "taker_fee": 0.0005,
            "limit_timeout_min": 30,
        },
        "risk": risk,
    })


def make_home_pos():
    return PaperPosition(
        inst_id="HOME-USDT-SWAP", side="long",
        entry_px=0.008414, size=317.0, leverage=2.0, margin=130.0,
        tp_px=0.009207998194004387, sl_px=0.008096398671703631,
        open_time=time.time() - 3300, ct_val=100.0, filled=True, signal_id="x",
    )


def main():
    # ---- 用例1: 修复默认开启 — 价格+5.5% 不落袋 (ROI 被 TP 距离撑高) ----
    eng = make_engine()
    pos = make_home_pos()
    tp_dist = (pos.tp_px - pos.entry_px) / pos.entry_px
    roi_target = eng._dynamic_roi_target(55.0)
    # ROI 目标量纲=保证金收益率(含杠杆), TP 时 = tp_dist × leverage
    eff = max(roi_target, tp_dist * pos.leverage * eng.dynamic_roi_tp_floor_pct)
    log.info(f"TP价格距离={tp_dist:.2%} ×2x杠杆=保证金{tp_dist*2:.1%} "
             f"配置ROI={roi_target:.1%} 生效ROI={eff:.1%}")
    assert tp_dist * pos.leverage > roi_target, "前置: TP 保证金收益应比配置 ROI 远"
    assert abs(eff - tp_dist * pos.leverage * 0.85) < 1e-9, \
        f"生效ROI应为 TP×杠杆×85%={tp_dist*2*0.85:.1%}"

    # +5.5% 价格 → 保证金收益约11% < 生效16% → 不应 dynamic_roi
    md = {"candles": [{"high": 0.008877, "low": 0.008414, "close": 0.008877}],
          "mark": 0.008414 * 1.055}
    ex = eng._check_exit(pos, md)
    log.info(f"  +5.5% → exit={ex} (期望 None, TP 未到且 ROI 未达)")
    assert ex is None, f"+5.5% 不应落袋, got {ex}"

    # +8.5% 价格 → 保证金收益约17% ≥ 16% → dynamic_roi 兜底落袋
    md2 = {"candles": [{"high": 0.009129, "low": 0.008414, "close": 0.009129}],
           "mark": 0.008414 * 1.085}
    ex2 = eng._check_exit(pos, md2)
    log.info(f"  +8.5% → exit={ex2} (期望 dynamic_roi)")
    assert ex2 is not None and ex2[1] == "dynamic_roi", f"+8.5% 应 ROI 落袋, got {ex2}"

    # +9.4% 价格 (=TP) → take_profit 优先
    md3 = {"candles": [{"high": 0.009208, "low": 0.008414, "close": 0.009208}],
           "mark": 0.009208}
    ex3 = eng._check_exit(pos, md3)
    log.info(f"  +9.4% → exit={ex3} (期望 take_profit)")
    assert ex3 is not None and ex3[1] == "take_profit", f"+9.4% 应 TP 优先, got {ex3}"

    # ---- 用例2: TP 距离 < 配置ROI 时 (如 TP=4%) 行为不变 ----
    pos2 = PaperPosition(
        inst_id="T-USDT-SWAP", side="long",
        entry_px=100.0, size=500.0, leverage=6.0, margin=166.67,
        tp_px=104.0, sl_px=97.0,
        open_time=time.time() - 1800, ct_val=1.0, filled=True, signal_id="x",
    )
    eff2 = max(eng._dynamic_roi_target(30.0),
               (104 - 100) / 100.0 * eng.dynamic_roi_tp_floor_pct)
    log.info(f"TP=4% 配置ROI=5% → 生效ROI={eff2:.1%} (取配置 5%)")
    assert abs(eff2 - 0.05) < 1e-9, f"TP<ROI 时应取配置值, got {eff2:.1%}"

    log.info("=" * 50)
    log.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
