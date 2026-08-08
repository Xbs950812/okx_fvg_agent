# -*- coding: utf-8 -*-
"""本地单测: 纸面引擎兜底成交 TP/SL 等比重算 (RR 保持)。

场景还原: HOME 挂单 entry=0.008206 TP=0.00898037 SL=0.00789625 (RR=2.50),
兜底成交 @0.008414 (偏离 +2.5%)。修复前 TP/SL 不动 → RR=1.09;
修复后按新成交价等比重算 → RR 保持 2.50。
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_trading import PaperPosition, PaperTradingEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("test_paper_assist_rr")


def make_engine():
    return PaperTradingEngine({
        "paper": {
            "enabled": True,
            "balance": 1000.0,
            "state_file": "test_paper_state.json",
            "maker_fee": 0.0002,
            "taker_fee": 0.0005,
            "limit_timeout_min": 30,
            "fill_assist_seconds": 120,
        },
    })


def make_pos():
    return PaperPosition(
        inst_id="HOME-USDT-SWAP", side="long",
        entry_px=0.008206, size=317.0, leverage=2.0, margin=130.0,
        tp_px=0.00898037, sl_px=0.00789625,
        open_time=1000.0, ct_val=100.0, filled=False, signal_id="x",
    )


def main():
    eng = make_engine()

    # 计算修复前 RR (挂单参数)
    pos = make_pos()
    rr0 = (pos.tp_px - pos.entry_px) / (pos.entry_px - pos.sl_px)
    log.info(f"挂单参数: entry={pos.entry_px} tp={pos.tp_px} sl={pos.sl_px} RR={rr0:.2f}")
    assert abs(rr0 - 2.50) < 0.05, f"挂单 RR 应为 2.50, got {rr0:.2f}"

    # 模拟兜底成交 @0.008414 (mark)
    md = {"mark": 0.008414}
    eng._fill_locked(pos, md, assist=True)
    log.info(f"兜底成交后: entry={pos.entry_px} tp={pos.tp_px} sl={pos.sl_px}")

    # 修复后: RR 应保持 (新 entry 下 TP/SL 等比重算)
    rr1 = (pos.tp_px - pos.entry_px) / (pos.entry_px - pos.sl_px)
    tp_dist = (pos.tp_px - pos.entry_px) / pos.entry_px
    sl_dist = (pos.entry_px - pos.sl_px) / pos.entry_px
    log.info(f"修复后: RR={rr1:.2f} tp_dist={tp_dist:.2%} sl_dist={sl_dist:.2%}")
    assert abs(rr1 - rr0) < 0.05, f"RR 应保持 {rr0:.2f}, got {rr1:.2f}"
    assert abs(tp_dist - 0.09437) < 0.005, f"tp_dist 应保持 9.44%"
    assert abs(sl_dist - 0.03775) < 0.005, f"sl_dist 应保持 3.78%"

    # 方向测试: short 镜像
    pos_s = PaperPosition(
        inst_id="X-USDT-SWAP", side="short",
        entry_px=0.0100, size=100.0, leverage=3.0, margin=100.0,
        tp_px=0.0092, sl_px=0.0105,  # tp_dist=8%, sl_dist=5%, RR=1.6
        open_time=1000.0, ct_val=10.0, filled=False, signal_id="x",
    )
    rr_s0 = (pos_s.entry_px - pos_s.tp_px) / (pos_s.sl_px - pos_s.entry_px)
    eng._fill_locked(pos_s, {"mark": 0.0103}, assist=True)  # 偏离 +3%
    rr_s1 = (pos_s.entry_px - pos_s.tp_px) / (pos_s.sl_px - pos_s.entry_px)
    log.info(f"short: 挂单 RR={rr_s0:.2f} → 兜底成交后 RR={rr_s1:.2f}")
    assert abs(rr_s1 - rr_s0) < 0.05, f"short RR 应保持 {rr_s0:.2f}, got {rr_s1:.2f}"

    # 无偏离 (正常限价成交) 时 TP/SL 不动
    pos2 = make_pos()
    eng._fill_locked(pos2, {"mark": pos2.entry_px}, assist=False)
    assert pos2.tp_px == 0.00898037 and pos2.sl_px == 0.00789625
    log.info("正常限价成交 TP/SL 不动 OK")

    log.info("=" * 50)
    log.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
