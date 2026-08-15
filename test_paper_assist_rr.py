# -*- coding: utf-8 -*-
"""本地单测: 纸面挂单"回补无望判定" (2026-08-11 重构替代 assist 强成交)。

背景: 原 fill_assist 在挂单超 120s 后按 mark 模拟市价成交(assist 强成交),
曾致 NEIRO +11.85%/LAB#1 -3.89% 追价入场, 入场价失真污染测试数据。
已移除强成交, 改为: 挂单超观察期后若 mark 偏离挂单价超阈值 → 撤单放弃(不追价);
偏离在阈值内 → 继续等自然回补。本测试验证该行为与正常限价成交语义。
"""
import logging
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_trading import PaperPosition, PaperTradingEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("test_paper_fill_gate")

fd, state_path = tempfile.mkstemp(prefix="fill_gate_test_", suffix=".json")
os.close(fd)


def make_engine():
    return PaperTradingEngine({
        "paper": {
            "enabled": True,
            "balance": 1000.0,
            "state_file": state_path,
            "maker_fee": 0.0002,
            "taker_fee": 0.0005,
            "limit_timeout_min": 30,
            "fill_assist_seconds": 120,
            "fill_assist_deviation_pct": 3.0,
        },
    })


def make_pos(open_age_s=200.0, entry=100.0, tp=104.0, sl=98.0, side="long"):
    return PaperPosition(
        inst_id="T-USDT-SWAP", side=side,
        entry_px=entry, size=100.0, leverage=3.0, margin=100.0,
        tp_px=tp, sl_px=sl,
        open_time=time.time() - open_age_s, ct_val=1.0, filled=False, signal_id="x",
    )


def main():
    # 场景1: 挂单超观察期且 mark 偏离超阈值 → 撤单放弃(不追价)
    eng = make_engine()
    pos = make_pos(open_age_s=200.0)
    eng._positions[pos.inst_id] = pos
    eng.set_market_data_provider(
        lambda i: {"candles": [{"high": 102.0, "low": 101.0, "close": 101.5}], "mark": 108.0})
    eng.update()  # mark 108 vs entry 100 → 偏离 8% > 3%; long 成交需 mark<=100 不满足
    assert pos.inst_id not in eng._positions, "偏离超阈值应撤单放弃"
    assert not pos.filled, "撤单放弃不得成交"
    log.info("场景1 OK: 偏离 8% > 3% → 撤单放弃, 不追价")

    # 场景2: 挂单超观察期但偏离在阈值内 → 继续等待自然回补
    eng2 = make_engine()
    pos2 = make_pos(open_age_s=200.0)
    eng2._positions[pos2.inst_id] = pos2
    eng2.set_market_data_provider(
        lambda i: {"candles": [{"high": 102.0, "low": 101.0, "close": 101.5}], "mark": 101.0})
    eng2.update()  # mark 101 vs entry 100 → 偏离 1% <= 3%; 未触及成交价
    assert pos2.inst_id in eng2._positions, "偏离在阈值内应继续挂单"
    assert not pos2.filled
    log.info("场景2 OK: 偏离 1% <= 3% → 继续等回补")

    # 场景3: 价格回补 → 正常限价成交, 成交价=挂单价, TP/SL 不动
    eng3 = make_engine()
    pos3 = make_pos(open_age_s=200.0)
    eng3._positions[pos3.inst_id] = pos3
    eng3.set_market_data_provider(
        lambda i: {"candles": [{"high": 101.0, "low": 99.0, "close": 99.5}], "mark": 99.0})
    eng3.update()  # long: mark 99 <= entry 100 → 限价成交 @100
    assert pos3.filled, "价格回补应限价成交"
    assert pos3.entry_px == 100.0, "成交价必须等于挂单价, 不追价"
    assert pos3.tp_px == 104.0 and pos3.sl_px == 98.0, "正常成交 TP/SL 不得重算"
    log.info("场景3 OK: 回补成交 @挂单价, TP/SL 不动")

    # 场景4: 观察期未到 → 即使偏离也不撤单
    eng4 = make_engine()
    pos4 = make_pos(open_age_s=10.0)  # 刚挂 10s < 120s
    eng4._positions[pos4.inst_id] = pos4
    eng4.set_market_data_provider(
        lambda i: {"candles": [{"high": 110.0, "low": 105.0, "close": 108.0}], "mark": 110.0})
    eng4.update()
    assert pos4.inst_id in eng4._positions, "观察期内不应撤单"
    log.info("场景4 OK: 观察期内保持挂单")

    os.unlink(state_path)
    log.info("=" * 50)
    log.info("ALL PASS")


if __name__ == "__main__":
    main()
