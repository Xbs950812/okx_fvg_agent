# -*- coding: utf-8 -*-
"""
全天值守监控脚本 — 只读，不干预 agent 运行。

每 300 秒将关键状态快照（余额/持仓/最近平仓/关键日志）追加到 duty_watch.log，
供用户次日询问"今日盈利"时快速汇总。

运行方式（无人值守）:
    python duty_watch.py
"""

import json
import logging
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "paper_state.json")
LOG_FILE = os.path.join(BASE_DIR, "duty_watch.log")

# 供手动指定 agent 输出日志（如 job 目录），缺省尝试常见位置
AGENT_LOG = ""
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    AGENT_LOG = sys.argv[1]

POLL_SECONDS = 300

# 关键事件关键词（用于从 agent 日志中抓取平仓/开仓）
KEY_EVENTS = (
    "平仓", "止损", "止盈", "成交", "开仓", "换仓",
    "Failed", "Error", "ERROR", "WARN",
)


def read_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def tail_key_events(agent_log: str, n_lines: int = 2000) -> list:
    """抓取 agent 日志尾部最近关键事件。"""
    if not agent_log or not os.path.exists(agent_log):
        return []
    try:
        with open(agent_log, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return []
    events = []
    for line in lines[-n_lines:]:
        if any(k in line for k in KEY_EVENTS):
            events.append(line.rstrip())
    # 只保留最后 20 条，避免日志无限膨胀
    return events[-20:]


def append_snapshot(state: dict, events: list) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n===== SNAPSHOT {ts} =====\n")
        balance = state.get("balance")
        f.write(f"balance={balance}\n")
        positions = state.get("positions") or []
        f.write(f"positions={len(positions)}\n")
        for p in positions:
            f.write(
                f"  {p.get('inst_id')} {p.get('side')} entry={p.get('entry_px')} "
                f"size={p.get('size')} lev={p.get('leverage')}x "
                f"mark={p.get('last_mark')} filled={p.get('filled')}\n"
            )
        trades = state.get("trades") or []
        f.write(f"trades_total={len(trades)}\n")
        if events:
            f.write("-- key events --\n")
            for e in events:
                f.write(f"  {e}\n")


def main() -> None:
    # 首次快照（基线）
    append_snapshot(read_state(), [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info(f"Duty watch started. state={STATE_FILE} agent_log={AGENT_LOG or 'N/A'}")
    while True:
        time.sleep(POLL_SECONDS)
        try:
            append_snapshot(read_state(), tail_key_events(AGENT_LOG))
        except Exception as e:
            logging.warning(f"snapshot failed: {e}")


if __name__ == "__main__":
    main()
