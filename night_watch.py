# -*- coding: utf-8 -*-
"""
夜间无人值守监控脚本 (night_watch) — 6 小时盯盘。

职责：
  1. 每 60s 检查 agent.log 新鲜度（>300s 无写入 → 告警；进程死亡 → 自动重启）
  2. 监控 paper_state.json 持仓变化（开仓/平仓/止损/止盈/换仓事件）
  3. 扫描 agent.log 尾部新增 ERROR/CRITICAL/风控断路器/暂停交易等关键事件
  4. 输出 night_watch.log（追加）+ night_watch_state.json（当前快照）
  5. 运行 6.5 小时后自动退出并输出汇总

纯标准库，无第三方依赖。启动命令：
  C:\\Users\\casey\\AppData\\Local\\Programs\\Python\\Python310\\python.exe night_watch.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_LOG = os.path.join(BASE_DIR, "agent.log")
PAPER_STATE = os.path.join(BASE_DIR, "paper_state.json")
AGENT_STATE = os.path.join(BASE_DIR, "agent_state.json")
WATCH_LOG = os.path.join(BASE_DIR, "night_watch.log")
WATCH_STATE = os.path.join(BASE_DIR, "night_watch_state.json")

PYTHON = r"C:\Users\casey\AppData\Local\Programs\Python\Python310\python.exe"
AGENT_CMD = [PYTHON, "agent.py", "--paper", "--auto", "--scan-interval", "120"]
AGENT_OUT = os.path.join(BASE_DIR, "night_watch_agent_out.log")

WATCH_HOURS = 6.5           # 监控时长（容差 0.5h）
CHECK_INTERVAL = 60         # 检查周期（秒）
STALE_LOG_SECONDS = 300     # 日志停滞阈值（秒）
MAX_RESTARTS = 3            # 自动重启上限
RESTART_COOLDOWN = 600      # 两次重启最小间隔（秒）

CRITICAL_KEYWORDS = [
    "ERROR", "CRITICAL", "Traceback",
    "Failed to save state", "绝对回撤断路器", "暂停交易",
    "连亏暂停", "exception", "Exception",
]

EXIT_KEYWORDS = [   # 平仓原因关键词（从 trades 的 reason 匹配）
    "stop_loss", "take_profit", "scale_out", "switch", "time_stop",
    "manual", "tp", "sl",
]


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, level="INFO", silent=False):
    line = f"{now_str()} | {level:7s} | {msg}"
    with open(WATCH_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if not silent:
        print(line)


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        log(f"保存 {path} 失败: {e}", "WARNING")


def log_age():
    try:
        return time.time() - os.path.getmtime(AGENT_LOG)
    except Exception:
        return float("inf")


def find_new_lines(state):
    """返回 agent.log 自上次读取以来的新增行。"""
    new_lines = []
    try:
        with open(AGENT_LOG, "r", encoding="utf-8", errors="replace") as f:
            f.seek(state.get("log_offset", 0))
            new_lines = f.readlines()
            state["log_offset"] = f.tell()
    except Exception as e:
        log(f"读取 agent.log 失败: {e}", "WARNING")
    return new_lines


def scan_critical(new_lines, last_hits, state):
    """扫描关键事件，避免重复告警（同关键词 10 分钟内只报一次）。"""
    hits = set()
    for line in new_lines:
        for kw in CRITICAL_KEYWORDS:
            if kw.lower() in line.lower():
                hits.add(kw)
    now = time.time()
    for kw in sorted(hits):
        if now - last_hits.get(kw, 0) > 600:
            log(f"!!! 关键事件 [{kw}]: {kw}", "ALERT")
            last_hits[kw] = now
    return hits


def snapshot_positions(paper):
    """返回当前持仓指纹列表。"""
    pos = paper.get("positions", []) if paper else []
    return [
        {
            "inst_id": p.get("inst_id"),
            "side": p.get("side"),
            "size": p.get("size"),
            "entry_px": p.get("entry_px"),
            "sl_px": p.get("sl_px"),
            "tp_px": p.get("tp_px"),
            "open_time": p.get("open_time"),
            "last_mark": p.get("last_mark"),
        }
        for p in pos
    ]


def detect_position_events(prev, curr, last_events, state):
    """对比前后持仓指纹，识别开仓/平仓/止损/止盈事件。"""
    prev_ids = {p["inst_id"]: p for p in prev}
    curr_ids = {p["inst_id"]: p for p in curr}
    now = time.time()
    events = []

    # 新开仓
    for inst_id, p in curr_ids.items():
        if inst_id not in prev_ids:
            key = f"open:{inst_id}"
            if now - last_events.get(key, 0) > 300:
                events.append(f"!!! 新开仓 {p['side']} {inst_id} size={p['size']} entry={p['entry_px']} "
                              f"sl={p['sl_px']} tp={p['tp_px']}")
                last_events[key] = now

    # 平仓（持仓消失）→ 从 trades 最后一条找原因
    paper = load_json(PAPER_STATE)
    trades = paper.get("trades", []) if paper else []
    for inst_id, p in prev_ids.items():
        if inst_id not in curr_ids:
            reason = "unknown"
            pnl = None
            if trades:
                t = trades[-1]
                reason = t.get("reason", "unknown")
                pnl = t.get("pnl")
            key = f"close:{inst_id}"
            if now - last_events.get(key, 0) > 300:
                events.append(f"!!! 平仓 {p['side']} {inst_id} reason={reason} pnl={pnl}")
                last_events[key] = now

    for ev in events:
        log(ev, "ALERT")
    return events


def check_agent_alive():
    """检查 python agent.py 进程是否存活。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*agent.py*' } | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=30,
        )
        pids = [l.strip() for l in out.stdout.splitlines() if l.strip().isdigit()]
        return pids
    except Exception as e:
        log(f"进程检查失败: {e}", "WARNING")
        return []


def restart_agent():
    """自动重启 agent（无人值守）。"""
    try:
        with open(AGENT_OUT, "a", encoding="utf-8") as f:
            f.write(f"\n=== {now_str()} 自动重启 agent ===\n")
        proc = subprocess.Popen(
            AGENT_CMD, cwd=BASE_DIR,
            stdout=open(AGENT_OUT, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        log(f"!!! 已自动重启 agent (PID {proc.pid})", "ALERT")
        return True
    except Exception as e:
        log(f"!!! 自动重启失败: {e}", "ALERT")
        return False


def main():
    start_ts = time.time()
    last_hits = {}
    last_events = {}
    state = {"log_offset": 0}
    # 初始化 log_offset：从文件尾部开始读
    try:
        state["log_offset"] = os.path.getsize(AGENT_LOG)
    except Exception:
        pass
    prev_positions = []
    restart_times = []
    balance_prev = None
    snapshot_ts = 0
    round_last = 0

    # 首次快照
    paper = load_json(PAPER_STATE)
    prev_positions = snapshot_positions(paper)
    if paper:
        balance_prev = paper.get("balance")
    save_json(WATCH_STATE, {
        "started_at": now_str(),
        "monitor_hours": WATCH_HOURS,
        "position_count": len(prev_positions),
        "balance": balance_prev,
    })

    log(f"夜间监控启动，共 {WATCH_HOURS}h，初始持仓 {len(prev_positions)} 个，余额 {balance_prev}")

    while True:
        elapsed = time.time() - start_ts
        if elapsed > WATCH_HOURS * 3600:
            log(f"监控时长 {WATCH_HOURS}h 已到，退出")
            break

        # 1. 日志新鲜度
        age = log_age()
        if age > STALE_LOG_SECONDS * 2:
            pids = check_agent_alive()
            log(f"!!! agent.log 已 {age:.0f}s 无更新 | 进程 PIDs: {pids or '无'}", "ALERT")
            if not pids:
                # 进程死亡 → 尝试重启
                if len(restart_times) >= MAX_RESTARTS:
                    log(f"!!! 已重启 {len(restart_times)} 次达上限，停止自动重启，需人工介入", "ALERT")
                elif restart_times and time.time() - restart_times[-1] < RESTART_COOLDOWN:
                    log("重启冷却期内，跳过", "WARNING")
                else:
                    restart_agent()
                    restart_times.append(time.time())
        elif age > STALE_LOG_SECONDS:
            log(f"agent.log 停滞 {age:.0f}s（正常单轮约 2min，观察中）", "WARNING")

        # 2. 扫描新增日志关键事件
        new_lines = find_new_lines(state)
        scan_critical(new_lines, last_hits, state)

        # 3. 持仓变化检测
        paper = load_json(PAPER_STATE)
        curr = snapshot_positions(paper)
        if prev_positions != curr:
            detect_position_events(prev_positions, curr, last_events, state)
            prev_positions = curr
        if paper:
            bal = paper.get("balance")
            if bal is not None and balance_prev is not None and abs(bal - balance_prev) > 1e-9:
                log(f"余额变化: {balance_prev:.4f} → {bal:.4f}")
            if bal is not None:
                balance_prev = bal

        # 4. 每 30 分钟记录持仓快照
        if time.time() - snapshot_ts >= 1800:
            snapshot_ts = time.time()
            paper = load_json(PAPER_STATE)
            curr = snapshot_positions(paper)
            detail = "; ".join(
                f"{p['inst_id']} {p['side']} {p['size']}张 entry={p['entry_px']} "
                f"sl={p['sl_px']} tp={p['tp_px']} mark={p['last_mark']}"
                for p in curr
            ) or "空仓"
            log(f"[快照] 持仓({len(curr)}): {detail} | 余额={balance_prev}")

        time.sleep(CHECK_INTERVAL)

    # 汇总
    paper = load_json(PAPER_STATE)
    curr = snapshot_positions(paper)
    ag_state = load_json(AGENT_STATE) or {}
    summary = {
        "finished_at": now_str(),
        "final_positions": curr,
        "balance": paper.get("balance") if paper else None,
        "agent_total_pnl": ag_state.get("total_pnl"),
        "restart_count": len(restart_times),
    }
    save_json(WATCH_STATE, summary)
    log(f"===== 监控汇总: 最终持仓 {len(curr)} 个 | 余额={summary['balance']} | "
        f"累计重启 {len(restart_times)} 次 =====")
    print(f"\n===== 监控汇总 =====\n{json.dumps(summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("监控被手动中断", "WARNING")
        sys.exit(0)
