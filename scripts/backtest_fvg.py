#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FVG 历史回测命令行工具。

用法:
    python scripts/backtest_fvg.py \
        --symbol BTC-USDT-SWAP \
        --timeframe 1H \
        --days 180 \
        --config config.json \
        --output backtest_results.html

输出: 胜率/盈亏比/利润因子/最大回撤/夏普 + 交易明细 + 自包含 HTML 报告。
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvg_detector import FVGDetector  # noqa: E402
from fvg_backtest import FVGBacktest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("backtest_fvg")

_BAR_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1H": 3_600_000, "2H": 7_200_000, "4H": 14_400_000, "1D": 86_400_000,
}


def load_client(config_path: str):
    """加载 config 并实例化 OKXClient（延迟导入，仅 CLI 路径需要）。"""
    from okx_client import OKXClient
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return OKXClient(cfg), cfg


def fetch_candles(client, symbol: str, timeframe: str, days: int):
    from strategy import candles_from_raw
    bar_ms = _BAR_MS.get(timeframe, 3_600_000)
    need = int(days * 86_400_000 / bar_ms) + 20
    raw = client.get_candles_enhanced(symbol, bar=timeframe, limit=min(need, 5000))
    if not raw:
        return []
    return candles_from_raw(raw)


def fmt_ts(ts: int) -> str:
    return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")


def render_html(args, res: dict, fvgs_n: int) -> str:
    """生成自包含 HTML 报告。"""
    rows = []
    for t in res["trades"]:
        rows.append(
            f"<tr><td>{fmt_ts(t.entry_ts)}</td><td>{fmt_ts(t.exit_ts)}</td>"
            f"<td>{t.direction}</td><td>{t.entry:.6g}</td><td>{t.exit:.6g}</td>"
            f"<td class='{'pos' if t.return_pct >= 0 else 'neg'}'>"
            f"{t.return_pct:+.2f}%</td><td>{t.exit_reason}</td>"
            f"<td>{t.quality_score:.2f}</td></tr>"
        )
    eq = res["equity_curve"]
    eq_str = ",".join(f"{v:.2f}" for v in eq) if eq else ""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>FVG Backtest — {args.symbol}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;margin:24px;background:#f6f8fa;color:#24292f}}
h1{{font-size:22px}} .cards{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;min-width:130px}}
.card b{{font-size:20px;display:block}} .card span{{color:#57606a;font-size:12px}}
table{{border-collapse:collapse;width:100%;background:#fff;margin-top:8px}}
th,td{{border:1px solid #d0d7de;padding:6px 10px;font-size:13px;text-align:left}}
th{{background:#f0f3f6}} .pos{{color:#1a7f37;font-weight:600}} .neg{{color:#cf222e;font-weight:600}}
</style></head><body>
<h1>FVG 回测报告 — {args.symbol} / {args.timeframe} / {args.days} 天</h1>
<div class="cards">
<div class="card"><span>检测 FVG 数</span><b>{fvgs_n}</b></div>
<div class="card"><span>交易数</span><b>{res['n_trades']}</b></div>
<div class="card"><span>胜率</span><b>{res['win_rate']:.1%}</b></div>
<div class="card"><span>盈亏比(利润因子)</span><b>{res['profit_factor']:.2f}</b></div>
<div class="card"><span>总收益</span><b>{res['total_return']:.2%}</b></div>
<div class="card"><span>最大回撤</span><b>{res['max_drawdown']:.2%}</b></div>
<div class="card"><span>夏普(每笔)</span><b>{res['sharpe_ratio']:.2f}</b></div>
</div>
<h3>交易明细 ({len(res['trades'])})</h3>
<table><tr><th>入场时间</th><th>出场时间</th><th>方向</th><th>入场价</th>
<th>出场价</th><th>收益率</th><th>退出原因</th><th>质量分</th></tr>
{''.join(rows) if rows else '<tr><td colspan=8>无成交</td></tr>'}
</table>
<h3>资金曲线</h3>
<svg width="100%" height="200" viewBox="0 0 1000 200" preserveAspectRatio="none"
     xmlns="http://www.w3.org/2000/svg">
<rect width="1000" height="200" fill="#fff" stroke="#d0d7de"/>
<polyline points="{_polyline(eq_str)}" fill="none" stroke="#1a7f37" stroke-width="2"/>
</svg>
</body></html>"""


def _polyline(eq_str: str) -> str:
    if not eq_str:
        return "0,200 1000,200"
    vals = [float(v) for v in eq_str.split(",")]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = max(len(vals) - 1, 1)
    pts = []
    for i, v in enumerate(vals):
        x = i / n * 1000
        y = 200 - (v - lo) / span * 190 - 5
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def main():
    ap = argparse.ArgumentParser(description="FVG 历史回测工具")
    ap.add_argument("--symbol", required=True, help="合约 ID，如 BTC-USDT-SWAP")
    ap.add_argument("--timeframe", default="1H", help="K 线周期，默认 1H")
    ap.add_argument("--days", type=int, default=180, help="回溯天数，默认 180")
    ap.add_argument("--config", default="config.json", help="配置文件路径")
    ap.add_argument("--output", default="backtest_results.html", help="HTML 输出路径")
    ap.add_argument("--detector-enabled", action="store_true", default=True,
                    help="启用 FVGDetector 独立检测器（默认开启）")
    args = ap.parse_args()

    client, cfg = load_client(args.config)
    logger.info(f"拉取 {args.symbol} {args.timeframe} 近 {args.days} 天 K 线...")
    candles = fetch_candles(client, args.symbol, args.timeframe, args.days)
    if len(candles) < 60:
        logger.error(f"K 线不足: {len(candles)} 根")
        sys.exit(1)
    logger.info(f"K 线 {len(candles)} 根 ({fmt_ts(candles[0].timestamp)} → "
                f"{fmt_ts(candles[-1].timestamp)})")

    strategy_cfg = cfg.get("strategy", {})
    detector = FVGDetector(strategy_cfg)
    fvgs = detector.detect({args.timeframe: candles})
    fvgs = detector.filter_by_quality(fvgs, {"current_price": candles[-1].close})
    logger.info(f"检测到 {len(fvgs)} 个 FVG")

    bt = FVGBacktest(initial_capital=cfg.get("risk", {}).get("initial_capital", 10000))
    res = bt.run(fvgs, candles, {
        "max_entry_bars": strategy_cfg.get("backtest_max_entry_bars", 50),
        "max_hold_bars": strategy_cfg.get("backtest_max_hold_bars", 50),
        "stop_width_mult": strategy_cfg.get("backtest_stop_width_mult", 1.5),
        "position_pct": strategy_cfg.get("backtest_position_pct", 0.1),
    })

    print("\n=== FVG 回测结果 ===")
    print(f"检测 FVG: {len(fvgs)} | 交易: {res['n_trades']}")
    print(f"胜率: {res['win_rate']:.1%} | 利润因子: {res['profit_factor']:.2f}")
    print(f"总收益: {res['total_return']:.2%} | 最大回撤: {res['max_drawdown']:.2%}")
    print(f"夏普: {res['sharpe_ratio']:.2f}")
    print(f"退出原因分布: {_exit_dist(res['trades'])}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(render_html(args, res, len(fvgs)))
    logger.info(f"HTML 报告已输出: {os.path.abspath(args.output)}")


def _exit_dist(trades) -> str:
    from collections import Counter
    return dict(Counter(t.exit_reason for t in trades)) or "无"


if __name__ == "__main__":
    main()
