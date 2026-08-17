"""
README 推广素材生成 — 社交预览横幅 + 架构流程图。

输出:
  docs/images/social_preview.png  — 1280x640, 链接分享卡片图 (上传到 GitHub
                                    Settings -> Social preview)
  docs/images/architecture.png    — 系统管线图 (README 用)

用法: python gen_promo_assets.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "docs", "images")

# GitHub dark 主题色
BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"
BOX = "#161b22"
BORDER = "#30363d"


def social_banner():
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 64)
    ax.axis("off")

    # 背景: 半透明蒙特卡洛风格资金曲线 (纯装饰性指数曲线)
    import math
    import random

    rng = random.Random(42)
    for k in range(7):
        x, y, pts = 0.0, 1.0, [1.0]
        for i in range(120):
            x += 1
            y *= 1.0 + rng.uniform(-0.03, 0.05)
            pts.append(y)
        xs = [i * 128 / 120 for i in range(len(pts))]
        base = [p * 30 for p in pts]  # 拉伸到底部区域
        ax.plot(xs, base, lw=1.0, alpha=0.10, color=GREEN if k % 2 else ACCENT)

    # 标题
    ax.text(6, 46, "FVG KILLER", fontsize=52, fontweight="bold",
            color=FG, family="DejaVu Sans", va="center")
    ax.text(6.2, 37.5,
            "ICT Fair Value Gap trading agent for OKX perpetual futures",
            fontsize=17, color=MUTED, va="center")

    # 特性标签 (圆角胶囊)
    chips = [
        ("197 tests passing", GREEN),
        ("Rolling Kelly risk engine", ACCENT),
        ("Live trading guards", PURPLE),
        ("Monte Carlo validated", ORANGE),
        ("10% profit royalty", RED),
    ]
    x = 6.5
    for label, color in chips:
        w = 2.1 * len(label) * 0.62 + 4
        ax.add_patch(FancyBboxPatch(
            (x, 24), w, 6.5, boxstyle="round,pad=0.6",
            facecolor=BOX, edgecolor=color, linewidth=1.4))
        ax.text(x + w / 2, 27.2, label, fontsize=12.5, color=color,
                ha="center", va="center", fontweight="bold")
        x += w + 2.6

    # 底部: 模块管线一览
    ax.text(6.5, 15, "FVG detect  ·  5-channel analysis  ·  multi-agent debate  ·  "
                     "regime detection  ·  461 alpha factors  ·  OCO protection  ·  "
                     "3-way reconciliation",
            fontsize=11.5, color=MUTED, va="center")

    ax.text(6.5, 7, "github.com/Xbs950812/okx_fvg_agent",
            fontsize=12, color=FG, va="center", family="monospace",
            fontweight="bold")

    ax.add_patch(FancyBboxPatch(
        (2, 2), 124, 60, boxstyle="round,pad=1.2",
        facecolor="none", edgecolor=BORDER, linewidth=2))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "social_preview.png")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    print(f"saved: {out}")


def _box(ax, x, y, w, h, title, lines, color, tsize=10.5, lsize=8.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35",
        facecolor=BOX, edgecolor=color, linewidth=1.5))
    ax.text(x + w / 2, y + h - 2.6, title, fontsize=tsize, fontweight="bold",
            color=color, ha="center", va="center")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 5.6 - i * 2.5, ln, fontsize=lsize,
                color=MUTED, ha="center", va="center")


def _arrow(ax, x1, y1, x2, y2, color=BORDER):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=1.4))


def architecture():
    fig, ax = plt.subplots(figsize=(13.2, 8.6), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 86)
    ax.axis("off")
    ax.set_title("FVG KILLER — System Pipeline", fontsize=17, fontweight="bold",
                 color=FG, pad=16)

    # ---- L1 数据层 ----
    _box(ax, 3, 70, 26, 12, "Market Data", ["OKX WebSocket tickers",
                                            "REST candles 1H/4H",
                                            "24h movers board"], ACCENT)
    _box(ax, 33, 70, 26, 12, "Background Research", ["Top-100 coin tracker",
                                                     "Smart-money (copy API)",
                                                     "research cache TTL"], ACCENT)
    _box(ax, 63, 70, 26, 12, "Anomaly Detection", ["3-sigma price move",
                                                   "5x volume spike",
                                                   "ExtremeMove gate"], PURPLE)
    _box(ax, 93, 70, 26, 12, "FVG Detection (ICT)", ["3-candle gap 1H/4H",
                                                     "CE mid-point rule",
                                                     "liquidity extension"], GREEN)

    # ---- L2 研判层 ----
    _box(ax, 8, 52, 30, 12, "Five-Channel Analysis", ["price action / structure",
                                                      "fund flow / sentiment",
                                                      "macro background"], ACCENT)
    _box(ax, 44, 52, 30, 12, "Multi-Agent Debate", ["6 analysts debate",
                                                    "reputation updates",
                                                    "structured verdict"], ACCENT)
    _box(ax, 80, 52, 30, 12, "Regime + Alpha Zoo", ["causal hysteresis FSM",
                                                    "461 factors",
                                                    "FreqAI online learning"], PURPLE)

    # ---- L3 风控层 ----
    _box(ax, 3, 32, 30, 12, "Risk Gate", ["rolling Kelly cap (EWMA)",
                                          "order-book depth 5%",
                                          "vol targeting / MoverDir"], ORANGE)
    _box(ax, 39, 32, 26, 12, "Market Guard", ["BTC crisis circuit",
                                              "funding/OI spikes",
                                              "breadth check"], RED)
    _box(ax, 71, 32, 22, 12, "Breakers", ["daily loss cap",
                                          "max daily trades",
                                          "drawdown 15%"], RED)
    _box(ax, 99, 32, 22, 12, "Rate Limiter", ["global token bucket",
                                              "10 QPS default"], ORANGE)

    # ---- L4 执行层 ----
    _box(ax, 8, 12, 30, 12, "Execution", ["limit entry @ FVG boundary",
                                          "OCO TP/SL + trailing",
                                          "pending-order lifecycle"], GREEN)
    _box(ax, 44, 52 - 0, 30, 0.1, "", [], BG)  # spacer no-op
    _box(ax, 44, 12, 30, 12, "Monitoring", ["3-way reconciliation",
                                            "slippage feedback loop",
                                            "funding-fee audit"], GREEN)
    _box(ax, 80, 12, 30, 12, "Royalty Module", ["10% of closed profits",
                                                "pool >= 20 USDT",
                                                "auto TRC20 withdraw"], ORANGE)

    # ---- 箭头 ----
    for x in (16, 46, 76, 106):
        _arrow(ax, x, 70, x, 64.5, ACCENT)
    _arrow(ax, 76, 76, 93, 76, GREEN)          # anomaly -> FVG
    _arrow(ax, 23, 64.5, 23, 58, MUTED)
    _arrow(ax, 59, 64.5, 59, 58, MUTED)
    _arrow(ax, 95, 64.5, 95, 58, MUTED)
    _arrow(ax, 38, 58, 44, 58, MUTED)          # five-channel -> debate
    _arrow(ax, 74, 58, 80, 58, MUTED)          # debate -> regime
    _arrow(ax, 23, 52, 18, 44, ORANGE)         # verdicts -> risk gate
    _arrow(ax, 95, 52, 110, 44, ORANGE)
    _arrow(ax, 95, 52, 82, 44, RED)
    _arrow(ax, 18, 32, 23, 24.5, GREEN)        # gate -> execution
    _arrow(ax, 38, 24, 44, 24, GREEN)          # exec -> monitoring
    _arrow(ax, 74, 24, 80, 24, ORANGE)         # monitoring -> royalty

    ax.text(66, 4,
            "paper / dry-run modes never place real orders  ·  "
            "royalty never transfers in simulation",
            fontsize=9.5, color=MUTED, ha="center", style="italic")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "architecture.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    social_banner()
    architecture()
