"""
README 素材生成 — 蒙特卡洛资金曲线对比图 (滚动 Kelly vs 固定 30%)。

复用 verify_kelly_monte_carlo.simulate_path 的 f_series（每笔风险比例）,
以完全一致的风险语义重建 equity 曲线, 保证图形与验证脚本数字同源:
  胜: equity × (1 + f%/100 × b)   负: equity × (1 − f%/100)

用法:
  python gen_readme_curves.py            # 生成 docs/images/montecarlo_curves.png
"""

import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verify_kelly_monte_carlo import simulate_path

P_WIN = 0.5        # 胜率 (验证脚本静态边 A 场景)
PAYOFF = 2.5       # 盈亏比
N_TRADES = 1000    # 每路径笔数
N_PATHS = 8        # 展示路径数
OUT = os.path.join(os.path.dirname(__file__), "docs", "images",
                   "montecarlo_curves.png")


def equity_curve(outcomes: list, f_series: list) -> list:
    """按 simulate_path 同语义重建 equity 倍数曲线 (起点 1.0)。"""
    eq, curve = 1.0, [1.0]
    for is_win, f in zip(outcomes, f_series):
        eq *= (1.0 + f / 100.0 * PAYOFF) if is_win else (1.0 - f / 100.0)
        curve.append(eq)
    return curve


def main():
    modes = [
        ("rolling_ewma", "Rolling Kelly (EWMA λ=0.97, production)", "tab:green"),
        ("fixed_30", "Fixed 30% risk per trade", "tab:red"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    fig.suptitle(
        f"Monte Carlo: {N_TRADES} trades/path × {N_PATHS} paths  "
        f"(win rate {P_WIN:.0%}, payoff {PAYOFF}×)",
        fontsize=12, fontweight="bold")

    stats = {}
    for ax, (mode, label, color) in zip(axes, modes):
        finals = []
        all_curves = []
        for path in range(N_PATHS):
            rng = random.Random(10_000 + path)   # 与验证脚本相同的种子族
            outcomes = [rng.random() < P_WIN for _ in range(N_TRADES)]
            res = simulate_path(outcomes, P_WIN, PAYOFF, mode)
            curve = equity_curve(outcomes, res["f_series"])
            all_curves.append(curve)
            finals.append(curve[-1])
            ax.plot(curve, lw=0.8, alpha=0.45, color=color)
        # 中位路径 (按最终倍数排序取中间一条) 加粗
        med_curve = sorted(all_curves, key=lambda c: c[-1])[len(all_curves) // 2]
        ax.plot(med_curve, lw=2.2, color=color, label="median path")
        ax.axhline(1.0, color="gray", lw=0.6, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"{label}\nfinal: median ×{sorted(finals)[len(finals)//2]:.1f}",
                     fontsize=10)
        ax.set_xlabel("trade #")
        ax.grid(alpha=0.25, which="both")
        stats[mode] = finals

    axes[0].set_ylabel("equity multiple (log scale)")
    axes[0].legend(loc="upper left", fontsize=9)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=150)
    print(f"saved: {OUT}")
    for mode, finals in stats.items():
        s = sorted(finals)
        print(f"  {mode:<14} median final = ×{s[len(s)//2]:.2f}")


if __name__ == "__main__":
    main()
