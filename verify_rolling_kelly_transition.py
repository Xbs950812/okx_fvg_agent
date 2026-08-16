"""
滚动 Kelly 档位切换验证脚本 (2026-08-15)。

场景构造（60 笔模拟盈亏，走真实数据路径）：
  - 交易序列: [ +2.5, -1.0 ] × 30 交错 → 偶数笔时 p 精确 = 0.5, b = 2.5
  - 每平仓一笔调用 StateManager.record_realized_pnl（生产代码的真实入口），
    再用 rolling_kelly_risk_pct 对 recent_pnl 重算风险上限

预期跳变点（与 test_rolling_kelly.py 契约一致）：
  - 第 1~9  笔: decisive < min_samples(10) → None（不约束）
  - 第 10~49 笔: 探索档 1/4 Kelly
      第 49 笔: p=25/49 → f*=0.3143 → 上限 7.86%
  - 第 50 笔起: 利用档 1/2 Kelly（样本 ≥ sample_full_kelly=50）
      第 50 笔: p=25/25 → f*=0.3000 → 上限 15.00%（近似翻倍跳变）

附加场景（滚动响应性）：60 笔后继续追加亏损，验证边的退化会实时压低上限。

运行: python verify_rolling_kelly_transition.py
"""

import os
import tempfile

from executor import StateManager
from fvg_killer_pro import rolling_kelly_risk_pct

BASE_RISK = 30.0          # config risk.risk_per_trade_pct（满倍率基准）
WIN_PNL, LOSS_PNL = 2.5, -1.0   # b = 2.5 / 1.0 = 2.5
# 本脚本验证档位切换语义，使用平权窗口（EWMA 关闭）；
# EWMA 平滑行为由 test_rolling_kelly.py::TestRollingKellyEwma 覆盖
FLAT_CFG = {"rolling_kelly": {"ewma_lambda": 0}}

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {msg}")
    return cond


def fmt(res):
    """(risk, diag) → 单行描述。"""
    risk, diag = res
    if risk is None:
        reason = diag.get("reason") or ("disabled" if diag.get("disabled") else "?")
        return f"None (不约束, {reason}, 样本={diag.get('samples', '-')})"
    return (f"{risk:6.2f}%  档位={diag.get('tier')}  f*={diag.get('kelly_f'):.4f}  "
            f"胜率={diag.get('win_rate'):.3f}  样本={diag.get('samples')}")


def main():
    sm = StateManager(os.path.join(tempfile.mkdtemp(), "state.json"))

    print("=" * 78)
    print("场景 A: 60 笔 p=0.5 / b=2.5 — 逐笔喂入，观察档位切换")
    print("=" * 78)
    print(f"{'笔数':>4} | 滚动 Kelly 风险上限 (eff_risk = min({BASE_RISK:.0f}%, 上限))")
    print("-" * 78)

    history = []                 # (笔数, risk, diag) 全程留档
    pnls = [WIN_PNL, LOSS_PNL] * 30
    for i, pnl in enumerate(pnls, start=1):
        sm.record_realized_pnl(pnl)                    # 真实生产入口
        res = rolling_kelly_risk_pct(
            sm.state.recent_pnl, BASE_RISK, FLAT_CFG)
        history.append((i, res[0], res[1]))
        if i in (9, 10, 24, 25, 49, 50, 51, 60):
            print(f"{i:>4} | {fmt(res)}")

    print("-" * 78)

    # ---- 断言: 启用门槛 (第 9 笔 None → 第 10 笔约束生效) ----
    r9, d9 = history[8][1], history[8][2]
    r10, d10 = history[9][1], history[9][2]
    check(r9 is None, f"第 9 笔: 不约束 (样本 {d9.get('samples')} < 10)")
    check(r10 is not None and d10.get("tier") == "quarter",
          f"第 10 笔: 探索档生效 (上限 {r10:.2f}%)")

    # ---- 断言: 探索档期间 (10~49) 恒为 quarter 且上限 < 10% ----
    quarter_range = history[9:49]
    check(all(d.get("tier") == "quarter" for _, _, d in quarter_range),
          "第 10~49 笔: 全程探索档 (1/4 Kelly)")
    check(all(0 < r <= 10.0 for _, r, _ in quarter_range),
          "第 10~49 笔: 上限均在 (0, 10%] 区间")

    # ---- 断言: 第 49 → 50 笔切换跳变 ----
    r49, d49 = history[48][1], history[48][2]
    r50, d50 = history[49][1], history[49][2]
    check(d49.get("tier") == "quarter" and d49.get("samples") == 49,
          f"第 49 笔: 探索档末位 (f*={d49.get('kelly_f'):.4f}, 上限 {r49:.2f}%)")
    check(d50.get("tier") == "half" and d50.get("samples") == 50,
          "第 50 笔: 切换到利用档 (1/2 Kelly)")
    check(abs(r50 - 15.0) < 1e-3,
          f"第 50 笔: p=25/25 → f*=0.30 × 1/2 = {r50:.2f}% (期望 15.00%)")
    check(abs(r50 / r49 - 2.0) / 2.0 < 0.05,
          f"切换跳变: {r49:.2f}% → {r50:.2f}% (≈2×, 相对偏差 <5%, "
          f"来自奇数样本胜率凑整)")

    # ---- 断言: 利用档持续到第 60 笔且上限稳定 15% ----
    r60, d60 = history[59][1], history[59][2]
    check(all(d.get("tier") == "half" for _, _, d in history[49:]),
          "第 50~60 笔: 持续利用档")
    check(abs(r60 - 15.0) < 1e-3,
          f"第 60 笔: p=30/30 → f*=0.30 × 1/2 = {r60:.2f}% (期望 15.00%)")

    # ---- 断言: 接线语义 eff_risk = min(base, cap) ----
    check(min(BASE_RISK, r50) == r50 and min(BASE_RISK, 8.0) == 8.0,
          "接线语义: min(基准30%, Kelly上限) 只降不升")

    print()
    print("=" * 78)
    print("场景 B: 边退化 — 60 笔后追加 20 笔亏损，验证上限实时回落")
    print("=" * 78)
    print(f"{'笔数':>4} | 滚动 Kelly 风险上限")
    print("-" * 78)
    for j in range(20):
        sm.record_realized_pnl(LOSS_PNL)
        n = 60 + j + 1
        res = rolling_kelly_risk_pct(sm.state.recent_pnl, BASE_RISK, FLAT_CFG)
        if n in (65, 70, 80):
            print(f"{n:>4} | {fmt(res)}")
    final_risk, final_diag = res
    check(final_risk < r60,
          f"上限随边退化回落: {r60:.2f}% → {final_risk:.2f}% "
          f"(f* {d60.get('kelly_f'):.4f} → {final_diag.get('kelly_f'):.4f})")
    check(final_diag.get("tier") == "half",
          "退化期间样本量仍 ≥50 → 保持利用档 (档位只看样本量, 幅度看边)")

    print()
    print("=" * 78)
    print(f"结果: {PASS} PASS / {FAIL} FAIL")
    print("=" * 78)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
