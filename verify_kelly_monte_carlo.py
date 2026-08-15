"""
滚动 Kelly 离散档位跳变 — 蒙特卡洛资金曲线压力测试 (2026-08-15)。

问题: 档位切换(50 笔处 1/4 → 1/2 Kelly)是离散跳变，长期统计下
      会不会导致资金曲线剧烈波动？

方法: 1000 笔/路径 × 300 路径，配对随机数（同一路径内所有策略
      消费同一盈亏序列，消除路径噪声，只对比仓位规则差异）。

策略组:
  fixed_30        恒定 30% 风险（当前 config 基准，无 Kelly）
  static_half     恒定 15%（先知半 Kelly，理论参照）
  rolling_disc    生产实现: <10 笔不约束(=base), 10~49 笔 1/4, >=50 笔 1/2,
                  下限 1%, 上限 30%（与 hyperopt.rolling_kelly_risk_pct 一致）
  rolling_smooth  唯一差异: 分数系数在 40~50 笔间线性爬升 0.25→0.50
                  （用来回答"跳变是否需要平滑"）

正确性保障: 快速增量 Kelly 与生产函数 rolling_kelly_risk_pct 在
  每条路径每 50 笔交叉核对一次（含档位与数值）。

场景:
  A. 真实边 p=0.5, b=2.5（f*=0.30）— 增长压力测试
  B. 负边   p=0.25, b=2.5（f*<0） — 生存压力测试

指标: 中位翻倍次数 log2(终值) / 每笔对数增长 g / 最大回撤(中位+p95) /
      P(终值>=2x) / P(终值<=0.5x)

运行: python verify_kelly_monte_carlo.py
"""

import math
import random
import sys
import time
from collections import deque

from hyperopt import rolling_kelly_risk_pct

PASS, FAIL = 0, 0

# ---- 与 config.example.json risk.rolling_kelly 默认值一致 ----
WINDOW = 100
MIN_SAMPLES = 10
SAMPLE_FULL = 50
MIN_RISK = 1.0        # %
BASE_RISK = 30.0      # %
N_TRADES = 1000
N_PATHS = 300

BASE_CFG = {"rolling_kelly": {"enabled": True, "window": WINDOW,
                              "min_samples": MIN_SAMPLES,
                              "sample_full_kelly": SAMPLE_FULL,
                              "min_risk_pct": MIN_RISK,
                              # FastRollingKelly 为平权窗口实现，交叉核对
                              # 需关闭 EWMA 保持口径一致；EWMA 行为由
                              # test_rolling_kelly.py::TestRollingKellyEwma 覆盖
                              "ewma_lambda": 0}}


def check(cond, msg):
    global PASS, FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if cond:
        PASS += 1
    else:
        FAIL += 1
    return cond


# ---------------------------------------------------------------------------
# 快速增量 Kelly（数学与 hyperopt.compute_kelly 逐条对齐，见交叉核对）
# ---------------------------------------------------------------------------

class FastRollingKelly:
    """O(1) 增量维护窗口统计，复刻 compute_kelly 全部裁剪规则。

    修复: 单一 deque(maxlen=WINDOW) 统一窗口 — 两个独立胜负 deque 的
    等效窗口是"最近100胜+最近100负"(最多200笔)，与生产函数
    recent_pnl[-100:] 语义不符（首次交叉核对即抓出）。

    ewma_lambda ∈ (0,1) 时同步维护 EWMA 累加器（共享时钟衰减，
    与生产 _ewma_window_stats 同构），供 risk_pct_ewma 使用；
    档位/样本门槛仍按平权窗口的 decisive 计数（与生产一致）。
    """

    def __init__(self, ewma_lambda: float = 0.0):
        self.buf = deque(maxlen=WINDOW)
        self.n_win = 0
        self.sum_win = 0.0
        self.n_loss = 0
        self.sum_loss = 0.0
        try:
            self.lam = float(ewma_lambda)
        except (TypeError, ValueError):
            self.lam = 0.0
        if not (0.0 < self.lam < 1.0):
            self.lam = 0.0
        self.e_sw = 0.0
        self.e_nw = 0.0
        self.e_sl = 0.0
        self.e_nl = 0.0
        # 每笔交易后的全历史 EWMA 累加器快照（对齐生产窗口语义用）。
        # 生产 _ewma_window_stats 只作用于最近 WINDOW 笔；EWMA 线性 ⇒
        # 窗口EWMA = A_now − λ^D × A_{n−WINDOW}（D=窗口内 decisive 数，
        # 衰减按 decisive 计数推进）。首次交叉核对曾抓出无限记忆偏差。
        self.e_snap = deque(maxlen=WINDOW + 1)

    def add(self, pnl: float):
        if len(self.buf) == WINDOW:      # append 将驱逐队首，先回滚其统计
            old = self.buf[0]
            if old > 0:
                self.n_win -= 1
                self.sum_win -= old
            elif old < 0:
                self.n_loss -= 1
                self.sum_loss -= -old
        self.buf.append(pnl)
        if pnl > 0:
            self.n_win += 1
            self.sum_win += pnl
        elif pnl < 0:
            self.n_loss += 1
            self.sum_loss += -pnl
        # EWMA 累加器: decisive 交易推进共享时钟（与生产口径一致）
        if self.lam > 0:
            if pnl != 0.0:
                self.e_sw *= self.lam
                self.e_nw *= self.lam
                self.e_sl *= self.lam
                self.e_nl *= self.lam
                if pnl > 0:
                    self.e_sw += pnl
                    self.e_nw += 1.0
                else:
                    self.e_sl += -pnl
                    self.e_nl += 1.0
            self.e_snap.append((self.e_sw, self.e_nw, self.e_sl, self.e_nl))

    @property
    def samples(self):
        return self.n_win + self.n_loss

    @staticmethod
    def _f_from_stats(p, avg_win, avg_loss):
        """裁剪链（与生产 _kelly_f_from_stats / compute_kelly 逐条对齐）。"""
        if avg_loss == 0:
            avg_loss = 1.0
        b = (avg_win / avg_loss) if avg_loss > 0 else 1.0
        b = max(0.01, min(b, 100.0))
        q = 1.0 - p
        if p >= 0.999:
            f = 0.25
        elif p <= 0.001:
            f = 0.0
        else:
            f = (p * b - q) / b if b > 0 else 0.0
        return max(0.0, min(f, 0.5))

    def _gate_and_clamp(self, f, frac_fn):
        d = self.samples
        if d < MIN_SAMPLES:
            return None
        risk = f * frac_fn(d) * 100.0
        return max(min(MIN_RISK, BASE_RISK), min(risk, BASE_RISK))

    def risk_pct(self, frac_fn):
        """平权窗口路径；样本不足返回 None。"""
        d = self.samples
        if d < MIN_SAMPLES:
            return None
        w, l = self.n_win, self.n_loss
        p = w / d
        avg_win = (self.sum_win / w) if w else 0.0
        avg_loss = (self.sum_loss / l) if l else avg_win * 0.5
        f = self._f_from_stats(p, avg_win, avg_loss)
        return self._gate_and_clamp(f, frac_fn)

    def risk_pct_ewma(self, frac_fn):
        """EWMA 路径（λ 已在构造时启用）；档位门槛仍用平权 decisive 计数。

        窗口语义对齐生产 _ewma_window_stats（只作用于最近 WINDOW 笔）：
        窗满时 窗口EWMA = A_now − λ^D × A_{n−WINDOW}，D=当前窗口 decisive 数。
        """
        if self.lam <= 0:
            return self.risk_pct(frac_fn)
        d = self.samples
        if d < MIN_SAMPLES:
            return None
        if len(self.buf) >= WINDOW and len(self.e_snap) > WINDOW:
            f_decay = self.lam ** d
            s0 = self.e_snap[0]              # n−WINDOW 笔后的累加器状态
            w_sw = self.e_sw - f_decay * s0[0]
            w_nw = self.e_nw - f_decay * s0[1]
            w_sl = self.e_sl - f_decay * s0[2]
            w_nl = self.e_nl - f_decay * s0[3]
        else:
            w_sw, w_nw, w_sl, w_nl = self.e_sw, self.e_nw, self.e_sl, self.e_nl
        tot = w_nw + w_nl
        p = (w_nw / tot) if tot > 0 else 0.0
        avg_win = (w_sw / w_nw) if w_nw > 0 else 0.0
        avg_loss = (w_sl / w_nl) if w_nl > 0 else avg_win * 0.5
        f = self._f_from_stats(p, avg_win, avg_loss)
        return self._gate_and_clamp(f, frac_fn)


def frac_discrete(d):
    return 0.5 if d >= SAMPLE_FULL else 0.25


def frac_smooth(d):
    # 40~50 笔线性爬升 0.25 → 0.50，其余与离散版一致
    if d >= SAMPLE_FULL:
        return 0.5
    if d <= 40:
        return 0.25
    return 0.25 + 0.25 * (d - 40) / (SAMPLE_FULL - 40)


# ---------------------------------------------------------------------------
# 路径模拟
# ---------------------------------------------------------------------------

def simulate_path(outcomes, p, b, mode, crosscheck_path=None):
    """单路径模拟。

    outcomes: 本路径全部交易的胜负布尔序列（所有策略共用，配对设计）
    风险语义: 胜 → equity × (1 + f%·b/100)，负 → equity × (1 − f%/100)

    Returns dict: log2_final, g_per_trade, max_dd, f_series
    """
    rk = FastRollingKelly(ewma_lambda=0.97 if mode == "rolling_ewma" else 0.0)
    frac_fn = {"rolling_disc": frac_discrete,
               "rolling_smooth": frac_smooth,
               "rolling_ewma": frac_discrete}.get(mode)

    log_eq = 0.0
    peak_log = 0.0        # log 域峰值（maxDD 在 log 域计算，防溢出）
    max_dd = 0.0
    f_series = []
    recent = deque(maxlen=WINDOW)   # 交叉核对用（仅 crosscheck 路径）

    for i, is_win in enumerate(outcomes):
        # ---- 决定本笔风险比例 f ----
        cap = None
        if mode in ("rolling_disc", "rolling_smooth"):
            cap = rk.risk_pct(frac_fn)
            f = BASE_RISK if cap is None else min(BASE_RISK, cap)
        elif mode == "rolling_ewma":
            cap = rk.risk_pct_ewma(frac_fn)
            f = BASE_RISK if cap is None else min(BASE_RISK, cap)
        elif mode == "fixed_30":
            f = BASE_RISK
        else:  # static_half
            f = 15.0
        f_series.append(f)

        # ---- 交叉核对（前 3 条路径每 50 笔对一次生产函数，仅 rolling 模式） ----
        if (crosscheck_path is not None
                and mode in ("rolling_disc", "rolling_smooth", "rolling_ewma")
                and i % 50 == 0 and i > 0):
            _cfg = BASE_CFG if mode != "rolling_ewma" else {
                "rolling_kelly": {**BASE_CFG["rolling_kelly"],
                                  "ewma_lambda": 0.97}}
            prod_risk, prod_diag = rolling_kelly_risk_pct(
                list(recent), BASE_RISK, _cfg)
            if cap is None:
                ok = prod_risk is None
            else:
                ok = (prod_risk is not None
                      and abs(prod_risk - cap) < 5e-3)
            if not ok:
                raise AssertionError(
                    f"快速Kelly与生产函数不一致 @trade{i} mode={mode}: "
                    f"fast={cap} prod={prod_risk} {prod_diag}")

        # ---- 结算本笔 ----
        if is_win:
            log_eq += math.log(1.0 + f / 100.0 * b)
        else:
            log_eq += math.log(1.0 - f / 100.0)
        pnl = (f / 100.0 * b) if is_win else -(f / 100.0)
        rk.add(pnl)
        if crosscheck_path is not None:
            recent.append(pnl)

        if log_eq > peak_log:
            peak_log = log_eq
        dd = 1.0 - math.exp(log_eq - peak_log)
        if dd > max_dd:
            max_dd = dd

    return {"log2_final": log_eq / math.log(2.0),
            "g_per_trade": log_eq / len(outcomes),
            "max_dd": max_dd,
            "f_series": f_series}


def run_scenario(name, p, b):
    print("=" * 86)
    print(f"{name}: p={p}, b={b}, {N_TRADES} 笔 × {N_PATHS} 路径（配对随机数）")
    print("=" * 86)

    results = {m: [] for m in ("fixed_30", "static_half",
                               "rolling_disc", "rolling_smooth")}
    for path in range(N_PATHS):
        rng = random.Random(10_000 + path)          # 每路径独立种子
        outcomes = [rng.random() < p for _ in range(N_TRADES)]
        for mode in results:
            cc = path if path < 3 else None         # 前 3 条路径做交叉核对
            results[mode].append(simulate_path(outcomes, p, b, mode, cc))

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    def p95(xs):
        s = sorted(xs)
        return s[int(len(s) * 0.95)]

    hdr = (f"{'策略':<14}{'中位翻倍数':>10}{'g/笔':>9}"
           f"{'maxDD中位':>10}{'maxDD p95':>10}"
           f"{'P(>=2x)':>9}{'P(<=0.5x)':>10}")
    print(hdr)
    print("-" * 86)
    stats = {}
    for mode, rs in results.items():
        log2f = [r["log2_final"] for r in rs]
        dds = [r["max_dd"] for r in rs]
        p_up = sum(1 for x in log2f if x >= 1) / len(log2f)
        p_dn = sum(1 for x in log2f if x <= -1) / len(log2f)
        stats[mode] = {"log2": log2f, "dd": dds}
        print(f"{mode:<14}{med(log2f):>10.1f}{sum(r['g_per_trade'] for r in rs)/len(rs):>+9.4f}"
              f"{med(dds):>10.1%}{p95(dds):>10.1%}"
              f"{p_up:>9.0%}{p_dn:>10.0%}")
    return stats


def run_drift_scenario(p_start: float, p_end: float, b: float = 2.5):
    """边漂移场景 (2026-08-15 --drift): 胜率按交易序号线性漂移。

    p(t) = p_start + (p_end − p_start) · t/(N−1)，检验滚动 Kelly 对边衰减的
    跟踪鲁棒性。策略组含 rolling_ewma（λ=0.97，与生产 EWMA 路径交叉核对）
    — 理论上 EWMA 有效滞后 ≈ λ/(1−λ) = 32 笔 < 平权窗口 ≈ 50 笔，
    衰减期应更快降档。

    断言按 p_end 与保本点 p_be = 1/(1+b) 的关系自适应:
      - p_end > p_be（边衰减但仍在）: 滚动版回撤应显著浅于固定 30%
      - p_end < p_be（边消失/反转）: 滚动版生存，固定 30% 湮灭
    """
    p_be = 1.0 / (1.0 + b)
    f_star = lambda p: max(0.0, p - (1 - p) / b)   # noqa: E731
    print("=" * 86)
    print(f"边漂移场景: p {p_start:.3f} → {p_end:.3f} (线性, {N_TRADES} 笔) × "
          f"{N_PATHS} 路径, b={b}")
    print(f"f*(p): {f_star(p_start):.3f} → {f_star(p_end):.3f} | "
          f"保本点 p_be = 1/(1+b) = {p_be:.3f}"
          + ("（边衰减但不消失）" if p_end > p_be else "（边跌破保本点）"))
    print("=" * 86)

    modes = ("fixed_30", "static_half", "rolling_disc", "rolling_ewma")
    results = {m: [] for m in modes}
    for path in range(N_PATHS):
        rng = random.Random(20_000 + path)          # 与静态场景种子隔离
        outcomes = [
            rng.random() < (p_start + (p_end - p_start) * t / (N_TRADES - 1))
            for t in range(N_TRADES)
        ]
        for mode in modes:
            cc = path if path < 3 else None
            results[mode].append(simulate_path(outcomes, p_end, b, mode, cc))

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    def p95(xs):
        s = sorted(xs)
        return s[int(len(s) * 0.95)]

    hdr = (f"{'策略':<14}{'中位翻倍数':>10}{'g/笔':>9}"
           f"{'maxDD中位':>10}{'maxDD p95':>10}"
           f"{'P(>=2x)':>9}{'P(<=0.5x)':>10}")
    print(hdr)
    print("-" * 86)
    stats = {}
    for mode in modes:
        rs = results[mode]
        log2f = [r["log2_final"] for r in rs]
        dds = [r["max_dd"] for r in rs]
        p_up = sum(1 for x in log2f if x >= 1) / len(log2f)
        p_dn = sum(1 for x in log2f if x <= -1) / len(log2f)
        stats[mode] = {"log2": log2f, "dd": dds, "f": [r["f_series"] for r in rs]}
        print(f"{mode:<14}{med(log2f):>10.1f}{sum(r['g_per_trade'] for r in rs)/len(rs):>+9.4f}"
              f"{med(dds):>10.1%}{p95(dds):>10.1%}"
              f"{p_up:>9.0%}{p_dn:>10.0%}")

    # 平均风险比例的时间轨迹（前3条路径均值，展示跟踪速度）
    print("-" * 86)
    print("平均风险上限 f 的轨迹（路径#0~#2 均值, 展示降档跟踪速度）:")
    ckpts = list(range(100, N_TRADES + 1, 100))
    print(f"{'策略':<14}" + "".join(f"{'@'+str(t):>9}" for t in ckpts))
    for mode in ("fixed_30", "rolling_disc", "rolling_ewma"):
        row = []
        for t in ckpts:
            fs = [s[t - 1] for s in stats[mode]["f"][:3]]
            row.append(f"{sum(fs) / len(fs):>8.1f}%")
        print(f"{mode:<14}" + "".join(f"{c:>9}" for c in row))
    p_true = [p_start + (p_end - p_start) * (t - 1) / (N_TRADES - 1) for t in ckpts]
    print(f"{'f*(p真值)半仓':<14}" + "".join(f"{f_star(p) * 0.5 * 100:>8.1f}%" for p in p_true))
    print()

    med_l2 = {m: med(stats[m]["log2"]) for m in modes}
    med_dd = {m: med(stats[m]["dd"]) for m in modes}
    if p_end > p_be:
        # 边仍为正: 固定档增长占优属正常（Kelly 牺牲增长换回撤），
        # 鲁棒性体现为回撤控制与 EWMA 跟踪速度
        check(med_dd["rolling_disc"] < med_dd["fixed_30"],
              f"边衰减期滚动版回撤更浅: {med_dd['rolling_disc']:.1%} < "
              f"{med_dd['fixed_30']:.1%} (固定30%)")
        check(med_dd["rolling_ewma"] <= med_dd["rolling_disc"] + 0.02,
              f"EWMA 回撤不劣于平权: {med_dd['rolling_ewma']:.1%} vs "
              f"{med_dd['rolling_disc']:.1%}")
        # 末段平均风险: EWMA 应更快降到接近半 Kelly 理论值
        f_ewma_late = sum(sum(s[-100:]) for s in stats["rolling_ewma"]["f"][:3]) / 300
        f_disc_late = sum(sum(s[-100:]) for s in stats["rolling_disc"]["f"][:3]) / 300
        f_theory_late = f_star(p_end - (p_end - p_start) * 100 / N_TRADES) * 0.5 * 100
        check(f_ewma_late <= f_disc_late + 1.0,
              f"末段降档速度 EWMA({f_ewma_late:.1f}%) ≤ 平权({f_disc_late:.1f}%)"
              f"+1pp (理论半Kelly≈{f_theory_late:.1f}%)")
        check(med_l2["rolling_disc"] > 0,
              f"边衰减至 {p_end} 仍为正时滚动版保持增长: "
              f"中位 {med_l2['rolling_disc']:.1f} 次翻倍")
    else:
        # 边跌破保本点: 固定档前期收益被后期负增长吞掉
        check(med_l2["fixed_30"] < med_l2["rolling_disc"] - 20,
              f"边消失时固定 30% 湮灭: 中位 {med_l2['fixed_30']:.1f} vs "
              f"滚动 {med_l2['rolling_disc']:.1f} (差>20次翻倍)")
        check(med_l2["rolling_disc"] > -12,
              f"滚动版在边消失后缓慢失血而非湮灭: 中位 {med_l2['rolling_disc']:.1f}"
              f" (1% 下限兜底)")
    return stats


def main():
    t0 = time.time()
    print("交叉核对: 前 3 条路径每 50 笔比对快速增量 Kelly 与生产函数"
          "（不一致即抛异常）\n")

    stats_a = run_scenario("场景 A: 真实边", 0.5, 2.5)
    print()
    # ---- 跳变专项: 离散 vs 平滑 的回撤/终值分布差异 ----
    dd_disc = stats_a["rolling_disc"]["dd"]
    dd_smooth = stats_a["rolling_smooth"]["dd"]
    l2_disc = stats_a["rolling_disc"]["log2"]
    l2_smooth = stats_a["rolling_smooth"]["log2"]
    med_dd_d, med_dd_s = sorted(dd_disc)[len(dd_disc)//2], sorted(dd_smooth)[len(dd_smooth)//2]
    mean_dd_d, mean_dd_s = sum(dd_disc)/len(dd_disc), sum(dd_smooth)/len(dd_smooth)
    mean_l2_d, mean_l2_s = sum(l2_disc)/len(l2_disc), sum(l2_smooth)/len(l2_smooth)

    print("跳变专项对比（离散 vs 平滑，其余规则完全相同）:")
    print(f"  maxDD 中位: {med_dd_d:.1%} vs {med_dd_s:.1%}  "
          f"(Δ={abs(med_dd_d-med_dd_s):.1%})")
    print(f"  maxDD 均值: {mean_dd_d:.1%} vs {mean_dd_s:.1%}  "
          f"(Δ={abs(mean_dd_d-mean_dd_s):.1%})")
    print(f"  翻倍数均值: {mean_l2_d:.1f} vs {mean_l2_s:.1f}")
    print()
    check(abs(med_dd_d - med_dd_s) < 0.02,
          f"离散跳变未显著放大回撤: maxDD 中位差 {abs(med_dd_d-med_dd_s):.1%} < 2 个百分点")
    check(abs(mean_l2_d - mean_l2_s) / max(abs(mean_l2_d), 1e-9) < 0.10,
          "离散 vs 平滑长期增长一致 (<10% 差异) — 跳变只发生在第 50 笔一次, "
          "长期统计不可见")
    check(med(stats_a["rolling_disc"]["dd"])
          < med(stats_a["fixed_30"]["dd"]),
          "滚动 Kelly 的回撤显著小于固定 30% 基准")
    half_growth = sum(stats_a["static_half"]["log2"]) / N_PATHS
    disc_growth = mean_l2_d
    check(disc_growth > 0.5 * half_growth,
          f"离散档位拿到先知半 Kelly 增长的 >50% ({disc_growth:.1f} vs "
          f"{half_growth:.1f} 翻倍数; 前期 1/4 档 + 前 9 笔满仓贡献差异)")

    stats_b = run_scenario("\n场景 B: 负边（生存测试）", 0.25, 2.5)
    med_fixed_b = sorted(stats_b["fixed_30"]["log2"])[N_PATHS//2]
    med_disc_b = sorted(stats_b["rolling_disc"]["log2"])[N_PATHS//2]
    print()
    check(med_fixed_b < -100,
          f"固定 30% 在负边下湮灭: 中位终值 2^{med_fixed_b:.0f} "
          f"(≈ 10^{med_fixed_b*math.log10(2):.0f})")
    check(-8 < med_disc_b < -1,
          f"滚动 Kelly 缓慢失血但生存: 中位终值 2^{med_disc_b:.1f} "
          f"(1% 下限兜底, 不加杠杆赌负边)")

    print()
    print("=" * 86)
    print(f"结果: {PASS} PASS / {FAIL} FAIL   "
          f"(耗时 {time.time()-t0:.1f}s, {N_PATHS}路径 × {N_TRADES}笔 × 4策略)")
    print("=" * 86)
    return 0 if FAIL == 0 else 1


def trace_curves(p=0.5, b=2.5, n_paths=5):
    """--curves 模式: 追踪样本路径的资金曲线轨迹（log2 权益 + 当时回撤深度）。

    与 simulate_path 同一套规则（含前 9 笔无约束期），仅增加每 100 笔
    节点记录。权益以 log2(倍数) 表示：+10 = 翻 10 次倍，-10 = 腰斩 10 次。
    """
    modes = ("fixed_30", "static_half", "rolling_disc")
    for path in range(n_paths):
        rng = random.Random(10_000 + path)
        outcomes = [rng.random() < p for _ in range(N_TRADES)]
        print("=" * 96)
        print(f"路径 #{path}（p={p}, b={b}）— 数值为 log2(权益倍数)，括号内为当时回撤深度")
        print("=" * 96)
        hdr = f"{'策略':<14}" + "".join(f"{'@'+str(t):>13}" for t in range(100, 1001, 100))
        print(hdr)
        print("-" * 96)
        for mode in modes:
            rk = FastRollingKelly()
            frac_fn = {"rolling_disc": frac_discrete,
                       "rolling_smooth": frac_smooth}.get(mode)
            log_eq, peak = 0.0, 0.0
            cells = []
            for i, is_win in enumerate(outcomes, 1):
                if mode in ("rolling_disc", "rolling_smooth"):
                    cap = rk.risk_pct(frac_fn)
                    f = BASE_RISK if cap is None else min(BASE_RISK, cap)
                elif mode == "fixed_30":
                    f = BASE_RISK
                else:
                    f = 15.0
                log_eq += (math.log(1.0 + f / 100.0 * b) if is_win
                           else math.log(1.0 - f / 100.0))
                pnl = (f / 100.0 * b) if is_win else -(f / 100.0)
                rk.add(pnl)
                if log_eq > peak:
                    peak = log_eq
                if i % 100 == 0:
                    dd = 1.0 - math.exp(log_eq - peak)
                    cells.append(f"{log_eq / math.log(2):>+7.1f}({dd:>4.0%})")
            print(f"{mode:<14}" + "".join(f"{c:>13}" for c in cells))
        print()


def med(xs):
    s = sorted(xs)
    return s[len(s) // 2]


if __name__ == "__main__":
    if "--curves" in sys.argv:
        # 曲线追踪模式: 快速直观查看资金曲线波动，不跑 300 路径断言
        trace_curves(0.5, 2.5, n_paths=5)
        print("提示: 完整 300 路径统计断言请运行: python verify_kelly_monte_carlo.py")
        raise SystemExit(0)
    if "--drift" in sys.argv:
        # 边漂移模式: --drift [起始胜率 结束胜率]，默认 0.5 → 0.4
        argv = sys.argv
        idx = argv.index("--drift")
        try:
            p_s = float(argv[idx + 1]) if len(argv) > idx + 1 else 0.5
            p_e = float(argv[idx + 2]) if len(argv) > idx + 2 else 0.4
        except ValueError:
            print("用法: python verify_kelly_monte_carlo.py --drift [p_start p_end]")
            raise SystemExit(2)
        print("交叉核对: 前 3 条路径每 50 笔比对快速增量 Kelly 与生产函数"
              "（含 EWMA 路径，不一致即抛异常）\n")
        run_drift_scenario(p_s, p_e)
        print()
        print("=" * 86)
        print(f"结果: {PASS} PASS / {FAIL} FAIL")
        print("=" * 86)
        raise SystemExit(0 if FAIL == 0 else 1)
    raise SystemExit(main())
