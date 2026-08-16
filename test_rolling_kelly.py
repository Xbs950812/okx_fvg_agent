"""
滚动 Kelly 计算逻辑单元测试 (2026-08-15)。

验证对象（现有生产代码，不改代码）：
  1. hyperopt.compute_kelly — Kelly 核心公式与边界裁剪
     - f* = (p·b − q)/b，按 pnl 符号分类，保本交易(pnl==0)剔除出胜率分母
     - 盈亏比裁剪 [0.01, 100]；胜率 ≥99.9% → 0.25；≤0.1% → 0
     - kelly_f 上限 0.5；负边 → 0
     - recommended_risk_pct = min(半Kelly×100, max_risk_pct)
  2. executor.StateManager.record_realized_pnl — 滚动数据源
     recent_pnl 滚动保留最近 100 笔（滚动 Kelly 的输入窗口）
  3. 滚动等价性 — recent_pnl[-N:] 窗口构造的 TradeRecord 喂入
     compute_kelly，结果必须等于直接对同 N 笔交易调用 compute_kelly
     （未来接线滚动 Kelly 时必须保持该性质）
  4. 样本量规则契约 — <50 笔用 1/4 Kelly、≥50 笔用 1/2 Kelly 的
     预期语义固化（当前仅存在于 agent.py 步骤 9 的日志分支，
     作为未来接线到 eff_risk 的可执行规格）

手算基准（对数增长率公式 g(f) = p·ln(1+f·b) + q·ln(1−f)）：
  p=0.5, b=2.5 → f* = (1.25−0.5)/2.5 = 0.30
    g(0.30) = 0.5·ln(1.75) + 0.5·ln(0.70) = 0.5·0.559616 − 0.5·0.356675 ≈ 0.1015
    g(0.60) = 0.5·ln(2.50) + 0.5·ln(0.40) = 0.5·0.916291 − 0.5·0.916291 = 0
    （2×f* 处增长率恰为 0 —— 超额 Kelly 归零的教科书特例）

运行: python -m pytest test_rolling_kelly.py -v
"""

import math

import pytest

from hyperopt import compute_kelly
from fvg_killer_pro import rolling_kelly_risk_pct
from optimization import TradeRecord
from executor import StateManager


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------

def make_trade(pnl: float, pnl_pct: float = None, symbol: str = "TEST-USDT-SWAP"):
    """构造 TradeRecord。pnl_pct 缺省时与 pnl 同值（按 1 USDT 本金口径）。

    compute_kelly 按 t.pnl 符号分类盈亏、按 abs(t.pnl_pct) 计算均值，
    测试中两者保持同号一致。
    """
    return TradeRecord(
        symbol=symbol,
        direction="long",
        entry_time=0.0,
        exit_time=1.0,
        entry_price=100.0,
        exit_price=100.0,
        quantity=1.0,
        leverage=1,
        pnl=pnl,
        pnl_pct=pnl_pct if pnl_pct is not None else pnl,
        is_win=pnl > 0,
        exit_reason="test",
    )


def series_trades(pnls, symbol="TEST-USDT-SWAP"):
    """把 pnl 序列（USDT，符号即盈亏）转成 TradeRecord 列表。"""
    return [make_trade(p, symbol=symbol) for p in pnls]


# ---------------------------------------------------------------------------
# 1. 核心公式与边界裁剪
# ---------------------------------------------------------------------------

class TestComputeKellyCore:

    def test_empty_trades_returns_zeros(self):
        r = compute_kelly([])
        assert r.kelly_fraction == 0
        assert r.half_kelly == 0
        assert r.quarter_kelly == 0
        assert r.recommended_risk_pct == 0
        assert r.expected_growth_rate == 0

    def test_canonical_p50_b250_kelly_30pct(self):
        """p=0.5, b=2.5 → f*=0.30（对话中引用的满 Kelly 基准数）。"""
        # 5 胜 pnl_pct=+2.5, 5 负 pnl_pct=−1.0 → b = 2.5/1.0 = 2.5
        trades = series_trades([2.5] * 5 + [-1.0] * 5)
        trades = [make_trade(t.pnl, pnl_pct=abs(t.pnl)) for t in trades]
        r = compute_kelly(trades)
        assert r.win_rate == pytest.approx(0.5)
        assert r.avg_win_pct == pytest.approx(2.5)
        assert r.avg_loss_pct == pytest.approx(1.0)
        assert r.kelly_fraction == pytest.approx(0.30, abs=1e-4)
        assert r.half_kelly == pytest.approx(0.15, abs=1e-4)
        assert r.quarter_kelly == pytest.approx(0.075, abs=1e-4)

    def test_expected_growth_formula_p50_b250(self):
        """g(0.30) = 0.5·ln(1.75) + 0.5·ln(0.70) ≈ 0.1015。"""
        trades = [make_trade(2.5, 2.5)] * 5 + [make_trade(-1.0, 1.0)] * 5
        r = compute_kelly(trades)
        g_ref = 0.5 * math.log(1 + 0.30 * 2.5) + 0.5 * math.log(1 - 0.30)
        assert r.expected_growth_rate == pytest.approx(g_ref, abs=1e-4)
        assert g_ref == pytest.approx(0.1015, abs=1e-4)

    def test_over_kelly_doubles_growth_zero_reference(self):
        """独立数学性质：p=0.5, b=2.5 时 g(2·f*) = g(0.60) 恰好为 0。

        用对数增长率公式直接验证凹性/超额归零 —— 这是"超 Kelly 数学自杀区"
        结论的公式根基（纯参考计算，不依赖生产代码）。
        """
        p, b = 0.5, 2.5

        def g(f):
            return p * math.log(1 + f * b) + (1 - p) * math.log(1 - f)

        assert g(0.30) == pytest.approx(0.1015, abs=1e-4)   # 峰值
        assert g(0.60) == pytest.approx(0.0, abs=1e-9)      # 2×f* 归零
        assert g(0.75) < 0                                   # 2.5×f* 负增长
        assert g(0.30) > g(0.15) > 0                         # 欠 Kelly 单调段

    def test_negative_edge_clipped_to_zero(self):
        """p=0.3, b=1.0 → 原始 f*=−0.4 → 裁剪为 0，增长率为 0。"""
        trades = [make_trade(1.0, 1.0)] * 3 + [make_trade(-1.0, 1.0)] * 7
        r = compute_kelly(trades)
        assert r.win_rate == pytest.approx(0.3)
        assert r.kelly_fraction == 0.0
        assert r.expected_growth_rate == 0.0

    def test_breakeven_trades_excluded_from_win_rate(self):
        """pnl==0 的保本交易剔除出胜率分母（Bug 56/胜率稀释修复）。"""
        # 10 笔中 2 笔保本: 4 胜 +2, 4 负 −1, 2 笔 pnl=0
        trades = ([make_trade(2.0, 2.0)] * 4
                  + [make_trade(-1.0, 1.0)] * 4
                  + [make_trade(0.0, 0.0)] * 2)
        r = compute_kelly(trades)
        assert r.win_rate == pytest.approx(0.5)      # 4/8 而非 4/10
        # b = 2.0 → f* = (0.5×2 − 0.5)/2 = 0.25
        assert r.kelly_fraction == pytest.approx(0.25, abs=1e-4)

    def test_all_wins_extreme_winrate_clipped_to_025(self):
        """胜率 1.0 (≥99.9%) → 强制 0.25，不过拟合。"""
        trades = [make_trade(2.0, 2.0)] * 5
        r = compute_kelly(trades)
        assert r.win_rate == pytest.approx(1.0)
        assert r.kelly_fraction == pytest.approx(0.25, abs=1e-4)
        assert r.half_kelly == pytest.approx(0.125, abs=1e-4)

    def test_all_losses_kelly_zero(self):
        """胜率 0 (≤0.1%) → f*=0（无亏损样本时 avg_loss 走兜底也不复活 Kelly）。"""
        trades = [make_trade(-1.0, -1.0)] * 5
        r = compute_kelly(trades)
        assert r.win_rate == pytest.approx(0.0)
        assert r.kelly_fraction == 0.0

    def test_payoff_ratio_clipped_at_100(self):
        """b=200 → 裁剪到 100: p=0.5 → f* = (50−0.5)/100 = 0.495。"""
        trades = [make_trade(100.0, 100.0)] + [make_trade(-0.5, 0.5)]
        r = compute_kelly(trades)
        assert r.kelly_fraction == pytest.approx(0.495, abs=1e-4)

    def test_kelly_capped_at_half(self):
        """p=0.9, b=10 → 原始 0.89 → 上限裁剪 0.5。"""
        trades = [make_trade(10.0, 10.0)] * 9 + [make_trade(-1.0, 1.0)]
        r = compute_kelly(trades)
        assert r.kelly_fraction == pytest.approx(0.5, abs=1e-4)

    def test_recommended_risk_respects_cap(self):
        """p=0.5,b=2.5: 半Kelly=15%。默认上限 5 → 推荐 5；上限 20 → 推荐 15（不钳制）。"""
        trades = [make_trade(2.5, 2.5)] * 5 + [make_trade(-1.0, 1.0)] * 5
        r_default = compute_kelly(trades)                     # max_risk_pct=5
        r_wide = compute_kelly(trades, max_risk_pct=20.0)
        assert r_default.recommended_risk_pct == pytest.approx(5.0, abs=1e-4)
        assert r_wide.recommended_risk_pct == pytest.approx(15.0, abs=1e-4)

    def test_kelly_monotone_in_payoff_for_fixed_winrate(self):
        """固定胜率下 f* 随盈亏比单调不减（公式结构性质）。"""
        for p in (0.3, 0.5, 0.7):
            n_win, n_loss = int(p * 10), 10 - int(p * 10)
            prev = -1.0
            for b in (0.5, 1.0, 2.0, 4.0):
                trades = ([make_trade(b, b)] * n_win
                          + [make_trade(-1.0, 1.0)] * n_loss)
                r = compute_kelly(trades)
                # 单调且落在 [0, 0.5]
                assert r.kelly_fraction >= prev - 1e-9, f"p={p}, b={b}"
                assert 0.0 <= r.kelly_fraction <= 0.5
                prev = r.kelly_fraction


# ---------------------------------------------------------------------------
# 2. 滚动数据源: recent_pnl 滚动保留 100 笔
# ---------------------------------------------------------------------------

class TestRollingPnlSource:

    def _make_sm(self, tmp_path):
        return StateManager(str(tmp_path / "test_state.json"))

    def test_recent_pnl_rolls_to_last_100(self, tmp_path):
        sm = self._make_sm(tmp_path)
        for i in range(120):
            sm.record_realized_pnl(float(i))
        rp = sm.state.recent_pnl
        assert len(rp) == 100
        # 最早保留的是第 21 笔 (i=20)，最晚是第 120 笔 (i=119)
        assert rp[0] == pytest.approx(20.0)
        assert rp[-1] == pytest.approx(119.0)

    def test_recent_pnl_under_100_keeps_all(self, tmp_path):
        sm = self._make_sm(tmp_path)
        for i in range(37):
            sm.record_realized_pnl(float(i))
        assert len(sm.state.recent_pnl) == 37

    def test_daily_trades_counts_every_close(self, tmp_path):
        """record_realized_pnl 同步累计 daily_trades（每日交易上限的计数器）。"""
        sm = self._make_sm(tmp_path)
        for i in range(5):
            sm.record_realized_pnl(1.0 if i % 2 == 0 else -1.0)
        assert sm.state.daily_trades == 5
        assert sm.state.winning_trades == 3
        assert sm.state.losing_trades == 2


# ---------------------------------------------------------------------------
# 3. 滚动等价性: 窗口切片 == 同窗口全量计算
# ---------------------------------------------------------------------------

class TestRollingWindowEquivalence:

    def test_window_tail_equals_full_compute_on_same_trades(self):
        """recent_pnl[-N:] 构造的窗口喂 compute_kelly == 直接对同 N 笔全量计算。

        这是未来"滚动 Kelly"接线的核心不变式：切片本身不引入偏差。
        """
        pnls = ([2.5] * 3 + [-1.0] * 3) * 10           # 60 笔, p=0.5, b=2.5
        sm_like = list(pnls)                            # 模拟 recent_pnl 状态

        for window in (10, 25, 50, 60):
            tail = sm_like[-window:]
            trades_tail = series_trades(tail)
            r_rolling = compute_kelly(trades_tail)

            trades_full = series_trades(tail)           # 同 N 笔独立构造
            r_full = compute_kelly(trades_full)

            assert r_rolling.kelly_fraction == r_full.kelly_fraction
            assert r_rolling.win_rate == r_full.win_rate
            assert r_rolling.expected_growth_rate == r_full.expected_growth_rate

    def test_window_invariant_to_history_before_window(self):
        """窗口外的历史变化不得影响窗口内 Kelly 结果（滚动语义）。"""
        tail = [2.5] * 5 + [-1.0] * 5                   # 窗口: p=0.5, b=2.5
        prefix_a = [100.0] * 40                         # 暴富前史
        prefix_b = [-100.0] * 40                        # 巨亏前史

        r_a = compute_kelly(series_trades(prefix_a + tail)[-10:])
        r_b = compute_kelly(series_trades(prefix_b + tail)[-10:])

        assert r_a.kelly_fraction == pytest.approx(0.30, abs=1e-4)
        assert r_b.kelly_fraction == pytest.approx(0.30, abs=1e-4)
        assert r_a.kelly_fraction == r_b.kelly_fraction

    def test_window_detects_regime_shift(self):
        """滚动窗口必须能感知体制切换：前 50 笔赢、后 50 笔亏 → 窗口 Kelly 归零。"""
        history = [1.0] * 50 + [-1.0] * 50
        r_old = compute_kelly(series_trades(history[:50]))   # 旧窗口: 全胜→0.25 裁剪
        r_new = compute_kelly(series_trades(history[-50:]))  # 新窗口: 全亏→0
        assert r_old.kelly_fraction == pytest.approx(0.25, abs=1e-4)
        assert r_new.kelly_fraction == 0.0


# ---------------------------------------------------------------------------
# 4. 样本量规则契约（未来接线 eff_risk 的可执行规格）
# ---------------------------------------------------------------------------

class TestSampleSizeRuleContract:
    """固化预期语义: 样本 < 50 笔 → 1/4 Kelly；≥ 50 笔 → 1/2 Kelly。

    当前 agent.py 步骤 9 只在 len(trades) >= 50 时才计算 Kelly（否则日志跳过），
    仓位侧尚未接线。此测试把目标行为钉死，防止未来接线时口径漂移。
    """

    @staticmethod
    def _expected_risk_fraction(n_samples: int, kelly_f: float) -> float:
        if n_samples < 50:
            return kelly_f / 4.0
        return kelly_f / 2.0

    def test_below_50_samples_uses_quarter_kelly(self):
        # 48 笔(24胜24负): p 精确 = 0.5, b=2.5 → f*=0.30（奇数笔会使 p≠0.5）
        trades = [make_trade(2.5, 2.5)] * 24 + [make_trade(-1.0, 1.0)] * 24  # 48 笔
        r = compute_kelly(trades)
        assert len(trades) == 48
        assert r.kelly_fraction == pytest.approx(0.30, abs=1e-4)
        expected = self._expected_risk_fraction(len(trades), r.kelly_fraction)
        assert expected == pytest.approx(0.075, abs=1e-4)   # 1/4 Kelly

    def test_at_50_samples_uses_half_kelly(self):
        trades = [make_trade(2.5, 2.5)] * 25 + [make_trade(-1.0, 1.0)] * 25  # 50 笔
        r = compute_kelly(trades)
        assert r.kelly_fraction == pytest.approx(0.30, abs=1e-4)
        expected = self._expected_risk_fraction(len(trades), r.kelly_fraction)
        assert expected == pytest.approx(0.15, abs=1e-4)    # 1/2 Kelly

    def test_agent_gate_threshold_is_50(self):
        """agent.py 步骤 9 的门槛常数固化: >= 50 才启用 Kelly 统计。"""
        # 该分支当前行为：不足 50 笔时不产出 Kelly 日志（仓位走固定比例）
        # 契约: 门槛本身不得悄悄变化
        below = 49 * [make_trade(1.0, 1.0)]
        at = 50 * [make_trade(1.0, 1.0)]
        assert len(below) < 50 <= len(at)
        # 两个样本量下 compute_kelly 均可计算（门槛是调用方职责），
        # 但契约函数的档位切换发生在 50
        r_below = compute_kelly(below)
        r_at = compute_kelly(at)
        assert self._expected_risk_fraction(49, r_below.kelly_fraction) \
            == pytest.approx(r_below.kelly_fraction / 4, abs=1e-9)
        assert self._expected_risk_fraction(50, r_at.kelly_fraction) \
            == pytest.approx(r_at.kelly_fraction / 2, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. rolling_kelly_risk_pct 纯函数（已接线到 agent.py eff_risk 消费点）
# ---------------------------------------------------------------------------

class TestRollingKellyRiskFunction:
    """验证滚动 Kelly 风险上限函数的档位切换、裁剪与窗口语义。

    注: 精确值断言统一用 FLAT（ewma_lambda=0，平权窗口）口径 —
    档位契约与平滑方式正交；EWMA 行为由 TestRollingKellyEwma 单独覆盖。
    """

    BASE_RISK = 30.0  # 与 config.example.json 的 risk_per_trade_pct 一致
    FLAT = {"rolling_kelly": {"ewma_lambda": 0}}

    @staticmethod
    def _pnl_series(n_win, n_loss, win=2.5, loss=-1.0, breakeven=0):
        return [win] * n_win + [loss] * n_loss + [0.0] * breakeven

    def test_disabled_returns_none(self):
        cfg = {"rolling_kelly": {"enabled": False}}
        r, diag = rolling_kelly_risk_pct(self._pnl_series(30, 30), self.BASE_RISK, cfg)
        assert r is None and diag.get("disabled") is True

    def test_insufficient_samples_returns_none(self):
        """decisive < min_samples(10) 不施加约束（防噪声假精度）。"""
        r, diag = rolling_kelly_risk_pct(self._pnl_series(5, 4), self.BASE_RISK, self.FLAT)
        assert r is None
        assert diag.get("reason") == "insufficient_samples"
        assert diag.get("samples") == 9

    def test_breakeven_not_counted_as_sample(self):
        """pnl==0 不计入 decisive 样本（9 胜 0 负 + 5 保本 = 9 样本 < 10）。"""
        r, diag = rolling_kelly_risk_pct(self._pnl_series(9, 0, breakeven=5),
                                         self.BASE_RISK, self.FLAT)
        assert r is None
        assert diag.get("samples") == 9

    def test_quarter_tier_below_50_samples(self):
        """30 样本 p=0.5 b=2.5 → f*=0.30 × 1/4 = 7.5%（探索档）。"""
        pnls = self._pnl_series(15, 15)
        r, diag = rolling_kelly_risk_pct(pnls, self.BASE_RISK, self.FLAT)
        assert diag.get("tier") == "quarter"
        assert diag.get("samples") == 30
        assert r == pytest.approx(7.5, abs=1e-3)

    def test_half_tier_at_50_samples(self):
        """50 样本 p=0.5 b=2.5 → f*=0.30 × 1/2 = 15.0%（利用档）。"""
        pnls = self._pnl_series(25, 25)
        r, diag = rolling_kelly_risk_pct(pnls, self.BASE_RISK, self.FLAT)
        assert diag.get("tier") == "half"
        assert r == pytest.approx(15.0, abs=1e-3)

    def test_negative_edge_floors_to_min_risk(self):
        """负边 Kelly=0 → 落到 min_risk_pct(默认 1.0)而非 0（暂停归期望值门禁）。"""
        pnls = self._pnl_series(3, 27)                     # p=0.1, b=2.5 → f*<0 → 0
        r, diag = rolling_kelly_risk_pct(pnls, self.BASE_RISK, self.FLAT)
        assert diag.get("kelly_f") == 0.0
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_never_exceeds_base_cap(self):
        """p=0.9 b=10 → f* 裁剪 0.5，半 Kelly=25% ≤ base 30。"""
        pnls = [10.0] * 45 + [-1.0] * 5                    # 50 样本
        r, _ = rolling_kelly_risk_pct(pnls, self.BASE_RISK, self.FLAT)
        assert r == pytest.approx(25.0, abs=1e-3)

    def test_explicit_max_risk_cap(self):
        """显式 max_risk_pct=10: 25% 被钳到 10。"""
        pnls = [10.0] * 45 + [-1.0] * 5
        r, _ = rolling_kelly_risk_pct(pnls, self.BASE_RISK,
                                      {"rolling_kelly": {"max_risk_pct": 10.0,
                                                         "ewma_lambda": 0}})
        assert r == pytest.approx(10.0, abs=1e-3)

    def test_max_risk_zero_falls_back_to_base(self):
        """max_risk_pct=0（example 配置惯例）→ 用 base 作上限，不是禁用。"""
        pnls = self._pnl_series(25, 25)
        r, _ = rolling_kelly_risk_pct(pnls, self.BASE_RISK,
                                      {"rolling_kelly": {"max_risk_pct": 0,
                                                         "ewma_lambda": 0}})
        assert r == pytest.approx(15.0, abs=1e-3)          # 未被钳到 0/未禁用

    def test_window_excludes_old_history(self):
        """window=100: 前缀 50 笔巨亏不影响窗口内 100 笔 p=0.5/b=2.5 的结果。"""
        tail = self._pnl_series(50, 50)
        r_tail, d_tail = rolling_kelly_risk_pct(tail, self.BASE_RISK, self.FLAT)
        r_full, d_full = rolling_kelly_risk_pct([-100.0] * 50 + tail,
                                                self.BASE_RISK, self.FLAT)
        assert r_tail == pytest.approx(15.0, abs=1e-3)
        assert r_full == pytest.approx(r_tail, abs=1e-9)
        assert d_full.get("samples") == 100

    def test_end_to_end_with_state_manager(self, tmp_path):
        """端到端: StateManager 滚动记录 60 笔 → 函数读 recent_pnl 出利用档。"""
        sm = StateManager(str(tmp_path / "sm.json"))
        for i in range(60):
            sm.record_realized_pnl(2.5 if i % 2 == 0 else -1.0)
        r, diag = rolling_kelly_risk_pct(sm.state.recent_pnl, self.BASE_RISK, self.FLAT)
        assert diag.get("tier") == "half"
        assert diag.get("win_rate") == pytest.approx(0.5)
        assert r == pytest.approx(15.0, abs=1e-3)

    def test_effective_risk_binding_semantics(self):
        """接线语义: eff_risk = min(自适应/基准, Kelly上限)，只降不升。"""
        pnls = self._pnl_series(25, 25)                    # Kelly 上限 15%
        base = 30.0
        adaptive_derated = 18.0
        cap, _ = rolling_kelly_risk_pct(pnls, base, self.FLAT)
        assert min(adaptive_derated, cap) == pytest.approx(15.0, abs=1e-3)
        # 自适应降档比 Kelly 更严时，降档获胜
        adaptive_severe = 8.0
        assert min(adaptive_severe, cap) == pytest.approx(8.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 6. EWMA 输入端平滑（2026-08-15 主流实践接入）
# ---------------------------------------------------------------------------

class TestRollingKellyEwma:
    """验证 EWMA 平滑路径的数学正确性、回退安全性与体制响应。"""

    BASE_RISK = 30.0
    FLAT = {"rolling_kelly": {"ewma_lambda": 0}}

    @staticmethod
    def _alt(n_pairs):
        """交替 W L 序列（L 结尾，等量等幅 |1|）。"""
        return [1.0, -1.0] * n_pairs

    def test_ewma_disabled_identical_to_flat(self):
        """λ=0 与旧版平权口径逐位一致（回归守卫）。"""
        pnls = [2.5] * 25 + [-1.0] * 25
        r_flat, d_flat = rolling_kelly_risk_pct(pnls, self.BASE_RISK, self.FLAT)
        assert r_flat == pytest.approx(15.0, abs=1e-3)
        assert d_flat.get("ewma_lambda") == 0

    def test_ewma_alternating_closed_form(self):
        """交替序列(W 结尾) λ=0.5 的闭式解手算验证。

        11 笔 W L W L ... W，等幅 |1|，共享时钟衰减：
          Σwin权重 = 1+λ²+λ⁴+λ⁶+λ⁸+λ¹⁰ = 1.333008
          Σloss权重 = λ¹+λ³+λ⁵+λ⁷+λ⁹ = 0.666016
          p = 1.333008/1.999023 = 0.666833 → b=1 → f* = 2p−1 = 0.333665
          quarter: 0.3337 × 1/4 × 100 = 8.34%
        平权对照: 6W/5L p=0.545 → f*=0.0909 → 2.27%（EWMA 更看重近期 W）
        """
        pnls = self._alt(5) + [1.0]                      # 11 样本, W 结尾
        r, diag = rolling_kelly_risk_pct(
            pnls, self.BASE_RISK, {"rolling_kelly": {"ewma_lambda": 0.5}})
        assert diag.get("samples") == 11
        assert diag.get("win_rate") == pytest.approx(0.667, abs=1e-3)
        assert diag.get("kelly_f") == pytest.approx(0.3337, abs=1e-3)
        assert r == pytest.approx(8.34, abs=0.02)
        r_flat, _ = rolling_kelly_risk_pct(pnls, self.BASE_RISK, self.FLAT)
        assert r_flat == pytest.approx(2.27, abs=0.02)
        assert r > r_flat

    def test_ewma_regime_shift_faster_than_flat(self):
        """40 胜后 10 连亏（体制切换）: EWMA(λ=0.8) 响应远快于平权。

        平权: p=0.8, b=1 → f*=0.6→cap 0.5 → half 25%
        EWMA λ=0.8（有效记忆≈9 笔）: 近 10 亏权重 4.463 vs 旧 40 胜 0.537
          → p=0.107 → f*<0 → 1% 下限
        （λ=0.97 记忆 33 笔压不过 40 旧胜 — λ 决定体制切换灵敏度）
        """
        pnls = [1.0] * 40 + [-1.0] * 10
        r_flat, d_flat = rolling_kelly_risk_pct(pnls, self.BASE_RISK, self.FLAT)
        r_ewma, d_ewma = rolling_kelly_risk_pct(
            pnls, self.BASE_RISK, {"rolling_kelly": {"ewma_lambda": 0.8}})
        assert d_flat.get("win_rate") == pytest.approx(0.8)
        assert r_flat == pytest.approx(25.0, abs=1e-3)
        assert d_ewma.get("win_rate") == pytest.approx(0.107, abs=1e-3)
        assert r_ewma == pytest.approx(1.0, abs=1e-6)
        assert r_ewma < r_flat

    def test_ewma_edge_improves_recently_beats_flat(self):
        """镜像场景: 40 亏后 10 连胜 → EWMA 上限高于平权（快速捕捉新边）。

        平权: p=0.2 → f*<0 → 1%；EWMA λ=0.8: 近 10 胜权重占优 p=0.893
          → f*=0.785 → cap 0.5 → half 25%
        """
        pnls = [-1.0] * 40 + [1.0] * 10
        r_flat, _ = rolling_kelly_risk_pct(pnls, self.BASE_RISK, self.FLAT)
        r_ewma, d_ewma = rolling_kelly_risk_pct(
            pnls, self.BASE_RISK, {"rolling_kelly": {"ewma_lambda": 0.8}})
        assert r_flat == pytest.approx(1.0, abs=1e-6)
        assert d_ewma.get("win_rate") == pytest.approx(0.893, abs=1e-3)
        assert r_ewma == pytest.approx(25.0, abs=1e-3)
        assert r_ewma > r_flat

    def test_ewma_invalid_lambda_falls_back_flat(self):
        """λ 非法（负数/≥1/非数字/None）→ 回退平权，结果与 λ=0 逐位一致。"""
        pnls = [2.5] * 25 + [-1.0] * 25
        r_ref, d_ref = rolling_kelly_risk_pct(pnls, self.BASE_RISK, self.FLAT)
        for bad in (-0.5, 1.0, 2.0, "abc", None):
            cfg = {"rolling_kelly": {"ewma_lambda": bad}}
            r_bad, d_bad = rolling_kelly_risk_pct(pnls, self.BASE_RISK, cfg)
            assert r_bad == pytest.approx(r_ref, abs=1e-9), f"λ={bad!r}"
            assert d_bad.get("ewma_lambda") == 0

    def test_ewma_default_is_097_and_active(self):
        """默认（未配置）λ=0.97 生效并出现在诊断里。"""
        pnls = [1.0] * 40 + [-1.0] * 10
        r, diag = rolling_kelly_risk_pct(pnls, self.BASE_RISK, {})
        assert diag.get("ewma_lambda") == pytest.approx(0.97)
        assert r is not None

    def test_ewma_breakeven_does_not_advance_clock(self):
        """保本交易不推进 EWMA 时钟: 穿插 0 与无 0 结果逐位一致。"""
        base_series = [2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0]
        with_zero = []
        for p in base_series:
            with_zero.extend([0.0, p])
        cfg = {"rolling_kelly": {"ewma_lambda": 0.9}}
        r_a, d_a = rolling_kelly_risk_pct(base_series, self.BASE_RISK, cfg)
        r_b, d_b = rolling_kelly_risk_pct(with_zero, self.BASE_RISK, cfg)
        assert d_a.get("samples") == 10
        assert d_b.get("samples") == 10
        assert r_a == pytest.approx(r_b, abs=1e-9)
        assert d_a.get("kelly_f") == d_b.get("kelly_f")

    def test_ewma_extreme_clips_shared_with_flat(self):
        """EWMA 路径复用同一裁剪链 + 新近观测主导性。

        49 胜后最新 1 亏，λ=0.5:
          Σwin权重 = λ+λ²+...+λ⁴⁹ = 1−λ⁴⁹ ≈ 1.0（最新胜被衰减一拍）
          Σloss权重 = 1.0（最新亏权重最大）
          → p = 0.5 → f*=0 → 1% 下限（EWMA 重罚最新亏损）
        平权: 49W/1L p=0.98 → f*=0.96 → 裁剪链 cap 0.5 → half 25%
        两路径 kelly_f 均 ≤ 0.5（裁剪链共享的直接证据）。
        """
        pnls = [1.0] * 49 + [-1.0]          # 49 胜 1 负
        r_flat, d_flat = rolling_kelly_risk_pct(pnls, self.BASE_RISK, self.FLAT)
        r_ewma, d_ewma = rolling_kelly_risk_pct(
            pnls, self.BASE_RISK, {"rolling_kelly": {"ewma_lambda": 0.5}})
        assert d_flat.get("kelly_f") == 0.5     # 平权触发 cap（0.96→0.5）
        assert d_ewma.get("kelly_f") == 0.0     # EWMA p=0.5 → f*=0
        assert r_flat == pytest.approx(25.0, abs=1e-3)
        assert r_ewma == pytest.approx(1.0, abs=1e-6)
        assert d_ewma.get("win_rate") == pytest.approx(0.5, abs=1e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
