"""
完整回测引擎 — 融合 Vibe-Trading 的回测 + 优化器 + 相关性分析。

借鉴 Vibe-Trading (23.6k⭐) 的 backtest 系统：
  - 相关性分析: 跨资产 Pearson/Spearman 相关矩阵 + 滚动窗口
  - 体制感知回测: 基于 FUSED/DIVERGENT/NEUTRAL 体制的绩效分解
  - 组合优化器: 等波动率 / 风险平价 / 最大分散化 / 均值-方差
  - 绩效指标: Sharpe, Sortino, Calmar, 最大回撤, 胜率, 盈亏比等

HunHeng_OS_V1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ===========================================================================
# 数据类型
# ===========================================================================

@dataclass
class BacktestTrade:
    """回测交易记录。"""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str = "long"       # "long" | "short"
    quantity: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_win: bool = False
    regime: str = "NEUTRAL"       # 入场时的体制
    fvg_score: float = 0.0
    holding_bars: int = 0


@dataclass
class BacktestMetrics:
    """回测绩效指标。"""
    # 收益
    total_return: float = 0.0
    annual_return: float = 0.0
    total_pnl: float = 0.0

    # 风险
    volatility: float = 0.0          # 年化波动率
    max_drawdown: float = 0.0        # 最大回撤 (%)
    max_drawdown_duration: int = 0   # 最大回撤持续期 (bar)
    var_95: float = 0.0              # 95% VaR
    cvar_95: float = 0.0             # 95% CVaR

    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_holding_bars: float = 0.0

    # 极值
    best_trade: float = 0.0
    worst_trade: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # 按体制分组
    regime_metrics: Dict[str, dict] = field(default_factory=dict)

    # 时间序列
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)


# ===========================================================================
# 相关性分析引擎
# ===========================================================================

class CorrelationAnalyzer:
    """跨资产相关性分析引擎。

    借鉴 Vibe-Trading 的 correlation.py 实现：
      - 滚动 Pearson/Spearman 相关矩阵
      - 边密度 (edge density) 计算
      - 相关性网络分析
    """

    def __init__(self, window: int = 90, method: str = "pearson"):
        self.window = window
        self.method = method

    def compute_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
    ) -> Tuple[List[str], np.ndarray]:
        """计算相关性矩阵。

        Args:
            returns_df: 多资产收益 DataFrame，列为资产代码

        Returns:
            (labels, matrix) — labels 为排序后的资产代码列表，matrix 为 N×N 相关矩阵
        """
        codes = sorted(returns_df.columns.tolist())
        n = len(codes)

        if n < 2:
            return codes, np.array([[1.0]])

        # 使用最近 window 条数据
        aligned = returns_df.dropna()
        if len(aligned) > self.window:
            aligned = aligned.iloc[-self.window:]

        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                xi = aligned[codes[i]].values
                xj = aligned[codes[j]].values
                valid = ~(np.isnan(xi) | np.isnan(xj))
                if valid.sum() < 2:
                    corr = 0.0
                elif self.method == "spearman":
                    corr, _ = spearmanr(xi[valid], xj[valid])
                else:
                    corr = np.corrcoef(xi[valid], xj[valid])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                matrix[i, j] = round(corr, 4)
                matrix[j, i] = round(corr, 4)

        return codes, matrix

    def compute_rolling_correlation(
        self,
        returns_df: pd.DataFrame,
        pair: Tuple[str, str],
    ) -> pd.Series:
        """计算资产对的滚动相关性。

        Args:
            returns_df: 多资产收益 DataFrame
            pair: (asset_a, asset_b)

        Returns:
            滚动相关性序列
        """
        a, b = pair
        if a not in returns_df.columns or b not in returns_df.columns:
            return pd.Series(dtype=float)

        aligned = returns_df[[a, b]].dropna()
        if len(aligned) < self.window:
            return pd.Series(dtype=float)

        rolling_corr = aligned[a].rolling(self.window).corr(aligned[b])
        return rolling_corr.dropna()

    def compute_edge_density(
        self,
        returns_df: pd.DataFrame,
        edge_threshold: float = 0.5,
    ) -> pd.Series:
        """计算边密度序列 — 高相关资产对的占比。

        Args:
            returns_df: 多资产收益 DataFrame
            edge_threshold: |ρ| 阈值

        Returns:
            边密度序列 [0, 1]
        """
        from regime_detector import compute_edge_density
        return compute_edge_density(
            returns_df,
            corr_window=self.window,
            edge_threshold=edge_threshold,
        )

    def find_highly_correlated_pairs(
        self,
        returns_df: pd.DataFrame,
        threshold: float = 0.7,
    ) -> List[Tuple[str, str, float]]:
        """找出高相关资产对。

        Returns:
            [(asset_a, asset_b, correlation), ...] 按相关性降序
        """
        labels, matrix = self.compute_correlation_matrix(returns_df)
        pairs = []
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(matrix[i, j]) >= threshold:
                    pairs.append((labels[i], labels[j], float(matrix[i, j])))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs


# ===========================================================================
# 组合优化器
# ===========================================================================

class PortfolioOptimizer:
    """组合优化器 — 借鉴 Vibe-Trading 的 optimizers 包。

    支持五种加权方案:
      - equal_weight: 等权 (1/N)
      - equal_volatility: 逆波动率加权
      - risk_parity: 风险平价 (等风险贡献)
      - max_diversification: 最大分散化
      - mean_variance: 最大 Sharpe (均值-方差优化)
    """

    def __init__(self, method: str = "equal_volatility"):
        self.method = method

    def optimize(
        self,
        returns_df: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """计算最优权重。

        Args:
            returns_df: 多资产收益 DataFrame
            **kwargs: 方法特定参数

        Returns:
            权重 Series，索引为资产代码
        """
        if returns_df.empty or returns_df.shape[1] == 0:
            return pd.Series(dtype=float)

        methods = {
            "equal_weight": self._equal_weight,
            "equal_volatility": self._equal_volatility,
            "risk_parity": self._risk_parity,
            "max_diversification": self._max_diversification,
            "mean_variance": self._mean_variance,
        }

        optimizer = methods.get(self.method, self._equal_weight)
        return optimizer(returns_df, **kwargs)

    def _equal_weight(self, returns_df: pd.DataFrame, **kwargs) -> pd.Series:
        n = returns_df.shape[1]
        weights = pd.Series(1.0 / n, index=returns_df.columns)
        return weights / weights.sum()

    def _equal_volatility(self, returns_df: pd.DataFrame, **kwargs) -> pd.Series:
        vol = returns_df.std()
        inv_vol = 1.0 / vol.replace(0, np.nan)
        weights = inv_vol / inv_vol.sum()
        return weights.fillna(1.0 / len(weights))

    def _risk_parity(self, returns_df: pd.DataFrame, **kwargs) -> pd.Series:
        """风险平价 — 等风险贡献 (ERC)。"""
        cov = returns_df.cov().to_numpy()
        n = cov.shape[0]

        if n == 0:
            return pd.Series(dtype=float)

        def _erc_objective(w: np.ndarray) -> float:
            w = np.abs(w) / np.abs(w).sum()
            portfolio_vol = np.sqrt(w @ cov @ w)
            marginal_contrib = cov @ w
            risk_contrib = w * marginal_contrib / portfolio_vol
            target_contrib = portfolio_vol / n
            return float(np.sum((risk_contrib - target_contrib) ** 2))

        x0 = np.ones(n) / n
        bounds = [(1e-6, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]

        try:
            result = minimize(_erc_objective, x0, bounds=bounds, constraints=constraints, method="SLSQP")
            if result.success:
                w = np.abs(result.x) / np.abs(result.x).sum()
                return pd.Series(w, index=returns_df.columns)
        except Exception as e:
            logger.debug("Risk parity optimization failed: %s", e)

        return self._equal_volatility(returns_df)

    def _max_diversification(self, returns_df: pd.DataFrame, **kwargs) -> pd.Series:
        """最大分散化 — 最大化分散化比率。"""
        cov = returns_df.cov().to_numpy()
        vol = returns_df.std().to_numpy()
        n = cov.shape[0]

        if n == 0:
            return pd.Series(dtype=float)

        def _objective(w: np.ndarray) -> float:
            w = np.abs(w) / np.abs(w).sum()
            portfolio_vol = np.sqrt(w @ cov @ w)
            weighted_vol = w @ vol
            if weighted_vol < 1e-10:
                return 1e10
            return float(-weighted_vol / portfolio_vol)

        x0 = np.ones(n) / n
        bounds = [(1e-6, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]

        try:
            result = minimize(_objective, x0, bounds=bounds, constraints=constraints, method="SLSQP")
            if result.success:
                w = np.abs(result.x) / np.abs(result.x).sum()
                return pd.Series(w, index=returns_df.columns)
        except Exception as e:
            logger.debug("Max diversification optimization failed: %s", e)

        return self._equal_volatility(returns_df)

    def _mean_variance(self, returns_df: pd.DataFrame, **kwargs) -> pd.Series:
        """均值-方差优化 — 最大化 Sharpe 比率。"""
        cov = returns_df.cov().to_numpy()
        mu = returns_df.mean().to_numpy()
        n = cov.shape[0]
        risk_free = kwargs.get("risk_free_rate", 0.0)

        if n == 0:
            return pd.Series(dtype=float)

        def _objective(w: np.ndarray) -> float:
            w = np.abs(w) / np.abs(w).sum()
            portfolio_return = w @ mu - risk_free
            portfolio_vol = np.sqrt(w @ cov @ w)
            if portfolio_vol < 1e-10:
                return -1e10
            return float(-portfolio_return / portfolio_vol)

        x0 = np.ones(n) / n
        bounds = [(1e-6, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]

        try:
            result = minimize(_objective, x0, bounds=bounds, constraints=constraints, method="SLSQP")
            if result.success:
                w = np.abs(result.x) / np.abs(result.x).sum()
                return pd.Series(w, index=returns_df.columns)
        except Exception as e:
            logger.debug("Mean-variance optimization failed: %s", e)

        return self._equal_volatility(returns_df)


# ===========================================================================
# 回测绩效计算
# ===========================================================================

class BacktestAnalyzer:
    """回测绩效分析器。

    计算全套绩效指标，支持按体制分组分析。
    """

    # 年化因子映射
    _TRADING_DAYS: Dict[str, int] = {"crypto": 365, "equity": 252, "forex": 260}
    _BARS_PER_DAY: Dict[str, int] = {
        "1H": 24, "4H": 6, "1D": 1, "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
    }

    def __init__(self, market: str = "crypto", interval: str = "1H"):
        self.market = market
        self.interval = interval

    @property
    def bars_per_year(self) -> int:
        trading_days = self._TRADING_DAYS.get(self.market, 365)
        bars_per_day = self._BARS_PER_DAY.get(self.interval, 24)
        return trading_days * bars_per_day

    def compute_metrics(
        self,
        trades: List[BacktestTrade],
        equity_curve: pd.Series,
        initial_capital: float = 10000.0,
    ) -> BacktestMetrics:
        """计算全套回测绩效指标。

        Args:
            trades: 交易记录列表
            equity_curve: 权益曲线 (index=datetime)
            initial_capital: 初始资金

        Returns:
            BacktestMetrics
        """
        metrics = BacktestMetrics()

        if equity_curve.empty:
            return metrics

        # ---- 收益 ----
        metrics.total_pnl = float(equity_curve.iloc[-1] - initial_capital)
        metrics.total_return = metrics.total_pnl / initial_capital

        # 日收益
        daily_eq = equity_curve.resample("D").last().ffill()
        daily_returns = daily_eq.pct_change().dropna()
        metrics.daily_returns = daily_returns

        if len(daily_returns) > 0:
            n_years = len(daily_returns) / 365
            if n_years > 0:
                metrics.annual_return = (1 + metrics.total_return) ** (1 / n_years) - 1

        # ---- 风险 ----
        if len(daily_returns) > 1:
            metrics.volatility = float(daily_returns.std() * np.sqrt(365))
            metrics.var_95 = float(np.percentile(daily_returns, 5))
            metrics.cvar_95 = float(daily_returns[daily_returns <= metrics.var_95].mean())

        # 最大回撤
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax.replace(0, 1e-10)
        metrics.max_drawdown = float(abs(drawdown.min())) if not drawdown.empty else 0.0
        metrics.drawdown_curve = drawdown

        # 最大回撤持续期
        if not drawdown.empty:
            dd_mask = drawdown < 0
            max_dur = 0
            current_dur = 0
            for is_dd in dd_mask:
                if is_dd:
                    current_dur += 1
                    max_dur = max(max_dur, current_dur)
                else:
                    current_dur = 0
            metrics.max_drawdown_duration = max_dur

        # ---- 风险调整收益 ----
        if metrics.volatility > 0 and len(daily_returns) > 0:
            risk_free_daily = 0.02 / 365  # 假设 2% 无风险利率
            metrics.sharpe_ratio = float(
                (daily_returns.mean() - risk_free_daily) / daily_returns.std() * np.sqrt(365)
            )

        if len(daily_returns) > 1:
            downside = daily_returns[daily_returns < 0]
            if len(downside) > 1 and downside.std() > 0:
                metrics.sortino_ratio = float(
                    daily_returns.mean() / downside.std() * np.sqrt(365)
                )

        if metrics.max_drawdown > 0 and not np.isnan(metrics.annual_return):
            metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown

        # ---- 交易统计 ----
        # 修复 Bug 50: 胜率统计基于 pnl 符号，保本交易单独统计，
        # 与 EdgeAnalyzer/StateManager 保持一致
        if trades:
            metrics.total_trades = len(trades)
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl < 0]
            break_evens = [t for t in trades if t.pnl == 0]
            metrics.winning_trades = len(wins)
            metrics.losing_trades = len(losses)
            metrics.break_even_trades = len(break_evens)
            decisive = metrics.winning_trades + metrics.losing_trades
            metrics.win_rate = metrics.winning_trades / decisive if decisive > 0 else 0.0

            metrics.avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
            metrics.avg_loss = abs(float(np.mean([t.pnl for t in losses]))) if losses else 0.0

            total_profit = sum(t.pnl for t in wins)
            total_loss = abs(sum(t.pnl for t in losses))
            metrics.profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

            metrics.expectancy = (
                metrics.win_rate * metrics.avg_win -
                (1 - metrics.win_rate) * metrics.avg_loss
            )

            metrics.best_trade = max(t.pnl for t in trades)
            metrics.worst_trade = min(t.pnl for t in trades)

            metrics.avg_holding_bars = float(np.mean([t.holding_bars for t in trades]))

            # 连赢/连亏（修复 Bug 48: 统一以 pnl 符号为准）
            cons_wins = 0
            cons_losses = 0
            max_cons_wins = 0
            max_cons_losses = 0
            for t in trades:
                if t.pnl > 0:
                    cons_wins += 1
                    cons_losses = 0
                    max_cons_wins = max(max_cons_wins, cons_wins)
                elif t.pnl < 0:
                    cons_losses += 1
                    cons_wins = 0
                    max_cons_losses = max(max_cons_losses, cons_losses)
                # pnl == 0 保本：不增加任何连赢/连亏计数
            metrics.max_consecutive_wins = max_cons_wins
            metrics.max_consecutive_losses = max_cons_losses

            # 按体制分组
            metrics.regime_metrics = self._compute_regime_metrics(trades)

        metrics.equity_curve = equity_curve

        return metrics

    def _compute_regime_metrics(
        self,
        trades: List[BacktestTrade],
    ) -> Dict[str, dict]:
        """按体制分组的绩效指标。"""
        regime_trades: Dict[str, List[BacktestTrade]] = {}
        for t in trades:
            regime_trades.setdefault(t.regime, []).append(t)

        metrics = {}
        for regime, rt in regime_trades.items():
            # 修复 Bug 65: 统一以 pnl 符号分类盈亏
            wins = [t for t in rt if t.pnl > 0]
            losses = [t for t in rt if t.pnl < 0]
            # 修复 Bug 43: 标记小样本（N<5）的 regime 为 low_sample，
            # 提示下游做策略对比时谨慎对待，避免被 N=1 偶然的 100% 胜率误导
            low_sample = len(rt) < 5
            decisive = len(wins) + len(losses)
            metrics[regime] = {
                "total_trades": len(rt),
                "win_rate": len(wins) / decisive if decisive > 0 else 0.0,
                "total_pnl": sum(t.pnl for t in rt),
                "avg_pnl": float(np.mean([t.pnl for t in rt])) if rt else 0.0,
                "avg_win": float(np.mean([t.pnl for t in wins])) if wins else 0.0,
                "avg_loss": abs(float(np.mean([t.pnl for t in losses]))) if losses else 0.0,
                "best_trade": max(t.pnl for t in rt) if rt else 0.0,
                "worst_trade": min(t.pnl for t in rt) if rt else 0.0,
                "low_sample": low_sample,
            }

        return metrics

    def format_report(self, metrics: BacktestMetrics) -> str:
        """格式化绩效报告。"""
        lines = [
            "",
            "╔" + "═" * 58 + "╗",
            "║  📊 回测绩效报告".ljust(61) + "║",
            "╠" + "═" * 58 + "╣",
            f"║  总收益率:     {metrics.total_return:>+8.2%}".ljust(61) + "║",
            f"║  年化收益率:   {metrics.annual_return:>+8.2%}".ljust(61) + "║",
            f"║  年化波动率:   {metrics.volatility:>8.2%}".ljust(61) + "║",
            f"║  最大回撤:     {metrics.max_drawdown:>8.2%}".ljust(61) + "║",
            "╠" + "═" * 58 + "╣",
            f"║  Sharpe:       {metrics.sharpe_ratio:>8.2f}".ljust(61) + "║",
            f"║  Sortino:      {metrics.sortino_ratio:>8.2f}".ljust(61) + "║",
            f"║  Calmar:       {metrics.calmar_ratio:>8.2f}".ljust(61) + "║",
            "╠" + "═" * 58 + "╣",
            f"║  总交易:       {metrics.total_trades:>8d}".ljust(61) + "║",
            f"║  胜率:         {metrics.win_rate:>8.1%}".ljust(61) + "║",
            f"║  盈亏比:       {metrics.profit_factor:>8.2f}".ljust(61) + "║",
            f"║  期望值:       {metrics.expectancy:>+8.2f}".ljust(61) + "║",
            f"║  最大连亏:     {metrics.max_consecutive_losses:>8d}".ljust(61) + "║",
            "╠" + "═" * 58 + "╣",
        ]

        # 按体制分组
        if metrics.regime_metrics:
            lines.append("║  按体制分组:".ljust(61) + "║")
            for regime, rm in sorted(metrics.regime_metrics.items()):
                lines.append(
                    f"║    {regime:<12} 胜率={rm['win_rate']:>5.1%}  "
                    f"PnL={rm['total_pnl']:>+8.2f}".ljust(61) + "║"
                )

        lines.append("╚" + "═" * 58 + "╝")
        return "\n".join(lines)


# ===========================================================================
# 回测运行器
# ===========================================================================

class BacktestRunner:
    """回测运行器 — 事件驱动的 FVG 策略回测。

    用法:
        runner = BacktestRunner(
            initial_capital=10000,
            leverage=3,
            risk_per_trade=0.01,
        )
        runner.set_data(candles_dict)
        metrics = runner.run()
        print(runner.format_report())
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 3,
        risk_per_trade: float = 0.01,
        commission: float = 0.0005,  # 0.05%
        slippage: float = 0.0001,    # 0.01%
        market: str = "crypto",
        interval: str = "1H",
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage

        self.analyzer = BacktestAnalyzer(market=market, interval=interval)
        self.correlation_analyzer = CorrelationAnalyzer()
        self.optimizer = PortfolioOptimizer(method="equal_volatility")

        self._candles: Dict[str, pd.DataFrame] = {}
        self._trades: List[BacktestTrade] = []
        self._equity_curve: pd.Series = pd.Series(dtype=float)
        self._regime_series: pd.Series = pd.Series(dtype=str)

    def set_data(self, candles_dict: Dict[str, pd.DataFrame]):
        """设置回测数据。

        Args:
            candles_dict: {inst_id: DataFrame(columns=[open,high,low,close,volume])}
        """
        self._candles = candles_dict

    def add_trade(self, trade: BacktestTrade):
        """添加交易记录。"""
        self._trades.append(trade)

    def set_equity_curve(self, equity: pd.Series, regime: pd.Series | None = None):
        """设置权益曲线和体制序列。"""
        self._equity_curve = equity
        if regime is not None:
            self._regime_series = regime

    def run(self) -> BacktestMetrics:
        """运行回测分析。"""
        return self.analyzer.compute_metrics(
            trades=self._trades,
            equity_curve=self._equity_curve,
            initial_capital=self.initial_capital,
        )

    def analyze_correlation(self, returns_df: pd.DataFrame) -> Dict:
        """运行相关性分析。

        Returns:
            {"matrix": ..., "labels": ..., "edge_density": ..., "high_corr_pairs": ...}
        """
        labels, matrix = self.correlation_analyzer.compute_correlation_matrix(returns_df)
        edge_density = self.correlation_analyzer.compute_edge_density(returns_df)
        high_pairs = self.correlation_analyzer.find_highly_correlated_pairs(returns_df)

        return {
            "labels": labels,
            "matrix": matrix.tolist(),
            "edge_density": edge_density.dropna().tolist() if not edge_density.empty else [],
            "high_corr_pairs": high_pairs,
        }

    def optimize_weights(
        self,
        returns_df: pd.DataFrame,
        method: str = "equal_volatility",
    ) -> pd.Series:
        """运行组合优化。

        Args:
            returns_df: 多资产收益 DataFrame
            method: 优化方法

        Returns:
            最优权重 Series
        """
        self.optimizer.method = method
        return self.optimizer.optimize(returns_df)

    def format_report(self, metrics: Optional[BacktestMetrics] = None) -> str:
        """格式化回测报告。"""
        if metrics is None:
            metrics = self.run()
        return self.analyzer.format_report(metrics)


# ===========================================================================
# 便捷函数
# ===========================================================================

def compute_correlation_matrix(
    returns_df: pd.DataFrame,
    window: int = 90,
    method: str = "pearson",
) -> Tuple[List[str], np.ndarray]:
    """便捷函数：计算相关性矩阵。"""
    analyzer = CorrelationAnalyzer(window=window, method=method)
    return analyzer.compute_correlation_matrix(returns_df)


def compute_edge_density(
    returns_df: pd.DataFrame,
    window: int = 60,
    edge_threshold: float = 0.5,
) -> pd.Series:
    """便捷函数：计算边密度。"""
    analyzer = CorrelationAnalyzer(window=window)
    return analyzer.compute_edge_density(returns_df, edge_threshold=edge_threshold)


def optimize_portfolio(
    returns_df: pd.DataFrame,
    method: str = "equal_volatility",
) -> pd.Series:
    """便捷函数：组合优化。"""
    optimizer = PortfolioOptimizer(method=method)
    return optimizer.optimize(returns_df)