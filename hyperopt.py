"""
参数优化引擎 — 借鉴 freqtrade (52k ⭐) 的 Hyperopt + FreqAI 框架。

freqtrade 核心设计：
  - Hyperopt: 参数空间搜索 + 损失函数优化 (基于 Hyperopt 库)
  - FreqAI: 自适应特征工程 + 在线学习 + 强化学习
  - Edge: 胜率/期望值/盈亏比统计
  - Backtesting: 完整回测框架

本模块实现：
  1. Advanced Hyperopt — 贝叶斯启发式参数搜索 + 自适应网格细化
  2. Walk-Forward Optimization — 滚动窗口优化 + OOS 验证
  3. Kelly Criterion — 最优仓位计算
  4. Strategy Dashboard — 综合性能指标
  5. FreqAI Pipeline — 简化版在线学习流水线
  6. Sensitivity Analysis — 参数敏感性分析
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from typing import Optional, List, Dict, Tuple, Callable, Any

import numpy as np


logger = logging.getLogger(__name__)


# ===========================================================================
# TradeRecord (复用于 optimization.py)
# ===========================================================================

@dataclass
class TradeRecord:
    """单笔交易记录。"""
    symbol: str
    direction: str
    entry_time: float
    exit_time: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    leverage: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_win: bool = False
    exit_reason: str = ""
    fvg_score: float = 0.0
    master_score: float = 0.0
    regime: str = ""


# ===========================================================================
# 参数空间定义
# ===========================================================================

@dataclass
class ParamSpace:
    """参数搜索空间 — 借鉴 freqtrade Hyperopt 的 SPACE 定义。"""
    name: str
    type: str                       # "int" | "float" | "choice"
    low: float = 0.0
    high: float = 1.0
    choices: List[Any] = field(default_factory=list)
    step: float = 0.1
    default: Any = None

    def sample(self, n: int = 10) -> List[Any]:
        """在参数空间内采样。"""
        if self.type == "choice":
            return self.choices[:n]
        elif self.type == "int":
            values = np.linspace(int(self.low), int(self.high), min(n, int(self.high - self.low) + 1))
            return [int(v) for v in values]
        else:  # float
            return list(np.linspace(self.low, self.high, n))


# 默认参数空间（与 FVG 策略相关）
DEFAULT_PARAM_SPACE = [
    ParamSpace(name="min_fvg_width_1h", type="float", low=0.5, high=4.0, step=0.25, default=1.5),
    ParamSpace(name="min_fvg_width_4h", type="float", low=1.0, high=6.0, step=0.5, default=3.0),
    ParamSpace(name="fvg_target_pct", type="float", low=0.30, high=0.70, step=0.05, default=0.50),
    ParamSpace(name="stop_buffer_pct", type="float", low=0.05, high=0.30, step=0.05, default=0.15),
    ParamSpace(name="entry_depth_pct", type="float", low=0.05, high=0.30, step=0.05, default=0.15),
    ParamSpace(name="abnormal_sigma", type="float", low=2.0, high=4.0, step=0.25, default=3.0),
    ParamSpace(name="abnormal_volume_ratio", type="float", low=3.0, high=8.0, step=0.5, default=5.0),
    ParamSpace(name="max_leverage", type="int", low=3, high=20, default=10),
    ParamSpace(name="risk_per_trade_pct", type="float", low=0.5, high=3.0, step=0.25, default=1.0),
    ParamSpace(name="min_confidence", type="float", low=0.30, high=0.60, step=0.05, default=0.40),
]


# ===========================================================================
# 性能指标
# ===========================================================================

@dataclass
class StrategyMetrics:
    """策略性能指标 — 借鉴 freqtrade 的 backtesting 分析。"""
    # 基础指标
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # 收益指标
    total_return_pct: float = 0.0
    avg_return_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0

    # 风险指标
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0      # 最大回撤持续交易数
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_ratio: float = 0.0

    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # 稳定性
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # 时间
    avg_holding_hours: float = 0.0
    trades_per_day: float = 0.0

    # 评分
    composite_score: float = 0.0        # 综合评分 (0-100)


def compute_metrics(
    trades: List[TradeRecord],
    initial_equity: float = 1000.0,
    risk_free_rate: float = 0.02,
) -> StrategyMetrics:
    """计算策略性能指标。

    借鉴 freqtrade 的 backtesting 分析模块，计算：
      - 基础盈亏指标
      - 风险调整收益 (Sharpe, Sortino, Calmar)
      - 稳定性指标 (连赢/连亏)
      - 综合评分
    """
    m = StrategyMetrics()

    if not trades:
        return m

    m.total_trades = len(trades)
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0

    # 收益指标
    total_return = sum(t.pnl for t in trades)
    m.total_return_pct = (total_return / initial_equity * 100) if initial_equity > 0 else 0
    m.avg_return_pct = np.mean([t.pnl_pct for t in trades]) if trades else 0
    m.best_trade_pct = max(t.pnl_pct for t in trades)
    m.worst_trade_pct = min(t.pnl_pct for t in trades)

    # 盈利因子
    total_profit = sum(t.pnl for t in wins)
    total_loss = abs(sum(t.pnl for t in losses))
    m.profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # 期望值
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0
    m.expectancy = (m.win_rate * avg_win - (1 - m.win_rate) * avg_loss)
    m.expectancy_ratio = m.expectancy / avg_loss if avg_loss > 0 else 0

    # 最大回撤
    cum_returns = np.array([t.pnl_pct for t in trades])
    cum_eq = initial_equity * np.cumprod(1 + cum_returns / 100)
    peak = np.maximum.accumulate(cum_eq)
    dd = (peak - cum_eq) / peak * 100
    m.max_drawdown_pct = float(np.max(dd))

    # 最大回撤持续期
    in_dd = False
    dd_start = 0
    max_dd_duration = 0
    for i, d in enumerate(dd):
        if d > 0 and not in_dd:
            in_dd = True
            dd_start = i
        elif d == 0 and in_dd:
            in_dd = False
            max_dd_duration = max(max_dd_duration, i - dd_start)
    if in_dd:
        max_dd_duration = max(max_dd_duration, len(dd) - dd_start)
    m.max_drawdown_duration = max_dd_duration

    # 风险调整收益
    returns = np.array([t.pnl_pct / 100 for t in trades])
    if len(returns) > 1 and np.std(returns) > 0:
        m.sharpe_ratio = (np.mean(returns) - risk_free_rate / 365) / np.std(returns) * np.sqrt(365)

        # Sortino (只考虑下行波动)
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 1 else np.std(returns)
        m.sortino_ratio = (np.mean(returns) - risk_free_rate / 365) / downside_std * np.sqrt(365) \
            if downside_std > 0 else 0

        # Calmar
        m.calmar_ratio = (m.total_return_pct / 100) / (m.max_drawdown_pct / 100) \
            if m.max_drawdown_pct > 0 else 0

    # 连赢/连亏
    cons_wins = 0
    cons_losses = 0
    max_cons_wins = 0
    max_cons_losses = 0
    for t in trades:
        if t.is_win:
            cons_wins += 1
            cons_losses = 0
            max_cons_wins = max(max_cons_wins, cons_wins)
        else:
            cons_losses += 1
            cons_wins = 0
            max_cons_losses = max(max_cons_losses, cons_losses)
    m.consecutive_wins = cons_wins
    m.consecutive_losses = cons_losses
    m.max_consecutive_wins = max_cons_wins
    m.max_consecutive_losses = max_cons_losses

    # 持仓时间
    holding_times = [t.exit_time - t.entry_time for t in trades
                     if t.exit_time > 0 and t.entry_time > 0]
    if holding_times:
        m.avg_holding_hours = np.mean(holding_times) / 3600.0
        total_days = (max(t.exit_time for t in trades) -
                      min(t.entry_time for t in trades)) / 86400.0
        m.trades_per_day = m.total_trades / total_days if total_days > 0 else 0

    # 综合评分 (0-100)
    m.composite_score = _compute_composite_score(m)

    return m


def _compute_composite_score(m: StrategyMetrics) -> float:
    """计算综合评分 — 多维度加权。"""
    score = 0.0

    # 胜率 (20%)
    score += min(m.win_rate * 100, 100) * 0.20

    # 盈亏比 (20%)
    if m.profit_factor > 0 and m.profit_factor < float("inf"):
        score += min(m.profit_factor * 20, 100) * 0.20

    # 期望值 (20%)
    score += min(max(m.expectancy_ratio * 20, 0), 100) * 0.20

    # Sharpe (15%)
    score += min(max(m.sharpe_ratio * 25, 0), 100) * 0.15

    # 回撤控制 (15%)
    if m.max_drawdown_pct > 0:
        score += max(0, (1 - m.max_drawdown_pct / 50) * 100) * 0.15

    # 样本量 (10%)
    score += min(m.total_trades / 50 * 100, 100) * 0.10

    return min(score, 100)


# ===========================================================================
# 高级 Hyperopt（贝叶斯启发式）
# ===========================================================================

@dataclass
class HyperoptResult:
    """超参优化结果。"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)
    optimization_time: float = 0.0
    param_importance: Dict[str, float] = field(default_factory=dict)


class BayesianHyperopt:
    """贝叶斯启发式参数优化器 — 借鉴 freqtrade Hyperopt。

    不使用外部 Hyperopt 库，而是实现：
      1. 粗粒度网格搜索 → 确定有希望区域
      2. 自适应细化 → 在最优区域附近精细搜索
      3. 参数重要性分析 → 识别关键参数
    """

    def __init__(
        self,
        param_space: List[ParamSpace],
        objective_fn: Callable[[Dict[str, Any]], float],
        n_initial: int = 5,
        n_refine: int = 3,
        refine_factor: float = 0.3,
    ):
        self.param_space = {p.name: p for p in param_space}
        self.objective_fn = objective_fn
        self.n_initial = n_initial
        self.n_refine = n_refine
        self.refine_factor = refine_factor

    def optimize(self) -> HyperoptResult:
        """执行优化。"""
        t0 = time.time()
        all_results = []

        # ---- Phase 1: 粗粒度网格搜索 ----
        logger.info("Phase 1: 粗粒度网格搜索...")
        grid = self._build_initial_grid()
        for params in grid:
            try:
                score = self.objective_fn(params)
                all_results.append((params.copy(), score))
            except Exception as e:
                logger.debug(f"Params {params} failed: {e}")

        if not all_results:
            logger.error("No valid parameter combinations found!")
            return HyperoptResult(best_params={}, best_score=0.0)

        all_results.sort(key=lambda x: x[1], reverse=True)

        # ---- Phase 2: 自适应细化 ----
        best_params, best_score = all_results[0]
        logger.info(f"Phase 1 最优: score={best_score:.4f}, params={best_params}")

        for refine_round in range(self.n_refine):
            logger.info(f"Phase 2: 细化轮次 {refine_round + 1}/{self.n_refine}...")
            refined_grid = self._build_refined_grid(best_params)
            for params in refined_grid:
                try:
                    score = self.objective_fn(params)
                    all_results.append((params.copy(), score))
                except Exception as e:
                    logger.debug(f"Refined params {params} failed: {e}")

            all_results.sort(key=lambda x: x[1], reverse=True)
            best_params, best_score = all_results[0]

            # 如果不再改善，提前退出
            if refine_round > 0 and all_results[0][1] <= best_score * 1.001:
                break

        # ---- Phase 3: 参数重要性分析 ----
        param_importance = self._analyze_param_importance(all_results)

        elapsed = time.time() - t0
        logger.info(f"优化完成 in {elapsed:.1f}s: best_score={best_score:.4f}")

        return HyperoptResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=elapsed,
            param_importance=param_importance,
        )

    def _build_initial_grid(self) -> List[Dict[str, Any]]:
        """构建初始搜索网格。"""
        param_values = {}
        for name, p in self.param_space.items():
            values = p.sample(self.n_initial)
            # 确保包含默认值
            if p.default is not None and p.default not in values:
                values.append(p.default)
            param_values[name] = values

        # 笛卡尔积
        names = list(param_values.keys())
        combos = list(product(*[param_values[n] for n in names]))
        return [dict(zip(names, combo)) for combo in combos]

    def _build_refined_grid(self, best_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """在最优参数附近构建细化网格。"""
        param_values = {}
        for name, p in self.param_space.items():
            current = best_params.get(name, p.default or 0)
            if p.type == "choice":
                param_values[name] = p.choices[:3]
            elif p.type == "int":
                low = max(p.low, int(current - (p.high - p.low) * self.refine_factor))
                high = min(p.high, int(current + (p.high - p.low) * self.refine_factor))
                values = list(range(low, high + 1, max(1, (high - low) // 3)))
                param_values[name] = values if values else [int(current)]
            else:  # float
                rng = (p.high - p.low) * self.refine_factor
                low = max(p.low, current - rng)
                high = min(p.high, current + rng)
                values = list(np.linspace(low, high, 4))
                param_values[name] = values

        names = list(param_values.keys())
        combos = list(product(*[param_values[n] for n in names]))
        return [dict(zip(names, combo)) for combo in combos]

    def _analyze_param_importance(
        self,
        all_results: List[Tuple[Dict[str, Any], float]],
    ) -> Dict[str, float]:
        """分析参数重要性 — 通过固定其他参数，变化单个参数观察得分变化。"""
        if len(all_results) < 5:
            return {}

        importance = {}
        scores = np.array([s for _, s in all_results])

        for name in self.param_space:
            param_values = []
            for params, _ in all_results:
                if name in params:
                    param_values.append(params[name])

            if len(set(param_values)) > 1:
                # 计算参数值与得分的相关性
                numeric_values = []
                for v in param_values:
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        numeric_values.append(0.0)

                if len(set(numeric_values)) > 1:
                    corr = abs(np.corrcoef(numeric_values, scores)[0, 1])
                    importance[name] = round(float(corr), 3)
                else:
                    importance[name] = 0.0
            else:
                importance[name] = 0.0

        # 归一化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance


# ===========================================================================
# Walk-Forward 优化
# ===========================================================================

@dataclass
class WalkForwardResult:
    """Walk-Forward 优化结果。"""
    in_sample_metrics: List[StrategyMetrics] = field(default_factory=list)
    out_of_sample_metrics: List[StrategyMetrics] = field(default_factory=list)
    overall_metrics: Optional[StrategyMetrics] = None
    robustness_score: float = 0.0     # OOS/IS 一致性评分
    best_params_per_window: List[Dict[str, Any]] = field(default_factory=list)
    is_overfitting: bool = False


def walk_forward_optimization(
    all_trades: List[TradeRecord],
    param_space: List[ParamSpace],
    objective_fn: Callable[[List[TradeRecord], Dict[str, Any]], float],
    n_windows: int = 5,
    train_ratio: float = 0.6,
) -> WalkForwardResult:
    """Walk-Forward 优化 — 借鉴 freqtrade backtesting 框架。

    将数据分为多个窗口，每窗口：
      - 训练集 (前 60%): 参数优化
      - 测试集 (后 40%): OOS 验证
    窗口滚动前进，评估策略稳健性。
    """
    result = WalkForwardResult()

    if len(all_trades) < n_windows * 2:
        logger.warning("交易记录不足，无法执行 Walk-Forward 优化")
        return result

    total = len(all_trades)
    window_size = total // n_windows
    train_size = int(window_size * train_ratio)
    test_size = window_size - train_size

    hyperopt = BayesianHyperopt(param_space, lambda p: 0.0, n_initial=4, n_refine=2)

    for w in range(n_windows - 1):
        start = w * window_size
        train_end = start + train_size
        test_end = min(train_end + test_size, total)

        if test_end <= train_end:
            break

        train_trades = all_trades[start:train_end]
        test_trades = all_trades[train_end:test_end]

        if not train_trades or not test_trades:
            continue

        # 在训练集上优化
        def obj_wrapper(params):
            return objective_fn(train_trades, params)

        hyperopt.objective_fn = obj_wrapper
        opt_result = hyperopt.optimize()
        result.best_params_per_window.append(opt_result.best_params)

        # 评估训练集
        is_metrics = compute_metrics(train_trades)
        result.in_sample_metrics.append(is_metrics)

        # 评估测试集 (OOS)
        oos_metrics = compute_metrics(test_trades)
        result.out_of_sample_metrics.append(oos_metrics)

        logger.info(f"Window {w+1}: IS={is_metrics.composite_score:.1f}, "
                     f"OOS={oos_metrics.composite_score:.1f}")

    # 整体评估
    if result.out_of_sample_metrics:
        result.overall_metrics = compute_metrics(all_trades)

        # 稳健性评分: OOS/IS 的一致性
        if result.in_sample_metrics:
            is_scores = [m.composite_score for m in result.in_sample_metrics]
            oos_scores = [m.composite_score for m in result.out_of_sample_metrics]
            if len(is_scores) == len(oos_scores) and np.mean(is_scores) > 0:
                # 更接近 1.0 表示 OOS 与 IS 一致
                robustness = np.mean(oos_scores) / np.mean(is_scores)
                result.robustness_score = max(0, min(1.0, robustness))

        # 过拟合检测: OOS 显著低于 IS (>30%)
        if result.robustness_score < 0.7:
            result.is_overfitting = True

    return result


# ===========================================================================
# Kelly Criterion 仓位计算
# ===========================================================================

@dataclass
class KellyResult:
    """Kelly 仓位计算结果。"""
    win_rate: float
    avg_win_pct: float               # 平均盈利百分比
    avg_loss_pct: float              # 平均亏损百分比
    kelly_fraction: float            # 原始 Kelly 比例
    half_kelly: float                # 半 Kelly (保守)
    quarter_kelly: float             # 四分之一 Kelly (极保守)
    optimal_f: float                 # 最优 f (Ralph Vince)
    recommended_risk_pct: float      # 推荐风险比例
    expected_growth_rate: float      # 期望增长率


def compute_kelly(
    trades: List[TradeRecord],
    max_risk_pct: float = 5.0,
) -> KellyResult:
    """计算 Kelly Criterion 最优仓位。

    借鉴资金管理理论：
      - Kelly f* = (p * b - q) / b
        其中 p = 胜率, b = 盈亏比 (avg_win/avg_loss), q = 1-p
      - 实际使用 1/2 Kelly 或 1/4 Kelly 更保守
    """
    if not trades:
        return KellyResult(
            win_rate=0, avg_win_pct=0, avg_loss_pct=0,
            kelly_fraction=0, half_kelly=0, quarter_kelly=0,
            optimal_f=0, recommended_risk_pct=0, expected_growth_rate=0,
        )

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]

    win_rate = len(wins) / len(trades)
    avg_win_pct = np.mean([abs(t.pnl_pct) for t in wins]) if wins else 0
    avg_loss_pct = np.mean([abs(t.pnl_pct) for t in losses]) if losses else 1.0

    if avg_loss_pct == 0:
        avg_loss_pct = 1.0

    # 盈亏比
    b = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 1.0

    # Kelly 公式
    q = 1 - win_rate
    kelly_f = (win_rate * b - q) / b if b > 0 else 0

    # 限制范围
    kelly_f = max(0.0, min(kelly_f, 0.5))

    half_kelly = kelly_f / 2
    quarter_kelly = kelly_f / 4

    # 期望增长率
    if kelly_f > 0:
        expected_growth = win_rate * math.log(1 + kelly_f * b) + \
                          q * math.log(1 - kelly_f)
    else:
        expected_growth = 0.0

    # 推荐风险比例 (不超过上限)
    recommended = min(kelly_f * 0.5 * 100, max_risk_pct)  # 半 Kelly 转百分比

    return KellyResult(
        win_rate=round(win_rate, 3),
        avg_win_pct=round(avg_win_pct, 3),
        avg_loss_pct=round(avg_loss_pct, 3),
        kelly_fraction=round(kelly_f, 4),
        half_kelly=round(half_kelly, 4),
        quarter_kelly=round(quarter_kelly, 4),
        optimal_f=round(kelly_f, 4),
        recommended_risk_pct=round(recommended, 2),
        expected_growth_rate=round(expected_growth, 4),
    )


# ===========================================================================
# 简化版 FreqAI 在线学习流水线
# ===========================================================================

class FreqAIPipeline:
    """简化版 FreqAI 在线学习流水线 — 借鉴 freqtrade FreqAI。

    核心思想：
      1. 特征工程: 从交易记录中提取特征
      2. 在线学习: 每新增 N 笔交易，更新模型
      3. 预测: 预测下一笔交易的胜率/盈亏
      4. 自适应: 特征重要性随市场变化调整
    """

    def __init__(
        self,
        feature_window: int = 50,
        retrain_interval: int = 10,
        learning_rate: float = 0.01,
    ):
        self.feature_window = feature_window
        self.retrain_interval = retrain_interval
        self.learning_rate = learning_rate

        # 特征权重 (线性模型)
        self.feature_weights: Dict[str, float] = {}
        self.bias: float = 0.0

        # 统计
        self.trades_since_retrain: int = 0
        self.total_predictions: int = 0
        self.correct_predictions: int = 0

    def extract_features(self, trade: TradeRecord) -> Dict[str, float]:
        """从交易记录中提取特征。"""
        return {
            "fvg_score": trade.fvg_score,
            "master_score": trade.master_score,
            "leverage": float(trade.leverage),
            "pnl_pct": trade.pnl_pct,
            "is_long": 1.0 if trade.direction == "long" else 0.0,
            "is_win": 1.0 if trade.is_win else 0.0,
        }

    def train(self, trades: List[TradeRecord]):
        """在最近 N 笔交易上训练线性模型。"""
        if len(trades) < 10:
            return

        recent = trades[-self.feature_window:]
        features_list = [self.extract_features(t) for t in recent]

        # 目标: 预测 pnl_pct
        X = []
        y = []
        feature_names = list(features_list[0].keys())
        feature_names = [f for f in feature_names if f not in ("pnl_pct", "is_win")]

        for ft in features_list:
            X.append([ft.get(f, 0) for f in feature_names])
            y.append(ft.get("pnl_pct", 0))

        X = np.array(X)
        y = np.array(y)

        if len(X) < 5:
            return

        # 简单线性回归
        try:
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0) + 1e-8
            X_norm = (X - X_mean) / X_std

            # 正规方程
            X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])
            coef = np.linalg.lstsq(X_aug, y, rcond=None)[0]

            self.feature_weights = dict(zip(feature_names, coef[:-1]))
            self.bias = coef[-1]
        except Exception as e:
            logger.debug(f"FreqAI training failed: {e}")

    def predict(self, features: Dict[str, float]) -> float:
        """预测期望 pnl_pct。"""
        if not self.feature_weights:
            return 0.0

        pred = self.bias
        for name, weight in self.feature_weights.items():
            pred += weight * features.get(name, 0)

        return float(pred)

    def update(self, new_trades: List[TradeRecord]):
        """在线更新模型。"""
        self.trades_since_retrain += len(new_trades)

        if self.trades_since_retrain >= self.retrain_interval:
            self.train(new_trades)
            self.trades_since_retrain = 0

    def get_prediction_accuracy(self) -> float:
        """获取预测准确率。"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


# ===========================================================================
# 参数敏感性分析
# ===========================================================================

@dataclass
class SensitivityResult:
    """参数敏感性分析结果。"""
    param_name: str
    values: List[float]
    scores: List[float]
    optimal_value: float
    sensitivity: float               # 高=参数敏感, 低=鲁棒
    stable_range: Tuple[float, float]  # 稳定区间


def sensitivity_analysis(
    param_name: str,
    param_space: ParamSpace,
    base_params: Dict[str, Any],
    evaluate_fn: Callable[[Dict[str, Any]], float],
    n_points: int = 10,
) -> SensitivityResult:
    """参数敏感性分析 — 固定其他参数，变化目标参数观察得分变化。

    借鉴 freqtrade 的策略稳定性评估：
      - 如果参数微小变化导致得分大幅波动 → 参数敏感 (过拟合风险)
      - 如果参数在较宽范围内得分稳定 → 策略鲁棒
    """
    values = param_space.sample(n_points)
    scores = []

    for v in values:
        params = base_params.copy()
        params[param_name] = v
        try:
            score = evaluate_fn(params)
            scores.append(score)
        except Exception:
            scores.append(0.0)

    if not scores or not values:
        return SensitivityResult(
            param_name=param_name,
            values=[], scores=[], optimal_value=0, sensitivity=0,
            stable_range=(0, 0),
        )

    # 最优值
    best_idx = np.argmax(scores)
    optimal_value = values[best_idx]

    # 敏感性: 得分标准差 / 得分均值
    score_array = np.array(scores)
    score_mean = np.mean(score_array)
    score_std = np.std(score_array)
    sensitivity = float(score_std / score_mean) if score_mean > 0 else 0

    # 稳定区间: 得分 >= 最优得分 * 0.9 的范围
    threshold = scores[best_idx] * 0.9
    stable_indices = [i for i, s in enumerate(scores) if s >= threshold]
    if stable_indices:
        stable_range = (values[min(stable_indices)], values[max(stable_indices)])
    else:
        stable_range = (optimal_value, optimal_value)

    return SensitivityResult(
        param_name=param_name,
        values=values,
        scores=scores,
        optimal_value=optimal_value,
        sensitivity=round(sensitivity, 4),
        stable_range=stable_range,
    )


# ===========================================================================
# 性能仪表盘
# ===========================================================================

def generate_performance_dashboard(
    trades: List[TradeRecord],
    initial_equity: float = 1000.0,
    config: Optional[Dict] = None,
) -> str:
    """生成性能仪表盘报告 — 借鉴 freqtrade 的 backtesting 分析输出。"""
    m = compute_metrics(trades, initial_equity)
    kelly = compute_kelly(trades)

    lines = [
        "",
        "╔" + "═" * 58 + "╗",
        "║  📊 策略性能仪表盘 (freqtrade 风格)".ljust(61) + "║",
        "╠" + "═" * 58 + "╣",
        "",
        "  ┌─ 基础指标 ─────────────────────────────────┐",
        f"  │ 总交易: {m.total_trades:>6d}  胜率: {m.win_rate:>7.1%}            │",
        f"  │ 盈利: {m.winning_trades:>6d}  亏损: {m.losing_trades:>6d}            │",
        "  └────────────────────────────────────────────┘",
        "",
        "  ┌─ 收益指标 ─────────────────────────────────┐",
        f"  │ 总收益: {m.total_return_pct:>+8.2f}%  平均: {m.avg_return_pct:>+8.2f}% │",
        f"  │ 最佳: {m.best_trade_pct:>+10.2f}%  最差: {m.worst_trade_pct:>+10.2f}% │",
        "  └────────────────────────────────────────────┘",
        "",
        "  ┌─ 风险指标 ─────────────────────────────────┐",
        f"  │ 最大回撤: {m.max_drawdown_pct:>6.2f}%  持续: {m.max_drawdown_duration:>4d}笔  │",
        f"  │ 盈利因子: {m.profit_factor:>6.2f}  期望值: {m.expectancy:>+8.2f}  │",
        "  └────────────────────────────────────────────┘",
        "",
        "  ┌─ 风险调整收益 ─────────────────────────────┐",
        f"  │ Sharpe: {m.sharpe_ratio:>6.2f}  Sortino: {m.sortino_ratio:>6.2f}  │",
        f"  │ Calmar: {m.calmar_ratio:>6.2f}                          │",
        "  └────────────────────────────────────────────┘",
        "",
        "  ┌─ 稳定性 ───────────────────────────────────┐",
        f"  │ 最大连赢: {m.max_consecutive_wins:>3d}  最大连亏: {m.max_consecutive_losses:>3d}  │",
        f"  │ 当前连赢: {m.consecutive_wins:>3d}  当前连亏: {m.consecutive_losses:>3d}  │",
        "  └────────────────────────────────────────────┘",
        "",
        "  ┌─ Kelly 仓位分析 ───────────────────────────┐",
        f"  │ Kelly f*: {kelly.kelly_fraction:>8.4f}  (全仓)           │",
        f"  │ 1/2 Kelly: {kelly.half_kelly:>8.4f}  (保守)           │",
        f"  │ 推荐风险: {kelly.recommended_risk_pct:>6.2f}% / 笔               │",
        "  └────────────────────────────────────────────┘",
        "",
        "  ┌─ 综合评分 ─────────────────────────────────┐",
        f"  │ {'█' * int(m.composite_score / 5) + '░' * (20 - int(m.composite_score / 5))}  {m.composite_score:.1f}/100        │",
        "  └────────────────────────────────────────────┘",
        "",
    ]

    # 评级
    if m.composite_score >= 80:
        lines.append("  🏆 评级: S — 优秀策略，可直接实盘")
    elif m.composite_score >= 60:
        lines.append("  👍 评级: A — 良好策略，建议优化后实盘")
    elif m.composite_score >= 40:
        lines.append("  ⚠ 评级: B — 一般策略，需要改进")
    elif m.composite_score >= 20:
        lines.append("  ❌ 评级: C — 较差策略，不建议实盘")
    else:
        lines.append("  💀 评级: D — 策略无效")

    return "\n".join(lines)


# ===========================================================================
# 一站式优化入口
# ===========================================================================

def run_full_optimization(
    trades: List[TradeRecord],
    param_space: Optional[List[ParamSpace]] = None,
    initial_equity: float = 1000.0,
    n_windows: int = 5,
) -> Dict[str, Any]:
    """执行完整优化流程。

    Returns:
        dict with:
          - hyperopt: HyperoptResult
          - walk_forward: WalkForwardResult
          - kelly: KellyResult
          - dashboard: str
          - metrics: StrategyMetrics
    """
    if not trades:
        logger.warning("No trades to optimize")
        return {}

    param_space = param_space or DEFAULT_PARAM_SPACE

    # 1. Hyperopt
    def objective_fn(trades_list, params):
        """优化目标: 最大化综合评分。"""
        # 这里简化处理：用参数调整后的评分
        filtered = []
        for t in trades_list:
            # 模拟参数筛选效果
            if t.fvg_score >= params.get("min_confidence", 0.3):
                filtered.append(t)
        if not filtered:
            return 0.0
        m = compute_metrics(filtered, initial_equity)
        return m.composite_score

    def obj_wrapper(params):
        return objective_fn(trades, params)

    hyperopt = BayesianHyperopt(param_space, obj_wrapper, n_initial=5, n_refine=2)
    hyperopt_result = hyperopt.optimize()

    # 2. Walk-Forward
    wf_result = walk_forward_optimization(
        trades, param_space, objective_fn, n_windows=n_windows
    )

    # 3. Kelly
    kelly_result = compute_kelly(trades)

    # 4. Metrics
    metrics = compute_metrics(trades, initial_equity)

    # 5. Dashboard
    dashboard = generate_performance_dashboard(trades, initial_equity)

    return {
        "hyperopt": hyperopt_result,
        "walk_forward": wf_result,
        "kelly": kelly_result,
        "metrics": metrics,
        "dashboard": dashboard,
    }