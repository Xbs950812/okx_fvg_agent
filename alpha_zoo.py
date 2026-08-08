"""
Alpha 因子库 — 借鉴 Vibe-Trading (23.6k ⭐) 的 Alpha Zoo 框架。

Vibe-Trading 核心设计：
  - Alpha Zoo: 因子注册表 + 运算符 + 安全门控
  - Correlation Regime: 相关性体制检测 + 因果滞后状态机
  - Memory Lifecycle: 质量评分 + Ebbinghaus 衰减 + 归档 GC
  - Factor Analysis: 因子分析框架 + 统计显著性检验

本模块实现：
  1. Alpha Factor Registry — 因子注册、发现、组合
  2. Factor Operators — 算术/比较/逻辑运算符
  3. Factor Backtesting — IC 分析、分位数收益、因子衰减
  4. Correlation Regime — 增强版因果滞后状态机
  5. Memory Lifecycle — 增强版记忆生命周期
  6. Factor Analysis — 统计显著性检验框架

HunHeng_OS_V1.0 — 集成 Vibe-Trading Alpha Zoo 461 因子 + 增强体制检测
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Callable, Set

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 从增强模块导入（Vibe-Trading 集成）
# ---------------------------------------------------------------------------

# 尝试导入增强版体制检测器（Schmitt-trigger + 边密度）
try:
    from regime_detector import (
        CausalHysteresisRegime as EnhancedCausalHysteresisRegime,
        EnhancedRegimeDetector,
        compute_edge_density,
        detect_regimes,
        MarketRegime as RegimeDetectorMarketRegime,
    )
    _REGIME_ENHANCED = True
except ImportError:
    _REGIME_ENHANCED = False

# 尝试导入因子库适配器（Alpha Zoo 461 因子）
try:
    from factor_zoo.adapter import FactorZooAdapter
    _FACTOR_ZOO_AVAILABLE = True
except ImportError:
    _FACTOR_ZOO_AVAILABLE = False


# ===========================================================================
# 市场体制枚举
# ===========================================================================

class MarketRegime(Enum):
    """市场体制 — 借鉴 Vibe-Trading 的 correlation-regime。"""
    FUSED = "FUSED"            # 融合/共振: 高相关 + 同向
    DIVERGENT = "DIVERGENT"    # 背离: 低相关 + 反向
    NEUTRAL = "NEUTRAL"        # 中性
    TRANSITIONING = "TRANSITIONING"  # 过渡中


# ===========================================================================
# Alpha 因子定义
# ===========================================================================

@dataclass
class AlphaFactor:
    """Alpha 因子定义 — 借鉴 Vibe-Trading Alpha Zoo。"""
    name: str                       # 因子名称
    category: str                   # 分类: price / volume / flow / structure / sentiment
    description: str = ""
    compute_fn: Optional[Callable] = None  # 计算函数
    params: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    # 元数据
    author: str = ""
    tags: List[str] = field(default_factory=list)
    expected_ic: float = 0.0        # 期望 IC (Information Coefficient)
    expected_ir: float = 0.0        # 期望 IR (Information Ratio)

    # 安全门控
    min_data_points: int = 30
    max_lookback: int = 200
    outlier_threshold: float = 5.0   # 异常值阈值 (sigma)


@dataclass
class FactorResult:
    """因子计算结果。"""
    factor_name: str
    values: np.ndarray              # 因子值
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 验证
    is_valid: bool = True
    nan_count: int = 0
    outlier_count: int = 0


# ===========================================================================
# 因子注册表
# ===========================================================================

class AlphaZoo:
    """Alpha 因子注册表 — 借鉴 Vibe-Trading Alpha Zoo。

    功能：
      - 注册/注销因子
      - 因子发现 (按分类/标签搜索)
      - 因子组合 (运算符)
      - 因子评估
    """

    def __init__(self):
        self._registry: Dict[str, AlphaFactor] = {}
        self._register_defaults()

    def _register_defaults(self):
        """注册默认因子 — 加密货币 FVG 策略相关。"""
        defaults = [
            AlphaFactor(
                name="fvg_width",
                category="price",
                description="FVG 缺口宽度 (价格百分比)",
                tags=["fvg", "price_action", "gap"],
                expected_ic=0.05,
                expected_ir=0.3,
            ),
            AlphaFactor(
                name="fvg_score",
                category="price",
                description="FVG 综合评分 (0-1)",
                tags=["fvg", "score", "composite"],
                expected_ic=0.08,
                expected_ir=0.5,
                min_data_points=20,
            ),
            AlphaFactor(
                name="volatility_ratio",
                category="price",
                description="波动率比率 (近/远)",
                tags=["volatility", "regime", "expansion"],
                expected_ic=0.03,
                expected_ir=0.2,
                outlier_threshold=4.0,
            ),
            AlphaFactor(
                name="orderbook_imbalance",
                category="structure",
                description="订单簿买卖失衡度",
                tags=["orderbook", "depth", "liquidity"],
                expected_ic=0.04,
                expected_ir=0.25,
            ),
            AlphaFactor(
                name="funding_rate",
                category="flow",
                description="资金费率",
                tags=["funding", "sentiment", "crowding"],
                expected_ic=0.06,
                expected_ir=0.35,
            ),
            AlphaFactor(
                name="oi_change",
                category="flow",
                description="持仓量变化率",
                tags=["open_interest", "flow", "position"],
                expected_ic=0.04,
                expected_ir=0.22,
            ),
            AlphaFactor(
                name="taker_volume_ratio",
                category="flow",
                description="主动买卖量比率",
                tags=["taker", "volume", "aggressiveness"],
                expected_ic=0.05,
                expected_ir=0.28,
            ),
            AlphaFactor(
                name="ls_ratio",
                category="sentiment",
                description="多空比",
                tags=["sentiment", "positioning", "crowd"],
                expected_ic=0.03,
                expected_ir=0.18,
            ),
            AlphaFactor(
                name="fear_greed",
                category="sentiment",
                description="恐慌贪婪指数 (0-100)",
                tags=["sentiment", "macro", "contrarian"],
                expected_ic=0.04,
                expected_ir=0.20,
            ),
            AlphaFactor(
                name="btc_dominance",
                category="macro",
                description="BTC 主导地位变化",
                tags=["macro", "btc", "rotation"],
                expected_ic=0.02,
                expected_ir=0.12,
            ),
            AlphaFactor(
                name="multi_period_resonance",
                category="price",
                description="多周期趋势共振 (1H/4H 一致性)",
                tags=["trend", "resonance", "multi_tf"],
                expected_ic=0.06,
                expected_ir=0.32,
            ),
            AlphaFactor(
                name="master_expert_score",
                category="composite",
                description="超级交易专家综合评分",
                tags=["composite", "expert", "master"],
                expected_ic=0.10,
                expected_ir=0.6,
                min_data_points=10,
            ),
        ]

        for factor in defaults:
            self.register(factor)

    def register(self, factor: AlphaFactor):
        """注册因子。"""
        if factor.name in self._registry:
            logger.warning(f"Factor '{factor.name}' already registered, overwriting")
        self._registry[factor.name] = factor

    def unregister(self, name: str):
        """注销因子。"""
        self._registry.pop(name, None)

    def get(self, name: str) -> Optional[AlphaFactor]:
        """获取因子。"""
        return self._registry.get(name)

    def list_by_category(self, category: str) -> List[AlphaFactor]:
        """按分类列出因子。"""
        return [f for f in self._registry.values() if f.category == category]

    def list_by_tag(self, tag: str) -> List[AlphaFactor]:
        """按标签搜索因子。"""
        return [f for f in self._registry.values() if tag in f.tags]

    def list_all(self) -> List[AlphaFactor]:
        """列出所有因子。"""
        return list(self._registry.values())

    def get_categories(self) -> List[str]:
        """获取所有分类。"""
        return sorted(set(f.category for f in self._registry.values()))


# ===========================================================================
# 因子运算符
# ===========================================================================

class FactorOperator:
    """因子运算符 — 借鉴 Vibe-Trading Alpha Zoo 的运算符系统。

    支持：
      - 算术: add, sub, mul, div, pow
      - 比较: gt, lt, eq, cross_above, cross_below
      - 逻辑: and, or, not
      - 变换: rank, zscore, clip, smooth
      - 组合: composite, weighted_sum
    """

    @staticmethod
    def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    @staticmethod
    def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    @staticmethod
    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    @staticmethod
    def div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return a / (b + eps)

    @staticmethod
    def pow(a: np.ndarray, n: float) -> np.ndarray:
        return np.power(a, n)

    @staticmethod
    def rank(values: np.ndarray) -> np.ndarray:
        """百分位排名 (0-1)。"""
        if len(values) == 0:
            return np.array([])
        n = len(values)
        if n <= 1:
            return np.zeros_like(values)
        ranks = np.argsort(np.argsort(values))
        return ranks / (n - 1)

    @staticmethod
    def zscore(values: np.ndarray) -> np.ndarray:
        """Z-Score 标准化。"""
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values)
        return (values - np.mean(values)) / std

    @staticmethod
    def clip(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
        return np.clip(values, lower, upper)

    @staticmethod
    def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
        """简单移动平均平滑。"""
        if len(values) < window:
            return values.copy()
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="same")

    @staticmethod
    def cross_above(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """上穿信号。"""
        result = np.zeros(len(a), dtype=bool)
        if len(a) < 2:
            return result
        result[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
        return result

    @staticmethod
    def cross_below(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """下穿信号。"""
        result = np.zeros(len(a), dtype=bool)
        if len(a) < 2:
            return result
        result[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
        return result

    @staticmethod
    def weighted_sum(factors: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """加权和组合。"""
        if not factors:
            return np.array([])
        total_weight = sum(weights.values())
        if total_weight <= 0:
            first_key = list(factors.keys())[0]
            return np.zeros_like(factors[first_key])
        first_key = list(factors.keys())[0]
        result = np.zeros_like(factors[first_key])
        for name, values in factors.items():
            w = weights.get(name, 0)
            result += values * w / total_weight
        return result

    @staticmethod
    def composite(factors: Dict[str, np.ndarray], method: str = "equal_weight") -> np.ndarray:
        """因子合成。"""
        if not factors:
            return np.array([])

        if method == "equal_weight":
            first_key = list(factors.keys())[0]
            result = np.zeros_like(factors[first_key])
            for values in factors.values():
                result += FactorOperator.zscore(values)
            return result / len(factors)

        elif method == "ic_weighted":
            # 需要 IC 权重，此处简化为等权
            return FactorOperator.composite(factors, "equal_weight")

        elif method == "max_ic":
            # 选取 IC 最高的因子（当前为占位实现，返回第一个因子；
            # 完整实现需要传入 IC 权重进行排序）
            # TODO: 实现 max_ic 选项 — 当前返回第一个因子，需要改为按 IC 排序
            logger.warning("max_ic composite method not fully implemented, returning first factor")
            return list(factors.values())[0]

        else:
            return FactorOperator.composite(factors, "equal_weight")


# ===========================================================================
# 因子回测
# ===========================================================================

@dataclass
class FactorBacktestResult:
    """因子回测结果。"""
    factor_name: str
    # IC 分析
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0              # IC / IC_std
    ic_positive_rate: float = 0.0    # IC > 0 的比例
    ic_series: List[float] = field(default_factory=list)

    # 分位数收益
    quantile_returns: Dict[int, float] = field(default_factory=dict)  # {q: return}
    top_bottom_spread: float = 0.0    # Q1 - Q5 收益差

    # 因子衰减
    decay_half_life: int = 0         # 半衰期 (期数)
    autocorrelation: List[float] = field(default_factory=list)

    # 统计检验
    t_statistic: float = 0.0
    p_value: float = 0.0
    is_significant: bool = False     # p < 0.05

    # 综合评分
    composite_score: float = 0.0     # 0-100


def backtest_factor(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    factor_name: str = "unknown",
    n_quantiles: int = 5,
    decay_lags: int = 20,
) -> FactorBacktestResult:
    """因子回测 — 借鉴 Vibe-Trading 的因子分析框架。

    Args:
        factor_values: 因子值序列 (N,)
        forward_returns: 前向收益序列 (N,) — 因子值对应的未来收益
        factor_name: 因子名称
        n_quantiles: 分位数数量
        decay_lags: 衰减分析滞后阶数

    Returns:
        FactorBacktestResult
    """
    result = FactorBacktestResult(factor_name=factor_name)

    if len(factor_values) < 30 or len(forward_returns) < 30:
        logger.warning(f"Not enough data for factor backtest: {factor_name}")
        return result

    # 对齐
    min_len = min(len(factor_values), len(forward_returns))
    factor_values = factor_values[-min_len:]
    forward_returns = forward_returns[-min_len:]

    # 去除 NaN
    valid = ~(np.isnan(factor_values) | np.isnan(forward_returns))
    fv = factor_values[valid]
    fr = forward_returns[valid]

    if len(fv) < 20:
        return result

    # ---- 1. IC 分析 ----
    # Spearman Rank IC
    ic_series = []
    window = min(20, len(fv) // 3)
    for i in range(window, len(fv)):
        fv_window = fv[i - window:i]
        fr_window = fr[i - window:i]
        if len(fv_window) > 2:
            rank_fv = np.argsort(np.argsort(fv_window))
            rank_fr = np.argsort(np.argsort(fr_window))
            ic = np.corrcoef(rank_fv, rank_fr)[0, 1]
            if not np.isnan(ic):
                ic_series.append(ic)

    if ic_series:
        result.ic_mean = float(np.mean(ic_series))
        result.ic_std = float(np.std(ic_series))
        result.ic_ir = result.ic_mean / result.ic_std if result.ic_std > 0 else 0
        result.ic_positive_rate = sum(1 for ic in ic_series if ic > 0) / len(ic_series)
        result.ic_series = ic_series

        # t 检验
        n = len(ic_series)
        if n > 1 and result.ic_std > 0:
            result.t_statistic = result.ic_mean / (result.ic_std / np.sqrt(n))
            # 使用 t 分布计算双尾 p-value；如果 scipy 不可用则回退到正态近似
            try:
                import scipy.stats as _st
                result.p_value = float(2 * (1 - _st.t.cdf(abs(result.t_statistic), df=n - 1)))
            except ImportError:
                # NOTE: 使用正态分布近似，小样本时可能低估 p-value。对于 n < 30 建议使用 t 分布。
                from math import erf
                result.p_value = float(1 - erf(abs(result.t_statistic) / np.sqrt(2)))
            result.is_significant = result.p_value < 0.05

    # ---- 2. 分位数收益 ----
    if len(fv) >= n_quantiles * 3:
        quantile_boundaries = np.percentile(fv, np.linspace(0, 100, n_quantiles + 1))
        for q in range(n_quantiles):
            mask = (fv >= quantile_boundaries[q]) & (fv < quantile_boundaries[q + 1])
            if q == n_quantiles - 1:
                mask = fv >= quantile_boundaries[q]
            if np.any(mask):
                result.quantile_returns[q] = float(np.mean(fr[mask]))

        # Q1 - Q5 收益差
        if 0 in result.quantile_returns and n_quantiles - 1 in result.quantile_returns:
            result.top_bottom_spread = (
                result.quantile_returns[n_quantiles - 1] -
                result.quantile_returns[0]
            )

    # ---- 3. 因子衰减 ----
    for lag in range(1, min(decay_lags + 1, len(fv) - 1)):
        if len(fv) > lag + 2:
            ac = np.corrcoef(fv[:-lag], fv[lag:])[0, 1]
            if not np.isnan(ac):
                result.autocorrelation.append(float(ac))
            else:
                result.autocorrelation.append(0.0)
        else:
            result.autocorrelation.append(0.0)

    # 半衰期: 自相关降到 0.5 以下的滞后阶数
    if len(result.autocorrelation) > 0:
        for i, ac in enumerate(result.autocorrelation):
            if ac < 0.5:
                result.decay_half_life = i + 1
                break
        if result.decay_half_life == 0:
            result.decay_half_life = len(result.autocorrelation)

    # ---- 4. 综合评分 ----
    result.composite_score = _compute_factor_score(result)

    return result


def _compute_factor_score(result: FactorBacktestResult) -> float:
    """计算因子综合评分 (0-100)。"""
    score = 0.0

    # IC IR (30%)
    score += min(max(result.ic_ir * 25, 0), 100) * 0.30

    # IC 稳定性 (20%)
    score += min(result.ic_positive_rate * 100, 100) * 0.20

    # 多空收益差 (20%)
    if abs(result.top_bottom_spread) > 0:
        score += min(abs(result.top_bottom_spread) * 500, 100) * 0.20

    # 统计显著性 (15%)
    if result.is_significant:
        score += 15

    # 衰减速度 (15%) — 衰减慢的因子更好
    if result.decay_half_life > 0:
        score += min(result.decay_half_life / 20 * 100, 100) * 0.15

    return min(score, 100)


# ===========================================================================
# 增强版相关性体制检测 (因果滞后状态机)
# ===========================================================================

@dataclass
class RegimeState:
    """体制状态 — 借鉴 Vibe-Trading 的因果滞后状态机。"""
    current_regime: MarketRegime = MarketRegime.NEUTRAL
    previous_regime: MarketRegime = MarketRegime.NEUTRAL
    transition_count: int = 0          # 过渡次数
    regime_duration: int = 0           # 当前体制持续时间
    min_regime_duration: int = 5       # 最短体制持续时间 (避免频繁切换)

    # 滞后参数
    hysteresis_threshold: float = 0.15  # 滞后阈值
    edge_density: float = 0.0          # 边密度

    # 历史
    regime_history: List[Tuple[float, str]] = field(default_factory=list)  # [(timestamp, regime), ...]


class CausalHysteresisRegime:
    """因果滞后状态机 — 借鉴 Vibe-Trading 的 correlation-regime 技能。

    改进：
      1. 滞后机制：不立即切换体制，需要持续确认
      2. 最短持续期：避免噪音导致的频繁切换
      3. 边密度：衡量体制边界强度
      4. 因果链：追踪体制切换的因果路径
    """

    def __init__(
        self,
        hysteresis_threshold: float = 0.15,
        min_regime_duration: int = 5,
        max_history: int = 100,
    ):
        self.state = RegimeState(
            hysteresis_threshold=hysteresis_threshold,
            min_regime_duration=min_regime_duration,
        )
        self.max_history = max_history

        # 过渡概率矩阵
        self.transition_counts: Dict[Tuple[str, str], int] = {}
        self.candidate_regime: Optional[MarketRegime] = None
        self.candidate_count: int = 0

    def update(
        self,
        correlation: float,
        trend_1h_sign: float,
        trend_4h_sign: float,
        timestamp: Optional[float] = None,
    ) -> MarketRegime:
        """更新体制状态。

        Args:
            correlation: 1H 与 4H 收益率的相关性
            trend_1h_sign: 1H 趋势方向符号
            trend_4h_sign: 4H 趋势方向符号
            timestamp: 时间戳

        Returns:
            当前体制
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. 确定候选体制
        candidate = self._classify_regime(correlation, trend_1h_sign, trend_4h_sign)

        # 2. 滞后处理
        if candidate != self.state.current_regime:
            if candidate == self.candidate_regime:
                self.candidate_count += 1
            else:
                self.candidate_regime = candidate
                self.candidate_count = 1

            # 需要持续 N 次确认才切换
            if self.candidate_count >= self.state.min_regime_duration:
                self._transition_to(candidate, timestamp)
                self.candidate_regime = None
                self.candidate_count = 0
        else:
            # 持续当前体制
            self.state.regime_duration += 1
            self.candidate_regime = None
            self.candidate_count = 0

        # 3. 更新边密度
        self.state.edge_density = self._compute_edge_density(correlation)

        # 4. 记录历史
        self.state.regime_history.append((timestamp, self.state.current_regime.value))
        if len(self.state.regime_history) > self.max_history:
            self.state.regime_history = self.state.regime_history[-self.max_history:]

        return self.state.current_regime

    def update_single_pair(
        self,
        returns: np.ndarray,
        symbol: Optional[str] = None,
    ) -> MarketRegime:
        """单品种直接更新 — 统一接口，对齐 EnhancedRegimeDetector。

        修复 Bug 38: 统一两个体制分类器的接口，
        让调用方可以无差别替换使用。底层仍走 update() 三参数接口。

        Args:
            returns: 该品种收益率序列 (1H 频率)
            symbol: 可选币种名（用于诊断日志）

        Returns:
            当前体制
        """
        if returns is None or len(returns) < 8:
            return self.state.current_regime

        # 计算 1H 与 4H 收益率趋势符号
        # 4H 收益率 = 1H 收益率的 4 步累计
        if len(returns) < 4:
            return self.state.current_regime
        ret_1h = float(np.sum(returns[-1:]))
        ret_4h = float(np.sum(returns[-4:]))
        trend_1h_sign = 1.0 if ret_1h > 0 else (-1.0 if ret_1h < 0 else 0.0)
        trend_4h_sign = 1.0 if ret_4h > 0 else (-1.0 if ret_4h < 0 else 0.0)

        # 修复 Bug C-3: 计算 1H 与 4H 收益率的交叉相关性（而非自相关）
        # 4H 收益率 = 1H 收益率的 4 步滚动累计
        if len(returns) < 8:
            corr = 0.0
        else:
            ret_1h_aligned = returns[3:]  # 对齐：从第4根开始
            ret_4h_rolling = np.array([float(np.sum(returns[i:i+4]))
                                       for i in range(len(returns) - 3)])
            std_1h = np.std(ret_1h_aligned)
            std_4h = np.std(ret_4h_rolling)
            if std_1h > 0 and std_4h > 0:
                corr = float(np.corrcoef(ret_1h_aligned, ret_4h_rolling)[0, 1])
            else:
                corr = 0.0

        regime = self.update(
            correlation=corr,
            trend_1h_sign=trend_1h_sign,
            trend_4h_sign=trend_4h_sign,
        )

        if symbol:
            logger.debug(
                f"[{symbol}] 1H ret={ret_1h:.4f}, 4H ret={ret_4h:.4f}, "
                f"corr={corr:.3f}, regime={regime.value}"
            )

        return regime

    def _classify_regime(
        self,
        correlation: float,
        trend_1h_sign: float,
        trend_4h_sign: float,
    ) -> MarketRegime:
        """分类当前体制。"""
        same_direction = (trend_1h_sign * trend_4h_sign) > 0

        if abs(correlation) > 0.7 and same_direction:
            return MarketRegime.FUSED
        elif abs(correlation) < 0.3:
            return MarketRegime.DIVERGENT
        elif abs(correlation) < 0.5 and not same_direction:
            return MarketRegime.DIVERGENT
        else:
            return MarketRegime.NEUTRAL

    def _transition_to(self, new_regime: MarketRegime, timestamp: float):
        """执行体制切换。"""
        old_regime = self.state.current_regime

        # 记录过渡
        key = (old_regime.value, new_regime.value)
        self.transition_counts[key] = self.transition_counts.get(key, 0) + 1

        # 更新状态
        self.state.previous_regime = old_regime
        self.state.current_regime = new_regime
        self.state.transition_count += 1
        self.state.regime_duration = 0

        logger.info(f"Regime transition: {old_regime.value} → {new_regime.value} "
                     f"(transition #{self.state.transition_count})")

    def _compute_edge_density(self, correlation: float) -> float:
        """计算边密度 — 衡量体制边界强度。"""
        # 相关性的绝对值越大，体制边界越清晰
        return abs(correlation)

    def get_transition_probability(self, from_regime: str, to_regime: str) -> float:
        """获取体制过渡概率。"""
        key = (from_regime, to_regime)
        count = self.transition_counts.get(key, 0)
        total = sum(c for k, c in self.transition_counts.items()
                    if k[0] == from_regime)
        return count / total if total > 0 else 0.0

    def get_regime_summary(self) -> Dict[str, Any]:
        """获取体制摘要。"""
        # 各体制时间占比
        regime_counts = {}
        for _, regime in self.state.regime_history:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        total = sum(regime_counts.values())
        regime_pcts = {k: v / total for k, v in regime_counts.items()} if total > 0 else {}

        return {
            "current_regime": self.state.current_regime.value,
            "previous_regime": self.state.previous_regime.value,
            "regime_duration": self.state.regime_duration,
            "transition_count": self.state.transition_count,
            "regime_distribution": regime_pcts,
            "edge_density": self.state.edge_density,
            "top_transitions": sorted(
                self.transition_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }


# 修复 2026-08-07: 优先使用增强版体制检测器 (regime_detector.EnhancedRegimeDetector
# 子类, Schmitt-trigger)。旧版实现 (上方 class CausalHysteresisRegime) 不支持
# regime_enter_threshold/exit_threshold/corr_window/smooth_window 配置参数 —
# agent/coin_tracker 从 alpha_zoo 导入时自动获得增强版, 且 update() 接口完全兼容。
if _REGIME_ENHANCED:
    CausalHysteresisRegime = EnhancedCausalHysteresisRegime


# ===========================================================================
# 增强版记忆生命周期
# ===========================================================================

@dataclass
class LifecycleEntry:
    """记忆条目 — 增强版。"""
    key: str
    category: str
    content: Any
    created_at: float
    quality_score: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    decay_factor: float = 1.0

    # 增强字段
    importance: float = 0.5           # 重要性 (0-1)
    reliability: float = 0.5          # 可靠性 (0-1)
    times_validated: int = 0          # 被验证次数
    tags: List[str] = field(default_factory=list)
    source: str = ""                  # 来源
    ttl: Optional[float] = None       # 过期时间


class EnhancedMemoryLifecycle:
    """增强版记忆生命周期 — 借鉴 Vibe-Trading 记忆管理系统。

    改进：
      1. 质量评分: 基于访问频率、验证次数、来源可靠性
      2. Ebbinghaus 衰减: 指数遗忘曲线
      3. 重要性加权: 高重要性条目衰减更慢
      4. 归档 GC: 低质量 + 低访问 → 归档
      5. 层级结构: Tier 1 (热) → Tier 2 (温) → Tier 3 (冷)
    """

    def __init__(
        self,
        half_life_days: float = 30.0,
        archive_threshold: float = 0.15,
        gc_threshold: float = 0.05,
    ):
        self.half_life_days = half_life_days
        self.archive_threshold = archive_threshold
        self.gc_threshold = gc_threshold

        # 三层存储
        self.tier1_hot: Dict[str, LifecycleEntry] = {}    # 最近访问
        self.tier2_warm: Dict[str, LifecycleEntry] = {}   # 中等访问
        self.tier3_cold: Dict[str, LifecycleEntry] = {}   # 低频访问

        self._tier_max_sizes = {"tier1": 100, "tier2": 500, "tier3": 2000}

    def store(self, entry: LifecycleEntry):
        """存储条目 — 自动分配到合适的层级。"""
        # 根据质量和访问频率分配层级
        score = entry.quality_score * entry.importance
        if score > 0.6 or entry.access_count > 5:
            self._promote_to_tier(entry, "tier1")
        elif score > 0.3:
            self._promote_to_tier(entry, "tier2")
        else:
            self._promote_to_tier(entry, "tier3")

    def retrieve(self, key: str) -> Optional[LifecycleEntry]:
        """检索条目并更新访问信息。"""
        # 层级递减搜索
        for tier in [self.tier1_hot, self.tier2_warm, self.tier3_cold]:
            if key in tier:
                entry = tier[key]
                entry.access_count += 1
                entry.last_accessed = time.time()
                # 强化记忆
                entry.decay_factor = min(1.0, entry.decay_factor + 0.05)
                # 提升层级
                if entry.access_count > 5 and tier is not self.tier1_hot:
                    self._promote_to_tier(entry, "tier1")
                return entry

        return None

    def _promote_to_tier(self, entry: LifecycleEntry, target_tier: str):
        """将条目提升到目标层级。"""
        # 从所有层级中移除
        for tier_dict in [self.tier1_hot, self.tier2_warm, self.tier3_cold]:
            tier_dict.pop(entry.key, None)

        # 添加到目标层级
        target = getattr(self, f"{target_tier}_hot" if target_tier == "tier1"
                         else f"{target_tier}_warm" if target_tier == "tier2"
                         else f"{target_tier}_cold")
        target[entry.key] = entry

        # 检查容量
        max_size = self._tier_max_sizes.get(target_tier, 500)
        if len(target) > max_size:
            self._evict_tier(target_tier)

    def _evict_tier(self, tier_name: str):
        """逐出层级中的低质量条目。"""
        tier = getattr(self, f"{tier_name}_hot" if tier_name == "tier1"
                       else f"{tier_name}_warm" if tier_name == "tier2"
                       else f"{tier_name}_cold")

        # 按综合评分排序，移除最低的
        entries = sorted(
            tier.items(),
            key=lambda x: x[1].quality_score * x[1].decay_factor * x[1].importance,
        )
        to_remove = entries[:len(entries) // 5]  # 移除底部 20%
        for key, _ in to_remove:
            del tier[key]

    def decay_all(self):
        """对所有条目执行 Ebbinghaus 衰减。"""
        for tier in [self.tier1_hot, self.tier2_warm, self.tier3_cold]:
            for entry in tier.values():
                self._apply_decay(entry)
        # H-8: 清理已完全衰减的条目
        self._cleanup_expired()

    def _apply_decay(self, entry: LifecycleEntry):
        """对单个条目应用 Ebbinghaus 衰减。

        Ebbinghaus 遗忘曲线:
          R = e^(-t / S)
          其中 S = 半衰期 * 重要性因子
        """
        if entry.last_accessed == 0.0:
            elapsed_days = max(0, (time.time() - entry.created_at) / 86400.0)
        else:
            elapsed_days = (time.time() - entry.last_accessed) / 86400.0
        if elapsed_days > 0:
            # 重要性越高，衰减越慢
            effective_half_life = self.half_life_days * (0.5 + entry.importance)
            entry.decay_factor = math.exp(
                -math.log(2) * elapsed_days / effective_half_life
            )

    def _cleanup_expired(self):
        """H-8: 清理已完全衰减的条目，防止内存泄漏。"""
        for tier in [self.tier1_hot, self.tier2_warm, self.tier3_cold]:
            expired = [k for k, v in tier.items() if v.decay_factor <= 0.001]
            for k in expired:
                del tier[k]

    def update_quality(self, key: str, was_correct: bool, learning_rate: float = 0.1):
        """根据验证结果更新质量评分。"""
        entry = self.retrieve(key)
        if not entry:
            return

        entry.times_validated += 1
        if was_correct:
            entry.quality_score = min(0.95, entry.quality_score + learning_rate)
            entry.reliability = min(0.95, entry.reliability + learning_rate * 0.5)
        else:
            entry.quality_score = max(0.05, entry.quality_score - learning_rate * 0.5)
            entry.reliability = max(0.05, entry.reliability - learning_rate * 0.5)

    def prune_archive(self) -> int:
        """归档低质量条目。"""
        removed = 0
        for tier in [self.tier3_cold, self.tier2_warm]:
            to_remove = []
            for key, entry in tier.items():
                score = entry.quality_score * entry.decay_factor
                if score < self.archive_threshold:
                    to_remove.append(key)
            for key in to_remove:
                tier.pop(key, None)
                removed += 1

        return removed

    def gc(self) -> int:
        """垃圾回收 — 删除极低质量条目。"""
        removed = 0
        for tier in [self.tier3_cold, self.tier2_warm, self.tier1_hot]:
            to_remove = []
            for key, entry in tier.items():
                score = entry.quality_score * entry.decay_factor
                if score < self.gc_threshold:
                    to_remove.append(key)
            for key in to_remove:
                tier.pop(key, None)
                removed += 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计。"""
        return {
            "tier1_hot": len(self.tier1_hot),
            "tier2_warm": len(self.tier2_warm),
            "tier3_cold": len(self.tier3_cold),
            "total": len(self.tier1_hot) + len(self.tier2_warm) + len(self.tier3_cold),
            "avg_quality_tier1": np.mean([e.quality_score for e in self.tier1_hot.values()]) if self.tier1_hot else 0,
            "avg_quality_tier2": np.mean([e.quality_score for e in self.tier2_warm.values()]) if self.tier2_warm else 0,
            "avg_quality_tier3": np.mean([e.quality_score for e in self.tier3_cold.values()]) if self.tier3_cold else 0,
        }


# ===========================================================================
# 因子分析框架
# ===========================================================================

@dataclass
class FactorAnalysisResult:
    """因子分析结果。"""
    factor_name: str
    # 单因子检验
    ic_mean: float = 0.0
    ic_ir: float = 0.0
    t_stat: float = 0.0
    p_value: float = 0.0

    # 多因子检验
    vif: float = 0.0                # 方差膨胀因子 (多重共线性)
    marginal_contribution: float = 0.0  # 边际贡献

    # 稳健性
    is_robust: bool = False
    robustness_score: float = 0.0

    # 结论
    recommendation: str = ""        # "strong_buy" | "buy" | "hold" | "avoid"
    confidence: float = 0.0


class FactorAnalyzer:
    """因子分析器 — 借鉴 Vibe-Trading 的 factor_analysis 工具。

    分析流程：
      1. 单因子检验: IC 均值、t 统计量
      2. 多因子检验: 多重共线性 (VIF)
      3. 稳健性检验: 子样本一致性
      4. 边际贡献: 因子对组合的增量贡献
      5. 综合评级
    """

    def __init__(self, alpha_zoo: Optional[AlphaZoo] = None):
        self.alpha_zoo = alpha_zoo or AlphaZoo()

    def analyze_factor(
        self,
        factor_name: str,
        factor_values: np.ndarray,
        forward_returns: np.ndarray,
        existing_factors: Optional[Dict[str, np.ndarray]] = None,
    ) -> FactorAnalysisResult:
        """分析单个因子。"""
        result = FactorAnalysisResult(factor_name=factor_name)

        if len(factor_values) < 30:
            return result

        # ---- 1. 单因子回测 ----
        bt_result = backtest_factor(factor_values, forward_returns, factor_name)
        result.ic_mean = bt_result.ic_mean
        result.ic_ir = bt_result.ic_ir
        result.t_stat = bt_result.t_statistic
        result.p_value = bt_result.p_value

        # ---- 2. 多重共线性检验 (VIF) ----
        if existing_factors and len(existing_factors) > 0:
            result.vif = self._compute_vif(factor_values, existing_factors)

        # ---- 3. 边际贡献 ----
        if existing_factors:
            result.marginal_contribution = self._compute_marginal_contribution(
                factor_values, existing_factors, forward_returns
            )

        # ---- 4. 稳健性 ----
        result.robustness_score = self._compute_robustness(factor_values, forward_returns)
        result.is_robust = result.robustness_score > 0.6

        # ---- 5. 综合推荐 ----
        result.recommendation, result.confidence = self._make_recommendation(result)

        return result

    def _compute_vif(
        self,
        factor_values: np.ndarray,
        existing_factors: Dict[str, np.ndarray],
    ) -> float:
        """计算方差膨胀因子 (VIF)。"""
        # 简化版：取与最高相关因子的 VIF
        max_vif = 0.0
        for name, values in existing_factors.items():
            min_len = min(len(factor_values), len(values))
            if min_len > 2:
                fv = factor_values[:min_len]
                ev = values[:min_len]
                valid = ~(np.isnan(fv) | np.isnan(ev))
                if valid.sum() > 2:
                    corr = abs(np.corrcoef(fv[valid], ev[valid])[0, 1])
                    if corr < 1.0:
                        vif = 1.0 / max(1.0 - corr ** 2, 1e-10)
                        vif = min(vif, 100.0)  # 上限截断，防止接近奇异时 VIF 爆炸
                        max_vif = max(max_vif, vif)

        return max_vif

    def _compute_marginal_contribution(
        self,
        factor_values: np.ndarray,
        existing_factors: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> float:
        """计算边际贡献 — 因子对现有组合的增量 IC。"""
        # 简化版：比较加入前后的 IC 变化
        existing_ic = 0.0
        if existing_factors:
            existing_composite = FactorOperator.composite(existing_factors)
            min_len = min(len(existing_composite), len(forward_returns))
            if min_len > 2:
                bt = backtest_factor(existing_composite[:min_len], forward_returns[:min_len])
                existing_ic = abs(bt.ic_mean)

        # 加入新因子后的 IC
        combined = existing_factors.copy() if existing_factors else {}
        combined["new"] = factor_values
        combined_composite = FactorOperator.composite(combined)
        min_len = min(len(combined_composite), len(forward_returns))
        if min_len > 2:
            bt = backtest_factor(combined_composite[:min_len], forward_returns[:min_len])
            combined_ic = abs(bt.ic_mean)
        else:
            combined_ic = existing_ic

        return combined_ic - existing_ic

    def _compute_robustness(
        self,
        factor_values: np.ndarray,
        forward_returns: np.ndarray,
        n_splits: int = 3,
    ) -> float:
        """子样本稳健性检验。"""
        if len(factor_values) < n_splits * 10 or len(forward_returns) < n_splits * 10:
            return 0.0

        split_size = len(factor_values) // n_splits
        if split_size < 1:
            return 0.0
        ic_means = []

        for i in range(n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < n_splits - 1 else len(factor_values)
            fv = factor_values[start:end]
            fr = forward_returns[start:end]
            if len(fv) > 10:
                bt = backtest_factor(fv, fr)
                ic_means.append(bt.ic_mean)

        if not ic_means:
            return 0.0

        # 一致性: 所有子样本 IC 同号
        same_sign = all(ic > 0 for ic in ic_means) or all(ic < 0 for ic in ic_means)
        sign_score = 0.5 if same_sign else 0.0

        # 稳定性: IC 标准差小
        ic_std = np.std(ic_means)
        ic_mean = np.mean(ic_means)
        stability = max(0, 1 - ic_std / (abs(ic_mean) + 1e-8))

        return (sign_score + stability) / 2

    def _make_recommendation(
        self,
        result: FactorAnalysisResult,
    ) -> Tuple[str, float]:
        """生成因子推荐。"""
        score = 0.0

        if result.p_value < 0.05:
            score += 0.3
        if abs(result.ic_ir) > 0.3:
            score += 0.2
        if result.vif < 5.0:
            score += 0.15
        if result.marginal_contribution > 0:
            score += 0.15
        if result.is_robust:
            score += 0.2

        if score >= 0.7:
            return "strong_include", score
        elif score >= 0.5:
            return "include", score
        elif score >= 0.3:
            return "consider", score
        else:
            return "exclude", score


# ===========================================================================
# 一站式因子分析
# ===========================================================================

def analyze_all_factors(
    factor_values: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
    alpha_zoo: Optional[AlphaZoo] = None,
) -> Dict[str, FactorAnalysisResult]:
    """分析所有因子。"""
    analyzer = FactorAnalyzer(alpha_zoo)
    results = {}

    existing = {}
    for name, values in factor_values.items():
        result = analyzer.analyze_factor(name, values, forward_returns, existing)
        results[name] = result
        # 只保留好的因子
        if result.recommendation in ("strong_include", "include"):
            existing[name] = values

    return results


# ===========================================================================
# 格式化输出
# ===========================================================================

def format_factor_report(results: Dict[str, FactorAnalysisResult]) -> str:
    """格式化因子分析报告。"""
    lines = [
        "",
        "╔" + "═" * 58 + "╗",
        "║  🧬 Alpha Zoo 因子分析报告".ljust(61) + "║",
        "╠" + "═" * 58 + "╣",
        f"║  {'Factor':<20} {'IC':>6} {'IR':>6} {'VIF':>6} {'Rec':>12} ║",
        "╠" + "═" * 58 + "╣",
    ]

    sorted_results = sorted(
        results.items(),
        key=lambda x: abs(x[1].ic_ir),
        reverse=True,
    )

    for name, r in sorted_results:
        rec = {"strong_include": "★★★ 强推", "include": "★★ 推荐",
               "consider": "★ 考虑", "exclude": "✗ 排除"}.get(r.recommendation, "?")
        lines.append(
            f"║  {name:<20} {r.ic_mean:>+5.3f} {r.ic_ir:>+5.2f} "
            f"{r.vif:>5.1f} {rec:>12} ║"
        )

    lines.append("╚" + "═" * 58 + "╝")
    return "\n".join(lines)