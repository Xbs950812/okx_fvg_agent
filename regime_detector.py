"""
增强版相关性体制检测 — 因果迟滞状态机 (Schmitt-Trigger Hysteresis)。

借鉴 Vibe-Trading (23.6k⭐) 的 regime.py 实现：
  - compute_edge_density: 多资产滚动相关性 → 边密度标量
  - detect_regimes: 双阈值迟滞状态机 (Schmitt trigger)
  - 因果性保证: 只使用 trailing/rolling 窗口，绝不使用 centered/future 窗口
  - 市场熔断检测: FUSED(共振) / DIVERGENT(背离) / NEUTRAL(中性)

HunHeng_OS_V1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# 市场体制枚举
# ===========================================================================

class MarketRegime(Enum):
    """市场体制状态。"""
    FUSED = "FUSED"              # 融合/共振: 高相关 + 同向 → 市场熔断风险
    DIVERGENT = "DIVERGENT"      # 背离: 低相关 + 反向
    NEUTRAL = "NEUTRAL"          # 中性
    TRANSITIONING = "TRANSITIONING"  # 过渡中


# ===========================================================================
# 边密度计算
# ===========================================================================

def compute_edge_density(
    returns: pd.DataFrame,
    corr_window: int = 60,
    edge_threshold: float = 0.5,
) -> pd.Series:
    """将滚动相关矩阵压缩为边密度序列。

    边密度 = |correlation| >= edge_threshold 的资产对占比。
    衡量市场"熔断"程度的标量指标，范围 [0, 1]。

    Args:
        returns: 多资产收益矩阵，列为资产代码，行为日期
        corr_window: 滚动窗口长度 (bar 数)
        edge_threshold: |ρ| 阈值，超过即计为一条"边"

    Returns:
        边密度序列，对齐 returns.index（warmup 期间为 NaN）
    """
    n_assets = returns.shape[1]
    if n_assets < 2:
        logger.warning("compute_edge_density: need >= 2 assets, got %d", n_assets)
        return pd.Series(np.nan, index=returns.index)

    n_pairs = n_assets * (n_assets - 1) // 2
    upper_mask = np.triu(np.ones((n_assets, n_assets), dtype=bool), k=1)

    density = pd.Series(np.nan, index=returns.index)
    for i in range(corr_window, len(returns) + 1):
        corr = returns.iloc[i - corr_window:i].corr().abs().to_numpy()
        density.iloc[i - 1] = float((corr[upper_mask] >= edge_threshold).sum()) / n_pairs

    return density


def compute_edge_density_single(
    values_a: np.ndarray,
    values_b: np.ndarray,
    corr_window: int = 60,
    edge_threshold: float = 0.5,
) -> pd.Series:
    """双资产边密度简化版 — 用于单对资产的滚动相关性分析。

    Args:
        values_a: 资产 A 的值序列
        values_b: 资产 B 的值序列
        corr_window: 滚动窗口
        edge_threshold: 边阈值

    Returns:
        边密度序列 (0 或 1/n_pairs)
    """
    if len(values_a) < corr_window or len(values_b) < corr_window:
        return pd.Series(np.nan, index=range(len(values_a)))

    min_len = min(len(values_a), len(values_b))
    density = pd.Series(np.nan, index=range(min_len))

    for i in range(corr_window, min_len + 1):
        a = values_a[i - corr_window:i]
        b = values_b[i - corr_window:i]
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() < 2:
            continue
        corr = np.corrcoef(a[valid], b[valid])[0, 1]
        density.iloc[i - 1] = 1.0 if abs(corr) >= edge_threshold else 0.0

    return density


# ===========================================================================
# 因果迟滞状态机 (Schmitt Trigger)
# ===========================================================================

def detect_regimes(
    density: pd.Series,
    smooth_window: int = 5,
    enter_threshold: float = 0.65,
    exit_threshold: float = 0.45,
) -> pd.DataFrame:
    """双阈值迟滞 (Schmitt-trigger) 体制检测状态机。

    当平滑后的边密度达到 enter_threshold 时进入 FUSED 状态，
    直到回落到 exit_threshold 以下才退出。
    中间的死区 (dead band) 用于抑制信号抖动。

    **因果性保证**: 平滑使用 trailing window (causal)，绝不使用 centered window。

    Args:
        density: 边密度序列 (来自 compute_edge_density)
        smooth_window: 因果平滑窗口 (trailing mean)
        enter_threshold: 进入 FUSED 的阈值
        exit_threshold: 退出 FUSED 的阈值 (必须 < enter_threshold)

    Returns:
        DataFrame，包含 density, smoothed, fused (0/1) 列
    """
    if exit_threshold >= enter_threshold:
        raise ValueError("exit_threshold must be below enter_threshold")

    # Trailing mean = causal. 绝不使用 centered window
    smoothed = density.rolling(smooth_window, min_periods=1).mean()

    fused = False
    states = np.zeros(len(smoothed), dtype=int)
    for i, value in enumerate(smoothed.to_numpy()):
        if np.isnan(value):
            states[i] = int(fused)
            continue
        if not fused and value >= enter_threshold:
            fused = True
        elif fused and value <= exit_threshold:
            fused = False
        states[i] = int(fused)

    return pd.DataFrame(
        {"density": density, "smoothed": smoothed, "fused": states},
        index=density.index,
    )


# ===========================================================================
# 增强版因果滞后体制检测器
# ===========================================================================

@dataclass
class RegimeState:
    """体制状态快照。"""
    current_regime: MarketRegime = MarketRegime.NEUTRAL
    previous_regime: MarketRegime = MarketRegime.NEUTRAL
    transition_count: int = 0
    regime_duration: int = 0
    min_regime_duration: int = 5
    hysteresis_threshold: float = 0.15
    edge_density: float = 0.0
    regime_history: List[Tuple[float, str]] = field(default_factory=list)


class EnhancedRegimeDetector:
    """增强版因果滞后体制检测器 — 融合 Vibe-Trading 的 Schmitt-trigger + 边密度。

    相比原版 CausalHysteresisRegime 的改进：
      1. 边密度计算: 多资产滚动相关性 → 体制强度标量
      2. Schmitt-trigger 迟滞: 双阈值抑制噪音抖动
      3. 因果性保证: 绝无 lookahead bias
      4. 多维度体制分类: 相关性 + 趋势方向 + 波动率
      5. FUSED 熔断区间检测: 连续 FUSED 区间的起止日期

    用法:
        detector = EnhancedRegimeDetector(
            enter_threshold=0.65,
            exit_threshold=0.45,
            corr_window=60,
            smooth_window=5,
        )

        # 单资产对 (1H vs 4H 收益率)
        regime = detector.update_single_pair(
            returns_1h=np.array([...]),
            returns_4h=np.array([...]),
            timestamps=[...],
        )

        # 多资产
        regime = detector.update_multi_asset(returns_df)
    """

    def __init__(
        self,
        enter_threshold: float = 0.65,
        exit_threshold: float = 0.45,
        corr_window: int = 60,
        smooth_window: int = 5,
        hysteresis_threshold: float = 0.15,
        min_regime_duration: int = 5,
        max_history: int = 500,
        symbol: str = "",
    ):
        # 2026-08-08: 币种标识用于切换日志定位（per-symbol 实例下必须可区分）
        self.symbol = symbol
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.corr_window = corr_window
        self.smooth_window = smooth_window
        self.hysteresis_threshold = hysteresis_threshold

        self.state = RegimeState(
            hysteresis_threshold=hysteresis_threshold,
            min_regime_duration=min_regime_duration,
        )
        self.max_history = max_history

        # 过渡概率矩阵
        self.transition_counts: Dict[Tuple[str, str], int] = {}
        self.candidate_regime: Optional[MarketRegime] = None
        self.candidate_count: int = 0

        # 存储最近的数据用于边密度计算
        self._density_buffer: List[float] = []
        self._smoothed_buffer: List[float] = []
        self._fused_buffer: List[int] = []
        self._max_buffer = 500

    def update_single_pair(
        self,
        returns_1h: np.ndarray,
        returns_4h: np.ndarray,
        timestamps: Optional[List[float]] = None,
    ) -> MarketRegime:
        """单资产对体制更新 (1H vs 4H 收益率)。

        Args:
            returns_1h: 1H 收益率序列
            returns_4h: 4H 收益率序列
            timestamps: 可选时间戳

        Returns:
            当前 MarketRegime
        """
        if len(returns_1h) < self.corr_window or len(returns_4h) < self.corr_window:
            return self.state.current_regime

        # 对齐长度
        min_len = min(len(returns_1h), len(returns_4h))
        r1 = returns_1h[-min_len:]
        r4 = returns_4h[-min_len:]

        # 计算滚动相关性
        corr = 0.0
        if len(r1) >= 2:
            valid = ~(np.isnan(r1) | np.isnan(r4))
            if valid.sum() >= 2:
                corr = float(np.corrcoef(r1[valid], r4[valid])[0, 1])
                if np.isnan(corr):
                    corr = 0.0

        # 计算趋势方向
        trend_1h_sign = float(np.sign(np.mean(r1))) if len(r1) > 0 else 0.0
        trend_4h_sign = float(np.sign(np.mean(r4))) if len(r4) > 0 else 0.0

        # 计算边密度
        edge_density = 1.0 if abs(corr) >= 0.5 else abs(corr)

        # 更新边密度缓冲区
        self._density_buffer.append(edge_density)
        if len(self._density_buffer) > self._max_buffer:
            self._density_buffer = self._density_buffer[-self._max_buffer:]

        # 计算平滑边密度 (causal trailing mean)
        if len(self._density_buffer) >= self.smooth_window:
            smoothed = float(np.mean(self._density_buffer[-self.smooth_window:]))
        else:
            smoothed = edge_density

        self._smoothed_buffer.append(smoothed)
        if len(self._smoothed_buffer) > self._max_buffer:
            self._smoothed_buffer = self._smoothed_buffer[-self._max_buffer:]

        # Schmitt-trigger 迟滞状态机
        is_fused = bool(self._fused_buffer[-1]) if self._fused_buffer else False
        if not is_fused and smoothed >= self.enter_threshold:
            is_fused = True
        elif is_fused and smoothed <= self.exit_threshold:
            is_fused = False

        self._fused_buffer.append(int(is_fused))
        if len(self._fused_buffer) > self._max_buffer:
            self._fused_buffer = self._fused_buffer[-self._max_buffer:]

        # 确定体制
        if is_fused:
            candidate = MarketRegime.FUSED
        elif abs(corr) < 0.3:
            candidate = MarketRegime.DIVERGENT
        elif abs(corr) < 0.5 and (trend_1h_sign * trend_4h_sign) <= 0:
            candidate = MarketRegime.DIVERGENT
        else:
            candidate = MarketRegime.NEUTRAL

        # 滞后处理：需要持续确认才切换
        if candidate != self.state.current_regime:
            if candidate == self.candidate_regime:
                self.candidate_count += 1
            else:
                self.candidate_regime = candidate
                self.candidate_count = 1

            if self.candidate_count >= self.state.min_regime_duration:
                self._transition_to(candidate)
                self.candidate_regime = None
                self.candidate_count = 0
        else:
            self.state.regime_duration += 1
            self.candidate_regime = None
            self.candidate_count = 0

        self.state.edge_density = smoothed

        return self.state.current_regime

    def update_multi_asset(
        self,
        returns_df: pd.DataFrame,
    ) -> MarketRegime:
        """多资产体制更新。

        Args:
            returns_df: 多资产收益 DataFrame，列为资产代码，行为日期

        Returns:
            当前 MarketRegime
        """
        if returns_df.shape[1] < 2:
            logger.warning("EnhancedRegimeDetector: need >= 2 assets for multi-asset mode")
            return self.state.current_regime

        if len(returns_df) < self.corr_window:
            return self.state.current_regime

        # 计算边密度
        density = compute_edge_density(
            returns_df,
            corr_window=self.corr_window,
            edge_threshold=0.5,
        )

        # 丢弃 NaN
        density = density.dropna()
        if len(density) < self.smooth_window:
            return self.state.current_regime

        # 检测体制
        regime_df = detect_regimes(
            density,
            smooth_window=self.smooth_window,
            enter_threshold=self.enter_threshold,
            exit_threshold=self.exit_threshold,
        )

        latest = regime_df.iloc[-1]
        is_fused = bool(latest["fused"])
        smoothed_val = float(latest["smoothed"])

        # 确定候选体制
        if is_fused:
            candidate = MarketRegime.FUSED
        else:
            # 检查最近的趋势一致性
            last_n = min(20, len(returns_df))
            recent = returns_df.iloc[-last_n:]
            signs = np.sign(recent.mean())
            same_direction = (signs > 0).all() or (signs < 0).all()

            if smoothed_val < 0.3:
                candidate = MarketRegime.DIVERGENT
            elif not same_direction:
                candidate = MarketRegime.DIVERGENT
            else:
                candidate = MarketRegime.NEUTRAL

        # 滞后处理
        if candidate != self.state.current_regime:
            if candidate == self.candidate_regime:
                self.candidate_count += 1
            else:
                self.candidate_regime = candidate
                self.candidate_count = 1

            if self.candidate_count >= self.state.min_regime_duration:
                self._transition_to(candidate)
                self.candidate_regime = None
                self.candidate_count = 0
        else:
            self.state.regime_duration += 1
            self.candidate_regime = None
            self.candidate_count = 0

        self.state.edge_density = smoothed_val

        return self.state.current_regime

    def update(
        self,
        correlation: float,
        trend_1h_sign: float,
        trend_4h_sign: float,
        timestamp: Optional[float] = None,
    ) -> MarketRegime:
        """兼容旧接口的单步更新。

        Args:
            correlation: 相关性系数
            trend_1h_sign: 1H 趋势符号
            trend_4h_sign: 4H 趋势符号
            timestamp: 时间戳

        Returns:
            当前体制
        """
        import time as _time
        if timestamp is None:
            timestamp = _time.time()

        # 边密度
        edge_density = abs(correlation)

        self._density_buffer.append(edge_density)
        if len(self._density_buffer) > self._max_buffer:
            self._density_buffer = self._density_buffer[-self._max_buffer:]

        if len(self._density_buffer) >= self.smooth_window:
            smoothed = float(np.mean(self._density_buffer[-self.smooth_window:]))
        else:
            smoothed = edge_density

        self._smoothed_buffer.append(smoothed)
        if len(self._smoothed_buffer) > self._max_buffer:
            self._smoothed_buffer = self._smoothed_buffer[-self._max_buffer:]

        # Schmitt-trigger
        is_fused = bool(self._fused_buffer[-1]) if self._fused_buffer else False
        if not is_fused and smoothed >= self.enter_threshold:
            is_fused = True
        elif is_fused and smoothed <= self.exit_threshold:
            is_fused = False

        self._fused_buffer.append(int(is_fused))
        if len(self._fused_buffer) > self._max_buffer:
            self._fused_buffer = self._fused_buffer[-self._max_buffer:]

        # 分类
        same_direction = (trend_1h_sign * trend_4h_sign) > 0

        if is_fused:
            candidate = MarketRegime.FUSED
        elif abs(correlation) < 0.3:
            candidate = MarketRegime.DIVERGENT
        elif abs(correlation) < 0.5 and not same_direction:
            candidate = MarketRegime.DIVERGENT
        else:
            candidate = MarketRegime.NEUTRAL

        # 滞后处理
        if candidate != self.state.current_regime:
            if candidate == self.candidate_regime:
                self.candidate_count += 1
            else:
                self.candidate_regime = candidate
                self.candidate_count = 1

            if self.candidate_count >= self.state.min_regime_duration:
                self._transition_to(candidate)
                self.candidate_regime = None
                self.candidate_count = 0
        else:
            self.state.regime_duration += 1
            self.candidate_regime = None
            self.candidate_count = 0

        self.state.edge_density = smoothed

        # 记录历史
        self.state.regime_history.append((timestamp, self.state.current_regime.value))
        if len(self.state.regime_history) > self.max_history:
            self.state.regime_history = self.state.regime_history[-self.max_history:]

        return self.state.current_regime

    def _transition_to(self, new_regime: MarketRegime):
        """执行体制切换。"""
        old_regime = self.state.current_regime

        key = (old_regime.value, new_regime.value)
        self.transition_counts[key] = self.transition_counts.get(key, 0) + 1

        self.state.previous_regime = old_regime
        self.state.current_regime = new_regime
        self.state.transition_count += 1
        self.state.regime_duration = 0

        logger.info(
            "[%s] Regime transition: %s -> %s (transition #%d, duration=%d)",
            self.symbol or "?",
            old_regime.value,
            new_regime.value,
            self.state.transition_count,
            self.state.regime_duration,
        )

    def get_transition_probability(self, from_regime: str, to_regime: str) -> float:
        """获取体制过渡概率。"""
        key = (from_regime, to_regime)
        count = self.transition_counts.get(key, 0)
        total = sum(c for k, c in self.transition_counts.items() if k[0] == from_regime)
        return count / total if total > 0 else 0.0

    def get_fused_episodes(self) -> List[Dict[str, Optional[str]]]:
        """获取历史 FUSED 区间列表。

        Returns:
            [{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD" | None}, ...]
            end=None 表示当前仍在 FUSED 中
        """
        if not self._fused_buffer or len(self._fused_buffer) < 2:
            return []

        from datetime import datetime
        episodes: List[Dict[str, Optional[str]]] = []
        start_idx: Optional[int] = None

        for i, state in enumerate(self._fused_buffer):
            if state:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                # 从 regime_history 获取日期
                start_date = self._get_date_for_index(start_idx)
                end_date = self._get_date_for_index(i - 1)
                episodes.append({"start": start_date, "end": end_date})
                start_idx = None

        if start_idx is not None:
            start_date = self._get_date_for_index(start_idx)
            episodes.append({"start": start_date, "end": None})

        return episodes

    def _get_date_for_index(self, idx: int) -> str:
        """从 regime_history 获取索引对应的日期。"""
        from datetime import datetime, timezone
        if idx < len(self.state.regime_history):
            ts = self.state.regime_history[idx][0]
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def get_regime_summary(self) -> Dict:
        """获取体制摘要。"""
        regime_counts: Dict[str, int] = {}
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
            "edge_density": round(self.state.edge_density, 4),
            "is_fused": bool(self._fused_buffer[-1]) if self._fused_buffer else False,
            "fused_episodes": self.get_fused_episodes(),
            "top_transitions": sorted(
                self.transition_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }


# ===========================================================================
# 兼容层：与现有 alpha_zoo.CausalHysteresisRegime 接口兼容
# ===========================================================================

class CausalHysteresisRegime(EnhancedRegimeDetector):
    """兼容旧接口的因果滞后体制检测器。

    继承 EnhancedRegimeDetector，提供与 alpha_zoo.CausalHysteresisRegime
    完全兼容的接口，同时内部使用增强版 Schmitt-trigger 实现。
    """

    def __init__(
        self,
        hysteresis_threshold: float = 0.15,
        min_regime_duration: int = 5,
        max_history: int = 100,
        # 修复 2026-08-07: 透传 EnhancedRegimeDetector 参数 —
        # 原实现只透传 hysteresis/min_duration, config 的 regime_enter_
        # threshold/exit/corr_window/smooth_window 全部未生效。
        enter_threshold: float = 0.65,
        exit_threshold: float = 0.45,
        corr_window: int = 60,
        smooth_window: int = 5,
        symbol: str = "",
    ):
        super().__init__(
            hysteresis_threshold=hysteresis_threshold,
            min_regime_duration=min_regime_duration,
            max_history=max_history,
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold,
            corr_window=corr_window,
            smooth_window=smooth_window,
            symbol=symbol,
        )