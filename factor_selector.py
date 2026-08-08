"""
FactorSelector：因子选择与降维。

目标：从 FactorZoo 的 461 个因子中筛选出 20~50 个核心因子，降低过拟合风险。

评估维度：
  - 因子相关性（剔除高度共线）
  - 因子贡献（IC / 与收益的预测能力）
  - 因子稳定性（IC 时序标准差）
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None
    import logging as _logging
    _logging.getLogger(__name__).warning("scipy not installed, factor_selector will use numpy fallback for IC")

logger = logging.getLogger(__name__)


@dataclass
class FactorMetrics:
    """单个因子评估指标。"""
    name: str
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0        # IC 均值 / IC 标准差
    corr_max: float = 0.0     # 与其他因子最大相关性
    contribution_score: float = 0.0


class FactorSelector:
    """因子选择器。"""

    def __init__(
        self,
        target_factor_count: int = 35,
        max_correlation: float = 0.7,
        min_ic_abs: float = 0.03,
        min_ic_ir: float = 0.3,
        lookback_window: int = 60,
    ):
        """
        Args:
            target_factor_count: 目标保留因子数
            max_correlation: 因子间最大允许相关性
            min_ic_abs: 最小 |IC| 阈值
            min_ic_ir: 最小 IC IR 阈值
            lookback_window: 滚动 IC 计算窗口
        """
        self.target_factor_count = target_factor_count
        self.max_correlation = max_correlation
        self.min_ic_abs = min_ic_abs
        self.min_ic_ir = min_ic_ir
        self.lookback_window = lookback_window

    # ------------------------------------------------------------------
    # 核心选择流程
    # ------------------------------------------------------------------
    def select(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> Tuple[List[str], Dict[str, FactorMetrics]]:
        """选择核心因子。

        Args:
            factor_df: index=date, columns=factor_names
            forward_returns: index=date, 值=未来一期收益

        Returns:
            (selected_factors, metrics_dict)
        """
        factors = [c for c in factor_df.columns if c != "returns"]
        if not factors:
            return [], {}

        # 对齐
        aligned = factor_df.join(forward_returns.rename("returns"), how="inner").dropna()
        if aligned.empty:
            logger.warning("[FactorSelector] empty aligned data")
            return [], {}

        # 1. 计算 IC 指标
        metrics = {}
        for f in factors:
            ic_series = _rolling_ic(
                aligned[f].values,
                aligned["returns"].values,
                window=self.lookback_window,
            )
            if len(ic_series) == 0:
                continue
            ic_mean = float(np.nanmean(ic_series))
            ic_std = float(np.nanstd(ic_series)) if len(ic_series) > 1 else 1e-6
            metrics[f] = FactorMetrics(
                name=f,
                ic_mean=ic_mean,
                ic_std=ic_std,
                ic_ir=ic_mean / ic_std if ic_std > 0 else 0.0,
            )

        # 2. 初筛：|IC| 与 IR 阈值
        passed = {
            k: v for k, v in metrics.items()
            if abs(v.ic_mean) >= self.min_ic_abs and v.ic_ir >= self.min_ic_ir
        }
        if not passed:
            logger.warning("[FactorSelector] no factor passed IC filter")
            return [], metrics

        # 3. 计算因子相关性并剔除高共线
        corr_matrix = aligned[list(passed.keys())].corr(method="spearman").abs()
        ranked = sorted(
            passed.values(),
            key=lambda m: (abs(m.ic_ir), abs(m.ic_mean)),
            reverse=True,
        )
        selected: List[str] = []
        for m in ranked:
            if len(selected) >= self.target_factor_count:
                break
            # 与已选因子的最大相关性
            if selected:
                max_corr = max(corr_matrix.loc[m.name, s] for s in selected)
            else:
                max_corr = 0.0
            m.corr_max = max_corr
            if max_corr <= self.max_correlation:
                selected.append(m.name)

        # 4. 计算贡献分
        for m in metrics.values():
            m.contribution_score = _contribution_score(m, self.max_correlation)

        logger.info(
            f"[FactorSelector] {len(factors)} factors -> {len(selected)} selected "
            f"(target={self.target_factor_count})"
        )
        return selected, metrics

    def report(self, metrics: Dict[str, FactorMetrics]) -> pd.DataFrame:
        """生成因子评估报告。"""
        rows = []
        for m in sorted(metrics.values(), key=lambda x: x.contribution_score, reverse=True):
            rows.append({
                "factor": m.name,
                "ic_mean": m.ic_mean,
                "ic_std": m.ic_std,
                "ic_ir": m.ic_ir,
                "corr_max": m.corr_max,
                "contribution": m.contribution_score,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _spearman_fallback(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation fallback when scipy is not available."""
    if spearmanr is not None:
        try:
            r, _ = spearmanr(x, y, nan_policy="omit")
            return float(r) if not np.isnan(r) else 0.0
        except Exception:
            pass
    # numpy fallback: correlate ranks
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return 0.0
    xr = np.argsort(np.argsort(x[mask])).astype(float)
    yr = np.argsort(np.argsort(y[mask])).astype(float)
    r = np.corrcoef(xr, yr)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def _rolling_ic(factor: np.ndarray, ret: np.ndarray, window: int = 60) -> np.ndarray:
    """滚动 Spearman IC。"""
    n = len(factor)
    if n < window + 1:
        if len(factor) > 1 and len(ret) > 1:
            try:
                r = _spearman_fallback(factor, ret)
                return np.array([r])
            except Exception:
                return np.array([])
        return np.array([])

    ics = []
    for i in range(window, n):
        x = factor[i - window:i]
        y = ret[i - window:i]
        try:
            r = _spearman_fallback(x, y)
            ics.append(r)
        except Exception:
            ics.append(np.nan)
    return np.array(ics, dtype=float)


def _contribution_score(m: FactorMetrics, max_corr_target: float) -> float:
    """综合贡献分：IC IR 经相关性惩罚。"""
    if m.ic_std <= 0:
        return 0.0
    corr_penalty = max(0.0, 1.0 - m.corr_max / max_corr_target)
    return abs(m.ic_ir) * corr_penalty
