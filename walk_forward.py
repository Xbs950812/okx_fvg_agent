"""
Walk Forward Analysis：滚动前向验证。

流程：
  1. 将历史数据按时间分为训练期（如 70%）和验证期（如 30%）。
  2. 在训练期上训练/优化策略参数。
  3. 在验证期上运行回测，得到样本外收益。
  4. 对比训练收益与验证收益，若衰减过大则标记 OVERFIT_WARNING。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """ Walk Forward 结果。"""
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""
    train_return: float = 0.0
    test_return: float = 0.0
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    decay_ratio: float = 0.0       # test_return / train_return
    overfit: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_return": self.train_return,
            "test_return": self.test_return,
            "train_sharpe": self.train_sharpe,
            "test_sharpe": self.test_sharpe,
            "decay_ratio": self.decay_ratio,
            "overfit": self.overfit,
            "notes": self.notes,
        }


class WalkForwardAnalyzer:
    """滚动前向验证器。

    支持两种模式：
      - 单次切分（train_ratio=0.7）
      - 多窗口滚动（n_windows）
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        n_windows: int = 1,
        overfit_threshold: float = 0.3,
        min_test_return: float = -0.5,
    ):
        """
        Args:
            train_ratio: 单次切分训练集比例
            n_windows: 滚动窗口数，>1 时启用滚动 Walk Forward
            overfit_threshold: 验证收益 < 训练收益 * threshold 视为过拟合
            min_test_return: 验证收益低于此值也视为过拟合
        """
        self.train_ratio = train_ratio
        self.n_windows = max(1, n_windows)
        self.overfit_threshold = overfit_threshold
        self.min_test_return = min_test_return

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def analyze(
        self,
        df: pd.DataFrame,
        backtest_fn: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]],
        param_optimizer: Optional[Callable[[pd.DataFrame], Dict[str, Any]]] = None,
        strategy_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[WalkForwardResult]:
        """执行 Walk Forward 分析。

        Args:
            df: 包含 OHLCV 等数据的 DataFrame，index 为时间
            backtest_fn: 回测函数，接收 (df, params) -> {"total_return": float, "sharpe_ratio": float, ...}
            param_optimizer: 参数优化函数，接收 train_df -> params dict；为 None 时使用默认参数
            strategy_kwargs: 默认策略参数字典

        Returns:
            WalkForwardResult 列表
        """
        if df.empty:
            logger.warning("[WalkForward] empty dataframe")
            return []

        df = df.sort_index()
        if self.n_windows == 1:
            return [self._run_single_split(df, backtest_fn, param_optimizer, strategy_kwargs)]
        return self._run_rolling(df, backtest_fn, param_optimizer, strategy_kwargs)

    # ------------------------------------------------------------------
    # 单次切分
    # ------------------------------------------------------------------
    def _run_single_split(
        self,
        df: pd.DataFrame,
        backtest_fn: Callable,
        param_optimizer: Optional[Callable],
        strategy_kwargs: Optional[Dict[str, Any]],
    ) -> WalkForwardResult:
        split_idx = int(len(df) * self.train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        params = self._optimize_or_default(train_df, backtest_fn, param_optimizer, strategy_kwargs)

        train_res = backtest_fn(train_df, params)
        test_res = backtest_fn(test_df, params)

        return self._build_result(train_df, test_df, train_res, test_res)

    # ------------------------------------------------------------------
    # 滚动切分
    # ------------------------------------------------------------------
    def _run_rolling(
        self,
        df: pd.DataFrame,
        backtest_fn: Callable,
        param_optimizer: Optional[Callable],
        strategy_kwargs: Optional[Dict[str, Any]],
    ) -> List[WalkForwardResult]:
        n = len(df)
        results = []
        # 每个窗口 test 占剩余部分的 1/n_windows
        test_ratio = 1.0 / self.n_windows
        for i in range(self.n_windows):
            test_start_idx = int(n * self.train_ratio + n * (1 - self.train_ratio) * i / self.n_windows)
            test_end_idx = int(n * self.train_ratio + n * (1 - self.train_ratio) * (i + 1) / self.n_windows)
            if test_start_idx >= n or test_start_idx >= test_end_idx:
                break
            train_df = df.iloc[:test_start_idx]
            test_df = df.iloc[test_start_idx:test_end_idx]

            params = self._optimize_or_default(train_df, backtest_fn, param_optimizer, strategy_kwargs)
            train_res = backtest_fn(train_df, params)
            test_res = backtest_fn(test_df, params)

            result = self._build_result(train_df, test_df, train_res, test_res)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # 参数优化 / 默认参数
    # ------------------------------------------------------------------
    def _optimize_or_default(
        self,
        train_df: pd.DataFrame,
        backtest_fn: Callable,
        param_optimizer: Optional[Callable],
        strategy_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if param_optimizer is not None:
            try:
                return param_optimizer(train_df)
            except Exception as e:
                logger.warning(f"[WalkForward] optimizer failed: {e}, using default params")
        return strategy_kwargs or {}

    # ------------------------------------------------------------------
    # 构建结果
    # ------------------------------------------------------------------
    def _build_result(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_res: Dict[str, Any],
        test_res: Dict[str, Any],
    ) -> WalkForwardResult:
        train_ret = float(train_res.get("total_return", 0.0))
        test_ret = float(test_res.get("total_return", 0.0))
        train_sharpe = float(train_res.get("sharpe_ratio", 0.0))
        test_sharpe = float(test_res.get("sharpe_ratio", 0.0))

        decay = test_ret / train_ret if train_ret != 0 else 0.0
        overfit = False
        notes = []

        if train_ret > 0 and test_ret < train_ret * self.overfit_threshold:
            overfit = True
            notes.append(
                f"验证收益 {test_ret:.2%} 低于训练收益 {train_ret:.2%} 的 {self.overfit_threshold:.0%}，"
                f"衰减比例 {decay:.2f}"
            )
        if test_ret < self.min_test_return:
            overfit = True
            notes.append(f"验证收益 {test_ret:.2%} 低于最低阈值 {self.min_test_return:.2%}")

        return WalkForwardResult(
            train_start=str(train_df.index[0]),
            train_end=str(train_df.index[-1]),
            test_start=str(test_df.index[0]),
            test_end=str(test_df.index[-1]),
            train_return=train_ret,
            test_return=test_ret,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            decay_ratio=decay,
            overfit=overfit,
            notes=notes,
        )

    def summary(self, results: List[WalkForwardResult]) -> Dict[str, Any]:
        """汇总多窗口结果。"""
        if not results:
            return {}
        overfit_count = sum(1 for r in results if r.overfit)
        avg_decay = np.mean([r.decay_ratio for r in results])
        return {
            "windows": len(results),
            "overfit_windows": overfit_count,
            "overfit_rate": overfit_count / len(results),
            "avg_decay_ratio": avg_decay,
            "avg_train_return": np.mean([r.train_return for r in results]),
            "avg_test_return": np.mean([r.test_return for r in results]),
        }
