"""
因子库适配器 — 桥接 Vibe-Trading Alpha Zoo 461 因子与现有 agent。

将 Vibe-Trading 的 Alpha Zoo 因子框架适配到 okx_fvg_agent 的数据格式：
  - 将 OKX K 线数据转换为 panel 格式 (pd.DataFrame, index=时间, columns=币种)
  - 调用 Registry.compute(alpha_id, panel) 计算因子值
  - 对加密货币上下文进行因子筛选和定制

HunHeng_OS_V1.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorZooAdapter:
    """因子库适配器 — 将 Alpha Zoo 因子集成到现有 FVG 策略中。

    用法:
        adapter = FactorZooAdapter()
        adapter.load_registry()

        # 构建 panel
        panel = adapter.build_panel(candles_dict)

        # 计算因子
        result = adapter.compute("alpha101_alpha_001", panel)

        # 列出可用因子
        crypto_factors = adapter.list_crypto_factors()
    """

    # 加密货币相关的因子分类
    CRYPTO_THEMES = {"momentum", "reversal", "volume", "volatility", "liquidity", "microstructure"}

    # 加密货币 Universe
    CRYPTO_UNIVERSE = "crypto"

    def __init__(self):
        self._registry = None
        self._loaded = False
        self._available_factors: List[str] = []
        self._crypto_factors: List[str] = []

    def load_registry(self) -> bool:
        """加载因子注册表，扫描所有 zoo 目录。"""
        try:
            from factor_zoo.registry import Registry
            self._registry = Registry()
            # Registry.__init__ already calls _scan() to discover all alphas
            self._loaded = True
            self._available_factors = self._registry.list()
            self._crypto_factors = self._registry.list(universe=self.CRYPTO_UNIVERSE)
            logger.info(
                "FactorZoo loaded: %d total factors, %d crypto-compatible",
                len(self._available_factors),
                len(self._crypto_factors),
            )
            return True
        except Exception as e:
            logger.error("FactorZoo load failed: %s", e)
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def build_panel(
        self,
        candles_dict: Dict[str, pd.DataFrame],
        include_vwap: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """将多个币种的 K 线数据构建为 panel 格式。

        Panel 格式: {column_name: DataFrame(index=datetime, columns=instrument)}
        这是 Alpha Zoo 因子计算的标准输入格式。

        Args:
            candles_dict: {inst_id: DataFrame(columns=[open,high,low,close,volume])}
            include_vwap: 是否计算 VWAP

        Returns:
            panel dict: {"open": df, "high": df, "low": df, "close": df, "volume": df}
        """
        panel: Dict[str, pd.DataFrame] = {}

        for col in ["open", "high", "low", "close", "volume"]:
            frames = {}
            for inst_id, df in candles_dict.items():
                if col in df.columns:
                    series = df[col].copy()
                    series.name = inst_id
                    frames[inst_id] = series
            if frames:
                panel[col] = pd.concat(frames, axis=1)

        if include_vwap and "high" in panel and "low" in panel and "close" in panel and "open" in panel:
            panel["vwap"] = (panel["high"] + panel["low"] + panel["close"] + panel["open"]) / 4.0

        return panel

    def compute(self, alpha_id: str, panel: Dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """计算单个因子值。

        Args:
            alpha_id: 因子 ID，如 "alpha101_alpha_001"
            panel: 面板数据

        Returns:
            因子值 DataFrame，或 None
        """
        if not self._loaded:
            logger.warning("FactorZoo not loaded, call load_registry() first")
            return None

        try:
            return self._registry.compute(alpha_id, panel)
        except Exception as e:
            logger.debug("FactorZoo compute %s failed: %s", alpha_id, e)
            return None

    def compute_batch(
        self,
        alpha_ids: List[str],
        panel: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """批量计算多个因子。

        Args:
            alpha_ids: 因子 ID 列表
            panel: 面板数据

        Returns:
            {alpha_id: factor_values_df}
        """
        results = {}
        for alpha_id in alpha_ids:
            result = self.compute(alpha_id, panel)
            if result is not None and not result.empty:
                results[alpha_id] = result
        return results

    def list_all(self) -> List[str]:
        """列出所有可用因子。"""
        if not self._loaded:
            return []
        return self._available_factors

    def list_crypto_factors(self) -> List[str]:
        """列出加密货币兼容的因子。"""
        if not self._loaded:
            return []
        return self._crypto_factors

    def list_by_theme(self, theme: str) -> List[str]:
        """按主题列出因子。"""
        if not self._loaded:
            return []
        return self._registry.list(theme=theme)

    def get_factor_meta(self, alpha_id: str) -> Dict[str, Any] | None:
        """获取因子元数据。"""
        if not self._loaded:
            return None
        alpha = self._registry.get(alpha_id)
        if alpha:
            return alpha.meta
        return None

    def get_registry_health(self) -> Dict[str, Any]:
        """获取注册表健康状态。"""
        if not self._loaded:
            return {"loaded": False}
        return self._registry.health()

    # ------------------------------------------------------------------
    # 便捷方法：直接计算加密货币常用因子
    # ------------------------------------------------------------------

    def compute_momentum_factors(
        self,
        candles_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """计算动量类因子综合得分。

        Returns:
            DataFrame(index=datetime, columns=instrument)，值为综合动量得分
        """
        panel = self.build_panel(candles_dict)
        momentum_alphas = self.list_by_theme("momentum")
        crypto_momentum = [a for a in momentum_alphas if a in self._crypto_factors]

        if not crypto_momentum:
            return pd.DataFrame()

        results = self.compute_batch(crypto_momentum[:10], panel)  # 最多 10 个
        if not results:
            return pd.DataFrame()

        # 等权合成
        first_key = list(results.keys())[0]
        combined = pd.DataFrame(0.0, index=results[first_key].index, columns=results[first_key].columns)
        for df in results.values():
            aligned = df.reindex_like(combined)
            combined = combined.add(aligned.fillna(0), fill_value=0)

        return combined / len(results)

    def compute_volatility_factors(
        self,
        candles_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """计算波动率类因子综合得分。"""
        panel = self.build_panel(candles_dict)
        vol_alphas = self.list_by_theme("volatility")
        crypto_vol = [a for a in vol_alphas if a in self._crypto_factors]

        if not crypto_vol:
            return pd.DataFrame()

        results = self.compute_batch(crypto_vol[:10], panel)
        if not results:
            return pd.DataFrame()

        first_key = list(results.keys())[0]
        combined = pd.DataFrame(0.0, index=results[first_key].index, columns=results[first_key].columns)
        for df in results.values():
            aligned = df.reindex_like(combined)
            combined = combined.add(aligned.fillna(0), fill_value=0)

        return combined / len(results)