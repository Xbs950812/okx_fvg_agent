"""
真实交易成本模型。

包含：
  - maker / taker 手续费
  - 资金费率估算
  - 滑点估算（基于 spread、volatility、深度）

所有回测与实盘 PnL 计算都应通过 CostModel 得到 gross/net。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    """交易成本估算结果。"""
    commission_open: float
    commission_close: float
    funding_estimate: float
    slippage_estimate: float
    total: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commission_open": self.commission_open,
            "commission_close": self.commission_close,
            "funding_estimate": self.funding_estimate,
            "slippage_estimate": self.slippage_estimate,
            "total": self.total,
        }


class CostModel:
    """交易成本模型。"""

    def __init__(
        self,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        use_taker_open: bool = False,
        use_taker_close: bool = True,
        slippage_base_pct: float = 0.05,
        slippage_vol_factor: float = 1.0,
        funding_hours: int = 8,
    ):
        """
        Args:
            maker_fee: maker 手续费（%）
            taker_fee: taker 手续费（%）
            use_taker_open: 开仓是否按 taker 成交
            use_taker_close: 平仓是否按 taker 成交
            slippage_base_pct: 基础滑点（%）
            slippage_vol_factor: 波动率对滑点的放大系数
            funding_hours: 资金费率结算间隔（小时）
        """
        self.maker_fee = maker_fee / 100.0
        self.taker_fee = taker_fee / 100.0
        self.use_taker_open = use_taker_open
        self.use_taker_close = use_taker_close
        self.slippage_base_pct = slippage_base_pct / 100.0
        self.slippage_vol_factor = slippage_vol_factor
        self.funding_hours = funding_hours

    # ------------------------------------------------------------------
    # 单笔交易估算
    # ------------------------------------------------------------------
    def estimate_trade(
        self,
        position_value: float,
        hold_hours: float,
        funding_rate: float = 0.0,
        spread_pct: float = 0.0,
        volatility: float = 0.0,
    ) -> CostEstimate:
        """估算一笔交易的总成本。

        Args:
            position_value: 仓位名义价值
            hold_hours: 预计持仓小时数
            funding_rate: 当前资金费率（年化或单次，按 OKX 单次值传入）
            spread_pct: 买卖价差 %
            volatility: 小时波动率（如 ATR/price）
        """
        fee_open = position_value * (self.taker_fee if self.use_taker_open else self.maker_fee)
        fee_close = position_value * (self.taker_fee if self.use_taker_close else self.maker_fee)

        # 资金费：按持仓小时数估算支付次数
        n_funding = max(1, int(hold_hours / self.funding_hours))
        funding_cost = position_value * funding_rate * n_funding

        # 滑点：基础滑点 + 波动率惩罚 + spread 惩罚
        slippage = self.slippage_base_pct
        slippage += volatility * self.slippage_vol_factor
        slippage += max(0, spread_pct / 100.0)
        slippage_cost = position_value * slippage

        return CostEstimate(
            commission_open=fee_open,
            commission_close=fee_close,
            funding_estimate=funding_cost,
            slippage_estimate=slippage_cost,
            total=fee_open + fee_close + funding_cost + slippage_cost,
        )

    # ------------------------------------------------------------------
    # 应用于回测交易
    # ------------------------------------------------------------------
    def apply_to_backtest_trade(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        direction: str,
        hold_bars: int = 1,
        bar_hours: int = 1,
        funding_rate: float = 0.0,
        spread_pct: float = 0.02,
        volatility: float = 0.0,
    ) -> Dict[str, float]:
        """对回测单笔交易计算 gross/net PnL。"""
        notional = quantity * entry_price
        hold_hours = hold_bars * bar_hours
        cost = self.estimate_trade(
            position_value=notional,
            hold_hours=hold_hours,
            funding_rate=funding_rate,
            spread_pct=spread_pct,
            volatility=volatility,
        )

        if direction == "long":
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        return {
            "gross_pnl": gross_pnl,
            "total_cost": cost.total,
            "net_pnl": gross_pnl - cost.total,
            "cost_detail": cost.to_dict(),
        }

    # ------------------------------------------------------------------
    # 与配置集成
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CostModel":
        backtest = config.get("backtest", {})
        risk = config.get("risk", {})
        return cls(
            maker_fee=backtest.get("commission", 0.0005) * 100.0,  # config 中为小数
            taker_fee=backtest.get("taker_fee", backtest.get("commission", 0.0005)) * 100.0,
            use_taker_open=backtest.get("use_taker_open", False),
            use_taker_close=backtest.get("use_taker_close", True),
            slippage_base_pct=backtest.get("slippage", 0.0001) * 100.0,
        )
