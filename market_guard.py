"""
MarketEmergencyGuard：市场状态熔断机制。

检测维度：
  - BTC 24h 涨跌幅
  - BTC 滚动波动率
  - 全市场跌幅比例（breadth）
  - 资金费率极端值
  - 持仓量异常变化

进入 CRISIS_MODE 后禁止开仓，只允许止损/减仓。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """市场状态快照。"""
    timestamp: float
    btc_return_24h: float = 0.0
    btc_volatility_24h: float = 0.0
    market_breadth: float = 1.0          # 正收益币种比例 (0-1)
    funding_extreme: float = 0.0          # 最大绝对资金费率
    oi_change_pct: float = 0.0            # BTC OI 24h 变化
    regime: str = "NORMAL"                # NORMAL | WARNING | CRISIS
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "btc_return_24h": self.btc_return_24h,
            "btc_volatility_24h": self.btc_volatility_24h,
            "market_breadth": self.market_breadth,
            "funding_extreme": self.funding_extreme,
            "oi_change_pct": self.oi_change_pct,
            "regime": self.regime,
            "reasons": self.reasons,
        }


class MarketEmergencyGuard:
    """市场熔断守卫。"""

    def __init__(
        self,
        btc_crisis_drop_pct: float = -8.0,
        volatility_spike_threshold: float = 5.0,
        breadth_crisis_threshold: float = 0.2,
        funding_extreme_threshold: float = 0.01,
        oi_spike_threshold: float = 0.3,
        recovery_cooldown_seconds: float = 1800.0,
    ):
        """
        Args:
            btc_crisis_drop_pct: BTC 24h 跌幅触发危机模式（%）
            volatility_spike_threshold: BTC 年化波动率相对近期均值倍数
            breadth_crisis_threshold: 市场正收益币种比例低于此值触发危机
            funding_extreme_threshold: 绝对资金费率超过此值视为极端
            oi_spike_threshold: OI 24h 变化超过此比例视为异常
            recovery_cooldown_seconds: 危机模式最短持续秒数
        """
        self.btc_crisis_drop_pct = btc_crisis_drop_pct
        self.volatility_spike_threshold = volatility_spike_threshold
        self.breadth_crisis_threshold = breadth_crisis_threshold
        self.funding_extreme_threshold = funding_extreme_threshold
        self.oi_spike_threshold = oi_spike_threshold
        self.recovery_cooldown_seconds = recovery_cooldown_seconds

        self._crisis_entered_at: Optional[float] = None
        self._last_state: Optional[MarketState] = None
        self._prev_oi: float = 0.0  # 上一轮 OI，用于计算 24h 变化

    # ------------------------------------------------------------------
    # 状态评估
    # ------------------------------------------------------------------
    def evaluate(
        self,
        btc_return_24h_pct: float,
        btc_prices: List[float],
        market_returns: Dict[str, float],
        funding_rates: Dict[str, float],
        open_interest_change_pct: float = 0.0,
    ) -> MarketState:
        """评估当前市场状态。"""
        now = time.time()
        reasons: List[str] = []

        # 数据完整性检查：若关键数据全部缺失，API 可能全挂，返回 UNKNOWN
        data_available = bool(btc_prices) or bool(market_returns) or bool(funding_rates)
        if not data_available:
            reasons.append("API 数据不可用，无法评估市场状态")
            return MarketState(
                timestamp=now,
                btc_return_24h=btc_return_24h_pct,
                btc_volatility_24h=0.0,
                market_breadth=1.0,
                funding_extreme=0.0,
                oi_change_pct=0.0,
                regime="UNKNOWN",
                reasons=reasons,
            )

        # BTC 波动率：24h 对数收益率年化
        btc_vol = _annualized_volatility(btc_prices)
        # 近期波动率均值（7 日近似：用更长序列最后 168 个点）
        recent_vol = _recent_volatility(btc_prices, lookback=168)
        vol_spike = btc_vol / recent_vol if recent_vol > 0 else 1.0

        # 市场广度
        positive = sum(1 for r in market_returns.values() if r > 0)
        breadth = positive / len(market_returns) if market_returns else 1.0

        # 资金费率极端值
        funding_extreme = max((abs(v) for v in funding_rates.values()), default=0.0)

        regime = "NORMAL"
        if btc_return_24h_pct <= self.btc_crisis_drop_pct:
            regime = "CRISIS"
            reasons.append(f"BTC 24h 跌幅 {btc_return_24h_pct:.2f}% <= 阈值 {self.btc_crisis_drop_pct:.2f}%")
        if vol_spike >= self.volatility_spike_threshold:
            regime = "CRISIS"
            reasons.append(f"BTC 波动率飙升 {vol_spike:.1f}x")
        if breadth <= self.breadth_crisis_threshold:
            regime = "CRISIS"
            reasons.append(f"市场广度 {breadth:.1%} <= 阈值 {self.breadth_crisis_threshold:.1%}")
        if funding_extreme >= self.funding_extreme_threshold:
            if regime == "NORMAL":
                regime = "WARNING"
            reasons.append(f"资金费率极端 {funding_extreme:.3%}")
        if abs(open_interest_change_pct) >= self.oi_spike_threshold:
            if regime == "NORMAL":
                regime = "WARNING"
            reasons.append(f"持仓量异常变化 {open_interest_change_pct:.1%}")

        # 危机模式冷却：进入后至少持续 cooldown 秒，避免频繁切换
        if regime != "CRISIS" and self._crisis_entered_at is not None:
            if now - self._crisis_entered_at < self.recovery_cooldown_seconds:
                regime = "CRISIS"
                reasons.append(
                    f"危机模式冷却中，还需 {self.recovery_cooldown_seconds - (now - self._crisis_entered_at):.0f}s"
                )
            else:
                self._crisis_entered_at = None

        if regime == "CRISIS" and self._crisis_entered_at is None:
            self._crisis_entered_at = now

        state = MarketState(
            timestamp=now,
            btc_return_24h=btc_return_24h_pct,
            btc_volatility_24h=btc_vol,
            market_breadth=breadth,
            funding_extreme=funding_extreme,
            oi_change_pct=open_interest_change_pct,
            regime=regime,
            reasons=reasons,
        )
        self._last_state = state
        return state

    def can_open_new_position(self, state: Optional[MarketState] = None) -> bool:
        """危机模式或未知状态下禁止开仓。"""
        s = state or self._last_state
        if s is None:
            return True
        return s.regime not in ("CRISIS", "UNKNOWN")

    def reduce_position_factor(self, state: Optional[MarketState] = None) -> float:
        """根据状态返回仓位缩放系数。"""
        s = state or self._last_state
        if s is None:
            return 1.0
        if s.regime == "CRISIS":
            return 0.0
        if s.regime == "WARNING":
            return 0.5
        return 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _annualized_volatility(prices: List[float], periods_per_year: float = 365 * 24) -> float:
    if len(prices) < 2:
        return 0.0
    rets = np.diff(np.log(np.maximum(prices, 1e-12)))
    return float(np.std(rets) * np.sqrt(periods_per_year))


def _recent_volatility(prices: List[float], lookback: int = 168) -> float:
    if len(prices) < 2:
        return 0.0
    recent = prices[-lookback:] if len(prices) > lookback else prices
    rets = np.diff(np.log(np.maximum(recent, 1e-12)))
    return float(np.std(rets) * np.sqrt(365 * 24))
