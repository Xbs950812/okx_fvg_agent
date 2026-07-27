"""
优化引擎 — 融合 freqtrade Hyperopt + Edge 分析 + FreqAI 自适应思想。

借鉴来源：freqtrade (52k ⭐)
  - Hyperopt: 参数空间搜索 + 损失函数优化
  - Edge: 胜率/期望值/盈亏比统计
  - FreqAI: 自适应特征工程 + 在线学习

核心能力：
  - 参数网格搜索（简化版 Hyperopt）
  - 交易统计（Edge 模块）
  - 自适应参数调整（基于近 N 笔表现）
  - Trailing Stop 动态止损
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np


logger = logging.getLogger(__name__)


# ===========================================================================
# 交易记录
# ===========================================================================

@dataclass
class TradeRecord:
    """单笔交易记录。"""
    symbol: str
    direction: str           # "long" | "short"
    entry_time: float        # 入场时间戳
    exit_time: float = 0.0   # 出场时间戳
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    leverage: int = 1
    pnl: float = 0.0         # 实现盈亏
    pnl_pct: float = 0.0     # 盈亏百分比
    is_win: bool = False
    exit_reason: str = ""    # tp / sl / manual / signal
    fvg_score: float = 0.0   # 入场时的 FVG 评分
    master_score: float = 0.0  # 入场时的专家评分
    regime: str = ""         # 入场时的市场体制


# ===========================================================================
# Edge 分析（胜率/期望值/盈亏比）
# ===========================================================================

@dataclass
class EdgeStats:
    """交易统计 — 借鉴 freqtrade Edge 模块。"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0        # 总盈利 / 总亏损
    expectancy: float = 0.0           # 期望值 (每笔交易)
    expectancy_ratio: float = 0.0     # 期望值比率 (expectancy / avg_loss)
    sharpe_ratio: float = 0.0         # 近似 Sharpe
    max_drawdown_pct: float = 0.0     # 最大回撤
    consecutive_losses: int = 0       # 当前连亏数
    max_consecutive_losses: int = 0   # 最大连亏数
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_holding_hours: float = 0.0

    # 按方向分组
    long_stats: Dict[str, float] = field(default_factory=dict)
    short_stats: Dict[str, float] = field(default_factory=dict)

    # 按体制分组
    regime_stats: Dict[str, dict] = field(default_factory=dict)

    # 按评分分组
    score_bucket_stats: Dict[str, dict] = field(default_factory=dict)


class EdgeAnalyzer:
    """Edge 分析器 — 统计交易数据，计算胜率/期望值/盈亏比。

    借鉴 freqtrade Edge 模块：统计每笔交易，计算：
      - 期望值 = win_rate * avg_win - (1-win_rate) * avg_loss
      - 盈亏比 = avg_win / avg_loss
      - 盈利因子 = Σ盈利 / Σ亏损
    """

    def __init__(self):
        self.trades: List[TradeRecord] = []

    def add_trade(self, trade: TradeRecord):
        self.trades.append(trade)

    def analyze(self, lookback: int = 100) -> EdgeStats:
        """分析最近 N 笔交易。"""
        recent = self.trades[-lookback:] if len(self.trades) > lookback else self.trades
        if not recent:
            return EdgeStats()

        stats = EdgeStats()
        stats.total_trades = len(recent)

        wins = [t for t in recent if t.is_win]
        losses = [t for t in recent if not t.is_win]
        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.win_rate = stats.winning_trades / stats.total_trades if stats.total_trades > 0 else 0

        stats.total_pnl = sum(t.pnl for t in recent)
        stats.avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        stats.avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0

        total_profit = sum(t.pnl for t in wins) if wins else 0
        total_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        stats.profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # 期望值
        stats.expectancy = (stats.win_rate * stats.avg_win -
                            (1 - stats.win_rate) * stats.avg_loss)
        stats.expectancy_ratio = (stats.expectancy / stats.avg_loss
                                  if stats.avg_loss > 0 else 0)

        # 最大回撤
        if recent:
            cum_pnl = np.cumsum([t.pnl for t in recent])
            peak = np.maximum.accumulate(cum_pnl)
            dd = peak - cum_pnl
            stats.max_drawdown_pct = (float(np.max(dd)) / abs(cum_pnl[0]) * 100
                                      if cum_pnl[0] != 0 else 0)

        # 连亏统计
        cons_loss = 0
        for t in reversed(recent):
            if not t.is_win:
                cons_loss += 1
            else:
                break
        stats.consecutive_losses = cons_loss
        stats.max_consecutive_losses = self._calc_max_consecutive_losses(recent)

        # 极值
        stats.best_trade_pnl = max(t.pnl for t in recent) if recent else 0
        stats.worst_trade_pnl = min(t.pnl for t in recent) if recent else 0

        # 平均持仓时间
        holding_times = [t.exit_time - t.entry_time for t in recent
                         if t.exit_time > 0 and t.entry_time > 0]
        if holding_times:
            stats.avg_holding_hours = np.mean(holding_times) / 3600.0

        # 近似 Sharpe
        if recent:
            returns = [t.pnl_pct for t in recent]
            if len(returns) > 1 and np.std(returns) > 0:
                stats.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365)

        # 按方向分组
        long_trades = [t for t in recent if t.direction == "long"]
        short_trades = [t for t in recent if t.direction == "short"]
        stats.long_stats = self._direction_stats(long_trades)
        stats.short_stats = self._direction_stats(short_trades)

        # 按评分分组
        for bucket_name, (lo, hi) in [("0.0-0.4", (0, 0.4)), ("0.4-0.6", (0.4, 0.6)),
                                       ("0.6-0.8", (0.6, 0.8)), ("0.8-1.0", (0.8, 1.0))]:
            bucket = [t for t in recent if lo <= t.master_score < hi]
            if bucket:
                stats.score_bucket_stats[bucket_name] = self._direction_stats(bucket)

        return stats

    def _direction_stats(self, trades: List[TradeRecord]) -> Dict[str, float]:
        if not trades:
            return {"count": 0, "win_rate": 0, "total_pnl": 0}
        wins = [t for t in trades if t.is_win]
        return {
            "count": len(trades),
            "win_rate": len(wins) / len(trades),
            "total_pnl": sum(t.pnl for t in trades),
            "avg_pnl": np.mean([t.pnl for t in trades]),
        }

    def _calc_max_consecutive_losses(self, trades: List[TradeRecord]) -> int:
        max_cons = 0
        current = 0
        for t in trades:
            if not t.is_win:
                current += 1
                max_cons = max(max_cons, current)
            else:
                current = 0
        return max_cons


# ===========================================================================
# 自适应参数调整（简化版 FreqAI）
# ===========================================================================

class AdaptiveParameterTuner:
    """自适应参数调整器 — 借鉴 FreqAI 的自适应思想。

    根据最近 N 笔交易的表现，动态调整策略参数：
      - 连亏 > 3 笔 → 降低杠杆、提高入场阈值
      - 连赢 > 5 笔 → 可以略微提高杠杆
      - 最大回撤 > 20% → 暂停交易
      - 体制切换时 → 重置参数到默认值
    """

    def __init__(self, config: dict):
        self.risk_cfg = config["risk"]
        self.strategy_cfg = config["strategy"]

        # 基准参数
        self.base_leverage = self.risk_cfg["max_leverage"]
        self.base_risk_pct = self.risk_cfg["risk_per_trade_pct"]
        self.base_min_score = 0.40  # 最低入场评分

        # 当前参数
        self.current_leverage = self.base_leverage
        self.current_risk_pct = self.base_risk_pct
        self.current_min_score = self.base_min_score
        self.trading_paused = False

    def adapt(self, edge_stats: EdgeStats, regime: str = ""):
        """根据近期表现自动调整参数。"""
        # 连亏保护
        if edge_stats.consecutive_losses >= 5:
            self.trading_paused = True
            logger.warning(f"连续 {edge_stats.consecutive_losses} 笔亏损，暂停交易！")
            self.current_leverage = self.base_leverage
            self.current_risk_pct = self.base_risk_pct
            self.current_min_score = self.base_min_score
            return

        if edge_stats.consecutive_losses >= 3:
            factor = max(0.3, 1.0 - edge_stats.consecutive_losses * 0.2)
            self.current_leverage = max(1, int(self.base_leverage * factor))
            self.current_risk_pct = self.base_risk_pct * factor
            self.current_min_score = min(0.80, self.base_min_score + 0.10 * edge_stats.consecutive_losses)
            logger.info(f"连亏 {edge_stats.consecutive_losses} 笔，自动降杠杆至 {self.current_leverage}x, "
                        f"风险比例 {self.current_risk_pct:.1f}%, 最低评分 {self.current_min_score:.2f}")
            return

        # 回撤保护
        if edge_stats.max_drawdown_pct > 20:
            self.trading_paused = True
            logger.warning(f"最大回撤 {edge_stats.max_drawdown_pct:.1f}% 超过 20%，暂停交易！")
            return

        # 盈利恢复
        self.trading_paused = False
        if edge_stats.win_rate > 0.6 and edge_stats.total_trades >= 10:
            # 稳定盈利，可略微提高杠杆
            self.current_leverage = min(self.base_leverage,
                                        int(self.base_leverage * 1.2))
            self.current_risk_pct = self.base_risk_pct
            self.current_min_score = self.base_min_score
        else:
            # 恢复默认
            self.current_leverage = self.base_leverage
            self.current_risk_pct = self.base_risk_pct
            self.current_min_score = self.base_min_score

    def get_effective_params(self) -> Tuple[int, float, float]:
        """获取当前有效参数。"""
        return self.current_leverage, self.current_risk_pct, self.current_min_score


# ===========================================================================
# Trailing Stop（动态止损）
# ===========================================================================

class TrailingStop:
    """动态止损 — 借鉴 freqtrade 的 trailing stop 机制。

    当价格朝有利方向移动时，止损跟随移动锁定利润。
    只在价格穿过止盈线一半时激活。
    """

    def __init__(self,
                 activation_pct: float = 0.50,   # 激活阈值（达到目标 50% 时激活）
                 trail_pct: float = 0.30,         # 追踪距离（动态止损距当前价的距离）
                 ):
        self.activation_pct = activation_pct
        self.trail_pct = trail_pct
        self._activated = False
        self._trailing_stop = 0.0
        self._best_price = 0.0

    def update(self, current_price: float, entry_price: float,
               stop_loss: float, take_profit: float,
               direction: str) -> float:
        """更新动态止损价位。

        Returns:
            当前有效的止损价
        """
        if direction == "long":
            progress = ((current_price - entry_price) /
                        (take_profit - entry_price) if take_profit != entry_price else 0)

            if progress >= self.activation_pct and not self._activated:
                self._activated = True
                self._best_price = current_price
                self._trailing_stop = current_price * (1 - self.trail_pct / 100)
                logger.info(f"Trailing Stop 激活! 初始追踪止损: {self._trailing_stop:.2f}")

            if self._activated:
                if current_price > self._best_price:
                    self._best_price = current_price
                new_stop = self._best_price * (1 - self.trail_pct / 100)
                self._trailing_stop = max(self._trailing_stop, new_stop)
                return max(stop_loss, self._trailing_stop)

        else:  # short
            progress = ((entry_price - current_price) /
                        (entry_price - take_profit) if entry_price != take_profit else 0)

            if progress >= self.activation_pct and not self._activated:
                self._activated = True
                self._best_price = current_price
                self._trailing_stop = current_price * (1 + self.trail_pct / 100)
                logger.info(f"Trailing Stop 激活! 初始追踪止损: {self._trailing_stop:.2f}")

            if self._activated:
                if current_price < self._best_price:
                    self._best_price = current_price
                new_stop = self._best_price * (1 + self.trail_pct / 100)
                self._trailing_stop = min(self._trailing_stop, new_stop)
                return min(stop_loss, self._trailing_stop)

        return stop_loss

    def reset(self):
        self._activated = False
        self._trailing_stop = 0.0
        self._best_price = 0.0


# ===========================================================================
# 参数网格搜索（简化版 Hyperopt）
# ===========================================================================

@dataclass
class ParamGrid:
    """参数搜索空间。"""
    name: str
    values: List[float]


def grid_search_hyperopt(
    records: List[TradeRecord],
    param_grids: List[ParamGrid],
    objective: Callable[[List[TradeRecord], dict], float],
    n_best: int = 3,
) -> List[Tuple[dict, float]]:
    """简化版 Hyperopt 网格搜索。

    对所有参数组合遍历，按目标函数评分排序，返回 Top N。

    Args:
        records: 历史交易记录
        param_grids: 参数网格
        objective: 目标函数 (records, params) -> score
        n_best: 返回最优组合数

    Returns:
        [(params, score), ...] 按评分降序
    """
    from itertools import product

    param_names = [g.name for g in param_grids]
    param_values = [g.values for g in param_grids]

    results = []
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        try:
            score = objective(records, params)
            results.append((params, score))
        except Exception as e:
            logger.debug(f"Hyperopt combo {params} failed: {e}")

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n_best]


def default_objective_sharpe(records: List[TradeRecord], params: dict) -> float:
    """默认目标函数：最大化 Sharpe-like 比率。

    可根据 params 中的参数重新评估每笔交易。
    """
    if not records:
        return 0.0

    returns = [t.pnl_pct for t in records]
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) if len(returns) > 1 else 1.0

    if std_ret == 0:
        return mean_ret * 100 if mean_ret > 0 else 0.0

    sharpe = mean_ret / std_ret * np.sqrt(len(records))

    # 惩罚连亏
    cons_loss = 0
    for t in reversed(records):
        if not t.is_win:
            cons_loss += 1
        else:
            break
    penalty = max(0, cons_loss - 3) * 0.1

    return sharpe - penalty


# ===========================================================================
# 投资组合级风控
# ===========================================================================

@dataclass
class PortfolioRisk:
    """投资组合级风控状态。"""
    current_exposure: float = 0.0     # 当前敞口 (USDT)
    max_exposure: float = 0.0         # 最大敞口
    daily_pnl: float = 0.0            # 当日盈亏
    weekly_pnl: float = 0.0           # 本周盈亏
    utilization_pct: float = 0.0      # 保证金使用率
    var_95: float = 0.0               # 95% VaR (近似)
    risk_score: float = 0.0           # 综合风险评分 (0-100, 越高越危险)


def assess_portfolio_risk(
    equity: float,
    margin_used: float,
    positions: List[dict],
    trades: List[TradeRecord],
    max_exposure_pct: float = 30.0,
) -> PortfolioRisk:
    """评估投资组合级风险。"""
    risk = PortfolioRisk()

    # 当前敞口
    risk.current_exposure = sum(
        abs(float(p.get("size", 0)) * float(p.get("mark_px", 0)))
        for p in positions
    )
    risk.max_exposure = equity * max_exposure_pct / 100.0
    risk.utilization_pct = (margin_used / equity * 100) if equity > 0 else 0

    # 近似 VaR (95%)
    if len(trades) >= 10:
        returns = [t.pnl_pct for t in trades[-50:]]
        risk.var_95 = abs(np.percentile(returns, 5)) if returns else 0

    # 综合风险评分
    risk_score = 0.0
    if risk.max_exposure > 0:
        risk_score += (risk.current_exposure / risk.max_exposure) * 40
    risk_score += min(risk.utilization_pct, 100) * 0.3
    risk_score += risk.var_95 * 100
    risk.risk_score = min(100, risk_score)

    return risk