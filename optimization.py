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
import threading
import time
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
    break_even_trades: int = 0          # 修复 Bug 48: 保本交易单独统计
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

    def __init__(self, max_trades: int = 500):
        self.trades: List[TradeRecord] = []
        self.max_trades = max_trades

    def add_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        # 修复: 内存无界膨胀防护 — 超过上限时裁剪最旧的交易
        # 不限制会导致 Hyperopt 贝叶斯优化指数级变慢，最终 OOM
        if len(self.trades) > self.max_trades:
            _trimmed = len(self.trades) - self.max_trades
            self.trades = self.trades[_trimmed:]
            logger.debug(f"[EdgeAnalyzer] 裁剪 {_trimmed} 笔旧交易，"
                         f"保留最近 {self.max_trades} 笔")

    def analyze(self, lookback: int = 100,
                equity_baseline: Optional[float] = None) -> EdgeStats:
        """分析最近 N 笔交易。

        Args:
            lookback: 统计最近多少笔
            equity_baseline: 权益基线(账户余额/权益)。回撤百分比必须相对权益计算，
                否则除以极小累计PnL峰值会被放大上千倍(实测 -1.16/-0.26→446%)。
        """
        recent = self.trades[-lookback:] if len(self.trades) > lookback else self.trades
        if not recent:
            return EdgeStats()

        stats = EdgeStats()
        stats.total_trades = len(recent)

        # 修复 Bug 48: 胜率统计以 pnl 符号为准，而非调用方传入的 is_win。
        # is_win 可能因调用错误导致 pnl 与 is_win 符号不一致。
        wins = [t for t in recent if t.pnl > 0]
        losses = [t for t in recent if t.pnl < 0]
        break_evens = [t for t in recent if abs(t.pnl) < 1e-9]
        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.break_even_trades = len(break_evens)
        decisive = stats.winning_trades + stats.losing_trades
        stats.win_rate = stats.winning_trades / decisive if decisive > 0 else 0.0

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

        # 最大回撤 — 标准公式: (peak - trough) / peak_equity * 100
        # L-16: 当前回撤基于累计 PnL 而非权益曲线，未考虑持仓浮盈浮亏的影响。
        # 截断情况下（如部分交易被剔除），回撤可能被低估。
        # 修复: 百分比分母必须是权益基线，而非累计PnL峰值。
        # 旧公式 max_dd/peak_cum_pnl 在累计PnL刚起步(如0.26 USDT)时会把
        # 绝对回撤(1.16)放大到 446%，触发 20% 阈值误暂停交易。
        if recent:
            cum_pnl = np.cumsum([t.pnl for t in recent])
            peak = np.maximum.accumulate(cum_pnl)
            dd = peak - cum_pnl
            max_dd = float(np.max(dd))
            if equity_baseline and equity_baseline > 0:
                stats.max_drawdown_pct = max_dd / float(equity_baseline) * 100.0
            else:
                # 无权益基线时的退化参考值: 相对累计PnL绝对规模(仅作展示，不触发风控)
                _scale = max(1.0, abs(float(cum_pnl[-1])) + max_dd)
                stats.max_drawdown_pct = max_dd / _scale * 100.0

        # 连亏统计（修复 Bug 48: 统一以 pnl 符号为准）
        cons_loss = 0
        for t in reversed(recent):
            if t.pnl < 0:
                cons_loss += 1
            elif t.pnl > 0:
                break
            # pnl == 0 保本：不计入连亏也不中断连亏
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

        # 近似 Sharpe（使用日均收益率进行年化）
        if recent:
            daily_returns = []
            for t in recent:
                # M-14: 跳过未平仓交易（exit_time <= 0），避免被当作持有 1 天
                if t.exit_time <= 0 or t.entry_time <= 0:
                    continue
                # 修复: pnl_pct 是百分比 (如 5.0 = 5%)，必须除以 100 转为小数
                # 原先直接用百分比除以持有天数，导致 daily_ret ≈ 2.5，
                # Sharpe = 2.5 * 365 / ... ≈ 912，严重失真
                holding_days = max(1.0, (t.exit_time - t.entry_time) / 86400.0)
                daily_ret = (t.pnl_pct / 100.0) / holding_days
                daily_returns.append(daily_ret)
            if len(daily_returns) > 1 and np.std(daily_returns, ddof=1) > 1e-10:
                daily_mean = np.mean(daily_returns)
                daily_std = np.std(daily_returns, ddof=1)
                # M-9: 扣除无风险利率，与 hyperopt.py 保持一致
                stats.sharpe_ratio = (daily_mean * 365 - 0.02) / (daily_std * np.sqrt(365))

        # 按方向分组
        long_trades = [t for t in recent if t.direction == "long"]
        short_trades = [t for t in recent if t.direction == "short"]
        stats.long_stats = self._direction_stats(long_trades)
        stats.short_stats = self._direction_stats(short_trades)

        # 按评分分组
        for bucket_name, (lo, hi) in [("0.0-0.4", (0, 0.4)), ("0.4-0.6", (0.4, 0.6)),
                                       ("0.6-0.8", (0.6, 0.8)), ("0.8-1.0", (0.8, 1.01))]:
            bucket = [t for t in recent if lo <= t.master_score < hi]
            if bucket:
                stats.score_bucket_stats[bucket_name] = self._direction_stats(bucket)

        return stats

    def _direction_stats(self, trades: List[TradeRecord]) -> Dict[str, float]:
        # 修复 Bug 48: 胜率分母剔除保本（pnl == 0）
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        decisive = len(wins) + len(losses)
        return {
            "count": len(trades),
            "win_rate": len(wins) / decisive if decisive > 0 else 0.0,
            "total_pnl": sum(t.pnl for t in trades),
            "avg_pnl": float(np.mean([t.pnl for t in trades])) if trades else 0.0,
        }

    def _calc_max_consecutive_losses(self, trades: List[TradeRecord]) -> int:
        # 修复 Bug 48: 只统计连续亏损（pnl < 0），保本（pnl == 0）不中断也不计入
        max_cons = 0
        current = 0
        for t in trades:
            if t.pnl < 0:
                current += 1
                max_cons = max(max_cons, current)
            elif t.pnl > 0:
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
      - 绝对回撤 > 15%（从峰值）→ 暂停交易，次日重置
      - 体制切换时 → 重置参数到默认值
    """

    def __init__(self, config: dict):
        self.risk_cfg = config["risk"]
        self.strategy_cfg = config["strategy"]
        opt_cfg = config.get("optimization", {}) or {}

        # 基准参数
        self.base_leverage = self.risk_cfg["max_leverage"]
        self.base_risk_pct = self.risk_cfg["risk_per_trade_pct"]
        self.base_min_score = 0.40  # 最低入场评分

        # 当前参数
        self.current_leverage = self.base_leverage
        self.current_risk_pct = self.base_risk_pct
        self.current_min_score = self.base_min_score
        self.trading_paused = False

        # 修复 2026-08-07: 连亏暂停参数化 (研究建议"连亏 2 笔暂停 24h"可配置)
        # 原硬编码 5/3 → 读 risk.consecutive_loss_pause / risk.consecutive_loss_derate;
        # 暂停时长 loss_pause_hours 到期后自动解除(原为无限期直到连亏恢复)。
        self.loss_pause_threshold: int = max(
            1, int(opt_cfg.get("consecutive_loss_pause", 5) or 5))
        self.loss_derate_threshold: int = max(
            1, int(opt_cfg.get("consecutive_loss_derate", 3) or 3))
        self.loss_pause_hours: float = float(
            opt_cfg.get("loss_pause_hours", 24) or 24)
        self.pause_until: float = 0.0  # 连亏暂停解除时间戳

        # 修复 P1-4: 绝对回撤断路器 — 从峰值回撤超过阈值时暂停
        self.peak_equity: float = 0.0              # 历史峰值权益
        self.absolute_drawdown_threshold_pct: float = 15.0  # 绝对回撤阈值 15%

    def update_equity(self, equity: float):
        """更新峰值权益追踪。"""
        if equity > self.peak_equity:
            self.peak_equity = equity

    def adapt(self, edge_stats: EdgeStats, regime: str = "", current_equity: float = 0.0):
        """根据近期表现自动调整参数。

        Args:
            edge_stats: 交易统计
            regime: 当前市场体制
            current_equity: 当前权益，用于绝对回撤断路器
        """
        # 修复 P1-4: 绝对回撤断路器 — 从峰值回撤 > 阈值时暂停交易
        # 比每日亏损限额更全面：防止连续阴跌（每天亏 3% 但不满日限额，累计亏 15%）
        if current_equity > 0:
            if self.peak_equity == 0:
                self.peak_equity = current_equity
            self.update_equity(current_equity)
            drawdown_from_peak = (self.peak_equity - current_equity) / self.peak_equity * 100
            if drawdown_from_peak > self.absolute_drawdown_threshold_pct:
                self.trading_paused = True
                # 修复 P2-2: 绝对回撤断路器此前只置 trading_paused 未设 pause_until，
                # 日志声称"暂停至次日"但恢复路径(adapt 中 pause_until>0 分支)永远
                # 不触发，只能靠滚动窗口回撤回落才恢复（语义不一致）。
                # 统一设 24h 冷却，到期后重新评估。
                self.pause_until = time.time() + 24 * 3600
                logger.warning(
                    f"绝对回撤断路器触发！从峰值 {self.peak_equity:.2f} 回撤 "
                    f"{drawdown_from_peak:.1f}% > {self.absolute_drawdown_threshold_pct}%，"
                    f"暂停交易 24h（{time.strftime('%Y-%m-%d %H:%M', time.localtime(self.pause_until))} 恢复评估）"
                )
                return

        # 修复 2026-08-07: 连亏暂停参数化 — 暂停时长到期自动解除
        # (原实现无限期暂停, 需连亏恢复才会解除; 现在 24h 后重新评估)
        if self.trading_paused and self.pause_until > 0:
            if time.time() >= self.pause_until:
                self.trading_paused = False
                self.pause_until = 0.0
                logger.warning("连亏暂停冷却期已过，恢复交易评估")
            else:
                _remain_h = (self.pause_until - time.time()) / 3600.0
                logger.warning(
                    f"连亏暂停中，剩余 {_remain_h:.1f}h 后恢复评估")
                return

        # 连亏保护
        if edge_stats.consecutive_losses >= self.loss_pause_threshold:
            self.trading_paused = True
            self.pause_until = time.time() + self.loss_pause_hours * 3600.0
            logger.warning(
                f"连续 {edge_stats.consecutive_losses} 笔亏损，暂停交易 "
                f"{self.loss_pause_hours:.0f}h！")
            self.current_leverage = self.base_leverage
            self.current_risk_pct = self.base_risk_pct
            self.current_min_score = self.base_min_score
            return

        if edge_stats.consecutive_losses >= self.loss_derate_threshold:
            self.trading_paused = False
            factor = max(0.3, 1.0 - edge_stats.consecutive_losses * 0.2)
            self.current_leverage = max(1, int(self.base_leverage * factor))
            self.current_risk_pct = self.base_risk_pct * factor
            self.current_min_score = min(0.70, self.base_min_score + 0.05 * edge_stats.consecutive_losses)
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
            upper_bound = self.risk_cfg["max_leverage"]
            self.current_leverage = min(upper_bound,
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

    修复 P1-3: 激活阈值和追踪距离基于 ATR 而非固定百分比。
    小 FVG 宽度（1.5%）导致止盈目标仅 0.75%，50% 激活 = 0.375% 即触发，
    然后 3% 追踪距离远超止盈范围，trailing stop 形同虚设。
    
    ATR 化后:
      - 激活: 价格朝有利方向移动 >= atr * atr_activation_multiplier 时激活
      - 追踪距离: atr * trail_atr_multiplier
    """

    def __init__(self,
                 activation_pct: float = 0.70,     # 回退兼容: TP 目标百分比激活（已弃用为主逻辑）
                 trail_pct: float = 0.03,           # 回退兼容: 固定百分比追踪距离（已弃用为主逻辑）
                 atr_activation_multiplier: float = 0.5,  # ATR 激活倍数: 0.5x ATR ≈ 1% 价格移动 (2% ATR 时)
                 trail_atr_multiplier: float = 1.5,      # ATR 追踪倍数: 1.5x ATR ≈ 3% 追踪距离 (2026-08-10 用户要求: 0.75→1.5, 防刚激活就被扫)
                 ):
        self.activation_pct = activation_pct
        self.trail_pct = trail_pct
        self.atr_activation_multiplier = atr_activation_multiplier
        self.trail_atr_multiplier = trail_atr_multiplier
        self._lock = threading.Lock()
        self._activated = False
        self._trailing_stop = 0.0
        self._best_price = 0.0
        self._atr = 0.0  # 缓存最近一次 atr 用于日志
        self._cached_atr = None  # C-2: 缓存最近有效 ATR，回退时使用

    def update(self, current_price: float, entry_price: float,
               stop_loss: float, take_profit: float,
               direction: str, atr: float = 0.0) -> float:
        """更新动态止损价位。

        Args:
            current_price: 当前价格
            entry_price: 入场价
            stop_loss: 原始止损价
            take_profit: 原始止盈价
            direction: "long" | "short"
            atr: 当前 ATR 值，用于 ATR 化激活阈值和追踪距离

        Returns:
            当前有效的止损价
        """
        with self._lock:
            if atr > 0:
                self._atr = atr
                self._cached_atr = atr  # C-2: 缓存有效 ATR 值
                # ATR 化激活阈值: price_move >= atr * atr_activation_multiplier
                activation_distance = atr * self.atr_activation_multiplier
                # ATR 化追踪距离: trail_distance = atr * trail_atr_multiplier
                trail_distance = atr * self.trail_atr_multiplier
            elif self._cached_atr is not None:
                # C-2: 当前 ATR 不可用时，使用缓存的有效 ATR 值
                logger.debug(f"ATR 不可用，回退到缓存值 {self._cached_atr:.4f}")
                activation_distance = self._cached_atr * self.atr_activation_multiplier
                trail_distance = self._cached_atr * self.trail_atr_multiplier
            else:
                # 回退: 无 ATR 且无缓存时使用百分比激活
                activation_distance = 0
                trail_distance = 0

            if direction == "long":
                price_move = current_price - entry_price

                # 激活判断: ATR 优先，回退到百分比
                if not self._activated:
                    if activation_distance > 0:
                        activated = price_move >= activation_distance
                    else:
                        tp_dist = take_profit - entry_price
                        if tp_dist <= current_price * 0.001:
                            return stop_loss
                        activated = (price_move / tp_dist) >= self.activation_pct

                    if activated:
                        self._activated = True
                        self._best_price = current_price
                        if trail_distance > 0:
                            self._trailing_stop = current_price - trail_distance
                        else:
                            self._trailing_stop = current_price * (1 - self.trail_pct)
                        logger.info(f"Trailing Stop 激活! ATR={self._atr:.4f} 初始止损: {self._trailing_stop:.2f}")

                if self._activated:
                    if current_price > self._best_price:
                        self._best_price = current_price
                    if trail_distance > 0:
                        new_stop = self._best_price - trail_distance
                    else:
                        new_stop = self._best_price * (1 - self.trail_pct)
                    self._trailing_stop = max(self._trailing_stop, new_stop)
                    return max(stop_loss, self._trailing_stop)

            else:  # short
                price_move = entry_price - current_price

                if not self._activated:
                    if activation_distance > 0:
                        activated = price_move >= activation_distance
                    else:
                        tp_dist = entry_price - take_profit
                        if tp_dist <= current_price * 0.001:
                            return stop_loss
                        activated = (price_move / tp_dist) >= self.activation_pct

                    if activated:
                        self._activated = True
                        self._best_price = current_price
                        if trail_distance > 0:
                            self._trailing_stop = current_price + trail_distance
                        else:
                            self._trailing_stop = current_price * (1 + self.trail_pct)
                        logger.info(f"Trailing Stop 激活! ATR={self._atr:.4f} 初始止损: {self._trailing_stop:.2f}")

                if self._activated:
                    if current_price < self._best_price:
                        self._best_price = current_price
                    if trail_distance > 0:
                        new_stop = self._best_price + trail_distance
                    else:
                        new_stop = self._best_price * (1 + self.trail_pct)
                    # 修复 C5: 做空方向使用 min() 而非 max()，止损价应向下移动
                    self._trailing_stop = min(self._trailing_stop, new_stop)
                    return min(stop_loss, self._trailing_stop)

            return stop_loss

    def reset(self):
        with self._lock:
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
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1.0

    if std_ret == 0:
        return mean_ret * 100 if mean_ret > 0 else 0.0

    sharpe = mean_ret / std_ret * np.sqrt(365)

    # 惩罚连亏（修复 Bug 48: 以 pnl 符号为准）
    cons_loss = 0
    for t in reversed(records):
        if t.pnl < 0:
            cons_loss += 1
        elif t.pnl > 0:
            break
        # pnl == 0 保本：不增加连亏也不中断
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
    """评估投资组合级风险。

    M-7: 当前实现为简化版本，基于各持仓独立评估后汇总，未计算跨品种
    相关系数矩阵。未来版本将集成多品种协方差矩阵，实现马科维茨风格
    的组合风险度量。

    NOTE: 当前实现不计算标的间相关系数矩阵，风险基于单标的独立评估。
    多标的组合风险 = sqrt(sum(weight_i^2 * risk_i^2)) 假设标的独立。
    """
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