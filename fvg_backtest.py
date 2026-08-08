"""
fvg_backtest.py — 独立 FVG 策略回测引擎。

在历史 K 线上回测 FVG 回补策略：
  1. FVG 形成后 N 根 K 线内价格回补到缺口 → 入场（bullish 买 gap_low，bearish 卖 gap_high）
  2. 止盈：回补完成（bullish 到 gap_high，bearish 到 gap_low）
  3. 止损：FVG 宽度的 1.5 倍
  4. 入场后 M 根内未触发 TP/SL → 时间退出（按收盘价）

输出：胜率、盈亏比、利润因子、最大回撤、夏普、资金曲线。
支持 compare_with_thresholds 对比不同 quality_score 阈值的表现，
用于确定最佳过滤参数（与 fvg_detector.FVGDetected 联动）。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """单笔回测交易。"""
    inst_id: str
    timeframe: str
    direction: str              # "bullish" | "bearish"
    entry_ts: int
    exit_ts: int
    entry: float
    exit: float
    return_pct: float           # 收益比例（含符号）
    exit_reason: str            # "tp" | "sl" | "time"
    quality_score: float = 0.0
    width_pct: float = 0.0


class FVGBacktest:
    """FVG 回测引擎。

    config 键:
        max_entry_bars: int     回补入场等待根数（默认 50）
        max_hold_bars: int      入场后持仓根数上限（默认 50）
        stop_width_mult: float  止损 = 缺口宽度 × mult（默认 1.5）
        position_pct: float     单笔仓位比例（默认 0.1）
        initial_capital: float  初始资金（默认 10000）
    """

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    # ------------------------------------------------------------------
    # 主回测
    # ------------------------------------------------------------------

    def run(
        self, fvgs: List, candles: List, config: Optional[dict] = None
    ) -> Dict[str, Any]:
        """在历史数据上回测 FVG 策略。

        Args:
            fvgs: FVGDetector 产出的 FVGDetected 列表（按时间正序）
            candles: 对应时间框架的完整 K 线（正序，须包含形成后数据）
            config: 回测参数

        Returns:
            {total_return, win_rate, profit_factor, max_drawdown,
             sharpe_ratio, trades, equity_curve}
        """
        cfg = config or {}
        max_entry_bars = int(cfg.get("max_entry_bars", 50))
        max_hold_bars = int(cfg.get("max_hold_bars", 50))
        stop_mult = float(cfg.get("stop_width_mult", 1.5))
        position_pct = float(cfg.get("position_pct", 0.1))

        self.trades = []
        self.capital = self.initial_capital
        self.equity_curve = [self.initial_capital]

        if not candles or not fvgs:
            return self._summary()

        for fvg in fvgs:
            trade = self._simulate(
                fvg, candles, max_entry_bars, max_hold_bars, stop_mult
            )
            if trade is None:
                continue
            # 资金按固定仓位比例累积
            pnl = self.capital * position_pct * trade.return_pct / 100.0
            self.capital += pnl
            self.trades.append(trade)
            self.equity_curve.append(self.capital)

        return self._summary()

    # ------------------------------------------------------------------
    # 内部模拟
    # ------------------------------------------------------------------

    def _simulate(
        self, fvg, candles: List,
        max_entry_bars: int, max_hold_bars: int, stop_mult: float,
    ) -> Optional[Trade]:
        """单 FVG 信号模拟。返回 Trade 或 None（未触发回补）。"""
        end_idx = fvg.end_idx
        if end_idx is None or end_idx >= len(candles) - 1:
            return None

        gap_high, gap_low = fvg.gap_high, fvg.gap_low
        gap_width = gap_high - gap_low
        if gap_width <= 0:
            return None

        # ---- 阶段 1: 等待回补入场（end_idx 之后 max_entry_bars 根内）----
        # 回补定义: 价格进入缺口区域即触发（bullish: low <= gap_high），
        # 入场价取缺口边界（bullish 在 gap_low 买入，bearish 在 gap_high 卖出）。
        entry_bar = None
        entry_price = None
        for j in range(end_idx + 1, min(len(candles), end_idx + 1 + max_entry_bars)):
            c = candles[j]
            if fvg.direction == "bullish" and c.low <= gap_high:
                entry_bar, entry_price = j, gap_low
                break
            if fvg.direction == "bearish" and c.high >= gap_low:
                entry_bar, entry_price = j, gap_high
                break
        if entry_bar is None:
            return None  # N 根内未回补 → 放弃

        # ---- 阶段 2: 入场后判断 TP / SL ----
        stop_price = (entry_price - stop_mult * gap_width
                      if fvg.direction == "bullish"
                      else entry_price + stop_mult * gap_width)

        exit_price = None
        exit_ts = candles[entry_bar].timestamp
        exit_reason = "time"
        last_bar = min(len(candles), entry_bar + 1 + max_hold_bars)
        for j in range(entry_bar, last_bar):
            c = candles[j]
            if fvg.direction == "bullish":
                # 保守顺序: 同根同时触发按 SL 计
                if c.low <= stop_price:
                    exit_price, exit_ts, exit_reason = stop_price, c.timestamp, "sl"
                    break
                if c.high >= gap_high:
                    exit_price, exit_ts, exit_reason = gap_high, c.timestamp, "tp"
                    break
            else:
                if c.high >= stop_price:
                    exit_price, exit_ts, exit_reason = stop_price, c.timestamp, "sl"
                    break
                if c.low <= gap_low:
                    exit_price, exit_ts, exit_reason = gap_low, c.timestamp, "tp"
                    break
        if exit_price is None:
            # 时间退出（按最后一根收盘价）
            exit_price = candles[last_bar - 1].close
            exit_ts = candles[last_bar - 1].timestamp
            exit_reason = "time"

        if fvg.direction == "bullish":
            return_pct = (exit_price - entry_price) / entry_price * 100
        else:
            return_pct = (entry_price - exit_price) / entry_price * 100

        return Trade(
            inst_id=fvg.inst_id or "",
            timeframe=fvg.timeframe,
            direction=fvg.direction,
            entry_ts=candles[entry_bar].timestamp,
            exit_ts=exit_ts,
            entry=entry_price,
            exit=exit_price,
            return_pct=return_pct,
            exit_reason=exit_reason,
            quality_score=float(getattr(fvg, "quality_score", 0.0)),
            width_pct=float(getattr(fvg, "width_pct", 0.0)),
        )

    # ------------------------------------------------------------------
    # 指标汇总
    # ------------------------------------------------------------------

    def _summary(self) -> Dict[str, Any]:
        n = len(self.trades)
        if n == 0:
            return {
                "total_return": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                "max_drawdown": 0.0, "sharpe_ratio": 0.0,
                "trades": [], "equity_curve": self.equity_curve,
                "n_trades": 0,
            }
        rets = np.array([t.return_pct for t in self.trades])
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        win_rate = float(len(wins) / n)
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_drawdown = float(max(0.0, -dd.min())) if len(dd) else 0.0

        if rets.std() > 1e-12:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(max(n, 1)))
        else:
            sharpe = 0.0

        return {
            "total_return": float(self.capital / self.initial_capital - 1),
            "win_rate": win_rate,
            "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "n_trades": n,
        }

    # ------------------------------------------------------------------
    # 阈值对比
    # ------------------------------------------------------------------

    def compare_with_thresholds(
        self, fvgs: List, candles: List,
        thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """对比不同 quality_score 阈值下的回测表现。

        Args:
            fvgs: FVGDetected 列表
            candles: K 线
            thresholds: 如 [0.0, 0.3, 0.4, 0.5, 0.6]，默认自动分桶

        Returns:
            DataFrame 行=阈值，列=指标
        """
        if thresholds is None:
            qs = [float(getattr(f, "quality_score", 0.0)) for f in fvgs]
            if not qs:
                return pd.DataFrame(columns=[
                    "quality_threshold", "n_trades", "win_rate",
                    "profit_factor", "total_return", "max_drawdown", "sharpe_ratio"])
            lo, hi = min(qs), max(qs)
            thresholds = [0.0]
            for q in np.linspace(lo + (hi - lo) / 5, hi, 4):
                thresholds.append(round(float(q), 3))
            thresholds = sorted(set(thresholds))

        rows = []
        for t in thresholds:
            filtered = [f for f in fvgs if float(getattr(f, "quality_score", 0.0)) >= t]
            bt = FVGBacktest(self.initial_capital)
            res = bt.run(filtered, candles)
            rows.append({
                "quality_threshold": round(float(t), 3),
                "n_trades": res["n_trades"],
                "win_rate": round(res["win_rate"], 3),
                "profit_factor": round(res["profit_factor"], 2),
                "total_return": round(res["total_return"], 4),
                "max_drawdown": round(res["max_drawdown"], 4),
                "sharpe_ratio": round(res["sharpe_ratio"], 3),
            })
        return pd.DataFrame(rows)
