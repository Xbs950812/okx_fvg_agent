"""
Daily Quant Report：每日量化报告生成。

内容：
  - 当日/累计交易次数、胜率、Profit Factor、Sharpe、最大回撤
  - 平均持仓时间
  - 最佳/最差策略表现
  - 模型表现：confidence 准确率、AI 风险评分准确率
  - 因子贡献
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from persistence import QuantDB

logger = logging.getLogger(__name__)


@dataclass
class DailyQuantReport:
    """每日量化报告。"""
    date: str
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_cost: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_holding_seconds: float = 0.0
    best_symbol: str = ""
    best_pnl: float = 0.0
    worst_symbol: str = ""
    worst_pnl: float = 0.0
    confidence_calibration: Dict[str, Any] = field(default_factory=dict)
    factor_contribution: List[Dict[str, Any]] = field(default_factory=list)
    ai_risk_accuracy: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "=" * 60,
            f"Daily Quant Report — {self.date}",
            "=" * 60,
            f"总交易次数: {self.total_trades}",
            f"胜/负: {self.win_count} / {self.loss_count}",
            f"胜率: {self.win_rate:.1%}",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Sharpe: {self.sharpe:.2f}",
            f"最大回撤: {self.max_drawdown_pct:.2f}%",
            f"毛盈亏: {self.gross_pnl:.2f} USDT",
            f"总成本: {self.total_cost:.2f} USDT",
            f"净盈亏: {self.net_pnl:.2f} USDT",
            f"平均持仓时间: {self.avg_holding_seconds / 3600:.1f} h",
            f"最佳: {self.best_symbol} ({self.best_pnl:.2f})",
            f"最差: {self.worst_symbol} ({self.worst_pnl:.2f})",
            "-" * 60,
            "Confidence Calibration:",
        ]
        for bin_name, stats in self.confidence_calibration.items():
            if isinstance(stats, dict):
                lines.append(
                    f"  {bin_name}: n={stats.get('n', 0)}, win_rate={stats.get('win_rate', 0):.1%}, "
                    f"avg_pnl={stats.get('avg_pnl', 0):.2f}"
                )
        lines.extend([
            "-" * 60,
            "AI Risk Accuracy:",
            f"  {self.ai_risk_accuracy:.1%}",
            "=" * 60,
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "total_trades": self.total_trades,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "total_cost": self.total_cost,
            "sharpe": self.sharpe,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_holding_seconds": self.avg_holding_seconds,
            "best_symbol": self.best_symbol,
            "best_pnl": self.best_pnl,
            "worst_symbol": self.worst_symbol,
            "worst_pnl": self.worst_pnl,
            "confidence_calibration": self.confidence_calibration,
            "factor_contribution": self.factor_contribution,
            "ai_risk_accuracy": self.ai_risk_accuracy,
            "notes": self.notes,
        }


class QuantReportGenerator:
    """每日量化报告生成器。"""

    def __init__(
        self,
        db: Optional[QuantDB] = None,
        db_path: str = "quant_agent.db",
        report_dir: str = "reports",
    ):
        self.db = db or QuantDB(db_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 生成报告
    # ------------------------------------------------------------------
    def generate(
        self,
        date: Optional[str] = None,
        equity_curve: Optional[pd.Series] = None,
    ) -> DailyQuantReport:
        """生成指定日期的报告。"""
        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_ts, end_ts = _day_bounds(date)

        trades = self.db.query(
            "SELECT * FROM trades WHERE exit_time >= ? AND exit_time < ? ORDER BY pnl DESC",
            (start_ts, end_ts),
        )

        report = DailyQuantReport(date=date)
        if not trades:
            report.notes.append("当日无平仓交易")
            return report

        report.total_trades = len(trades)
        wins = [t for t in trades if t.get("is_win")]
        losses = [t for t in trades if not t.get("is_win")]
        report.win_count = len(wins)
        report.loss_count = len(losses)
        report.win_rate = report.win_count / report.total_trades if report.total_trades else 0.0

        gross_wins = sum(t.get("pnl", 0) for t in wins)
        gross_losses = abs(sum(t.get("pnl", 0) for t in losses))
        report.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        report.total_cost = sum(t.get("cost_total", 0) for t in trades)
        report.net_pnl = sum(t.get("pnl", 0) for t in trades)
        report.gross_pnl = report.net_pnl + report.total_cost

        holdings = [t.get("holding_time", 0) for t in trades if t.get("holding_time")]
        report.avg_holding_seconds = float(np.mean(holdings)) if holdings else 0.0

        sorted_by_pnl = sorted(trades, key=lambda x: x.get("pnl", 0))
        if sorted_by_pnl:
            worst = sorted_by_pnl[0]
            best = sorted_by_pnl[-1]
            report.best_symbol = best.get("symbol", "")
            report.best_pnl = best.get("pnl", 0)
            report.worst_symbol = worst.get("symbol", "")
            report.worst_pnl = worst.get("pnl", 0)

        if equity_curve is not None and not equity_curve.empty:
            report.sharpe = _compute_sharpe(equity_curve)
            report.max_drawdown_pct = _compute_max_drawdown(equity_curve)

        # 置信度校准（使用最近 90 天数据）
        report.confidence_calibration = self._confidence_summary(days=90)

        # 因子贡献（最近 90 天）
        report.factor_contribution = self._factor_contribution(days=90)

        return report

    def save(self, report: DailyQuantReport) -> str:
        """保存报告到 reports/YYYY-MM-DD_quant_report.txt。"""
        path = self.report_dir / f"{report.date}_quant_report.txt"
        path.write_text(report.to_text(), encoding="utf-8")
        logger.info(f"[QuantReport] saved to {path}")
        return str(path)

    # ------------------------------------------------------------------
    # 辅助统计
    # ------------------------------------------------------------------
    def _confidence_summary(self, days: int = 90) -> Dict[str, Any]:
        end_ts = datetime.now(timezone.utc).timestamp()
        start_ts = end_ts - days * 86400
        rows = self.db.query(
            """
            SELECT
                CASE
                    WHEN s.confidence >= 0.8 THEN '0.8+'
                    WHEN s.confidence >= 0.7 THEN '0.7-0.8'
                    WHEN s.confidence >= 0.6 THEN '0.6-0.7'
                    WHEN s.confidence >= 0.5 THEN '0.5-0.6'
                    WHEN s.confidence >= 0.4 THEN '0.4-0.5'
                    ELSE '<0.4'
                END AS conf_bin,
                COUNT(*) AS n,
                AVG(t.is_win) AS win_rate,
                AVG(t.pnl) AS avg_pnl
            FROM signals s
            JOIN trades t ON s.signal_id = t.signal_id
            WHERE t.exit_time >= ? AND t.exit_time < ?
            GROUP BY conf_bin
            """,
            (start_ts, end_ts),
        )
        return {r["conf_bin"]: dict(r) for r in rows}

    def _factor_contribution(self, days: int = 90) -> List[Dict[str, Any]]:
        end_ts = datetime.now(timezone.utc).timestamp()
        start_ts = end_ts - days * 86400
        rows = self.db.query(
            """
            SELECT factor_name, AVG(ABS(score) * weight) AS contribution
            FROM factor_scores
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY factor_name
            ORDER BY contribution DESC
            LIMIT 20
            """,
            (start_ts, end_ts),
        )
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _day_bounds(date_str: str) -> tuple:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = dt.timestamp()
    end = start + 86400
    return start, end


def _compute_sharpe(equity: pd.Series, periods_per_year: float = 365) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    rets = equity.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(periods_per_year))


def _compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min() * 100)
