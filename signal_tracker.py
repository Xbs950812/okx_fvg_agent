"""
SignalPerformanceTracker：信号预测统计与置信度校准系统。

每次产生信号时记录预测字段；平仓后记录结果并计算 MFE/MAE。
定期生成 confidence calibration report，若 confidence 与实际胜率脱节则给出权重调整建议。
"""

import json
import logging
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from persistence import QuantDB

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """信号预测记录。"""
    symbol: str
    direction: str
    entry_price: float
    confidence: float = 0.0
    master_score: float = 0.0
    factor_score: float = 0.0
    regime: str = "NEUTRAL"
    volatility: float = 0.0
    funding_rate: float = 0.0
    market_condition: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0
    leverage: int = 1
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=lambda: time.time())
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "master_score": self.master_score,
            "factor_score": self.factor_score,
            "regime": self.regime,
            "volatility": self.volatility,
            "funding_rate": self.funding_rate,
            "market_condition": self.market_condition,
            "extra": self.extra,
        }


@dataclass
class TradeOutcome:
    """交易结果记录。"""
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    pnl: float
    pnl_pct: float
    mfe: float = 0.0
    mae: float = 0.0
    cost_total: float = 0.0
    regime: str = "NEUTRAL"
    confidence: float = 0.0
    master_score: float = 0.0
    factor_score: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def holding_time(self) -> float:
        return max(0.0, self.exit_time - self.entry_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "holding_time": self.holding_time,
            "mfe": self.mfe,
            "mae": self.mae,
            "is_win": self.is_win,
            "regime": self.regime,
            "confidence": self.confidence,
            "master_score": self.master_score,
            "factor_score": self.factor_score,
            "cost_total": self.cost_total,
            "extra": self.extra,
        }


class SignalPerformanceTracker:
    """信号性能追踪器。"""

    def __init__(self, db: Optional[QuantDB] = None, db_path: str = "quant_agent.db"):
        self.db = db or QuantDB(db_path)
        self._open_signals: Dict[str, SignalRecord] = {}

    # ------------------------------------------------------------------
    # 记录信号
    # ------------------------------------------------------------------
    def record_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        confidence: float = 0.0,
        master_score: float = 0.0,
        factor_score: float = 0.0,
        regime: str = "NEUTRAL",
        volatility: float = 0.0,
        funding_rate: float = 0.0,
        market_condition: str = "",
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        leverage: int = 1,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        rec = SignalRecord(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            confidence=confidence,
            master_score=master_score,
            factor_score=factor_score,
            regime=regime,
            volatility=volatility,
            funding_rate=funding_rate,
            market_condition=market_condition,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            extra=extra or {},
        )
        self._open_signals[rec.signal_id] = rec
        self.db.save_signal(rec.to_dict())
        logger.debug(
            f"[SignalTracker] recorded {rec.signal_id} {symbol} {direction} "
            f"conf={confidence:.2f} score={master_score:.2f}"
        )
        return rec.signal_id

    # ------------------------------------------------------------------
    # 更新信号 MFE/MAE（主循环可定期调用）
    # ------------------------------------------------------------------
    def update_mfe_mae(self, signal_id: str, high_price: float, low_price: float):
        """根据持仓期间最高/最低价更新 MFE/MAE。

        应在每轮循环中对活跃持仓调用一次。
        """
        rec = self._open_signals.get(signal_id)
        if rec is None:
            return
        direction = rec.direction
        entry = rec.entry_price
        if direction == "long":
            mfe = (high_price - entry) / entry if entry > 0 else 0.0
            mae = (entry - low_price) / entry if entry > 0 else 0.0
        else:
            mfe = (entry - low_price) / entry if entry > 0 else 0.0
            mae = (high_price - entry) / entry if entry > 0 else 0.0
        rec.extra["mfe"] = max(rec.extra.get("mfe", 0.0), mfe)
        rec.extra["mae"] = max(rec.extra.get("mae", 0.0), mae)

    # ------------------------------------------------------------------
    # 记录平仓
    # ------------------------------------------------------------------
    def record_trade(
        self,
        signal_id: str,
        exit_price: float,
        exit_time: Optional[float] = None,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        cost_total: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        rec = self._open_signals.pop(signal_id, None)
        exit_ts = exit_time or time.time()
        if rec is None:
            # 尝试从数据库恢复信号元数据
            rows = self.db.query(
                "SELECT * FROM signals WHERE signal_id = ? LIMIT 1", (signal_id,)
            )
            if not rows:
                logger.warning(f"[SignalTracker] signal_id {signal_id} not found")
                return False
            meta = rows[0]
            rec = SignalRecord(
                symbol=meta["symbol"],
                direction=meta["direction"],
                entry_price=meta["entry_price"],
                confidence=meta.get("confidence") or 0.0,
                master_score=meta.get("master_score") or 0.0,
                factor_score=meta.get("factor_score") or 0.0,
                regime=meta.get("regime") or "NEUTRAL",
                timestamp=meta["timestamp"],
                signal_id=signal_id,
                extra=json.loads(meta.get("extra") or "{}"),
            )

        outcome = TradeOutcome(
            signal_id=signal_id,
            symbol=rec.symbol,
            direction=rec.direction,
            entry_price=rec.entry_price,
            exit_price=exit_price,
            entry_time=rec.timestamp,
            exit_time=exit_ts,
            pnl=pnl,
            pnl_pct=pnl_pct,
            mfe=rec.extra.get("mfe", 0.0),
            mae=rec.extra.get("mae", 0.0),
            cost_total=cost_total,
            regime=rec.regime,
            confidence=rec.confidence,
            master_score=rec.master_score,
            factor_score=rec.factor_score,
            extra=extra or {},
        )
        self.db.save_trade(outcome.to_dict())
        logger.info(
            f"[SignalTracker] closed {signal_id} {rec.symbol} PnL={pnl:.2f} "
            f"conf={rec.confidence:.2f} is_win={outcome.is_win}"
        )
        return True

    # ------------------------------------------------------------------
    # 置信度校准
    # ------------------------------------------------------------------
    def calibration_report(self, min_samples: int = 5) -> Dict[str, Any]:
        bins = self.db.get_confidence_calibration(min_samples=min_samples)
        summary = {
            "total_trades": 0,
            "overall_win_rate": 0.0,
            "overconfident_bins": [],
            "underconfident_bins": [],
            "bins": bins,
            "recommendations": [],
        }
        if not bins:
            return summary

        total = sum(b["n"] for b in bins)
        weighted_wins = sum(b["n"] * b["win_rate"] for b in bins if b["win_rate"] is not None)
        summary["total_trades"] = total
        summary["overall_win_rate"] = weighted_wins / total if total > 0 else 0.0

        for b in bins:
            try:
                lo, hi = _parse_conf_bin(b["conf_bin"])
                mid = (lo + hi) / 2.0
                actual_wr = b["win_rate"] or 0.0
                # 允许 ±0.1 容差
                if mid - actual_wr > 0.1:
                    summary["overconfident_bins"].append({
                        "bin": b["conf_bin"],
                        "expected": round(mid, 2),
                        "actual": round(actual_wr, 2),
                    })
                    summary["recommendations"].append(
                        f"置信度区间 {b['conf_bin']} 实际胜率 {actual_wr:.1%} 显著低于预期 {mid:.1%}，"
                        f"建议降低该区间模型权重或收紧过滤。"
                    )
                elif actual_wr - mid > 0.1:
                    summary["underconfident_bins"].append({
                        "bin": b["conf_bin"],
                        "expected": round(mid, 2),
                        "actual": round(actual_wr, 2),
                    })
            except Exception:
                continue

        return summary

    def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.db.get_recent_trades(limit)

    def get_recent_signals(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.db.get_recent_signals(limit)

    def close(self):
        self.db.close()


def _parse_conf_bin(label: str) -> tuple:
    """把 '0.7-0.8' 或 '0.8+' 解析为 (low, high)。"""
    label = label.strip()
    if "+" in label:
        lo = float(label.replace("+", ""))
        return lo, 1.0
    if "<" in label:
        return 0.0, float(label.replace("<", ""))
    parts = label.split("-")
    return float(parts[0]), float(parts[1])
