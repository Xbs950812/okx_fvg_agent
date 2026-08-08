"""
SQLite 持久化层。

保存：signals、trades、model_predictions、factor_scores、market_regime。
所有写入带线程锁，支持原子操作。
"""

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QuantDB:
    """量化 Agent 专用 SQLite 数据库。"""

    def __init__(self, db_path: str = "quant_agent.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        """获取线程本地连接。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    @contextmanager
    def _transaction(self):
        with self._lock:
            conn = self._conn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _init_schema(self):
        with self._transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    confidence REAL,
                    master_score REAL,
                    factor_score REAL,
                    regime TEXT,
                    volatility REAL,
                    funding_rate REAL,
                    market_condition TEXT,
                    extra TEXT,
                    created_at REAL DEFAULT (strftime('%s','now'))
                );

                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(timestamp);

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_time REAL,
                    exit_time REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    holding_time REAL,
                    mfe REAL,
                    mae REAL,
                    is_win INTEGER,
                    regime TEXT,
                    confidence REAL,
                    master_score REAL,
                    factor_score REAL,
                    cost_total REAL,
                    extra TEXT,
                    created_at REAL DEFAULT (strftime('%s','now'))
                );

                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(exit_time);

                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    symbol TEXT,
                    timestamp REAL NOT NULL,
                    prediction TEXT,
                    confidence REAL,
                    feature_vector TEXT,
                    outcome TEXT,
                    pnl REAL,
                    created_at REAL DEFAULT (strftime('%s','now'))
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions(model_name, timestamp);

                CREATE TABLE IF NOT EXISTS factor_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp REAL NOT NULL,
                    factor_name TEXT NOT NULL,
                    score REAL,
                    weight REAL,
                    created_at REAL DEFAULT (strftime('%s','now'))
                );

                CREATE INDEX IF NOT EXISTS idx_factor_symbol ON factor_scores(symbol, timestamp);
                CREATE INDEX IF NOT EXISTS idx_factor_name ON factor_scores(factor_name, timestamp);

                CREATE TABLE IF NOT EXISTS market_regime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT,
                    regime TEXT,
                    btc_return_24h REAL,
                    btc_volatility REAL,
                    market_breadth REAL,
                    funding_extreme REAL,
                    extra TEXT,
                    created_at REAL DEFAULT (strftime('%s','now'))
                );

                CREATE INDEX IF NOT EXISTS idx_regime_time ON market_regime(timestamp);
                """
            )

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------
    def save_signal(self, record: Dict[str, Any]) -> bool:
        try:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO signals
                    (signal_id, symbol, timestamp, direction, entry_price, stop_loss, take_profit,
                     confidence, master_score, factor_score, regime, volatility, funding_rate,
                     market_condition, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.get("signal_id"),
                        record.get("symbol"),
                        record.get("timestamp"),
                        record.get("direction"),
                        record.get("entry_price"),
                        record.get("stop_loss"),
                        record.get("take_profit"),
                        record.get("confidence"),
                        record.get("master_score"),
                        record.get("factor_score"),
                        record.get("regime"),
                        record.get("volatility"),
                        record.get("funding_rate"),
                        record.get("market_condition"),
                        json.dumps(record.get("extra", {}), ensure_ascii=False),
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"save_signal failed: {e}")
            return False

    # ------------------------------------------------------------------
    # trades
    # ------------------------------------------------------------------
    def save_trade(self, record: Dict[str, Any]) -> bool:
        try:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO trades
                    (signal_id, symbol, direction, entry_time, exit_time, entry_price, exit_price,
                     pnl, pnl_pct, holding_time, mfe, mae, is_win, regime, confidence,
                     master_score, factor_score, cost_total, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.get("signal_id"),
                        record.get("symbol"),
                        record.get("direction"),
                        record.get("entry_time"),
                        record.get("exit_time"),
                        record.get("entry_price"),
                        record.get("exit_price"),
                        record.get("pnl"),
                        record.get("pnl_pct"),
                        record.get("holding_time"),
                        record.get("mfe"),
                        record.get("mae"),
                        1 if record.get("is_win") else 0,
                        record.get("regime"),
                        record.get("confidence"),
                        record.get("master_score"),
                        record.get("factor_score"),
                        record.get("cost_total"),
                        json.dumps(record.get("extra", {}), ensure_ascii=False),
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"save_trade failed: {e}")
            return False

    # ------------------------------------------------------------------
    # model predictions
    # ------------------------------------------------------------------
    def save_prediction(self, record: Dict[str, Any]) -> bool:
        try:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO model_predictions
                    (model_name, symbol, timestamp, prediction, confidence, feature_vector, outcome, pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.get("model_name"),
                        record.get("symbol"),
                        record.get("timestamp"),
                        record.get("prediction"),
                        record.get("confidence"),
                        json.dumps(record.get("feature_vector", {}), ensure_ascii=False),
                        record.get("outcome"),
                        record.get("pnl"),
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"save_prediction failed: {e}")
            return False

    # ------------------------------------------------------------------
    # factor scores
    # ------------------------------------------------------------------
    def save_factor_scores(self, records: List[Dict[str, Any]]) -> bool:
        if not records:
            return True
        try:
            with self._transaction() as conn:
                for r in records:
                    conn.execute(
                        """
                        INSERT INTO factor_scores
                        (symbol, timestamp, factor_name, score, weight)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            r.get("symbol"),
                            r.get("timestamp"),
                            r.get("factor_name"),
                            r.get("score"),
                            r.get("weight"),
                        ),
                    )
            return True
        except Exception as e:
            logger.error(f"save_factor_scores failed: {e}")
            return False

    # ------------------------------------------------------------------
    # market regime
    # ------------------------------------------------------------------
    def save_market_regime(self, record: Dict[str, Any]) -> bool:
        try:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO market_regime
                    (timestamp, symbol, regime, btc_return_24h, btc_volatility, market_breadth,
                     funding_extreme, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.get("timestamp"),
                        record.get("symbol"),
                        record.get("regime"),
                        record.get("btc_return_24h"),
                        record.get("btc_volatility"),
                        record.get("market_breadth"),
                        record.get("funding_extreme"),
                        json.dumps(record.get("extra", {}), ensure_ascii=False),
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"save_market_regime failed: {e}")
            return False

    # ------------------------------------------------------------------
    # query helpers
    # ------------------------------------------------------------------
    def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        try:
            with self._lock:
                conn = self._conn()
                cur = conn.execute(sql, params)
                rows = cur.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"query failed: {e}")
            return []

    def get_recent_signals(self, limit: int = 1000) -> List[Dict[str, Any]]:
        return self.query(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
        )

    def get_recent_trades(self, limit: int = 1000) -> List[Dict[str, Any]]:
        return self.query(
            "SELECT * FROM trades ORDER BY exit_time DESC LIMIT ?", (limit,)
        )

    def get_confidence_calibration(self, min_samples: int = 5) -> List[Dict[str, Any]]:
        """按 confidence 分箱统计历史表现。"""
        sql = """
        SELECT
            CASE
                WHEN s.confidence >= 0.8 THEN '0.8+'
                WHEN s.confidence >= 0.7 THEN '0.7-0.8'
                WHEN s.confidence >= 0.6 THEN '0.6-0.7'
                WHEN s.confidence >= 0.5 THEN '0.5-0.6'
                WHEN s.confidence >= 0.4 THEN '0.4-0.5'
                WHEN s.confidence >= 0.3 THEN '0.3-0.4'
                ELSE '<0.3'
            END AS conf_bin,
            COUNT(*) AS n,
            AVG(t.is_win) AS win_rate,
            AVG(t.pnl) AS avg_pnl,
            AVG(t.pnl_pct) AS avg_pnl_pct,
            MIN(t.pnl) AS worst_pnl,
            MAX(t.pnl) AS best_pnl
        FROM signals s
        LEFT JOIN trades t ON s.signal_id = t.signal_id
        WHERE t.exit_time IS NOT NULL
        GROUP BY conf_bin
        HAVING COUNT(*) >= ?
        ORDER BY MIN(s.confidence)
        """
        return self.query(sql, (min_samples,))

    def get_trades_by_model(self, model_name: str, limit: int = 500) -> List[Dict[str, Any]]:
        return self.query(
            "SELECT * FROM model_predictions WHERE model_name = ? ORDER BY timestamp DESC LIMIT ?",
            (model_name, limit),
        )

    def close(self):
        with self._lock:
            if hasattr(self._local, "conn") and self._local.conn:
                try:
                    self._local.conn.close()
                except Exception:
                    pass
                self._local.conn = None
