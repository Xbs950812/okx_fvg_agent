"""
fvg_training_pipeline.py — FVG 训练数据采集与 ML 模型训练流水线。

离线流水线：
  1. collect_training_data(): 从 OKX 历史 K 线采集训练样本
     - features: FVGDetector.compute_features() 的 15 维特征
     - label: 1=该 FVG 在后续 24h 内盈利超过止损的 1.5 倍，0=否则
  2. train_and_save(): 训练 FVGMLRanker 并保存到 models/ 目录
  3. validate_cross_timeframe(): 跨品种/时间框架验证，检测过拟合

零外部依赖：仅使用 numpy/pandas；OKX SDK 通过注入的 client 使用（本模块
不 import okx），模型后端由 fvg_ml_ranker 自动探测（xgboost/lightgbm/
sklearn/内置线性，可插拔）。
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fvg_detector import FVGDetector, FVGDetected
from fvg_ml_ranker import FVGMLRanker

logger = logging.getLogger(__name__)

# 时间框架 → K 线毫秒数（用于 24h 窗口换算）
_BAR_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1H": 3_600_000, "2H": 7_200_000, "4H": 14_400_000, "1D": 86_400_000,
}

# label 判定参数（与 fvg_backtest 的止损口径一致）
_STOP_MULT = 1.5      # 止损 = FVG 宽度 × 1.5
_TARGET_MULT = 1.5    # 盈利目标 = 止损的 1.5 倍
_WINDOW_HOURS = 24    # 判定时间窗口


class FVGTrainingPipeline:
    """管理训练数据的采集和模型训练。"""

    def __init__(
        self,
        db_path: str = "fvg_training_data.db",
        strategy_cfg: Optional[dict] = None,
        timeframes: Optional[List[str]] = None,
        feature_extractor: Optional[Any] = None,
    ):
        self.db_path = db_path
        self.strategy_cfg = strategy_cfg or {}
        self.timeframes = timeframes or list(self.strategy_cfg.get(
            "min_fvg_width_pct", {"1H": 1.5, "4H": 3.0}).keys()) or ["1H", "4H"]
        # 可选特征提取器: callable(fvg, candles, detector) -> Dict[str, float]。
        # 用于训练时注入 4H 感知的汇流特征 (train_fvg_model.extract_features_with_confluence)。
        self.feature_extractor = feature_extractor
        self._ensure_db()

    # ------------------------------------------------------------------
    # 数据采集
    # ------------------------------------------------------------------

    def collect_training_data(
        self, client, inst_ids: List[str], days_back: int = 365
    ) -> int:
        """从历史数据中采集训练样本。

        Args:
            client: OKXClient 实例（注入，本模块不依赖 SDK）
            inst_ids: 合约 ID 列表，如 ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
            days_back: 回溯天数

        Returns:
            新增样本数
        """
        added = 0
        detector = FVGDetector(self.strategy_cfg)
        for inst_id in inst_ids:
            for tf in self.timeframes:
                candles = self._fetch_history(client, inst_id, tf, days_back)
                if not candles or len(candles) < 60:
                    logger.warning(f"{inst_id} {tf} 历史数据不足，跳过")
                    continue
                try:
                    fvgs = detector.detect({tf: candles})
                except Exception as e:
                    logger.warning(f"{inst_id} {tf} 检测异常: {e}")
                    continue
                for fvg in fvgs:
                    label = self._label(fvg, candles, tf)
                    if label is None:
                        continue
                    # 特征提取（context 用形成后第一根收盘价作为当前价）
                    try:
                        if self.feature_extractor is not None:
                            feats = self.feature_extractor(fvg, candles, detector)
                            if not feats:
                                continue
                        else:
                            detector._last_context = {
                                "current_price": self._post_price(candles, fvg),
                            }
                            feats = detector.compute_features(fvg, candles)
                    except Exception as e:
                        logger.debug(f"特征提取异常 {inst_id}: {e}")
                        continue
                    self._insert_sample(
                        inst_id, tf, fvg, feats, label
                    )
                    added += 1
                logger.info(f"{inst_id} {tf}: {len(fvgs)} FVG, 新增样本 {added}")
        logger.info(f"collect_training_data 完成: 累计新增 {added} 样本")
        return added

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train_and_save(
        self,
        model_type: str = "xgboost",
        params: Optional[dict] = None,
        model_path: Optional[str] = None,
        split_ratio: float = 0.8,
    ) -> FVGMLRanker:
        """训练模型并保存到 models/ 目录。

        Args:
            model_type: 模型后端（xgboost/lightgbm/sklearn/linear，
                        传 auto 则自动探测可用后端）
            params: 训练超参数
            model_path: 保存路径，默认 models/fvg_ranker.pkl
            split_ratio: 训练/验证切分比例

        Returns:
            训练好的 FVGMLRanker
        """
        X, y = self.load_samples()
        if len(X) < 20:
            logger.warning(f"样本不足 ({len(X)})，无法训练。先运行 collect_training_data")
            return FVGMLRanker()
        vc = y.value_counts().to_dict()
        if y.nunique() < 2:
            logger.warning(
                f"样本仅含单一类别 (label 分布 {vc})，模型退化为恒等预测。"
                "建议增加样本量/品种后重训，正样本不足时 min_ml_score 过滤意义有限"
            )

        ranker = FVGMLRanker()
        if model_type != "auto":
            # 强制指定后端（若该库不可用则回退探测）
            import importlib
            try:
                if model_type == "xgboost":
                    importlib.import_module("xgboost")
                elif model_type == "lightgbm":
                    importlib.import_module("lightgbm")
                elif model_type == "sklearn":
                    importlib.import_module("sklearn")
                ranker.backend = model_type
            except ImportError:
                logger.warning(f"指定后端 {model_type} 不可用，回退自动探测")
        ranker.train(X, y, params)

        model_path = model_path or "models/fvg_ranker.pkl"
        os.makedirs(os.path.dirname(os.path.abspath(model_path)) or ".", exist_ok=True)
        ranker.model_path = model_path
        ranker.save(model_path)

        # 验证集评估
        _n = max(1, int(len(X) * split_ratio))
        Xv, yv = X.iloc[_n:], y.iloc[_n:]
        if len(Xv) > 0:
            preds = ranker.predict_batch(Xv.to_dict("records"))
            acc = float(np.mean([1 if (p >= 0.5) == (int(l) == 1) else 0
                                 for p, l in zip(preds, yv)]))
            logger.info(f"验证集 (n={len(Xv)}) 准确率: {acc:.2%}")
        return ranker

    # ------------------------------------------------------------------
    # 跨品种验证（过拟合检测）
    # ------------------------------------------------------------------

    def validate_cross_timeframe(
        self, model: Optional[FVGMLRanker], test_symbols: List[str]
    ) -> Dict[str, Any]:
        """跨品种/跨时间框架验证，检测过拟合。

        用训练时未见过的币种样本评估模型泛化能力：
          - 整体准确率 / 正样本召回率
          - 高分样本（ML>0.6）的实际正样本占比（越高于全局占比越好）
          - 按时间框架分组的准确率

        Args:
            model: FVGMLRanker；None 时用基准（全部预测 0.5）
            test_symbols: 测试币种列表

        Returns:
            dict: 评估指标
        """
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT inst_id, timeframe, features_json, label FROM fvg_samples"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            logger.warning("无样本数据")
            return {}

        # 测试币种过滤（模糊匹配前缀）
        df_rows = []
        labels = []
        for inst_id, tf, feat_json, label in rows:
            if any(inst_id.startswith(t) or t in inst_id for t in test_symbols):
                df_rows.append(json.loads(feat_json))
                labels.append(int(label))
        if len(df_rows) < 10:
            logger.warning(f"测试币种样本不足 ({len(df_rows)})")
            return {}

        Xt = pd.DataFrame(df_rows)
        yt = np.array(labels)

        if model is not None and model._trained:
            preds = np.array(model.predict_batch(df_rows))
            acc = float(np.mean((preds >= 0.5) == (yt == 1)))
            pos_recall = float(np.mean(preds[yt == 1] >= 0.5)) if (yt == 1).any() else 0.0
            base_pos = float(yt.mean())
            hi = preds >= 0.6
            hi_precision = float(yt[hi].mean()) if hi.any() else 0.0
        else:
            acc = float(max(base := yt.mean(), 1 - base))
            pos_recall, hi_precision = base, base

        # 按时间框架分组
        tf_inst = [r[1] for r in rows if any(
            t in r[0] for t in test_symbols)]
        by_tf: Dict[str, float] = {}
        for tf in sorted(set(tf_inst)):
            mask = [i for i, r in enumerate(rows) if r[1] == tf and any(
                t in r[0] for t in test_symbols)]
            if len(mask) >= 5:
                m = np.array([labels[i] for i in mask])
                by_tf[tf] = round(float(m.mean()), 3)

        result = {
            "test_symbols": test_symbols,
            "n_test": len(Xt),
            "base_positive_rate": round(float(yt.mean()), 3),
            "accuracy": round(acc, 3),
            "positive_recall": round(pos_recall, 3),
            "high_score_positive_rate": round(hi_precision, 3),
            "by_timeframe": by_tf,
        }
        logger.info(f"跨品种验证: {result}")
        return result

    # ------------------------------------------------------------------
    # 样本加载
    # ------------------------------------------------------------------

    def load_samples(self) -> Tuple[pd.DataFrame, pd.Series]:
        """从 SQLite 加载全部样本 → (X, y)。"""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT features_json, label FROM fvg_samples"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return pd.DataFrame(), pd.Series(dtype=int)
        feats = [json.loads(r[0]) for r in rows]
        labels = pd.Series([int(r[1]) for r in rows])
        return pd.DataFrame(feats), labels

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _ensure_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fvg_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inst_id TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    features_json TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fvg_samples_inst ON "
                "fvg_samples(inst_id, timeframe)"
            )
            conn.commit()
        finally:
            conn.close()

    def _insert_sample(
        self, inst_id: str, tf: str, fvg: FVGDetected,
        feats: Dict[str, float], label: int,
    ):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO fvg_samples "
                "(inst_id, timeframe, ts, features_json, label, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (inst_id, tf, int(fvg.formation_ts),
                 json.dumps(feats), int(label), time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def _fetch_history(self, client, inst_id: str, bar: str,
                       days_back: int) -> List:
        """拉取历史 K 线（OKXHistoryLoader 内部自动分页）。"""
        bar_ms = _BAR_MS.get(bar, 3_600_000)
        need = int(days_back * 86_400_000 / bar_ms) + 20
        try:
            raw = client.get_candles_enhanced(inst_id, bar=bar, limit=min(need, 5000))
        except Exception as e:
            logger.warning(f"历史K线获取失败 {inst_id} {bar}: {e}")
            return []
        if not raw:
            return []
        # 转 Candle（复用 strategy 转换，鸭子类型：含 timestamp/open/high/low/close/volume）
        from strategy import candles_from_raw
        return candles_from_raw(raw)

    def _post_price(self, candles: List, fvg: FVGDetected) -> float:
        """形成后第一根 K 线收盘价（特征提取的 current_price）。"""
        idx = fvg.end_idx + 1
        if idx < len(candles):
            return float(candles[idx].close)
        return float(candles[-1].close)

    def _label(self, fvg: FVGDetected, candles: List, tf: str) -> Optional[int]:
        """计算样本标签。

        1=后续 24h 内盈利超过止损的 1.5 倍（即价格达到入场 +2.25×缺口宽度）
        0=先触发止损（入场 -1.5×缺口宽度）或 24h 未达标
        None=数据不足（缺口形成后无后续 K 线）
        """
        bar_ms = _BAR_MS.get(tf, 3_600_000)
        win = max(1, int(_WINDOW_HOURS * 3_600_000 / bar_ms))
        gap = fvg.gap_high - fvg.gap_low
        if gap <= 0:
            return None
        stop_span = gap * _STOP_MULT
        target = stop_span * _TARGET_MULT  # = 2.25 × gap

        end = min(len(candles), fvg.end_idx + 1 + win)
        if end <= fvg.end_idx + 1:
            return None
        for j in range(fvg.end_idx + 1, end):
            c = candles[j]
            if fvg.direction == "bullish":
                if c.low <= fvg.gap_low - stop_span:
                    return 0
                if c.high >= fvg.gap_low + target:
                    return 1
            else:
                if c.high >= fvg.gap_high + stop_span:
                    return 0
                if c.low <= fvg.gap_high - target:
                    return 1
        return 0  # 24h 内未达到盈利目标
