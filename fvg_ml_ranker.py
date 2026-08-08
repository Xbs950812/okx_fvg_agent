"""
fvg_ml_ranker.py — FVG 信号 ML 二次评分器。

对 FVGDetector.compute_features 提取的特征做二次评分，预测"该 FVG 后续
盈利"的概率（0~1），用于在规则质量过滤之上增加一层 ML 过滤。

模型后端自动探测（按优先级）：
  xgboost → lightgbm → sklearn(RandomForest/GradientBoosting) → 内置 numpy 逻辑回归

内置逻辑回归保证零额外依赖也能训练/预测（特征需先标准化，训练时自动
估计均值/标准差并持久化，预测时复用）。
"""

from __future__ import annotations

import logging
import pickle
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 模型后端探测（模块级，只探测一次）
_BACKEND = None


def _detect_backend() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    for name, mod in (("xgboost", "xgboost"),
                      ("lightgbm", "lightgbm"),
                      ("sklearn", "sklearn.ensemble")):
        try:
            __import__(mod)
            _BACKEND = name
            logger.info(f"FVGMLRanker 后端: {name}")
            return name
        except ImportError:
            continue
    _BACKEND = "linear"
    logger.info("FVGMLRanker 后端: 内置 numpy 逻辑回归（未安装 xgboost/lightgbm/sklearn）")
    return _BACKEND


class FVGMLRanker:
    """FVG ML 排名器。

    用法:
        ranker = FVGMLRanker("fvg_model.pkl")
        ranker.train(X, y)                 # 训练并保存
        p = ranker.predict(features)       # 单个预测 0~1
        ps = ranker.predict_batch(feat_list)
        ranker.get_feature_importance()
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.scaler = None                # (mean, std) 用于标准化（linear/sklearn）
        self.feature_names: List[str] = []
        self.backend = _detect_backend()
        self._trained = False
        if model_path:
            self._load_or_train(model_path)

    # ------------------------------------------------------------------
    # 加载 / 训练
    # ------------------------------------------------------------------

    def _load_or_train(self, model_path: str):
        try:
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            self.model = obj.get("model")
            self.scaler = obj.get("scaler")
            self.feature_names = obj.get("feature_names", [])
            self.backend = obj.get("backend", self.backend)
            self._trained = bool(self.model is not None)
            logger.info(f"FVGMLRanker 加载模型 {model_path} (backend={self.backend}, "
                        f"{len(self.feature_names)} 特征)")
        except FileNotFoundError:
            logger.warning(f"模型文件不存在 {model_path}，处于未训练状态")
        except Exception as e:
            logger.warning(f"模型加载失败 {model_path}: {e}，处于未训练状态")

    def train(self, X: pd.DataFrame, y: pd.Series, params: Optional[dict] = None):
        """训练模型。X 列名即 feature_names。y ∈ {0,1}，1=有效 FVG（后续盈利）。"""
        params = params or {}
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.feature_names = list(X.columns)

        # 标准化（linear / sklearn 需要；树模型不影响但统一做）
        Xv = X.values.astype(np.float64)
        mean = np.nanmean(Xv, axis=0)
        std = np.nanstd(Xv, axis=0)
        std[std < 1e-9] = 1.0
        Xs = (Xv - mean) / std
        self.scaler = (mean, std)

        if self.backend == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=int(params.get("max_depth", 4)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            )
            self.model.fit(Xs, y.values)
        elif self.backend == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=int(params.get("max_depth", 4)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                verbose=-1,
            )
            self.model.fit(Xs, y.values)
        elif self.backend == "sklearn":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=int(params.get("max_depth", 4)),
                random_state=int(params.get("random_state", 42)),
            )
            self.model.fit(Xs, y.values)
        else:
            # 内置 numpy 逻辑回归（L2 正则梯度下降）
            self.model = self._fit_logistic(Xs, y.values, params)

        self._trained = True
        if self.model_path:
            self.save(self.model_path)

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray,
                      params: Optional[dict]) -> dict:
        """内置 L2 逻辑回归（牛顿-拉弗森或梯度下降）。"""
        n, d = X.shape
        lr = float((params or {}).get("learning_rate", 0.1))
        epochs = int((params or {}).get("epochs", 500))
        l2 = float((params or {}).get("l2", 0.01))
        Xb = np.hstack([np.ones((n, 1)), X])
        w = np.zeros(d + 1)
        for _ in range(epochs):
            z = Xb @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            grad = Xb.T @ (p - y) + l2 * np.hstack([np.array([0.0]), w[1:]])
            step = lr * grad
            if np.all(np.abs(step) < 1e-8):
                break
            w -= step
        return {"coef": w}

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, float]) -> float:
        """预测单个 FVG 有效概率（0~1）。未训练时返回 0.5 中性值。"""
        if not self._trained or self.model is None:
            return 0.5
        return float(self.predict_batch([features])[0])

    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        if not features_list or not self._trained or self.model is None:
            return [0.5] * len(features_list)
        rows = []
        for feats in features_list:
            rows.append([float(feats.get(name, 0.0)) for name in self.feature_names])
        X = np.array(rows, dtype=np.float64)
        mean, std = self.scaler
        X = (X - mean) / std

        if self.backend == "linear":
            w = self.model["coef"]
            Xb = np.hstack([np.ones((len(X), 1)), X])
            z = Xb @ w
            prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            return [float(p) for p in prob]
        prob = self.model.predict_proba(X)
        if prob.shape[1] == 1:
            # 单类别模型（训练集全 0 或全 1）：predict_proba 只返回该类别列。
            # classes_[0]==1 → 该列即 P(class=1)；==0 → 取 1-p。
            cls = int(getattr(self.model, "classes_", [0])[0])
            return [float(p[0]) if cls == 1 else float(1 - p[0]) for p in prob]
        return [float(p[1]) for p in prob]

    # ------------------------------------------------------------------
    # 特征重要性
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """返回特征重要性排序（降序）。"""
        if not self._trained or self.model is None or not self.feature_names:
            return pd.DataFrame(columns=["feature", "importance"])
        if self.backend == "xgboost":
            imp = self.model.feature_importances_
        elif self.backend == "lightgbm":
            imp = self.model.feature_importances_
        elif self.backend == "sklearn":
            imp = self.model.feature_importances_
        else:
            w = np.abs(self.model["coef"][1:])
            imp = w / (w.sum() + 1e-12)
        imp = np.asarray(imp, dtype=np.float64)
        if imp.sum() > 0:
            imp = imp / imp.sum()
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str):
        obj = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "backend": self.backend,
        }
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"FVGMLRanker 模型已保存: {path}")
