# -*- coding: utf-8 -*-
"""
train_fvg_model.py — FVG ML 模型训练脚本 (HunHeng_OS_V1.0)。

用法:
  python train_fvg_model.py --dry-run                        # 模拟数据快速验证
  python train_fvg_model.py --days 180 --model xgboost       # 正式训练
  python train_fvg_model.py --days 90 --symbols BTC-USDT-SWAP,ETH-USDT-SWAP --model auto
  python train_fvg_model.py --days 365 --source trades       # 历史采集 + 成交结果回填

样本定义 (与 fvg_backtest / fvg_training_pipeline 口径一致):
  正样本 label=1: FVG 形成后 24h 内价格达到 入场+2.25×缺口宽度
                  (入场=缺口边界; 止损=1.5×缺口宽度; 目标=止损×1.5)
  负样本 label=0: 先触发止损 或 24h 未达目标
  --source trades 额外回填: quant_agent.db 中已平仓交易若带 FVG 结构
                  (agent.py _record_signal_quant 从本版本起把 extra.fvg_* 落库),
                  用真实盈亏覆盖 label; 被扫描但未成交的 FVG(signals 无对应
                  trades) 维持历史标签并计入负样本.

输出:
  models/fvg_ranker.pkl   — FVGMLRanker 模型 (agent.py ml.enabled=true 时加载)
  控制台报告 — 样本分布 / AUC(留出集+全量) / 准确率 / 混淆矩阵 / 特征重要性

技术要求:
  零外部依赖训练路径 (xgboost 缺失自动回退 lightgbm → sklearn → 内置 numpy 逻辑回归);
  sklearn 可用时使用 roc_auc_score/confusion_matrix, 否则内置秩和法实现。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fvg_training_pipeline import FVGTrainingPipeline  # noqa: E402
from fvg_ml_ranker import FVGMLRanker  # noqa: E402

logger = logging.getLogger("train_fvg_model")

H = 3_600_000
_TS0 = 1_700_000_000_000

_DEFAULT_SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
                    "BNB-USDT-SWAP", "XRP-USDT-SWAP"]
_DEFAULT_DB = "fvg_training_data.db"
_DEFAULT_MODEL = "models/fvg_ranker.pkl"


# ---------------------------------------------------------------------------
# 评估指标（sklearn 可用时优先，否则内置实现，保证零依赖可用）
# ---------------------------------------------------------------------------

def _auc_manual(y_true, y_score) -> float:
    """内置 AUC（Mann-Whitney U 秩和法），单类别时返回 nan。"""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(y_score, dtype=float)
    i, n = 0, len(y_score)
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = float(ranks[y_true == 1].sum())
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def _roc_auc(y_true, y_score) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return _auc_manual(y_true, y_score)


def _confusion(y_true, y_pred) -> Dict[str, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
    }


# ---------------------------------------------------------------------------
# 数据采集
# ---------------------------------------------------------------------------

class _MockClient:
    """模拟 OKXClient.get_candles_enhanced（--dry-run 用）。

    每 60 根一个看涨冲动周期（跳空+4 → 爬升 0.4/根 → 回落），构造出
    gap≈2.3% 且后续价格能摸到 2.25×gap 的 FVG（label=1）与未达标
    FVG（label=0），保证双类别可训练。
    """

    def __init__(self, days: int = 60):
        self.days = days

    def get_candles_enhanced(self, inst_id: str, bar: str = "1H", limit: int = 200):
        n = min(limit, self.days * 24)
        rows = []
        for i in range(n):
            ts = _TS0 + i * H
            p = i % 60
            base = 100 + 0.4 * math.sin(i / 7)
            if p <= 9:
                o = base
            elif p == 10:            # 冲动蜡烛：跳空 +4
                o = base + 4
            elif p == 11:            # 回补但仍留缺口：低点 base+3
                o = base + 3
            elif p <= 40:            # 持续爬升：0.4/根
                o = base + 3 + (p - 11) * 0.4
            else:                    # 回落
                o = base + 14.6 - (p - 40) * 0.5
            c = o + 0.2
            h = max(o, c) + 0.3
            l = min(o, c) - 0.3
            v = 900 if p in (10, 11) else 100
            rows.append([str(ts), str(o), str(h), str(l), str(c), str(v),
                         "0", "0", "1"])
        return rows[::-1]  # OKX 时间倒序


def _load_client(config_path: str):
    """从 config.json 构造 OKXClient（正式训练数据源）。"""
    try:
        from okx_client import OKXClient  # noqa: PLC0415
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return OKXClient(config), config
    except Exception as e:
        logger.error(f"OKXClient 构造失败 ({config_path}): {e}")
        raise


def collect_historical(pipe: FVGTrainingPipeline, client, symbols: List[str],
                       days_back: int) -> int:
    """历史 K 线采集（正向收益标签，与回测引擎口径一致）。"""
    return pipe.collect_training_data(client, symbols, days_back)


def extract_features_with_confluence(fvg, candles_1h, candles_4h, context,
                                     confluence_cfg=None, strategy_cfg=None):
    """compute_features + 4H 感知汇流特征 (任务4)。

    基础特征由 fvg_detector.compute_features 产出 (25 维, 含 1H 兜底的汇流列),
    汇流列再用真实 4H K 线经 ConfluenceChecker 重算覆盖 —— 列集合不变。

    Args:
        fvg: FVGDetected / strategy.FVG
        candles_1h: 1H K 线 (正序)
        candles_4h: 4H K 线 (正序, 可空 → 回落 1H 兜底)
        context: {"current_price": float, ...}
        confluence_cfg: ConfluenceChecker 配置
        strategy_cfg: FVGDetector 配置

    Returns:
        Dict[str, float]: 25 维特征 (15 原始 + 10 汇流)
    """
    from fvg_detector import FVGDetector  # noqa: PLC0415
    from confluence import ConfluenceChecker  # noqa: PLC0415

    det = FVGDetector({**(strategy_cfg or {}), "confluence": confluence_cfg or {}})
    base = det.compute_features(fvg, candles_1h or [])
    cc = ConfluenceChecker(confluence_cfg or {})
    cr = cc.check(fvg, candles_1h or [], candles_4h or [], context or {})
    conf = cc.get_confluence_features(cr)
    return {**base, **conf}


def make_confluence_extractor(client, confluence_cfg: dict):
    """构造 4H 感知的汇流特征提取器 (训练流水线 feature_extractor 钩子)。

    对每个 (inst, HTF) 缓存一次 HTF K 线, 避免逐样本重复拉取。
    返回 callable(fvg, candles, detector) -> Dict[str, float]。
    """
    from confluence import ConfluenceChecker  # noqa: PLC0415
    from strategy import candles_from_raw  # noqa: PLC0415

    cc = ConfluenceChecker(confluence_cfg or {})
    htf_cache: Dict[Tuple[str, str], List] = {}
    htf_map = {"1m": "1H", "5m": "1H", "15m": "4H", "30m": "4H",
               "1H": "4H", "2H": "1D", "4H": "1D", "1D": "1W"}

    def extractor(fvg, candles, detector) -> Dict[str, float]:
        if not candles:
            return {}
        # 当前价: 形成后第一根收盘价, 否则最后收盘价
        idx = getattr(fvg, "end_idx", -1) + 1
        if 0 <= idx < len(candles):
            cur = float(candles[idx].close)
        else:
            cur = float(candles[-1].close)
        ctx = {"current_price": cur}
        if detector is not None:
            detector._last_context = ctx
            base = detector.compute_features(fvg, candles)
        else:
            base = extract_features_with_confluence(fvg, candles, [], ctx)
        # HTF K 线 (缓存)
        inst = getattr(fvg, "inst_id", "") or ""
        tf = getattr(fvg, "timeframe", "1H") or "1H"
        htf_tf = htf_map.get(tf, "4H")
        key = (inst, htf_tf)
        c4 = htf_cache.get(key)
        if c4 is None:
            try:
                raw = client.get_candles_enhanced(inst, bar=htf_tf, limit=200)
                c4 = candles_from_raw(raw) if raw else []
            except Exception as e:
                logger.debug(f"HTF K线获取失败 {inst} {htf_tf}: {e}")
                c4 = []
            htf_cache[key] = c4
        cr = cc.check(fvg, candles, c4, ctx)
        conf = cc.get_confluence_features(cr)
        return {**base, **conf}

    return extractor


def collect_trades_overlay(pipe: FVGTrainingPipeline, quant_db_path: str,
                           max_trades: int = 200) -> int:
    """用已成交结果回填样本标签。

    从 quant_agent.db 读已平仓交易；对含 extra.fvg_* 结构且能在训练库中
    匹配到 (symbol, timeframe, 形成时间) 的样本，用真实 is_win 覆盖 label。
    注意: 仅覆盖训练库中已存在的样本（特征来自历史 K 线），不新增行。
    """
    import sqlite3  # noqa: PLC0415
    conn = sqlite3.connect(quant_db_path)
    try:
        rows = conn.execute(
            "SELECT signal_id, symbol, direction, entry_time, is_win "
            "FROM trades ORDER BY exit_time DESC LIMIT ?", (max_trades,)
        ).fetchall()
    except sqlite3.Error as e:
        logger.warning(f"读取 quant_agent.db 交易失败: {e}")
        return 0
    finally:
        conn.close()
    if not rows:
        logger.info("quant_agent.db 无已平仓交易，跳过回填")
        return 0

    bar_ms = {"1H": H, "4H": 4 * H, "1D": 24 * H}
    n_overlaid = 0
    for signal_id, symbol, direction, entry_time, is_win in rows:
        # 读取该 signal 的 FVG 结构
        dconn = sqlite3.connect(quant_db_path)
        try:
            sig = dconn.execute(
                "SELECT extra FROM signals WHERE signal_id = ? LIMIT 1",
                (signal_id,)
            ).fetchone()
        except sqlite3.Error:
            sig = None
        finally:
            dconn.close()
        if not sig:
            continue
        extra = json.loads(sig[0] or "{}")
        tf = extra.get("fvg_timeframe")
        fvg_ts = int(extra.get("fvg_candle_ts") or 0)
        if not tf or not fvg_ts:
            continue  # 旧版本信号未落 FVG 结构，跳过
        # 匹配训练库样本（同 symbol + 同 timeframe + 形成时间相近）
        conn2 = sqlite3.connect(pipe.db_path)
        try:
            matched = conn2.execute(
                "SELECT id, label FROM fvg_samples "
                "WHERE inst_id=? AND timeframe=? AND "
                "ABS(ts - ?) <= ? LIMIT 1",
                (symbol, tf, fvg_ts, int(bar_ms.get(tf, H) * 2)),
            ).fetchone()
        except sqlite3.Error:
            matched = None
        finally:
            conn2.close()
        if matched:
            new_label = 1 if int(is_win) == 1 else 0
            conn3 = sqlite3.connect(pipe.db_path)
            try:
                conn3.execute("UPDATE fvg_samples SET label=? WHERE id=?",
                              (new_label, matched[0]))
                conn3.commit()
            finally:
                conn3.close()
            n_overlaid += 1
    logger.info(f"成交结果回填: {n_overlaid} 条标签已用真实盈亏覆盖")
    return n_overlaid


# ---------------------------------------------------------------------------
# 训练 + 评估
# ---------------------------------------------------------------------------

def train_and_report(pipe: FVGTrainingPipeline, model_type: str,
                     params: Optional[dict], model_path: str,
                     split_ratio: float = 0.8) -> Tuple[FVGMLRanker, Dict[str, Any]]:
    """训练模型并输出报告（留出集评估 + 全量重训保存）。"""
    X, y = pipe.load_samples()
    if len(X) < 20:
        logger.warning(f"样本不足 ({len(X)})，无法训练。先跑 --days 更大的采集")
        return FVGMLRanker(), {"error": "样本不足"}

    ranker = FVGMLRanker()
    if model_type != "auto":
        import importlib  # noqa: PLC0415
        try:
            importlib.import_module(model_type if model_type != "sklearn" else "sklearn")
            ranker.backend = model_type
        except ImportError:
            logger.warning(f"指定后端 {model_type} 不可用，回退自动探测")

    # ---- 留出集评估（先只在训练子集上训练，避免数据泄漏）----
    _n = max(1, int(len(X) * split_ratio))
    Xtr, ytr = X.iloc[:_n], y.iloc[:_n]
    Xva, yva = X.iloc[_n:], y.iloc[_n:]
    if len(Xva) > 0:
        ranker.train(Xtr, ytr, params)
        val_preds = np.array(ranker.predict_batch(Xva.to_dict("records")))
        val_labels = (val_preds >= 0.5).astype(int)
        val_cm = _confusion(yva.values, val_labels)
        val_acc = float(np.mean(val_labels == yva.values))
        val_auc = _roc_auc(yva.values, val_preds) if len(yva) >= 10 else float("nan")
    else:
        val_cm, val_acc, val_auc = {}, float("nan"), float("nan")

    # ---- 全量重训并保存（保证保存模型用了全部数据）----
    ranker.train(X, y, params)
    os.makedirs(os.path.dirname(os.path.abspath(model_path)) or ".", exist_ok=True)
    ranker.model_path = model_path
    ranker.save(model_path)

    # ---- 全量 in-sample 指标 ----
    all_preds = np.array(ranker.predict_batch(X.to_dict("records")))
    all_labels = (all_preds >= 0.5).astype(int)
    cm = _confusion(y.values, all_labels)
    acc = float(np.mean(all_labels == y.values))
    auc = _roc_auc(y.values, all_preds)

    # ---- 特征重要性 ----
    imp = ranker.get_feature_importance()
    top = imp.head(10).to_dict("records") if len(imp) else []

    report = {
        "n_samples": int(len(X)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "positive_rate": round(float((y == 1).mean()), 3),
        "majority_baseline_acc": round(max(float(y.mean()), 1 - float(y.mean())), 3),
        "backend": ranker.backend,
        "holdout": {
            "n": int(len(Xva)),
            "accuracy": round(val_acc, 3) if not math.isnan(val_acc) else None,
            "auc": round(val_auc, 3) if not math.isnan(val_auc) else None,
            "confusion": val_cm,
        },
        "in_sample": {
            "n": int(len(X)),
            "accuracy": round(acc, 3),
            "auc": round(auc, 3) if not math.isnan(auc) else None,
            "confusion": cm,
        },
        "feature_importance_top10": top,
        "model_path": model_path,
    }
    return ranker, report


def print_report(report: Dict[str, Any]):
    if report.get("error"):
        print(f"\n训练失败: {report['error']}")
        return
    print("\n" + "=" * 56)
    print("FVG ML 模型训练报告")
    print("=" * 56)
    print(f"后端           : {report['backend']}")
    print(f"样本量         : {report['n_samples']} "
          f"(正={report['n_positive']}, 负={report['n_negative']}, "
          f"正样本率={report['positive_rate']:.1%})")
    print(f"多数类基准     : 准确率 {report['majority_baseline_acc']:.1%} (AUC 0.50)")
    h = report.get("holdout") or {}
    print(f"留出集 (n={h.get('n')}): 准确率 {h.get('accuracy')}  AUC {h.get('auc')}")
    cm = h.get("confusion") or {}
    if cm:
        print(f"   混淆矩阵 TP={cm.get('tp')} FP={cm.get('fp')} "
              f"TN={cm.get('tn')} FN={cm.get('fn')}")
    s = report.get("in_sample") or {}
    print(f"全量训练集   : 准确率 {s.get('accuracy')}  AUC {s.get('auc')}")
    print(f"模型已保存     : {report['model_path']}")
    print("-" * 56)
    print("特征重要性 Top10:")
    for i, row in enumerate(report.get("feature_importance_top10", []), 1):
        print(f"  {i:2d}. {row['feature']:<24s} {row['importance']:.3f}")
    print("=" * 56)
    print("解释: AUC > 多数类基准(0.5) 说明 ML 分层能力存在; 实盘建议 AUC≥0.58 且\n"
          "     留出集与全量差距 <0.08 才开启 ml.enabled=true（防止过拟合上线）。")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def run_dry_run(model_type: str = "auto", workdir: Optional[str] = None,
                use_confluence: bool = True) -> Dict[str, Any]:
    """模拟数据快速验证：采集→训练→报告（不落正式模型/库）。"""
    workdir = workdir or tempfile.gettempdir()
    db = os.path.join(workdir, "fvg_dryrun.db")
    model = os.path.join(workdir, "fvg_dryrun.pkl")
    for p in (db, model):
        if os.path.exists(p):
            os.remove(p)
    cfg = {"min_fvg_width_pct": {"1H": 1.5, "4H": 3.0},
           "abnormal_sigma": 3.0, "abnormal_volume_ratio": 5.0,
           "abnormal_lookback": {"1H": 50, "4H": 50}}
    pipe = FVGTrainingPipeline(db_path=db, strategy_cfg=cfg, timeframes=["1H"])
    if use_confluence:
        pipe.feature_extractor = make_confluence_extractor(
            _MockClient(days=60), {})
    n = collect_historical(pipe, _MockClient(days=60),
                           ["AAA-USDT-SWAP", "BBB-USDT-SWAP"], days_back=60)
    print(f"[dry-run] 采集样本: {n}")
    ranker, report = train_and_report(pipe, model_type, None, model)
    print_report(report)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="FVG ML 模型训练")
    ap.add_argument("--dry-run", action="store_true",
                    help="用模拟数据快速验证整条流水线")
    ap.add_argument("--days", type=int, default=180, help="回溯天数 (默认 180)")
    ap.add_argument("--symbols", type=str, default="",
                    help="合约列表，逗号分隔 (默认核心 5 币种)")
    ap.add_argument("--model", type=str, default="auto",
                    choices=["auto", "xgboost", "lightgbm", "sklearn", "linear"],
                    help="模型后端 (默认 auto 自动探测)")
    ap.add_argument("--config", type=str, default="config.json",
                    help="config.json 路径 (OKX 密钥/代理)")
    ap.add_argument("--db", type=str, default=_DEFAULT_DB,
                    help="训练样本 SQLite 路径")
    ap.add_argument("--model-path", type=str, default=_DEFAULT_MODEL,
                    help="模型输出路径")
    ap.add_argument("--source", type=str, default="historical",
                    choices=["historical", "trades"],
                    help="historical=仅历史K线; trades=历史+成交结果回填(需 quant_agent.db)")
    ap.add_argument("--quant-db", type=str, default="quant_agent.db",
                    help="quant_agent.db 路径 (--source trades 时用)")
    ap.add_argument("--params", type=str, default="",
                    help='训练超参 JSON，如 \'{"n_estimators": 300}\'')
    ap.add_argument("--no-confluence", action="store_true",
                    help="禁用 4H 感知汇流特征 (默认按 config.confluence.enabled 启用)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.dry_run:
        run_dry_run(model_type=args.model, use_confluence=not args.no_confluence)
        return 0

    client, config = _load_client(args.config)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] \
        or _DEFAULT_SYMBOLS
    params = json.loads(args.params) if args.params else None

    pipe = FVGTrainingPipeline(
        db_path=args.db,
        strategy_cfg=config.get("strategy", {}),
    )
    con_cfg = config.get("confluence", {})
    if (not args.no_confluence and con_cfg.get("enabled", True)):
        pipe.feature_extractor = make_confluence_extractor(client, con_cfg)
        logger.info("已启用 4H 感知汇流特征提取 (config.confluence)")
    n = collect_historical(pipe, client, symbols, args.days)
    print(f"历史采集: {n} 样本 (db={args.db})")
    if n == 0:
        logger.error("未采集到任何样本，检查 --symbols/网络/API 密钥")
        return 1

    if args.source == "trades":
        collect_trades_overlay(pipe, args.quant_db)

    ranker, report = train_and_report(pipe, args.model, params, args.model_path)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
