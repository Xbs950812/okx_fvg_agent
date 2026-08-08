"""
fvg_detector.py — 独立 FVG 检测模块。

从单体 agent 架构中拆分的纯检测层，不依赖任何外部 API / 状态管理器 /
strategy 模块（避免循环依赖），K 线对象采用鸭子类型（duck-typing），
接受任何含 open/high/low/close/volume/timestamp 属性的对象。

核心能力：
  - detect(): 标准 ICT 三蜡烛 FVG 检测（看涨/看跌），与 strategy.detect_fvg
    算法一致，附带异常波动标记（MAD 鲁棒 z-score + 量比）
  - filter_by_quality(): 基于市场上下文的规则质量过滤
  - compute_features(): 为 ML 二次评分提取 15 维特征向量

数据流（第一层检测 → 第二层 ML 评分 → 第三层回测）：
  candles_by_tf → detect() → FVGDetected[]
  FVGDetected[] + context → filter_by_quality() → FVGDetected[]
  FVGDetected + candles → compute_features() → {feature: value}
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class FVGDetected:
    """独立检测器产出的 FVG 结构（与 strategy.FVG 分离，避免耦合）。"""
    inst_id: str                    # 合约 ID
    timeframe: str                  # "1H" | "4H"
    start_idx: int                  # 缺口左蜡烛索引（c0）
    end_idx: int                    # 缺口右蜡烛索引（c2）
    gap_high: float                 # 缺口上沿
    gap_low: float                  # 缺口下沿
    width_pct: float                # 缺口宽度百分比
    volume_at_formation: float      # 推动蜡烛（中间那根）成交量
    formation_price: float          # 形成时价格（推动蜡烛收盘价）
    direction: str                  # "bullish" | "bearish"
    is_abnormal: bool               # 是否异常波动（σ + 量比双条件）
    quality_score: float            # 初始质量评分（0-1，规则计算）
    sigma: float = 0.0              # 异常 σ（MAD 鲁棒 z-score）
    volume_ratio: float = 1.0       # 推动蜡烛量比（vs 前 lookback 均量）
    formation_ts: int = 0           # 形成时间戳（c2.timestamp）


# ---------------------------------------------------------------------------
# 检测器
# ---------------------------------------------------------------------------

class FVGDetector:
    """独立 FVG 检测器。

    config 键:
        min_fvg_width_pct: dict  各时间框架最小缺口宽度百分比，如 {"1H": 1.5, "4H": 3.0}
        abnormal_sigma: float    异常波动标准差倍数（默认 3.0）
        abnormal_volume_ratio: float  异常成交量倍数（默认 5.0）
        abnormal_lookback: dict  异常检测回溯窗口，如 {"1H": 50, "4H": 50}
        min_candle_count: int    最少 K 线数量（默认 3）
        min_quality: float       质量过滤下限（0-1，默认 0.0 关闭）
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self._last_context: dict = {}   # 最近一次 filter/特征提取的市场上下文

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------

    def detect(self, candles_by_tf: Dict[str, List]) -> List[FVGDetected]:
        """纯 FVG 检测逻辑（多时间框架）。

        标准 ICT 三蜡烛模式：
          - 看涨 FVG: c0.high < c2.low → 缺口 [c0.high, c2.low]
          - 看跌 FVG: c0.low > c2.high → 缺口 [c2.high, c0.low]

        Args:
            candles_by_tf: {"1H": [...], "4H": [...]}，K 线须正序（旧→新）

        Returns:
            正序 FVGDetected 列表
        """
        fvgs: List[FVGDetected] = []
        min_candle_count = int(self.config.get("min_candle_count", 3))

        for tf, candles in (candles_by_tf or {}).items():
            if not candles or len(candles) < min_candle_count:
                continue
            min_width = float(self.config.get("min_fvg_width_pct", {}).get(tf, 1.5))
            lookback = int(self.config.get("abnormal_lookback", {}).get(tf, 50))
            sigma_th = float(self.config.get("abnormal_sigma", 3.0))
            vol_th = float(self.config.get("abnormal_volume_ratio", 5.0))

            for i in range(len(candles) - 2):
                c0, c1, c2 = candles[i], candles[i + 1], candles[i + 2]

                # ---- 看涨 FVG ----
                if c0.high < c2.low:
                    gap_high, gap_low = c2.low, c0.high
                    if gap_low <= 0:
                        continue
                    width_pct = (gap_high - gap_low) / gap_low * 100
                    if width_pct >= min_width:
                        is_ab, sigma, vol_ratio = self._detect_abnormal(
                            candles, i + 1, sigma_th, vol_th, lookback
                        )
                        q = self._quality_score(
                            width_pct, is_ab, vol_ratio, sigma
                        )
                        fvgs.append(FVGDetected(
                            inst_id="", timeframe=tf,
                            start_idx=i, end_idx=i + 2,
                            gap_high=gap_high, gap_low=gap_low,
                            width_pct=width_pct,
                            volume_at_formation=c1.volume,
                            formation_price=c1.close,
                            direction="bullish",
                            is_abnormal=is_ab, quality_score=q,
                            sigma=sigma, volume_ratio=vol_ratio,
                            formation_ts=c2.timestamp,
                        ))

                # ---- 看跌 FVG ----
                elif c0.low > c2.high:
                    gap_high, gap_low = c0.low, c2.high
                    if gap_low <= 0:
                        continue
                    width_pct = (gap_high - gap_low) / gap_low * 100
                    if width_pct >= min_width:
                        is_ab, sigma, vol_ratio = self._detect_abnormal(
                            candles, i + 1, sigma_th, vol_th, lookback
                        )
                        q = self._quality_score(
                            width_pct, is_ab, vol_ratio, sigma
                        )
                        fvgs.append(FVGDetected(
                            inst_id="", timeframe=tf,
                            start_idx=i, end_idx=i + 2,
                            gap_high=gap_high, gap_low=gap_low,
                            width_pct=width_pct,
                            volume_at_formation=c1.volume,
                            formation_price=c1.close,
                            direction="bearish",
                            is_abnormal=is_ab, quality_score=q,
                            sigma=sigma, volume_ratio=vol_ratio,
                            formation_ts=c2.timestamp,
                        ))

        return fvgs

    # ------------------------------------------------------------------
    # 质量过滤
    # ------------------------------------------------------------------

    def filter_by_quality(
        self, fvgs: List[FVGDetected], context: dict
    ) -> List[FVGDetected]:
        """基于市场上下文的质量规则过滤。

        上下文缺失对应键时该项跳过（保守放行，保证与旧行为兼容）。
        context 键:
            current_price: float   当前价格
            atr: float            平均真实波幅
            volume_profile: dict  成交量分布（预留）
            funding_rate: float   资金费率
            spread_pct: float     买卖价差（%）

        规则:
            1. 价格已远离缺口（> 2x 缺口宽度）→ 失效
            2. ATR 相对缺口宽度异常（width/atr > 15）→ 巨型缺口否决
            3. spread 超限（> 0.5%）→ 流动性不足否决
            4. 资金费率方向反向且超限（与过滤链互补）→ 否决
            5. quality_score < min_quality → 否决
        """
        self._last_context = context or {}
        if not fvgs:
            return fvgs
        ctx = self._last_context
        min_quality = float(self.config.get("min_quality", 0.0))

        out: List[FVGDetected] = []
        for f in fvgs:
            # 规则 1: 价格远离缺口失效
            cp = ctx.get("current_price")
            if cp and cp > 0:
                _span = (f.gap_high - f.gap_low)
                if f.direction == "bullish" and cp < f.gap_low - 2 * _span:
                    continue
                if f.direction == "bearish" and cp > f.gap_high + 2 * _span:
                    continue

            # 规则 2: ATR 巨型缺口否决（width/atr 过大说明缺口异常）
            atr = ctx.get("atr")
            if atr and atr > 0:
                _gap = f.gap_high - f.gap_low
                if _gap > 0 and _gap / atr > 15.0:
                    continue

            # 规则 3: 价差超限
            spread = ctx.get("spread_pct")
            if spread is not None and float(spread) > 0.5:
                continue

            # 规则 4: 资金费率反向超限（做多吃高正费率 / 做空吃高负费率）
            fr = ctx.get("funding_rate")
            if fr is not None and abs(float(fr)) > 0.01:
                if f.direction == "bullish" and float(fr) > 0.01:
                    continue
                if f.direction == "bearish" and float(fr) < -0.01:
                    continue

            # 规则 5: 质量下限
            if min_quality > 0 and f.quality_score < min_quality:
                continue

            out.append(f)

        if len(out) < len(fvgs):
            logger.debug(f"filter_by_quality: {len(fvgs)} → {len(out)}")
        return out

    # ------------------------------------------------------------------
    # 特征提取（ML 二次评分输入）
    # ------------------------------------------------------------------

    def compute_features(
        self, fvg: FVGDetected, candles: List
    ) -> Dict[str, float]:
        """为 ML 模型提取 25 维特征向量 (15 原始 + 10 汇流)。

        依赖外部数据（BTC 相关性 / 历史回补率 / 订单簿）的特征在数据
        不可用时返回中性值（0 或 0.5），不抛异常——保证单根 K 线也能算。
        汇流特征 (16-25) 在 config.confluence.enabled 时经 ConfluenceChecker
        计算并惰性缓存于 fvg 上，数据不足/异常时返回中性值。
        """
        feats: Dict[str, float] = {}
        ctx = self._last_context or {}

        # 1. 缺口宽度百分比
        feats["fvg_width_pct"] = float(fvg.width_pct)

        # 2. 缺口宽度 / ATR(14)
        atr14 = self._atr(candles, 14)
        feats["atr_ratio"] = (
            (fvg.gap_high - fvg.gap_low) / atr14 if atr14 > 0 else 0.0
        )

        # 3. 形成时成交量 / 前 20 根均量
        feats["volume_ratio"] = float(fvg.volume_ratio)

        # 4-5. 价格到 MA20 / MA50 的距离百分比
        ma20 = self._sma(candles, 20)
        ma50 = self._sma(candles, 50)
        last_close = candles[-1].close if candles else 0.0
        feats["distance_to_ma20"] = (
            (last_close - ma20) / ma20 * 100 if ma20 > 0 else 0.0
        )
        feats["distance_to_ma50"] = (
            (last_close - ma50) / ma50 * 100 if ma50 > 0 else 0.0
        )

        # 6. 形成前 20 根趋势强度（线性回归斜率归一化到价格比例）
        feats["prev_trend_strength"] = self._trend_strength(candles, fvg.start_idx, 20)

        # 7. 形成前 20 根波动率（对数收益标准差）
        feats["prev_volatility"] = self._prev_volatility(candles, fvg.start_idx, 20)

        # 8. 缺口在近期价格区间（60 根）的位置 0~1
        feats["gap_position"] = self._gap_position(candles, fvg)

        # 9. 当前价格回到缺口内的程度（0=未回补，1=完全回补）
        feats["retracement_pct"] = self._retracement(fvg, ctx.get("current_price"))

        # 10. 动量背离（价格创新高但 RSI 未创新高）
        feats["momentum_divergence"] = self._momentum_divergence(candles, fvg)

        # 11. 缺口附近流动性深度估算（附近 5% 带宽内成交量占比）
        feats["liquidity_around_gap"] = self._liquidity_around_gap(candles, fvg)

        # 12. 历史回补率（无统计时中性 0.5）
        hist = self.config.get("hist_fill_rates", {})
        feats["historical_fill_rate"] = float(
            hist.get(fvg.timeframe, hist.get("*", 0.5))
        )

        # 13. 与 BTC 相关性（无 BTC 数据时中性 0）
        feats["corr_with_btc"] = self._corr_with_btc(candles, fvg)

        # 14. 资金费率 z-score（无历史时中性 0）
        feats["funding_rate_zscore"] = self._funding_zscore(ctx.get("funding_rate"))

        # 15. 订单簿不平衡度（无数据时中性 0）
        ob = ctx.get("order_book_imbalance")
        feats["order_book_imbalance"] = float(ob) if ob is not None else 0.0

        # 16-24. 汇流确认特征 (config.confluence.enabled 且数据可得时计算;
        #        结果惰性缓存于 fvg 上避免重复计算)
        _cr = self._confluence_result(fvg, candles)
        feats["confluence_score"] = self._get_confluence_score(fvg, candles)
        feats["bias_aligned"] = self._is_bias_aligned(fvg, candles)
        feats["liquidity_swept"] = self._is_liquidity_swept(fvg, candles)
        feats["structure_broken"] = self._is_structure_broken(fvg, candles)
        feats["orderflow_positive"] = self._is_orderflow_positive(fvg, candles)
        feats["htf_nested"] = self._is_nested_in_htf(fvg, candles)
        feats["in_premium_zone"] = self._is_in_premium_zone(fvg, candles)
        feats["in_good_time"] = self._is_in_good_time_window(fvg)
        feats["num_conditions_met"] = int(_cr.get("num_conditions_met", 0))
        # 25. 入口质量分数 (excellent=1.0 / good=0.65 / poor=0.25)
        # 与 get_confluence_features 的 entry_quality_score 同口径,
        # 保证 train_fvg_model 的 {**base, **confluence_features} 合并不产生新列
        _eq_map = {"excellent": 1.0, "good": 0.65, "poor": 0.25}
        feats["entry_quality_score"] = float(_eq_map.get(
            _cr.get("entry_quality", "poor"), 0.25))

        return feats

    # ------------------------------------------------------------------
    # 汇流特征辅助 (ConfluenceChecker 惰性调用, 防重复计算)
    # ------------------------------------------------------------------

    def _confluence_result(self, fvg: FVGDetected, candles: List) -> dict:
        """惰性计算汇流确认结果并缓存于 fvg 上 (避免每特征重复计算)。

        单根 K 线/未启用时返回中性结果, 不抛异常。
        """
        res = getattr(fvg, "_confluence_result", None)
        if res is None:
            res = self._compute_confluence(fvg, candles)
            try:
                fvg._confluence_result = res
            except Exception:
                pass
        return res

    def _compute_confluence(self, fvg: FVGDetected, candles: List) -> dict:
        """运行 ConfluenceChecker.check (无 4H 数据时 1H 兼作 HTF 回退)。"""
        if not self.config.get("confluence", {}).get("enabled", True):
            return self._neutral_confluence()
        if not candles:
            return self._neutral_confluence()
        try:
            from confluence import ConfluenceChecker  # noqa: PLC0415 (惰性, 防环)
        except ImportError:
            return self._neutral_confluence()
        ctx = dict(self._last_context or {})
        ctx.setdefault("current_price", float(candles[-1].close))
        try:
            cc = ConfluenceChecker(self.config.get("confluence", {}))
            return cc.check(fvg, candles, candles, ctx)
        except Exception as e:
            logger.debug(f"汇流检查异常，返回中性: {e}")
            return self._neutral_confluence()

    @staticmethod
    def _neutral_confluence() -> dict:
        """汇流结果中性值 (所有条件未满足)。"""
        return {
            "confluence_score": 0.0,
            "conditions_met": [],
            "conditions_failed": [
                "bias_alignment", "liquidity_sweep", "structure_break",
                "orderflow", "htf_nesting", "price_zone", "time_window",
            ],
            "details": {
                "bias_alignment": {"met": False, "score": 0.0, "bias": "neutral"},
                "liquidity_sweep": {"met": False, "score": 0.0, "swept_pool": "none"},
                "structure_break": {"met": False, "score": 0.0, "type": None},
                "orderflow": {"met": False, "score": 0.0, "delta": 0.0},
                "htf_nesting": {"met": False, "score": 0.0, "htf_fvg": {}},
                "price_zone": {"met": False, "score": 0.0, "zone": "equilibrium"},
                "time_window": {"met": False, "score": 0.0, "window": "unknown"},
            },
            "recommendation": "neutral",
            "entry_quality": "poor",
            "num_conditions_met": 0,
        }

    @staticmethod
    def _flag(cr: dict, key: str) -> int:
        """读取汇流结果中某条件的 met 标记 (0/1)。"""
        return int(bool((cr.get("details", {}).get(key) or {}).get("met")))

    def _get_confluence_score(self, fvg: FVGDetected, candles: List) -> float:
        return float(self._confluence_result(fvg, candles).get(
            "confluence_score", 0.0))

    def _is_bias_aligned(self, fvg: FVGDetected, candles: List) -> int:
        """大周期方向与 FVG 方向一致 (0/1)。"""
        return self._flag(self._confluence_result(fvg, candles), "bias_alignment")

    def _is_liquidity_swept(self, fvg: FVGDetected, candles: List) -> int:
        """形成位附近发生顺向流动性猎杀 (0/1)。"""
        return self._flag(self._confluence_result(fvg, candles), "liquidity_sweep")

    def _is_structure_broken(self, fvg: FVGDetected, candles: List) -> int:
        """结构破坏方向与 FVG 一致 (0/1)。"""
        return self._flag(self._confluence_result(fvg, candles), "structure_break")

    def _is_orderflow_positive(self, fvg: FVGDetected, candles: List) -> int:
        """订单流 Delta 与 FVG 方向一致 (0/1)。"""
        return self._flag(self._confluence_result(fvg, candles), "orderflow")

    def _is_nested_in_htf(self, fvg: FVGDetected, candles: List) -> int:
        """FVG 嵌套在更大周期 FVG 缺口内 (0/1)。"""
        return self._flag(self._confluence_result(fvg, candles), "htf_nesting")

    def _is_in_premium_zone(self, fvg: FVGDetected, candles: List) -> int:
        """价格处于溢价区 (VWAP 上方, 0/1)。"""
        return int((self._confluence_result(fvg, candles).get("details", {})
                    .get("price_zone", {}) or {}).get("zone") == "premium")

    def _is_in_good_time_window(self, fvg: FVGDetected) -> int:
        """当前处于可交易时段 (0/1)。读取缓存结果, 未缓存时为中性 0。"""
        res = getattr(fvg, "_confluence_result", None)
        if res is None:
            return 0
        return int(bool((res.get("details", {}).get("time_window") or {})
                        .get("met")))

    # ------------------------------------------------------------------
    # 汇流检测 (detect + 逐 FVG 汇流评分)
    # ------------------------------------------------------------------

    def detect_with_confluence(
        self,
        candles_by_tf: Dict[str, List],
        context: dict,
    ) -> List[tuple]:
        """检测 FVG 并同时返回每个 FVG 的汇流确认结果。

        Args:
            candles_by_tf: {"1H": [...], "4H": [...]}, K 线正序
            context: 市场上下文 (current_price/funding_rate/spread 等)

        Returns:
            List of (FVGDetected, confluence_result)
        """
        fvgs = self.detect(candles_by_tf)
        fvgs = self.filter_by_quality(fvgs, context)

        from confluence import ConfluenceChecker  # noqa: PLC0415 (惰性, 防环)
        cc = ConfluenceChecker(self.config.get("confluence", {}))

        results: List[tuple] = []
        for fvg in fvgs:
            cr = cc.check(
                fvg,
                candles_by_tf.get("1H", []),
                candles_by_tf.get("4H", []),
                context,
            )
            results.append((fvg, cr))
        return results

    # ------------------------------------------------------------------
    # 内部算法
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_abnormal(
        candles: List, idx: int,
        sigma_threshold: float, volume_ratio_threshold: float, lookback: int,
    ):
        """复刻 strategy.detect_abnormal_candle：MAD 鲁棒 z-score + 量比。"""
        if idx < lookback or idx < 1:
            return False, 0.0, 1.0
        returns, volumes = [], []
        for i in range(idx - lookback, idx):
            if i < 1:
                continue
            if candles[i - 1].close <= 0 or candles[i].close <= 0:
                continue
            returns.append(abs(math.log(candles[i].close / candles[i - 1].close)))
            volumes.append(candles[i].volume)
        if len(returns) < 20:
            return False, 0.0, 1.0
        ret_arr = np.asarray(returns, dtype=float)
        median_ret = float(np.median(ret_arr))
        mad = float(np.median(np.abs(ret_arr - median_ret)))
        if mad < 1e-10:
            return False, 0.0, 1.0
        if candles[idx - 1].close <= 0 or candles[idx].close <= 0:
            return False, 0.0, 1.0
        current_ret = abs(math.log(candles[idx].close / candles[idx - 1].close))
        sigma = (current_ret - median_ret) / (mad * 1.4826)
        mean_vol = float(np.mean(volumes))
        volume_ratio = candles[idx].volume / mean_vol if mean_vol > 1e-10 else 1.0
        is_abnormal = (sigma >= sigma_threshold) and (volume_ratio >= volume_ratio_threshold)
        return is_abnormal, sigma, volume_ratio

    @staticmethod
    def _quality_score(width_pct: float, is_abnormal: bool,
                       volume_ratio: float, sigma: float) -> float:
        """规则初始质量分（0-1）。

        分量：
          - 宽度：缺口越大越可信（1%-4% 映射到 0.3-0.8）
          - 异常波动：伴随放量跳空（机构行为）加分
          - 量比：1x-5x 映射加分
          - σ：2-6 映射加分
        """
        q = 0.3
        # 宽度分量（4%+ 封顶 0.8）
        q += max(0.0, min(0.5, (width_pct - 1.0) / 6.0))
        # 异常波动加分
        if is_abnormal:
            q += 0.15
        # 量比加分（5x 封顶 +0.1）
        q += max(0.0, min(0.1, (volume_ratio - 1.0) / 40.0))
        # σ 加分（6 封顶 +0.1）
        q += max(0.0, min(0.1, (sigma - 2.0) / 40.0))
        return float(min(max(q, 0.0), 1.0))

    @staticmethod
    def _sma(candles: List, period: int) -> float:
        if not candles or len(candles) < period:
            return 0.0
        closes = [c.close for c in candles[-period:] if c.close > 0]
        if len(closes) < period:
            return 0.0
        return sum(closes) / period

    @staticmethod
    def _atr(candles: List, period: int = 14) -> float:
        if not candles or len(candles) < period + 1:
            return 0.0
        trs = []
        for i in range(-period, 0):
            c0, c1 = candles[i - 1], candles[i]
            tr = max(c1.high - c1.low,
                     abs(c1.high - c0.close),
                     abs(c1.low - c0.close))
            trs.append(tr)
        return float(np.mean(trs)) if trs else 0.0

    @staticmethod
    def _trend_strength(candles: List, end_idx: int, window: int) -> float:
        """前 window 根收盘价线性回归斜率，归一化为价格比例（%/根）。"""
        if end_idx < window or not candles:
            return 0.0
        seg = [c.close for c in candles[end_idx - window:end_idx] if c.close > 0]
        if len(seg) < 5:
            return 0.0
        x = np.arange(len(seg), dtype=float)
        slope, _ = np.polyfit(x, seg, 1)
        base = float(np.mean(seg))
        return float(slope / base * 100) if base > 0 else 0.0

    @staticmethod
    def _prev_volatility(candles: List, end_idx: int, window: int) -> float:
        if end_idx < window + 1 or not candles:
            return 0.0
        rets = []
        for i in range(end_idx - window + 1, end_idx + 1):
            if candles[i - 1].close > 0 and candles[i].close > 0:
                rets.append(math.log(candles[i].close / candles[i - 1].close))
        return float(np.std(rets)) if len(rets) > 3 else 0.0

    @staticmethod
    def _gap_position(candles: List, fvg: FVGDetected) -> float:
        """缺口在近期 60 根高低区间中的位置（0=底部，1=顶部）。"""
        seg = candles[-60:] if len(candles) >= 60 else candles
        if not seg:
            return 0.5
        hi = max(c.high for c in seg)
        lo = min(c.low for c in seg)
        if hi <= lo:
            return 0.5
        mid = (fvg.gap_high + fvg.gap_low) / 2
        return float(min(max((mid - lo) / (hi - lo), 0.0), 1.0))

    @staticmethod
    def _retracement(fvg: FVGDetected, current_price: Optional[float]) -> float:
        """价格回到缺口内的程度。bullish 用 gap_low 为 0、gap_high 为 1。"""
        if current_price is None or current_price <= 0:
            return 0.5
        gap = fvg.gap_high - fvg.gap_low
        if gap <= 0:
            return 0.5
        if fvg.direction == "bullish":
            frac = (current_price - fvg.gap_low) / gap
        else:
            frac = (fvg.gap_high - current_price) / gap
        return float(min(max(frac, 0.0), 1.0))

    @staticmethod
    def _rsi(candles: List, period: int = 14) -> float:
        if len(candles) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(-period, 0):
            chg = candles[i].close - candles[i - 1].close
            gains.append(max(chg, 0.0))
            losses.append(max(-chg, 0.0))
        avg_g = float(np.mean(gains))
        avg_l = float(np.mean(losses))
        if avg_l == 0:
            return 100.0
        rs = avg_g / avg_l
        return float(100 - 100 / (1 + rs))

    def _momentum_divergence(self, candles: List, fvg: FVGDetected) -> float:
        """动量背离：价格创新高但 RSI 未创新高（顶背离=-1，底背离=+1）。"""
        if len(candles) < 30:
            return 0.0
        lookback = int(self.config.get("divergence_lookback", 14))
        seg = candles[-lookback:]
        prices = [c.close for c in seg]
        rsis = [self._rsi(candles[:len(candles) - lookback + i + 1], 14)
                for i in range(lookback)]
        if len(prices) < 5 or len(rsis) < 5:
            return 0.0
        px_peak = max(prices)
        px_trough = min(prices)
        # 顶部背离：当前价接近区间高点但 RSI 明显低于区间内 RSI 峰值
        rsi_peak_prev = max(rsis[:-1])
        rsi_low_prev = min(rsis[:-1])
        if prices[-1] >= px_peak * 0.999 and rsis[-1] < rsi_peak_prev - 5:
            return -1.0 if fvg.direction == "bullish" else 1.0
        if prices[-1] <= px_trough * 1.001 and rsis[-1] > rsi_low_prev + 5:
            return 1.0 if fvg.direction == "bullish" else -1.0
        return 0.0

    @staticmethod
    def _liquidity_around_gap(candles: List, fvg: FVGDetected) -> float:
        """缺口附近（外扩 5% 带宽）成交量占总成交量比例（估算流动性深度）。"""
        seg = candles[-60:] if len(candles) >= 60 else candles
        if not seg:
            return 0.0
        span = fvg.gap_high - fvg.gap_low
        band = span * 0.05
        lo_b = fvg.gap_low - band
        hi_b = fvg.gap_high + band
        tot_v = sum(c.volume for c in seg) or 0.0
        if tot_v <= 0:
            return 0.0
        near_v = sum(c.volume for c in seg
                     if lo_b <= (c.high + c.low) / 2 <= hi_b)
        return float(near_v / tot_v)

    @staticmethod
    def _corr_with_btc(candles: List, fvg: FVGDetected) -> float:
        """与 BTC 的相关性。btc 数据通过 config['btc_candles'] 注入。"""
        btc = getattr(fvg, "_btc_candles", None) or None
        if btc is None or len(btc) < 10 or len(candles) < 10:
            return 0.0
        a = np.array([c.close for c in candles[-min(len(candles), len(btc)):]])
        b = np.array([c.close for c in btc[-min(len(candles), len(btc)):]])
        if len(a) < 10 or len(b) < 10:
            return 0.0
        ra = np.diff(np.log(a))
        rb = np.diff(np.log(b))
        if len(ra) < 5 or np.std(ra) == 0 or np.std(rb) == 0:
            return 0.0
        corr = float(np.corrcoef(ra, rb)[0, 1])
        return corr if abs(corr) < 5 else 0.0

    @staticmethod
    def _funding_zscore(funding_rate: Optional[float]) -> float:
        """资金费率 z-score（默认以 0 为均值、0.0003 为波动基准）。"""
        if funding_rate is None:
            return 0.0
        return float(funding_rate / 0.0003)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def detect_fvgs(
    candles_by_tf: Dict[str, List],
    config: Optional[dict] = None,
    context: Optional[dict] = None,
) -> List[FVGDetected]:
    """便捷入口：detect + filter_by_quality 一步完成。"""
    detector = FVGDetector(config or {})
    fvgs = detector.detect(candles_by_tf)
    if context:
        fvgs = detector.filter_by_quality(fvgs, context)
    return fvgs


def from_legacy_fvg(legacy_fvg) -> FVGDetected:
    """将 strategy.FVG（旧结构）适配为 FVGDetected。

    用于 ML 集成：agent 主循环中的 Signal.fvg 是 strategy.FVG（含
    top/bottom/fvg_index/impulse_candle 等字段），compute_features 需要
    FVGDetected 结构。鸭子类型访问属性，不 import strategy 避免循环依赖。

    Args:
        legacy_fvg: strategy.FVG 或任何含 top/bottom/direction/timeframe/
                    width_pct/fvg_index/impulse_candle/is_abnormal/sigma/
                    volume_ratio/candle_ts 属性的对象

    Returns:
        FVGDetected
    """
    direction = getattr(legacy_fvg, "direction", "long")
    impulse = getattr(legacy_fvg, "impulse_candle", None)
    fvg_index = getattr(legacy_fvg, "fvg_index", -1)
    return FVGDetected(
        inst_id="",
        timeframe=getattr(legacy_fvg, "timeframe", ""),
        start_idx=max(0, fvg_index - 1),
        end_idx=fvg_index + 1,
        gap_high=float(getattr(legacy_fvg, "top", 0.0)),
        gap_low=float(getattr(legacy_fvg, "bottom", 0.0)),
        width_pct=float(getattr(legacy_fvg, "width_pct", 0.0)),
        volume_at_formation=float(getattr(impulse, "volume", 0.0) if impulse else 0.0),
        formation_price=float(getattr(impulse, "close", 0.0) if impulse else 0.0),
        direction="bullish" if direction == "long" else "bearish",
        is_abnormal=bool(getattr(legacy_fvg, "is_abnormal", False)),
        quality_score=0.0,
        sigma=float(getattr(legacy_fvg, "sigma", 0.0)),
        volume_ratio=float(getattr(legacy_fvg, "volume_ratio", 1.0)),
        formation_ts=int(getattr(legacy_fvg, "candle_ts", 0)),
    )
