# -*- coding: utf-8 -*-
"""
confluence.py — 多因素汇流确认引擎 (HunHeng_OS_V1.0)。

将 ICT / Smart Money 的"汇流确认"逻辑抽象为可独立使用的组件，
供 fvg_detector / agent 在信号执行前做多因素确认（层级决策）：

  1. HTFBiasDetector     — 大周期方向偏执 (HTF Bias): EMA 排列 + VWAP 位置 + 结构高低点
  2. LiquidityPoolDetector — 流动性池识别: BSL/SSL + 流动性猎杀 (Sweep)
  3. StructureBreaker    — 市场结构破坏 (ChoCH): 高低点结构 + 突破确认
  4. OrderFlowFilter     — 订单流过滤 (模拟): 吸收 (Absorption) + 买卖力量差 (Delta)
  5. ConfluenceChecker   — 汇流确认总控: 组合上述检测器, 对 FVG 信号做 7 条件汇流评分

设计约束:
  - 零外部依赖 (仅 numpy; K 线采用鸭子类型, 接受任何含
    open/high/low/close/volume/timestamp 属性的对象, 与 fvg_detector 一致)
  - 不 import strategy / agent (避免循环依赖)
  - 所有阈值从 config 读取 (配置驱动, 不硬编码)
  - 数据不足时返回中性结果 + reasons 说明, 不抛异常 (保守放行)

数据流:
  HTF candles ─→ HTFBiasDetector.detect ─→ bias/confidence
  最近 K 线  ─→ LiquidityPoolDetector.identify_pools ─→ BSL/SSL 距离 + 猎杀标记
  最近 K 线  ─→ StructureBreaker.detect ─→ regime/is_broken
  最近 K 线  ─→ OrderFlowFilter ─→ absorption / delta
  FVG + 1H/4H K 线 + context ─→ ConfluenceChecker.check ─→ 7 条件汇流评分
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA (指数移动平均), 与 pandas ewm(adjust=False) 同构。"""
    period = max(1, int(period))
    a = 2.0 / (period + 1.0)
    out = np.empty(len(values), dtype=float)
    if len(values) == 0:
        return out
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = a * float(values[i]) + (1.0 - a) * out[i - 1]
    return out


def _find_swings(
    candles: List[Any], lookback: int,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """分形 Swing High / Swing Low。

    一根 K 线是 Swing High: 其 high 严格大于左右各 lookback 根 K 线的高;
    Swing Low 同理 (low 严格小于左右 lookback 根的低)。

    Returns:
        (swing_highs, swing_lows), 每项为 [(index, price), ...] 按时间正序
    """
    n = len(candles)
    lookback = max(1, int(lookback))
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []
    if n < 2 * lookback + 1:
        return highs, lows
    for i in range(lookback, n - lookback):
        h = float(candles[i].high)
        l = float(candles[i].low)
        is_high = all(float(candles[i - k].high) < h and float(candles[i + k].high) < h
                      for k in range(1, lookback + 1))
        is_low = all(float(candles[i - k].low) > l and float(candles[i + k].low) > l
                     for k in range(1, lookback + 1))
        if is_high:
            highs.append((i, h))
        if is_low:
            lows.append((i, l))
    return highs, lows


def _safe_close(candles: List[Any]) -> float:
    return float(candles[-1].close) if candles else 0.0


def _swing_slope(vals: List[float]) -> float:
    """Swing 价格序列线性回归斜率, 归一化为每步相对变化率。

    正值=序列整体抬升, 负值=下降, 接近 0=区间震荡。
    相比"最后两根比较", 对震荡市更稳健 (不易被末端噪声误导)。
    """
    if len(vals) < 3:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    slope = float(np.polyfit(x, vals, 1)[0])
    base = float(np.mean(vals))
    return slope / base if base > 0 else 0.0


# ---------------------------------------------------------------------------
# 1.1 大周期方向偏执
# ---------------------------------------------------------------------------

class HTFBiasDetector:
    """检测大周期 (4H/1D) 的市场方向偏执 (HTF Bias)。

    方法:
      - EMA 排列: 价格与 EMA20/50/200 的排列方向
      - VWAP 位置: 价格相对锚定 VWAP 的偏离百分比
      - 结构高低点: 近期 Swing High/Low 的序列方向 (HH/HL / LH/LL)

    config 键:
      - ema_periods: List[int]  默认 [20, 50, 200]
      - vwap_period: int        默认 20 (锚定窗口根数)
      - lookback_bars: int      默认 100 (检测回溯窗口)
      - swing_lookback: int     默认 3 (结构方向用的分形左右根数)
      - vwap_near_pct: float    默认 0.1 (%); |偏离| 低于此视为 "near"
      - ema_alignment_min_pct: float 默认 0.1 (%); 价格与 EMA20 的偏离低于
        此值不参与排列判定 (防震荡市随机翻转)
      - structure_slope_threshold: float 默认 0.0005 (Swing 序列斜率阈值,
        低于此视为区间震荡, 防止末端噪声误判方向)
      - min_recent_bars: int    默认 30 (数据不足阈值)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.ema_periods: List[int] = [
            int(p) for p in cfg.get("ema_periods", [20, 50, 200])
        ]
        self.vwap_period: int = int(cfg.get("vwap_period", 20))
        self.lookback_bars: int = int(cfg.get("lookback_bars", 100))
        self.swing_lookback: int = int(cfg.get("swing_lookback", 3))
        self.vwap_near_pct: float = float(cfg.get("vwap_near_pct", 0.1))
        self.ema_alignment_min_pct: float = float(
            cfg.get("ema_alignment_min_pct", 0.1))
        self.structure_slope_threshold: float = float(
            cfg.get("structure_slope_threshold", 0.0005))
        self.min_recent_bars: int = int(cfg.get("min_recent_bars", 30))

    # ------------------------------------------------------------------
    def compute_vwap(self, candles: List[Any]) -> float:
        """锚定 VWAP (最近 vwap_period 根的量加权典型价)。

        供汇流确认的"溢价/折价区"判断复用, 与 detect() 内部口径一致。

        Args:
            candles: 正序 K 线

        Returns:
            VWAP 值; 无成交量/数据不足时返回 0.0
        """
        if not candles:
            return 0.0
        seg = candles[-self.lookback_bars:]
        start = max(0, len(seg) - self.vwap_period)
        tp_sum = 0.0
        vol_sum = 0.0
        for c in seg[start:]:
            vol = float(c.volume)
            if vol > 0:
                tp_sum += (float(c.high) + float(c.low) + float(c.close)) / 3.0 * vol
                vol_sum += vol
        return tp_sum / vol_sum if vol_sum > 0 else 0.0

    # ------------------------------------------------------------------
    def detect(self, candles: List[Any]) -> Dict[str, Any]:
        """检测当前大周期方向。

        Args:
            candles: 大周期 K 线 (正序, 旧→新), 鸭子类型

        Returns:
            {
                "bias": "bullish" | "bearish" | "neutral",
                "ema_alignment": "bullish" | "bearish" | "mixed",
                "vwap_position": float,   # 价格相对 VWAP 偏离百分比
                "structure_direction": "up" | "down" | "sideways",
                "confidence": float,      # 0-1
                "reasons": List[str],
                "breakdown": {
                    "ema_20_50_cross": "bullish" | "bearish" | None,
                    "price_vs_vwap": "above" | "below" | "near",
                    "swing_highs_lows": "HH_HL" | "LH_LL" | "sideways",
                }
            }
        """
        neutral = self._neutral()
        if not candles or len(candles) < self.min_recent_bars:
            return neutral
        seg = candles[-self.lookback_bars:]
        closes = np.array([float(c.close) for c in seg], dtype=float)
        if np.any(closes <= 0):
            return neutral

        price = _safe_close(seg)
        reasons: List[str] = []
        breakdown: Dict[str, Any] = {}

        # ---- EMA 排列 ----
        ema_alignment = "mixed"
        emas: Dict[int, float] = {}
        valid_periods = [p for p in self.ema_periods if p <= len(closes)]
        for p in valid_periods:
            emas[p] = float(_ema(closes, p)[-1])
        if len(valid_periods) >= 2 and all(p < len(closes) for p in valid_periods):
            sorted_emas = [emas[p] for p in sorted(valid_periods)]
            dev_pct = abs(price - sorted_emas[0]) / sorted_emas[0] * 100.0 \
                if sorted_emas[0] > 0 else 0.0
            if (dev_pct >= self.ema_alignment_min_pct
                    and price > sorted_emas[0] and all(
                        sorted_emas[i] > sorted_emas[i + 1]
                        for i in range(len(sorted_emas) - 1))):
                ema_alignment = "bullish"
                reasons.append("EMA 多头排列 (价格 > EMA%s)" % ",".join(
                    str(p) for p in sorted(valid_periods)))
            elif (dev_pct >= self.ema_alignment_min_pct
                    and price < sorted_emas[0] and all(
                        sorted_emas[i] < sorted_emas[i + 1]
                        for i in range(len(sorted_emas) - 1))):
                ema_alignment = "bearish"
                reasons.append("EMA 空头排列")
            else:
                ema_alignment = "mixed"
                reasons.append(f"EMA 排列交错/偏离不足 ({dev_pct:.2f}% < "
                               f"{self.ema_alignment_min_pct}%)")
        else:
            reasons.append("EMA 周期超数据长度，排列判定跳过")

        # ---- VWAP 位置 (锚定最近 vwap_period 根) ----
        vwap = self.compute_vwap(seg)
        if vwap > 0:
            vwap_position = (price - vwap) / vwap * 100.0
            if abs(vwap_position) <= self.vwap_near_pct:
                price_vs_vwap = "near"
            elif vwap_position > 0:
                price_vs_vwap = "above"
            else:
                price_vs_vwap = "below"
            reasons.append(f"VWAP 偏离 {vwap_position:+.2f}% ({price_vs_vwap})")
        else:
            vwap_position, price_vs_vwap = 0.0, "near"
            reasons.append("成交量缺失，VWAP 判定跳过")
        breakdown["price_vs_vwap"] = price_vs_vwap

        # ---- EMA20/50 交叉 ----
        ema_20_50_cross: Optional[str] = None
        if 20 in emas and 50 in emas and len(closes) > 50:
            e20 = _ema(closes, 20)
            e50 = _ema(closes, 50)
            for i in range(len(closes) - 1, max(0, len(closes) - 5), -1):
                if e20[i - 1] <= e50[i - 1] and e20[i] > e50[i]:
                    ema_20_50_cross = "bullish"
                    break
                if e20[i - 1] >= e50[i - 1] and e20[i] < e50[i]:
                    ema_20_50_cross = "bearish"
                    break
        breakdown["ema_20_50_cross"] = ema_20_50_cross

        # ---- 结构方向 (Swing High/Low 序列斜率) ----
        highs, lows = _find_swings(seg, self.swing_lookback)
        structure_direction = "sideways"
        swing_label = "sideways"
        if len(highs) >= 3 and len(lows) >= 3:
            h_slope = _swing_slope([p for _, p in highs[-4:]])
            l_slope = _swing_slope([p for _, p in lows[-4:]])
            th = self.structure_slope_threshold
            if l_slope > th and h_slope > -th * 0.5:   # 低点抬升为主
                structure_direction, swing_label = "up", "HH_HL"
            elif l_slope < -th and h_slope < th * 0.5:  # 高点压低为主
                structure_direction, swing_label = "down", "LH_LL"
            else:
                structure_direction, swing_label = "sideways", "sideways"
            reasons.append(f"结构 {swing_label}")
        else:
            reasons.append("Swing 数量不足，结构判定侧向")
        breakdown["swing_highs_lows"] = swing_label

        # ---- 综合方向 + 置信度 ----
        # 评分权重: EMA 排列 0.4 / VWAP 位置 0.2 / 结构 0.4
        score = 0.0
        if ema_alignment == "bullish":
            score += 0.4
        elif ema_alignment == "bearish":
            score -= 0.4
        if price_vs_vwap == "above":
            score += 0.2
        elif price_vs_vwap == "below":
            score -= 0.2
        if structure_direction == "up":
            score += 0.4
        elif structure_direction == "down":
            score -= 0.4

        if score > 0.05:
            bias = "bullish"
        elif score < -0.05:
            bias = "bearish"
        else:
            bias = "neutral"
        confidence = float(min(max(abs(score), 0.0), 1.0))
        if bias == "neutral":
            reasons.append("多因素相互抵消 → 中性")

        return {
            "bias": bias,
            "ema_alignment": ema_alignment,
            "vwap_position": round(vwap_position, 4),
            "structure_direction": structure_direction,
            "confidence": round(confidence, 4),
            "reasons": reasons,
            "breakdown": breakdown,
        }

    def _neutral(self) -> Dict[str, Any]:
        return {
            "bias": "neutral",
            "ema_alignment": "mixed",
            "vwap_position": 0.0,
            "structure_direction": "sideways",
            "confidence": 0.0,
            "reasons": ["数据不足 (K线数 < min_recent_bars)"],
            "breakdown": {
                "ema_20_50_cross": None,
                "price_vs_vwap": "near",
                "swing_highs_lows": "sideways",
            },
        }


# ---------------------------------------------------------------------------
# 1.2 流动性池检测器
# ---------------------------------------------------------------------------

class LiquidityPoolDetector:
    """识别关键的流动性池 (买方/卖方流动性) 与流动性猎杀。

    概念:
      - BSL (Buy-side Liquidity): 买方流动性, 位于近期 Swing Low 下方
      - SSL (Sell-side Liquidity): 卖方流动性, 位于近期 Swing High 上方
      - Sweep: 价格穿透流动性池 (插针) 后快速反转

    config 键:
      - swing_lookback: int            默认 5 (分形左右 K 线数)
      - liquidity_zone_padding: float  默认 0.001 (池区间缓冲比例)
      - max_pools_per_side: int        默认 3 (每侧返回的池数量)
      - lookback_bars: int             默认 200 (池识别窗口)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.swing_lookback: int = int(cfg.get("swing_lookback", 5))
        self.padding: float = float(cfg.get("liquidity_zone_padding", 0.001))
        self.max_pools_per_side: int = int(cfg.get("max_pools_per_side", 3))
        self.lookback_bars: int = int(cfg.get("lookback_bars", 200))

    # ------------------------------------------------------------------
    def find_swing_points(self, candles: List[Any]) -> Dict[str, List[float]]:
        """找出所有 Swing High / Swing Low 价格。

        Args:
            candles: 正序 K 线

        Returns:
            {"swing_highs": [p, ...], "swing_lows": [p, ...]}
        """
        seg = candles[-self.lookback_bars:]
        highs, lows = _find_swings(seg, self.swing_lookback)
        return {
            "swing_highs": [round(p, 8) for _, p in highs],
            "swing_lows": [round(p, 8) for _, p in lows],
        }

    # ------------------------------------------------------------------
    def identify_pools(self, candles: List[Any]) -> Dict[str, Any]:
        """识别当前最近的买方/卖方流动性池。

        Args:
            candles: 正序 K 线

        Returns:
            {
                "nearest_bsl": float | None,     # 最近买方流动性价位 (下方)
                "nearest_ssl": float | None,     # 最近卖方流动性价位 (上方)
                "bsl_distance_pct": float | None,
                "ssl_distance_pct": float | None,
                "pools": [{"type": "bsl"|"ssl", "price": float,
                           "is_swept": bool}, ...]
            }
        """
        empty: Dict[str, Any] = {
            "nearest_bsl": None, "nearest_ssl": None,
            "bsl_distance_pct": None, "ssl_distance_pct": None,
            "pools": [],
        }
        if not candles:
            return empty
        seg = candles[-self.lookback_bars:]
        price = _safe_close(seg)
        if price <= 0:
            return empty
        highs, lows = _find_swings(seg, self.swing_lookback)
        if not highs and not lows:
            return empty

        # 候选池: BSL=低于当前价的 Swing Low, SSL=高于当前价的 Swing High
        bsl_pools = [(i, p) for i, p in lows if p < price]
        ssl_pools = [(i, p) for i, p in highs if p > price]

        def _swept(pool_idx: int, level: float, below: bool) -> bool:
            """该池是否已被价格穿越。below=True → 向下穿越 (BSL)。"""
            if pool_idx >= len(seg):
                return False
            if below:
                return any(float(c.low) <= level * (1.0 - self.padding)
                           for c in seg[pool_idx:])
            return any(float(c.high) >= level * (1.0 + self.padding)
                       for c in seg[pool_idx:])

        pools: List[Dict[str, Any]] = []
        nearest_bsl: Optional[float] = None
        nearest_ssl: Optional[float] = None
        bsl_dist = ssl_dist = None

        if bsl_pools:
            _, nearest_bsl = max(bsl_pools, key=lambda x: x[1])
            bsl_dist = abs(price - nearest_bsl) / price * 100.0
            pools.extend({
                "type": "bsl", "price": round(p, 8),
                "is_swept": _swept(i, p, below=True),
            } for i, p in sorted(bsl_pools,
                                 key=lambda x: abs(price - x[1]))[
                                     :self.max_pools_per_side])
        if ssl_pools:
            _, nearest_ssl = min(ssl_pools, key=lambda x: x[1])
            ssl_dist = abs(nearest_ssl - price) / price * 100.0
            pools.extend({
                "type": "ssl", "price": round(p, 8),
                "is_swept": _swept(i, p, below=False),
            } for i, p in sorted(ssl_pools,
                                 key=lambda x: abs(x[1] - price))[
                                     :self.max_pools_per_side])

        pools.sort(key=lambda x: abs(x["price"] - price))
        return {
            "nearest_bsl": round(nearest_bsl, 8) if nearest_bsl else None,
            "nearest_ssl": round(nearest_ssl, 8) if nearest_ssl else None,
            "bsl_distance_pct": round(bsl_dist, 4) if bsl_dist is not None else None,
            "ssl_distance_pct": round(ssl_dist, 4) if ssl_dist is not None else None,
            "pools": pools,
        }

    # ------------------------------------------------------------------
    def detect_sweep(self, candles: List[Any], idx: int) -> Dict[str, Any]:
        """检测某根 K 线是否发生流动性猎杀, 并返回被猎杀的池。

        条件 (做空猎杀 / BSL 为例): 该 K 线 low 穿透最近未触及的 Swing Low,
        且该 K 线 (或下一根) 收盘重新收回到池价位上方 → 快速反转确认。

        Args:
            candles: 正序 K 线
            idx: 待判断的 K 线索引 (须在 1..len-2 范围, 需下一根 K 线)

        Returns:
            {"is_sweep": bool, "pool": "bsl"|"ssl"|None, "swept_level": float|None}
        """
        base: Dict[str, Any] = {"is_sweep": False, "pool": None, "swept_level": None}
        n = len(candles)
        if idx <= 0 or idx >= n - 1:
            return base
        seg = candles[:idx + 1]
        if len(seg) < 2 * self.swing_lookback + 1:
            return base
        highs, lows = _find_swings(seg[:-1], self.swing_lookback)
        if not highs and not lows:
            return base

        c0, c1 = candles[idx], candles[idx + 1]
        lo, hi = float(c0.low), float(c0.high)

        # 向下猎杀 BSL: 穿透最近 Swing Low 后收回
        if lows:
            _, last_swing_low = max(lows, key=lambda x: x[0])
            level = last_swing_low * (1.0 - self.padding)
            if (lo <= level and last_swing_low < float(c0.open)
                    and float(c0.close) > last_swing_low):
                return {"is_sweep": True, "pool": "bsl", "swept_level": last_swing_low}
            if lo <= level and float(c1.close) > last_swing_low \
                    and last_swing_low < float(c0.open):
                return {"is_sweep": True, "pool": "bsl", "swept_level": last_swing_low}

        # 向上猎杀 SSL: 穿透最近 Swing High 后收回
        if highs:
            _, last_swing_high = max(highs, key=lambda x: x[0])
            level = last_swing_high * (1.0 + self.padding)
            if (hi >= level and last_swing_high > float(c0.open)
                    and float(c0.close) < last_swing_high):
                return {"is_sweep": True, "pool": "ssl", "swept_level": last_swing_high}
            if hi >= level and float(c1.close) < last_swing_high \
                    and last_swing_high > float(c0.open):
                return {"is_sweep": True, "pool": "ssl", "swept_level": last_swing_high}
        return base

    def is_liquidity_sweep(self, candles: List[Any], idx: int) -> bool:
        """判断某根 K 线是否发生流动性猎杀 (兼容布尔接口)。"""
        return bool(self.detect_sweep(candles, idx)["is_sweep"])


# ---------------------------------------------------------------------------
# 1.3 市场结构破坏检测器
# ---------------------------------------------------------------------------

class StructureBreaker:
    """检测市场结构破坏 (ChoCH - Change of Character)。

    概念:
      - 破坏多头结构: 价格收盘跌破前一个 Higher Low
      - 破坏空头结构: 价格收盘突破前一个 Lower High
      - 结构破坏是趋势反转的重要确认信号

    config 键:
      - swing_lookback: int          默认 5 (分形左右 K 线数)
      - min_break_distance: float    默认 0.002 (最小突破距离, 百分比)
      - structure_slope_threshold: float 默认 0.0005 (Swing 序列斜率阈值,
        低于此视为区间震荡, regime=neutral)
      - lookback_bars: int           默认 200
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.swing_lookback: int = int(cfg.get("swing_lookback", 5))
        self.min_break_distance: float = float(cfg.get("min_break_distance", 0.002))
        self.structure_slope_threshold: float = float(
            cfg.get("structure_slope_threshold", 0.0005))
        self.lookback_bars: int = int(cfg.get("lookback_bars", 200))

    # ------------------------------------------------------------------
    def detect(self, candles: List[Any]) -> Dict[str, Any]:
        """检测当前市场的结构状态。

        Args:
            candles: 正序 K 线

        Returns:
            {
                "regime": "bullish" | "bearish" | "neutral",
                "is_broken": bool,
                "last_break_type": "bullish_break" | "bearish_break" | None,
                "break_bar_index": int,
                "structure_levels": {
                    "last_higher_high": float | None,
                    "last_higher_low": float | None,
                    "last_lower_high": float | None,
                    "last_lower_low": float | None,
                }
            }
        """
        neutral = self._neutral()
        if not candles:
            return neutral
        seg = candles[-self.lookback_bars:]
        n = len(seg)
        if n < 2 * self.swing_lookback + 3:
            return neutral
        price = _safe_close(seg)
        if price <= 0:
            return neutral
        highs, lows = _find_swings(seg, self.swing_lookback)
        if not highs and not lows:
            return neutral

        h_vals = [p for _, p in highs]
        l_vals = [p for _, p in lows]
        th = self.structure_slope_threshold

        # ---- 结构方向 (Swing 序列斜率) ----
        struct_dir = "sideways"
        if len(h_vals) >= 3 and len(l_vals) >= 3:
            h_slope = _swing_slope(h_vals[-4:])
            l_slope = _swing_slope(l_vals[-4:])
            if l_slope > th and h_slope > -th * 0.5:
                struct_dir = "up"
            elif l_slope < -th and h_slope < th * 0.5:
                struct_dir = "down"

        # ---- 结构级别 (最近模式, 全局 min/max 会拉到陈旧位 → 禁止) ----
        def _recent(seq: List[float], higher: bool) -> Optional[float]:
            """最近一个满足模式 (higher=比前一个高) 的级别; 无则用最新。"""
            for i in range(len(seq) - 1, 0, -1):
                if (seq[i] > seq[i - 1]) if higher else (seq[i] < seq[i - 1]):
                    return seq[i]
            return seq[-1] if seq else None

        last_higher_high = _recent(h_vals, higher=True)
        last_higher_low = _recent(l_vals, higher=True)
        last_lower_high = _recent(h_vals, higher=False)
        last_lower_low = _recent(l_vals, higher=False)

        # ---- 最近已完成 K 线的结构破坏 ----
        # 关键: break 只在与当前结构同向的模式下检查, 防止上升趋势里
        # 价格恒在陈旧 Lower High 上方而误报 bullish_break。
        is_broken = False
        last_break_type: Optional[str] = None
        break_bar_index = -1
        bar = seg[-2]  # 最近一根已收盘 K 线
        min_dist = self.min_break_distance * 100.0

        # 多头结构被破坏: 收盘跌破最近的 Higher Low (up/sideways 才成立)
        if struct_dir in ("up", "sideways") and last_higher_low:
            if (last_higher_low - float(bar.close)) / last_higher_low * 100.0 \
                    >= min_dist:
                is_broken, last_break_type = True, "bearish_break"
                break_bar_index = n - 2
        # 空头结构被破坏: 收盘突破最近的 Lower High (down/sideways 才成立)
        if (not is_broken) and struct_dir in ("down", "sideways") \
                and last_lower_high:
            if (float(bar.close) - last_lower_high) / last_lower_high * 100.0 \
                    >= min_dist:
                is_broken, last_break_type = True, "bullish_break"
                break_bar_index = n - 2

        # ---- Regime ----
        regime = "neutral"
        if struct_dir == "up":
            regime = "bullish"
        elif struct_dir == "down":
            regime = "bearish"
        if last_break_type == "bearish_break":
            regime = "bearish"   # 多头结构被破坏 → 转空
        elif last_break_type == "bullish_break":
            regime = "bullish"   # 空头结构被破坏 → 转多

        return {
            "regime": regime,
            "is_broken": is_broken,
            "last_break_type": last_break_type,
            "break_bar_index": break_bar_index,
            "structure_levels": {
                "last_higher_high": round(last_higher_high, 8)
                if last_higher_high else None,
                "last_higher_low": round(last_higher_low, 8)
                if last_higher_low else None,
                "last_lower_high": round(last_lower_high, 8)
                if last_lower_high else None,
                "last_lower_low": round(last_lower_low, 8)
                if last_lower_low else None,
            },
        }

    def _neutral(self) -> Dict[str, Any]:
        return {
            "regime": "neutral",
            "is_broken": False,
            "last_break_type": None,
            "break_bar_index": -1,
            "structure_levels": {
                "last_higher_high": None, "last_higher_low": None,
                "last_lower_high": None, "last_lower_low": None,
            },
        }


# ---------------------------------------------------------------------------
# 1.4 订单流过滤器 (模拟)
# ---------------------------------------------------------------------------

class OrderFlowFilter:
    """模拟订单流数据的过滤逻辑。

    无真实订单簿数据时, 用成交量 + 价格行为模拟:
      - Absorption: 关键位附近放量长影线 (价格停滞 + 承接/抛压吸收)
      - Delta Divergence: 买卖力量差 (量加权 body 占比), 范围 [-1, 1]

    config 键:
      - volume_threshold: float     默认 1.5 (成交量放大倍数)
      - absorption_threshold: float 默认 0.3 (影线占比下限)
      - body_threshold: float       默认 0.5 (实体占比上限, 吸收=小实体)
      - volume_lookback: int        默认 20 (均量窗口)
      - delta_window: int           默认 50 (Delta 聚合窗口)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.volume_threshold: float = float(cfg.get("volume_threshold", 1.5))
        self.absorption_threshold: float = float(
            cfg.get("absorption_threshold", 0.3))
        self.body_threshold: float = float(cfg.get("body_threshold", 0.5))
        self.volume_lookback: int = int(cfg.get("volume_lookback", 20))
        self.delta_window: int = int(cfg.get("delta_window", 50))

    # ------------------------------------------------------------------
    def detect_absorption(self, candles: List[Any], idx: int) -> Dict[str, Any]:
        """检测关键位附近的"吸收"行为。

        特征: 放量 + 长影线 + 小实体 (价格在关键位停滞, 一方吸收对手盘)。

        Args:
            candles: 正序 K 线
            idx: 待检测的 K 线索引

        Returns:
            {
                "is_absorption": bool,
                "volume_ratio": float,   # 当前量 / 前 volume_lookback 均量
                "wick_ratio": float,     # 主影线长度 / 全幅
                "body_ratio": float,     # 实体长度 / 全幅
                "side": "buy" | "sell" | None,  # 吸收方向 (承接/抛压)
                "reason": str,
            }
        """
        result: Dict[str, Any] = {
            "is_absorption": False, "volume_ratio": 0.0,
            "wick_ratio": 0.0, "body_ratio": 0.0,
            "side": None, "reason": "数据不足",
        }
        n = len(candles)
        if idx <= 0 or idx >= n:
            return result
        c = candles[idx]
        rng = float(c.high) - float(c.low)
        if rng <= 1e-12:
            result["reason"] = "K线振幅为零"
            return result

        # 均量
        win = candles[max(0, idx - self.volume_lookback):idx]
        vols = [float(x.volume) for x in win if float(x.volume) > 0]
        mean_vol = float(np.mean(vols)) if vols else 0.0
        volume_ratio = (float(c.volume) / mean_vol
                        if mean_vol > 1e-12 else 1.0)

        upper_wick = float(c.high) - max(float(c.open), float(c.close))
        lower_wick = min(float(c.open), float(c.close)) - float(c.low)
        wick_ratio = max(upper_wick, lower_wick) / rng
        body_ratio = abs(float(c.close) - float(c.open)) / rng

        is_abs = (volume_ratio >= self.volume_threshold
                  and wick_ratio >= self.absorption_threshold
                  and body_ratio <= self.body_threshold)
        side: Optional[str] = None
        if is_abs:
            side = "sell" if upper_wick >= lower_wick else "buy"
            result.update({
                "is_absorption": True,
                "side": side,
                "reason": f"放量({volume_ratio:.2f}x) 长{side}影线吸收"
                          f"(wick={wick_ratio:.2f}, body={body_ratio:.2f})",
            })
        else:
            why = []
            if volume_ratio < self.volume_threshold:
                why.append(f"量比{volume_ratio:.2f}<{self.volume_threshold}")
            if wick_ratio < self.absorption_threshold:
                why.append(f"影线{wick_ratio:.2f}<{self.absorption_threshold}")
            if body_ratio > self.body_threshold:
                why.append(f"实体{body_ratio:.2f}>{self.body_threshold}")
            result["reason"] = "非吸收: " + ", ".join(why)
        result.update({
            "volume_ratio": round(volume_ratio, 3),
            "wick_ratio": round(wick_ratio, 3),
            "body_ratio": round(body_ratio, 3),
        })
        return result

    # ------------------------------------------------------------------
    def detect_delta_divergence(self, candles: List[Any]) -> float:
        """估算买卖力量差 (Delta), 返回 [-1, 1]。

        单根 K 线 Delta ≈ sign(close-open) × |close-open|/range × volume,
        聚合最近 delta_window 根后除以总成交量归一化。
        负值=卖压重, 正值=买压强。

        Args:
            candles: 正序 K 线

        Returns:
            float in [-1, 1]; 无数据/无量时返回 0.0
        """
        seg = candles[-self.delta_window:]
        if not seg:
            return 0.0
        num = 0.0
        den = 0.0
        for c in seg:
            rng = float(c.high) - float(c.low)
            vol = float(c.volume)
            if vol <= 0:
                continue
            body = float(c.close) - float(c.open)
            if rng <= 1e-12:
                d = 1.0 if body > 0 else (-1.0 if body < 0 else 0.0)
            else:
                d = body / rng
            num += d * vol
            den += vol
        if den <= 1e-12:
            return 0.0
        return round(float(max(-1.0, min(1.0, num / den))), 4)


# ---------------------------------------------------------------------------
# 1.5 汇流确认总控
# ---------------------------------------------------------------------------

class ConfluenceChecker:
    """对 FVG 信号进行多因素汇流确认 (汇流确认总控)。

    汇流条件 (7 项, 权重默认总和为 1; time_window 默认权重 0 仅作软过滤):
      1. bias_alignment   — 大周期方向一致 (HTF Bias Alignment)
      2. liquidity_sweep  — 流动性猎杀确认 (Liquidity Sweep)
      3. structure_break  — 结构破坏确认 (Structure Break / ChoCH)
      4. orderflow        — 订单流确认 (Orderflow, 模拟 Delta)
      5. htf_nesting      — 多时间框架嵌套 (HTF FVG Nesting)
      6. price_zone       — 价格位置 (溢价/折价区, 基于 VWAP)
      7. time_window      — 时间窗口 (UTC 低流动性时段 16-22 软过滤)

    config 键:
      - htf_bias / liquidity / structure / orderflow: 各子检测器配置
      - fvg_detector: HTF 嵌套检测用的 FVGDetector 配置
      - weights: dict, 各条件权重 (未列出条件用默认值)
      - zone_tolerance_pct: float 默认 0.1 (均衡区容差 %, 兼容旧 schema)
      - price_zones: {"premium_threshold": 0.02, "discount_threshold": 0.02}
      - time_window: {"low_liquidity_start": 16, "low_liquidity_end": 22} (UTC, 旧 schema)
      - time_windows: {"enabled": bool, "preferred_hours": [...], "avoid_hours": [...]}
        (新 schema; 命中 avoid_hours 视为低流动性时段, 命中 preferred_hours 加分)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.bias_detector = HTFBiasDetector(cfg.get("htf_bias", {}))
        self.liquidity_detector = LiquidityPoolDetector(cfg.get("liquidity", {}))
        self.structure_breaker = StructureBreaker(cfg.get("structure", {}))
        self.orderflow_filter = OrderFlowFilter(cfg.get("orderflow", {}))
        self.weights: Dict[str, float] = {
            "bias_alignment": 0.25,
            "liquidity_sweep": 0.25,
            "structure_break": 0.15,
            "orderflow": 0.15,
            "htf_nesting": 0.10,
            "price_zone": 0.10,
        }
        self.weights.update(
            {k: float(v) for k, v in (cfg.get("weights") or {}).items()})
        # 价格区 (新 schema price_zones 优先, 否则旧 zone_tolerance_pct)
        _pz = cfg.get("price_zones") or {}
        self.premium_threshold: float = float(
            _pz.get("premium_threshold", cfg.get("zone_tolerance_pct", 0.1)))
        self.discount_threshold: float = float(
            _pz.get("discount_threshold", cfg.get("zone_tolerance_pct", 0.1)))
        # 时间窗口 (新 schema time_windows 优先, 否则旧 low_liquidity_start/end)
        _tw = cfg.get("time_window") or {}
        _tws = cfg.get("time_windows") or {}
        self.time_windows_enabled: bool = bool(_tws.get("enabled", True))
        self.preferred_hours: List[int] = sorted({
            int(h) for h in _tws.get(
                "preferred_hours", [2, 4, 5, 6, 7, 8, 13, 14, 15])})
        self.avoid_hours: List[int] = sorted({
            int(h) for h in _tws.get(
                "avoid_hours", [0, 1, 16, 17, 18, 19, 20, 21, 22, 23])})
        self.low_liq_start: int = int(_tw.get("low_liquidity_start", 16))
        self.low_liq_end: int = int(_tw.get("low_liquidity_end", 22))
        self._detector_cfg: dict = cfg.get("fvg_detector", {})
        # FVG 前置硬门 (2026-08-07 落地): 孤立 FVG(既无流动性扫掠确认也无
        # 结构破坏确认)不交易。研究: 无汇流 FVG=抛硬币, sweep/MSS 确认才有边。
        # true=硬门(至少一项确认才交易), false=仅软评分(激进挡位可关)。
        self.require_sweep_or_structure: bool = bool(
            cfg.get("require_sweep_or_structure", True))

    # ------------------------------------------------------------------
    def check(
        self,
        fvg: Any,
        candles_1h: List[Any],
        candles_4h: List[Any],
        context: dict,
    ) -> Dict[str, Any]:
        """对一个 FVG 进行全面汇流确认。

        Args:
            fvg: FVGDetected (或含 direction/gap_low/gap_high/start_idx/end_idx
                 属性的兼容对象)
            candles_1h: 1H K 线 (正序)
            candles_4h: 4H K 线 (正序, HTF 偏执与嵌套用)
            context: 含 current_price / funding_rate / spread 等

        Returns:
            {
                "confluence_score": float,     # 0-1 综合汇流得分
                "conditions_met": List[str],
                "conditions_failed": List[str],
                "details": { 各条件 {met, score, ...} },
                "recommendation": "strong_buy"|"buy"|"neutral"|"reject",
                "entry_quality": "excellent"|"good"|"poor",
                "num_conditions_met": int,
            }
        """
        c1 = list(candles_1h or [])
        c4 = list(candles_4h or []) or c1
        htf = c4 if len(c4) >= 30 else c1
        ctx = context or {}
        price = ctx.get("current_price") or _safe_close(c1) \
            or float(getattr(fvg, "formation_price", 0.0))

        # 无任何 K 线 → 无汇流证据 → 纯中性 (不放行也不否决)
        if not c1:
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

        # 归一化 FVG 字段 (兼容 FVGDetected 与 strategy.FVG)
        fdir = "bullish" if getattr(fvg, "direction", "bullish") in (
            "bullish", "long") else "bearish"
        gl = float(getattr(fvg, "gap_low", getattr(fvg, "bottom", 0.0)))
        gh = float(getattr(fvg, "gap_high", getattr(fvg, "top", 0.0)))
        start_idx = int(getattr(fvg, "start_idx", getattr(fvg, "fvg_index", -1)))
        end_idx = int(getattr(fvg, "end_idx", getattr(fvg, "fvg_index", -1)))

        details: Dict[str, Any] = {}

        # 1) HTF 方向对齐
        bias = self.bias_detector.detect(htf)
        b_bias = str(bias.get("bias", "neutral"))
        b_conf = float(bias.get("confidence", 0.0))
        b_opp = ("bearish" if fdir == "bullish" else "bullish")
        if b_bias == fdir and b_conf >= 0.4:
            details["bias_alignment"] = {
                "met": True, "score": round(0.5 + 0.5 * b_conf, 4), "bias": b_bias}
        elif b_bias == b_opp:
            details["bias_alignment"] = {
                "met": False, "score": round(0.5 * (1.0 - b_conf), 4), "bias": b_bias}
        else:  # neutral
            details["bias_alignment"] = {"met": False, "score": 0.5, "bias": b_bias}

        # 2) 流动性猎杀确认 (形成位附近的猎杀, 池方向须与 FVG 一致)
        swept_pool = "none"
        _lo = max(1, start_idx - 2)
        _hi = min(len(c1) - 1, end_idx + 3)
        for i in range(_lo, _hi):
            _sw = self.liquidity_detector.detect_sweep(c1, i)
            if _sw.get("is_sweep"):
                swept_pool = _sw.get("pool") or "bsl"
                break
        fav_pool = "bsl" if fdir == "bullish" else "ssl"
        if swept_pool == fav_pool:
            details["liquidity_sweep"] = {
                "met": True, "score": 0.85, "swept_pool": swept_pool}
        elif swept_pool != "none":
            details["liquidity_sweep"] = {
                "met": False, "score": 0.1, "swept_pool": swept_pool}
        else:
            details["liquidity_sweep"] = {
                "met": False, "score": 0.3, "swept_pool": "none"}

        # 3) 结构破坏确认 (break 方向与 FVG 一致, 否则看 regime 顺向)
        sb = self.structure_breaker.detect(c1)
        sb_type = sb.get("last_break_type")
        sb_regime = str(sb.get("regime", "neutral"))
        break_fav = (fdir == "bullish" and sb_type == "bullish_break") or \
                    (fdir == "bearish" and sb_type == "bearish_break")
        if break_fav:
            details["structure_break"] = {"met": True, "score": 0.9, "type": sb_type}
        elif sb_regime == fdir:
            details["structure_break"] = {"met": True, "score": 0.65, "type": sb_type}
        else:
            details["structure_break"] = {"met": False, "score": 0.3, "type": sb_type}

        # 4) 订单流确认 (模拟 Delta)
        delta = self.orderflow_filter.detect_delta_divergence(c1)
        of_fav = (fdir == "bullish" and delta > 0.1) or \
                 (fdir == "bearish" and delta < -0.1)
        if of_fav:
            details["orderflow"] = {
                "met": True, "score": round(min(1.0, 0.6 + abs(delta)), 4),
                "delta": delta}
        else:
            details["orderflow"] = {"met": False, "score": 0.4, "delta": delta}

        # 5) HTF 嵌套 (1H FVG 是否落在更大的 4H FVG 缺口内)
        htf_fvg = self._find_nesting(fdir, gl, gh, start_idx, end_idx, c4)
        if htf_fvg is not None:
            details["htf_nesting"] = {
                "met": True, "score": 0.8, "htf_fvg": htf_fvg}
        else:
            details["htf_nesting"] = {"met": False, "score": 0.3, "htf_fvg": {}}

        # 6) 价格位置 (溢价/折价区, 基于 VWAP)
        zone = "equilibrium"
        p_met, p_score = False, 0.5
        vwap = self.bias_detector.compute_vwap(htf)
        if vwap > 0 and price > 0:
            dev = (price - vwap) / vwap * 100.0
            if dev > self.premium_threshold:
                zone = "premium"
            elif dev < -self.discount_threshold:
                zone = "discount"
            zone_fav = (fdir == "bullish" and zone == "discount") or \
                       (fdir == "bearish" and zone == "premium")
            if zone_fav:
                p_met, p_score = True, round(min(1.0, 0.6 + abs(dev) / 2.0), 4)
            else:
                p_met, p_score = False, 0.4
        details["price_zone"] = {
            "met": p_met, "score": round(p_score, 4), "zone": zone}

        # 7) 时间窗口 (软过滤, 默认不参与加权)
        window, t_met, t_score = self._time_window(c1, fvg)
        details["time_window"] = {
            "met": t_met, "score": t_score, "window": window}

        # ---- 加权综合 (time_window 默认权重 0) ----
        w_used = {k: v for k, v in self.weights.items() if k != "time_window"}
        w_sum = sum(w_used.values()) or 1.0
        total = 0.0
        for k, w in w_used.items():
            total += w * float(details.get(k, {}).get("score", 0.0))
        cscore = round(float(min(max(total / w_sum, 0.0), 1.0)), 4)

        met_list = [k for k, v in details.items() if v["met"]]
        failed_list = [k for k, v in details.items() if not v["met"]]

        # ---- 推荐级别 ----
        if cscore >= 0.7:
            rec = "strong_buy"
        elif cscore >= 0.55:
            rec = "buy"
        elif cscore >= 0.4:
            rec = "neutral"
        else:
            rec = "reject"
        # 低流动性时段软过滤: 降级 (与"UTC16-22 阈值提高"策略一致)
        if window == "low_liquidity" and rec in ("strong_buy", "buy"):
            rec = "neutral"

        # ---- 入口质量 ----
        if (cscore >= 0.75
                and details["bias_alignment"]["met"]
                and details["liquidity_sweep"]["met"]
                and details["structure_break"]["met"]):
            eq = "excellent"
        elif cscore >= 0.6:
            eq = "good"
        else:
            eq = "poor"

        # ---- FVG 前置硬门 (2026-08-07): 孤立 FVG 不交易 ----
        # 研究: 无汇流 FVG=抛硬币; sweep(流动性扫掠)或 MSS(结构破坏)至少
        # 一项确认才有 edge。两项都未确认 → 强制 reject + poor,
        # 配合 agent 的 confluence_reject_poor 拦截实现硬门。
        if (self.require_sweep_or_structure
                and not (details["liquidity_sweep"]["met"]
                         or details["structure_break"]["met"])):
            rec = "reject"
            eq = "poor"

        return {
            "confluence_score": cscore,
            "conditions_met": met_list,
            "conditions_failed": failed_list,
            "details": details,
            "recommendation": rec,
            "entry_quality": eq,
            "num_conditions_met": len(met_list),
        }

    # ------------------------------------------------------------------
    def get_confluence_features(self, result: Dict[str, Any]) -> Dict[str, float]:
        """将汇流确认结果转换为 ML 特征向量 (用于扩展 compute_features)。

        Args:
            result: check() 的返回结果

        Returns:
            {
                "confluence_score": float,
                "bias_aligned": 0/1, "liquidity_swept": 0/1,
                "structure_broken": 0/1, "orderflow_positive": 0/1,
                "htf_nested": 0/1, "in_premium_zone": 0/1,
                "in_good_time": 0/1, "num_conditions_met": int,
                "entry_quality_score": float,
            }
        """
        d = result.get("details", {})
        zone = str(d.get("price_zone", {}).get("zone", "equilibrium"))
        eq_map = {"excellent": 1.0, "good": 0.65, "poor": 0.25}
        return {
            "confluence_score": float(result.get("confluence_score", 0.0)),
            "bias_aligned": int(bool(d.get("bias_alignment", {}).get("met"))),
            "liquidity_swept": int(bool(d.get("liquidity_sweep", {}).get("met"))),
            "structure_broken": int(bool(d.get("structure_break", {}).get("met"))),
            "orderflow_positive": int(bool(d.get("orderflow", {}).get("met"))),
            "htf_nested": int(bool(d.get("htf_nesting", {}).get("met"))),
            "in_premium_zone": int(zone == "premium"),
            "in_good_time": int(bool(d.get("time_window", {}).get("met"))),
            "num_conditions_met": int(result.get("num_conditions_met", 0)),
            "entry_quality_score": float(eq_map.get(
                result.get("entry_quality", "poor"), 0.25)),
        }

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _find_nesting(self, fdir: str, gl: float, gh: float,
                      start_idx: int, end_idx: int,
                      candles_4h: List[Any]) -> Optional[Dict[str, Any]]:
        """在 4H K 线中查找包含当前 FVG 缺口且方向一致的大周期 FVG。"""
        if not candles_4h or len(candles_4h) < 5:
            return None
        try:
            from fvg_detector import FVGDetector  # noqa: PLC0415 (惰性, 防环)
        except ImportError:
            return None
        htf_fvgs = FVGDetector(self._detector_cfg).detect({"4H": candles_4h})
        for hf in htf_fvgs:
            if hf.direction != fdir:
                continue
            if hf.start_idx == start_idx and hf.end_idx == end_idx:
                continue  # 跳过自身 (c1/c4 相同时)
            if hf.gap_low <= gl and hf.gap_high >= gh:
                return {
                    "timeframe": hf.timeframe,
                    "direction": hf.direction,
                    "gap_low": round(hf.gap_low, 8),
                    "gap_high": round(hf.gap_high, 8),
                    "start_idx": hf.start_idx,
                    "end_idx": hf.end_idx,
                    "quality_score": round(hf.quality_score, 4),
                }
        return None

    def _time_window(self, candles: List[Any], fvg: Any) -> Tuple[str, bool, float]:
        """按最后 K 线时间戳 (UTC) 判定交易时段。

        新 schema (time_windows.enabled): 命中 avoid_hours → 低流动性;
        命中 preferred_hours → 可交易; 其余 → 中性可交易。
        旧 schema (time_window): 按 low_liquidity_start/end 区间。
        """
        ts = int(getattr(fvg, "formation_ts", 0)) or 0
        if candles:
            ts = int(getattr(candles[-1], "timestamp", 0)) or ts
        if not ts:
            return "unknown", False, 0.0
        hour = (ts // 3_600_000) % 24

        if self.time_windows_enabled:
            if hour in self.avoid_hours:
                return "low_liquidity", False, 0.2
            if hour in self.preferred_hours:
                if hour < 8:
                    return "asia", True, 0.7
                if hour < 13:
                    return "london", True, 0.85
                if hour < 16:
                    return "newyork", True, 0.9
                return "late", True, 0.6
            return "neutral", True, 0.55

        s, e = self.low_liq_start, self.low_liq_end
        if s < e and s <= hour < e:
            return "low_liquidity", False, 0.2
        if e < s and (hour >= s or hour < e):   # 跨天窗口
            return "low_liquidity", False, 0.2
        if hour < 8:
            return "asia", True, 0.7
        if hour < 13:
            return "london", True, 0.85
        if hour < 16:
            return "newyork", True, 0.9
        return "late", True, 0.6


__all__ = [
    "HTFBiasDetector",
    "LiquidityPoolDetector",
    "StructureBreaker",
    "OrderFlowFilter",
    "ConfluenceChecker",
]
