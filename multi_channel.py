"""
多渠道信息融合分析系统 — 超级交易专家引擎。

人设：超级交易专家 (Master Trader)
  - 10 年+ 加密货币交易经验，经历过 3 轮牛熊
  - 风格：多维度证据链交叉验证，绝不做单维度决策
  - 语言：专业犀利，一针见血，不废话
  - 座右铭："市场永远是对的，但证据链不会骗人。"

五通道分析框架：
  ┌─────────────────────────────────────────────────────┐
  │  Ch1 价格行为 (40%)   FVG + 波动率 + 多周期共振     │
  │  Ch2 市场结构 (25%)   订单簿失衡 + 流动性墙          │
  │  Ch3 资金流向 (20%)   资金费率 + OI + 主动量          │
  │  Ch4 市场情绪 (10%)   多空比 + 恐慌贪婪指数           │
  │  Ch5 宏观背景 ( 5%)   BTC 主导 + 板块轮动             │
  └─────────────────────────────────────────────────────┘
         ↓ 加权融合 ↓
  ┌─────────────────────────────────────────────────────┐
  │  超级交易专家研判引擎                                │
  │  · 证据链交叉验证  · 矛盾检测  · 置信度量化          │
  │  · 风险提示         · 仓位建议   · 入场时机判断       │
  └─────────────────────────────────────────────────────┘
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

import numpy as np

from okx_client import OKXClient
from strategy import Candle, FVG, Signal, candles_from_raw, detect_fvg


logger = logging.getLogger(__name__)


# ===========================================================================
# 数据类
# ===========================================================================

@dataclass
class ChannelReport:
    """单个通道的分析报告。"""
    channel_name: str
    weight: float                    # 通道权重 (0-1)
    bullish_score: float             # 看涨分 (0-1)
    bearish_score: float             # 看跌分 (0-1)
    net_score: float                 # 净分 = bullish - bearish (-1 ~ +1)
    confidence: float                # 本通道置信度 (0-1)
    observations: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MasterAnalysis:
    """超级交易专家综合研判结果。"""
    symbol: str
    timestamp: float
    # 融合评分
    final_score: float               # 最终综合分 (-1 ~ +1, 正=看多, 负=看空)
    final_confidence: float          # 综合置信度 (0-1)
    direction: str                   # "long" | "short" | "neutral"
    # 各通道
    channels: List[ChannelReport] = field(default_factory=list)
    # 专家研判
    expert_verdict: str = ""         # 专家结论（自然语言）
    expert_reasoning: str = ""       # 推理过程
    key_risks: List[str] = field(default_factory=list)
    trade_suggestion: str = ""       # 交易建议
    # 一致性检查
    channel_agreement: float = 0.0   # 通道一致性 (0-1, 越高越一致)
    contradiction_flag: bool = False  # 是否存在矛盾信号


# ===========================================================================
# 通道 1：价格行为分析 (40%)
# ===========================================================================

def analyze_price_action(
    candles_1h: List[Candle],
    candles_4h: List[Candle],
    current_price: float,
    fvg_signals: List[Signal],
    config: dict,
) -> ChannelReport:
    """价格行为通道分析。

    分析维度：
      - FVG 信号质量 & 方向
      - 波动率状态（扩张/收缩）
      - 多周期趋势一致性
      - 极端波动标记
    """
    obs = []
    reds = []
    bullish = 0.0
    bearish = 0.0

    # ---- 1. FVG 信号评估 ----
    long_fvgs = [s for s in fvg_signals if s.position_side == "long"]
    short_fvgs = [s for s in fvg_signals if s.position_side == "short"]

    best_long = max([s.score for s in long_fvgs]) if long_fvgs else 0
    best_short = max([s.score for s in short_fvgs]) if short_fvgs else 0

    if best_long > 0.6:
        obs.append(f"检测到高质量看涨 FVG (评分 {best_long:.2f})，缺口可能回补")
        bullish += 0.30
    if best_short > 0.6:
        obs.append(f"检测到高质量看跌 FVG (评分 {best_short:.2f})，缺口可能回补")
        bearish += 0.30

    # 异常波动标记
    abnormal_fvgs = [s for s in fvg_signals if s.fvg.is_abnormal]
    if abnormal_fvgs:
        avg_sigma = np.mean([s.fvg.sigma for s in abnormal_fvgs])
        obs.append(f"伴随 {len(abnormal_fvgs)} 个异常波动 FVG (平均 {avg_sigma:.1f}σ)")

    # ---- 2. 多周期趋势一致性 ----
    # 1H 趋势
    if len(candles_1h) >= 20:
        ma10_1h = np.mean([c.close for c in candles_1h[-10:]])
        ma20_1h = np.mean([c.close for c in candles_1h[-20:]])
        trend_1h = "up" if ma10_1h > ma20_1h else "down"
    else:
        trend_1h = "unknown"

    # 4H 趋势
    if len(candles_4h) >= 20:
        ma10_4h = np.mean([c.close for c in candles_4h[-10:]])
        ma20_4h = np.mean([c.close for c in candles_4h[-20:]])
        trend_4h = "up" if ma10_4h > ma20_4h else "down"
    else:
        trend_4h = "unknown"

    if trend_1h == "up" and trend_4h == "up":
        obs.append("1H/4H 趋势共振向上，多头结构完整")
        bullish += 0.20
    elif trend_1h == "down" and trend_4h == "down":
        obs.append("1H/4H 趋势共振向下，空头结构完整")
        bearish += 0.20
    elif trend_1h != trend_4h and trend_1h != "unknown" and trend_4h != "unknown":
        obs.append(f"多周期背离：1H={trend_1h}, 4H={trend_4h}，大周期约束优先")
        reds.append("多周期趋势不一致，信号可靠性降低")

    # ---- 3. 波动率状态 ----
    if len(candles_1h) >= 30:
        returns = [abs(math.log(candles_1h[i].close / candles_1h[i - 1].close))
                   for i in range(1, len(candles_1h))]
        current_vol = np.std(returns[-10:]) if len(returns) >= 10 else 0
        hist_vol = np.std(returns) if len(returns) >= 10 else 0
        if hist_vol > 0:
            vol_ratio = current_vol / hist_vol
            if vol_ratio > 1.5:
                obs.append(f"波动率扩张 ({vol_ratio:.1f}x)，可能酝酿大行情")
            elif vol_ratio < 0.5:
                obs.append(f"波动率收缩 ({vol_ratio:.1f}x)，可能蓄力突破")

    # ---- 4. 价格位置 ----
    if len(candles_4h) >= 20:
        high_20 = max(c.high for c in candles_4h[-20:])
        low_20 = min(c.low for c in candles_4h[-20:])
        price_position = (current_price - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
        if price_position > 0.8:
            obs.append(f"价格处于 4H 高位区间 ({price_position:.0%})，追多风险大")
            reds.append("价格处于区间高位，不宜追多")
        elif price_position < 0.2:
            obs.append(f"价格处于 4H 低位区间 ({price_position:.0%})，追空风险大")
            reds.append("价格处于区间低位，不宜追空")

    # 归一化
    total = bullish + bearish
    if total > 0:
        bullish /= total
        bearish /= total

    confidence = min(0.95, 0.5 + len(obs) * 0.10 - len(reds) * 0.15)

    return ChannelReport(
        channel_name="价格行为",
        weight=0.40,
        bullish_score=round(bullish, 3),
        bearish_score=round(bearish, 3),
        net_score=round(bullish - bearish, 3),
        confidence=round(max(0.1, confidence), 3),
        observations=obs if obs else ["价格行为信号中性，无明确方向"],
        red_flags=reds,
        raw_data={
            "trend_1h": trend_1h,
            "trend_4h": trend_4h,
            "fvg_long_count": len(long_fvgs),
            "fvg_short_count": len(short_fvgs),
            "abnormal_fvg_count": len(abnormal_fvgs),
        },
    )


# ===========================================================================
# 通道 2：市场结构分析 (25%)
# ===========================================================================

def analyze_market_structure(
    client: OKXClient,
    inst_id: str,
    current_price: float,
    config: dict,
) -> ChannelReport:
    """市场结构通道分析。

    分析维度：
      - 订单簿买卖失衡度
      - 流动性墙检测
      - 深度集中度
    """
    obs = []
    reds = []
    bullish = 0.0
    bearish = 0.0

    order_book = client.get_order_book(inst_id, sz=20)
    if not order_book:
        return ChannelReport(
            channel_name="市场结构",
            weight=0.25,
            bullish_score=0.0,
            bearish_score=0.0,
            net_score=0.0,
            confidence=0.0,
            observations=["订单簿数据不可用"],
        )

    # ---- 1. 买卖深度失衡 ----
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    bid_volume = sum(float(b[1]) for b in bids[:10])
    ask_volume = sum(float(a[1]) for a in asks[:10])

    if bid_volume + ask_volume > 0:
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    else:
        imbalance = 0.0

    if imbalance > 0.3:
        obs.append(f"买方深度显著占优 (失衡度 {imbalance:.1%})，买盘承接强劲")
        bullish += 0.35
    elif imbalance > 0.1:
        obs.append(f"买方深度略占优 (失衡度 {imbalance:.1%})")
        bullish += 0.15
    elif imbalance < -0.3:
        obs.append(f"卖方深度显著占优 (失衡度 {imbalance:.1%})，卖盘压制明显")
        bearish += 0.35
    elif imbalance < -0.1:
        obs.append(f"卖方深度略占优 (失衡度 {imbalance:.1%})")
        bearish += 0.15
    else:
        obs.append(f"买卖深度均衡 (失衡度 {imbalance:.1%})，无明确偏向")

    # ---- 2. 流动性墙检测 ----
    # 寻找价格附近 ±2% 内的最大挂单
    best_bid = float(bids[0][0]) if bids else 0
    best_ask = float(asks[0][0]) if asks else 0
    spread_pct = (best_ask - best_bid) / best_ask * 100 if best_ask > 0 else 0

    # 买方流动性墙
    bid_walls = []
    for b in bids:
        px = float(b[0])
        sz = float(b[1])
        if px >= current_price * 0.98:
            if sz > np.mean([float(x[1]) for x in bids[:5]]) * 2:
                bid_walls.append((px, sz))

    # 卖方流动性墙
    ask_walls = []
    for a in asks:
        px = float(a[0])
        sz = float(a[1])
        if px <= current_price * 1.02:
            if sz > np.mean([float(x[1]) for x in asks[:5]]) * 2:
                ask_walls.append((px, sz))

    if bid_walls:
        wall_desc = ", ".join([f"{px:.2f}({sz:.0f}张)" for px, sz in bid_walls[:2]])
        obs.append(f"下方流动性墙: {wall_desc}")
        bullish += 0.10

    if ask_walls:
        wall_desc = ", ".join([f"{px:.2f}({sz:.0f}张)" for px, sz in ask_walls[:2]])
        obs.append(f"上方流动性墙: {wall_desc}")
        bearish += 0.10

    # ---- 3. 价差 ----
    if spread_pct > 0.5:
        obs.append(f"买卖价差较宽 ({spread_pct:.2f}%)，流动性偏弱")
        reds.append(f"价差 {spread_pct:.2f}% 偏高")

    # 归一化
    total = bullish + bearish
    if total > 0:
        bullish /= total
        bearish /= total

    confidence = min(0.90, 0.5 + len(obs) * 0.08)

    return ChannelReport(
        channel_name="市场结构",
        weight=0.25,
        bullish_score=round(bullish, 3),
        bearish_score=round(bearish, 3),
        net_score=round(bullish - bearish, 3),
        confidence=round(max(0.1, confidence), 3),
        observations=obs,
        red_flags=reds,
        raw_data={
            "imbalance": round(imbalance, 4),
            "spread_pct": round(spread_pct, 4),
            "bid_walls": len(bid_walls),
            "ask_walls": len(ask_walls),
        },
    )


# ===========================================================================
# 通道 3：资金流向分析 (20%)
# ===========================================================================

def analyze_capital_flow(
    client: OKXClient,
    inst_id: str,
    config: dict,
) -> ChannelReport:
    """资金流向通道分析。

    分析维度：
      - 资金费率：当前水平 + 趋势
      - OI 变化
      - 主动买卖量比
    """
    obs = []
    reds = []
    bullish = 0.0
    bearish = 0.0

    # ---- 1. 资金费率分析 ----
    funding_rate = client.get_funding_rate(inst_id)
    funding_history = client.get_funding_rate_history(inst_id, limit=24)

    if funding_rate is not None:
        if funding_rate > 0.005:  # 0.5% 以上极度看多
            obs.append(f"资金费率极度偏多 ({funding_rate:.4%})，多头拥挤，警惕回调")
            reds.append(f"资金费率 {funding_rate:.4%} 过高，多头拥挤风险")
            bearish += 0.25
        elif funding_rate > 0.001:  # 0.1% 以上偏多
            obs.append(f"资金费率偏多 ({funding_rate:.4%})，市场情绪偏乐观")
            bullish += 0.10
        elif funding_rate < -0.005:
            obs.append(f"资金费率极度偏空 ({funding_rate:.4%})，空头拥挤，可能反弹")
            reds.append(f"资金费率 {funding_rate:.4%} 过低，空头拥挤风险")
            bullish += 0.25
        elif funding_rate < -0.001:
            obs.append(f"资金费率偏空 ({funding_rate:.4%})，市场情绪偏悲观")
            bearish += 0.10
        else:
            obs.append(f"资金费率中性 ({funding_rate:.4%})，多空平衡")

        # 资金费率趋势
        if funding_history and len(funding_history) >= 8:
            recent_rates = [float(r.get("fundingRate", "0")) for r in funding_history[:8]]
            early_rates = [float(r.get("fundingRate", "0")) for r in funding_history[8:16]]
            if recent_rates and early_rates:
                recent_avg = np.mean(recent_rates)
                early_avg = np.mean(early_rates)
                if early_avg != 0:
                    trend = (recent_avg - early_avg) / abs(early_avg)
                    if trend > 0.5:
                        obs.append(f"资金费率趋势上升，多头情绪升温")
                    elif trend < -0.5:
                        obs.append(f"资金费率趋势下降，空头情绪升温")

    # ---- 2. OI 分析 ----
    oi = client.get_open_interest(inst_id)
    if oi is not None:
        obs.append(f"当前 OI: {oi:,.0f} 张")

    # ---- 3. 主动买卖量分析 ----
    taker_data = client.get_taker_volume(inst_id, bar="1H", limit=12)
    if taker_data and len(taker_data) >= 4:
        buy_vol_total = sum(float(t.get("buyVol", "0")) for t in taker_data[:4])
        sell_vol_total = sum(float(t.get("sellVol", "0")) for t in taker_data[:4])
        if buy_vol_total + sell_vol_total > 0:
            taker_ratio = buy_vol_total / (buy_vol_total + sell_vol_total)
            if taker_ratio > 0.55:
                obs.append(f"主动买量占优 ({taker_ratio:.1%})，买方积极")
                bullish += 0.15
            elif taker_ratio < 0.45:
                obs.append(f"主动卖量占优 ({1 - taker_ratio:.1%})，卖方积极")
                bearish += 0.15
            else:
                obs.append(f"主动买卖量均衡 ({taker_ratio:.1%})")

    # 归一化
    total = bullish + bearish
    if total > 0:
        bullish /= total
        bearish /= total

    confidence = min(0.90, 0.5 + len(obs) * 0.08 - len(reds) * 0.10)

    return ChannelReport(
        channel_name="资金流向",
        weight=0.20,
        bullish_score=round(bullish, 3),
        bearish_score=round(bearish, 3),
        net_score=round(bullish - bearish, 3),
        confidence=round(max(0.1, confidence), 3),
        observations=obs,
        red_flags=reds,
        raw_data={
            "funding_rate": funding_rate,
            "oi": oi,
        },
    )


# ===========================================================================
# 通道 4：市场情绪分析 (10%)
# ===========================================================================

def analyze_market_sentiment(
    client: OKXClient,
    inst_id: str,
    config: dict,
) -> ChannelReport:
    """市场情绪通道分析。

    分析维度：
      - 多空比
      - 恐慌贪婪指数（外部 API）
      - 全网情绪综合判断
    """
    obs = []
    reds = []
    bullish = 0.0
    bearish = 0.0

    # ---- 1. 多空比 ----
    ls_ratio = client.get_long_short_ratio(inst_id, period="1H")
    if ls_ratio is not None and ls_ratio > 0:
        if ls_ratio > 2.0:
            obs.append(f"多空比极高 ({ls_ratio:.2f})，市场过度看多，警惕反转")
            reds.append(f"多空比 {ls_ratio:.2f} 极端，反向风险大")
            bearish += 0.20
        elif ls_ratio > 1.2:
            obs.append(f"多空比偏多 ({ls_ratio:.2f})，散户偏乐观")
            bullish += 0.05
        elif ls_ratio < 0.5:
            obs.append(f"多空比极低 ({ls_ratio:.2f})，市场过度看空，可能反弹")
            reds.append(f"多空比 {ls_ratio:.2f} 极端，反向风险大")
            bullish += 0.20
        elif ls_ratio < 0.8:
            obs.append(f"多空比偏空 ({ls_ratio:.2f})，散户偏悲观")
            bearish += 0.05
        else:
            obs.append(f"多空比中性 ({ls_ratio:.2f})")

    # ---- 2. 全网恐慌贪婪指数 ----
    fgi = _fetch_fear_greed_index()
    if fgi is not None:
        if fgi >= 75:
            obs.append(f"恐慌贪婪指数: {fgi} (极度贪婪)，市场过热")
            reds.append("极度贪婪区域，回调风险加大")
            bearish += 0.15
        elif fgi >= 60:
            obs.append(f"恐慌贪婪指数: {fgi} (贪婪)")
        elif fgi <= 25:
            obs.append(f"恐慌贪婪指数: {fgi} (极度恐慌)，市场恐慌往往是机会")
            bullish += 0.15
        elif fgi <= 40:
            obs.append(f"恐慌贪婪指数: {fgi} (恐慌)")
        else:
            obs.append(f"恐慌贪婪指数: {fgi} (中性)")

    # 归一化
    total = bullish + bearish
    if total > 0:
        bullish /= total
        bearish /= total

    confidence = min(0.85, 0.4 + len(obs) * 0.08)

    return ChannelReport(
        channel_name="市场情绪",
        weight=0.10,
        bullish_score=round(bullish, 3),
        bearish_score=round(bearish, 3),
        net_score=round(bullish - bearish, 3),
        confidence=round(max(0.1, confidence), 3),
        observations=obs,
        red_flags=reds,
        raw_data={
            "ls_ratio": ls_ratio,
            "fear_greed_index": fgi,
        },
    )


def _fetch_fear_greed_index() -> Optional[int]:
    """获取加密货币恐慌贪婪指数 (0-100)。"""
    import requests as req
    try:
        resp = req.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=10,
            headers={"User-Agent": "OKX-FVG-Agent/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            return int(data["data"][0]["value"])
    except Exception:
        pass
    return None


# ===========================================================================
# 通道 5：宏观背景分析 (5%)
# ===========================================================================

def analyze_macro_context(
    client: OKXClient,
    inst_id: str,
    config: dict,
) -> ChannelReport:
    """宏观背景通道分析。

    分析维度：
      - BTC 主导地位趋势
      - 标的与 BTC 相关性方向
    """
    obs = []
    reds = []
    bullish = 0.0
    bearish = 0.0

    # ---- 1. BTC 主导地位 ----
    # 用 BTC 指数价格变化近似判断
    btc_candles = client.get_index_candles("BTC-USDT", bar="1D", limit=7)
    eth_candles = client.get_index_candles("ETH-USDT", bar="1D", limit=7)

    if btc_candles and eth_candles and len(btc_candles) >= 3 and len(eth_candles) >= 3:
        btc_ret = (float(btc_candles[0][4]) / float(btc_candles[-1][4]) - 1) * 100
        eth_ret = (float(eth_candles[0][4]) / float(eth_candles[-1][4]) - 1) * 100

        if btc_ret > eth_ret + 2:
            obs.append(f"BTC 主导走强 (BTC {btc_ret:+.1f}% vs ETH {eth_ret:+.1f}%)，山寨可能承压")
            if inst_id != "BTC-USDT-SWAP":
                bearish += 0.05
        elif eth_ret > btc_ret + 2:
            obs.append(f"山寨走强 (ETH {eth_ret:+.1f}% vs BTC {btc_ret:+.1f}%)，风险偏好回升")
            if inst_id != "BTC-USDT-SWAP":
                bullish += 0.05

    # ---- 2. 标的性质 ----
    if inst_id == "BTC-USDT-SWAP":
        obs.append("BTC 是市场锚定资产，宏观环境直接影响方向")
    elif inst_id == "ETH-USDT-SWAP":
        obs.append("ETH 受 BTC 走势和生态叙事双重驱动")

    if not obs:
        obs.append("宏观背景信号中性")

    # 归一化
    total = bullish + bearish
    if total > 0:
        bullish /= total
        bearish /= total

    return ChannelReport(
        channel_name="宏观背景",
        weight=0.05,
        bullish_score=round(bullish, 3),
        bearish_score=round(bearish, 3),
        net_score=round(bullish - bearish, 3),
        confidence=round(max(0.1, 0.3 + len(obs) * 0.05), 3),
        observations=obs,
        red_flags=reds,
        raw_data={},
    )


# ===========================================================================
# 超级交易专家研判引擎
# ===========================================================================

class MasterTraderEngine:
    """超级交易专家 — 多通道融合研判引擎。

    职责：
      1. 加权融合五通道评分
      2. 证据链交叉验证
      3. 矛盾检测
      4. 生成自然语言专家研判
      5. 输出交易建议
    """

    # 通道权重配置
    DEFAULT_WEIGHTS = {
        "价格行为": 0.40,
        "市场结构": 0.25,
        "资金流向": 0.20,
        "市场情绪": 0.10,
        "宏观背景": 0.05,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def analyze(
        self,
        symbol: str,
        channels: List[ChannelReport],
        fvg_signals: List[Signal],
    ) -> MasterAnalysis:
        """执行超级交易专家综合研判。

        Args:
            symbol: 交易对
            channels: 五通道分析报告
            fvg_signals: FVG 信号列表

        Returns:
            MasterAnalysis 综合研判结果
        """
        # ---- 1. 加权融合 ----
        weighted_score = 0.0
        total_confidence = 0.0
        total_weight = 0.0

        for ch in channels:
            w = self.weights.get(ch.channel_name, ch.weight)
            weighted_score += ch.net_score * w * ch.confidence
            total_confidence += w * ch.confidence
            total_weight += w

        if total_confidence > 0:
            final_score = weighted_score / total_confidence
        else:
            final_score = 0.0

        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.0

        # ---- 2. 方向判断 ----
        if final_score > 0.15:
            direction = "long"
        elif final_score < -0.15:
            direction = "short"
        else:
            direction = "neutral"

        # ---- 3. 通道一致性检查 ----
        net_scores = [ch.net_score for ch in channels if ch.confidence > 0.2]
        if len(net_scores) >= 3:
            positive = sum(1 for s in net_scores if s > 0.05)
            negative = sum(1 for s in net_scores if s < -0.05)
            agreement = max(positive, negative) / len(net_scores)
        else:
            agreement = 0.5

        contradiction = (agreement < 0.6) and (len(net_scores) >= 3)

        # ---- 4. 收集红旗 ----
        all_reds = []
        for ch in channels:
            all_reds.extend(ch.red_flags)

        # ---- 5. 生成专家研判 ----
        expert_verdict, expert_reasoning, trade_suggestion = self._generate_expert_analysis(
            symbol=symbol,
            channels=channels,
            final_score=final_score,
            final_confidence=final_confidence,
            direction=direction,
            agreement=agreement,
            all_reds=all_reds,
            fvg_signals=fvg_signals,
        )

        return MasterAnalysis(
            symbol=symbol,
            timestamp=time.time(),
            final_score=round(final_score, 3),
            final_confidence=round(final_confidence, 3),
            direction=direction,
            channels=channels,
            expert_verdict=expert_verdict,
            expert_reasoning=expert_reasoning,
            key_risks=all_reds,
            trade_suggestion=trade_suggestion,
            channel_agreement=round(agreement, 3),
            contradiction_flag=contradiction,
        )

    def _generate_expert_analysis(
        self,
        symbol: str,
        channels: List[ChannelReport],
        final_score: float,
        final_confidence: float,
        direction: str,
        agreement: float,
        all_reds: List[str],
        fvg_signals: List[Signal],
    ) -> Tuple[str, str, str]:
        """生成自然语言专家分析 — 人设驱动。"""

        # 通道摘要
        ch_summary = []
        for ch in channels:
            if ch.confidence > 0.2:
                direction_word = "偏多" if ch.net_score > 0.1 else (
                    "偏空" if ch.net_score < -0.1 else "中性"
                )
                ch_summary.append(f"  [{ch.channel_name}] {direction_word} "
                                  f"(置信度 {ch.confidence:.0%})")

        # ---- 专家结论 ----
        if direction == "long":
            if final_confidence > 0.6:
                verdict = (
                    f"「{symbol}」多维度证据链指向多头方向。\n"
                    f"综合评分 {final_score:+.2f}，置信度 {final_confidence:.0%}。\n"
                    f"这是经过五通道交叉验证的结论，不是凭感觉拍脑袋。"
                )
            elif final_confidence > 0.4:
                verdict = (
                    f"「{symbol}」整体偏向多头，但证据链还不够硬。\n"
                    f"综合评分 {final_score:+.2f}，置信度 {final_confidence:.0%}。\n"
                    f"有利润空间，但别上头，仓位控制好。"
                )
            else:
                verdict = (
                    f"「{symbol}」微弱偏多，信号不够强。\n"
                    f"综合评分 {final_score:+.2f}，置信度 {final_confidence:.0%}。\n"
                    f"这种情况宁可错过，不要做错。"
                )
        elif direction == "short":
            if final_confidence > 0.6:
                verdict = (
                    f"「{symbol}」多维度证据链指向空头方向。\n"
                    f"综合评分 {final_score:+.2f}，置信度 {final_confidence:.0%}。\n"
                    f"五通道数据汇聚指向同一个方向，这不是巧合。"
                )
            elif final_confidence > 0.4:
                verdict = (
                    f"「{symbol}」整体偏向空头，但信号还需要确认。\n"
                    f"综合评分 {final_score:+.2f}，置信度 {final_confidence:.0%}。\n"
                    f"可以轻仓试探，等确认后再加码。"
                )
            else:
                verdict = (
                    f"「{symbol}」微弱偏空，信号不够强。\n"
                    f"综合评分 {final_score:+.2f}，置信度 {final_confidence:.0%}。\n"
                    f"空头信号弱，不值得大动干戈。"
                )
        else:
            verdict = (
                f"「{symbol}」多空力量胶着，五通道信号方向不一致。\n"
                f"综合评分 {final_score:+.2f}，置信度 {final_confidence:.0%}。\n"
                f"这种情况我选择观望 — 不做也是一种交易。"
            )

        # ---- 推理过程 ----
        reasoning_lines = [
            "【推理过程】",
            f"通道一致性: {agreement:.0%} ({'一致' if agreement >= 0.6 else '分歧' if agreement >= 0.4 else '严重分歧'})",
        ]
        reasoning_lines.extend(ch_summary)

        if all_reds:
            reasoning_lines.append(f"\n⚠ 风险警示 ({len(all_reds)} 项):")
            for r in all_reds:
                reasoning_lines.append(f"  · {r}")

        if fvg_signals:
            best_fvg = fvg_signals[0]
            reasoning_lines.append(
                f"\n最优 FVG: {best_fvg.position_side.upper()} "
                f"{best_fvg.fvg.timeframe} "
                f"宽度={best_fvg.fvg.width_pct:.2f}% "
                f"评分={best_fvg.score:.2f}"
            )

        reasoning = "\n".join(reasoning_lines)

        # ---- 交易建议 ----
        if direction == "neutral":
            suggestion = "建议：观望。没有明确方向时，现金就是最好的仓位。"
        elif final_confidence < 0.4:
            suggestion = "建议：观望或极轻仓试探。信号不够强，不做不亏钱。"
        elif agreement < 0.5:
            suggestion = "建议：轻仓入场，严格止损。通道分歧意味着不确定性大。"
        elif len(all_reds) >= 2:
            suggestion = "建议：谨慎入场。存在多个风险信号，仓位控制在平时一半。"
        else:
            suggestion = "建议：按策略信号正常入场。多通道一致，证据链充分。"

        return verdict, reasoning, suggestion


# ===========================================================================
# 一站式分析入口
# ===========================================================================

def full_multi_channel_analysis(
    client: OKXClient,
    inst_id: str,
    current_price: float,
    candles_1h: List[Candle],
    candles_4h: List[Candle],
    fvg_signals: List[Signal],
    config: dict,
    engine: Optional[MasterTraderEngine] = None,
) -> MasterAnalysis:
    """执行完整的多通道分析。

    Args:
        client: OKX 客户端
        inst_id: 合约 ID
        current_price: 当前价格
        candles_1h: 1H K 线数据
        candles_4h: 4H K 线数据
        fvg_signals: 已生成的 FVG 信号
        config: 完整配置
        engine: 专家引擎（可选）

    Returns:
        MasterAnalysis 综合研判
    """
    if engine is None:
        engine = MasterTraderEngine()

    channels = []

    # Ch1: 价格行为
    try:
        ch1 = analyze_price_action(candles_1h, candles_4h, current_price, fvg_signals, config)
        channels.append(ch1)
    except Exception as e:
        logger.error(f"Ch1 价格行为分析失败: {e}")

    # Ch2: 市场结构
    try:
        ch2 = analyze_market_structure(client, inst_id, current_price, config)
        channels.append(ch2)
    except Exception as e:
        logger.error(f"Ch2 市场结构分析失败: {e}")

    # Ch3: 资金流向
    try:
        ch3 = analyze_capital_flow(client, inst_id, config)
        channels.append(ch3)
    except Exception as e:
        logger.error(f"Ch3 资金流向分析失败: {e}")

    # Ch4: 市场情绪
    try:
        ch4 = analyze_market_sentiment(client, inst_id, config)
        channels.append(ch4)
    except Exception as e:
        logger.error(f"Ch4 市场情绪分析失败: {e}")

    # Ch5: 宏观背景
    try:
        ch5 = analyze_macro_context(client, inst_id, config)
        channels.append(ch5)
    except Exception as e:
        logger.error(f"Ch5 宏观背景分析失败: {e}")

    # 专家研判
    analysis = engine.analyze(
        symbol=inst_id.replace("-USDT-SWAP", ""),
        channels=channels,
        fvg_signals=fvg_signals,
    )

    return analysis


# ===========================================================================
# 格式化输出
# ===========================================================================

def format_analysis_report(analysis: MasterAnalysis) -> str:
    """将综合研判格式化为可打印的报告。"""
    lines = [
        "",
        "╔" + "═" * 58 + "╗",
        f"║  🏆 超级交易专家 · 综合研判报告".ljust(61) + "║",
        f"║  标的: {analysis.symbol:<50}║",
        f"║  时间: {datetime.fromtimestamp(analysis.timestamp).strftime('%Y-%m-%d %H:%M:%S'):<50}║",
        "╠" + "═" * 58 + "╣",
    ]

    # 各通道摘要
    for ch in analysis.channels:
        if ch.confidence < 0.2:
            continue
        bar_len = max(1, int(abs(ch.net_score) * 20))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        direction = "多" if ch.net_score > 0.05 else ("空" if ch.net_score < -0.05 else "中")
        lines.append(
            f"║  [{ch.channel_name:<5}] {direction} {bar} "
            f"{ch.net_score:+.2f} (置信度 {ch.confidence:.0%})".ljust(61) + "║"
        )

    # 综合评分
    lines.append("╠" + "═" * 58 + "╣")
    score_bar_len = max(1, int(abs(analysis.final_score) * 30))
    score_bar = "█" * score_bar_len + "░" * (30 - score_bar_len)
    direction_word = "看多" if analysis.direction == "long" else (
        "看空" if analysis.direction == "short" else "中性"
    )
    lines.append(
        f"║  综合: {direction_word} {score_bar} "
        f"{analysis.final_score:+.2f}".ljust(61) + "║"
    )
    lines.append(
        f"║  置信度: {analysis.final_confidence:.0%}  |  "
        f"通道一致性: {analysis.channel_agreement:.0%}".ljust(61) + "║"
    )
    lines.append("╠" + "═" * 58 + "╣")

    # 专家结论
    lines.append("║  【专家结论】".ljust(61) + "║")
    for vline in analysis.expert_verdict.split("\n"):
        lines.append(f"║  {vline}".ljust(61) + "║")

    lines.append("╠" + "═" * 58 + "╣")

    # 推理过程
    for rline in analysis.expert_reasoning.split("\n"):
        lines.append(f"║  {rline}".ljust(61) + "║")

    lines.append("╠" + "═" * 58 + "╣")

    # 交易建议
    lines.append(f"║  {analysis.trade_suggestion}".ljust(61) + "║")
    lines.append("╚" + "═" * 58 + "╝")

    return "\n".join(lines)