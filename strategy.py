"""
策略模块 — 异常波动检测 + FVG 识别 + 信号过滤。

核心逻辑：
  - 异常检测：价格偏离 ≥3σ + 成交量 ≥5x 均量 → 排除基本面突破
  - FVG 检测：标准 ICT 三蜡烛缺口识别
  - 信号生成：限价单入场位 + 止盈止损计算
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np

# 独立 FVG 检测器（第一层检测，见 fvg_detector.py）
from fvg_detector import FVGDetector, FVGDetected


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    """单根 K 线。"""
    timestamp: int       # unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float        # 成交量（张数）


@dataclass
class FVG:
    """Fair Value Gap 结构。"""
    direction: str           # "long" | "short"
    top: float               # FVG 上界
    bottom: float            # FVG 下界
    width_pct: float         # 缺口宽度 (%)
    candle_ts: int           # FVG 形成时的 K 线时间戳
    timeframe: str           # "1H" | "4H"
    impulse_candle: Candle   # 推动蜡烛 (中间那根)
    fvg_index: int = -1      # 推动蜡烛在 candles 列表中的索引
    is_abnormal: bool = False  # 是否伴随异常波动
    sigma: float = 0.0       # 异常 sigma 值
    volume_ratio: float = 1.0  # 量比


@dataclass
class Signal:
    """交易信号。"""
    inst_id: str             # 合约 ID, e.g. "BTC-USDT-SWAP"
    fvg: FVG
    entry_price: float       # 限价入场价
    stop_loss: float         # 止损价
    take_profit: float       # 止盈价
    leverage: int            # 建议杠杆
    position_side: str       # "long" | "short"
    score: float = 0.0       # 信号评分 (0-1)
    reason: str = ""         # 信号原因
    ml_score: float = 0.0    # ML 二次评分概率 (0-1, 由 fvg_ml_ranker 填充, 默认 0=未评估)
    spread_pct: float = 0.0  # 买卖价差 (%), 汇流确认上下文用
    confluence_score: float = 0.0   # 汇流综合得分 (0-1, ConfluenceChecker 填充)
    confluence_details: dict = field(default_factory=dict)  # 汇流确认明细
    entry_quality: str = "poor"     # 入口质量 "excellent"|"good"|"poor"
    use_conditional_entry: bool = False   # 2026-08-10: 深挂触发单 — 价格先走到
    # 距回补位一个阈值窗口处触发, 触发后才挂限价进场(避免提前深挂空转)
    entry_trigger_px: float = 0.0         # conditional 触发单的触发价 (0=无)


@dataclass
class IFVG:
    """Inversion Fair Value Gap — 已填满并被反向突破的 FVG（极性强转）。

    ICT 概念: 看涨 FVG 被价格跌破(close<bottom)填满后继续下行，原支撑
    翻转为阻力 → 做空信号；看跌 FVG 被突破(close>top)后继续上行，原
    阻力翻转为支撑 → 做多信号。
    """
    direction: str          # 反转后的交易方向 "long" | "short"
    original: FVG           # 原始 FVG
    breakout_idx: int       # 反向突破蜡烛索引（candles 正序）
    breakout_price: float   # 触发价（该蜡烛收盘价）
    age_bars: int           # 距最新 K 线的根数


# ---------------------------------------------------------------------------
# 蜡烛数据转换
# ---------------------------------------------------------------------------

def candles_from_raw(raw: List[list], reverse: bool = True) -> List[Candle]:
    """将 OKX 原始 K 线数据转为 Candle 列表。
    OKX 返回: [[ts, o, h, l, c, vol, volCcy, ...], ...]
    默认时间倒序（最新在前），reverse=True 转为正序。
    """
    candles = []
    for row in raw:
        c = Candle(
            timestamp=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        candles.append(c)
    if reverse:
        candles.reverse()
    return candles


# ---------------------------------------------------------------------------
# 异常波动检测
# ---------------------------------------------------------------------------

def detect_abnormal_candle(
    candles: List[Candle],
    idx: int,
    sigma_threshold: float = 3.0,
    volume_ratio_threshold: float = 5.0,
    lookback: int = 50,
) -> Tuple[bool, float, float]:
    """检测 candles[idx] 是否为异常波动蜡烛。

    条件：
      1. 涨跌幅偏离均值 ≥ sigma_threshold * σ
      2. 成交量 ≥ volume_ratio_threshold * 均量

    Returns:
        (is_abnormal, sigma, volume_ratio)
    """
    if idx < lookback or idx < 1:
        return False, 0.0, 1.0

    # 修复 Bug 29: 统一使用 close-to-close 对数收益（与 agent.py 体制检测一致），
    # 避免单根蜡烛 c/o 收益与跨时间尺度收益的统计口径不一致
    returns = []
    volumes = []
    for i in range(idx - lookback, idx):
        if i < 1:
            continue
        if candles[i - 1].close <= 0 or candles[i].close <= 0:
            continue
        ret = math.log(candles[i].close / candles[i - 1].close)
        returns.append(abs(ret))
        volumes.append(candles[i].volume)

    if len(returns) < 20:
        return False, 0.0, 1.0

    median_ret = np.median(returns)
    mad = np.median(np.abs(returns - median_ret))
    if mad < 1e-10:
        return False, 0.0, 1.0

    # 注：使用 MAD (Median Absolute Deviation) 而非标准差作为离差度量。
    # MAD 对异常值更鲁棒，乘以 1.4826 转换为正态分布近似标准差。
    # RSS (Robust Standard Score) = (x - median) / (MAD * 1.4826) 等价于
    # 标准化 z-score 的鲁棒版本，行业标准做法。
    if candles[idx - 1].close <= 0 or candles[idx].close <= 0:
        return False, 0.0, 1.0
    current_ret = abs(math.log(candles[idx].close / candles[idx - 1].close))
    sigma = (current_ret - median_ret) / (mad * 1.4826)

    mean_vol = np.mean(volumes)
    if mean_vol < 1e-10:
        volume_ratio = 1.0
    else:
        volume_ratio = candles[idx].volume / mean_vol

    is_abnormal = (sigma >= sigma_threshold) and (volume_ratio >= volume_ratio_threshold)
    return is_abnormal, sigma, volume_ratio


# ---------------------------------------------------------------------------
# FVG 检测
# ---------------------------------------------------------------------------

def detect_fvg(
    candles: List[Candle],
    timeframe: str,
    min_width_pct: float = 1.5,
    sigma_threshold: float = 3.0,
    volume_ratio_threshold: float = 5.0,
    lookback: int = 50,
) -> List[FVG]:
    """检测 Fair Value Gaps。

    标准 ICT 三蜡烛 FVG 模式：
      - 看涨 FVG: candles[i].high < candles[i+2].low
        → 缺口在 [candles[i].high, candles[i+2].low]
      - 看跌 FVG: candles[i].low > candles[i+2].high
        → 缺口在 [candles[i+2].high, candles[i].low]

    蜡烛索引: i, i+1(推动), i+2 (从左到右)

    Args:
        candles: 正序蜡烛列表
        timeframe: 时间周期
        min_width_pct: 最小 FVG 宽度 (%)
        sigma_threshold: 异常 sigma 阈值
        volume_ratio_threshold: 异常量比阈值
        lookback: 异常检测回溯窗口

    Returns:
        FVG 列表（按时间倒序）
    """
    fvgs = []
    if len(candles) < 3:
        return fvgs

    for i in range(len(candles) - 2):
        c0 = candles[i]       # 左蜡烛
        c1 = candles[i + 1]   # 推动蜡烛 (impulse)
        c2 = candles[i + 2]   # 右蜡烛

        # ---- 看涨 FVG ----
        if c0.high < c2.low:
            fvg_top = c2.low
            fvg_bottom = c0.high
            if fvg_bottom <= 0:
                continue
            width_pct = (fvg_top - fvg_bottom) / fvg_bottom * 100

            if width_pct >= min_width_pct:
                is_ab, sigma, vol_ratio = detect_abnormal_candle(
                    candles, i + 1, sigma_threshold, volume_ratio_threshold, lookback
                )
                fvgs.append(FVG(
                    direction="long",
                    top=fvg_top,
                    bottom=fvg_bottom,
                    width_pct=width_pct,
                    candle_ts=c2.timestamp,
                    timeframe=timeframe,
                    impulse_candle=c1,
                    fvg_index=i + 1,
                    is_abnormal=is_ab,
                    sigma=sigma,
                    volume_ratio=vol_ratio,
                ))

        # ---- 看跌 FVG ----
        elif c0.low > c2.high:
            fvg_top = c0.low
            fvg_bottom = c2.high
            if fvg_bottom <= 0:
                continue
            width_pct = (fvg_top - fvg_bottom) / fvg_bottom * 100

            if width_pct >= min_width_pct:
                is_ab, sigma, vol_ratio = detect_abnormal_candle(
                    candles, i + 1, sigma_threshold, volume_ratio_threshold, lookback
                )
                fvgs.append(FVG(
                    direction="short",
                    top=fvg_top,
                    bottom=fvg_bottom,
                    width_pct=width_pct,
                    candle_ts=c2.timestamp,
                    timeframe=timeframe,
                    impulse_candle=c1,
                    fvg_index=i + 1,
                    is_abnormal=is_ab,
                    sigma=sigma,
                    volume_ratio=vol_ratio,
                ))

    return fvgs


# ---------------------------------------------------------------------------
# iFVG (Inversion Fair Value Gap) 反转检测
# ---------------------------------------------------------------------------

def detect_ifvg(
    candles: List[Candle],
    fvgs: List[FVG],
    max_age_bars: int = 100,
) -> List[IFVG]:
    """从已检测的历史 FVG 中识别 Inversion FVG（极性强转）。

    对每个 FVG 追踪其形成后第一根反向收盘突破边界的蜡烛：
      - 看涨 FVG: close < bottom → 缺口被填满并跌破 → iFVG 做空
      - 看跌 FVG: close > top   → 缺口被填满并突破 → iFVG 做多

    突破确认采用收盘价（而非影线），过滤插针假突破，避免前视偏差。

    Args:
        candles: 正序蜡烛列表
        fvgs: detect_fvg 已返回的 FVG 列表（复用，避免重复检测）
        max_age_bars: 只保留距最新 K 线不超过该根数的 iFVG（太旧已失效）

    Returns:
        iFVG 列表（按 age 升序，最新在前）
    """
    ifvgs = []
    n = len(candles)
    for fvg in fvgs:
        if fvg.fvg_index < 0:
            continue
        for j in range(fvg.fvg_index + 1, n):
            c = candles[j]
            if fvg.direction == "long" and c.close < fvg.bottom:
                ifvgs.append(IFVG(
                    direction="short",
                    original=fvg,
                    breakout_idx=j,
                    breakout_price=c.close,
                    age_bars=n - 1 - j,
                ))
                break
            if fvg.direction == "short" and c.close > fvg.top:
                ifvgs.append(IFVG(
                    direction="long",
                    original=fvg,
                    breakout_idx=j,
                    breakout_price=c.close,
                    age_bars=n - 1 - j,
                ))
                break
    ifvgs = [i for i in ifvgs if i.age_bars <= max_age_bars]
    ifvgs.sort(key=lambda i: i.age_bars)
    return ifvgs


def _ifvg_to_fvg(ifvg: IFVG, candles: List[Candle]) -> Optional[FVG]:
    """将 iFVG 转为等效 FVG（反转后的新缺口结构），复用 generate_signal 完整信号链。

    反转结构:
      - iFVG 做多(原看跌 FVG 被向上突破): 新看涨缺口 bottom=原FVG上沿(转性
        支撑), top=突破后近端高点；价格回踩该缺口时按做多信号处理
      - iFVG 做空(原看涨 FVG 被向下跌破): 新看跌缺口 top=原FVG下沿(转性
        阻力), bottom=突破后近端低点

    Returns:
        等效 FVG 或 None（数据异常）
    """
    orig = ifvg.original
    width = abs(orig.top - orig.bottom)
    if width <= 0 or ifvg.breakout_idx < 0 or ifvg.breakout_idx >= len(candles):
        return None
    breakout = candles[ifvg.breakout_idx]
    look = candles[ifvg.breakout_idx:ifvg.breakout_idx + 5]
    if not look:
        look = [breakout]

    if ifvg.direction == "long":
        bottom = orig.top
        top = max(c.high for c in look)
        if top <= bottom:  # 突破后无有效位移，用原宽度兜底
            top = bottom + width
    else:
        top = orig.bottom
        bottom = min(c.low for c in look)
        if bottom >= top:
            bottom = top - width

    if bottom <= 0:
        return None

    return FVG(
        direction=ifvg.direction,
        top=top,
        bottom=bottom,
        width_pct=abs(top - bottom) / bottom * 100,
        candle_ts=breakout.timestamp,
        timeframe=orig.timeframe,
        impulse_candle=breakout,
        fvg_index=ifvg.breakout_idx,
        is_abnormal=orig.is_abnormal,
        sigma=orig.sigma,
        volume_ratio=orig.volume_ratio,
    )


# ---------------------------------------------------------------------------
# 信号生成
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Improvement 1: FVG 入场前确认蜡烛 (P1-2a)
# ---------------------------------------------------------------------------

def _match_confirmation(last: Candle, prev: Candle, direction: str) -> bool:
    """纯形态确认：做多看涨吞没/锤子线，做空看跌吞没/上吊线。"""
    if direction == "long":
        # 做多确认：看涨吞没 或 锤子线（下影线 >= 实体 * 2）
        body = abs(last.close - last.open)
        lower_wick = min(last.open, last.close) - last.low
        upper_wick = last.high - max(last.open, last.close)

        # 看涨吞没：前阴后阳，后阳实体包前阴实体
        bullish_engulfing = (
            prev.close < prev.open  # 前阴
            and last.close > last.open  # 后阳
            and last.open <= prev.close  # 后开 <= 前收
            and last.close >= prev.open  # 后收 >= 前开
        )

        # 锤子线：下影 >= 实体 * 2, 上影 < 实体 * 0.3
        hammer = (body > 0 and lower_wick >= body * 2 and upper_wick < body * 0.3)

        return bullish_engulfing or hammer

    # 做空确认：看跌吞没 或 上吊线（上影线 >= 实体 * 2）
    body = abs(last.close - last.open)
    upper_wick = last.high - max(last.open, last.close)
    lower_wick = min(last.open, last.close) - last.low

    # 看跌吞没
    bearish_engulfing = (
        prev.close > prev.open  # 前阳
        and last.close < last.open  # 后阴
        and last.open >= prev.close  # 后开 >= 前收
        and last.close <= prev.open  # 后收 <= 前开
    )

    # 上吊线：上影 >= 实体 * 2, 下影 < 实体 * 0.3
    shooting_star = (body > 0 and upper_wick >= body * 2 and lower_wick < body * 0.3)

    return bearish_engulfing or shooting_star


def _check_confirmation_candle(
    candles: list,
    fvg_top: float,
    fvg_bottom: float,
    direction: str,
    current_price: float,
    lookback: int = 10,
) -> bool:
    """检查 FVG 区域是否有确认蜡烛（兼容流动性猎手挂单模式）。

    确认蜡烛条件：
    1. 形态确认（pin bar / engulfing / hammer），方向正确
    2. 确认蜡烛须与 FVG 区域相关：
       - 实时价在 FVG 区域内 → 直接用最新两根蜡烛确认（原行为）
       - 挂单模式下价格在 FVG 外 → 回溯最近 lookback 根蜡烛，找"回补蜡烛"
         （影线触及 FVG 区间或收盘落在 FVG 内），对回补蜡烛做形态确认

    修复: 原逻辑强制要求 current_price 位于 FVG 区域内，而流动性猎手挂单
    模式下价格通常在 FVG 外（做多等回踩前低下方、做空等反弹前高上方），
    导致确认恒不通过、评分恒被 ×0.7。调整后确认基于回补位附近的历史蜡烛。

    Args:
        candles: 最近 N 根蜡烛
        fvg_top: FVG 上沿
        fvg_bottom: FVG 下沿
        direction: "long" 或 "short"
        current_price: 当前价格
        lookback: 挂单模式下回溯的蜡烛数

    Returns:
        True 如果存在确认蜡烛
    """
    if len(candles) < 2:
        return False

    # 实时价在 FVG 区域内：用最新两根蜡烛直接确认
    if fvg_bottom <= current_price <= fvg_top:
        return _match_confirmation(candles[-1], candles[-2], direction)

    # 挂单模式：回溯最近 lookback 根蜡烛，找回补蜡烛做形态确认
    start = max(0, len(candles) - lookback)
    window = candles[start:]
    for i in range(len(window) - 1, 0, -1):
        last = window[i]
        prev = window[i - 1]
        # 回补判定：该蜡烛影线/实体触及 FVG 区间，或收盘落在 FVG 内
        touched = (last.low <= fvg_top and last.high >= fvg_bottom) or \
                  (fvg_bottom <= last.close <= fvg_top)
        if not touched:
            continue
        if _match_confirmation(last, prev, direction):
            return True

    return False


# ---------------------------------------------------------------------------
# Improvement 2: FVG 成交量验证 (P1-2b)
# ---------------------------------------------------------------------------

def _check_volume_confirmation(
    candles: list,
    fvg_index: int,
    lookback: int = 20,
) -> Tuple[bool, float]:
    """检查 FVG 填充是否有成交量支撑。

    Args:
        candles: 完整蜡烛列表
        fvg_index: FVG 形成蜡烛的索引（中间蜡烛）
        lookback: 回看蜡烛数

    Returns:
        (is_confirmed, volume_ratio)
    """
    fill_candles = candles[fvg_index + 2 : min(len(candles), fvg_index + 5)]
    if len(fill_candles) == 0:
        return False, 0.0
    fill_volume = sum(c.volume for c in fill_candles)

    # FVG 形成前的平均成交量
    start_idx = max(0, fvg_index - lookback)
    pre_candles = candles[start_idx:max(0, fvg_index - 1)]
    if not pre_candles:
        return False, 0.0

    avg_volume = sum(c.volume for c in pre_candles) / len(pre_candles)

    if avg_volume <= 0:
        return False, 0.0

    # Normalize to actual candle count
    volume_ratio = fill_volume / (avg_volume * len(fill_candles))

    # 无量回补 = 假回补
    if volume_ratio < 0.5:
        return False, volume_ratio

    return True, volume_ratio


# ---------------------------------------------------------------------------
# Improvement 3: FVG Degree 量化过滤 (P2-3)
# ---------------------------------------------------------------------------

def _compute_fvg_degree(
    candles: list,
    fvg_index: int,
) -> Optional[float]:
    """计算 FVG Degree 指标（Kondapally 2026）。

    Degree = |β₁| = FVG 形成期间 tick 数据的线性回归斜率。

    由于没有 tick 数据，使用 FVG 形成蜡烛的 OHLC 近似：
    Degree ≈ abs((c2.close - c0.open) / (c2.timestamp - c0.timestamp))

    分类：
    - ≤ 0.00015: 低度（机构共识）— 3.2x 更强反应
    - 0.00015-0.0004: 中度
    - > 0.0004: 高度（噪音）— 反应不可靠

    Returns:
        Degree 值，或 None（无法计算）
    """
    if len(candles) < fvg_index + 2:
        return None

    c0 = candles[fvg_index - 1]  # 第一根蜡烛
    c2 = candles[fvg_index + 1]  # 第三根蜡烛

    # 时间跨度（秒）
    # timestamp 是 unix ms，转为秒
    time_span = (c2.timestamp - c0.timestamp) / 1000.0
    if time_span <= 0:
        return None

    # 价格变化
    price_change = abs(c2.close - c0.open)

    # Degree = 价格变化 / 时间跨度
    degree = price_change / time_span

    return degree


def _is_high_quality_fvg(degree: Optional[float]) -> bool:
    """判断 FVG 是否为高质量（低度）。"""
    if degree is None:
        return False  # 无法判断时保守处理
    return degree <= 0.0004  # 中低度视为可接受


# ---------------------------------------------------------------------------
# Improvement 4: 回踩走势状态检查 (用户要求 — 挂单前回溯近期K线状态与走势)
# ---------------------------------------------------------------------------

def _check_pullback_state(
    candles: Optional[List[Candle]],
    entry_price: float,
    current_price: float,
    direction: str,
    lookback: int = 8,
) -> Tuple[bool, float]:
    """回溯近期 K 线状态与走势，判断当前是否处于合理的回踩阶段。

    流动性猎手在 FVG 上方/下方挂限价单等回踩，但"回踩"与"破位下跌"必须
    区分：健康回踩（缩量、企稳）→ 挂单合理；放量破位 → 挂单=下跌中接飞刀。

    检查项（做多为例，做空镜像）：
      1. 超跌否决: 当前价已击穿挂单价 entry → 价格深度破位，限价单若挂出将
         立即成交在下跌趋势中，直接否决挂单。
      2. 连续下跌: 最近 lookback 根已收盘 K 线全部收跌（无一根反弹）→ 下跌
         动能强，回补概率低 → ×0.8。
      3. 放量下跌: 下跌 K 线均量 > 1.5 × 上涨 K 线均量 → 放量下杀（机构
         出货/破位），而非缩量健康回踩 → ×0.7。
      4. 止跌迹象: 最后一根已收盘 K 线收阳（回落中出现反抽）→ 回踩接近
         尾声，企稳概率大 → ×1.05。

    Returns:
        (ok, factor) — ok=False 否决挂单；factor 为评分系数（默认 1.0）
    """
    if not candles or len(candles) < 2:
        return True, 1.0

    # 用已收盘 K 线统计（最新一根可能未收盘，不参与形态/量价统计）
    window = list(candles[-lookback - 1:-1])
    if len(window) < 2:
        return True, 1.0

    if direction == "long":
        # 1. 超跌否决: 价格已跌穿挂单位 → 破位，接飞刀
        if current_price < entry_price:
            logger.debug(f"[Pullback] {direction} 当前价 {current_price} 已击穿挂单价 "
                         f"{entry_price}，超跌/破位，否决挂单")
            return False, 0.0
    else:
        # 做空镜像: 价格已涨穿挂单位
        if current_price > entry_price:
            logger.debug(f"[Pullback] {direction} 当前价 {current_price} 已突破挂单价 "
                         f"{entry_price}，超涨/破位，否决挂单")
            return False, 0.0

    factor = 1.0
    up_vols = [c.volume for c in window if c.close >= c.open]
    down_vols = [c.volume for c in window if c.close < c.open]
    last = window[-1]

    if direction == "long":
        # 2. 连续下跌（全部收跌，无一根反弹）
        if all(c.close < c.open for c in window):
            factor *= 0.8
            logger.debug(f"[Pullback] {direction} 最近 {len(window)} 根连续收跌，下跌动能强")
        # 3. 放量下跌（下跌均量 > 1.5x 上涨均量）→ 破位嫌疑；全部收跌另计 0.8
        if down_vols and up_vols and \
                (sum(down_vols) / len(down_vols)) > 1.5 * (sum(up_vols) / len(up_vols)):
            factor *= 0.7
            logger.debug(f"[Pullback] {direction} 放量下跌（下跌均量 > 1.5x 上涨均量），疑似破位")
        # 4. 止跌迹象: 最后一根已收盘 K 线收阳
        if last.close > last.open:
            factor *= 1.05
    else:
        # 做空镜像
        if all(c.close > c.open for c in window):
            factor *= 0.8
            logger.debug(f"[Pullback] {direction} 最近 {len(window)} 根连续收涨，上涨动能强")
        if up_vols and down_vols and \
                (sum(up_vols) / len(up_vols)) > 1.5 * (sum(down_vols) / len(down_vols)):
            factor *= 0.7
            logger.debug(f"[Pullback] {direction} 放量上涨（上涨均量 > 1.5x 下跌均量），疑似破位")
        if last.close < last.open:
            factor *= 1.05

    return True, factor


# ---------------------------------------------------------------------------
# Improvement 5: 综合技术参考 (用户要求 — 走势/布林带/多空持仓比 与
#                 成交量/K线形态 共同研判，混合模式: 极端逆势否决 + 背离降分 + 共振加分)
# ---------------------------------------------------------------------------

def _compute_bollinger(
    candles: List[Candle],
    period: int = 20,
    num_std: float = 2.0,
) -> Optional[Dict[str, float]]:
    """计算布林带 (period SMA ± num_std×σ)。

    Returns:
        {"upper", "middle", "lower", "bandwidth", "pct_b"} 或 None（数据不足）
      - bandwidth = (upper - lower) / middle：带宽，越小表示波动率越压缩（挤压）
      - pct_b = (price - lower) / (upper - lower)：价格在带内位置
        (<0 跌破下轨, >1 突破上轨)
    """
    if candles is None or len(candles) < period:
        return None
    window = [c.close for c in candles[-period:]]
    if any(c <= 0 for c in window):
        return None
    sma = sum(window) / period
    variance = sum((x - sma) ** 2 for x in window) / period
    std = math.sqrt(variance)
    if sma <= 0 or std <= 0:
        return None
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma
    price = candles[-1].close
    pct_b = (price - lower) / (upper - lower) if upper != lower else 0.5
    return {
        "upper": upper,
        "middle": sma,
        "lower": lower,
        "bandwidth": bandwidth,
        "pct_b": pct_b,
    }


def _compute_rsi_series(
    candles: List[Candle],
    period: int = 14,
) -> List[float]:
    """RSI 序列（Wilder 平滑）。rsis[k] 对应 closes[k+period] 结束的窗口。

    数据不足返回空列表。avg_loss=0（全程上涨）时 RSI=100。
    """
    if candles is None:
        return []
    closes = [c.close for c in candles if c.close > 0]
    if len(closes) < period + 1:
        return []
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [100.0 if avg_loss <= 1e-12 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)]
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsis.append(100.0 if avg_loss <= 1e-12 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss))
    return rsis


def _compute_adx(
    candles: List[Candle],
    period: int = 14,
) -> Optional[Tuple[float, float, float]]:
    """ADX + 方向性指标 DI（Wilder 平滑）。

    Returns:
        (adx, pdi, ndi) 或 None
      - ADX>25 趋势市（FVG 回补有效）；ADX<20 震荡市（假回补风险高）
      - +DI>-DI 多方主导；-DI>+DI 空方主导（ADX 仅强度、DI 定方向）
    """
    if candles is None or len(candles) < period * 2 + 1:
        return None
    trs, pdms, ndms = [], [], []
    for i in range(1, len(candles)):
        h, l = candles[i].high, candles[i].low
        pc = candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        up = h - candles[i - 1].high
        dn = candles[i - 1].low - l
        pdms.append(up if (up > dn and up > 0) else 0.0)
        ndms.append(dn if (dn > up and dn > 0) else 0.0)
    atr = sum(trs[:period]) / period
    pdi_s = sum(pdms[:period]) / period
    ndi_s = sum(ndms[:period]) / period
    dxs = []
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        pdi_s = (pdi_s * (period - 1) + pdms[i]) / period
        ndi_s = (ndi_s * (period - 1) + ndms[i]) / period
        pdi = 100.0 * pdi_s / atr if atr > 0 else 0.0
        ndi = 100.0 * ndi_s / atr if atr > 0 else 0.0
        dx = 100.0 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) > 0 else 0.0
        dxs.append(dx)
    if not dxs:
        return None
    adx = sum(dxs[-period:]) / period
    return adx, pdi, ndi


def _extreme_move_reject_reason(
    candles: Optional[List[Candle]],
    current_price: float,
    atr_period: int = 14,
    min_adx: float = 0.0,
    min_atr_pct: float = 0.0,
) -> Optional[str]:
    """FVG Hunter 硬门禁 — 只吃确定性极端行情 (2026-08-08 横盘监控复盘)。

    横盘折磨行情 (低 ADX 无趋势 或 低 ATR 波动太小) 里 FVG 回补易被噪音扫掉,
    入场后长时间无方向空耗 (实测 ADA 4h 横盘 0.1987~0.2001, ±0.35%)。

    判定标准 (行业标准, 非凭空参数):
      - ADX(14) ≥ min_adx: 趋势强度。Wilder 1978: >25 趋势市(FVG回补有效)/
        <20 震荡市(假回补风险高), 与 adx_trend_threshold 同值。
      - ATR(14)/现价 ≥ min_atr_pct: 单根 K 线平均振幅, 保证"上涨下跌都很大"。

    Returns:
        拒绝原因字符串 (含 ADX/ATR% 实测值) 或 None (放行)。
        数据不足时 fail-open 放行 (防新币/数据缺失阻塞, 与 ATRGrade 同策略)。
    """
    if min_adx <= 0 and min_atr_pct <= 0:
        return None  # 门禁关闭
    if not candles or len(candles) < atr_period * 2:
        return None  # fail-open: K线不足不做裁决
    reasons = []
    _adx = _compute_adx(candles, period=atr_period)
    _adx_val = _adx[0] if _adx else None
    if _adx_val is not None and min_adx > 0 and _adx_val < min_adx:
        reasons.append(f"ADX={_adx_val:.0f}<{min_adx:.0f}(横盘无趋势)")
    _atr = _compute_atr_wilder(candles, atr_period)
    _atr_pct = (_atr / current_price * 100.0) \
        if (_atr > 0 and current_price > 0) else 0.0
    if min_atr_pct > 0 and _atr_pct < min_atr_pct:
        reasons.append(f"ATR%={_atr_pct:.2f}<{min_atr_pct:.2f}(波动太小)")
    return "; ".join(reasons) if reasons else None


def _compute_vwap(candles: List[Candle]) -> Optional[float]:
    """当日锚定 VWAP（从最后一根K线的自然日 0 点开始累计）。

    典型价 = (H+L+C)/3；VWAP = Σ(典型价×量) / Σ量。
    机构日内成本参考线：价格回踩/贴合 VWAP 常为支撑阻力位。
    当日数据不足或量为 0 返回 None。
    """
    if not candles:
        return None
    from datetime import datetime, timezone
    last_day = datetime.fromtimestamp(
        candles[-1].timestamp / 1000.0, tz=timezone.utc
    ).date()
    total_pv = 0.0
    total_v = 0.0
    for c in reversed(candles):
        day = datetime.fromtimestamp(c.timestamp / 1000.0, tz=timezone.utc).date()
        if day != last_day:
            break
        tp = (c.high + c.low + c.close) / 3.0
        total_pv += tp * c.volume
        total_v += c.volume
    if total_v <= 0:
        return None
    return total_pv / total_v


def _detect_rsi_divergence(
    closes: List[float],
    rsis: List[float],
    lookback: int = 14,
) -> str:
    """简化双摆动点 RSI 背离检测。

    rsis[k] 与 closes[k+period] 对齐，取最近 lookback 段（排除最后一根
    未确认摆动点）。
      - 顶背离(bearish): 后一价格高点 >= 前一价格高点，但后一 RSI 高点 < 前一 RSI 高点
        → 价格创新高而动量未确认，上涨衰竭（做空顺向/做多逆向）
      - 底背离(bullish): 后一价格低点 <= 前一价格低点，但后一 RSI 低点 > 前一 RSI 低点
        → 价格创新低而抛压衰竭（做多顺向/做空逆向）

    Returns:
        "bullish" | "bearish" | "none"
    """
    if len(closes) < lookback * 2 or len(rsis) < lookback:
        return "none"
    seg_c = closes[-lookback - 1:-1]  # 排除最后一根，避免未完成摆动点
    seg_r = rsis[-lookback:]
    seg_c = seg_c[-len(seg_r):]
    if len(seg_c) < 4 or len(seg_r) < 4:
        return "none"

    # 两段式摆动点: 前/后各半段内取极值，比较价格与动量的背离
    half = len(seg_c) // 2
    if half < 1:
        return "none"
    seg_a = seg_c[:half]
    seg_b = seg_c[half:]
    _ra = seg_r[:half]
    _rb = seg_r[half:]

    # 顶背离: 后段高点价格 >= 前段高点价格，但后段 RSI 高点 < 前段 RSI 高点
    i_prev = int(np.argmax(seg_a))
    i_last = half + int(np.argmax(seg_b))
    if seg_c[i_last] >= seg_c[i_prev] and _rb[int(np.argmax(seg_b))] < _ra[i_prev]:
        return "bearish"

    # 底背离: 后段低点价格 <= 前段低点价格，但后段 RSI 低点 > 前段 RSI 低点
    j_prev = int(np.argmin(seg_a))
    j_last = half + int(np.argmin(seg_b))
    if seg_c[j_last] <= seg_c[j_prev] and _rb[int(np.argmin(seg_b))] > _ra[j_prev]:
        return "bullish"
    return "none"


def _check_technical_state(
    candles: Optional[List[Candle]],
    current_price: float,
    direction: str,
    long_short_ratio: Optional[float] = None,
    entry_price: Optional[float] = None,
    bb_period: int = 20,
    bb_std: float = 2.0,
    squeeze_threshold: float = 0.6,
    trend_ma_period: int = 20,
    rsi_period: int = 14,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    adx_period: int = 14,
    adx_trend_threshold: float = 25.0,
    adx_range_threshold: float = 20.0,
    vwap_tolerance_pct: float = 0.5,
    divergence_lookback: int = 14,
    lsr_strong_high: float = 1.3,
    lsr_strong_low: float = 0.7,
    bb_veto_low: float = -0.2,
    bb_veto_high: float = 1.2,
) -> Tuple[bool, float, str]:
    """综合技术参考检查（混合模式，强化版）。

    与 _check_pullback_state（量价走势）互补，这里覆盖:
      1. 走势(趋势): close vs SMA(trend_ma_period)
      2. 布林带: 价格带内位置 %B + 带宽收口(挤压)
      3. 多空持仓比: long_short_ratio 方向一致性
      4. RSI: 超买超卖 + 双摆动点背离（强化版）
      5. ADX: 趋势强度（>25 趋势市 / <20 震荡市）+ DI 方向
      6. VWAP: 机构成本线方向一致性 + 入场位贴合共振

    规则（做多为例，做空镜像）:
      - 否决: %B < bb_veto_low(-0.2) → 价格跌破下轨 0.2 带宽仍在破位下跌，不接飞刀
      - 否决: %B > bb_veto_high(1.2) → 价格突破上轨 0.2 带宽已超买追高，不追单
        （行业标准 %B>1 超买 / <0 超卖；±0.2 带宽作为否决缓冲，防误杀）
      - 降分 ×0.9: 做多时 close < SMA（趋势未向上，回补概率低）
      - 降分 ×0.9: 做多时 %B > 1.0（价格在上轨上方，追高）
      - 加分 ×1.05: 带宽收口（bandwidth < 近窗口均值 × squeeze_threshold，
        波动率压缩=变盘前兆，FVG 回补概率高；行业参考 BBW<4% 为挤压）
      - 多空比: 做多 ratio ≥ lsr_strong_high(1.3) → ×1.05 顺向；
        ratio ≤ lsr_strong_low(0.7) → ×0.9 逆向
        （行业参考: >1.3 显著偏多, <0.7 显著偏空, 0.9~1.1 平衡区）
      - RSI: 做多 RSI≤30 → ×1.05（超卖企稳）；RSI≥70 → ×0.85（超买追高）
        RSI 顶背离 → 做多 ×0.85 / 做空 ×1.05；底背离 → 做多 ×1.05 / 做空 ×0.85
      - ADX: ≥25 趋势市 → ×1.05；≤20 震荡市 → ×0.85（假回补风险）
        +DI>-DI 多方主导 → 做多 ×1.05 / 做空 ×0.9（-DI 镜像）
      - VWAP: 做多价在 VWAP 上方 → ×1.05 / 下方 → ×0.9；
        |entry-VWAP|/VWAP ≤ vwap_tolerance_pct → ×1.05 共振

    Returns:
        (ok, factor, detail) — ok=False 否决；factor 评分系数；detail 说明
    """
    if not candles or len(candles) < max(bb_period, trend_ma_period) + 1:
        return True, 1.0, "技术数据不足"

    closes = [c.close for c in candles if c.close > 0]
    if len(closes) < trend_ma_period:
        return True, 1.0, "技术数据不足"

    factor = 1.0
    detail = []

    # ---- 1. 走势(趋势): close vs SMA ----
    sma = sum(closes[-trend_ma_period:]) / trend_ma_period
    if direction == "long":
        if current_price < sma:
            factor *= 0.9
            detail.append(f"价格在SMA{trend_ma_period}下方(趋势未向上)")
    else:
        if current_price > sma:
            factor *= 0.9
            detail.append(f"价格在SMA{trend_ma_period}上方(趋势未向下)")

    # ---- 2. 布林带: 位置 %B + 带宽挤压 ----
    bb = _compute_bollinger(candles, period=bb_period, num_std=bb_std)
    if bb:
        pct_b = bb["pct_b"]
        if direction == "long":
            if pct_b < bb_veto_low:
                return False, 0.0, f"跌破布林下轨过深(%B={pct_b:.2f}，<{bb_veto_low:.1f})，破位下跌中"
            if pct_b > bb_veto_high:
                return False, 0.0, f"突破布林上轨过深(%B={pct_b:.2f}，>{bb_veto_high:.1f})，超买追高"
            if pct_b > 1.0:
                factor *= 0.9
                detail.append(f"价格在布林上轨上方(%B={pct_b:.2f})，追高")
        else:
            if pct_b > bb_veto_high:
                return False, 0.0, f"突破布林上轨过深(%B={pct_b:.2f}，>{bb_veto_high:.1f})，破位上涨中"
            if pct_b < bb_veto_low:
                return False, 0.0, f"跌破布林下轨过深(%B={pct_b:.2f}，<{bb_veto_low:.1f})，超卖抄底"
            if pct_b < 0.0:
                factor *= 0.9
                detail.append(f"价格在布林下轨下方(%B={pct_b:.2f})，超卖")

        # 带宽收口(挤压): 当前带宽 < 近窗口均值 × squeeze_threshold → 变盘前兆
        _hist_span = min(len(candles), 60)
        _hist = candles[-_hist_span:]
        if len(_hist) >= bb_period + 1:
            _bws = []
            for i in range(bb_period, len(_hist)):
                _w = [c.close for c in _hist[i - bb_period:i + 1]]
                if all(x > 0 for x in _w):
                    _m = sum(_w) / len(_w)
                    _s = math.sqrt(sum((x - _m) ** 2 for x in _w) / len(_w))
                    if _m > 0 and _s > 0:
                        _bws.append(((_m + bb_std * _s) - (_m - bb_std * _s)) / _m)
            if _bws and bb["bandwidth"] < (sum(_bws) / len(_bws)) * squeeze_threshold:
                factor *= 1.05
                detail.append(f"布林带宽收口(挤压={bb['bandwidth']:.4f})")

    # ---- 3. 多空持仓比（行业参考: >1.3 显著偏多, <0.7 显著偏空, 0.9~1.1 平衡）----
    if long_short_ratio is not None and long_short_ratio > 0:
        if direction == "long":
            if long_short_ratio >= lsr_strong_high:
                factor *= 1.05
                detail.append(f"多空比{long_short_ratio:.2f}顺向(多头占优)")
            elif long_short_ratio <= lsr_strong_low:
                factor *= 0.9
                detail.append(f"多空比{long_short_ratio:.2f}逆向(空头占优)")
        else:
            if long_short_ratio <= lsr_strong_low:
                factor *= 1.05
                detail.append(f"多空比{long_short_ratio:.2f}顺向(空头占优)")
            elif long_short_ratio >= lsr_strong_high:
                factor *= 0.9
                detail.append(f"多空比{long_short_ratio:.2f}逆向(多头占优)")

    # ---- 4. RSI: 超买超卖 + 双摆动点背离（强化版）----
    # 行业标准: RSI>70 超买 / <30 超卖（Wilder 1978）；趋势市中可上调至 80/40、
    # 下调至 60/20（Constance Brown），静态 70/30 为跨市场默认。
    rsis = _compute_rsi_series(candles, period=rsi_period)
    if rsis:
        rsi_now = rsis[-1]
        if direction == "long":
            if rsi_now >= rsi_overbought:
                factor *= 0.85
                detail.append(f"RSI={rsi_now:.0f}超买(追高)")
            elif rsi_now <= rsi_oversold:
                factor *= 1.05
                detail.append(f"RSI={rsi_now:.0f}超卖(企稳)")
        else:
            if rsi_now <= rsi_oversold:
                factor *= 0.85
                detail.append(f"RSI={rsi_now:.0f}超卖(追空)")
            elif rsi_now >= rsi_overbought:
                factor *= 1.05
                detail.append(f"RSI={rsi_now:.0f}超买(顺向)")
        # 背离
        div = _detect_rsi_divergence(closes, rsis, lookback=divergence_lookback)
        if div == "bullish":
            if direction == "long":
                factor *= 1.05
                detail.append("RSI底背离(抛压衰竭)")
            else:
                factor *= 0.85
                detail.append("RSI底背离(逆做空)")
        elif div == "bearish":
            if direction == "long":
                factor *= 0.85
                detail.append("RSI顶背离(上涨衰竭)")
            else:
                factor *= 1.05
                detail.append("RSI顶背离(顺做空)")

    # ---- 5. ADX: 趋势强度 + DI 方向 ----
    # 行业标准: ADX>25 趋势市 / <20 震荡市（Wilder 1978）；ADX>40 极强趋势
    # 警惕衰竭/反转（趋势跟随的最佳区间 25~40）。
    _adx = _compute_adx(candles, period=adx_period)
    if _adx:
        adx, pdi, ndi = _adx
        if adx >= adx_trend_threshold:
            factor *= 1.05
            detail.append(f"ADX={adx:.0f}趋势市")
        elif adx <= adx_range_threshold:
            factor *= 0.85
            detail.append(f"ADX={adx:.0f}震荡市(假回补风险)")
        if pdi > ndi:
            if direction == "long":
                factor *= 1.05
                detail.append("+DI主导(多方占优)")
            else:
                factor *= 0.9
                detail.append("+DI主导(逆做空)")
        elif ndi > pdi:
            if direction == "long":
                factor *= 0.9
                detail.append("-DI主导(逆做多)")
            else:
                factor *= 1.05
                detail.append("-DI主导(空方占优)")

    # ---- 6. VWAP: 机构成本线方向一致性 + 入场位贴合共振 ----
    _vwap = _compute_vwap(candles)
    if _vwap and _vwap > 0:
        if direction == "long":
            if current_price >= _vwap:
                factor *= 1.05
                detail.append("价在VWAP上方(机构成本上)")
            else:
                factor *= 0.9
                detail.append("价在VWAP下方(机构成本下)")
        else:
            if current_price <= _vwap:
                factor *= 1.05
                detail.append("价在VWAP下方(机构成本下)")
            else:
                factor *= 0.9
                detail.append("价在VWAP上方(机构成本上)")
        if entry_price and entry_price > 0:
            _tol = vwap_tolerance_pct / 100.0
            if abs(entry_price - _vwap) / _vwap <= _tol:
                factor *= 1.05
                detail.append(f"入场位贴合VWAP({_vwap:.6g})")

    return True, factor, "; ".join(detail) if detail else "技术状态中性"


# ---------------------------------------------------------------------------
# Improvement 7: Qlib Alpha158 风格因子 (微软 Qlib 44k⭐ 标准化因子集子集)
# ---------------------------------------------------------------------------

def _compute_alpha158_style(
    candles: List[Candle],
) -> Optional[Dict[str, float]]:
    """计算 Alpha158 风格核心因子（Qlib 标准化定义的子集）。

    微软 Qlib (44k⭐) 的 Alpha158 是行业标准的可复现因子集。这里只取对
    FVG 回补方向判断有价值、且计算开销低的子集（避免全量 158 因子）：

      - ret5/ret10/ret20: 对数收益（动量）
      - ma5/ma10/ma20: 移动平均线（趋势排列）
      - slope20: MA20 斜率（近5根变化，归一化）
      - std10: 10 日波动率（波动收缩）
      - vol_ratio: 近5日均量 / 前15日均量（量能变化）
      - pos_20d: 价格在 20 日高低区间分位（0-1）

    Returns:
        因子 dict 或 None（数据不足）
    """
    if candles is None or len(candles) < 30:
        return None
    closes = [c.close for c in candles if c.close > 0]
    if len(closes) < 30:
        return None

    def _sma(win):
        return sum(closes[-win:]) / win

    ret5 = math.log(closes[-1] / closes[-6]) if closes[-6] > 0 else 0.0
    ret10 = math.log(closes[-1] / closes[-11]) if closes[-11] > 0 else 0.0
    ret20 = math.log(closes[-1] / closes[-21]) if closes[-21] > 0 else 0.0

    ma5, ma10, ma20 = _sma(5), _sma(10), _sma(20)

    # MA20 斜率: 当前 MA20 与 5 根前 MA20 的归一化变化
    _ma20_prev = sum(closes[-25:-5]) / 20
    slope20 = (ma20 - _ma20_prev) / ma20 if ma20 > 0 else 0.0

    # 10 日波动率
    _w10 = closes[-10:]
    _m10 = sum(_w10) / 10
    std10 = math.sqrt(sum((x - _m10) ** 2 for x in _w10) / 10) / _m10 if _m10 > 0 else 0.0

    # 量比: 近5日均量 / 前15日均量
    _v_near = [c.volume for c in candles[-5:] if c.volume > 0]
    _v_far = [c.volume for c in candles[-20:-5] if c.volume > 0]
    vol_ratio = (sum(_v_near) / len(_v_near) / (sum(_v_far) / len(_v_far))
                 if _v_near and _v_far and sum(_v_far) > 0 else 1.0)

    # 20 日高低区间分位
    _w20 = candles[-20:]
    _hi = max(c.high for c in _w20)
    _lo = min(c.low for c in _w20)
    pos_20d = (closes[-1] - _lo) / (_hi - _lo) if _hi > _lo else 0.5

    return {
        "ret5": ret5, "ret10": ret10, "ret20": ret20,
        "ma5": ma5, "ma10": ma10, "ma20": ma20,
        "slope20": slope20, "std10": std10,
        "vol_ratio": vol_ratio, "pos_20d": pos_20d,
    }


def _check_alpha_confluence(
    candles: Optional[List[Candle]],
    current_price: float,
    direction: str,
) -> Tuple[float, str]:
    """Alpha158 风格因子的方向一致性评分（Qlib 标准化因子子集）。

    与 _check_technical_state 互补，聚焦动量/均线排列/趋势斜率/波动收缩
    等未被布林/RSI/ADX/VWAP 覆盖的维度。全因子一致时加分，背离时降分，
    factor 裁剪到 [0.85, 1.15]，避免单因子过度影响总分。

    规则（做多为例，做空镜像）:
      - 短期动量: ret5>0 → ×1.03；ret5<0 且 ret20<0 → ×0.95
      - 均线排列: close>ma5>ma10>ma20 → ×1.03；close<ma20 → ×0.95
      - 趋势斜率: slope20>0 → ×1.02；slope20<0 → ×0.98
      - 波动收缩: std10 < 近60根中位数 → ×1.02（低波动回补更稳）
      - 低位区间: pos_20d<0.2 → ×1.02（超卖低位）；>0.8 → ×0.97（高位）

    Returns:
        (factor, detail)
    """
    f = _compute_alpha158_style(candles)
    if f is None:
        return 1.0, ""
    factor = 1.0
    detail = []

    if direction == "long":
        if f["ret5"] > 0:
            factor *= 1.03
            detail.append(f"Ret5={f['ret5']*100:.1f}%动量顺向")
        elif f["ret5"] < 0 and f["ret20"] < 0:
            factor *= 0.95
            detail.append(f"Ret5={f['ret5']*100:.1f}%连续负动量")
        if current_price > f["ma5"] > f["ma10"] > f["ma20"]:
            factor *= 1.03
            detail.append("均线多头排列")
        elif current_price < f["ma20"]:
            factor *= 0.95
            detail.append("价格在MA20下方")
        if f["slope20"] > 0:
            factor *= 1.02
            detail.append("MA20斜率向上")
        elif f["slope20"] < 0:
            factor *= 0.98
            detail.append("MA20斜率向下")
        if f["pos_20d"] < 0.2:
            factor *= 1.02
            detail.append(f"20日低位区(pos={f['pos_20d']:.2f})")
        elif f["pos_20d"] > 0.8:
            factor *= 0.97
            detail.append(f"20日高位区(pos={f['pos_20d']:.2f})")
    else:
        if f["ret5"] < 0:
            factor *= 1.03
            detail.append(f"Ret5={f['ret5']*100:.1f}%动量顺向")
        elif f["ret5"] > 0 and f["ret20"] > 0:
            factor *= 0.95
            detail.append(f"Ret5={f['ret5']*100:.1f}%连续正动量")
        if current_price < f["ma5"] < f["ma10"] < f["ma20"]:
            factor *= 1.03
            detail.append("均线空头排列")
        elif current_price > f["ma20"]:
            factor *= 0.95
            detail.append("价格在MA20上方")
        if f["slope20"] < 0:
            factor *= 1.02
            detail.append("MA20斜率向下")
        elif f["slope20"] > 0:
            factor *= 0.98
            detail.append("MA20斜率向上")
        if f["pos_20d"] > 0.8:
            factor *= 1.02
            detail.append(f"20日高位区(pos={f['pos_20d']:.2f})")
        elif f["pos_20d"] < 0.2:
            factor *= 0.97
            detail.append(f"20日低位区(pos={f['pos_20d']:.2f})")

    # 波动收缩（方向无关，低波动回补更稳定）
    if candles and len(candles) >= 60:
        _stds = []
        _c60 = [c.close for c in candles[-60:] if c.close > 0]
        for i in range(30, len(_c60)):
            _w = _c60[i - 10:i]
            if len(_w) == 10:
                _m = sum(_w) / 10
                _s = math.sqrt(sum((x - _m) ** 2 for x in _w) / 10)
                if _m > 0:
                    _stds.append(_s / _m)
        if _stds:
            _med = sorted(_stds)[len(_stds) // 2]
            if f["std10"] < _med:
                factor *= 1.02
                detail.append("波动收缩")

    factor = min(max(factor, 0.85), 1.15)
    return factor, "; ".join(detail) if detail else ""


# ---------------------------------------------------------------------------
# 信号生成
# ---------------------------------------------------------------------------

def _is_fvg_fresh(
    fvg: FVG,
    candles: Optional[List[Candle]],
    max_age_bars: int,
) -> Tuple[bool, int]:
    """检查 FVG 是否过期（距最新 K 线根数 ≤ max_age_bars 视为新鲜）。

    修复: 200 根 K 线窗口内任意老的 FVG 都生成信号（实测 BEAT 信号 entry
    偏离现价 148%、SLX 偏离 32%），过期缺口不仅难以成交还污染多通道分析。
    max_age_bars <= 0 时不过滤（兼容旧行为/测试）。

    Returns:
        (fresh, age_bars) — age_bars = -1 表示无法判定（视为新鲜）
    """
    if max_age_bars <= 0 or candles is None or fvg.fvg_index < 0:
        return True, -1
    age = len(candles) - 1 - fvg.fvg_index
    return age <= max_age_bars, age


def generate_signal(
    inst_id: str,
    fvg: FVG,
    current_price: float,
    entry_depth_pct: float = 0.15,
    fvg_target_pct: float = 0.50,
    stop_buffer_pct: float = 0.15,
    max_leverage: int = 10,
    funding_rate: Optional[float] = None,
    max_funding_rate_abs: float = 0.01,
    funding_confluence_min_abs: float = 0.0003,
    funding_confluence_max_abs: float = 0.001,
    # 修复: 流动性猎手挂单 — 限价单挂在前低下方/前高上方 (外扩 3-5%)，
    # 等价格扫穿前高/前低流动性后回补成交，避免在流动性密集区接单被扫穿。
    liquidity_extension_pct: float = 4.0,
    # 裁剪下限可配置: 实盘保持 3.0 (策略要求 3-5%)；纸面模式经
    # scan_fvg_all_timeframes 传入更小下限，缩短挂单距离加速成交(测试用)。
    liquidity_extension_min_pct: float = 3.0,
    # 挂单距离上限(%): 深挂(流动性猎手)入场价与现价偏离超过该阈值时，
    # 回退到 FVG 回补位，避免挂单远离现价永不成交(实测 RLS 偏离 7.2% 空转)。
    max_entry_distance_pct: float = 5.0,
    # ATR 挂钩深度阈值 (2026-08-10 用户要求): 有效挂单距离 =
    # max(max_entry_distance_pct, entry_distance_atr_mult × ATR%)。
    # 涨跌幅榜极端波动币 ATR 大, 固定 1.5% 会把大波动机会全部滤掉
    # ("找不到好行情"根因之二)。0=关闭挂钩(退化为固定阈值)。
    entry_distance_atr_mult: float = 4.0,
    # 深挂 conditional 触发单上限 (2026-08-10 用户要求): 挂单距现价超过有效
    # 阈值但 ≤ 该上限时, 不拒绝 — 改挂触发单(价格先走到距回补位一个阈值
    # 窗口处触发, 触发后才挂限价进场), 避免提前深挂空转也不错过深回补。
    # 超过该上限仍拒绝(防挂永不触发的单)。0=关闭 conditional 改用旧拒绝逻辑。
    max_conditional_distance_pct: float = 15.0,
    # 最低盈亏比: 止盈距离 < SL距离 × min_risk_reward 时，把止盈推远到
    # 该盈亏比(如 2.5 = 至少 1:2.5)，满足"止盈太少"诉求。
    min_risk_reward: float = 0.0,
    # FVG 新鲜度上限(根): 距最新 K 线超过该根数的 FVG 视为过期直接丢弃。
    # 修复: 旧 FVG 缺口远离现价（实测偏离 148%），生成信号难以成交且污染多通道。
    max_fvg_age_bars: int = 24,
    # FVG ATR 分级下限: 缺口宽度 / ATR(14) 低于该值视为 C 级弱缺口直接丢弃。
    # 修复(5400缺口研究/2026-08-07): A级(≥1×ATR)持仓率53%/+0.48R期望, B级(≥0.5×ATR)
    # ~32%/~0EV, C级(<0.5×ATR)仅20%/-0.34R 是负期望陷阱。0=关闭分级。
    min_fvg_atr_ratio: float = 0.0,
    # FVG Hunter 硬门禁 (2026-08-08): 只吃确定性极端行情 —
    # 横盘折磨行情(低 ADX 无趋势 或 低 ATR 波动太小)直接否决入场。
    # 行业标准: ADX≥25 趋势市(Wilder 1978, 与 adx_trend_threshold 同值);
    # ATR%≥min 保证单根 K 线平均振幅够大。0=关闭门禁。
    extreme_move_min_adx: float = 0.0,
    extreme_move_min_atr_pct: float = 0.0,
    # 职业交易标准 (ATR 动态止损): 止损距离不得小于 ATR(14)×atr_stop_multiplier，
    # 异常波动币自动放宽，避免固定百分比止损被正常噪音扫掉
    # (SAHARA 1.08% 止损 9x 杠杆被扫 -15.24 的教训)。
    atr_period: int = 14,
    atr_stop_multiplier: float = 2.0,
    # 窄止损拒绝: 止损距离 < ATR×atr_reject_ratio 时直接拒绝信号
    # (止损在噪音带内必被扫，职业标准: 止损必须超越 1.5x~2.5x ATR)。
    atr_reject_ratio: float = 0.8,
    # 杠杆联动修正 (职业标准): 杠杆 ≤ leverage_stop_budget_pct / 止损距离%，
    # 封顶 max_leverage。止损越窄杠杆越低 (5x→4%止损, 10x→2.5%, 20x→1.2%)。
    leverage_stop_budget_pct: float = 2.5,
    swing_lookback_bars: int = 24,
    pullback_lookback: int = 8,
    max_tp_distance_pct: float = 25.0,
    long_short_ratio: Optional[float] = None,
    tech_params: Optional[dict] = None,
    alpha158_enabled: bool = True,
    spread_pct: float = 0.0,
    max_spread_pct: float = 0.5,  # L-3: 可通过 strategy config max_spread_pct 覆盖
    regime: str = "NEUTRAL",  # 修复 P2-5: 体制感知评分
    candles: Optional[List[Candle]] = None,  # 用于确认蜡烛/成交量/Degree 检查
) -> Optional[Signal]:
    """从 FVG 生成交易信号。

    Args:
        inst_id: 合约 ID
        fvg: 检测到的 FVG
        current_price: 当前价格
        entry_depth_pct: 入场深度（进入 FVG 的百分比，0.15 = 15%）
        fvg_target_pct: 止盈目标（FVG 宽度的百分比，0.50 = 50%）
        stop_buffer_pct: 止损缓冲（FVG 边界外侧的百分比，0.15 = 15%）
        max_leverage: 最大杠杆
        funding_rate: 当前资金费率
        max_funding_rate_abs: 最大允许资金费率绝对值
        spread_pct: 当前买卖价差 (%)
        max_spread_pct: 最大允许价差 (%)
        regime: 市场体制 (FUSED/DIVERGENT/NEUTRAL)，用于体制感知评分

    Returns:
        Signal 或 None（被过滤）
    """
    fvg_width = abs(fvg.top - fvg.bottom)

    # ---- ATR 挂钩深度阈值 (2026-08-10 用户要求) ----
    # 有效挂单距离 = max(max_entry_distance_pct, entry_distance_atr_mult × ATR%)。
    # 涨跌幅榜极端波动币 ATR 大(±10%+ 波动), 固定 1.5% 硬阈值会把大波动
    # 机会全部滤掉。ATR 数据不足时退化为固定阈值。
    _atr_dist = _compute_atr_wilder(candles, atr_period)
    _atr_pct = (_atr_dist / current_price * 100.0) if (_atr_dist > 0 and current_price > 0) else 0.0
    _eff_entry_dist = max_entry_distance_pct
    if _atr_pct > 0 and entry_distance_atr_mult > 0:
        _eff_entry_dist = max(max_entry_distance_pct, entry_distance_atr_mult * _atr_pct)
    if _eff_entry_dist != max_entry_distance_pct:
        logger.debug(
            f"[EntryLimit] {inst_id} ATR挂钩阈值: ATR={_atr_pct:.2f}% × "
            f"{entry_distance_atr_mult:.0f} = {_eff_entry_dist:.2f}% "
            f"(基础 {max_entry_distance_pct}%)")
    use_conditional_entry = False   # 深挂触发单标记
    entry_trigger_px = 0.0          # conditional 触发价

    # ---- 过滤器 0: FVG 新鲜度 ----
    # 修复: 旧 FVG 缺口远离现价（BEAT entry 偏离现价 148%），不仅难成交
    # 还污染多通道价格行为分析。距最新 K 线超过 max_fvg_age_bars 根直接丢弃。
    _fresh, _fvg_age = _is_fvg_fresh(fvg, candles, max_fvg_age_bars)
    if not _fresh:
        logger.info(
            f"[Freshness] {inst_id} {fvg.timeframe} FVG 距最新 K 线 "
            f"{_fvg_age} 根 > {max_fvg_age_bars}，已过期，丢弃信号")
        return None

    # ---- 过滤器 0.5: FVG ATR 分级 ----
    # 修复(5400缺口研究/2026-08-07): 孤立 C 级缺口(宽度<0.5×ATR)持仓率仅20%
    # 是负期望陷阱(-0.34R); A 级(≥1×ATR)持仓率53%(+0.48R)。ATR 数据不足时
    # 跳过分级(放行, 防新币/数据缺失阻塞)。
    if min_fvg_atr_ratio > 0:
        _atr_grade = _compute_atr_wilder(candles, atr_period)
        if _atr_grade > 0:
            _atr_ratio = fvg_width / _atr_grade
            if _atr_ratio < min_fvg_atr_ratio:
                logger.info(
                    f"[ATRGrade] {inst_id} {fvg.timeframe} FVG 宽度 {fvg_width:.5f} / "
                    f"ATR {_atr_grade:.5f} = {_atr_ratio:.2f} < {min_fvg_atr_ratio}，"
                    f"C级弱缺口负期望，丢弃信号")
                return None
        else:
            logger.debug(f"[ATRGrade] {inst_id} ATR 数据不足({len(candles) if candles else 0}根)，跳过分级")

    # ---- 过滤器 0.75: FVG Hunter 硬门禁 (只吃确定性极端行情) ----
    # 2026-08-08 横盘监控复盘: ADA 4h 横盘 0.1987~0.2001 (±0.35%) 入场后
    # 空耗 4 小时无方向。横盘折磨行情 FVG 回补易被噪音扫掉, 直接否决入场。
    # ADX≥25 趋势市 + ATR%≥min 波动(上涨下跌都很大) 才放行; 数据不足 fail-open。
    if extreme_move_min_adx > 0 or extreme_move_min_atr_pct > 0:
        _em_reason = _extreme_move_reject_reason(
            candles, current_price, atr_period,
            extreme_move_min_adx, extreme_move_min_atr_pct)
        if _em_reason:
            logger.info(
                f"[ExtremeMove] {inst_id} {fvg.timeframe} 横盘/低波动拒绝: "
                f"{_em_reason} (FVG Hunter 只吃确定性极端行情)")
            return None

    # ---- 过滤器 1: 资金费率 ----
    # 修复 C4: 方向感知过滤 — 做多只过滤正费率（你支付），做空只过滤负费率（你支付）
    if funding_rate is not None and max_funding_rate_abs > 0:
        if fvg.direction == "long" and funding_rate > max_funding_rate_abs:
            logger.debug(f"[Filter] {inst_id} 做多 + 高正费率 {funding_rate:.4%} > {max_funding_rate_abs:.4%}")
            return None
        if fvg.direction == "short" and funding_rate < -max_funding_rate_abs:
            logger.debug(f"[Filter] {inst_id} 做空 + 高负费率 {funding_rate:.4%} < -{max_funding_rate_abs:.4%}")
            return None

    # ---- 过滤器 2: 买卖价差 ----
    if spread_pct > max_spread_pct:
        logger.debug(f"[Filter] {inst_id} spread {spread_pct:.2%} "
                     f"exceeds limit {max_spread_pct:.2%}")
        return None

    # ---- 过滤器 3: 价格是否已离开 FVG 区域 ----
    if fvg.direction == "long":
        # 做多: 当前价格应该 >= FVG 顶部（价格在 FVG 上方，等待回踩）
        # 如果价格已经在 FVG 内部或下方，需要确认
        if current_price < fvg.bottom:
            # 价格已经跌破 FVG 下沿，FVG 可能已失效
            return None
        _deep_entry = False
        _swing_ref = None
        _liq_pct = max(liquidity_extension_min_pct, min(5.0, liquidity_extension_pct))
        if current_price < fvg.top:
            # 价格在 FVG 区域内，可以直接入场
            entry_price = current_price
        else:
            # 修复: 流动性猎手 — 价格在 FVG 上方，限价单挂在前低下方
            # (外扩 liquidity_extension_pct%)。价格扫穿前低流动性后回补成交，
            # 避免在前低/前高这类流动性密集区直接接单被继续扫穿。
            _swing_ref = None
            if candles and len(candles) >= 3:
                _lows = [c.low for c in candles[-swing_lookback_bars:] if c.low > 0]
                if _lows:
                    _swing_ref = min(_lows)
            if _swing_ref and _swing_ref > 0:
                entry_price = _swing_ref * (1 - _liq_pct / 100.0)
                _deep_entry = True
            else:
                # 无前低数据时回退原 FVG 回补位
                entry_price = max(fvg.top - fvg_width * entry_depth_pct, fvg.bottom)

        # 挂单距离限制: 深挂偏离现价过大时回退 FVG 回补位（防远离现价空转）
        # 阈值随 ATR 挂钩放大 (_eff_entry_dist): 极端波动币允许更深挂
        if _deep_entry and _eff_entry_dist > 0 and current_price > 0:
            _dev = (current_price - entry_price) / current_price * 100.0
            if _dev > _eff_entry_dist:
                logger.info(
                    f"[EntryLimit] {inst_id} long 深挂偏离 {_dev:.2f}% > "
                    f"{_eff_entry_dist:.2f}%，回退 FVG 回补位")
                _deep_entry = False
                entry_price = max(fvg.top - fvg_width * entry_depth_pct, fvg.bottom)

        # 止盈: FVG 回补目标 + 额外溢价（做多期望价格反弹超越 FVG 顶部）
        take_profit = fvg.top + fvg_width * fvg_target_pct
        # 止损: FVG 下沿外侧；挂单加深(前低下方)时原 FVG 止损可能高于
        # 入场价（方向错误），用前低下方 2x 外扩作保底止损
        _sl_fvg = fvg.bottom - fvg_width * stop_buffer_pct
        if _deep_entry:
            stop_loss = min(_sl_fvg, _swing_ref * (1 - 2 * _liq_pct / 100.0))
        else:
            stop_loss = _sl_fvg

        # ---- ATR 止损下限 (职业标准): SL 距 entry ≥ ATR×multiplier ----
        # 异常波动币的 FVG 结构止损可能仅 1-2% (SAHARA 1.08% 被扫 -15.24)，
        # 远小于正常波动。止损必须放在 ATR(14)×2.0 波动带之外。
        _atr_val = _compute_atr_wilder(candles, atr_period)
        if _atr_val > 0:
            _sl_min = entry_price - _atr_val * atr_stop_multiplier
            if stop_loss > _sl_min:
                logger.info(
                    f"[ATRStop] {inst_id} long 止损距 {stop_loss:.6g} 过近 "
                    f"(ATR={_atr_val:.6g}×{atr_stop_multiplier:.1f})，"
                    f"放宽到 {_sl_min:.6g}")
                stop_loss = _sl_min
            # 窄止损拒绝: 止损距离 < ATR×atr_reject_ratio → 噪音带内必被扫
            _sl_dist = entry_price - stop_loss
            if _sl_dist < _atr_val * atr_reject_ratio:
                logger.info(
                    f"[ATRStop] {inst_id} long 止损距离 {_sl_dist:.6g} "
                    f"< ATR×{atr_reject_ratio:.1f}={_atr_val * atr_reject_ratio:.6g}，"
                    f"止损在噪音带内，拒绝信号")
                return None

        # 止盈盈亏比下限: TP 距离 < SL距离 × min_risk_reward 时推远止盈
        _rr = 0.0
        if min_risk_reward > 0 and stop_loss < entry_price:
            _rr = (take_profit - entry_price) / (entry_price - stop_loss)
            if _rr < min_risk_reward:
                take_profit = entry_price + (entry_price - stop_loss) * min_risk_reward
                _rr = min_risk_reward

        _entry_dev = abs(entry_price - current_price) / current_price * 100.0 if current_price > 0 else 0.0
        logger.info(
            f"[EntryLog] {inst_id} long entry={entry_price:.6g} cur={current_price:.6g} "
            f"dev={_entry_dev:.2f}% deep={_deep_entry} RR={_rr:.2f} "
            f"tp={take_profit:.6g} sl={stop_loss:.6g}")

        position_side = "long"

    else:  # short
        # 做空: 当前价格应该 <= FVG 底部（价格在 FVG 下方，等待反弹）
        if current_price > fvg.top:
            # 价格已经突破 FVG 上沿，FVG 可能已失效
            return None
        _deep_entry = False
        _swing_ref = None
        _liq_pct = max(liquidity_extension_min_pct, min(5.0, liquidity_extension_pct))
        if current_price > fvg.bottom:
            # 价格在 FVG 区域内，可以直接入场
            entry_price = current_price
        else:
            # 修复: 流动性猎手 — 价格在 FVG 下方，限价单挂在前高上方
            # (外扩 liquidity_extension_pct%)。价格扫穿前高流动性后回补成交。
            _swing_ref = None
            if candles and len(candles) >= 3:
                _highs = [c.high for c in candles[-swing_lookback_bars:] if c.high > 0]
                if _highs:
                    _swing_ref = max(_highs)
            if _swing_ref and _swing_ref > 0:
                entry_price = _swing_ref * (1 + _liq_pct / 100.0)
                _deep_entry = True
            else:
                # 无前高数据时回退原 FVG 回补位
                entry_price = min(fvg.bottom + fvg_width * entry_depth_pct, fvg.top)

        # 挂单距离限制: 深挂偏离现价过大时回退 FVG 回补位（防远离现价空转）
        # 阈值随 ATR 挂钩放大 (_eff_entry_dist): 极端波动币允许更深挂
        if _deep_entry and _eff_entry_dist > 0 and current_price > 0:
            _dev = (entry_price - current_price) / current_price * 100.0
            if _dev > _eff_entry_dist:
                logger.info(
                    f"[EntryLimit] {inst_id} short 深挂偏离 {_dev:.2f}% > "
                    f"{_eff_entry_dist:.2f}%，回退 FVG 回补位")
                _deep_entry = False
                entry_price = min(fvg.bottom + fvg_width * entry_depth_pct, fvg.top)

        # 止盈: FVG 回补目标 + 额外折价（做空期望价格跌破 FVG 底部）
        take_profit = fvg.bottom - fvg_width * fvg_target_pct
        # 止损: FVG 上沿外侧；挂单加深(前高上方)时原 FVG 止损可能低于
        # 入场价（方向错误），用前高上方 2x 外扩作保底止损
        _sl_fvg = fvg.top + fvg_width * stop_buffer_pct
        if _deep_entry:
            stop_loss = max(_sl_fvg, _swing_ref * (1 + 2 * _liq_pct / 100.0))
        else:
            stop_loss = _sl_fvg

        # ---- ATR 止损下限 (职业标准): SL 距 entry ≥ ATR×multiplier ----
        _atr_val = _compute_atr_wilder(candles, atr_period)
        if _atr_val > 0:
            _sl_min = entry_price + _atr_val * atr_stop_multiplier
            if stop_loss < _sl_min:
                logger.info(
                    f"[ATRStop] {inst_id} short 止损距 {stop_loss:.6g} 过近 "
                    f"(ATR={_atr_val:.6g}×{atr_stop_multiplier:.1f})，"
                    f"放宽到 {_sl_min:.6g}")
                stop_loss = _sl_min
            # 窄止损拒绝: 止损距离 < ATR×atr_reject_ratio → 噪音带内必被扫
            _sl_dist = stop_loss - entry_price
            if _sl_dist < _atr_val * atr_reject_ratio:
                logger.info(
                    f"[ATRStop] {inst_id} short 止损距离 {_sl_dist:.6g} "
                    f"< ATR×{atr_reject_ratio:.1f}={_atr_val * atr_reject_ratio:.6g}，"
                    f"止损在噪音带内，拒绝信号")
                return None

        # 止盈盈亏比下限: TP 距离 < SL距离 × min_risk_reward 时推远止盈
        _rr = 0.0
        if min_risk_reward > 0 and stop_loss > entry_price:
            _rr = (entry_price - take_profit) / (stop_loss - entry_price)
            if _rr < min_risk_reward:
                take_profit = entry_price - (stop_loss - entry_price) * min_risk_reward
                _rr = min_risk_reward

        _entry_dev = abs(entry_price - current_price) / current_price * 100.0 if current_price > 0 else 0.0
        logger.info(
            f"[EntryLog] {inst_id} short entry={entry_price:.6g} cur={current_price:.6g} "
            f"dev={_entry_dev:.2f}% deep={_deep_entry} RR={_rr:.2f} "
            f"tp={take_profit:.6g} sl={stop_loss:.6g}")

        position_side = "short"

    # ---- 最终挂单距离上限检查 ----
    # 深挂路径回退到 FVG 回补位后，大 FVG 的回补位本身仍可能远离现价
    # (实测 BICO 回退后仍偏离 3-19%)，同样难成交。无论哪条路径，最终
    # 挂单价距现价超过有效阈值 (_eff_entry_dist, ATR 挂钩) 时:
    #   - 偏离 ≤ max_conditional_distance_pct → 改用 conditional 触发单
    #     (2026-08-10 用户要求): 价格先走到距回补位一个阈值窗口处触发,
    #     触发后才挂限价进场 — 不提前深挂空转, 也不错过深回补机会
    #   - 偏离 > max_conditional_distance_pct → 拒绝(防挂永不触发的单)
    if _eff_entry_dist > 0 and current_price > 0:
        _final_dev = abs(entry_price - current_price) / current_price * 100.0
        if _final_dev > _eff_entry_dist:
            if max_conditional_distance_pct > 0 and _final_dev <= max_conditional_distance_pct:
                use_conditional_entry = True
                if position_side == "long":
                    entry_trigger_px = entry_price * (1 + _eff_entry_dist / 100.0)
                    entry_trigger_px = min(entry_trigger_px, current_price)
                else:
                    entry_trigger_px = entry_price * (1 - _eff_entry_dist / 100.0)
                    entry_trigger_px = max(entry_trigger_px, current_price)
                logger.info(
                    f"[EntryLimit] {inst_id} {position_side} 深挂偏离 "
                    f"{_final_dev:.2f}% > {_eff_entry_dist:.2f}%，改用 conditional "
                    f"触发单 (trigger={entry_trigger_px:.6g} entry={entry_price:.6g})")
            else:
                logger.info(
                    f"[EntryLimit] {inst_id} {position_side} 挂单距现价 "
                    f"{_final_dev:.2f}% > 有效阈值 {_eff_entry_dist:.2f}% "
                    f"(conditional 上限 {max_conditional_distance_pct}%)，拒绝挂单")
                return None

    # ---- 止盈目标距离合理性检查（修复 MMT 巨型缺口 TP 形同虚设）----
    # 异常波动币的巨型缺口会把 TP 算到远离 entry 的不现实位置
    # （实测 MMT: entry=0.4931 TP=0.1993，偏离 -60%，止盈永远不会触发
    #   = 只有止损没有止盈，且白挂单空转）。TP 距离超过阈值直接否决。
    if entry_price > 0:
        if position_side == "long":
            _tp_dist = (take_profit - entry_price) / entry_price
        else:
            _tp_dist = (entry_price - take_profit) / entry_price
        if _tp_dist > max_tp_distance_pct / 100.0:
            logger.info(
                f"[TPCheck] {inst_id} TP距离 {_tp_dist:.1%} > "
                f"{max_tp_distance_pct:.0f}%，巨型缺口止盈不现实，否决信号"
            )
            return None

    # ---- 评分 ----
    score = _calculate_signal_score(
        fvg=fvg,
        current_price=current_price,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        regime=regime,  # 修复 P2-5: 体制感知
    )

    # ---- 资金费率顺向加分（费率收割顺风）----
    # 做空吃正费率 / 做多吃负费率 = 每 8h 确定性收资金费，接近 100% 胜率的
    # 顺风策略（需止盈止损兜底反向行情）。适度费率 ∈ [min_abs, max_abs] 加分；
    # 过热费率（>max_abs）说明市场拥挤、逆着强动量，不加分防接刀。
    if funding_rate is not None and funding_confluence_min_abs > 0:
        _fr_same_dir = (
            (fvg.direction == "short" and funding_rate > 0)
            or (fvg.direction == "long" and funding_rate < 0)
        )
        _fr_abs = abs(funding_rate)
        if (_fr_same_dir
                and funding_confluence_min_abs <= _fr_abs <= funding_confluence_max_abs):
            score *= 1.05
            logger.debug(f"[Funding] {inst_id} 费率{funding_rate:+.4%}顺向，"
                         f"score → {score:.3f}")

    # ---- Improvement 1: FVG 入场前确认蜡烛 (P1-2a) ----
    if candles is not None and fvg.fvg_index >= 0:
        if not _check_confirmation_candle(
            candles, fvg.top, fvg.bottom, fvg.direction, current_price
        ):
            score *= 0.7  # 无确认蜡烛，30% 惩罚
            logger.debug(f"[Confirmation] {inst_id} no confirmation candle, score reduced to {score:.3f}")

    # ---- Improvement 2: FVG 成交量验证 (P1-2b) ----
    if candles is not None and fvg.fvg_index >= 0:
        vol_ok, vol_ratio = _check_volume_confirmation(candles, fvg.fvg_index)
        if not vol_ok:
            score *= 0.5  # 无量回补，50% 惩罚
            logger.debug(f"[Volume] {inst_id} low volume fill ratio={vol_ratio:.2f}, score reduced to {score:.3f}")

    # ---- Improvement 3: FVG Degree 量化过滤 (P2-3) ----
    if candles is not None and fvg.fvg_index >= 0:
        fvg_degree = _compute_fvg_degree(candles, fvg.fvg_index)
        if fvg_degree is not None and fvg_degree > 0.0004:
            # 高度 FVG（噪音），大幅降低置信度
            score *= 0.5
            logger.debug(f"[Degree] {inst_id} high-degree FVG degree={fvg_degree:.6f}, score reduced to {score:.3f}")
        elif fvg_degree is not None and fvg_degree <= 0.00015:
            # 低度 FVG（高质量），提升置信度
            score *= 1.15
            logger.debug(f"[Degree] {inst_id} low-degree FVG degree={fvg_degree:.6f}, score boosted to {score:.3f}")

    # ---- Improvement 4: 回踩走势状态检查（挂单前回溯近期K线状态与走势）----
    # 用户要求: 挂单前回溯前面若干根K线的状态和价格走势，区分"健康回踩"
    # 与"破位下跌"，避免在高位出现看跌形态（长上影/放量阴线）时盲目做多。
    if candles is not None:
        _pb_ok, _pb_factor = _check_pullback_state(
            candles, entry_price, current_price, position_side,
            lookback=pullback_lookback,
        )
        if not _pb_ok:
            logger.info(f"[Pullback] {inst_id} 回踩状态否决挂单 "
                        f"({position_side} entry={entry_price} cur={current_price})")
            return None
        if _pb_factor != 1.0:
            score *= _pb_factor
            logger.debug(f"[Pullback] {inst_id} 走势因子 {_pb_factor:.2f}，score → {score:.3f}")

    # ---- Improvement 5: 综合技术参考（走势/布林带/RSI/ADX/VWAP/多空持仓比）----
    # 混合模式: 极端逆势否决 + 一般背离降分 + 共振加分。
    if candles is not None:
        _tech_ok, _tech_factor, _tech_detail = _check_technical_state(
            candles, current_price, position_side,
            long_short_ratio=long_short_ratio,
            entry_price=entry_price,
            **(tech_params or {}),
        )
        if not _tech_ok:
            logger.info(f"[Technical] {inst_id} 技术状态否决挂单: {_tech_detail}")
            return None
        if _tech_factor != 1.0:
            score *= _tech_factor
            logger.debug(f"[Technical] {inst_id} {_tech_detail} "
                         f"factor={_tech_factor:.2f}，score → {score:.3f}")

    # ---- Improvement 7: Qlib Alpha158 风格因子方向一致性 ----
    # 微软 Qlib 标准化因子集子集（动量/均线排列/趋势斜率/波动收缩），
    # 与现有技术检查互补，为信号评分提供因子学维度的交叉验证。
    if candles is not None and alpha158_enabled:
        _af_factor, _af_detail = _check_alpha_confluence(candles, current_price, position_side)
        if _af_factor != 1.0:
            score *= _af_factor
            logger.debug(f"[Alpha] {inst_id} {_af_detail} "
                         f"factor={_af_factor:.2f}，score → {score:.3f}")

    # ---- 杠杆建议 (职业标准: 杠杆与止损距离反向联动) ----
    # 行业参考 (XeroGravity): 5x→4%止损, 10x→2.5%, 20x→1.2%。
    # 公式: leverage = min(max_leverage, max(1, leverage_stop_budget_pct / 止损距离%))
    # 止损 1% 时最高 2.5x, 止损 4% 时最高 ~6x (预算 25 / 4)。封顶 max_leverage。
    # 修复: 原 LEVERAGE_RISK_FACTOR=10 导致止损 1% 仍给 10x，杠杆过高。
    _stop_pct = 0.0
    if fvg.direction == "long":
        _stop_pct = (entry_price - stop_loss) / entry_price
    else:
        _stop_pct = (stop_loss - entry_price) / entry_price

    if entry_price <= 0 or _stop_pct <= 0:
        suggested_leverage = 1
    else:
        # 防止极小值导致杠杆过大
        _stop_pct = max(_stop_pct, 0.001)
        _budget = max(1.0, float(leverage_stop_budget_pct or 2.5))
        suggested_leverage = min(
            max_leverage,
            max(1, int(_budget / (_stop_pct * 100.0)))
        )

    # Bug L-11: 评分可能溢出 (0,1) — 加 clamp 保护
    score = min(max(score, 0.0), 1.0)

    return Signal(
        inst_id=inst_id,
        fvg=fvg,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        leverage=suggested_leverage,
        position_side=position_side,
        score=score,
        reason=f"FVG {fvg.direction} {fvg.timeframe} "
               f"width={fvg.width_pct:.2f}% "
               f"abnormal={fvg.is_abnormal} sigma={fvg.sigma:.1f}",
        use_conditional_entry=use_conditional_entry,
        entry_trigger_px=entry_trigger_px,
    )


def _calculate_signal_score(
    fvg: FVG,
    current_price: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    regime: str = "NEUTRAL",  # 修复 P2-5: 体制感知评分
) -> float:
    """计算信号评分 (0-1)。

    修复 P2-5: 加入市场体制调整。
      - FUSED (融合): 多周期共振，信号更可靠，+10%
      - DIVERGENT (背离): 多周期矛盾，信号可靠性降低，-20%
      - NEUTRAL: 不调整
    顶级交易员共识: 同一 FVG 结构在不同体制下回补率差异巨大。
    """
    score = 0.2  # 基础分 (M-5: 从 0.50 降至 0.20，扩大区分度范围 0.16~1.0)

    # FVG 宽度加分 (越宽越好, 上限 0.15)
    width_bonus = min(fvg.width_pct / 10.0, 0.15)
    score += width_bonus

    # 异常波动加分 (上限 0.15)
    if fvg.is_abnormal:
        sigma_bonus = min(fvg.sigma / 20.0, 0.10)
        vol_bonus = min(fvg.volume_ratio / 20.0, 0.05)
        score += sigma_bonus + vol_bonus

    # 入场距离 FVG 边界越近越好 (上限 0.10)
    if fvg.direction == "long":
        entry_proximity = (entry_price - fvg.top) / fvg.top if fvg.top > 0 else 0
    else:
        entry_proximity = (fvg.bottom - entry_price) / fvg.bottom if fvg.bottom > 0 else 0
    proximity_bonus = max(0, 0.10 - abs(entry_proximity) * 2)
    score += proximity_bonus

    # RR 比加分 (上限 0.10) — 使用实际止盈价计算，与 generate_signal 一致
    if fvg.direction == "long":
        reward = take_profit - entry_price    # 实际止盈到入场价距离
        risk = entry_price - stop_loss
    else:
        reward = entry_price - take_profit    # 入场价到实际止盈距离
        risk = stop_loss - entry_price
    if risk > 0:
        rr = reward / risk
        rr_bonus = min(rr / 10.0, 0.10)
        score += rr_bonus

    # 修复 P2-5: 体制调整 — FUSED 加分，DIVERGENT 减分
    if regime == "FUSED":
        score *= 1.10  # 多周期共振，信号更可靠
    elif regime == "DIVERGENT":
        score *= 0.80  # 多周期矛盾，信号可靠性降低

    return min(score, 1.0)


def _compute_atr_wilder(candles: Optional[List[Candle]], period: int = 14) -> float:
    """计算 1H K 线 ATR (Average True Range, Wilder's 平滑)。

    职业交易标准 (XeroGravity/Binance): 止损必须放在"正常波动带"之外，
    用 ATR(14)×2.0 度量正常波动，避免固定百分比止损被噪音扫掉。
    数据不足 (<period+1 根) 或异常时返回 0.0（调用方回退不阻塞）。

    Returns:
        ATR 绝对值(价格单位)；数据不足返回 0.0
    """
    if not candles or len(candles) < period + 1:
        return 0.0
    tr_list = []
    try:
        for i in range(1, len(candles)):
            h = float(candles[i].high)
            l = float(candles[i].low)
            prev_c = float(candles[i - 1].close)
            if h <= 0 or l <= 0 or prev_c <= 0:
                continue
            tr_list.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    except (TypeError, ValueError, AttributeError):
        return 0.0
    if len(tr_list) < period:
        return 0.0
    # 初始简单平均 + Wilder's 平滑 (与 trade_analyzer.compute_atr 一致)
    atr = float(np.mean(tr_list[:period]))
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
    return atr


# ---------------------------------------------------------------------------
# 扫描多个时间周期
# ---------------------------------------------------------------------------

def _detected_to_legacy(d: FVGDetected, candles: List[Candle]) -> FVG:
    """将独立检测器输出的 FVGDetected 转回 strategy.FVG。

    独立检测器（fvg_detector.py）作为第一层检测入口，输出新结构
    FVGDetected；这里转回 FVG 以复用现有的确认链/技术检查/评分逻辑。
    """
    if d.start_idx + 1 < len(candles):
        impulse = candles[d.start_idx + 1]
    else:
        impulse = candles[d.end_idx]
    return FVG(
        direction="long" if d.direction == "bullish" else "short",
        top=d.gap_high,
        bottom=d.gap_low,
        width_pct=d.width_pct,
        candle_ts=d.formation_ts,
        timeframe=d.timeframe,
        impulse_candle=impulse,
        fvg_index=d.start_idx + 1,
        is_abnormal=d.is_abnormal,
        sigma=d.sigma,
        volume_ratio=d.volume_ratio,
    )


def scan_fvg_all_timeframes(
    inst_id: str,
    candles_by_tf: Dict[str, List[Candle]],
    current_price: float,
    config: dict,
    funding_rate: Optional[float] = None,
    spread_pct: float = 0.0,
    regime: str = "NEUTRAL",  # 修复 P2-5: 体制感知
    long_short_ratio: Optional[float] = None,  # 多空持仓比(综合技术参考)
) -> List[Signal]:
    """在多个时间周期上扫描 FVG 并生成信号。

    H-4: 已知限制 — 当前各时间周期独立扫描，无真正的跨周期融合验证。
    未来版本将实现 FVG 层级对齐（如 4H FVG 内部包含 1H FVG 时增强信号）。

    当前简易融合: 若 1H 和 4H 同时存在同方向 FVG，信号评分 +10%。

    Args:
        inst_id: 合约 ID
        candles_by_tf: {"1H": [...], "4H": [...]}
        current_price: 当前价格
        config: 完整配置
        funding_rate: 资金费率
        spread_pct: 买卖价差
        regime: 市场体制 (FUSED/DIVERGENT/NEUTRAL)

    Returns:
        信号列表（按评分降序）
    """
    signals = []
    strategy_cfg = config.get("strategy", {})
    # 纸面模式专用覆盖: dry_run + paper.enabled 时，用 paper.liquidity_extension_pct
    # 缩短流动性猎手挂单距离（如 0.8% 贴近现价），加速限价回补成交，便于测试
    # 完整交易闭环。实盘保持 strategy.liquidity_extension_pct(3-5%) 策略要求不变。
    _paper_cfg = config.get("paper", {}) if isinstance(config, dict) else {}
    _liq_ext = strategy_cfg.get("liquidity_extension_pct", 4.0)
    _liq_min = 3.0
    if (
        isinstance(config, dict)
        and config.get("agent", {}).get("dry_run")
        and _paper_cfg.get("enabled", False)
        and "liquidity_extension_pct" in _paper_cfg
    ):
        try:
            _liq_ext = float(_paper_cfg["liquidity_extension_pct"])
            _liq_min = float(_paper_cfg.get("liquidity_extension_min_pct", 0.5))
        except (TypeError, ValueError):
            pass
    # 独立 FVG 检测器（第一层检测入口，见 fvg_detector.py）
    _detector = FVGDetector(strategy_cfg)

    for tf in strategy_cfg.get("timeframes", []):
        candles = candles_by_tf.get(tf, [])
        if len(candles) < 3:
            continue

        if strategy_cfg.get("fvg_detector_enabled", True):
            # 独立检测器（第一层）：detect + 质量过滤
            _detected = _detector.detect({tf: candles})
            _detected = _detector.filter_by_quality(_detected, {
                "current_price": current_price,
                "funding_rate": funding_rate,
                "spread_pct": spread_pct,
            })
            fvgs = [_detected_to_legacy(d, candles) for d in _detected]
        else:
            # 兼容分支: 直接使用旧版检测函数
            min_width = strategy_cfg.get("min_fvg_width_pct", {}).get(tf, 1.5)
            lookback = strategy_cfg.get("abnormal_lookback", {}).get(tf, 50)

            fvgs = detect_fvg(
                candles,
                timeframe=tf,
                min_width_pct=min_width,
                sigma_threshold=strategy_cfg.get("abnormal_sigma", 3.0),
                volume_ratio_threshold=strategy_cfg.get("abnormal_volume_ratio", 5.0),
                lookback=lookback,
            )

        # 共用信号生成参数（普通 FVG 与 iFVG 复用同一挂单/过滤/评分链）
        _sig_kwargs = dict(
            inst_id=inst_id,
            entry_depth_pct=strategy_cfg.get("entry_depth_pct", 0.15),
            fvg_target_pct=strategy_cfg.get("fvg_target_pct", 0.50),
            stop_buffer_pct=strategy_cfg.get("stop_buffer_pct", 0.15),
            liquidity_extension_pct=_liq_ext,
            liquidity_extension_min_pct=_liq_min,
            max_entry_distance_pct=float(strategy_cfg.get("max_entry_distance_pct", 5.0)),
            entry_distance_atr_mult=float(strategy_cfg.get("entry_distance_atr_mult", 4.0)),
            max_conditional_distance_pct=float(
                strategy_cfg.get("max_conditional_distance_pct", 15.0)),
            min_risk_reward=float(strategy_cfg.get("min_risk_reward", 0.0)),
            max_fvg_age_bars=int(strategy_cfg.get("max_fvg_age_bars", 24)),
            min_fvg_atr_ratio=float(strategy_cfg.get("min_fvg_atr_ratio", 0.0)),
            extreme_move_min_adx=float(strategy_cfg.get("extreme_move_min_adx", 0.0)),
            extreme_move_min_atr_pct=float(strategy_cfg.get("extreme_move_min_atr_pct", 0.0)),
            atr_period=int(strategy_cfg.get("atr_period", 14)),
            atr_stop_multiplier=float(strategy_cfg.get("atr_stop_multiplier", 2.0)),
            atr_reject_ratio=float(strategy_cfg.get("atr_reject_ratio", 0.8)),
            leverage_stop_budget_pct=float(strategy_cfg.get("leverage_stop_budget_pct", 2.5)),
            swing_lookback_bars=strategy_cfg.get("swing_lookback_bars", 24),
            pullback_lookback=strategy_cfg.get("pullback_lookback", 8),
            max_tp_distance_pct=strategy_cfg.get("max_tp_distance_pct", 25.0),
            long_short_ratio=long_short_ratio,
            alpha158_enabled=strategy_cfg.get("alpha158_enabled", True),
            tech_params={
                "bb_period": strategy_cfg.get("bb_period", 20),
                "bb_std": strategy_cfg.get("bb_std", 2.0),
                "squeeze_threshold": strategy_cfg.get("bb_squeeze_threshold", 0.6),
                "trend_ma_period": strategy_cfg.get("trend_ma_period", 20),
                "rsi_period": strategy_cfg.get("rsi_period", 14),
                "rsi_overbought": strategy_cfg.get("rsi_overbought", 70.0),
                "rsi_oversold": strategy_cfg.get("rsi_oversold", 30.0),
                "adx_period": strategy_cfg.get("adx_period", 14),
                "adx_trend_threshold": strategy_cfg.get("adx_trend_threshold", 25.0),
                "adx_range_threshold": strategy_cfg.get("adx_range_threshold", 20.0),
                "vwap_tolerance_pct": strategy_cfg.get("vwap_tolerance_pct", 0.5),
                "divergence_lookback": strategy_cfg.get("divergence_lookback", 14),
                "lsr_strong_high": strategy_cfg.get("lsr_strong_high", 1.3),
                "lsr_strong_low": strategy_cfg.get("lsr_strong_low", 0.7),
                "bb_veto_low": strategy_cfg.get("bb_veto_low", -0.2),
                "bb_veto_high": strategy_cfg.get("bb_veto_high", 1.2),
            },
            max_leverage=config.get("risk", {}).get("max_leverage", 10),
            funding_rate=funding_rate,
            max_funding_rate_abs=strategy_cfg.get("max_funding_rate_abs", 0.01),
            funding_confluence_min_abs=strategy_cfg.get("funding_confluence_min_abs", 0.0003),
            funding_confluence_max_abs=strategy_cfg.get("funding_confluence_max_abs", 0.001),
            spread_pct=spread_pct,
            max_spread_pct=strategy_cfg.get("max_spread_pct", 0.5),
            regime=regime,  # 修复 P2-5: 体制感知
            candles=candles,  # 用于确认蜡烛/成交量/Degree 检查
        )

        for fvg in fvgs:
            signal = generate_signal(
                fvg=fvg,
                current_price=current_price,
                **_sig_kwargs,
            )
            if signal:
                signals.append(signal)

        # ---- iFVG (Inversion Fair Value Gap) 反转信号 ----
        # ICT 高级概念: 已填满的 FVG 被反向突破后极性强转（支撑↔阻力），
        # 转向同方向交易。转为等效 FVG 复用完整信号链。
        if strategy_cfg.get("ifvg_enabled", True) and fvgs:
            _ifvgs = detect_ifvg(
                candles,
                fvgs,
                max_age_bars=strategy_cfg.get("ifvg_max_age_bars", 100),
            )
            for _ifvg in _ifvgs:
                _eq = _ifvg_to_fvg(_ifvg, candles)
                if _eq is None:
                    continue
                _sig = generate_signal(
                    fvg=_eq,
                    current_price=current_price,
                    **_sig_kwargs,
                )
                if _sig:
                    _sig.reason = f"iFVG反转{_ifvg.direction} " + _sig.reason
                    signals.append(_sig)

    # H-4: 简易多周期融合 — 若 1H 和 4H 同时存在同方向 FVG，增强信号
    _has_1h_long = any(s.fvg.timeframe == "1H" and s.position_side == "long" for s in signals)
    _has_1h_short = any(s.fvg.timeframe == "1H" and s.position_side == "short" for s in signals)
    _has_4h_long = any(s.fvg.timeframe == "4H" and s.position_side == "long" for s in signals)
    _has_4h_short = any(s.fvg.timeframe == "4H" and s.position_side == "short" for s in signals)
    _fusion_long = _has_1h_long and _has_4h_long
    _fusion_short = _has_1h_short and _has_4h_short
    if _fusion_long or _fusion_short:
        for s in signals:
            if (_fusion_long and s.position_side == "long") or \
               (_fusion_short and s.position_side == "short"):
                s.score = min(s.score * 1.10, 1.0)

    # 按评分降序
    signals.sort(key=lambda s: s.score, reverse=True)
    return signals