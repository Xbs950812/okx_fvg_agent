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
    if idx < lookback:
        return False, 0.0, 1.0

    # 计算收益率（对数收益）
    returns = []
    volumes = []
    for i in range(idx - lookback, idx):
        ret = math.log(candles[i].close / candles[i].open)
        returns.append(abs(ret))
        volumes.append(candles[i].volume)

    if len(returns) < 20:
        return False, 0.0, 1.0

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret < 1e-10:
        return False, 0.0, 1.0

    current_ret = abs(math.log(candles[idx].close / candles[idx].open))
    sigma = (current_ret - mean_ret) / std_ret

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
                    is_abnormal=is_ab,
                    sigma=sigma,
                    volume_ratio=vol_ratio,
                ))

        # ---- 看跌 FVG ----
        if c0.low > c2.high:
            fvg_top = c0.low
            fvg_bottom = c2.high
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
                    is_abnormal=is_ab,
                    sigma=sigma,
                    volume_ratio=vol_ratio,
                ))

    return fvgs


# ---------------------------------------------------------------------------
# 信号生成
# ---------------------------------------------------------------------------

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
    spread_pct: float = 0.0,
    max_spread_pct: float = 0.5,
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

    Returns:
        Signal 或 None（被过滤）
    """
    fvg_width = fvg.top - fvg.bottom

    # ---- 过滤器 1: 资金费率 ----
    if funding_rate is not None and abs(funding_rate) > max_funding_rate_abs:
        logger.debug(f"[Filter] {inst_id} funding rate {funding_rate:.4%} "
                     f"exceeds limit {max_funding_rate_abs:.4%}")
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
        if current_price < fvg.top:
            # 价格在 FVG 区域内，可以直接入场
            entry_price = current_price
        else:
            # 价格在 FVG 上方，挂限价单在 FVG 上沿附近
            entry_price = fvg.top - fvg_width * entry_depth_pct

        # 止盈: FVG 回补目标 + 额外溢价（做多期望价格反弹超越 FVG 顶部）
        take_profit = fvg.top + fvg_width * fvg_target_pct
        # 止损: FVG 下沿外侧
        stop_loss = fvg.bottom - fvg_width * stop_buffer_pct

        position_side = "long"

    else:  # short
        # 做空: 当前价格应该 <= FVG 底部（价格在 FVG 下方，等待反弹）
        if current_price > fvg.top:
            # 价格已经突破 FVG 上沿，FVG 可能已失效
            return None
        if current_price > fvg.bottom:
            # 价格在 FVG 区域内，可以直接入场
            entry_price = current_price
        else:
            # 价格在 FVG 下方，挂限价单在 FVG 下沿附近
            entry_price = fvg.bottom + fvg_width * entry_depth_pct

        # 止盈: FVG 回补目标 + 额外折价（做空期望价格跌破 FVG 底部）
        take_profit = fvg.bottom - fvg_width * fvg_target_pct
        # 止损: FVG 上沿外侧
        stop_loss = fvg.top + fvg_width * stop_buffer_pct

        position_side = "short"

    # ---- 评分 ----
    score = _calculate_signal_score(
        fvg=fvg,
        current_price=current_price,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    # ---- 杠杆建议 ----
    # 基于止损距离动态调整杠杆
    if fvg.direction == "long":
        stop_distance_pct = (entry_price - stop_loss) / entry_price
    else:
        stop_distance_pct = (stop_loss - entry_price) / entry_price

    # 止损距离越大，可用杠杆越低
    suggested_leverage = min(
        max_leverage,
        max(1, int(1.0 / (stop_distance_pct * 10)))  # 止损距离 * 10 的倒数
    )

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
    )


def _calculate_signal_score(
    fvg: FVG,
    current_price: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> float:
    """计算信号评分 (0-1)。"""
    score = 0.5  # 基础分

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
        entry_proximity = (current_price - fvg.top) / fvg.top if fvg.top > 0 else 0
    else:
        entry_proximity = (fvg.bottom - current_price) / fvg.bottom if fvg.bottom > 0 else 0
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

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# 扫描多个时间周期
# ---------------------------------------------------------------------------

def scan_fvg_all_timeframes(
    inst_id: str,
    candles_by_tf: Dict[str, List[Candle]],
    current_price: float,
    config: dict,
    funding_rate: Optional[float] = None,
    spread_pct: float = 0.0,
) -> List[Signal]:
    """在多个时间周期上扫描 FVG 并生成信号。

    Args:
        inst_id: 合约 ID
        candles_by_tf: {"1H": [...], "4H": [...]}
        current_price: 当前价格
        config: 完整配置
        funding_rate: 资金费率
        spread_pct: 买卖价差

    Returns:
        信号列表（按评分降序）
    """
    signals = []
    strategy_cfg = config["strategy"]

    for tf in strategy_cfg["timeframes"]:
        candles = candles_by_tf.get(tf, [])
        if len(candles) < 3:
            continue

        min_width = strategy_cfg["min_fvg_width_pct"].get(tf, 1.5)

        fvgs = detect_fvg(
            candles,
            timeframe=tf,
            min_width_pct=min_width,
            sigma_threshold=strategy_cfg["abnormal_sigma"],
            volume_ratio_threshold=strategy_cfg["abnormal_volume_ratio"],
        )

        for fvg in fvgs:
            signal = generate_signal(
                inst_id=inst_id,
                fvg=fvg,
                current_price=current_price,
                entry_depth_pct=strategy_cfg["entry_depth_pct"],
                fvg_target_pct=strategy_cfg["fvg_target_pct"],
                stop_buffer_pct=strategy_cfg["stop_buffer_pct"],
                max_leverage=config["risk"]["max_leverage"],
                funding_rate=funding_rate,
                max_funding_rate_abs=strategy_cfg["max_funding_rate_abs"],
                spread_pct=spread_pct,
                max_spread_pct=strategy_cfg["max_spread_pct"],
            )
            if signal:
                signals.append(signal)

    # 按评分降序
    signals.sort(key=lambda s: s.score, reverse=True)
    return signals