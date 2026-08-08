"""
历史交易超详细分析模块 — 扫描 OKX 账户所有历史交易并进行深度复盘。

功能：
  1. 分页拉取所有历史账单（含已实现盈亏）
  2. 配对开仓/平仓，还原完整交易周期
  3. 拉取建仓时点多时间框架 K 线（月/周/日/4H/1H/15min）
  4. 计算技术指标：布林带、RSI、MACD、成交量分布、多空比、资金费率
  5. 分析交易成功/失败原因（以盈亏为基准）
  6. 生成 HTML 详细报告并上传云端

用法:
  python trade_analyzer.py                    # 分析最近 90 天
  python trade_analyzer.py --days 180         # 分析最近 180 天
  python trade_analyzer.py --upload           # 分析并上传云端
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np

# 时区
BEIJING_TZ = timezone(timedelta(hours=8))

logger = logging.getLogger(__name__)


# ===========================================================================
# 数据类
# ===========================================================================

@dataclass
class CandleSnapshot:
    """单根 K 线快照（建仓时刻附近）。"""
    timestamp: int          # unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float           # 成交量（张数）


@dataclass
class TimeframeAnalysis:
    """单个时间框架的技术指标分析。"""
    timeframe: str                              # "1M" | "1W" | "1D" | "4H" | "1H" | "15m"
    # 建仓时刻蜡烛
    entry_candle: Optional[CandleSnapshot] = None
    # 布林带 (20周期)
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width_pct: float = 0.0                   # 带宽百分比
    bb_position_pct: float = 0.0                # 价格在带中的位置 (0=下轨, 1=上轨)
    # RSI (14周期)
    rsi: Optional[float] = None
    rsi_zone: str = "中性"                       # "超买" | "超卖" | "偏多" | "偏空" | "中性"
    # MACD
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_trend: str = "中性"                     # "金叉" | "死叉" | "多头" | "空头" | "中性"
    # 成交量
    volume_ratio: float = 1.0                    # 当前量 / 均量
    volume_zone: str = "正常"                    # "放量" | "缩量" | "正常"
    # 趋势
    ma_20: float = 0.0
    ma_50: float = 0.0
    ma_200: float = 0.0
    price_vs_ma20_pct: float = 0.0
    trend_direction: str = "横盘"                # "上升" | "下降" | "横盘"
    # ATR (14周期)
    atr: float = 0.0
    atr_pct: float = 0.0                         # ATR 占价格百分比


@dataclass
class TradeAnalysis:
    """单笔交易的完整分析。"""
    # 基本信息
    trade_id: str = ""
    inst_id: str = ""                            # 合约 ID
    symbol: str = ""                             # 币种简称
    direction: str = ""                          # "long" | "short"
    # 时间
    entry_time: float = 0.0                      # 建仓时间戳 (秒)
    entry_time_str: str = ""                     # 格式化时间字符串
    exit_time: float = 0.0                       # 平仓时间戳
    exit_time_str: str = ""
    holding_hours: float = 0.0
    # 价格
    entry_price: float = 0.0
    exit_price: float = 0.0
    mark_price_entry: float = 0.0                # 建仓时标记价格
    # 数量
    quantity: float = 0.0                        # 合约张数
    leverage: int = 1
    # 盈亏
    pnl: float = 0.0                             # 已实现盈亏 (USDT)
    pnl_pct: float = 0.0                         # 收益率
    fee: float = 0.0                             # 手续费
    is_win: bool = False
    is_loss: bool = False
    is_breakeven: bool = False
    # 多时间框架分析
    timeframe_analysis: Dict[str, TimeframeAnalysis] = field(default_factory=dict)
    # 市场环境
    funding_rate_entry: float = 0.0              # 建仓时资金费率
    long_short_ratio: float = 0.0                # 建仓时多空比
    open_interest: float = 0.0                   # 建仓时持仓量
    spread_pct: float = 0.0                      # 建仓时买卖价差
    # 分析结论
    success_reasons: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    key_observations: List[str] = field(default_factory=list)
    grade: str = ""                              # "A" | "B" | "C" | "D" | "F"
    grade_reason: str = ""


# ===========================================================================
# 技术指标计算
# ===========================================================================

def compute_bollinger_bands(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[float, float, float]:
    """计算布林带。

    Args:
        closes: 收盘价序列（正序）
        period: 周期
        num_std: 标准差倍数

    Returns:
        (upper, middle, lower)
    """
    if len(closes) < period:
        return 0.0, 0.0, 0.0
    recent = closes[-period:]
    middle = float(np.mean(recent))
    std = float(np.std(recent, ddof=1))
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_rsi(closes: np.ndarray, period: int = 14) -> Optional[float]:
    """计算 RSI (Wilder's smoothing)。"""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # 初始简单平均
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    # Wilder's 平滑
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def compute_macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[float, float, float]:
    """计算 MACD。

    Returns:
        (macd_line, signal_line, histogram)
    """
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0

    def _ema(data: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line[-1] - signal_line[-1]
    return float(macd_line[-1]), float(signal_line[-1]), float(histogram)


def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """计算 ATR (Average True Range, Wilder's smoothing)。"""
    if len(highs) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)
    # 初始简单平均
    atr = float(np.mean(tr_list[:period]))
    # Wilder's 平滑
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
    return atr


def compute_ma(closes: np.ndarray, period: int) -> float:
    """计算移动均线。"""
    if len(closes) < period:
        return 0.0
    return float(np.mean(closes[-period:]))


def classify_trend(closes: np.ndarray, current_price: Optional[float] = None) -> str:
    """根据价格与均线关系判断趋势方向。"""
    if len(closes) < 50:
        return "数据不足"
    ma20 = compute_ma(closes, 20)
    ma50 = compute_ma(closes, 50)
    if current_price is None:
        current_price = closes[-1]
    if current_price > ma20 > ma50:
        return "上升"
    elif current_price < ma20 < ma50:
        return "下降"
    else:
        return "横盘"


# ===========================================================================
# 交易分析引擎
# ===========================================================================

class TradeAnalyzer:
    """历史交易超详细分析引擎。

    流程：
      1. 从 OKX API 拉取全部历史账单
      2. 配对开仓/平仓账单，还原完整交易
      3. 逐笔拉取多时间框架 K 线 + 技术指标
      4. 分析成功/失败原因
      5. 生成报告并上传云端
    """

    # 多时间框架定义
    TIMEFRAMES = ["1M", "1W", "1D", "4H", "1H", "15m"]
    # OKX API bar 参数映射（标准格式）
    BAR_MAP = {
        "1M": "1M",
        "1W": "1W",
        "1D": "1D",
        "4H": "4H",
        "1H": "1H",
        "15m": "15m",
    }
    # 各时间框架需要的 K 线数量（用于指标计算）
    CANDLE_COUNTS = {
        "1M": 60,    # 60 个月 = 5 年
        "1W": 100,   # 100 周 ≈ 2 年
        "1D": 200,   # 200 天
        "4H": 200,   # 200 根 4H ≈ 33 天
        "1H": 200,   # 200 小时
        "15m": 200,  # 200 根 15m ≈ 2 天
    }

    def __init__(self, client, config: dict):
        """
        Args:
            client: OKXClient 实例
            config: 完整配置
        """
        self.client = client
        self.config = config
        self.analyses: List[TradeAnalysis] = []
        self._market_cache: Dict[str, Tuple[float, float]] = {}  # TTL 缓存
        self._cache_lock = threading.Lock()  # 修复 Bug C-8: 保护 _market_cache 并发访问

    # ------------------------------------------------------------------
    # 步骤 1: 拉取历史账单
    # ------------------------------------------------------------------

    def fetch_all_bills(self, days_back: int = 90) -> List[dict]:
        """拉取所有历史账单，过滤出开仓和平仓记录。

        OKX 账单类型 (billType):
          1: 划转  2: 交易  3: 交割  4: 自动换币
          5: 提币  6: 充值  7: 结算  8: 交割手续费
          9: 手续费  10: 资金费  11: 免息金额
          101: 平多  102: 平空  201: 开多  202: 开空
          203: 买入  204: 卖出  205: 交割多头  206: 交割空头

        Returns:
            过滤后的交易账单列表
        """
        logger.info(f"正在拉取最近 {days_back} 天的历史账单...")
        all_bills = self.client.get_all_bills_paginated(
            inst_type="SWAP",
            days_back=days_back,
        )
        logger.info(f"共获取 {len(all_bills)} 条账单记录")

        # 过滤出交易相关账单
        trade_bill_types = {"101", "102", "201", "202", "2", "205", "206"}
        trade_bills = [b for b in all_bills if str(b.get("billType", "")) in trade_bill_types]
        logger.info(f"其中交易账单: {len(trade_bills)} 条")
        return trade_bills

    # ------------------------------------------------------------------
    # 步骤 2: 配对开仓/平仓 -> 还原交易
    # ------------------------------------------------------------------

    def pair_trades(self, bills: List[dict]) -> List[TradeAnalysis]:
        """将开仓/平仓账单配对，还原完整交易。

        策略：按 instId + 时间顺序，开仓后匹配最近的平仓。
        """
        # 按 instId 分组
        by_inst: Dict[str, List[dict]] = {}
        for b in bills:
            inst_id = b.get("instId", "")
            if not inst_id:
                continue
            if inst_id not in by_inst:
                by_inst[inst_id] = []
            by_inst[inst_id].append(b)

        trades: List[TradeAnalysis] = []

        for inst_id, inst_bills in by_inst.items():
            # 按时间戳升序
            inst_bills.sort(key=lambda x: int(x.get("ts") or "0"))

            # 过滤出开仓和平仓记录
            # 修复 Bug C-1: 移除 billType="2"（通用交易），它在 SWAP 合约中语义不明确，
            # 且 op_side 判定只处理了 "201"/"202"，"2" 会被错误归类为 short
            open_types = {"201", "202"}           # 开多、开空
            close_types = {"101", "102", "205", "206"}  # 平多、平空、交割

            pending_opens: List[dict] = []
            # Track which opens have been matched (FIFO matching with used set)
            # M-22: used_opens 正确追踪已完成匹配的开仓索引，部分平仓不加入 used_opens，
            # 允许同一开仓被多次部分平仓匹配。
            used_opens: set = set()

            for b in inst_bills:
                bt = str(b.get("billType", ""))
                if bt in open_types:
                    pending_opens.append(b)
                elif bt in close_types and pending_opens:
                    # 匹配最近的同方向开仓
                    open_side = "long" if bt in ("101", "205") else "short"

                    # 找匹配的开仓
                    matched = None
                    for i, op in enumerate(pending_opens):
                        if i in used_opens:
                            continue
                        op_bt = str(op.get("billType", ""))
                        op_side = "long" if op_bt in ("201",) else "short"
                        if op_side == open_side:
                            matched = i
                            break

                    if matched is not None:
                        op = pending_opens[matched]
                        close_vol = float(b.get("sz", "0"))
                        open_vol = float(op.get("sz", "0"))
                        # 简化实现：部分平仓时减少剩余量，完全平仓才移除
                        # 注：若 OKX 账单返回的是单笔完整平仓，则 close_vol ≈ open_vol
                        if close_vol < open_vol:
                            # 部分平仓：减少 pending_opens 中的剩余量
                            op["sz"] = str(open_vol - close_vol)
                            logger.debug(
                                "部分平仓: %s 剩余 %s 张 (已平 %s/%s)",
                                inst_id, op["sz"], close_vol, open_vol,
                            )
                        else:
                            used_opens.add(matched)
                        trade = self._build_trade_from_bills(op, b, inst_id)
                        if trade:
                            trades.append(trade)

        # 按建仓时间降序
        trades.sort(key=lambda t: t.entry_time, reverse=True)
        logger.info(f"配对完成，共 {len(trades)} 笔完整交易")
        return trades

    def _get_ct_val(self, inst_id: str) -> float:
        """获取合约面值 (ctVal)。

        USDT 本位永续合约面值因币种而异。
        """
        # 硬编码常见币种面值
        ct_val_map = {
            "BTC": 0.01,
            "ETH": 0.1,
            "BCH": 0.1,
            "LTC": 1.0,
            "ETC": 1.0,
            "LINK": 1.0,
            "XRP": 10.0,
            "EOS": 10.0,
            "TRX": 10.0,
            "ADA": 10.0,
            "SOL": 1.0,
            "DOT": 1.0,
            "DOGE": 100.0,
            "AVAX": 1.0,
            "MATIC": 10.0,
            "SUI": 1.0,
            "APT": 1.0,
            "ARB": 10.0,
            "OP": 10.0,
            "FIL": 1.0,
            "ATOM": 1.0,
            "NEAR": 1.0,
            "UNI": 1.0,
            "AAVE": 0.1,
            "PEPE": 100000.0,
            "SHIB": 100000.0,
            "FLOKI": 100000.0,
            "BONK": 100000.0,
            "WIF": 10.0,
        }
        symbol = inst_id.split("-")[0] if "-" in inst_id else inst_id
        if symbol in ct_val_map:
            return ct_val_map[symbol]

        # 尝试从客户端获取合约信息
        try:
            if hasattr(self.client, "get_instrument_info"):
                info = self.client.get_instrument_info(inst_id)
                if info and "ctVal" in info:
                    return float(info["ctVal"])
        except Exception:
            pass

        # 修复 Bug H-14: 未知币种走默认值 0.1（保守估计），记录警告
        logger.warning(
            "未找到 %s (%s) 的合约面值，使用默认值 0.1（保守估计），收益率计算可能不准确",
            inst_id, symbol,
        )
        return 0.1  # 默认值（保守估计，避免高估收益率）

    def _build_trade_from_bills(
        self,
        open_bill: dict,
        close_bill: dict,
        inst_id: str,
    ) -> Optional[TradeAnalysis]:
        """从开仓/平仓账单构建 TradeAnalysis。"""
        entry_ts = int(open_bill.get("ts") or "0") / 1000.0
        exit_ts = int(close_bill.get("ts") or "0") / 1000.0

        if entry_ts <= 0 or exit_ts <= 0:
            return None

        entry_px = float(open_bill.get("px", "0"))
        exit_px = float(close_bill.get("px", "0"))
        pnl = float(close_bill.get("pnl", "0"))
        fee = float(close_bill.get("fee", "0"))
        sz = float(open_bill.get("sz", "0"))

        if entry_px <= 0 or sz <= 0:
            return None

        # 方向判断
        open_bt = str(open_bill.get("billType", ""))
        if open_bt == "201":
            direction = "long"
        elif open_bt == "202":
            direction = "short"
        else:
            # 通过 pnl 符号推断方向（不可靠: 多单也可能亏损）
            direction = "long" if pnl >= 0 else "short"
            logger.warning(
                "方向由盈亏推断 (%s): 盈亏=%+.2f, 方向=%s, 此推断可能不准确",
                inst_id, pnl, direction,
            )

        # 收益率 — 修复 Bug C-2: 使用合约面值 ctVal 计算精确收益率
        # USDT 本位永续合约面值因币种而异（BTC=0.01, ETH=0.1 等），直接 entry_px*sz 会失真
        ct_val = self._get_ct_val(inst_id)
        position_value = entry_px * sz * ct_val
        pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0

        symbol = inst_id.split("-")[0] if "-" in inst_id else inst_id

        trade = TradeAnalysis(
            trade_id=open_bill.get("billId", ""),
            inst_id=inst_id,
            symbol=symbol,
            direction=direction,
            entry_time=entry_ts,
            entry_time_str=datetime.fromtimestamp(entry_ts, tz=BEIJING_TZ).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            exit_time=exit_ts,
            exit_time_str=datetime.fromtimestamp(exit_ts, tz=BEIJING_TZ).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            holding_hours=(exit_ts - entry_ts) / 3600.0,
            entry_price=entry_px,
            exit_price=exit_px,
            quantity=sz,
            leverage=int(float(open_bill.get("lever", "1"))),
            pnl=pnl,
            pnl_pct=pnl_pct,
            fee=fee,
            is_win=pnl > 0,
            is_loss=pnl < 0,
            is_breakeven=pnl == 0,
        )
        return trade

    # ------------------------------------------------------------------
    # 步骤 3: 多时间框架技术分析
    # ------------------------------------------------------------------

    def analyze_timeframe(
        self,
        inst_id: str,
        entry_ts: float,
        tf: str,
    ) -> TimeframeAnalysis:
        """对单个时间框架进行技术指标分析。

        Args:
            inst_id: 合约 ID
            entry_ts: 建仓时间戳 (秒)
            tf: 时间框架

        Returns:
            TimeframeAnalysis
        """
        result = TimeframeAnalysis(timeframe=tf)
        bar = self.BAR_MAP.get(tf, "1H")
        limit = self.CANDLE_COUNTS.get(tf, 200)

        try:
            raw = self.client.get_candles_enhanced(inst_id, bar=bar, limit=limit)
            if not raw:
                raw = self.client.get_candles(inst_id, bar=bar, limit=limit)
            if not raw:
                return result

            # 转换为正序数组
            candles_list = []
            for row in raw:
                candles_list.append({
                    "ts": int(row[0]),
                    "o": float(row[1]),
                    "h": float(row[2]),
                    "l": float(row[3]),
                    "c": float(row[4]),
                    "vol": float(row[5]),
                })
            candles_list.sort(key=lambda x: x["ts"])

            closes = np.array([c["c"] for c in candles_list])
            highs = np.array([c["h"] for c in candles_list])
            lows = np.array([c["l"] for c in candles_list])
            volumes = np.array([c["vol"] for c in candles_list])

            # 找到建仓时刻最近的蜡烛 — 仅搜索 ≤ entry_ms 的蜡烛，消除 look-ahead bias
            entry_ms = entry_ts * 1000
            best_idx = 0
            best_diff = float("inf")
            for i, c in enumerate(candles_list):
                if c["ts"] > entry_ms:
                    continue
                diff = entry_ms - c["ts"]
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            # 修复 Bug H-11: 时间偏差过大（>1小时），返回空分析
            if best_diff > 3600 * 1000:
                return result

            # 修复 Bug C-6: 截断到建仓时刻，消除 look-ahead bias
            closes = closes[:best_idx + 1]
            highs = highs[:best_idx + 1]
            lows = lows[:best_idx + 1]
            volumes = volumes[:best_idx + 1]

            # 修复 Bug B-3: 截断后重新检查数据量，防止指标静默归零
            if len(closes) < 20:
                return result

            entry_candle = candles_list[best_idx]
            result.entry_candle = CandleSnapshot(
                timestamp=entry_candle["ts"],
                open=entry_candle["o"],
                high=entry_candle["h"],
                low=entry_candle["l"],
                close=entry_candle["c"],
                volume=entry_candle["vol"],
            )

            # 修复 Bug C-4: 历史分析中使用建仓时刻蜡烛收盘价，而非数据末尾的"当前价"
            current_price = entry_candle["c"]

            # 布林带
            bb_upper, bb_middle, bb_lower = compute_bollinger_bands(closes)
            result.bb_upper = bb_upper
            result.bb_middle = bb_middle
            result.bb_lower = bb_lower
            if bb_middle > 0:
                result.bb_width_pct = (bb_upper - bb_lower) / bb_middle * 100
            if bb_upper > bb_lower:
                result.bb_position_pct = (current_price - bb_lower) / (bb_upper - bb_lower)

            # RSI
            result.rsi = compute_rsi(closes)
            if result.rsi is not None:
                if result.rsi >= 70:
                    result.rsi_zone = "超买"
                elif result.rsi <= 30:
                    result.rsi_zone = "超卖"
                elif result.rsi >= 55:
                    result.rsi_zone = "偏多"
                elif result.rsi <= 45:
                    result.rsi_zone = "偏空"
                else:
                    result.rsi_zone = "中性"

            # MACD
            macd, macd_sig, macd_hist = compute_macd(closes)
            result.macd = macd
            result.macd_signal = macd_sig
            result.macd_histogram = macd_hist
            if macd > macd_sig and macd_hist > 0:
                result.macd_trend = "多头"
            elif macd < macd_sig and macd_hist < 0:
                result.macd_trend = "空头"
            elif macd > macd_sig and macd_hist < 0:
                result.macd_trend = "顶背离"
            elif macd < macd_sig and macd_hist > 0:
                result.macd_trend = "底背离"
            else:
                result.macd_trend = "中性"

            # 均线
            result.ma_20 = compute_ma(closes, 20)
            result.ma_50 = compute_ma(closes, 50) if len(closes) >= 50 else 0.0
            result.ma_200 = compute_ma(closes, 200) if len(closes) >= 200 else 0.0
            if result.ma_20 > 0:
                result.price_vs_ma20_pct = (current_price - result.ma_20) / result.ma_20 * 100

            # 趋势
            result.trend_direction = classify_trend(closes, current_price)

            # ATR
            result.atr = compute_atr(highs, lows, closes)
            if current_price > 0:
                result.atr_pct = result.atr / current_price * 100

            # 成交量 — 修复 Bug C-7: 使用 volumes[-21:-1] 计算均量，排除当前蜡烛避免 look-ahead bias
            if len(volumes) >= 21:
                avg_vol = float(np.mean(volumes[-21:-1]))
                if avg_vol > 0:
                    result.volume_ratio = volumes[-1] / avg_vol
            if result.volume_ratio >= 2.0:
                result.volume_zone = "放量"
            elif result.volume_ratio <= 0.5:
                result.volume_zone = "缩量"
            else:
                result.volume_zone = "正常"

        except Exception as e:
            logger.warning(f"分析 {inst_id} {tf} 失败: {e}")

        return result

    def analyze_all_timeframes(
        self,
        inst_id: str,
        entry_ts: float,
    ) -> Dict[str, TimeframeAnalysis]:
        """对所有时间框架进行技术分析。"""
        results: Dict[str, TimeframeAnalysis] = {}
        for tf in self.TIMEFRAMES:
            results[tf] = self.analyze_timeframe(inst_id, entry_ts, tf)
            time.sleep(0.1)  # API 速率限制
        return results

    # ------------------------------------------------------------------
    # 步骤 4: 市场环境数据
    # ------------------------------------------------------------------

    def fetch_market_context(
        self,
        inst_id: str,
        entry_ts: float,
    ) -> Tuple[float, float, float, float]:
        """获取建仓时刻的市场环境数据（历史数据不可得则返回当前值）。

        Returns:
            (funding_rate, long_short_ratio, open_interest, spread_pct)
        """
        funding = 0.0
        ls_ratio = 0.0
        oi = 0.0
        spread = 0.0

        now = time.time()
        cache_key = f"mc:{inst_id}"

        # 检查缓存 — 修复 Bug C-8: 使用锁保护并发访问
        with self._cache_lock:
            if cache_key in self._market_cache:
                cached_time, cached_funding, cached_ls = self._market_cache[cache_key]
                if now - cached_time < 60:
                    funding = cached_funding
                    ls_ratio = cached_ls
                else:
                    del self._market_cache[cache_key]

        if funding == 0.0:
            try:
                # 资金费率（当前最新）
                fr = self.client.get_funding_rate(inst_id)
                if fr is not None:
                    funding = fr
            except Exception:
                pass

        if ls_ratio == 0.0:
            try:
                # 多空比
                lsr = self.client.get_long_short_ratio(inst_id)
                if lsr is not None:
                    ls_ratio = lsr
            except Exception:
                pass

        # 缓存 funding 和 ls_ratio — 修复 Bug C-8: 使用锁保护并发写入
        with self._cache_lock:
            if cache_key not in self._market_cache:
                self._market_cache[cache_key] = (now, funding, ls_ratio)

        try:
            # 持仓量
            oi_val = self.client.get_open_interest(inst_id)
            if oi_val is not None:
                oi = oi_val
        except Exception:
            pass

        try:
            # 买卖价差
            ob = self.client.get_order_book(inst_id, sz=5)
            if ob:
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                if bids and asks:
                    bid_px = float(bids[0][0])
                    ask_px = float(asks[0][0])
                    if ask_px > 0:
                        spread = (ask_px - bid_px) / ask_px * 100
        except Exception:
            pass

        return funding, ls_ratio, oi, spread

    # ------------------------------------------------------------------
    # 步骤 5: 成功/失败原因分析
    # ------------------------------------------------------------------

    def analyze_trade_outcome(self, trade: TradeAnalysis):
        """分析交易成功/失败的原因。

        基于多时间框架技术指标 + 市场环境 + 持仓数据综合判断。
        """
        reasons_win: List[str] = []
        reasons_loss: List[str] = []
        observations: List[str] = []

        # ---- 1. 趋势方向匹配 ----
        tf_main = trade.timeframe_analysis.get("4H") or trade.timeframe_analysis.get("1H")
        if tf_main:
            trend = tf_main.trend_direction
            if trade.direction == "long" and trend == "上升":
                reasons_win.append("4H 趋势向上，做多方向与趋势一致")
            elif trade.direction == "short" and trend == "下降":
                reasons_win.append("4H 趋势向下，做空方向与趋势一致")
            elif trade.direction == "long" and trend == "下降":
                reasons_loss.append("4H 趋势向下，逆势做多风险高")
            elif trade.direction == "short" and trend == "上升":
                reasons_loss.append("4H 趋势向上，逆势做空风险高")

        # ---- 2. RSI 极端区域 ----
        for tf_name, tf_analysis in trade.timeframe_analysis.items():
            if tf_name in ("1H", "4H"):
                if tf_analysis.rsi_zone == "超买" and trade.direction == "long":
                    reasons_loss.append(f"{tf_name} RSI={tf_analysis.rsi:.0f} 超买区域追多，回撤风险大")
                elif tf_analysis.rsi_zone == "超卖" and trade.direction == "short":
                    reasons_loss.append(f"{tf_name} RSI={tf_analysis.rsi:.0f} 超卖区域追空，反弹风险大")
                elif tf_analysis.rsi_zone == "超卖" and trade.direction == "long":
                    reasons_win.append(f"{tf_name} RSI={tf_analysis.rsi:.0f} 超卖区域抄底，反弹概率高")
                elif tf_analysis.rsi_zone == "超买" and trade.direction == "short":
                    reasons_win.append(f"{tf_name} RSI={tf_analysis.rsi:.0f} 超买区域做空，回调概率高")

        # ---- 3. 布林带位置 ----
        for tf_name, tf_analysis in trade.timeframe_analysis.items():
            if tf_name in ("4H", "1D"):
                pos = tf_analysis.bb_position_pct
                if pos > 0.9 and trade.direction == "long":
                    reasons_loss.append(f"{tf_name} 价格在布林上轨({pos:.0%})，做多追高")
                elif pos < 0.1 and trade.direction == "short":
                    reasons_loss.append(f"{tf_name} 价格在布林下轨({pos:.0%})，做空杀跌")
                elif pos < 0.1 and trade.direction == "long":
                    reasons_win.append(f"{tf_name} 价格在布林下轨({pos:.0%})，低位做多")
                elif pos > 0.9 and trade.direction == "short":
                    reasons_win.append(f"{tf_name} 价格在布林上轨({pos:.0%})，高位做空")

        # ---- 4. MACD 信号 ----
        for tf_name, tf_analysis in trade.timeframe_analysis.items():
            if tf_name in ("4H", "1D"):
                macd_t = tf_analysis.macd_trend
                if macd_t == "多头" and trade.direction == "long":
                    reasons_win.append(f"{tf_name} MACD 多头排列，支撑做多")
                elif macd_t == "空头" and trade.direction == "short":
                    reasons_win.append(f"{tf_name} MACD 空头排列，支撑做空")
                elif macd_t == "空头" and trade.direction == "long":
                    reasons_loss.append(f"{tf_name} MACD 空头排列，做多逆势")
                elif macd_t == "多头" and trade.direction == "short":
                    reasons_loss.append(f"{tf_name} MACD 多头排列，做空逆势")

        # ---- 5. 成交量分析 ----
        for tf_name, tf_analysis in trade.timeframe_analysis.items():
            if tf_name in ("1H", "4H"):
                if tf_analysis.volume_zone == "放量":
                    if trade.direction == "long":
                        reasons_win.append(f"{tf_name} 放量上涨，动能充足")
                    else:
                        reasons_loss.append(f"{tf_name} 放量时做空，可能遭遇轧空")
                elif tf_analysis.volume_zone == "缩量":
                    observations.append(f"{tf_name} 成交量萎缩，市场参与度低")

        # ---- 6. 资金费率 ----
        if abs(trade.funding_rate_entry) > 0.005:
            if trade.funding_rate_entry > 0 and trade.direction == "long":
                reasons_loss.append(
                    f"资金费率 {trade.funding_rate_entry*100:+.3f}% 偏高，做多成本高"
                )
            elif trade.funding_rate_entry < 0 and trade.direction == "short":
                reasons_loss.append(
                    f"资金费率 {trade.funding_rate_entry*100:+.3f}% 偏负，做空成本高"
                )

        # ---- 7. 多空比 ----
        if trade.long_short_ratio > 2.0:
            if trade.direction == "long":
                reasons_loss.append(f"多空比 {trade.long_short_ratio:.2f}，市场过度看多，拥挤风险")
            else:
                reasons_win.append(f"多空比 {trade.long_short_ratio:.2f}，市场过度看多，做空博弈")
        elif trade.long_short_ratio < 0.5:
            if trade.direction == "short":
                reasons_loss.append(f"多空比 {trade.long_short_ratio:.2f}，市场过度看空，拥挤风险")
            else:
                reasons_win.append(f"多空比 {trade.long_short_ratio:.2f}，市场过度看空，做多博弈")

        # ---- 8. 持仓时间 ----
        if trade.holding_hours > 72:
            observations.append(f"持仓 {trade.holding_hours:.0f} 小时，周期较长")

        # ---- 9. 价差 ----
        if trade.spread_pct > 0.5:
            reasons_loss.append(f"建仓时价差 {trade.spread_pct:.2f}%，流动性差")

        # ---- 10. 多周期共振 ----
        tf_1h = trade.timeframe_analysis.get("1H")
        tf_4h = trade.timeframe_analysis.get("4H")
        if tf_1h and tf_4h:
            if (tf_1h.trend_direction == tf_4h.trend_direction == "上升"
                    and trade.direction == "long"):
                reasons_win.append("1H+4H 趋势共振向上，多周期确认")
            elif (tf_1h.trend_direction == tf_4h.trend_direction == "下降"
                    and trade.direction == "short"):
                reasons_win.append("1H+4H 趋势共振向下，多周期确认")
            elif (tf_1h.trend_direction != tf_4h.trend_direction):
                observations.append("1H 与 4H 趋势不一致，多周期分歧")

        trade.success_reasons = list(dict.fromkeys(reasons_win))  # 去重保序
        trade.failure_reasons = list(dict.fromkeys(reasons_loss))
        trade.key_observations = list(dict.fromkeys(observations))

        # ---- 评级 ----
        trade.grade, trade.grade_reason = self._grade_trade(trade)

    def _grade_trade(self, trade: TradeAnalysis) -> Tuple[str, str]:
        """对交易进行评级 (A-F)。

        评分方法：
          - 基础分 50，盈亏维度 ±30~40（最大贡献），成功/失败原因每项 ±3，
            趋势匹配/逆势每项 ±5
          - 各维度量纲不同（盈亏百分比 vs 原因计数 vs 趋势方向），
            直接加减可能导致某一维度过度主导最终评分
          - 最终分数通过 max(0, min(100, score)) 归一化到 0-100 范围
          - 建议后续引入标准化权重或使用 z-score 归一化各维度
        """
        score = 50  # 基础分

        # 盈亏影响
        if trade.is_win:
            score += 30
            if trade.pnl_pct > 5:
                score += 10
            elif trade.pnl_pct > 2:
                score += 5
        elif trade.is_loss:
            score -= 30
            if trade.pnl_pct < -5:
                score -= 10
            elif trade.pnl_pct < -2:
                score -= 5

        # 成功原因加分
        score += len(trade.success_reasons) * 3
        # 失败原因扣分
        score -= len(trade.failure_reasons) * 3

        # 趋势匹配加分
        for r in trade.success_reasons:
            if "趋势一致" in r or "共振" in r:
                score += 5
        for r in trade.failure_reasons:
            if "逆势" in r:
                score -= 5

        # 修复 Bug L-4: 归一化分数到 0-100 范围
        score = max(0, min(100, score))

        if score >= 85:
            return "A", "交易质量优秀，多指标共振确认"
        elif score >= 70:
            return "B", "交易质量良好，有明确的技术面支撑"
        elif score >= 55:
            return "C", "交易质量一般，存在部分瑕疵"
        elif score >= 40:
            return "D", "交易质量较差，有明显逆势或追高/杀跌行为"
        else:
            return "F", "交易质量极差，多项指标严重背离"

    # ------------------------------------------------------------------
    # 步骤 6: 完整分析流程
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        days_back: int = 90,
        max_trades: int = 100,
    ) -> List[TradeAnalysis]:
        """执行完整的历史交易分析流程。

        Args:
            days_back: 回溯天数
            max_trades: 最大分析笔数

        Returns:
            TradeAnalysis 列表
        """
        logger.info("=" * 60)
        logger.info("  历史交易超详细分析 — 开始")
        logger.info("=" * 60)

        # 1. 拉取账单
        bills = self.fetch_all_bills(days_back)

        if not bills:
            logger.warning("未获取到任何交易账单，请确认账户有历史交易记录")
            return []

        # 2. 配对交易
        trades = self.pair_trades(bills)

        if not trades:
            logger.warning("未能配对出完整交易，请检查账单数据")
            return []

        # 限制分析数量
        trades = trades[:max_trades]

        logger.info(f"\n开始分析 {len(trades)} 笔交易...\n")

        # 3. 逐笔分析
        for i, trade in enumerate(trades):
            logger.info(
                f"[{i+1}/{len(trades)}] {trade.symbol} "
                f"{trade.direction.upper()} | "
                f"入场: {trade.entry_time_str} | "
                f"盈亏: {trade.pnl:+.2f} USDT"
            )

            # 3a. 多时间框架 K 线分析
            logger.info(f"  拉取多时间框架 K 线...")
            trade.timeframe_analysis = self.analyze_all_timeframes(
                trade.inst_id, trade.entry_time
            )

            # 3b. 市场环境
            funding, ls_ratio, oi, spread = self.fetch_market_context(
                trade.inst_id, trade.entry_time
            )
            trade.funding_rate_entry = funding
            trade.long_short_ratio = ls_ratio
            trade.open_interest = oi
            trade.spread_pct = spread

            # 3c. 成功/失败原因分析
            self.analyze_trade_outcome(trade)

            # 打印摘要
            status = "盈利" if trade.is_win else ("亏损" if trade.is_loss else "保本")
            logger.info(
                f"  结果: {status} | 评级: {trade.grade} | "
                f"成功因素: {len(trade.success_reasons)} | "
                f"失败因素: {len(trade.failure_reasons)}"
            )
            time.sleep(0.3)  # API 速率控制

        self.analyses = trades
        logger.info(f"\n分析完成! 共 {len(trades)} 笔交易")
        return trades

    # ------------------------------------------------------------------
    # 步骤 7: 统计摘要
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """生成统计摘要。"""
        if not self.analyses:
            return {}

        total = len(self.analyses)
        wins = sum(1 for t in self.analyses if t.is_win)
        losses = sum(1 for t in self.analyses if t.is_loss)
        breakevens = sum(1 for t in self.analyses if t.is_breakeven)
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        total_pnl = sum(t.pnl for t in self.analyses)
        avg_pnl = total_pnl / total if total > 0 else 0
        best_trade = max(self.analyses, key=lambda t: t.pnl)
        worst_trade = min(self.analyses, key=lambda t: t.pnl)

        # 方向分布
        longs = sum(1 for t in self.analyses if t.direction == "long")
        shorts = sum(1 for t in self.analyses if t.direction == "short")
        long_win_rate = (
            sum(1 for t in self.analyses if t.direction == "long" and t.is_win)
            / max(longs, 1) * 100
        )
        short_win_rate = (
            sum(1 for t in self.analyses if t.direction == "short" and t.is_win)
            / max(shorts, 1) * 100
        )

        # 评级分布
        grades: Dict[str, int] = {}
        for t in self.analyses:
            grades[t.grade] = grades.get(t.grade, 0) + 1

        # 常见失败原因统计
        failure_reasons: Dict[str, int] = {}
        for t in self.analyses:
            for r in t.failure_reasons:
                key = r[:50]  # 截断作为 key
                failure_reasons[key] = failure_reasons.get(key, 0) + 1
        top_failures = sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)[:10]

        # 常见成功原因统计
        success_reasons: Dict[str, int] = {}
        for t in self.analyses:
            for r in t.success_reasons:
                key = r[:50]
                success_reasons[key] = success_reasons.get(key, 0) + 1
        top_successes = sorted(success_reasons.items(), key=lambda x: x[1], reverse=True)[:10]

        # 按币种统计
        by_symbol: Dict[str, dict] = {}
        for t in self.analyses:
            s = t.symbol
            if s not in by_symbol:
                by_symbol[s] = {"count": 0, "wins": 0, "pnl": 0.0}
            by_symbol[s]["count"] += 1
            if t.is_win:
                by_symbol[s]["wins"] += 1
            by_symbol[s]["pnl"] += t.pnl

        # 持仓时长分析
        holding_hours = [t.holding_hours for t in self.analyses]
        avg_holding = np.mean(holding_hours) if holding_hours else 0
        median_holding = np.median(holding_hours) if holding_hours else 0

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "breakevens": breakevens,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "longs": longs,
            "shorts": shorts,
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
            "grades": grades,
            "top_failures": top_failures,
            "top_successes": top_successes,
            "by_symbol": by_symbol,
            "avg_holding_hours": avg_holding,
            "median_holding_hours": median_holding,
        }


# ===========================================================================
# HTML 报告生成
# ===========================================================================

def generate_html_report(
    analyses: List[TradeAnalysis],
    summary: dict,
    config: dict,
) -> str:
    """生成超详细 HTML 分析报告。

    Args:
        analyses: 交易分析列表
        summary: 统计摘要
        config: 配置

    Returns:
        HTML 字符串
    """
    now = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")
    total = summary.get("total_trades", 0)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OKX 历史交易超详细分析报告</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; line-height: 1.6; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 12px; margin-bottom: 20px; font-size: 24px; }}
  h2 {{ color: #f0f6fc; margin: 30px 0 15px; font-size: 20px; border-left: 3px solid #58a6ff; padding-left: 12px; }}
  h3 {{ color: #58a6ff; margin: 20px 0 10px; font-size: 16px; }}
  .meta {{ color: #8b949e; font-size: 13px; margin-bottom: 25px; }}

  /* 摘要卡片 */
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 20px; text-align: center; }}
  .card .value {{ font-size: 26px; font-weight: bold; color: #58a6ff; }}
  .card .label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
  .card.win .value {{ color: #3fb950; }}
  .card.loss .value {{ color: #f85149; }}
  .card.warn .value {{ color: #d2991d; }}

  /* 表格 */
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
  th {{ background: #161b22; color: #8b949e; text-align: left; padding: 10px 8px; border-bottom: 2px solid #30363d; white-space: nowrap; position: sticky; top: 0; }}
  td {{ padding: 8px; border-bottom: 1px solid #21262d; vertical-align: top; }}
  tr:hover {{ background: #1c2128; }}
  .table-wrap {{ max-height: 600px; overflow-y: auto; border: 1px solid #30363d; border-radius: 6px; }}

  /* 标签 */
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
  .tag-long {{ background: #1b3d1b; color: #3fb950; }}
  .tag-short {{ background: #3d1b1b; color: #f85149; }}
  .tag-win {{ background: #1b3d1b; color: #3fb950; }}
  .tag-loss {{ background: #3d1b1b; color: #f85149; }}
  .tag-be {{ background: #1b2b3d; color: #58a6ff; }}
  .tag-grade-a {{ background: #1b3d1b; color: #3fb950; }}
  .tag-grade-b {{ background: #1b3d2b; color: #7ee787; }}
  .tag-grade-c {{ background: #3d3d1b; color: #d2991d; }}
  .tag-grade-d {{ background: #3d2b1b; color: #f0883e; }}
  .tag-grade-f {{ background: #3d1b1b; color: #f85149; }}

  /* 交易详情 */
  .trade-detail {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin: 20px 0; overflow: hidden; }}
  .trade-header {{ background: #1c2128; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }}
  .trade-body {{ padding: 20px; }}
  .trade-body .section {{ margin-bottom: 20px; }}
  .trade-body .section-title {{ color: #58a6ff; font-size: 14px; font-weight: bold; margin-bottom: 8px; }}

  .indicator-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
  .indicator-item {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 10px 14px; }}
  .indicator-item .tf {{ color: #58a6ff; font-size: 12px; font-weight: bold; }}
  .indicator-item .row {{ display: flex; justify-content: space-between; font-size: 12px; margin: 2px 0; }}
  .indicator-item .row .key {{ color: #8b949e; }}
  .indicator-item .row .val {{ color: #c9d1d9; }}

  .reason-list {{ list-style: none; padding: 0; }}
  .reason-list li {{ padding: 6px 12px; margin: 4px 0; border-radius: 4px; font-size: 13px; }}
  .reason-list li.success {{ background: #1b3d1b22; border-left: 3px solid #3fb950; }}
  .reason-list li.failure {{ background: #3d1b1b22; border-left: 3px solid #f85149; }}
  .reason-list li.observation {{ background: #1b2b3d22; border-left: 3px solid #58a6ff; }}

  .footer {{ margin-top: 40px; padding: 20px 0; border-top: 1px solid #30363d; color: #8b949e; font-size: 12px; text-align: center; }}

  .tabs {{ display: flex; gap: 0; margin-bottom: 20px; border-bottom: 2px solid #30363d; }}
  .tab {{ padding: 10px 20px; cursor: pointer; color: #8b949e; border-bottom: 2px solid transparent; margin-bottom: -2px; font-size: 14px; }}
  .tab.active {{ color: #58a6ff; border-bottom-color: #58a6ff; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  @media (max-width: 768px) {{
    .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .indicator-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">
<h1>OKX 历史交易超详细分析报告</h1>
<p class="meta">生成时间: {now} (北京时间) | 分析周期: 最近 {total} 笔交易</p>

<!-- 统计摘要 -->
<h2>总览</h2>
<div class="summary-grid">
  <div class="card">
    <div class="value">{total}</div>
    <div class="label">总交易笔数</div>
  </div>
  <div class="card win">
    <div class="value">{summary.get('wins', 0)}</div>
    <div class="label">盈利笔数</div>
  </div>
  <div class="card loss">
    <div class="value">{summary.get('losses', 0)}</div>
    <div class="label">亏损笔数</div>
  </div>
  <div class="card">
    <div class="value">{summary.get('win_rate', 0):.1f}%</div>
    <div class="label">胜率</div>
  </div>
  <div class="card {'win' if summary.get('total_pnl', 0) > 0 else ('loss' if summary.get('total_pnl', 0) < 0 else '')}">
    <div class="value">{summary.get('total_pnl', 0):+.2f}</div>
    <div class="label">总盈亏 (USDT)</div>
  </div>
  <div class="card">
    <div class="value">{summary.get('avg_pnl', 0):+.2f}</div>
    <div class="label">平均盈亏 (USDT)</div>
  </div>
  <div class="card">
    <div class="value">{summary.get('long_win_rate', 0):.1f}%</div>
    <div class="label">做多胜率 ({summary.get('longs', 0)}笔)</div>
  </div>
  <div class="card">
    <div class="value">{summary.get('short_win_rate', 0):.1f}%</div>
    <div class="label">做空胜率 ({summary.get('shorts', 0)}笔)</div>
  </div>
  <div class="card">
    <div class="value">{summary.get('avg_holding_hours', 0):.1f}h</div>
    <div class="label">平均持仓时长</div>
  </div>
</div>

<!-- 评级分布 -->
<h2>交易评级分布</h2>
<div class="summary-grid">
"""
    grades = summary.get("grades", {})
    for grade in ["A", "B", "C", "D", "F"]:
        count = grades.get(grade, 0)
        pct = count / total * 100 if total > 0 else 0
        html += f"""  <div class="card">
    <div class="value">{count} <span style="font-size:14px;">({pct:.1f}%)</span></div>
    <div class="label"><span class="tag tag-grade-{grade.lower()}">{grade} 级</span></div>
  </div>
"""

    html += """</div>

<!-- 常见成功/失败原因 -->
<div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
<div>
<h2>Top 成功因素</h2>
<table>
<thead><tr><th>#</th><th>原因</th><th>出现次数</th></tr></thead>
<tbody>
"""
    for i, (reason, count) in enumerate(summary.get("top_successes", [])[:10]):
        html += f"""<tr><td>{i+1}</td><td>{html.escape(reason)}</td><td>{count}</td></tr>
"""
    html += """</tbody></table>
</div>
<div>
<h2>Top 失败因素</h2>
<table>
<thead><tr><th>#</th><th>原因</th><th>出现次数</th></tr></thead>
<tbody>
"""
    for i, (reason, count) in enumerate(summary.get("top_failures", [])[:10]):
        html += f"""<tr><td>{i+1}</td><td>{html.escape(reason)}</td><td>{count}</td></tr>
"""
    html += """</tbody></table>
</div>
</div>

<!-- 币种维度 -->
<h2>按币种统计</h2>
<table>
<thead><tr><th>币种</th><th>交易笔数</th><th>胜率</th><th>总盈亏</th><th>平均盈亏</th></tr></thead>
<tbody>
"""
    by_symbol = summary.get("by_symbol", {})
    for sym, data in sorted(by_symbol.items(),
                             key=lambda x: x[1]["pnl"], reverse=True):
        cnt = data["count"]
        wr = data["wins"] / cnt * 100 if cnt > 0 else 0
        avg_p = data["pnl"] / cnt if cnt > 0 else 0
        pnl_class = "win" if data["pnl"] >= 0 else "loss"
        html += f"""<tr>
  <td><strong>{html.escape(sym)}</strong></td>
  <td>{cnt}</td>
  <td>{wr:.1f}%</td>
  <td class="{pnl_class}">{data['pnl']:+.2f}</td>
  <td class="{pnl_class}">{avg_p:+.2f}</td>
</tr>
"""

    html += """</tbody></table>

<!-- 逐笔交易详情 -->
<h2>逐笔交易详情</h2>
"""
    for i, t in enumerate(analyses):
        pnl_class = "tag-win" if t.is_win else ("tag-loss" if t.is_loss else "tag-be")
        pnl_text = "盈利" if t.is_win else ("亏损" if t.is_loss else "保本")
        dir_class = "tag-long" if t.direction == "long" else "tag-short"
        dir_text = "做多" if t.direction == "long" else "做空"

        html += f"""
<div class="trade-detail">
  <div class="trade-header">
    <div>
      <strong style="font-size:16px;">#{i+1} {html.escape(t.symbol)}</strong>
      <span class="tag {dir_class}">{dir_text}</span>
      <span class="tag {pnl_class}">{pnl_text} {t.pnl:+.2f} USDT</span>
      <span class="tag tag-grade-{t.grade.lower()}">评级: {html.escape(t.grade)}</span>
    </div>
    <div style="color:#8b949e;font-size:12px;">
      入场: {html.escape(t.entry_time_str)} | 持仓: {t.holding_hours:.1f}h | 杠杆: {t.leverage}x
    </div>
  </div>
  <div class="trade-body">
    <!-- 基本信息 -->
    <div class="section">
      <div class="section-title">基本信息</div>
      <div class="indicator-grid">
        <div class="indicator-item">
          <div class="row"><span class="key">入场价</span><span class="val">{t.entry_price:.4f}</span></div>
          <div class="row"><span class="key">平仓价</span><span class="val">{t.exit_price:.4f}</span></div>
          <div class="row"><span class="key">数量</span><span class="val">{t.quantity:.0f} 张</span></div>
          <div class="row"><span class="key">杠杆</span><span class="val">{t.leverage}x</span></div>
        </div>
        <div class="indicator-item">
          <div class="row"><span class="key">盈亏</span><span class="val" style="color:{'#3fb950' if t.pnl >= 0 else '#f85149'}">{t.pnl:+.2f} USDT</span></div>
          <div class="row"><span class="key">收益率</span><span class="val" style="color:{'#3fb950' if t.pnl_pct >= 0 else '#f85149'}">{t.pnl_pct:+.2f}%</span></div>
          <div class="row"><span class="key">手续费</span><span class="val">{t.fee:.4f}</span></div>
          <div class="row"><span class="key">持仓时长</span><span class="val">{t.holding_hours:.1f}h</span></div>
        </div>
        <div class="indicator-item">
          <div class="row"><span class="key">资金费率</span><span class="val">{t.funding_rate_entry*100:+.4f}%</span></div>
          <div class="row"><span class="key">多空比</span><span class="val">{t.long_short_ratio:.2f}</span></div>
          <div class="row"><span class="key">持仓量</span><span class="val">{t.open_interest:.0f}</span></div>
          <div class="row"><span class="key">价差</span><span class="val">{t.spread_pct:.3f}%</span></div>
        </div>
      </div>
    </div>

    <!-- 多时间框架指标 -->
    <div class="section">
      <div class="section-title">多时间框架技术指标</div>
      <div style="overflow-x:auto;">
      <table>
      <thead><tr>
        <th>指标</th>
"""
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            html += f"<th>{tf}</th>"
        html += "</tr></thead><tbody>"

        # 价格
        html += "<tr><td><strong>价格</strong></td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            if ta and ta.entry_candle:
                html += f"<td>{ta.entry_candle.close:.4f}</td>"
            else:
                html += "<td>-</td>"
        html += "</tr>"

        # 布林带
        html += "<tr><td><strong>布林上轨</strong></td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            html += f"<td>{ta.bb_upper:.4f}</td>" if ta else "<td>-</td>"
        html += "</tr>"
        html += "<tr><td><strong>布林中轨</strong></td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            html += f"<td>{ta.bb_middle:.4f}</td>" if ta else "<td>-</td>"
        html += "</tr>"
        html += "<tr><td><strong>布林下轨</strong></td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            html += f"<td>{ta.bb_lower:.4f}</td>" if ta else "<td>-</td>"
        html += "</tr>"
        html += "<tr><td>布林位置</td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            if ta:
                color = "#3fb950" if ta.bb_position_pct < 0.2 else ("#f85149" if ta.bb_position_pct > 0.8 else "#c9d1d9")
                html += f"<td style='color:{color}'>{ta.bb_position_pct:.0%}</td>"
            else:
                html += "<td>-</td>"
        html += "</tr>"

        # RSI
        html += "<tr><td><strong>RSI(14)</strong></td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            if ta and ta.rsi is not None:
                color = "#f85149" if ta.rsi >= 70 else ("#3fb950" if ta.rsi <= 30 else "#c9d1d9")
                html += f"<td style='color:{color}'>{ta.rsi:.0f} ({ta.rsi_zone})</td>"
            else:
                html += "<td>-</td>"
        html += "</tr>"

        # MACD
        html += "<tr><td><strong>MACD</strong></td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            if ta:
                color = "#3fb950" if ta.macd_histogram > 0 else "#f85149"
                html += f"<td style='color:{color}'>{ta.macd_histogram:.4f} ({ta.macd_trend})</td>"
            else:
                html += "<td>-</td>"
        html += "</tr>"

        # 趋势
        html += "<tr><td><strong>趋势</strong></td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            if ta:
                color = "#3fb950" if ta.trend_direction == "上升" else ("#f85149" if ta.trend_direction == "下降" else "#d2991d")
                html += f"<td style='color:{color}'>{ta.trend_direction}</td>"
            else:
                html += "<td>-</td>"
        html += "</tr>"

        # 成交量
        html += "<tr><td><strong>量比</strong></td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            if ta:
                color = "#3fb950" if ta.volume_zone == "放量" else ("#8b949e" if ta.volume_zone == "缩量" else "#c9d1d9")
                html += f"<td style='color:{color}'>{ta.volume_ratio:.1f}x ({ta.volume_zone})</td>"
            else:
                html += "<td>-</td>"
        html += "</tr>"

        # ATR
        html += "<tr><td><strong>ATR(14)</strong></td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            if ta:
                html += f"<td>{ta.atr:.4f} ({ta.atr_pct:.2f}%)</td>"
            else:
                html += "<td>-</td>"
        html += "</tr>"

        # 均线
        html += "<tr><td><strong>MA20</strong></td>"
        for tf in ["1M", "1W", "1D", "4H", "1H", "15m"]:
            ta = t.timeframe_analysis.get(tf)
            if ta:
                color = "#3fb950" if ta.price_vs_ma20_pct > 0 else "#f85149"
                html += f"<td style='color:{color}'>{ta.ma_20:.4f} ({ta.price_vs_ma20_pct:+.1f}%)</td>"
            else:
                html += "<td>-</td>"
        html += "</tr>"

        html += """</tbody></table>
      </div>
    </div>
"""

        # 原因分析
        html += """    <div class="section">
      <div class="section-title">分析结论</div>
"""
        if t.success_reasons:
            html += '      <ul class="reason-list">\n'
            for r in t.success_reasons:
                html += f'        <li class="success">[成功因素] {html.escape(r)}</li>\n'
            html += '      </ul>\n'
        if t.failure_reasons:
            html += '      <ul class="reason-list">\n'
            for r in t.failure_reasons:
                html += f'        <li class="failure">[失败因素] {html.escape(r)}</li>\n'
            html += '      </ul>\n'
        if t.key_observations:
            html += '      <ul class="reason-list">\n'
            for r in t.key_observations:
                html += f'        <li class="observation">[观察] {html.escape(r)}</li>\n'
            html += '      </ul>\n'
        html += f"""      <p style="margin-top:8px;color:#8b949e;">
        评级: <span class="tag tag-grade-{t.grade.lower()}">{html.escape(t.grade)}</span> — {html.escape(t.grade_reason)}
      </p>
    </div>
  </div>
</div>
"""

    html += f"""
<div class="footer">
  <p>OKX FVG Trading Agent — 历史交易超详细分析报告</p>
  <p>生成时间: {now} (北京时间) | 本报告仅供参考，不构成投资建议</p>
</div>
</div>
</body>
</html>"""

    return html


# ===========================================================================
# 云端上传
# ===========================================================================

def upload_to_cloud(
    html_content: str,
    config: dict,
    filename: str = "",
) -> bool:
    """上传报告到云端（夸克网盘提示 / 本地保存后上传）。

    由于夸克网盘无公开 API，采用以下策略：
      1. 保存到本地 reports/ 目录
      2. 提示用户手动上传到夸克网盘
      3. 如果配置了 SMTP，自动发送邮件

    Args:
        html_content: HTML 报告内容
        config: 配置
        filename: 文件名

    Returns:
        是否成功
    """
    if not filename:
        filename = f"trade_analysis_{datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')}.html"

    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)
    filepath = os.path.join(report_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"报告已保存到本地: {filepath}")
    except Exception as e:
        logger.error(f"保存报告失败: {e}")
        return False

    # 尝试发送邮件
    email_cfg = config.get("report", {}).get("email", {})
    if email_cfg.get("enabled", False):
        try:
            from report import send_email_report
            success = send_email_report(html_content, "历史交易分析", config, filepath)
            if success:
                logger.info("报告已通过邮件发送")
        except Exception as e:
            logger.warning(f"邮件发送失败: {e}")

    # 提示上传夸克网盘
    print(f"\n{'='*60}")
    print(f"  报告已生成: {filepath}")
    print(f"  如需云端分享，请手动上传到夸克网盘:")
    print(f"  https://pan.quark.cn/s/8320adb53d0b")
    print(f"{'='*60}\n")

    return True


# ===========================================================================
# 入口函数
# ===========================================================================

def run_trade_analysis(
    client,
    config: dict,
    days_back: int = 90,
    max_trades: int = 100,
    upload: bool = True,
) -> Optional[str]:
    """一站式历史交易分析入口。

    Args:
        client: OKXClient 实例
        config: 配置
        days_back: 回溯天数
        max_trades: 最大分析笔数
        upload: 是否上传云端

    Returns:
        报告文件路径，失败返回 None
    """
    analyzer = TradeAnalyzer(client, config)

    # 执行分析
    trades = analyzer.run_full_analysis(days_back=days_back, max_trades=max_trades)

    if not trades:
        logger.warning("没有可分析的交易记录")
        return None

    # 生成摘要
    summary = analyzer.get_summary()

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"  分析摘要")
    print(f"{'='*60}")
    print(f"  总交易: {summary['total_trades']} | "
          f"胜率: {summary['win_rate']:.1f}% | "
          f"总盈亏: {summary['total_pnl']:+.2f} USDT")
    print(f"  做多胜率: {summary['long_win_rate']:.1f}% | "
          f"做空胜率: {summary['short_win_rate']:.1f}%")
    print(f"  平均持仓: {summary['avg_holding_hours']:.1f}h")
    print(f"{'='*60}")

    # 生成 HTML 报告
    logger.info("正在生成 HTML 报告...")
    html = generate_html_report(trades, summary, config)

    # 上传云端
    if upload:
        upload_to_cloud(html, config)

    return html


# ===========================================================================
# CLI 入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OKX 历史交易超详细分析工具"
    )
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(__file__), "config.json"),
        help="配置文件路径"
    )
    parser.add_argument(
        "--days", "--天数",
        dest="days",
        type=int, default=90,
        help="回溯天数，默认 90"
    )
    parser.add_argument(
        "--max-trades", "--最大笔数",
        dest="max_trades",
        type=int, default=100,
        help="最多分析笔数，默认 100"
    )
    parser.add_argument(
        "--upload", "--上传",
        dest="upload",
        action="store_true",
        default=True,
        help="生成报告并上传云端"
    )
    parser.add_argument(
        "--no-upload", "--不上传",
        dest="upload",
        action="store_false",
        help="仅本地分析，不上传"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 加载配置
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 初始化客户端
    from okx_client import OKXClient
    client = OKXClient(config)

    # 运行分析
    run_trade_analysis(
        client=client,
        config=config,
        days_back=args.days,
        max_trades=args.max_trades,
        upload=args.upload,
    )


if __name__ == "__main__":
    main()