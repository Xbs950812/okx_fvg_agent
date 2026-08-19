"""
FVG KILLER（公允价值缺口杀手）— OKX 永续合约 FVG 交易主循环。

融合 GitHub Top 3 开源项目精华：
  - freqtrade (52k ⭐): Hyperopt 参数优化 + Edge 分析 + Trailing Stop + FreqAI + Kelly
  - TradingAgents (86k ⭐): 多 Agent 辩论引擎 + 分析师信誉 + 决策反思 + 跨品种经验迁移
  - Vibe-Trading (23.6k ⭐): Alpha Zoo 461 因子库 + 因果滞后体制检测 + 记忆生命周期
    + OKX history-candles 增强加载 + 回测引擎 + 组合优化器 + 相关性分析

运行流程:
  1. 扫描 Top N 合约标的
  2. 五通道数据采集 + 超级交易专家分析
  3. 多空辩论引擎 (TradingAgents) + 因果滞后体制检测 (Vibe-Trading)
  4. Alpha 因子分析 (Vibe-Trading Alpha Zoo 461 因子) + FreqAI 在线预测 (freqtrade)
  5. 自适应参数调整 + 风控门禁
  6. 执行交易（限价单 + Trailing Stop）
  7. 监控持仓，记录决策日志
  8. 定期反思 + Kelly 仓位分析 + Hyperopt 优化
  9. 休眠 → 下一轮扫描

用法:
  python agent.py                    # 实盘模式
  python agent.py --演练             # 演练模式，只分析不下单
  python agent.py --演练 --单轮       # 演练模式，只跑一轮
  python agent.py --演练 --轮次 10    # 演练模式，跑 10 轮
"""

import argparse
import copy
import decimal
import json
import logging
import math
import os
import sys
import time
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd

from okx_client import OKXClient, OKXQueryError
from ws_ticker_cache import WsTickerCache
from strategy import (
    Candle, Signal,
    candles_from_raw,
    scan_fvg_all_timeframes,
)
from fvg_detector import FVGDetector, from_legacy_fvg
from fvg_ml_ranker import FVGMLRanker
from confluence import ConfluenceChecker
from paper_trading import PaperTradingEngine
from executor import (
    StateManager,
    execute_signal,
    monitor_positions,
    manage_pending_orders,
    calculate_spread,
    print_summary,
    get_tradable_coins,
    resolve_full_leverage,
)
from multi_channel import (
    MasterAnalysis,
    MasterTraderEngine,
    full_multi_channel_analysis,
    format_analysis_report,
)
from optimization import (
    TradeRecord, EdgeAnalyzer,
    AdaptiveParameterTuner, TrailingStop,
)
from memory import (
    DecisionLog, MemoryManager,
)
from debate_engine import (
    TradingAgentsDebateEngine,
)
from hyperopt import (
    compute_kelly,
    FreqAIPipeline,
    run_full_optimization,
)
# PRO 模块（可选）: 滚动Kelly/波动率目标/对账/限流等 v3.3 增强功能。
# 开源核心版无 fvg_killer_pro 时自动降级到 v3.2 行为（各调用点守卫）。
try:
    import fvg_killer_pro as _PRO
except ImportError:
    _PRO = None
from alpha_zoo import (
    CausalHysteresisRegime,
)
from coin_tracker import (
    CoinResearchCache, CoinTracker,
    warmup_research,
)
from report import (
    SessionReporter, generate_and_send_report,
)
from trade_analyzer import (
    run_trade_analysis,
)
from persistence import QuantDB
from signal_tracker import SignalPerformanceTracker
from market_guard import MarketEmergencyGuard
from factor_selector import FactorSelector
from walk_forward import WalkForwardAnalyzer
from quant_report import QuantReportGenerator
from risk_committee import RiskCommittee
from cost_model import CostModel
from royalty import RoyaltyManager

# ---- Vibe-Trading 增强模块 (可选导入) ----
try:
    from factor_zoo.adapter import FactorZooAdapter
    _FACTOR_ZOO_AVAILABLE = True
except ImportError:
    _FACTOR_ZOO_AVAILABLE = False

try:
    from backtest_engine import (
        BacktestRunner,
    )
    _BACKTEST_AVAILABLE = True
except ImportError:
    _BACKTEST_AVAILABLE = False

AGENT_VERSION = "v3.3.1"
AGENT_NAME = "FVG KILLER"              # 公允价值缺口杀手

# 修复 M-2: 模块级清理注册表，替代 main_loop 函数对象上的动态属性挂载
# _ws_cache / _tracker 等资源不应挂在函数对象上，造成序列化和 GC 风险
_cleanup_registry: Dict[str, Any] = {}

# 成单率漏斗 (2026-08-07 调研): 信号数→挂单数→成交数, 检测深挂资金空转。
# 等待时长统计(2026-08-07 落地补全): placed_at 记录挂单时间戳, 成交时算
# 挂单等待分钟数, 平均等待超 max_wait_minutes 报警(挂单长时间不成交=深挂空转)。
_FILL_FUNNEL: Dict[str, Any] = {
    "signals": 0,
    "placed": 0,
    "filled": 0,
    "counted_filled": set(),
    "placed_at": {},          # inst_id → 挂单时间戳 (等待时长统计)
    "wait_times_min": [],     # 已成交挂单的等待时长(分钟)
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

class _FlushStreamHandler(logging.StreamHandler):
    """emit 后立即 flush 的 StreamHandler。

    修复: Windows 下 stdout 被管道/重定向时为块缓冲(非行缓冲)，
    终端里 INFO 日志会成块延迟出现，无法实时看到 [EntryLimit]/[EntryLog]
    等诊断日志。每次 emit 强制 flush 保证实时可见。
    """

    def emit(self, record):
        try:
            super().emit(record)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(config: dict):
    """配置日志：同时输出到控制台和文件。"""
    level = getattr(logging, config["agent"].get("log_level", "INFO"))
    log_file = config["agent"].get("log_file", "agent.log")

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # 控制台 — 统一为实时 flush 的 handler
    _console = None
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, _FlushStreamHandler):
            _console = h
            break
    if _console is not None:
        # 已存在(如第三方 basicConfig 添加) → 升级为实时 flush + 对齐级别
        _console.setLevel(level)
        _console.setFormatter(fmt)
        _console.flush = getattr(_console, "flush", lambda: None)
    else:
        ch = _FlushStreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # 文件 — 按天轮转，防止无限增长
    if log_file:
        from logging.handlers import TimedRotatingFileHandler
        fh = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1,
            backupCount=30, encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # 减少第三方库日志噪音
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 加载配置
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 研判挡位系统
# ---------------------------------------------------------------------------

def check_signal_price_fresh(client, signal, max_dev_pct: float = 3.0):
    """执行前价格新鲜度校验。

    修复: 异常波动币种 (abnormal=True) 的信号基于缓存 K 线生成，执行时
    实时价可能已剧烈脱离信号价（曾出现 deviation 85% 的"开单→撤单"空转，
    白白消耗手续费）。执行前用实时标记价对比信号 entry，偏离超阈值直接跳过。

    Returns:
        (ok, live_price) — ok=False 表示价格已脱节应跳过执行
    """
    if signal is None:
        return True, 0.0
    try:
        entry = float(signal.entry_price or 0)
    except (TypeError, ValueError):
        return True, 0.0
    if entry <= 0:
        return True, 0.0
    try:
        live = client.get_mark_price(signal.inst_id)
        if live is None or live <= 0:
            return True, 0.0  # 无法获取实时价，不拦截
    except Exception:
        return True, 0.0
    dev = abs(live - entry) / entry
    if dev > max_dev_pct / 100.0:
        logger.info(
            f"[PriceCheck] {signal.inst_id} 信号价 {entry} vs 实时价 {live} "
            f"偏离 {dev:.2%} > {max_dev_pct:.1f}%，信号已脱节，跳过执行"
        )
        return False, live
    return True, live


def _switch_cost_surcharge(config: dict) -> float:
    """方案B: 换仓往返成本换算成评分门槛附加项。

    频繁换仓 = 平仓费 + 开仓费 + 双向滑点（实测 STRK 换仓单笔损耗 1~2.3 USDT）。
    换仓门槛从固定 min_switch_score_improvement 提升为:
        effective = min_switch_score_improvement + round_trip_cost / edge_per_score

    Args:
        config: 完整配置

    Returns:
        评分附加项(≥0)。配置为 0 或非法时返回 0（退化为基础门槛）。
    """
    scfg = config.get("strategy", {}) or {}
    edge_per_score = float(scfg.get("switch_cost_edge_pct", 1.5))
    if edge_per_score <= 0:
        return 0.0
    cost_pct = float(scfg.get("switch_round_trip_cost_pct", 0.3))
    if cost_pct <= 0:
        return 0.0
    return cost_pct / edge_per_score


def _switch_funding_guards(
    config: dict,
    client: Any,
    old_inst_id: str,
    target_rate: Optional[float],
    target_side: str,
) -> Tuple[bool, str]:
    """换仓资金费双守卫。

    方案A(结算保护): 持仓距下次资金费结算 < switch_funding_lockout_hours 时不换仓，
        让当前持仓跨过结算点收完费率再换（8h 结算点 00/08/16 UTC 收不到=白扛）。
    方案D(费率顺向过滤): 换仓目标方向吃费率才换 —— 做空目标需正费率/做多需负费率，
        否则换过去还要付费率（杜绝"换仓后开始付费率"）。

    Returns:
        (ok, reason) — ok=False 表示应拦截换仓，reason 为日志说明
    """
    scfg = config.get("strategy", {}) or {}

    # 方案D: 目标费率顺向
    if scfg.get("switch_funding_direction_required", True) and target_rate is not None:
        min_abs = float(scfg.get("funding_confluence_min_abs", 0.0003))
        if target_side == "short" and target_rate < -min_abs:
            return False, (
                f"[SwitchFund] 目标做空但费率 {target_rate:+.4%} 负向(做空需付费率)，"
                f"拒绝换仓")
        if target_side == "long" and target_rate > min_abs:
            return False, (
                f"[SwitchFund] 目标做多但费率 {target_rate:+.4%} 正向(做多需付费率)，"
                f"拒绝换仓")

    # 方案A: 结算保护 — 持仓临近结算点不换
    lockout_h = float(scfg.get("switch_funding_lockout_hours", 3))
    if lockout_h > 0 and client is not None and old_inst_id:
        try:
            _finfo = client.get_funding_info(old_inst_id)
            if _finfo:
                _next_ts = float(_finfo[1])
                _wait_h = (_next_ts - time.time()) / 3600.0
                if 0 < _wait_h < lockout_h:
                    return False, (
                        f"[SwitchFund] 持仓 {old_inst_id} 距资金费结算 "
                        f"{_wait_h:.2f}h < {lockout_h:.1f}h，等收完费率再换")
        except Exception as _e:
            logger.debug(f"[SwitchFund] {old_inst_id} 结算时间查询失败: {_e}")
    return True, ""


def _switch_guards(
    config: dict,
    client: Any,
    cache: Optional["CoinResearchCache"],
    cur_inst_id: str,
    cur_score: float,
    cur_c_time: float,
    new_inst_id: str,
    new_score: float,
    new_side: str,
    new_funding_rate: Optional[float],
) -> Tuple[bool, str]:
    """换仓统一守卫 — 换仓绞肉机防线。

    整合四重守卫（主换仓路径与缓存预换仓路径共用，杜绝"换仓绞肉机" —
    实测 11 笔平仓 100% 都是 signal_switch，5~40 分钟换一仓白吃双倍手续费）：

      1. 最小持仓时长 (risk.min_hold_hours): 持仓不足该时长禁止换仓，
         给 FVG 论点留足展开时间 (FVG 回补周期小时级，5 分钟换仓必死)。
      2. 评分门槛: 新信号 final_score 需 ≥ 持仓 master_score +
         min_switch_score_improvement + 往返成本等价分 (方案B)。
      3. 资金费双守卫 (方案A 结算保护 + 方案D 目标费率顺向)。
      4. 相关性: 1H 对数收益率相关 > 0.7 拒绝换仓 (高相关切换=白交手续费)。

    Returns:
        (ok, reason) — ok=False 表示拒绝换仓，reason 为日志说明
    """
    # ---- 守卫 1: 最小持仓时长 (换仓冷却) ----
    min_hold_h = float(config.get("risk", {}).get("min_hold_hours", 4.0) or 0)
    if min_hold_h > 0 and cur_c_time > 0:
        _hold_h = (time.time() - cur_c_time) / 3600.0
        if _hold_h < min_hold_h:
            return False, (
                f"[Switch] 持仓 {cur_inst_id} 仅 {_hold_h:.1f}h < "
                f"min_hold {min_hold_h:.0f}h，换仓冷却期内，维持当前持仓")

    # ---- 守卫 2: 评分门槛 (基础分差 + 往返成本等价分) ----
    _min_improve = float(
        config.get("strategy", {}).get("min_switch_score_improvement", 0.2) or 0)
    _cost_score = _switch_cost_surcharge(config)
    _eff_improve = _min_improve + _cost_score
    if new_score < cur_score + _eff_improve:
        return False, (
            f"[Switch] {new_inst_id} 评分 {new_score:.2f} 未明显优于 "
            f"持仓 {cur_inst_id} 评分 {cur_score:.2f} "
            f"(需 +{_eff_improve:.2f} = 基础{_min_improve:.2f} "
            f"+ 成本{_cost_score:.2f})，维持当前持仓")

    # ---- 守卫 3: 资金费双守卫 (结算保护 + 目标费率顺向) ----
    _fund_ok, _fund_reason = _switch_funding_guards(
        config, client, cur_inst_id, new_funding_rate, new_side)
    if not _fund_ok:
        return False, _fund_reason

    # ---- 守卫 4: 相关性 (高相关切换 = 白交手续费) ----
    _corr = _compute_correlation_1h(cache, cur_inst_id, new_inst_id)
    if _corr is not None and _corr > 0.7:
        return False, (
            f"[Switch] 相关性 {_corr:.2f} > 0.7，拒绝换仓: "
            f"{cur_inst_id} → {new_inst_id}")
    if _corr is None:
        logger.debug(
            f"[Switch] 相关性检查: {cur_inst_id} ↔ {new_inst_id} "
            f"样本不足或数据缺失，跳过相关性检查")
    return True, ""


def _pick_switch_candidate(
    cached_signals: list, positions: dict
) -> Optional[Tuple[Any, dict]]:
    """从缓存候选中选出换仓目标（跳过无有效 FVG 信号的条目）。

    候选选择与换仓比较统一使用 final_score — 原实现按 final_confidence 选
    最优但按 final_score 比较，度量不一致导致"选中置信度高但评分未必高"的
    标的反复触发换仓。

    FVG Hunter 硬门禁一致性 (2026-08-08): 缓存研究分(final_score)可能虚高
    但信号已被 ExtremeMove/宽度/新鲜度门禁拒绝 (PIPPIN 实测: 研究分 +1.00
    但 4H ADX=13 全部信号被拒，仍混入换仓候选)。无有效 FVG 信号的条目
    不得参与换仓 — 与主扫描路径 (all_signals.extend(entry.signals)) 口径
    一致，杜绝横盘币绕门禁入场。

    Returns:
        (entry, coin_info) 或 None（无合格候选）
    """
    _best_cached = None
    _best_cached_score = -999.0
    for _cached_entry, _cached_coin_info in (cached_signals or []):
        if _cached_entry.inst_id in positions:
            continue
        if not _cached_entry.signals:
            logger.debug(
                f"[Switch] {_cached_entry.inst_id} 无有效 FVG 信号"
                f"(门禁已拒绝)，排除换仓候选")
            continue
        _cand_score = (
            _cached_entry.analysis.final_score
            if _cached_entry.analysis and _cached_entry.analysis.final_score
            else 0.0
        )
        if _cand_score > _best_cached_score:
            _best_cached_score = _cand_score
            _best_cached = (_cached_entry, _cached_coin_info)
    return _best_cached


def _weak_signal_multi_gate(
    config: dict,
    client: Any,
    signal: Signal,
    analysis: Optional[Any],
    candles_1h: List[Candle],
) -> Tuple[bool, str]:
    """弱信号多指标共振审核（"赌一把"防线）。

    用户要求: 弱信号(想赌一把)下单时必须参考多个技术指标——交易量/
    换手率/多空比/资金费率/趋势，杜绝 SAHARA 式失算(仅凭单一 FVG 结构
    + 负分信号就赌方向, 低位"地板空"被反弹扫损 -15.24 USDT)。

    判定: signal.score < score_threshold 或 final_confidence <
    confidence_threshold 时视为弱信号，触发共振审核；强信号直接放行
    (强信号已经过多通道分析/辩论/ML/汇流完整管线把关)。

    共振指标 (至少 min_confluence 项顺向才放行):
      1. 量能:   近3根均量 / 前20根均量 ≥ volume_ratio_min (放量顺向)
      2. 多空比: long_short_ratio 方向顺向 (long≥lsr_long_min, short≤lsr_short_max)
      3. 资金费率: 做空吃正费率 / 做多吃负费率 (费率收割顺风)
      4. 换手:   OI>0 时近24根累计量/OI ≥ turnover_min (活跃度高)
      5. 趋势:   close 与 SMA(trend_ma) 方向顺向

    数据不足(某项指标不可得)时该项不计入顺向计数；可用指标 < min_confluence
    时放行(不阻塞主循环，与 ML/汇流一致的降级策略)。任何指标异常不抛异常。

    Returns:
        (ok, reason) — ok=False 表示弱信号共振不足应拒绝下单
    """
    wcfg = config.get("strategy", {}).get("weak_signal_gate", {}) or {}
    if not wcfg.get("enabled", True):
        return True, ""

    # 强信号判定阈值随挡位缩放 — 修复: 硬编码 0.45/0.50 是保守挡位标准,
    # 激进挡位主门禁 min_confidence=0.25/min_factor_score=0.20, 配置为 0 时
    # 自动取挡位阈值。否则 conf 0.25~0.50 的高分信号(如 ADA score=1.00
    # conf=40%)被主门禁判定为强信号, 却被 WeakGate 按 0.50 标准当弱信号
    # 推入共振审核 → 2/4 共振不足被拒, 挡位缩放完全失效。
    _agg_mode = int(config.get("agent", {}).get("aggressiveness", 3) or 3)
    _agg_t = get_aggressiveness_thresholds(_agg_mode)
    try:
        score_threshold = float(wcfg.get("score_threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        score_threshold = 0.0
    try:
        conf_threshold = float(wcfg.get("confidence_threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf_threshold = 0.0
    if score_threshold <= 0:
        score_threshold = _agg_t["min_factor_score"] / 100.0
    if conf_threshold <= 0:
        conf_threshold = _agg_t["min_confidence"]

    # ---- 弱信号判定 ----
    sig_score = float(getattr(signal, "score", 0.0) or 0.0)
    conf = 0.0
    if analysis is not None:
        try:
            conf = float(getattr(analysis, "final_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
    if sig_score >= score_threshold and conf >= conf_threshold:
        return True, ""  # 强信号，直接放行

    inst_id = signal.inst_id
    direction = signal.position_side

    # ---- 收集各指标 ----
    checks: Dict[str, bool] = {}  # 指标名 -> 是否顺向
    try:
        min_confluence = max(1, int(wcfg.get("min_confluence", 3)))
    except (TypeError, ValueError):
        min_confluence = 3

    # 1. 量能 (放量顺向)
    try:
        vols = [float(c.volume or 0) for c in candles_1h if getattr(c, "volume", 0) > 0]
        if len(vols) >= 24:
            _recent = sum(vols[-3:]) / 3.0
            _base = sum(vols[-23:-3]) / 20.0
            if _base > 0:
                _vr = _recent / _base
                _vr_min = float(wcfg.get("volume_ratio_min", 1.2))
                checks["量能"] = _vr >= _vr_min
            else:
                checks["量能"] = False
        # 数据不足时不判定该项(保持未判定状态)
    except Exception:
        pass

    # 2. 多空比 (方向顺向)
    try:
        _lsr = client.get_long_short_ratio(inst_id, period="1H")
        if _lsr is not None and _lsr > 0:
            _lsr_long_min = float(wcfg.get("lsr_long_min", 1.05))
            _lsr_short_max = float(wcfg.get("lsr_short_max", 0.95))
            if direction == "long":
                checks["多空比"] = _lsr >= _lsr_long_min
            else:
                checks["多空比"] = _lsr <= _lsr_short_max
    except Exception:
        pass

    # 3. 资金费率 (做空吃正/做多吃负)
    try:
        _fr = client.get_funding_rate(inst_id)
        if _fr is not None:
            if direction == "short":
                checks["资金费率"] = _fr > 0
            else:
                checks["资金费率"] = _fr < 0
    except Exception:
        pass

    # 4. 换手 (近24根累计量 / OI, 活跃度)
    try:
        _oi = client.get_open_interest(inst_id)
        if _oi is not None and _oi > 0:
            _vol24 = sum(float(c.volume or 0)
                         for c in candles_1h[-24:]
                         if getattr(c, "volume", 0) > 0)
            if _vol24 > 0:
                _turnover = _vol24 / _oi
                _turnover_min = float(wcfg.get("turnover_min", 2.0))
                checks["换手"] = _turnover >= _turnover_min
    except Exception:
        pass

    # 5. 趋势 (close vs SMA)
    try:
        closes = [float(c.close) for c in candles_1h if getattr(c, "close", 0) > 0]
        _ma = int(wcfg.get("trend_ma", 20))
        if len(closes) >= _ma:
            _sma = sum(closes[-_ma:]) / _ma
            _last = closes[-1]
            if direction == "long":
                checks["趋势"] = _last > _sma
            else:
                checks["趋势"] = _last < _sma
    except Exception:
        pass

    # ---- 汇总判定 ----
    total_ok = sum(1 for v in checks.values() if v)
    total = len(checks)
    detail = " ".join(f"{k}={'✓' if v else '✗'}" for k, v in checks.items())
    if total < min_confluence:
        # 修复: 弱信号数据不足不再降级放行 — 数据越稀缺越放行与设计意图相反。
        # 可用指标不足无法证明共振，直接拒绝弱信号（强信号仍走前置放行逻辑）。
        logger.info(
            f"[WeakGate] {inst_id} 弱信号但可用指标 {total} < {min_confluence}，"
            f"数据不足无法共振，拒绝下单 (score={sig_score:.2f} conf={conf:.0%})")
        return False, (
            f"[WeakGate] {inst_id} 弱信号可用指标 {total} < {min_confluence}，"
            f"数据不足无法共振，拒绝下单")
    if total_ok >= min_confluence:
        logger.info(
            f"[WeakGate] {inst_id} 弱信号共振 {total_ok}/{total} 顺向 "
            f"(score={sig_score:.2f} conf={conf:.0%}) {detail} → 放行")
        return True, ""
    logger.info(
        f"[WeakGate] {inst_id} 弱信号共振 {total_ok}/{total} < {min_confluence}，"
        f"拒绝赌单 (score={sig_score:.2f} conf={conf:.0%}) {detail}")
    return False, (
        f"[WeakGate] {inst_id} 弱信号多指标共振不足 "
        f"({total_ok}/{total} < {min_confluence})，拒绝下单")


def _open_funding_settlement_guard(
    client: Optional[OKXClient],
    inst_id: str,
    side: str,
    funding_rate: Optional[float],
    config: dict,
) -> Tuple[bool, str]:
    """直接开仓结算窗口守卫(2026-08-07 调研落地)。

    距下次资金费结算 < settlement_lockout_hours 且目标方向需付费率时,
    推迟到结算后开仓, 避免刚开仓就被吸一口费率 (8h结算点 00/08/16 UTC)。

    注: 仅做结算窗口推迟, 不做方向否决 — 方向否决对首次开仓过严(正费率
    市场会永远做不了多)。极端费率(|rate|>max_funding_rate_abs=0.01)已由
    generate_signal 过滤器覆盖。

    Returns:
        (ok, reason)
    """
    og = config.get("strategy", {}).get("open_funding_guard") or {}
    if not og.get("enabled", True):
        return True, ""
    try:
        lockout_h = float(og.get("settlement_lockout_hours", 1.0))
    except (TypeError, ValueError):
        lockout_h = 1.0
    if lockout_h <= 0 or client is None or funding_rate is None:
        return True, ""
    _pays = (side == "long" and funding_rate > 0) or (
        side == "short" and funding_rate < 0)
    if not _pays:
        return True, ""
    try:
        _finfo = client.get_funding_info(inst_id)
        if not _finfo:
            return True, ""
        _next_ts = float(_finfo[1])
        _wait_h = (_next_ts - time.time()) / 3600.0
        if 0 < _wait_h < lockout_h:
            return False, (
                f"[OpenFund] {inst_id} {side} 需付费率且距结算 "
                f"{_wait_h:.2f}h < {lockout_h:.1f}h，推迟到结算后开仓")
    except Exception as _e:
        logger.debug(f"[OpenFund] {inst_id} 结算时间查询失败: {_e}")
    return True, ""


def _expectancy_guard(
    state_manager: StateManager,
    config: dict,
    inst_id: str,
) -> Tuple[bool, str]:
    """期望值门禁(2026-08-07 调研): 近 window 笔已实现盈亏均值 < min_avg_pnl
    时暂停开仓; 均值未达暂停线但仍为负(avg < degrade_avg_pnl)时降频——
    冷却 degrade_cooldown_minutes 内拦截开仓, 冷却到期后重新评估。研究原文
    要求"自动降频/暂停"双档, 避免负期望策略持续开仓白亏手续费。

    Returns:
        (ok, reason)
    """
    eg = config.get("risk", {}).get("expectancy_guard") or {}
    if not eg.get("enabled", True):
        return True, ""
    try:
        window = int(eg.get("window", 20))
        min_avg = float(eg.get("min_avg_pnl", 0.0))
    except (TypeError, ValueError):
        return True, ""
    if window <= 0:
        return True, ""  # 非法窗口(≤0)不拦截; 防止 recent[-window:] 变全列表切片
    try:
        with state_manager.lock():
            recent = list(state_manager.state.recent_pnl or [])
            _deg_until = float(state_manager.state.ev_degrade_until or 0.0)
    except Exception:
        return True, ""
    if len(recent) < max(window, 5):
        return True, ""  # 样本不足不拦截
    _w = recent[-window:]
    avg = sum(_w) / len(_w)

    # ---- 档位 1: 暂停 (均值 < min_avg_pnl, 硬停止) ----
    if avg < min_avg:
        return False, (
            f"[Expectancy] {inst_id} 暂停开仓: 近{len(_w)}笔已实现盈亏均值 "
            f"{avg:+.3f} < {min_avg:+.3f}(负期望)，等待盈利单转正")

    # ---- 档位 2: 降频 (均值在 [min_avg, degrade_avg_pnl) 区间, 冷却期拦截) ----
    if eg.get("degrade_enabled", True):
        try:
            _deg_avg = float(eg.get("degrade_avg_pnl", 0.0))
        except (TypeError, ValueError):
            _deg_avg = 0.0
        if avg < _deg_avg:
            _now = time.time()
            if _now < _deg_until:
                return False, (
                    f"[Expectancy] {inst_id} 降频冷却中(均值 {avg:+.3f}<{_deg_avg:+.3f})，"
                    f"本轮跳过开仓")
            try:
                _cooldown_min = float(eg.get("degrade_cooldown_minutes", 60))
            except (TypeError, ValueError):
                _cooldown_min = 60.0
            if _cooldown_min <= 0:
                _cooldown_min = 60.0
            try:
                with state_manager.lock():
                    state_manager.state.ev_degrade_until = _now + _cooldown_min * 60.0
            except Exception:
                pass
            return False, (
                f"[Expectancy] {inst_id} 触发降频(均值 {avg:+.3f}<{_deg_avg:+.3f})，"
                f"冷却 {_cooldown_min:.0f}min 内暂停开仓")
    return True, ""


def _check_ce_invalidation(
    cache: Optional["CoinResearchCache"],
    inst_id: str,
    pos: Dict[str, Any],
    state_manager: StateManager,
    config: dict,
) -> Tuple[bool, float]:
    """CE 中点失效检查（53年DXY回测/2026-08-07 seasonaledge）。

    价格实体收盘越过缺口中间值(consequent encroachment)即认定结构失效:
      多头 → 1H 实体收盘 < FVG 中点 → 失效
      空头 → 1H 实体收盘 > FVG 中点 → 失效
    失效后止损提到成本价(0R)，而不是死等原止损(-1R) —
    回测: CE 提前退出把 785 笔全损砍到 -0.30R。

    Returns:
        (invalidated, fvg_mid)
    """
    cecfg = config.get("strategy", {}).get("ce_invalidation") or {}
    if not cecfg.get("enabled", True):
        return False, 0.0
    # 从 active_signals 读取该持仓的信号 FVG 中点
    fvg_mid = 0.0
    try:
        with state_manager.lock():
            fvg_mid = float(
                state_manager.state.active_signals.get(
                    inst_id, {}).get("signal_fvg_mid", 0.0) or 0.0)
    except Exception:
        fvg_mid = 0.0
    if fvg_mid <= 0:
        return False, 0.0
    if cache is None:
        return False, 0.0
    entry = cache.get(inst_id)
    if entry is None:
        return False, 0.0
    candle_1h = (entry.candles_by_tf.get("1H") or [None])[-1]
    if candle_1h is None:
        return False, 0.0
    try:
        body_close = float(candle_1h.close)
    except (TypeError, ValueError, AttributeError):
        return False, 0.0
    if body_close <= 0:
        return False, 0.0
    side = pos.get("pos_side", "long")
    if side == "long" and body_close < fvg_mid:
        return True, fvg_mid
    if side == "short" and body_close > fvg_mid:
        return True, fvg_mid
    return False, 0.0


def _htf_alignment_gate(
    signal: Signal,
    candles_htf: Optional[List[Candle]],
    config: dict,
) -> Tuple[bool, str]:
    """HTF 方向门（大周期定方向，2026-08-07 调研落地）。

    首小时区间回测(seasonaledge 2987笔): 顺趋势突破 64% 胜率 / 逆趋势假突破
    仅 3.9% 胜率。信号方向需与高周期均线方向一致:
      做多 → HTF 最新价 > SMA(ma_period)
      做空 → HTF 最新价 < SMA(ma_period)
    无 HTF 数据 / 数据不足时放行(不阻塞, 防数据缺失饿死主循环)。

    Returns:
        (ok, reason)
    """
    if signal is None:
        return True, ""
    hcfg = config.get("strategy", {}).get("htf_alignment") or {}
    if not hcfg.get("enabled", True):
        return True, ""
    if not candles_htf or len(candles_htf) < 3:
        return True, ""
    ma_period = int(hcfg.get("ma_period", 20))
    if len(candles_htf) < ma_period + 1:
        return True, ""
    closes = [float(c.close) for c in candles_htf[-ma_period:]]
    if not closes or min(closes) <= 0:
        return True, ""
    sma = sum(closes) / len(closes)
    last_px = float(candles_htf[-1].close)
    if last_px <= 0:
        return True, ""
    direction = signal.position_side
    if direction == "long" and last_px < sma:
        return False, (
            f"[HTFGate] {signal.inst_id} 做多但高周期价 {last_px} < "
            f"SMA{ma_period} {sma:.6g}，逆大周期趋势，拒绝（顺趋势64% vs 逆趋势3.9%）")
    if direction == "short" and last_px > sma:
        return False, (
            f"[HTFGate] {signal.inst_id} 做空但高周期价 {last_px} > "
            f"SMA{ma_period} {sma:.6g}，逆大周期趋势，拒绝（顺趋势64% vs 逆趋势3.9%）")
    return True, ""


def _direction_momentum_gate(
    signal: Signal,
    candles_1h: Optional[List[Candle]],
    config: dict,
) -> Tuple[bool, str]:
    """方向动量一致性门 (2026-08-10 方案A): 开仓方向与 1H 短期趋势明确冲突时否决。

    背景: HTF 方向门(4H SMA20)是滞后指标 — 1H 已转跌时 4H 仍向上, 系统逆着
    1H 短期趋势开多 (实测 PUMP score=0.91 做多 @0.002855 后价格一路跌到
    -4.6%, 方向错的根因之一是逆 1H 趋势开仓)。
    本门用 1H SMA(trend_ma_period) + SMA 斜率判定短期趋势:
      做多冲突 = close < SMA 且 SMA 下行 (明确空头趋势)
      做空冲突 = close > SMA 且 SMA 上行 (明确多头趋势)
    仅"明确冲突"否决; 横盘(SMA 走平)/数据不足放行, 避免误杀。

    Returns:
        (ok, reason)
    """
    if signal is None:
        return True, ""
    mcfg = config.get("strategy", {}).get("direction_momentum_gate") or {}
    if not mcfg.get("enabled", True):
        return True, ""
    if not candles_1h or len(candles_1h) < 10:
        return True, ""
    ma_period = int(mcfg.get("ma_period", 20) or 20)
    if len(candles_1h) < ma_period + 6:
        return True, ""
    try:
        closes_now = [float(c.close) for c in candles_1h[-ma_period:] if c.close > 0]
        if len(closes_now) < ma_period:
            return True, ""
        sma_now = sum(closes_now) / len(closes_now)
        # SMA 斜率: 当前 SMA vs 5 根前的 SMA (同一均线窗口两个时间点)
        closes_prev = [float(c.close) for c in candles_1h[-(ma_period + 5):-5]
                       if c.close > 0]
        if len(closes_prev) < ma_period:
            return True, ""
        sma_prev = sum(closes_prev) / len(closes_prev)
        last_px = float(candles_1h[-1].close or 0)
    except (TypeError, ValueError):
        return True, ""
    if last_px <= 0 or sma_now <= 0 or sma_prev <= 0:
        return True, ""
    direction = signal.position_side
    if direction == "long" and last_px < sma_now and sma_now < sma_prev:
        return False, (
            f"[MomentumGate] {signal.inst_id} 做多但 1H 短期趋势向下 "
            f"(close {last_px:.6g} < SMA{ma_period} {sma_now:.6g}, "
            f"SMA 下行 {sma_prev:.6g}→{sma_now:.6g})，逆 1H 趋势，拒绝")
    if direction == "short" and last_px > sma_now and sma_now > sma_prev:
        return False, (
            f"[MomentumGate] {signal.inst_id} 做空但 1H 短期趋势向上 "
            f"(close {last_px:.6g} > SMA{ma_period} {sma_now:.6g}, "
            f"SMA 上行 {sma_prev:.6g}→{sma_now:.6g})，逆 1H 趋势，拒绝")
    return True, ""


def _red_flag_gate(
    signal: Signal,
    analysis: Optional[Any],
    config: dict,
) -> Tuple[bool, str]:
    """区间位置红旗门禁 — 强制执行多通道红旗。

    教训固化 (SAHARA/CHZ 复盘): "价格处于区间低位，不宜追空" 等红旗过去
    只被写进 memory 日志从未执行，导致系统在区间底部反复开空被反弹扫损
    (SAHARA -15.24 / CHZ -6.68 USDT)。本门禁对方向性红旗直接否决同向信号。

    覆盖所有开仓路径: 直接开仓/换仓/反手/金字塔加仓/平仓后重开 (由
    _execute_signal_with_quant_enhancements 统一调用)。

    Returns:
        (ok, reason) — ok=False 表示应拒绝该信号
    """
    if signal is None:
        return True, ""
    if not config.get("strategy", {}).get("enforce_red_flags", True):
        return True, ""
    if analysis is None:
        return True, ""
    flags = getattr(analysis, "key_risks", None) or []
    if not flags:
        return True, ""
    direction = signal.position_side
    for _f in flags:
        _fs = str(_f)
        if direction == "short" and "不宜追空" in _fs:
            return False, (
                f"[RedFlagGate] {signal.inst_id} 做空被区间位置红旗否决: {_fs}")
        if direction == "long" and "不宜追多" in _fs:
            return False, (
                f"[RedFlagGate] {signal.inst_id} 做多被区间位置红旗否决: {_fs}")
    return True, ""


def get_aggressiveness_thresholds(mode: int) -> dict:
    """根据研判挡位返回调整后的阈值参数。

    挡位说明:
      1 = 激进 — 每天务必找到一个币种建仓，大幅降低阈值，无可选信号时强制选最优
      2 = 均衡 — 2-3 天操作一笔，适度降低阈值
      3 = 保守 — 默认参数，严格门禁

    Returns:
        dict: 包含所有调整后阈值的字典
    """
    levels = {
        # 挡位 1: 激进
        1: {
            "min_confidence": 0.25,       # 原 0.40
            "min_agreement": 0.25,         # 原 0.50
            "min_prediction_confidence": -2.0,  # 原 -0.5，几乎不过滤
            "min_fvg_width_1h": 0.8,       # 原 1.5
            "min_fvg_width_4h": 1.5,       # 原 3.0
            "abnormal_sigma": 2.0,         # 原 3.0
            "abnormal_volume_ratio": 3.0,   # 原 5.0
            "min_factor_score": 20,         # 原 40
            "label": "激进",
        },
        # 挡位 2: 均衡
        2: {
            "min_confidence": 0.33,
            "min_agreement": 0.35,
            "min_prediction_confidence": -1.0,
            "min_fvg_width_1h": 1.2,
            "min_fvg_width_4h": 2.2,
            "abnormal_sigma": 2.5,
            "abnormal_volume_ratio": 4.0,
            "min_factor_score": 30,
            "label": "均衡",
        },
        # 挡位 3: 保守（默认）
        3: {
            "min_confidence": 0.40,
            "min_agreement": 0.50,
            "min_prediction_confidence": -0.5,
            "min_fvg_width_1h": 1.5,
            "min_fvg_width_4h": 3.0,
            "abnormal_sigma": 3.0,
            "abnormal_volume_ratio": 5.0,
            "min_factor_score": 40,
            "label": "保守",
        },
    }
    if mode not in levels:
        logger.warning(f"未知挡位 {mode}，使用保守挡位 3")
        return levels[3]
    return levels[mode]


def apply_aggressiveness_to_config(config: dict, thresholds: dict) -> dict:
    """将挡位阈值应用到策略配置中，返回调整后的配置副本。

    不修改原始 config，返回深拷贝。
    """
    cfg = copy.deepcopy(config)
    s = cfg["strategy"]
    t = thresholds

    # 调整 FVG 最小宽度
    if "min_fvg_width_pct" not in s:
        s["min_fvg_width_pct"] = {}
    s["min_fvg_width_pct"]["1H"] = t["min_fvg_width_1h"]
    s["min_fvg_width_pct"]["4H"] = t["min_fvg_width_4h"]

    # 调整异常检测
    s["abnormal_sigma"] = t["abnormal_sigma"]
    s["abnormal_volume_ratio"] = t["abnormal_volume_ratio"]

    # 调整多通道与预测阈值
    s["min_confidence"] = t["min_confidence"]
    s["min_agreement"] = t["min_agreement"]
    s["min_prediction_confidence"] = t["min_prediction_confidence"]

    # 调整 Alpha Zoo 因子评分阈值
    if "alpha_zoo" in cfg:
        cfg["alpha_zoo"]["min_factor_score"] = t["min_factor_score"]

    return cfg


# ---------------------------------------------------------------------------
# 单轮扫描结果
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """单轮扫描结果，包含中间数据供后续多通道分析复用。"""
    signals: List[Signal]
    candles_by_tf: Dict[str, List[Candle]]
    funding_rate: Optional[float]
    spread: float


def scan_round(
    client: OKXClient,
    coin: dict,
    config: dict,
    ws_cache: Optional["WsTickerCache"] = None,
) -> ScanResult:
    """对单个合约执行一轮完整扫描。

    Returns:
        ScanResult: 信号列表 + K 线数据 + 资金费率 + 价差，供后续多通道分析复用
    """
    inst_id = coin["instId"]
    original_coin = coin  # 保留原始数据，避免 WS 覆盖丢失 bidPx/askPx

    # WebSocket 缓存回退: 优先使用实时推送的价格，延迟 < 50ms
    # 修复: 只更新 last 价格，不覆盖整个 coin 对象
    # 原因: ws_ticker 可能只包含 last 字段，覆盖会丢失 bidPx/askPx
    # 导致 calculate_spread(0, 0) = 0，造成流动性完美的假象
    if ws_cache is not None:
        ws_ticker = ws_cache.get(inst_id)
        if ws_ticker and ws_ticker.get("last", 0) > 0:
            current_price = ws_ticker["last"]
        else:
            current_price = coin.get("last", 0)
    else:
        current_price = coin.get("last", 0)

    # 价格有效性守卫: 0/负数/None 直接跳过
    if not current_price or current_price <= 0:
        logger.warning(f"Invalid price {current_price} for {inst_id}, skip scan")
        return ScanResult(signals=[], candles_by_tf={}, funding_rate=None, spread=0)

    # 从原始 coin 提取买卖盘，不受 WS 缓存覆盖影响
    bid_px = original_coin.get("bidPx", 0)
    ask_px = original_coin.get("askPx", 0)

    logger.debug(f"Scanning {inst_id} @ {current_price}")

    # ---- 获取 K 线 ----
    candles_by_tf: Dict[str, List[Candle]] = {}
    for tf in config["strategy"]["timeframes"]:
        # 使用增强版加载器获取历史数据（支持 history-candles 端点 + 限速重试）
        raw = client.get_candles_enhanced(inst_id, bar=tf, limit=200)
        if not raw:
            # 回退到 SDK 原生方法
            raw = client.get_candles(inst_id, bar=tf, limit=200)
        if raw:
            candles_by_tf[tf] = candles_from_raw(raw)
        else:
            logger.warning(f"No candle data for {inst_id} {tf}")

    if not candles_by_tf:
        return ScanResult(signals=[], candles_by_tf={}, funding_rate=None, spread=0)

    # ---- 获取资金费率 ----
    funding_rate = client.get_funding_rate(inst_id)

    # ---- 获取多空持仓比（综合技术参考，失败返回 None 不阻塞）----
    long_short_ratio = client.get_long_short_ratio(inst_id, period="1H")

    # ---- 计算价差 ----
    spread_pct = calculate_spread(bid_px, ask_px)

    # ---- 扫描信号 ----
    signals = scan_fvg_all_timeframes(
        inst_id=inst_id,
        candles_by_tf=candles_by_tf,
        current_price=current_price,
        config=config,
        funding_rate=funding_rate,
        spread_pct=spread_pct,
        long_short_ratio=long_short_ratio,
    )

    return ScanResult(
        signals=signals,
        candles_by_tf=candles_by_tf,
        funding_rate=funding_rate,
        spread=spread_pct,
    )


# ---------------------------------------------------------------------------
# 风控门禁
# ---------------------------------------------------------------------------

def _position_notional_usd(client: OKXClient, pos: dict) -> float:
    """计算单笔持仓名义价值 (USDT) = |pos| × ctVal × markPx。

    修复 2026-08-13: 原敞口计算缺 ctVal（合约面值）——NEIRO(ctVal=1000)/
    LAB(ctVal=10) 等高面值币种名义敞口被低估 10~1000 倍，max_exposure_pct
    风控形同虚设。OKX position 的 pos 字段是张数，须乘 ctVal 得 USDT 价值。
    """
    try:
        _sz = abs(float(pos.get("pos", "0") or 0))
        _px = float(pos.get("markPx", "0") or 0)
        if _sz <= 0 or _px <= 0:
            return 0.0
        _ct_val = 0.01
        _inst = pos.get("instId", "")
        if _inst:
            _info = client.get_instrument_info(_inst)
            if _info:
                _ct_val = float(_info.get("ctVal", "0.01") or 0.01)
        return _sz * _ct_val * _px
    except Exception:
        return 0.0


def risk_gate(
    client: OKXClient,
    state_manager: StateManager,
    equity: float,
    config: dict,
    active_count: int,  # 必传参数，防止调用方遗漏导致静默绕过多仓检查
) -> Tuple[bool, str]:
    """风控检查，返回 (是否通过, 原因)。

    检查项：
      1. 当前持仓数是否已达上限
      2. 当日亏损是否已达上限
      3. 权益是否归零
      4. 挂单数是否过多
    """
    risk_cfg = config["risk"]
    state = state_manager.state

    # 权益检查
    if equity <= 0:
        return False, "Equity <= 0"

    # 修复: 使用 >= 严格限制最大持仓数，防止 Off-by-One 边界穿透
    # 当 active_count == max_positions 时，风控门禁必须拦截，
    # 换仓逻辑（先平后开）在步骤 1.5 中独立处理，不依赖此处的宽松检查
    # active_count 基于实际持仓数（来自 monitor_positions），比 state.active_signals 更准确
    if active_count >= risk_cfg["max_positions"]:
        return False, f"Positions exceed max ({active_count}/{risk_cfg['max_positions']})"

    # 每日亏损检查
    # 修复 H7: 使用 daily_start_equity 作为基准（与 monitor_positions 一致）
    max_daily_loss = state_manager.state.daily_start_equity * risk_cfg["max_daily_loss_pct"] / 100.0
    # 修复 H7: 使用 daily_start_equity + 累计已实现 PnL 语义检查
    if state.daily_loss <= -max_daily_loss and max_daily_loss > 0:
        return False, (f"Daily loss limit reached: "
                       f"{state.daily_loss:.2f} >= {max_daily_loss:.2f}")

    # 修复 2026-08-07: risk.max_exposure_pct 未生效 — 名义敞口上限检查
    # 持仓名义价值(Σ |pos×markPx|) 不得超过 equity×max_exposure_pct%。
    # config 已配 30%; 原 assess_portfolio_risk 定义了但从未被调用。
    _max_exp = float(risk_cfg.get("max_exposure_pct", 30.0) or 0)
    if _max_exp > 0:
        _notional = 0.0
        try:
            _poss = client.get_positions()
            # 修复 P0-B (fail-closed): 持仓查询失败(None)时敞口未知，
            # 必须拦截开仓而不是当"零敞口"放行
            if _poss is None:
                return False, "Exposure check unavailable (get_positions failed)"
            _notional = sum(_position_notional_usd(client, p) for p in _poss)
        except (ConnectionError, TimeoutError, OSError, ValueError, OKXQueryError) as _ee:
            logger.warning(f"get_positions 失败，拒绝开仓(fail-closed): {_ee}")
            return False, "Exposure check failed (fail-closed)"
        except Exception:
            _notional = 0.0
        _exp_limit = equity * _max_exp / 100.0
        if _exp_limit > 0 and _notional > _exp_limit * 1.001:
            return False, (f"Exposure limit: 名义 {_notional:.2f} > "
                           f"上限 {_exp_limit:.2f} (equity×{_max_exp:.0f}%)")

    # 挂单数检查（避免挂单堆积）
    try:
        pending = client.get_pending_orders() or []
    except (ConnectionError, TimeoutError, OSError, ValueError, OKXQueryError) as _pe:
        logger.warning(f"get_pending_orders 失败，跳过挂单数检查: {_pe}")
        pending = []
    if len(pending) >= risk_cfg["max_positions"] * 2:
        return False, f"Too many pending orders ({len(pending)})"

    return True, "OK"


def _risk_breaker_triggered(state_manager, config,
                            adaptive_tuner=None) -> Tuple[bool, str]:
    """修复 P1-5: daily_loss / 自适应暂停的统一断路器查询。

    换仓预检(step 1.6)与平仓后开新仓(step 1.5)都在 risk_gate(step 4)之前执行，
    此前可绕过日亏限额与自适应暂停（亏损日继续平旧开新）。任何开仓路径
    （换仓/反手/平仓后重开）必须先查此门，命中则一律禁止。
    """
    _risk = config.get("risk", {}) or {}
    _ds = state_manager.state.daily_start_equity
    try:
        _mll = _ds * float(_risk.get("max_daily_loss_pct", 10.0) or 0) / 100.0
    except (TypeError, ValueError):
        _mll = 0.0
    if _mll > 0 and state_manager.state.daily_loss <= -_mll:
        return True, (f"daily_loss 已达上限 ({state_manager.state.daily_loss:.2f} USDT)，"
                      f"禁止换仓/反手/新开仓")
    # 每日最大交易次数 (2026-08-14 防过度交易): 超过上限当日禁止再开仓。
    # 换仓绞肉机(5~40分钟换一仓)会白吃双倍手续费，顶级交易员设每日交易上限。
    try:
        _max_daily_trades = int(_risk.get("max_daily_trades", 0) or 0)
    except (TypeError, ValueError):
        _max_daily_trades = 0
    if _max_daily_trades > 0 and state_manager.state.daily_trades >= _max_daily_trades:
        return True, (
            f"当日交易次数已达上限 ({state_manager.state.daily_trades}/"
            f"{_max_daily_trades})，禁止换仓/反手/新开仓")
    if adaptive_tuner is not None:
        try:
            _pu = float(getattr(adaptive_tuner, "pause_until", 0) or 0)
            if _pu > time.time():
                return True, (
                    f"自适应暂停中(连亏/回撤断路器, 至 "
                    f"{datetime.fromtimestamp(_pu, timezone.utc).strftime('%H:%M')} UTC)"
                )
            if getattr(adaptive_tuner, "trading_paused", False):
                return True, "自适应调参器 trading_paused"
        except Exception:
            pass
    return False, ""


def _exposure_cap_allows_add(client, equity, config) -> bool:
    """修复 P1-6: 聚合名义敞口是否允许再加一笔（保留 30% 余量给新单）。

    金字塔加仓此前只查"币种仍在持仓"，无聚合敞口上限 —— 多轮 0.5× 加仓
    可把单币名义敞口推到数倍于 max_exposure_pct，且每笔各挂独立止损单，
    触发时只能部分平仓。持仓查询失败时拒绝加仓（fail-closed）。
    """
    _risk = config.get("risk", {}) or {}
    try:
        _max_exp = float(_risk.get("max_exposure_pct", 30.0) or 0)
    except (TypeError, ValueError):
        _max_exp = 0.0
    if _max_exp <= 0:
        return True
    try:
        _poss = client.get_positions()
        if _poss is None:
            return False  # fail-closed: 查询失败不得加仓
    except Exception:
        return False
    _notional = sum(_position_notional_usd(client, p) for p in _poss)
    _limit = equity * _max_exp / 100.0
    # 当前敞口 ≥ 70% 上限时不再加仓（为新单预留 ≥30% 空间，加仓后不超限）
    return _notional <= _limit * 0.7


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _round_to_tick(price: float, tick_sz: float) -> float:
    """将价格按交易所 tickSz 精度向下取整（截断多余小数位）。

    OKX 等交易所对每个合约有严格的 tickSz（价格精度，如 0.1 或 0.001）。
    如果提交的价格精度不匹配，API 会直接拒单返回 "Invalid price format"。

    止损价使用 floor（向下取整），确保止损不会被意外抬高到无效价位。
    止盈/入场价可根据需要选择 round。

    修复 L-2: 使用 Decimal 避免浮点精度误差。
    math.floor(1.0 / 0.1) * 0.1 在某些平台可能产生 0.9999999 而非 1.0。

    Args:
        price: 原始价格
        tick_sz: 合约最小价格变动单位（从 instrument_info['tickSz'] 获取）

    Returns:
        合规化截断后的价格
    """
    if tick_sz <= 0:
        return price
    # 使用 Decimal 进行精确截断，避免浮点精度问题
    try:
        d_price = decimal.Decimal(str(price))
        d_tick = decimal.Decimal(str(tick_sz))
        # 整除（Decimal // 为向下取整除法）对齐到 tick 精度
        d_result = (d_price // d_tick) * d_tick
        return float(d_result)
    except Exception:
        # Decimal 回退: 传统 math.floor 方式
        return math.floor(price / tick_sz) * tick_sz


def _format_price_for_exchange(price: float, tick_sz: float) -> str:
    """将价格按 tickSz 精度格式化后转为交易所兼容的字符串。

    修复: 浮动精度问题——直接 str(price) 可能产生 65432.12345000001，
    导致 OKX API 拒绝。先按 tickSz 精度截断再格式化。

    Args:
        price: 原始价格
        tick_sz: 合约 tickSz

    Returns:
        格式化后的价格字符串，如 "65432.1"
    """
    rounded = _round_to_tick(price, tick_sz)
    # 根据 tickSz 计算小数位数，优先使用 Decimal 精确处理非 10 幂 tickSz
    if tick_sz > 0:
        try:
            decimals = max(0, -int(decimal.Decimal(str(tick_sz)).as_tuple().exponent))
        except (decimal.InvalidOperation, decimal.DecimalException, ValueError, TypeError):
            decimals = max(0, int(-math.floor(math.log10(tick_sz))))
    else:
        decimals = 0
    return f"{rounded:.{decimals}f}"


def _estimate_funding_cost(
    client: "OKXClient",
    inst_id: str,
    pos_size: float,
    avg_px: float,
    hold_hours: float,
    funding_rate: Optional[float] = None,
    pos_side: str = "long",
) -> float:
    """估算持仓期间的资金费率总成本。

    资金费率每 8 小时结算一次（OKX 永续合约标准）。
    long: 费率 > 0 时支付，费率 < 0 时收取
    short: 费率 > 0 时收取，费率 < 0 时支付

    Args:
        client: OKX 客户端
        inst_id: 合约 ID
        pos_size: 持仓张数
        avg_px: 开仓均价
        hold_hours: 持仓时长（小时）
        funding_rate: 当前资金费率（None 时自动获取）
        pos_side: 持仓方向，"long" 或 "short"

    Returns:
        估算资金费率成本 (USDT)，负数表示支出
    """
    if hold_hours <= 0 or pos_size <= 0 or avg_px <= 0:
        return 0.0

    # 获取合约面值
    try:
        info = client.get_instrument_info(inst_id)
        ct_val = float(info.get("ctVal", "0.01")) if info else 0.01
    except (ValueError, TypeError, KeyError, OSError):
        ct_val = 0.01

    # 获取当前资金费率（如果未提供）
    if funding_rate is None:
        try:
            fr = client.get_funding_rate(inst_id)
            funding_rate = fr if fr is not None else 0.0
        except (ConnectionError, TimeoutError, OSError, ValueError):
            funding_rate = 0.0

    # 持仓价值 = 张数 * 面值 * 开仓均价
    position_value = pos_size * ct_val * avg_px

    # 结算次数（每 8 小时一次，连续比例近似 — 修复 2026-08-13:
    # 原 max(1, int(hold_hours/8)) 对 <8h 持仓按满周期高估成本）
    funding_cycles = hold_hours / 8.0

    # 资金费率成本（方向感知）
    # long: pay funding if rate > 0, receive if rate < 0
    # short: pay funding if rate < 0, receive if rate > 0
    if pos_side == "long":
        funding_cost = -position_value * funding_rate * funding_cycles
    else:
        funding_cost = position_value * funding_rate * funding_cycles

    return funding_cost


def _estimate_fee_cost(
    client: "OKXClient",
    inst_id: str,
    pos_size: float,
    avg_px: float,
    is_taker: bool = True,
) -> float:
    """估算开平仓手续费。

    OKX 标准费率: taker 0.05%, maker 0.02%
    开仓 + 平仓 = 2 倍

    Args:
        client: OKX 客户端
        inst_id: 合约 ID
        pos_size: 持仓张数
        avg_px: 开仓均价
        is_taker: 是否 taker 费率

    Returns:
        估算手续费 (USDT)，始终为负数（支出）
    """
    if pos_size <= 0 or avg_px <= 0:
        return 0.0

    try:
        info = client.get_instrument_info(inst_id)
        ct_val = float(info.get("ctVal", "0.01")) if info else 0.01
    except (ValueError, TypeError, KeyError, OSError):
        ct_val = 0.01

    position_value = pos_size * ct_val * avg_px
    # 修复 2026-08-13: 开仓为限价单(maker 0.02%)，平仓按 is_taker 区分
    # taker 0.05% / maker 0.02%。原实现统一 taker×2(0.10%) 高估 2.5 倍。
    fee_rate_open = 0.0002                                   # 限价开仓 maker
    fee_rate_close = 0.0005 if is_taker else 0.0002          # 平仓 taker/maker
    total_fee = -position_value * (fee_rate_open + fee_rate_close)

    return total_fee


def _compute_atr_from_cache(
    cache: Optional["CoinResearchCache"],
    inst_id: str,
    current_price: float,
    period: int = 14,
    fallback_pct: float = 0.02,
) -> float:
    """从缓存 1H K 线数据动态计算 ATR（True Range Wilder's 平滑），回退到保守估值。

    修复: 不再使用 current_price * 0.02 硬编码。
    不同币种波动率差异巨大（BTC ~1.5%, alt ~3-5%），硬编码会导致止损过紧或过松。
    """
    if cache is None:
        return current_price * fallback_pct
    entry = cache.get(inst_id)
    if entry is None:
        return current_price * fallback_pct
    candles_1h = entry.candles_by_tf.get("1H", [])
    if len(candles_1h) < period + 1:
        return current_price * fallback_pct

    try:
        tr_list = []
        for i in range(1, len(candles_1h)):
            h, l, prev_c = candles_1h[i].high, candles_1h[i].low, candles_1h[i - 1].close
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            tr_list.append(tr)
        if not tr_list:
            return current_price * fallback_pct
        # 初始简单平均 + Wilder's 平滑
        atr = float(np.mean(tr_list[:min(period, len(tr_list))]))
        for i in range(period, len(tr_list)):
            atr = (atr * (period - 1) + tr_list[i]) / period
        return atr
    except (ValueError, TypeError, ZeroDivisionError, IndexError):
        return current_price * fallback_pct


def _compute_correlation_1h(
    cache: Optional["CoinResearchCache"],
    inst_id_a: str,
    inst_id_b: str,
    min_samples: int = 8,
) -> Optional[float]:
    """计算两个币种 1H 对数收益率的时间戳对齐相关性。

    修复 H-3: 提取为共享函数，消除两处重复的 ~30 行相关性计算逻辑。
    两处分别位于换仓预检步骤和仓位已满换仓步骤。

    返回相关系数 float，若样本不足或数据缺失则返回 None。
    """
    if cache is None:
        return None
    entry_a = cache.get(inst_id_a)
    entry_b = cache.get(inst_id_b)
    if not entry_a or not entry_b:
        return None
    candles_a = entry_a.candles_by_tf.get("1H", [])
    candles_b = entry_b.candles_by_tf.get("1H", [])
    if len(candles_a) < 12 or len(candles_b) < 12:
        return None

    ts_a = {c.timestamp: i for i, c in enumerate(candles_a)}
    ts_b = {c.timestamp: i for i, c in enumerate(candles_b)}
    aligned_a = []
    aligned_b = []
    for c in candles_b:
        if c.timestamp in ts_a:
            oi = ts_a[c.timestamp]
            if oi > 0:
                if candles_a[oi].close <= 0 or candles_a[oi - 1].close <= 0:
                    continue
                ni = ts_b.get(c.timestamp)
                if ni is not None and ni > 0:
                    if candles_b[ni].close <= 0 or candles_b[ni - 1].close <= 0:
                        continue
                    aligned_a.append(
                        math.log(candles_a[oi].close / candles_a[oi - 1].close)
                    )
                    aligned_b.append(
                        math.log(candles_b[ni].close / candles_b[ni - 1].close)
                    )
    if len(aligned_a) >= min_samples and len(aligned_b) >= min_samples:
        corr = float(np.corrcoef(aligned_a, aligned_b)[0, 1])
        if not np.isnan(corr):
            return corr
    return None


def _run_reflection_safe(memory: "MemoryManager", recent_decisions: list):
    """后台线程安全执行反思生成，异常不传播到主线程。"""
    try:
        memory.generate_reflection(recent_decisions)
        logger.info("交易反思报告生成完成")
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.error(f"反思生成失败（后台线程）: {e}")


def _run_hyperopt_safe(trades: list, initial_equity: float, config: Optional[dict] = None):
    """后台线程安全执行 Hyperopt 贝叶斯优化，异常不传播到主线程。"""
    try:
        # 修复 2026-08-07: hyperopt.n_initial/n_refine/walk_forward_windows
        # 从 config 读取 (原调用恒用默认值, 配置项未生效)。
        _hp_cfg = (config or {}).get("hyperopt", {}) or {}
        opt_result = run_full_optimization(
            trades=trades,
            initial_equity=initial_equity,
            n_windows=int(_hp_cfg.get("walk_forward_windows", 5) or 5),
            n_initial=int(_hp_cfg.get("n_initial", 5) or 5),
            n_refine=int(_hp_cfg.get("n_refine", 3) or 3),
        )
        if opt_result:
            logger.info(f"[Hyperopt] best_score={opt_result['metrics'].composite_score:.1f}")
            logger.info(f"[Hyperopt] best_params={opt_result['hyperopt'].best_params}")
            logger.info(f"[Hyperopt] is_overfitting={opt_result['walk_forward'].is_overfitting}")
            if opt_result.get("dashboard"):
                logger.info(opt_result["dashboard"])
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.error(f"Hyperopt 优化失败（后台线程）: {e}")


# ---------------------------------------------------------------------------
# 量化增强辅助函数
# ---------------------------------------------------------------------------

def _evaluate_market_guard(
    client: OKXClient,
    market_guard: MarketEmergencyGuard,
    ws_cache: Optional["WsTickerCache"],
) -> "MarketState":
    """评估市场熔断状态。

    收集 BTC 24h 收益、滚动价格、全市场收益、资金费率、持仓量变化。
    """
    try:
        btc_ticker = ws_cache.get("BTC-USDT-SWAP") if ws_cache else None
        if not btc_ticker:
            btc_tickers = client.get_tickers(inst_type="SWAP")
            for t in btc_tickers:
                if t.get("instId") == "BTC-USDT-SWAP":
                    btc_ticker = t
                    break

        btc_return_24h = 0.0
        if btc_ticker:
            open24h = float(btc_ticker.get("open24h", "0") or 0)
            last = float(btc_ticker.get("last", "0") or 0)
            if open24h > 0 and last > 0:
                btc_return_24h = (last - open24h) / open24h * 100.0

        # BTC 滚动价格（最近 24h 1H K 线）
        btc_prices: List[float] = []
        try:
            raw = client.get_candles_enhanced("BTC-USDT-SWAP", bar="1H", limit=168)
            if raw:
                candles = candles_from_raw(raw)
                btc_prices = [c.close for c in candles]
        except Exception as e:
            logger.debug(f"[MarketGuard] BTC 1H K 线获取失败: {e}")

        # 全市场 24h 收益
        market_returns: Dict[str, float] = {}
        try:
            tickers = client.get_tickers(inst_type="SWAP")
            for t in tickers:
                inst_id = t.get("instId", "")
                if not inst_id.endswith("-USDT-SWAP"):
                    continue
                open24h = float(t.get("open24h", "0") or 0)
                last = float(t.get("last", "0") or 0)
                if open24h > 0:
                    market_returns[inst_id] = (last - open24h) / open24h
        except Exception as e:
            logger.debug(f"[MarketGuard] 市场收益获取失败: {e}")

        # 资金费率
        funding_rates: Dict[str, float] = {}
        try:
            for inst_id in ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]:
                fr = client.get_funding_rate(inst_id)
                if fr is not None:
                    funding_rates[inst_id] = fr
        except Exception as e:
            logger.debug(f"[MarketGuard] 资金费率获取失败: {e}")

        # BTC 持仓量变化（真实 24h）— 修复 2026-08-13: 原实现用 _prev_oi 作
        # 单轮(约 2 分钟)基准，算出的"变化"量级远小于 oi_spike_threshold(30%)，
        # 注释却写"24h"，OI 异常飙升检测形同虚设。改为：首次采样记录基准，
        # 仅在采样跨度 ≥ 24h 时计算真实 24h 变化并滚动基准。
        oi_change_pct = 0.0
        try:
            oi_current = client.get_open_interest("BTC-USDT-SWAP")
            if oi_current is not None:
                _prev_oi = getattr(market_guard, "_prev_oi", 0.0) or 0.0
                _prev_ts = getattr(market_guard, "_prev_oi_ts", 0.0) or 0.0
                if _prev_oi <= 0:
                    market_guard._prev_oi = oi_current
                    market_guard._prev_oi_ts = time.time()
                else:
                    _span_h = (time.time() - _prev_ts) / 3600.0
                    if _span_h >= 24.0:
                        oi_change_pct = (oi_current - _prev_oi) / _prev_oi
                        market_guard._prev_oi = oi_current
                        market_guard._prev_oi_ts = time.time()
        except Exception as e:
            logger.debug(f"[MarketGuard] 持仓量获取失败: {e}")

        state = market_guard.evaluate(
            btc_return_24h_pct=btc_return_24h,
            btc_prices=btc_prices,
            market_returns=market_returns,
            funding_rates=funding_rates,
            open_interest_change_pct=oi_change_pct,
        )
        logger.info(
            f"[MarketGuard] regime={state.regime} "
            f"btc_24h={state.btc_return_24h:.2f}% "
            f"breadth={state.market_breadth:.1%} "
            f"funding_extreme={state.funding_extreme:.3%}"
        )
        return state
    except Exception as e:
        logger.warning(f"[MarketGuard] 评估失败: {e}，返回 UNKNOWN 保守模式")
        # 关键修复：API 全部失败时不应返回 NORMAL（允许开仓），
        # 应返回 UNKNOWN 状态告知上层无法判断，由上层决定是否禁止开仓
        try:
            return market_guard.evaluate(
                btc_return_24h_pct=0.0,
                btc_prices=[],
                market_returns={},
                funding_rates={},
                open_interest_change_pct=0.0,
            )
        except Exception:
            # evaluate 内部也失败时，直接构造 UNKNOWN 状态返回
            from market_guard import MarketState
            return MarketState(
                timestamp=time.time(),
                btc_return_24h=0.0,
                btc_volatility_24h=0.0,
                market_breadth=1.0,
                funding_extreme=0.0,
                oi_change_pct=0.0,
                regime="UNKNOWN",
                reasons=["评估完全失败，数据不可用"],
            )


def _gather_risk_committee_inputs(
    client: OKXClient,
    inst_id: str,
    candles_1h: List[Candle],
    funding_rate: Optional[float],
) -> Dict[str, Any]:
    """为 RiskCommittee 收集市场数据与技术指标。"""
    market_data: Dict[str, Any] = {
        "symbol": inst_id,
        "returns_24h": 0.0,
        "volatility": 0.0,
        "funding_rate": funding_rate or 0.0,
    }
    try:
        ticker = client.get_tickers(inst_type="SWAP")
        for t in ticker:
            if t.get("instId") == inst_id:
                open24h = float(t.get("open24h", "0") or 0)
                last = float(t.get("last", "0") or 0)
                if open24h > 0:
                    market_data["returns_24h"] = (last - open24h) / open24h
                break
    except Exception:
        pass

    # 用 1H K 线估算小时波动率（ATR / price）
    if len(candles_1h) >= 20:
        try:
            rets = []
            trs = []
            for i in range(1, len(candles_1h)):
                rets.append(math.log(candles_1h[i].close / candles_1h[i - 1].close))
                h, l, prev_c = candles_1h[i].high, candles_1h[i].low, candles_1h[i - 1].close
                trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
            market_data["volatility"] = float(np.std(rets)) if rets else 0.0
            market_data["atr_14"] = float(np.mean(trs[-14:])) if trs else 0.0
        except Exception:
            pass

    technical_indicators: Dict[str, Any] = {}
    if len(candles_1h) >= 20:
        closes = np.array([c.close for c in candles_1h])
        technical_indicators["rsi"] = float(_compute_rsi(closes))
        technical_indicators["trend_strength"] = float(np.sign(np.mean(np.diff(closes))))

    return {
        "market_data": market_data,
        "technical_indicators": technical_indicators,
        "flow_data": {},
        "sentiment": {},
    }


def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """计算 RSI（容错版）。"""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _record_signal_quant(
    signal_tracker: SignalPerformanceTracker,
    signal: Signal,
    analysis: Optional[MasterAnalysis],
    regime: str,
    volatility: float,
    funding_rate: Optional[float],
    market_condition: str,
) -> str:
    """记录信号到 SignalPerformanceTracker，返回 signal_id。"""
    confidence = analysis.final_confidence if analysis else signal.score
    master_score = analysis.final_score if analysis else signal.score
    factor_score = signal.score
    _fvg = signal.fvg
    _impulse = getattr(_fvg, "impulse_candle", None)
    signal_id = signal_tracker.record_signal(
        symbol=signal.inst_id,
        direction=signal.position_side,
        entry_price=signal.entry_price,
        confidence=confidence,
        master_score=master_score,
        factor_score=factor_score,
        regime=regime,
        volatility=volatility,
        funding_rate=funding_rate or 0.0,
        market_condition=market_condition,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        leverage=signal.leverage,
        extra={
            # FVG 完整结构持久化 → train_fvg_model.py --source trades 可据此
            # 用真实成交结果回填训练样本（本版本起落库）
            "fvg_timeframe": _fvg.timeframe,
            "fvg_width_pct": _fvg.width_pct,
            "fvg_top": _fvg.top,
            "fvg_bottom": _fvg.bottom,
            "fvg_direction": _fvg.direction,
            "fvg_index": getattr(_fvg, "fvg_index", -1),
            "fvg_is_abnormal": bool(getattr(_fvg, "is_abnormal", False)),
            "fvg_sigma": float(getattr(_fvg, "sigma", 0.0)),
            "fvg_volume_ratio": float(getattr(_fvg, "volume_ratio", 1.0)),
            "fvg_candle_ts": int(getattr(_fvg, "candle_ts", 0)),
            "fvg_impulse": {
                "ts": _impulse.timestamp, "o": _impulse.open,
                "h": _impulse.high, "l": _impulse.low,
                "c": _impulse.close, "v": _impulse.volume,
            } if _impulse else None,
            "score": signal.score,
            # 汇流确认结果落库 (供离线训练与概念漂移监控)
            "confluence_score": float(getattr(signal, "confluence_score", 0.0)),
            "entry_quality": str(getattr(signal, "entry_quality", "poor")),
        },
    )
    return signal_id


def _estimate_trade_cost(
    cost_model: CostModel,
    position_value: float,
    hold_hours: float,
    funding_rate: float,
    spread_pct: float,
    volatility: float,
) -> "CostEstimate":
    """估算交易成本。"""
    return cost_model.estimate_trade(
        position_value=position_value,
        hold_hours=hold_hours,
        funding_rate=funding_rate,
        spread_pct=spread_pct,
        volatility=volatility,
    )


def _record_trade_quant(
    signal_tracker: SignalPerformanceTracker,
    signal_id: str,
    exit_price: float,
    pnl: float,
    cost_total: float,
    pos_info: Dict[str, Any],
):
    """记录平仓结果。"""
    entry_price = pos_info.get("avg_px", 0.0) or pos_info.get("entry_price", 0.0)
    size = pos_info.get("size", 0.0)
    direction = pos_info.get("pos_side", "")
    entry_time = pos_info.get("c_time", 0.0) or pos_info.get("entry_time", 0.0)
    exit_time = time.time()
    holding_time = max(0.0, exit_time - entry_time)
    pnl_pct = 0.0
    if size > 0 and entry_price > 0:
        # 合约杠杆交易：pnl_pct = pnl / (position_value) * 100
        position_value = size * entry_price
        if position_value > 0:
            pnl_pct = pnl / position_value * 100.0
    signal_tracker.record_trade(
        signal_id=signal_id,
        exit_price=exit_price,
        exit_time=exit_time,
        pnl=pnl,
        pnl_pct=pnl_pct,
        cost_total=cost_total,
        extra={
            "holding_time": holding_time,
            "size": size,
            "direction": direction,
        },
    )


def _update_mfe_mae_quant(
    signal_tracker: SignalPerformanceTracker,
    active_signals: Dict[str, dict],
    ws_cache: Optional["WsTickerCache"],
    client: OKXClient,
):
    """用当前标记价格更新活跃信号的 MFE/MAE — 开仓以来最高/最低价。"""
    for inst_id, pos in active_signals.items():
        signal_id = pos.get("signal_id")
        if not signal_id:
            continue
        try:
            mark_px = pos.get("mark_px", 0.0)
            # 修复 2026-08-13: active_signals 条目无 mark_px 字段（仅实盘
            # monitor_positions 对账时填充，纸面模式不走该路径），导致 MFE/MAE
            # 追踪恒被 skip。此处回退用 WS 实时 last 价兜底，恢复纸面模式追踪。
            if mark_px <= 0 and ws_cache is not None:
                try:
                    _t = ws_cache.get(inst_id)
                    if _t and float(_t.get("last", 0) or 0) > 0:
                        mark_px = float(_t["last"])
                except Exception:
                    mark_px = 0.0
            if mark_px <= 0:
                continue
            # 持久化在 active_signals 中，不依赖 24h high/low（可能遗漏更早的极值）
            # 始终追踪实际最高价和最低价，方向语义由 update_mfe_mae 内部处理
            if "mfe_high" not in pos:
                pos["mfe_high"] = mark_px
            if "mae_low" not in pos:
                pos["mae_low"] = mark_px
            pos["mfe_high"] = max(pos["mfe_high"], mark_px)
            pos["mae_low"] = min(pos["mae_low"], mark_px)
            signal_tracker.update_mfe_mae(signal_id, pos["mfe_high"], pos["mae_low"])
        except Exception as e:
            logger.debug(f"[SignalTracker] MFE/MAE 更新失败 {inst_id}: {e}")


def _execute_signal_with_quant_enhancements(
    client: OKXClient,
    signal: Signal,
    equity: float,
    config: dict,
    instrument_info: dict,
    state_manager: StateManager,
    risk_committee: Optional[RiskCommittee],
    market_guard: Optional[MarketEmergencyGuard],
    market_state: Optional["MarketState"],
    signal_tracker: Optional[SignalPerformanceTracker],
    analysis: Optional[MasterAnalysis],
    debate_result: Optional[Any],
    candles_1h: List[Candle],
    funding_rate: Optional[float],
    regime: str,
    paper_engine: Optional[PaperTradingEngine] = None,
    candles_htf: Optional[List[Candle]] = None,
) -> Optional[str]:
    """执行信号，集成风险委员会、市场熔断与信号记录。

    返回交易所订单号；若被风险委员会/市场熔断拦截或下单失败，返回 None。
    纸面模式下（paper_engine 非空），下单走 dry-run 假单路径，同时由
    纸面引擎建立虚拟持仓（限价挂单等待回补成交）。
    """
    # ---- 1. 市场熔断：危机模式禁止开仓 ----
    if market_guard and not market_guard.can_open_new_position(market_state):
        logger.warning(
            f"[MarketGuard] CRISIS 模式禁止开仓，跳过 {signal.inst_id}"
        )
        return None

    # ---- 1.5 区间位置红旗门禁 (教训固化: SAHARA/CHZ 区间底部开空被扫) ----
    # 放在 AI 风控之前、最前顺位，覆盖直接开仓/换仓/反手/加仓全部路径
    _rf_ok, _rf_reason = _red_flag_gate(signal, analysis, config)
    if not _rf_ok:
        logger.info(_rf_reason)
        return None

    # ---- 1.6 HTF 方向门 (大周期定方向) ----
    # 2026-08-07 调研: 顺趋势 64% vs 逆趋势 3.9% 胜率。无 4H 数据时放行。
    _htf_ok, _htf_reason = _htf_alignment_gate(signal, candles_htf, config)
    if not _htf_ok:
        logger.info(_htf_reason)
        return None

    # ---- 1.6b 方向动量一致性门 (1H 短期趋势, 2026-08-10 方案A) ----
    # HTF(4H)滞后: 1H 已转跌时 4H 仍向上 → 逆 1H 趋势开多(实测 PUMP
    # score=0.91 做多后 -4.6%)。方向与 1H SMA+斜率明确冲突时否决。
    _mom_ok, _mom_reason = _direction_momentum_gate(signal, candles_1h, config)
    if not _mom_ok:
        logger.info(_mom_reason)
        return None

    # ---- 1.7 期望值门禁 (负期望暂停开仓) ----
    # 2026-08-07 调研: 近 window 笔均值为负 → 继续开仓=持续亏手续费。
    _exp_ok, _exp_reason = _expectancy_guard(state_manager, config, signal.inst_id)
    if not _exp_ok:
        logger.info(_exp_reason)
        return None

    # ---- 1.8 结算窗口守卫 (直接开仓需付费率时推迟) ----
    # 2026-08-07 调研: 距结算 < lockout 且目标方向需付费率 → 结算后开。
    _fund_ok, _fund_reason = _open_funding_settlement_guard(
        client, signal.inst_id, signal.position_side, funding_rate, config)
    if not _fund_ok:
        logger.info(_fund_reason)
        return None

    exec_config = copy.deepcopy(config)
    # 波动率目标仓位 (v3.3 / PRO): ATR% 高的币降保证金、ATR% 低的币用满。
    # 开源核心版自动跳过（不缩放）。
    if _PRO is not None:
        try:
            _vt_scale = _PRO.vol_targeting_scale(
                candles_1h, float(signal.entry_price or 0), config)
            if _vt_scale != 1.0 and "risk" in exec_config:
                _orig_margin = float(
                    exec_config["risk"].get("margin_pct", 30.0) or 0)
                _new_margin = _orig_margin * _vt_scale
                exec_config["risk"]["margin_pct"] = _new_margin
                logger.info(
                    f"[VolTarget] {signal.inst_id} 波动率目标缩放 "
                    f"{_vt_scale:.2f}×: margin_pct {_orig_margin:.0f}% → "
                    f"{_new_margin:.1f}% (ATR% 偏离目标)")
        except Exception as _vt_e:
            logger.debug(f"[VolTarget] {signal.inst_id} 缩放失败(放行): {_vt_e}")
    # 修复 P2-1: 接线 market_guard 的 WARNING 减仓因子（此前从未被调用，
    # WARNING 状态——资金费极端/OI 异常——不产生任何仓位缩放）。
    # CRISIS 由 can_open_new_position 硬拦截，WARNING 走减半仓。
    if market_guard is not None and market_state is not None:
        try:
            _mg_factor = market_guard.reduce_position_factor(market_state)
            if _mg_factor < 1.0 and "risk" in exec_config:
                exec_config["risk"]["risk_per_trade_pct"] *= _mg_factor
                logger.info(
                    f"[MarketGuard] 状态 {market_state.regime}，仓位系数 {_mg_factor:.0%}"
                )
        except Exception:
            pass
    inputs: Optional[Dict[str, Any]] = None

    # ---- 2. AI 风险委员会评估 ----
    if risk_committee:
        inputs = _gather_risk_committee_inputs(
            client, signal.inst_id, candles_1h, funding_rate
        )
        if debate_result is not None:
            assessment = risk_committee.assess_from_debate(
                debate_result, market_data=inputs["market_data"]
            )
        else:
            assessment = risk_committee.assess(
                inputs["market_data"],
                technical_indicators=inputs.get("technical_indicators"),
                flow_data=inputs.get("flow_data"),
                sentiment=inputs.get("sentiment"),
            )
        logger.info(
            f"[RiskCommittee] {signal.inst_id} risk_score={assessment.risk_score} "
            f"position_factor={assessment.position_factor:.0%} "
            f"allow_trade={assessment.allow_trade}"
        )
        if not assessment.allow_trade:
            logger.warning(f"[RiskCommittee] 风险过高，禁止交易 {signal.inst_id}")
            return None
        if assessment.position_factor < 1.0 and "risk" in exec_config:
            exec_config["risk"]["risk_per_trade_pct"] *= assessment.position_factor
            logger.info(
                f"[RiskCommittee] 风险分 {assessment.risk_score}，"
                f"仓位系数 {assessment.position_factor:.0%}"
            )

    # ---- 2.5 弱信号多指标共振审核（"赌一把"防线）----
    # 用户要求: 弱信号(想赌一把)下单前必须参考多个技术指标——交易量/
    # 换手率/多空比/资金费率/趋势，至少 N 项顺向才允许下单。
    # 放在 AI 风控委员会之后、下单之前，覆盖所有开仓路径
    # (直接开仓/换仓/反手/金字塔加仓/缓存路径)。
    if config.get("strategy", {}).get("weak_signal_gate", {}).get("enabled", True):
        try:
            _weak_ok, _weak_reason = _weak_signal_multi_gate(
                config, client, signal, analysis, candles_1h
            )
            if not _weak_ok:
                logger.info(_weak_reason)
                return None
        except Exception as _we:
            # 修复 P2-3: 弱信号共振审核异常时 fail-closed（拒绝开仓）。
            # 这是风险过滤门——组件故障时放行 = 少一道闸，宁可错过不可冒险。
            # （与 ML/汇流评分类门的放行策略区分：评分类可降级，风控类必须拦截）
            logger.warning(
                f"[WeakGate] {signal.inst_id} 审核异常，拒绝开仓(fail-closed): {_we}"
            )
            return None

    # ---- 3. 执行下单 ----
    # 修复 2026-08-10: 同轮内纸面平仓后, 传入的 equity 是轮首快照(含已平仓
    # 浮盈), 直接用于开仓会超额 — 实测 13:54 平仓 -5.29 后, 13:56 同轮开新仓
    # 仍用 38.01(含已平仓浮盈) → 保证金 11.40, 实际余额 25.35 只该用 7.60。
    # 下单前刷新为当前真实权益, 保证以损定量/保证金口径正确。
    if paper_engine is not None:
        try:
            _fresh_equity = paper_engine.get_equity()
            if _fresh_equity is not None and _fresh_equity > 0:
                equity = _fresh_equity
        except Exception as _pe:
            logger.debug(
                f"[Paper] {signal.inst_id} 开仓前权益刷新失败: {_pe}")
    # 满倍率模式 (2026-08-09 用户要求): 执行杠杆 = 币种最大杠杆 (tiers.maxLever)，
    # 统一在执行前覆盖 signal.leverage，保证实盘(dry-run)与纸面引擎口径一致。
    # 剩余余额全部留在账户当爆仓缓冲 (isolated 逐仓爆仓只损该仓 30% 保证金)。
    try:
        _exec_risk = exec_config.get("risk", {}) if isinstance(exec_config, dict) else {}
        signal.leverage = resolve_full_leverage(
            client, signal.inst_id, int(signal.leverage or 1), _exec_risk)
    except Exception as _lev_e:
        logger.warning(
            f"[Leverage] {signal.inst_id} 满杠杆解析失败，用信号杠杆 "
            f"{signal.leverage}x: {_lev_e}")
    ord_id = execute_signal(client, signal, equity, exec_config, instrument_info)
    if not ord_id:
        return None

    # ---- 3.5 纸面模式: 同步建立虚拟持仓 ----
    # 使用 exec_config（含风险委员会/加仓缩放后的风险比例），保证虚拟仓位
    # 口径与实盘下单完全一致。纸面开仓失败不影响主循环（日志告警）。
    if paper_engine is not None:
        try:
            paper_engine.open_position(
                signal=signal,
                instrument_info=instrument_info,
                risk_cfg=exec_config.get("risk", {}),
                equity=equity,
            )
        except Exception as _pe:
            logger.warning(f"[Paper] {signal.inst_id} 纸面开仓失败: {_pe}")

    # 修复: 记录信号 TP — FVG 限价单在回补位（当前价下方）等回踩期间，
    # TP（回补位上方但低于当前价）会被 OKX 以 51277/51279 拒绝（TP 须相对
    # 实时价方向正确）。成交后由 trailing 用此信号 TP 补挂（相对成交价
    # 方向必然正确），否则持仓会一直缺止盈。
    try:
        with state_manager.lock():
            _sig_slot = state_manager.state.active_signals.setdefault(
                signal.inst_id, {}
            )
            # 修复: 新开仓重置分批止盈档位标记 — 旧持仓残留的 scaled_out
            # 若不清零, 同币种二次开仓会直接跳过第一档进入第二档检查。
            _sig_slot["scaled_out"] = 0
            _sig_slot["signal_tp"] = float(signal.take_profit)
            # 修复 2026-08-10: 记录实际执行杠杆 — 满倍率下 execute_signal 可能
            # 因"止损先于爆仓"降杠杆(50x→15x), signal.leverage 此处已是被
            # 降杠杆后的值(finally 恢复在外层), 落盘供复盘审计/监控查看。
            _sig_slot["exec_leverage"] = float(signal.leverage)
            # CE 中点失效(2026-08-07): 记录信号 FVG 中点, trailing 据此判断
            # 结构失效(实体收盘越过中点→止损提到成本价), 不再死等原止损。
            try:
                _sig_slot["signal_fvg_mid"] = float(
                    (signal.fvg.top + signal.fvg.bottom) / 2.0)
            except (TypeError, ValueError, AttributeError):
                _sig_slot.pop("signal_fvg_mid", None)
    except Exception as _e:
        logger.debug(f"记录 signal_tp 失败: {_e}")

    # ---- 4. 记录信号到 SignalPerformanceTracker ----
    signal_id = None
    if signal_tracker:
        if inputs is None:
            inputs = _gather_risk_committee_inputs(
                client, signal.inst_id, candles_1h, funding_rate
            )
        signal_id = _record_signal_quant(
            signal_tracker,
            signal,
            analysis,
            regime,
            volatility=inputs["market_data"].get("volatility", 0.0),
            funding_rate=funding_rate,
            market_condition=str(market_state.regime if market_state else regime),
        )

    # ---- 5. 关联 signal_id 到 active_signals，便于平仓时追溯 ----
    if signal_id:
        try:
            with state_manager.lock():
                # 不依赖 monitor_positions 刷新（交易所可能尚未处理订单），
                # 直接在 active_signals 中写入或更新条目
                if signal.inst_id not in state_manager.state.active_signals:
                    # 预留条目，等 monitor_positions 下一轮刷新时合并
                    state_manager.state.active_signals[signal.inst_id] = {}
                pos = state_manager.state.active_signals[signal.inst_id]
                pos["inst_id"] = signal.inst_id
                pos["signal_id"] = signal_id
                pos["confidence"] = (
                    analysis.final_confidence if analysis else signal.score
                )
                pos["master_score"] = (
                    analysis.final_score if analysis else signal.score
                )
                pos["factor_score"] = signal.score
                pos["regime"] = regime
                pos["entry_time"] = time.time()
        except Exception as e:
            logger.debug(f"[SignalTracker] 关联 signal_id 失败: {e}")

    if ord_id:
        _FILL_FUNNEL["placed"] += 1  # 成单率漏斗: 挂单成功数
        _FILL_FUNNEL["placed_at"][signal.inst_id] = time.time()  # 等待时长统计
    return ord_id


def _run_factor_selection_safe(
    factor_selector: FactorSelector,
    factor_zoo_adapter: Optional["FactorZooAdapter"],
    cache: Optional["CoinResearchCache"],
    quant_db: QuantDB,
):
    """后台安全执行因子选择并保存结果。"""
    try:
        if factor_zoo_adapter is None or cache is None:
            return
        # 构建 panel
        candles_dict: Dict[str, pd.DataFrame] = {}
        for entry in cache.get_all_entries():
            candles_1h = entry.candles_by_tf.get("1H", [])
            if len(candles_1h) < 50:
                continue
            df = pd.DataFrame(
                {
                    "open": [c.open for c in candles_1h],
                    "high": [c.high for c in candles_1h],
                    "low": [c.low for c in candles_1h],
                    "close": [c.close for c in candles_1h],
                    "volume": [c.volume for c in candles_1h],
                },
                index=pd.to_datetime([c.timestamp for c in candles_1h], unit="ms"),
            )
            candles_dict[entry.inst_id] = df

        if len(candles_dict) < 3:
            logger.warning("[FactorSelector] 币种不足，跳过因子选择")
            return

        panel = factor_zoo_adapter.build_panel(candles_dict)
        all_factors = factor_zoo_adapter.list_crypto_factors()
        if len(all_factors) < 10:
            logger.warning("[FactorSelector] 可用因子不足，跳过因子选择")
            return

        # 计算因子值 — 优先用 BTC 单独计算，回退到横截面均值
        btc_df = candles_dict.get("BTC-USDT-SWAP")
        ref_df = btc_df if btc_df is not None else next(iter(panel.values()))
        factor_df = pd.DataFrame(index=ref_df.index)
        for alpha_id in all_factors[:80]:  # 限制计算量，扩展至 80
            try:
                values = factor_zoo_adapter.compute(alpha_id, panel)
                if values is None or values.empty:
                    continue
                # 用 BTC 的因子值（若 panel 包含 BTC），否则用横截面均值
                if btc_df is not None and "BTC-USDT-SWAP" in values.columns:
                    factor_df[alpha_id] = values["BTC-USDT-SWAP"].reindex(factor_df.index)
                elif btc_df is not None and isinstance(values.index, pd.MultiIndex):
                    factor_df[alpha_id] = values.mean(axis=1)
                else:
                    factor_df[alpha_id] = values.mean(axis=1)
            except Exception:
                continue

        if factor_df.empty or len(factor_df.columns) < 5:
            logger.warning("[FactorSelector] 有效因子不足，跳过因子选择")
            return

        # 计算未来 1H 收益作为 forward returns — 使用 ref_df 的收盘价
        # forward_returns[t] = close[t+1]/close[t] - 1
        if "close" in ref_df.columns:
            forward_returns = ref_df["close"].pct_change().shift(-1)
            forward_returns = forward_returns.reindex(factor_df.index).fillna(0.0)
        else:
            forward_returns = pd.Series(0.0, index=factor_df.index)

        selected, metrics = factor_selector.select(factor_df, forward_returns)
        logger.info(f"[FactorSelector] 核心因子: {len(selected)} 个")

        # 保存因子得分
        now = time.time()
        records = []
        for m in metrics.values():
            if m.name in selected:
                records.append({
                    "symbol": "",
                    "timestamp": now,
                    "factor_name": m.name,
                    "score": m.ic_mean,
                    "weight": m.contribution_score,
                })
        if records:
            quant_db.save_factor_scores(records)
    except Exception as e:
        logger.error(f"[FactorSelector] 后台因子选择失败: {e}")


def _run_walk_forward_safe(
    walk_forward: WalkForwardAnalyzer,
    backtest_runner: Optional["BacktestRunner"],
    client: OKXClient,
    config: dict,
):
    """后台安全执行 Walk Forward 验证。"""
    try:
        if backtest_runner is None:
            return
        # 获取 BTC 历史 K 线
        raw = client.get_candles_enhanced("BTC-USDT-SWAP", bar="4H", limit=2000)
        if not raw:
            logger.warning("[WalkForward] 无法获取历史 K 线")
            return
        candles = candles_from_raw(raw)
        df = pd.DataFrame(
            {
                "open": [c.open for c in candles],
                "high": [c.high for c in candles],
                "low": [c.low for c in candles],
                "close": [c.close for c in candles],
                "volume": [c.volume for c in candles],
            },
            index=pd.to_datetime([c.timestamp for c in candles], unit="ms"),
        )

        def _bt_fn(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
            # 使用传入的 train/test 切分数据做逐 bar 回测，而非重新拉取 API
            if data.empty or len(data) < 30:
                return {"total_return": 0.0, "sharpe_ratio": 0.0}
            closes = data["close"].values
            total_return = 0.0
            daily_rets = []
            in_position = False
            entry_price = 0.0
            position_side = ""
            # 简化回测：MA20 突破入场 + 固定止损 2% / 止盈 4%
            for i in range(30, len(closes)):
                if not in_position:
                    # 入场条件：价格穿越 MA20
                    ma20 = float(np.mean(closes[i-20:i]))
                    if closes[i] > ma20 and closes[i-1] <= ma20:
                        entry_price = closes[i]
                        position_side = "long"
                        in_position = True
                    elif closes[i] < ma20 and closes[i-1] >= ma20:
                        entry_price = closes[i]
                        position_side = "short"
                        in_position = True
                else:
                    if position_side == "long":
                        pnl_pct = (closes[i] - entry_price) / entry_price
                        # 止损 -2% 或 止盈 +4%
                        if pnl_pct <= -0.02 or pnl_pct >= 0.04 or i == len(closes) - 1:
                            total_return += pnl_pct * 0.01  # 1% risk per trade
                            daily_rets.append(pnl_pct * 0.01)
                            in_position = False
                    else:
                        pnl_pct = (entry_price - closes[i]) / entry_price
                        if pnl_pct <= -0.02 or pnl_pct >= 0.04 or i == len(closes) - 1:
                            total_return += pnl_pct * 0.01
                            daily_rets.append(pnl_pct * 0.01)
                            in_position = False
            sharpe = 0.0
            if len(daily_rets) > 1:
                mu = float(np.mean(daily_rets))
                sigma = float(np.std(daily_rets))
                sharpe = mu / sigma * np.sqrt(252) if sigma > 0 else 0.0
            return {"total_return": total_return, "sharpe_ratio": sharpe}

        results = walk_forward.analyze(df, _bt_fn)
        summary = walk_forward.summary(results)
        logger.info(f"[WalkForward] windows={summary.get('windows')} "
                    f"overfit_rate={summary.get('overfit_rate', 0):.1%} "
                    f"avg_decay={summary.get('avg_decay_ratio', 0):.2f}")
        if summary.get("overfit_rate", 0) > 0.5:
            logger.warning("[WalkForward] 过拟合警告：超过 50% 窗口验证收益显著衰减")
    except Exception as e:
        logger.error(f"[WalkForward] 后台验证失败: {e}")


def _generate_daily_report_safe(
    quant_report: QuantReportGenerator,
    equity_curve: Optional[pd.Series] = None,
):
    """后台安全生成每日量化报告。"""
    try:
        report = quant_report.generate(equity_curve=equity_curve)
        path = quant_report.save(report)
        logger.info(f"[QuantReport] 日报已保存: {path}")
        cal = report.confidence_calibration
        if cal:
            for bin_name, stats in cal.items():
                if isinstance(stats, dict):
                    logger.info(
                        f"[QuantReport] Calibration {bin_name}: "
                        f"n={stats.get('n', 0)}, win_rate={stats.get('win_rate', 0):.1%}"
                    )
    except Exception as e:
        logger.error(f"[QuantReport] 日报生成失败: {e}")


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

# BTC K 线缓存（ML 特征 corr_with_btc 注入）：tf -> (fetched_ts, List[Candle])
_BTC_CANDLE_CACHE: Dict[str, Tuple[float, List[Candle]]] = {}


def _btc_candles_cached(client, tf: str, ttl_s: float = 300.0) -> Optional[List[Candle]]:
    """获取并缓存 BTC K 线，供 ML 特征 corr_with_btc 注入。

    失败时返回 None，调用方忽略（corr_with_btc 特征回落中性 0，不阻塞主循环）。
    """
    global _BTC_CANDLE_CACHE
    _hit = _BTC_CANDLE_CACHE.get(tf)
    if _hit and (time.time() - _hit[0]) < ttl_s:
        return _hit[1]
    try:
        raw = client.get_candles_enhanced("BTC-USDT-SWAP", bar=tf, limit=200)
        candles = candles_from_raw(raw) if raw else []
    except Exception as e:
        logger.debug(f"[ML] BTC {tf} K 线获取失败: {e}")
        return None
    if candles:
        _BTC_CANDLE_CACHE[tf] = (time.time(), candles)
    return candles or None


def _paper_market_data(
    client: OKXClient,
    ws_cache: Optional[WsTickerCache],
    inst_id: str,
) -> Optional[dict]:
    """纸面交易行情源：最近 1H K 线（用于限价回补/TP/SL 判定）+ 当前价。

    价格优先级: WS 缓存最新买价 > 标记价格 > 最近收盘价。
    全部失败返回 None（纸面引擎保守跳过，不误成交不误平仓）。
    """
    try:
        # 修复(2026-08-09): 取 3 根不足以计算 ATR(14) → 纸面移动止损恒退化
        # 为固定百分比路径 (RAVE +3.4% 未激活 TS 实测)。增至 20 根，
        # 供 _atr14(Wilder, 需 14+1 根) 与限价回补/TP/SL 判定使用。
        raw = client.get_candles_enhanced(inst_id, "1H", 20) or []
        candles = candles_from_raw(raw) if raw else []
        mark = None
        if ws_cache is not None:
            try:
                _t = ws_cache.get(inst_id)
                if isinstance(_t, dict):
                    # 修复: WS 缓存字段名为 "bidPx" (与 REST get_tickers 同构)，
                    # 此前误用 "bid_px" 导致 WS 买价永远取不到，纸面价格优先级降级
                    _bp = _t.get("bidPx")
                    if _bp:
                        mark = float(_bp)
            except Exception:
                mark = None
        if mark is None:
            mark = client.get_mark_price(inst_id)
        if mark is None and candles:
            mark = candles[-1].close
        return {"candles": candles, "mark": mark}
    except Exception as _e:
        logger.debug(f"[Paper] {inst_id} 行情获取失败: {_e}")
        return None


def main_loop(config: dict, once: bool = False, max_rounds: int = 0):
    """Agent v2.0 主循环 — 多通道融合 + 自适应优化 + 记忆反思。"""
    logger = logging.getLogger("agent")
    client = OKXClient(config)

    # ---- WebSocket 行情缓存 (替代 REST 轮询) ----
    # 修复: tickers 频道不支持 instType 模糊订阅，改为按 instId 订阅 Top N 币种
    ws_cache = WsTickerCache(
        proxy=config.get("okx", {}).get("proxy"),
        inst_provider=lambda: [
            c["instId"] for c in get_tradable_coins(client, config)
        ],
    )
    ws_cache.start()
    _cleanup_registry["ws_cache"] = ws_cache  # For restart cleanup

    # ---- 初始化模块 ----
    state_path = os.path.join(os.path.dirname(__file__), "agent_state.json")
    state_manager = StateManager(state_path)
    _cleanup_registry["state_manager"] = state_manager  # For restart cleanup

    # 多通道专家引擎
    mc_cfg = config.get("multi_channel", {})
    expert_engine = MasterTraderEngine(
        weights=mc_cfg.get("channel_weights")
    ) if mc_cfg.get("enabled", True) else None

    # 优化模块
    opt_cfg = config.get("optimization", {})
    edge_analyzer = EdgeAnalyzer()
    adaptive_tuner = AdaptiveParameterTuner(config) if opt_cfg.get("adaptive_enabled", True) else None
    _ts_enabled = opt_cfg.get("trailing_stop_enabled", True)
    _ts_activation = opt_cfg.get("trailing_stop_activation_pct", 0.50)
    _ts_trail = opt_cfg.get("trailing_stop_trail_pct", 0.03)
    # 每个持仓独立维护 TrailingStop 实例，避免状态串扰
    trailing_stops: Dict[str, TrailingStop] = {} if _ts_enabled else None
    # 记录每个持仓当前的止损 algo_id，用于 trailing stop 更新时撤销旧单
    trailing_algo_ids: Dict[str, str] = {}
    # 修复: 追踪止损防抖 — 记录每个持仓上次提交的止损价，避免微小波动触发高频撤改单
    _last_submitted_sl: Dict[str, float] = {}
    # 修复: 缓存每个合约的 tickSz，避免重复 API 调用
    _tick_sz_cache: Dict[str, float] = {}
    # 修复: 缓存每个合约的 lotSz（最小下单步长），用于止损单 sz 格式化
    _lot_sz_cache: Dict[str, float] = {}

    # 记忆模块
    mem_cfg = config.get("memory", {})
    memory = MemoryManager(
        memory_dir=mem_cfg.get("memory_dir", "memory"),
        half_life_days=float(mem_cfg.get("half_life_days", 30) or 30),
        archive_threshold=float(mem_cfg.get("archive_threshold", 0.1) or 0.1),
    ) if mem_cfg.get("enabled", True) else None

    # 辩论引擎 (TradingAgents 86k⭐)
    debate_cfg = config.get("debate_engine", {})
    debate_engine = TradingAgentsDebateEngine(
        debate_rounds=debate_cfg.get("debate_rounds", 2),
        min_agreement=debate_cfg.get("min_agreement", 0.50),
        checkpoint_dir=debate_cfg.get("checkpoint_dir") if debate_cfg.get("save_checkpoints") else None,
    ) if debate_cfg.get("enabled", True) else None

    # Alpha 因子库 (Vibe-Trading 23.6k⭐)
    az_cfg = config.get("alpha_zoo", {})

    # 因果滞后体制检测 (Vibe-Trading)
    # 修复 2026-08-07: 透传全部 regime 参数 — 原只传 hysteresis/min_duration,
    # config 的 regime_enter_threshold/exit/corr_window/smooth_window 未生效。
    regime_detector = CausalHysteresisRegime(
        hysteresis_threshold=az_cfg.get("hysteresis_threshold", 0.15),
        min_regime_duration=az_cfg.get("min_regime_duration", 5),
        enter_threshold=az_cfg.get("regime_enter_threshold", 0.65),
        exit_threshold=az_cfg.get("regime_exit_threshold", 0.45),
        corr_window=az_cfg.get("regime_corr_window", 60),
        smooth_window=az_cfg.get("regime_smooth_window", 5),
    ) if az_cfg.get("enabled", True) else None
    # 修复: 体制检测器必须 per-symbol 独立实例 — 共享实例被多币种顺序调用
    # 会互相污染状态 (candidate_count/current_regime)，一个币种触发 FUSED
    # 后其余币种全部继承 FUSED。此 registry 由主循环与 CoinTracker 共享，
    # 保证同一币种使用同一个状态机实例。
    regime_detector_map: Dict[str, CausalHysteresisRegime] = {}

    # Alpha Zoo 461 因子库适配器 (Vibe-Trading)
    factor_zoo_adapter = None
    if _FACTOR_ZOO_AVAILABLE and az_cfg.get("factor_zoo_enabled", True):
        factor_zoo_adapter = FactorZooAdapter()
        if factor_zoo_adapter.load_registry():
            n_factors = len(factor_zoo_adapter.list_all())
            n_crypto = len(factor_zoo_adapter.list_crypto_factors())
            logger.info(f"  FactorZoo: {n_factors} total, {n_crypto} crypto-compatible")
        else:
            logger.warning("  FactorZoo: failed to load, falling back to built-in factors")
            factor_zoo_adapter = None

    scan_interval = config["agent"]["scan_interval_seconds"]
    risk_cfg = config["risk"]

    # 回测引擎 (Vibe-Trading)
    backtest_runner = None
    if _BACKTEST_AVAILABLE and config.get("backtest", {}).get("enabled", True):
        bt_cfg = config.get("backtest", {})
        backtest_runner = BacktestRunner(
            initial_capital=bt_cfg.get("initial_capital", 10000),
            leverage=bt_cfg.get("leverage", risk_cfg["max_leverage"]),
            risk_per_trade=bt_cfg.get("risk_per_trade", risk_cfg["risk_per_trade_pct"] / 100),
            market="crypto",
            interval="1H",
        )
        logger.info("  BacktestRunner: enabled")

    # FreqAI 在线学习 (freqtrade 52k⭐)
    fa_cfg = config.get("freqai", {})
    freqai = FreqAIPipeline(
        feature_window=fa_cfg.get("feature_window", 50),
        retrain_interval=fa_cfg.get("retrain_interval", 10),
    ) if fa_cfg.get("enabled", True) else None

    # ---- Confluence 汇流确认 (任务3, 多因素汇流评分过滤) ----
    con_cfg = config.get("confluence", {})
    confluence_checker = None
    if con_cfg.get("enabled", True):
        try:
            confluence_checker = ConfluenceChecker(con_cfg)
            logger.info("  Confluence Checker: enabled")
        except Exception as e:
            logger.warning(f"Confluence Checker 初始化失败，禁用: {e}")
            confluence_checker = None
    else:
        logger.info("  Confluence Checker: disabled (config.confluence.enabled=false)")

    # ---- FVG ML 二次评分 (第二阶段 ML 增强, 可插拔) ----
    # 模型文件存在且 ml.enabled=true 时启用；缺失时自动禁用（不阻塞主循环）。
    ml_ranker = None
    # 注入汇流配置 → compute_features 的 16-25 维汇流特征可用
    _ml_detector = FVGDetector(
        {**config.get("strategy", {}), "confluence": con_cfg})
    _ml_cfg = config.get("ml", {})
    if _ml_cfg.get("enabled", False):
        _ml_path = _ml_cfg.get("model_path", "models/fvg_ranker.pkl")
        _ml_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), _ml_path)
        if os.path.exists(_ml_abs):
            try:
                ml_ranker = FVGMLRanker(_ml_abs)
                logger.info(f"  FVG ML Ranker: enabled ({_ml_abs})")
            except Exception as e:
                logger.warning(f"FVG ML Ranker 加载失败，禁用: {e}")
                ml_ranker = None
        else:
            logger.info(
                f"  FVG ML Ranker: 模型文件不存在 {_ml_abs}，ML 过滤禁用 "
                f"(可先运行 fvg_training_pipeline.py 训练)"
            )

    # Hyperopt 优化器 (freqtrade) — 定期触发
    hp_cfg = config.get("hyperopt", {})
    hyperopt_enabled = hp_cfg.get("enabled", True)
    hyperopt_interval = hp_cfg.get("optimize_interval_rounds", 50)
    # 修复: 非法配置（0/负值）防护 — round_count % hyperopt_interval 会抛
    # ZeroDivisionError 且 main() 重启兜底不捕获该异常，导致无人值守中断
    if hyperopt_interval <= 0:
        logger.warning(f"hyperopt.optimize_interval_rounds={hyperopt_interval} 非法，重置为 50")
        hyperopt_interval = 50

    # ---- 时段报告 ----
    session_reporter = SessionReporter(config) if config.get("report", {}).get("enabled", True) else None
    if session_reporter:
        logger.info(f"  Session Report: enabled (times: {session_reporter.session_times})")

    # ---- 量化增强模块 ----
    _persistence_enabled = config.get("persistence", {}).get("enabled", True)
    quant_db = QuantDB(
        db_path=config.get("persistence", {}).get("db_path", "quant_agent.db")
    ) if _persistence_enabled else None
    signal_tracker = SignalPerformanceTracker(quant_db) if quant_db else None

    _mg_enabled = config.get("market_guard", {}).get("enabled", True)
    # 修复: 剥离非构造函数键（enabled/_comment），否则按官方 config.example.json
    # 配置时 MarketEmergencyGuard(**config段) 会抛 TypeError: unexpected keyword argument
    _mg_cfg = {k: v for k, v in config.get("market_guard", {}).items()
               if k not in ("enabled", "_comment")}
    market_guard = MarketEmergencyGuard(**_mg_cfg) if _mg_enabled else None

    _rc_enabled = config.get("risk_committee", {}).get("enabled", True)
    # 修复: 同上 — 剥离 enabled/_comment 后传入构造函数
    _rc_cfg = {k: v for k, v in config.get("risk_committee", {}).items()
               if k not in ("enabled", "_comment")}
    risk_committee = RiskCommittee(
        debate_engine=debate_engine,
        **_rc_cfg,
    ) if _rc_enabled else None

    cost_model = CostModel.from_config(config)

    _qr_enabled = config.get("quant_report", {}).get("enabled", True)
    quant_report = QuantReportGenerator(
        quant_db,
        report_dir=config.get("quant_report", {}).get(
            "report_dir", config.get("report", {}).get("report_dir", "reports")
        ),
    ) if _qr_enabled and quant_db else None

    _fs_enabled = config.get("factor_selector", {}).get("enabled", True)
    # 修复: 剥离非构造函数键（enabled/_comment/reselect_interval_rounds —
    # 后者在步骤 12 单独读取），否则按官方 config.example.json 配置必抛 TypeError
    _fs_cfg = {k: v for k, v in config.get("factor_selector", {}).items()
               if k not in ("enabled", "_comment", "reselect_interval_rounds")}
    factor_selector = FactorSelector(**_fs_cfg) if _fs_enabled else None

    _wf_enabled = config.get("walk_forward", {}).get("enabled", True)
    # 修复: 同上 — interval_rounds 在步骤 13 单独读取，不属于构造函数参数
    _wf_cfg = {k: v for k, v in config.get("walk_forward", {}).items()
               if k not in ("enabled", "_comment", "interval_rounds")}
    # 修复 2026-08-07: config 无 walk_forward 段 — 原 _wf_cfg 恒空导致
    # WalkForwardAnalyzer(n_windows=1) 用默认值, hyperopt.walk_forward_windows
    # 配置未生效。无独立段时从 hyperopt 段补 n_windows。
    if not _wf_cfg:
        _wf_cfg = {"n_windows": int(
            config.get("hyperopt", {}).get("walk_forward_windows", 5) or 5)}
    walk_forward = WalkForwardAnalyzer(**_wf_cfg) if _wf_enabled else None

    logger.info(
        f"  QuantEnhancements: SignalTracker={signal_tracker is not None}, "
        f"MarketGuard={market_guard is not None}, RiskCommittee={risk_committee is not None}, "
        f"CostModel=True, QuantReport={quant_report is not None}, "
        f"FactorSelector={factor_selector is not None}, WalkForward={walk_forward is not None}"
    )

    # ---- 研判挡位（必须在 tracker/warmup 之前计算，供其使用 scan_config） ----
    agg_mode = config["agent"].get("aggressiveness", 3)
    agg_thresholds = get_aggressiveness_thresholds(agg_mode)
    logger.info(f"  Aggressiveness: {agg_mode} ({agg_thresholds['label']})")
    # 应用挡位到策略配置（影响 FVG 检测宽严度）
    scan_config = apply_aggressiveness_to_config(config, agg_thresholds)
    # 修复 2026-08-13: 低流动性时段 ×1.2 的基准值（挡位调整后的固定基准）。
    # 每轮低流动性调整都从该基准计算，而非 scan_config 当前值——否则若上轮
    # 在 get_positions 失败/换仓/risk_gate 拦截等 continue 分支提前退出、漏执行
    # 恢复代码，min_fvg_width_pct 会按 1.2^n 指数膨胀，FVG 宽度门槛越来越严
    # 最终信号静默饿死。
    _base_fvg_width = copy.deepcopy(scan_config["strategy"]["min_fvg_width_pct"])

    # 币种追踪研究 (后台线程) — 闲置时间持续研究，不等建仓才分析
    tracker_cfg = config.get("coin_tracker", {})
    tracker_enabled = tracker_cfg.get("enabled", True)
    cache = None
    tracker = None
    if tracker_enabled:
        cache = CoinResearchCache(
            max_entries=tracker_cfg.get("max_cache_entries", 200),
            candle_ttl=tracker_cfg.get("candle_ttl_seconds", 60),
            research_ttl=tracker_cfg.get("research_ttl_seconds", 120),
        )
        tracker = CoinTracker(
            client=client,
            config=config,
            cache=cache,
            scan_config=scan_config,
            expert_engine=expert_engine,
            regime_detector=regime_detector,
            debate_engine=debate_engine,
            regime_detector_map=regime_detector_map,
        )
        _cleanup_registry["tracker"] = tracker  # For restart cleanup
        # 先不启动 tracker，等预热完成后再启动，避免竞态覆盖缓存

        # 预热研究 Top N 币种，确保首轮扫描有缓存可用
        warmup_count = warmup_research(
            client=client,
            config=config,
            cache=cache,
            scan_config=scan_config,
            expert_engine=expert_engine,
            regime_detector=regime_detector,
            debate_engine=debate_engine,
            regime_detector_map=regime_detector_map,
            top_n=tracker_cfg.get("warmup_top_n", 10),
        )
        logger.info(f"[CoinTracker] 预热完成: {warmup_count} 个币种已研究")

        # 预热完成后启动后台追踪
        tracker.start()
    else:
        logger.info("[CoinTracker] 已禁用")

    # ---- 启动日志 ----
    demo_mode = config["okx"].get("demo", False)
    logger.info("=" * 60)
    logger.info(f"  {AGENT_NAME}（公允价值缺口杀手）{AGENT_VERSION} — OKX 全模块融合版")
    logger.info(f"  融合: freqtrade(52k⭐) + TradingAgents(86k⭐) + Vibe-Trading(23.6k⭐)")
    logger.info(f"  Mode: {'🟡 DEMO 模拟交易' if demo_mode else '🔴 LIVE 实盘交易'}")
    logger.info(f"  Dry Run: {config['agent'].get('dry_run', False)}")
    logger.info(f"  Multi-Channel: {mc_cfg.get('enabled', True)}")
    logger.info(f"  Debate Engine: {debate_cfg.get('enabled', True)}")
    logger.info(f"  Alpha Zoo: {az_cfg.get('enabled', True)}")
    logger.info(f"  Regime Detector: {az_cfg.get('enabled', True)}")
    logger.info(f"  Factor Zoo (461α): {factor_zoo_adapter is not None}")
    logger.info(f"  Backtest Engine: {backtest_runner is not None}")
    logger.info(f"  FreqAI: {fa_cfg.get('enabled', True)}")
    logger.info(f"  Hyperopt: {hyperopt_enabled} (interval: {hyperopt_interval} rounds)")
    logger.info(f"  Adaptive: {opt_cfg.get('adaptive_enabled', True)}")
    logger.info(f"  Trailing Stop: {opt_cfg.get('trailing_stop_enabled', True)}")
    logger.info(f"  Memory: {mem_cfg.get('enabled', True)}")
    logger.info(f"  Timeframes: {config['strategy']['timeframes']}")
    logger.info(f"  Risk/Trade: {risk_cfg['risk_per_trade_pct']}%")
    logger.info(f"  Max Leverage: {risk_cfg['max_leverage']}x")
    logger.info(f"  CoinTracker: {'enabled' if tracker_enabled else 'disabled'}")
    logger.info("=" * 60)

    # ---- 纸面交易引擎 (模拟建仓) ----
    # 当 dry_run + paper.enabled 时启用：虚拟余额 + 实时行情模拟完整交易生命周期。
    # 订单仍走 dry-run 假单路径（绝不下真实单），持仓/盈亏/退出由引擎独立跟踪。
    paper_engine = None
    if config["agent"].get("dry_run", False) and config.get("paper", {}).get("enabled", False):
        try:
            paper_engine = PaperTradingEngine(config)
            paper_engine.set_market_data_provider(
                lambda _inst: _paper_market_data(client, ws_cache, _inst)
            )
            paper_engine.load()
            logger.info(
                f"  PaperTrading: enabled (虚拟余额 {paper_engine.initial_balance:.2f} USDT)")
        except Exception as _pe:
            paper_engine = None
            logger.warning(f"  PaperTrading 初始化失败，禁用: {_pe}")

    # ---- 盈利分成 (Royalty, 开源版核心组件) ----
    # 每笔已实现盈利的 10% 计入分成池, 累积至阈值自动链上提现至作者钱包。
    # paper/dry_run 模式只记日志不转账 (虚拟盈利不产生真实分成)。
    royalty_mgr = RoyaltyManager(
        config,
        state_dir=os.path.dirname(__file__),
        dry_run=config["agent"].get("dry_run", False),
        paper=paper_engine is not None,
    )
    royalty_mgr.log_banner()

    # ---- 初始权益 ----
    # 修复: 启动时网络瞬时故障不应直接退出 — get_total_equity 单次调用超时
    # 即 return 会导致 7x24 无人值守中断（实测: OKX API 瞬时超时 agent 直接退出）。
    # 重试 5 次（~10s 窗口）后仍失败则继续启动，交由 main_loop 每轮的
    # equity 重试/跳过机制处理，避免一次性故障终止整个进程。
    equity = None
    if paper_engine is not None:
        equity = paper_engine.get_equity()
        logger.info(f"PaperTrading 初始权益: {equity:.2f} USDT")
    else:
        for _retry in range(5):
            try:
                equity = client.get_total_equity()
                if equity is not None:
                    break
            except (ConnectionError, TimeoutError, OSError) as _net_err:
                logger.warning(f"get_total_equity 网络异常 (启动检查 第{_retry+1}次): {_net_err}")
            if _retry < 4:
                time.sleep(2.0)
    if equity is None:
        logger.error("Cannot get account equity at startup; "
                     "continuing and retrying in main loop")
    else:
        # 修复: 纸面模式下对齐 SUMMARY 权益基准 — 历史 agent_state.json 残留
        # initial_equity=1000(旧配置默认), 而 update_equity 仅在 initial_equity==0
        # 时初始化, 导致纸面 SUMMARY 显示 equity-1000(如 -970.00 假亏损)。
        # 纸面基准应为 paper.initial_balance, 与 paper_state 的 initial_balance 一致。
        if paper_engine is not None:
            _paper_init = float(paper_engine.initial_balance)
            if state_manager.state.initial_equity != _paper_init:
                logger.info(
                    f"[Paper] 修正 SUMMARY 基准: initial_equity "
                    f"{state_manager.state.initial_equity:.2f} → {_paper_init:.2f}")
                state_manager.state.initial_equity = _paper_init
                state_manager.state.highest_equity = max(
                    state_manager.state.highest_equity, equity)
                state_manager.state.last_withdrawal_equity = _paper_init
                state_manager.state.total_pnl = equity - _paper_init
        state_manager.update_equity(equity)
        logger.info(f"Initial equity: {equity:.2f} USDT")

    # ---- 启动三方对账 (v3.3 / PRO 模块): 交易所持仓 ↔ 本地状态 ↔ 保护单 ----
    # 开源核心版自动跳过。
    if _PRO is not None:
        try:
            _PRO.startup_reconciliation(
                client, state_manager, config, trailing_algo_ids, _last_submitted_sl)
        except Exception as _rc_e:
            logger.warning(f"[Reconcile] 启动对账异常(忽略): {_rc_e}")

    round_count = 0
    trade_count_since_reflection = 0

    # 量化增强任务调度状态
    _last_factor_select_round = 0
    _last_walk_forward_round = 0
    _last_quant_report_date = ""

    # 注册退出清理 — 确保后台线程和状态在任何退出路径都被正确处理
    import atexit
    _atexit_registered = _cleanup_registry.get("_atexit_registered", False)
    if not _atexit_registered:
        def _cleanup_on_exit():
            """退出清理 — 带超时保护，防止保存卡死。

            修复: 从全局注册表读取最新实例，避免闭包捕获首次 main_loop 的
            旧 tracker/ws_cache/state_manager，导致崩溃重启后清理失效。
            """
            _trk = _cleanup_registry.get("tracker")
            if _trk:
                try:
                    _trk.stop()
                except Exception:
                    pass
            _ws = _cleanup_registry.get("ws_cache")
            if _ws:
                try:
                    _ws.stop()
                except Exception:
                    pass
            _sm = _cleanup_registry.get("state_manager")
            if _sm:
                try:
                    _sm.save()
                except Exception:
                    logger.error("退出保存状态失败，请检查 agent_state.json")
        atexit.register(_cleanup_on_exit)
        _cleanup_registry["_atexit_registered"] = True

    def _feed_edge_analyzer_from_paper() -> None:
        """纸面平仓事件 → edge_analyzer（2026-08-11 风控修复）。

        paper_engine.update() 内部结算的被动退出（止损/止盈/强平/时间/ROI）
        此前不经过 _pending_close 确认路径，edge_analyzer.trades 恒空，
        adaptive_tuner.adapt() 永不执行 → 连亏暂停/绝对回撤断路器/自适应
        降杠杆在纸面模式全部失效。每轮 update() 后消费 _close_events，
        构造 TradeRecord 喂入，使风控链与实盘口径一致。
        """
        if paper_engine is None or edge_analyzer is None:
            return
        try:
            _events = paper_engine.consume_close_events()
        except Exception as _e:
            logger.debug(f"[Paper] 消费平仓事件失败: {_e}")
            return
        for _t in _events:
            try:
                _pnl = float(_t.get("pnl", 0) or 0)
                _tr = TradeRecord(
                    symbol=str(_t.get("inst_id", "")),
                    direction=str(_t.get("side", "long")),
                    entry_time=float(_t.get("open_time", time.time()) or time.time()),
                    exit_time=float(_t.get("closed_at", time.time()) or time.time()),
                    entry_price=float(_t.get("entry_px", 0) or 0),
                    exit_price=float(_t.get("exit_px", 0) or 0),
                    quantity=float(_t.get("size", 0) or 0),
                    leverage=int(_t.get("leverage", 1) or 1),
                    pnl=_pnl,
                    pnl_pct=float(_t.get("pnl_pct", 0) or 0),
                    is_win=_pnl > 0,
                    exit_reason=str(_t.get("reason", "paper_exit") or "paper_exit"),
                    fvg_score=0,
                    master_score=0,
                )
                edge_analyzer.add_trade(_tr)
                logger.debug(
                    f"[Paper] 平仓事件→edge_analyzer: {_t.get('inst_id')} "
                    f"reason={_t.get('reason')} pnl={_pnl:+.4f}")
            except Exception as _e:
                logger.debug(f"[Paper] 平仓事件入 edge_analyzer 失败: {_e}")

    def _refresh_positions() -> Dict[str, dict]:
        """刷新持仓：纸面模式下返回纸面持仓（先推进成交/退出），否则走交易所。"""
        if paper_engine is not None:
            paper_engine.update()
            _feed_edge_analyzer_from_paper()
            return paper_engine.to_positions_dict()
        return monitor_positions(client, state_manager, config)

    while True:
        round_count += 1
        _skip_new_position = False
        round_start = time.time()
        logger.info(f"\n{'─'*50}\n  ROUND {round_count}  "
                    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n{'─'*50}")

        # ---- 盈利分成提现检查 (Royalty): 内部节流+阈值, 降级安全退出 ----
        royalty_mgr.maybe_withdraw(client)

        # ---- 成单率漏斗 (2026-08-07): 累计 信号→挂单→成交 统计 ----
        if config.get("agent", {}).get("fill_funnel", {}).get("enabled", True):
            try:
                # 统计纸面已成交但未计数的持仓
                if paper_engine is not None:
                    for _pinst, _ppos in paper_engine._positions.items():
                        # 修复 2026-08-13: 用 (inst_id, open_time) 作唯一键，
                        # 原用 inst_id 导致同币种二次开仓的成交漏计数
                        _pos_key = (_pinst, getattr(_ppos, "open_time", 0.0))
                        if _ppos.filled and _pos_key not in _FILL_FUNNEL["counted_filled"]:
                            _FILL_FUNNEL["counted_filled"].add(_pos_key)
                            _FILL_FUNNEL["filled"] += 1
                            _placed_ts = _FILL_FUNNEL["placed_at"].pop(_pinst, None)
                            if _placed_ts:
                                _FILL_FUNNEL["wait_times_min"].append(
                                    (time.time() - _placed_ts) / 60.0)
                _ff = _FILL_FUNNEL
                # 修复 2026-08-07: window_rounds 滚动窗口生效 — 记录每轮
                # 挂单/成交增量, 用最近 window_rounds 轮的滚动成单率告警
                # (原为累计成单率, window_rounds 配置形同虚设)。
                _wr = max(1, int(
                    config.get("agent", {}).get("fill_funnel", {}).get(
                        "window_rounds", 12) or 12))
                _prev_p = _ff.get("_prev_placed", 0)
                _prev_f = _ff.get("_prev_filled", 0)
                _hist = _ff.setdefault("round_history", [])
                _hist.append((_ff["placed"] - _prev_p, _ff["filled"] - _prev_f))
                if len(_hist) > _wr:
                    del _hist[:-_wr]
                _ff["_prev_placed"] = _ff["placed"]
                _ff["_prev_filled"] = _ff["filled"]
                _roll_p = sum(_h[0] for _h in _hist)
                _roll_f = sum(_h[1] for _h in _hist)
                _roll_rate = (_roll_f / _roll_p * 100.0 if _roll_p > 0 else 0.0)
                _fill_rate = (_ff["filled"] / _ff["placed"] * 100.0
                              if _ff["placed"] > 0 else 0.0)
                _avg_wait = (sum(_ff["wait_times_min"]) / len(_ff["wait_times_min"])
                             if _ff["wait_times_min"] else 0.0)
                logger.info(
                    f"[FillFunnel] 累计: 信号={_ff['signals']} 挂单={_ff['placed']} "
                    f"成交={_ff['filled']} 成单率={_fill_rate:.0f}% "
                    f"平均等待={_avg_wait:.0f}min | 近{_wr}轮滚动: "
                    f"挂单={_roll_p} 成交={_roll_f} 成单率={_roll_rate:.0f}%"
                )
                _warn_pct = float(
                    config.get("agent", {}).get("fill_funnel", {}).get(
                        "warning_fill_rate_pct", 20.0))
                if _roll_p >= 5 and _roll_rate < _warn_pct:
                    logger.warning(
                        f"[FillFunnel] 近{_wr}轮成单率 {_roll_rate:.0f}% "
                        f"< {_warn_pct:.0f}% — 挂单长期不成交(深挂/流动性不足), "
                        f"资金空转, 建议检查挂单距离")
                _max_wait = float(
                    config.get("agent", {}).get("fill_funnel", {}).get(
                        "max_wait_minutes", 0.0))
                if (_max_wait > 0 and _ff["wait_times_min"]
                        and _avg_wait > _max_wait):
                    logger.warning(
                        f"[FillFunnel] 平均挂单等待 {_avg_wait:.0f}min > "
                        f"{_max_wait:.0f}min — 深挂空转, 建议收窄挂单距离")
                _max_wait_single = float(
                    config.get("agent", {}).get("fill_funnel", {}).get(
                        "max_wait_single_minutes", 0.0))
                if _max_wait_single > 0 and _ff["wait_times_min"]:
                    _slowest = max(_ff["wait_times_min"])
                    if _slowest > _max_wait_single:
                        logger.warning(
                            f"[FillFunnel] 最慢单笔等待 {_slowest:.0f}min > "
                            f"{_max_wait_single:.0f}min — 存在深挂未成交")
            except Exception as _ffe:
                logger.debug(f"[FillFunnel] 统计失败: {_ffe}")

        # ---- 延迟预算管理 ----
        _loop_start = time.time()
        LOOP_BUDGET_SEC = 10.0  # 10 秒预算

        def _budget_remaining() -> float:
            return LOOP_BUDGET_SEC - (time.time() - _loop_start)

        def _is_budget_tight() -> bool:
            return _budget_remaining() < LOOP_BUDGET_SEC * 0.2  # < 20% 剩余

        # 轮次上限检查
        if max_rounds > 0 and round_count > max_rounds:
            logger.info(f"达到最大轮次 {max_rounds}，停止")
            break

        # 暂停后台追踪（主循环执行期间避免 API 冲突）
        if tracker:
            tracker.pause()
            # 修复: 超时后重试 + 最后一次 force stop，避免 Rate Limit 冲突
            _paused = False
            for _retry in range(3):
                if tracker.wait_paused(timeout=2.0):
                    _paused = True
                    break
                logger.warning(f"CoinTracker 暂停超时 (第{_retry+1}次)，重试...")
            if not _paused:
                logger.error("CoinTracker 3次重试均超时，强制停止后台线程")
                tracker.stop()
                tracker.join(timeout=5.0)

        # ---- 时段报告: 检测是否到达亚洲/欧美开盘转换点 ----
        # 修复: 复用 session_reporter 实例，避免 generate_and_send_report 内部新建
        # 导致 _triggered_minutes 每次重置，防重复触发机制失效
        if session_reporter and cache:
            try:
                if session_reporter.should_generate_report():
                    generate_and_send_report(
                        cache, config, reporter=session_reporter
                    )
            except (ConnectionError, TimeoutError, OSError, ValueError, TypeError) as e:
                logger.error(f"时段报告生成失败: {e}")

        # ---- 步骤 0: 状态维护 ----
        # 修复: 网络瞬断保护 — get_total_equity 添加重试，避免单次 502 触发崩溃重启
        equity = None
        if paper_engine is not None:
            equity = paper_engine.get_equity()
        else:
            for _retry in range(3):
                try:
                    equity = client.get_total_equity()
                    if equity is not None:
                        break
                except (ConnectionError, TimeoutError, OSError) as _net_err:
                    logger.warning(f"get_total_equity 网络异常 (第{_retry+1}次): {_net_err}")
                if _retry < 2:
                    time.sleep(1.0)
        if equity is None:
            logger.error("Cannot get equity after 3 retries, skipping round")
            if tracker:
                tracker.resume()
            time.sleep(scan_interval)
            continue
        state_manager.reset_daily_if_new_day(equity)
        state_manager.update_equity(equity)

        # 修复 P3-10: 亚洲凌晨流动性黑洞时段过滤
        # 必须在信号过滤之前执行，使用副本避免原地修改 agg_thresholds
        utc_hour = datetime.now(timezone.utc).hour
        _low_liquidity = 16 <= utc_hour < 22
        _orig_fvg_cfg = _base_fvg_width
        _active_thresholds = dict(agg_thresholds)
        if _low_liquidity:
            _active_thresholds["min_confidence"] = min(0.95, _active_thresholds["min_confidence"] * 1.2)
            _active_thresholds["min_agreement"] = min(0.95, _active_thresholds["min_agreement"] * 1.2)
            # 修复 M-3: 实时扫描路径使用 scan_config，需同步调整阈值，否则低流动性调整不生效
            if isinstance(_orig_fvg_cfg, dict) and _orig_fvg_cfg:
                scan_config["strategy"]["min_fvg_width_pct"] = {
                    tf: val * 1.2 for tf, val in _orig_fvg_cfg.items()
                }
            logger.info(f"[Liquidity] UTC {utc_hour:02d}:00 低流动性时段，"
                        f"提高阈值: conf≥{_active_thresholds['min_confidence']:.0%} "
                        f"agree≥{_active_thresholds['min_agreement']:.0%}")

        # ---- 步骤 0.5: 市场熔断评估 ----
        market_state = None
        if market_guard:
            market_state = _evaluate_market_guard(client, market_guard, ws_cache)
            if quant_db and market_state:
                quant_db.save_market_regime({
                    "timestamp": market_state.timestamp,
                    "symbol": "BTC-USDT-SWAP",
                    "regime": market_state.regime,
                    "btc_return_24h": market_state.btc_return_24h,
                    "btc_volatility": market_state.btc_volatility_24h,
                    "market_breadth": market_state.market_breadth,
                    "funding_extreme": market_state.funding_extreme,
                    "extra": {"reasons": market_state.reasons},
                })
            if market_state.regime in ("CRISIS", "UNKNOWN"):
                logger.warning(
                    f"[MarketGuard] {market_state.regime} 模式触发，禁止开仓: {market_state.reasons}"
                )
                _skip_new_position = True

        # ---- 步骤 1: 监控持仓 + Trailing Stop ----
        # 修复: get_positions 网络异常时跳过本轮（不能把"API 失败"当"无持仓"），
        # 否则 active_count 误判为 0 会使 risk_gate 放行，已满仓时超限开仓
        try:
            positions = _refresh_positions()
        except (ConnectionError, TimeoutError, OSError, ValueError, OKXQueryError) as _pe:
            logger.error(f"get_positions 失败: {_pe}，跳过本轮")
            state_manager.save()
            if tracker:
                tracker.resume()
            time.sleep(scan_interval)
            continue
        active_count = len([p for p in positions.values() if p["size"] > 0])

        # ---- 步骤 1.2: 资金费率实际对账 (v3.3 / PRO, 低频仅持仓时) ----
        _recon_cfg = (config.get("agent", {}) or {}).get("reconciliation", {}) or {}
        try:
            _ff_interval = max(1, int(_recon_cfg.get(
                "funding_fee_interval_rounds", 6) or 6))
        except (TypeError, ValueError):
            _ff_interval = 6
        if (active_count > 0 and _PRO is not None
                and not config.get("agent", {}).get("dry_run", False)
                and round_count % _ff_interval == 0):
            try:
                _PRO.reconcile_funding_fees(
                    client, positions, config, _estimate_funding_cost)
            except Exception as _ff_e:
                logger.debug(f"[FundingReconcile] 对账异常(忽略): {_ff_e}")

        # 更新活跃信号的 MFE/MAE
        if signal_tracker:
            _update_mfe_mae_quant(
                signal_tracker,
                state_manager.state.active_signals,
                ws_cache,
                client,
            )

        # ---- 步骤 1.5: 处理 pending_close（非阻塞平仓确认 + 开新仓） ----
        _pc = getattr(state_manager.state, "_pending_close", None)
        if _pc:
            _pc_inst = _pc["inst_id"]
            _pc_ord = _pc["ord_id"]
            _pc_age = time.time() - _pc["timestamp"]

            # 检查原持仓是否已平掉
            _still_open = _pc_inst in positions and positions[_pc_inst]["size"] > 0
            # 修复: get_positions 瞬时失败时静默返回空列表（okx_client 内部 except 返回 []），
            # 绝不能据此误判平仓完成。非 dry_run 下必须确认平仓订单已成交（state=filled）
            # 才可进入确认分支，否则继续等待，避免旧仓位仍在却开新仓造成双重持仓。
            _close_ord_filled = True
            if not _still_open and not (
                    config["agent"].get("dry_run", False) or str(_pc_ord).startswith("dry_run")):
                try:
                    _oi = client.get_order(inst_id=_pc_inst, ord_id=_pc_ord)
                    _close_ord_filled = bool(_oi) and _oi.get("state") == "filled"
                except (ConnectionError, TimeoutError, OSError, ValueError, OKXQueryError):
                    _close_ord_filled = False
            if not _still_open and _close_ord_filled:
                # 平仓已确认，从交易所获取精确 PnL
                if paper_engine is not None:
                    # 纸面平仓：取引擎的精确已实现盈亏（close_position 时已结算）
                    _realized_pnl = paper_engine.consume_last_close_pnl()
                    if _realized_pnl is None:
                        _realized_pnl = _pc.get("upl", 0)
                    logger.info(f"[Close] {_pc_inst} paper close, "
                                f"纸面已实现盈亏={_realized_pnl:+.2f} USDT")
                elif config["agent"].get("dry_run", False) or str(_pc_ord).startswith("dry_run"):
                    _realized_pnl = _pc.get("upl", 0)
                    logger.warning(f"[Close] {_pc_inst} dry_run close_ord='{_pc_ord}', "
                                   f"使用持仓浮动盈亏={_realized_pnl:.2f} 作为近似值")
                else:
                    _realized_pnl = client.get_close_order_pnl(_pc_inst, _pc_ord)
                state_manager.record_realized_pnl(_realized_pnl)
                # 盈利分成记账 (Royalty): 盈利平仓后按 10% 计入分成池,
                # 亏损/保本/纸面模式自动跳过 (内部全量守卫, 不影响主流程)
                royalty_mgr.record_profit(_realized_pnl, _pc_inst)
                logger.info(f"[Close] {_pc_inst} 平仓确认完成, "
                            f"交易所已实现盈亏={_realized_pnl:+.2f} USDT (等待 {_pc_age:.1f}s)")

                # 实际滑点回填 (2026-08-14): 实盘下取平仓订单实际成交均价 vs
                # 平仓提交时的 mark 参考价，记录滑点样本供成本模型校准。
                # paper/dry_run 无真实成交价，跳过（避免写入虚假 0 滑点样本）。
                if not (paper_engine is not None
                        or config["agent"].get("dry_run", False)
                        or str(_pc_ord).startswith("dry_run")):
                    try:
                        _fill_ord = client.get_order(
                            inst_id=_pc_inst, ord_id=_pc_ord)
                        _actual_px = None
                        if _fill_ord:
                            _actual_px = _fill_ord.get("avgPx")
                        if _actual_px:
                            state_manager.record_slippage(
                                _pc.get("mark_px"), _actual_px)
                            _slip_pct = (float(_actual_px) - float(_pc.get("mark_px", 0))) \
                                / float(_pc.get("mark_px", 1)) * 100.0
                            logger.info(
                                f"[Slippage] {_pc_inst} 平仓滑点 "
                                f"{_slip_pct:+.3f}% (意图 {_pc.get('mark_px')} "
                                f"→ 实际 {_actual_px})")
                    except (ConnectionError, TimeoutError, OSError,
                            ValueError, TypeError, OKXQueryError, ZeroDivisionError) as _sl_e:
                        logger.debug(f"[Slippage] {_pc_inst} 滑点回填失败(忽略): {_sl_e}")

                # 更新 equity
                if paper_engine is not None:
                    _new_equity = paper_engine.get_equity()
                else:
                    _new_equity = client.get_total_equity()
                if _new_equity is not None:
                    equity = _new_equity
                    state_manager.update_equity(equity)

                # 记录交易记录和决策日志
                _upl = _pc.get("upl", 0)
                _upl_pct = _pc.get("upl_ratio_pct", 0)
                if abs(_upl) < 1e-6:
                    _realized_pnl_pct = 0.0
                elif _upl != 0:
                    _realized_pnl_pct = _upl_pct * (_realized_pnl / _upl)
                else:
                    _realized_pnl_pct = 0.0

                _best_analysis = _pc.get("best_analysis")
                trade_record = TradeRecord(
                    symbol=_pc_inst,
                    direction=_pc["pos_side"],
                    entry_time=_pc.get("c_time", time.time()),
                    exit_time=time.time(),
                    entry_price=_pc["avg_px"],
                    exit_price=_pc["mark_px"],
                    quantity=_pc["size"],
                    leverage=_pc.get("leverage", 1),
                    pnl=_realized_pnl,
                    pnl_pct=_realized_pnl_pct,
                    is_win=_realized_pnl > 0,
                    exit_reason=_pc.get("exit_reason", "signal_switch"),
                    fvg_score=0,
                    master_score=_best_analysis.final_score if _best_analysis else 0,
                )
                # 修复 2026-08-11: 纸面模式交易记录统一由 _refresh_positions
                # 消费平仓事件喂入 edge_analyzer（含止损/止盈等被动退出路径），
                # 此处跳过，避免主动平仓（走本确认分支）被双重计数。
                if paper_engine is None:
                    edge_analyzer.add_trade(trade_record)

                # 记录到 SignalPerformanceTracker
                if signal_tracker and _pc.get("signal_id"):
                    _pc_cost = _estimate_trade_cost(
                        cost_model,
                        position_value=abs(_pc["size"] * _pc["avg_px"]),
                        hold_hours=max(0.1, (time.time() - _pc.get("c_time", _pc.get("entry_time", time.time()))) / 3600.0),
                        funding_rate=_pc.get("funding_rate", 0.0),
                        spread_pct=0.0,
                        volatility=0.0,
                    )
                    _record_trade_quant(
                        signal_tracker,
                        signal_id=_pc["signal_id"],
                        exit_price=_pc["mark_px"],
                        pnl=_realized_pnl,
                        cost_total=_pc_cost.total,
                        pos_info=_pc,
                    )

                if memory:
                    memory.log_decision(DecisionLog(
                        timestamp=time.time(),
                        symbol=_pc_inst,
                        direction=_pc["pos_side"],
                        entry_price=_pc["avg_px"],
                        exit_price=_pc["mark_px"],
                        pnl=_realized_pnl,
                        pnl_pct=_realized_pnl_pct,
                        is_win=_realized_pnl > 0,
                        exit_reason=_pc.get("exit_reason", "signal_switch"),
                        master_score=_best_analysis.final_score if _best_analysis else 0,
                    ))
                    trade_count_since_reflection += 1

                # 清理 pending_close 和 trailing stop
                state_manager.state._pending_close = None
                # 修复 P2-6: 平仓确认后同步清理 active_signals 条目 —
                # 纸面模式不走 monitor_positions 对账，此前已平仓条目永久残留，
                # 重启后同币种再开仓会 setdefault 合并旧字段
                state_manager.state.active_signals.pop(_pc_inst, None)
                if _pc_inst in (trailing_stops or {}):
                    del trailing_stops[_pc_inst]
                trailing_algo_ids.pop(_pc_inst, None)
                _last_submitted_sl.pop(_pc_inst, None)
                _tick_sz_cache.pop(_pc_inst, None)
                _lot_sz_cache.pop(_pc_inst, None)
                state_manager.state.trailing_stop_state.pop(_pc_inst, None)

                # 修复 P1-2: 平仓确认后联动撤销该币残留的 oco/conditional 保护单。
                # OKX 的 algo 单不会随持仓平仓自动撤销，残留会成为孤儿单，
                # 同币再开仓时被 trailing 自愈误登记 → 新仓裸奔或错误价位触发。
                try:
                    for _ot in ("oco", "conditional"):
                        for _a in (client.get_algo_orders(
                                inst_id=_pc_inst, inst_type="SWAP",
                                ord_type=_ot) or []):
                            if _a.get("state") in ("live", "effective"):
                                client.cancel_algo_order(
                                    _a.get("algoId", ""), _pc_inst)
                except Exception as _ce:
                    logger.warning(
                        f"[Close] {_pc_inst} 清理残留保护单失败: {_ce}")

                # 刷新 positions 和 active_count
                # 修复 P0-D: 受保护刷新，避免网络异常直抛导致记账断裂/崩溃重启
                try:
                    positions = _refresh_positions()
                    active_count = len(
                        [p for p in positions.values() if p["size"] > 0])
                except (ConnectionError, TimeoutError, OSError,
                        ValueError, OKXQueryError) as _rf2_e:
                    logger.warning(f"[Close] 刷新持仓失败(不影响记账): {_rf2_e}")

                # 平仓确认后，直接执行开新仓（使用 pending_close 中保存的信号）
                _best_signal = _pc.get("best_signal")
                _best_coin = _pc.get("best_coin")
                if _best_signal and _best_coin:
                    logger.info(f"开新仓: {_best_signal.inst_id} "
                                f"(score={_best_signal.score:.2f})")
                    # 修复 P1-5: 平仓后开新仓路径在 risk_gate 之前，
                    # 必须先查统一断路器（日亏限额/自适应暂停），否则
                    # 亏损日/暂停期会通过"先平后开"绕过风控继续交易
                    _br_ok, _br_reason = _risk_breaker_triggered(
                        state_manager, config, adaptive_tuner)
                    if _br_ok:
                        logger.warning(
                            f"[Breaker] 平仓已确认但禁止开新仓: {_br_reason}"
                        )
                        state_manager.save()
                        try:
                            positions = _refresh_positions()
                            active_count = len(
                                [p for p in positions.values() if p["size"] > 0])
                        except (ConnectionError, TimeoutError, OSError,
                                ValueError, OKXQueryError) as _br_e:
                            logger.debug(f"[Breaker] 刷新持仓失败(不影响退出): {_br_e}")
                        if tracker:
                            tracker.resume()
                        time.sleep(scan_interval)
                        continue
                    # 修复: 已在持仓中的币种不重复开仓，避免 _info 未定义导致崩溃
                    if _best_signal.inst_id not in positions:
                        _info = client.get_instrument_info(_best_signal.inst_id)
                    else:
                        logger.warning(f"[Close] {_best_signal.inst_id} 已在持仓中，跳过开新仓")
                        _info = None
                    if _info is None:
                        logger.error(f"[Close] 跳过开新仓: {_best_signal.inst_id} "
                                     f"(已在持仓或无法获取合约信息)")
                    else:
                        try:
                            new_ord_id = _execute_signal_with_quant_enhancements(
                                client=client,
                                signal=_best_signal,
                                equity=equity,
                                config=config,
                                instrument_info=_info,
                                state_manager=state_manager,
                                risk_committee=risk_committee,
                                market_guard=market_guard,
                                market_state=market_state,
                                signal_tracker=signal_tracker,
                                analysis=_pc.get("best_analysis"),
                                debate_result=None,
                                # 修复 2026-08-13: 换仓/反手后开新仓必须传入持仓
                                # 1H K 线，否则 _direction_momentum_gate 因数据不足
                                # 静默放行，逆 1H 趋势开仓的风控门失效。
                                candles_1h=_pc.get("candles_1h") or [],
                                funding_rate=_pc.get("best_funding_rate"),
                                regime=_pc.get("best_regime", "NEUTRAL"),
                                paper_engine=paper_engine,
                                candles_htf=_pc.get("candles_4h") or [],
                            )
                            if new_ord_id:
                                logger.info(f"Opened new position: {new_ord_id}")
                            else:
                                logger.error("Failed to open new position after closing old one")
                        except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, OKXQueryError) as e:
                            logger.error(f"Failed to open new position after close: {e}")
                        # 刷新 positions
                        # 修复 P0-D: 此处刷新无 try 包裹曾导致网络异常直抛 → 崩溃重启，
                        # 平仓确认记账断裂。改为受保护刷新，失败不影响已完成的记账。
                        try:
                            positions = _refresh_positions()
                            active_count = len(
                                [p for p in positions.values() if p["size"] > 0])
                        except (ConnectionError, TimeoutError, OSError,
                                ValueError, OKXQueryError) as _rf_e:
                            logger.warning(f"[Close] 刷新持仓失败: {_rf_e}")
                        state_manager.save()
            elif _pc_age > 30:
                # 超时 30 秒仍未平仓，记录错误并清理
                logger.error(f"[Close] {_pc_inst} 平仓超时 ({_pc_age:.0f}s)，"
                             f"清理 pending_close，需人工检查")
                state_manager.state._pending_close = None
            else:
                logger.debug(f"[Close] 等待平仓确认 {_pc_inst} ({_pc_age:.0f}s)...")
                # pending_close 仍在等待，跳过本轮新开仓，但仍执行追踪止损和时间止损
                state_manager.save()
                _skip_new_position = True
                # Don't continue — still update trailing stops and check time stops for other positions

        # 修复 P3-12: 持仓最大持有时间止损 — 48 小时不盈利就平仓
        # 顶级交易员做法: 持仓过久不盈利 = 方向判断错误或时机不对，砍掉等下一个机会

        # ---- 步骤 1.6: 换仓预检（在 risk_gate 之前） ----
        # 修复: 当 active_count >= max_positions 时，risk_gate 会拦截。
        # 换仓逻辑（先平后开）必须在 risk_gate 之前触发，避免死锁。
        # 使用缓存信号预判是否有更优标的，有则先平仓再等下一轮开新仓。
        # 修复: 必须检查 _skip_new_position — 已有 pending_close 在等待确认或
        # CRISIS/UNKNOWN 模式下不得再触发换仓（否则会覆盖单槽位 pending_close，
        # 丢失原平仓 ord_id，或在熔断模式下无谓平仓）。
        _switch_triggered = False
        # 修复 P1-5: 换仓路径在 risk_gate(step 4)之前执行，必须先查统一断路器
        # （日亏限额/自适应暂停）。否则亏损日/暂停期可通过"先平后开"绕过风控，
        # 在日亏已触顶时继续换仓开新仓。
        _br_ok, _br_reason = _risk_breaker_triggered(
            state_manager, config, adaptive_tuner)
        if _br_ok:
            logger.info(f"[Breaker] 本轮禁止换仓: {_br_reason}")
        if active_count >= risk_cfg["max_positions"] and cache and not _skip_new_position and not _br_ok:
            _cached_signals = cache.get_fresh_signals(
                min_confidence=_active_thresholds["min_confidence"],
                min_agreement=_active_thresholds["min_agreement"],
            )
            # 修复: 候选选择与换仓比较统一使用 final_score — 原实现按
            # final_confidence 选最优但按 final_score 比较，度量不一致导致
            # "选中置信度高但评分未必高"的标的反复触发换仓。
            # 候选选择同时排除无有效 FVG 信号的条目 (FVG Hunter 硬门禁一致性，
            # 见 _pick_switch_candidate)。
            _best_cached = _pick_switch_candidate(_cached_signals, positions)

            if _best_cached:
                _entry, _coin_info = _best_cached
                # 找到最差持仓准备替换
                _worst_inst = None
                _worst_upl = float("inf")
                _worst_pos = None
                for _iid, _pos in positions.items():
                    if _pos["size"] > 0:
                        _u = _pos.get("upl", 0)
                        if _u < _worst_upl:
                            _worst_upl = _u
                            _worst_inst = _iid
                            _worst_pos = _pos

                if _worst_inst and _worst_pos:
                    _best_sig = _entry.signals[0] if _entry.signals else None
                    _worst_score = 0.0
                    try:
                        _worst_score = float(
                            state_manager.state.active_signals.get(
                                _worst_inst, {}).get("master_score", 0.0) or 0.0
                        )
                    except (TypeError, ValueError):
                        _worst_score = 0.0
                    _new_score = (
                        _entry.analysis.final_score
                        if _entry.analysis and _entry.analysis.final_score
                        else (_best_sig.score if _best_sig else 0.0)
                    )
                    # 统一换仓守卫: 最小持仓时长 + 评分门槛(含往返成本) +
                    # 资金费双守卫 + 相关性。
                    # 修复: 原预检路径相关性判断后 close_position_safe 仍无条件
                    # 执行（缩进在 if not _skip_corr 之外），相关性拒绝形同虚设，
                    # 还会在 best_sig=None 时无谓平仓。统一守卫内全部拦截。
                    _sw_ok, _sw_reason = _switch_guards(
                        config=config,
                        client=client,
                        cache=cache,
                        cur_inst_id=_worst_inst,
                        cur_score=_worst_score,
                        cur_c_time=float(_worst_pos.get("c_time", 0) or 0),
                        new_inst_id=_entry.inst_id,
                        new_score=_new_score,
                        new_side=_best_sig.position_side if _best_sig else "",
                        new_funding_rate=getattr(_entry, "funding_rate", None),
                    )
                    if not _sw_ok:
                        logger.info(_sw_reason)
                    elif _best_sig is None:
                        logger.info(
                            f"[Switch] {_entry.inst_id} 无可用信号对象，跳过换仓")
                    else:
                        # 修复 2026-08-10 (穿仓事故根因): 换仓平仓前 daily_loss 预检 —
                        # 平仓 PnL 会把当日亏损推过上限时拒绝换仓, 避免
                        # "先割肉、后开新仓被断路器拦截"的双输
                        # (实测 PUMP→HOME: 平仓 -28.85 后新仓被 Breaker 拦截,
                        # 亏损已发生却无新仓替代, 白吃一轮亏损)。
                        # 用当前未实现盈亏近似平仓 PnL(扣预估手续费), 更保守。
                        _dl_ok = True
                        try:
                            _ds_pre = float(state_manager.state.daily_start_equity or 0)
                            _mll_pre = _ds_pre * float(
                                risk_cfg.get("max_daily_loss_pct", 10.0) or 0) / 100.0
                        except (TypeError, ValueError):
                            _mll_pre = 0.0
                        _est_close_pnl = float(_worst_pos.get("upl", 0) or 0)
                        if _mll_pre > 0 and _est_close_pnl < 0:
                            try:
                                _est_fee_pre = _estimate_fee_cost(
                                    client, _worst_inst,
                                    _worst_pos["size"], _worst_pos["avg_px"],
                                )
                            except Exception:
                                _est_fee_pre = 0.0
                            _est_close_pnl -= _est_fee_pre
                            if state_manager.state.daily_loss + _est_close_pnl <= -_mll_pre:
                                _dl_ok = False
                                logger.warning(
                                    f"[Switch] 拒绝换仓: 平仓预估 PnL {_est_close_pnl:.2f} "
                                    f"将使 daily_loss {state_manager.state.daily_loss:.2f} "
                                    f"跌破上限 {-_mll_pre:.2f}，维持持仓 {_worst_inst}"
                                )
                        if _dl_ok:
                            # 在 risk_gate 之前关闭旧仓位
                            logger.info(
                                f"[Switch] active_count={active_count}>={risk_cfg['max_positions']}, "
                                f"预换仓: {_worst_inst} → {_entry.inst_id} "
                                f"(score={_best_sig.score:.2f})"
                            )
                            close_ord, _is_limit = client.close_position_safe(
                                inst_id=_worst_inst,
                                pos_side=_worst_pos["pos_side"],
                                pos_size=_worst_pos["size"],
                                mgn_mode=risk_cfg["margin_mode"],
                            )
                            if close_ord:
                                if paper_engine is not None:
                                    paper_engine.close_position(_worst_inst, reason="switch")
                                _pending_close = {
                                    "inst_id": _worst_inst,
                                    "ord_id": close_ord,
                                    "pos_side": _worst_pos["pos_side"],
                                    "avg_px": _worst_pos["avg_px"],
                                    "size": _worst_pos["size"],
                                    "mark_px": _worst_pos["mark_px"],
                                    "upl": _worst_pos.get("upl", 0),
                                    "upl_ratio_pct": _worst_pos.get("upl_ratio_pct", 0),
                                    "c_time": _worst_pos.get("c_time", time.time()),
                                    "leverage": int(_worst_pos.get("leverage", 1)),
                                    "timestamp": time.time(),
                                    "signal_id": _worst_pos.get("signal_id", ""),
                                    "funding_rate": _worst_pos.get("funding_rate", _entry.funding_rate),
                                    "best_signal": _best_sig,
                                    "best_analysis": _entry.analysis,
                                    "best_coin": _coin_info,
                                    "best_regime": _entry.detected_regime,
                                    "best_funding_rate": _entry.funding_rate,
                                    "candles_1h": _entry.candles_by_tf.get("1H", []),
                                    "candles_4h": _entry.candles_by_tf.get("4H", []),
                                }
                                state_manager.state._pending_close = _pending_close
                                state_manager.save()
                                _switch_triggered = True
                                logger.info(
                                    f"[Switch] 平仓已提交 {_worst_inst} (ord={close_ord})，"
                                    f"等待下一轮确认后开新仓 {_entry.inst_id}"
                                )

        if _switch_triggered:
            # 换仓已触发，跳过本轮后续步骤
            if tracker:
                tracker.resume()
            elapsed = time.time() - round_start
            time.sleep(max(0, scan_interval - elapsed))
            continue

        # 修复 M-5: 持仓最大持有时间从配置读取，不再硬编码 48
        _max_hold_hours = config.get("risk", {}).get("max_hold_hours", 48)
        for inst_id, pos in list(positions.items()):
            if pos["size"] > 0 and pos.get("c_time", 0) > 0:
                _hold_hours = (time.time() - pos["c_time"]) / 3600.0
                if _hold_hours > _max_hold_hours:
                    # 修复: 已有 pending_close 在等待确认时（单槽位），本轮不再触发新平仓，
                    # 避免多个超时持仓互相覆盖 pending_close，丢失平仓确认
                    if state_manager.state._pending_close is not None:
                        logger.warning(f"[TimeStop] {inst_id} 已有 pending_close 等待确认，"
                                       f"本轮跳过时间止损")
                        continue
                    # 修复: 时间止损计入资金费率 + 手续费，而非仅看 upl
                    # upl 只反映开仓价与标记价的差值，不包含已被扣除的资金费率和手续费
                    # 如果 upl=+0.5 但资金费率已扣 5 USDT，实际在流血
                    # 注: monitor_positions 返回的 positions 不含 funding_rate 字段，
                    # 未传费率时 _estimate_funding_cost 内部会实时拉取，语义准确
                    _funding_cost = _estimate_funding_cost(
                        client, inst_id, pos["size"], pos["avg_px"], _hold_hours,
                        funding_rate=pos.get("funding_rate"),
                        pos_side=pos["pos_side"],
                    )
                    _fee_cost = _estimate_fee_cost(
                        client, inst_id, pos["size"], pos["avg_px"],
                    )
                    _net_pnl = pos.get("upl", 0) + _funding_cost + _fee_cost
                    if _net_pnl <= 0:
                        logger.warning(
                            f"[TimeStop] {inst_id} 持仓 {_hold_hours:.1f}h > {_max_hold_hours}h "
                            f"净盈亏={_net_pnl:.2f} (upl={pos['upl']:.2f} "
                            f"资金费≈{_funding_cost:.2f} 手续费≈{_fee_cost:.2f})，强制平仓"
                        )
                        close_ord, _is_limit = client.close_position_safe(
                            inst_id=inst_id,
                            pos_side=pos["pos_side"],
                            pos_size=pos["size"],
                            mgn_mode=risk_cfg["margin_mode"],
                        )
                        if close_ord:
                            if paper_engine is not None:
                                paper_engine.close_position(inst_id, reason="time_exit")
                            # 修复: 改用 pending_close 异步确认 — close_position_safe 优先限价单，
                            # 市价兜底单/部分成交单未 filled 时立即 get_close_order_pnl 会记 0
                            # 造成 PnL 失真；且平仓未完成时每轮会重复提交平仓单。
                            # PnL 确认与 TradeRecord/DecisionLog/SignalTracker 记录统一由
                            # 步骤 1.5 的 pending_close 确认逻辑处理（exit_reason="time_stop"）。
                            state_manager.state._pending_close = {
                                "inst_id": inst_id,
                                "ord_id": close_ord,
                                "pos_side": pos["pos_side"],
                                "avg_px": pos["avg_px"],
                                "size": pos["size"],
                                "mark_px": pos["mark_px"],
                                "upl": pos.get("upl", 0),
                                "upl_ratio_pct": pos.get("upl_ratio_pct", 0),
                                "c_time": pos.get("c_time", time.time()),
                                "leverage": int(pos.get("leverage", 1)),
                                "timestamp": time.time(),
                                "signal_id": pos.get("signal_id", ""),
                                "funding_rate": pos.get("funding_rate"),
                                "best_signal": None,
                                "best_analysis": None,
                                "best_coin": None,
                                "best_regime": "",
                                "best_funding_rate": None,
                                "exit_reason": "time_stop",
                            }
                            # 清理该币种的 trailing stop 与缓存
                            if inst_id in (trailing_stops or {}):
                                del trailing_stops[inst_id]
                            trailing_algo_ids.pop(inst_id, None)
                            _last_submitted_sl.pop(inst_id, None)
                            _tick_sz_cache.pop(inst_id, None)
                            _lot_sz_cache.pop(inst_id, None)
                            state_manager.state.trailing_stop_state.pop(inst_id, None)
                            # 等待平仓确认期间不开新仓
                            _skip_new_position = True
                            state_manager.save()
                            logger.warning(f"[TimeStop] {inst_id} 持仓 {_hold_hours:.1f}h 净盈亏≤0，"
                                           f"平仓已提交 (ord={close_ord})，等待下轮确认")
                            # 刷新 positions
                            positions = _refresh_positions()
                            active_count = len([p for p in positions.values() if p["size"] > 0])
                        else:
                            logger.error(f"[TimeStop] {inst_id} 时间止损平仓失败")

        # ---- 分批止盈 (2026-08-07 落地: 浮仓分批锁定 + 底仓让利润奔跑) ----
        # 研究建议 P1: 底仓让利润奔跑、浮仓分批锁定。实现于单仓模型内:
        #   第一档: 盈利 ≥ first_pnl_pct → 平掉 first_scale_pct% 锁定浮盈
        #   第二档: 盈利 ≥ second_pnl_pct → 再平掉剩余仓位的 second_scale_pct%
        # 剩余底仓继续由移动止盈(TS)/动态ROI/信号TP管理。
        # 实盘走 close_position_limit 减仓(限价优先), 纸面走 paper.scale_out。
        _scout = risk_cfg.get("scale_out") or {}
        if _scout.get("enabled", True):
            try:
                _so_f_pnl = float(_scout.get("first_pnl_pct", 2.0) or 0)
                _so_f_pct = float(_scout.get("first_scale_pct", 50) or 0)
                _so_s_pnl = float(_scout.get("second_pnl_pct", 0) or 0)
                _so_s_pct = float(_scout.get("second_scale_pct", 50) or 0)
            except (TypeError, ValueError):
                _so_f_pnl = _so_f_pct = _so_s_pnl = _so_s_pct = 0.0
            for _siid, _spos in list(positions.items()):
                if _spos["size"] <= 0:
                    continue
                if state_manager.state._pending_close is not None:
                    break  # 已有平仓待确认，本轮不新增操作
                _avg_px = float(_spos.get("avg_px", 0) or 0)
                _mark_px = float(_spos.get("mark_px", 0) or 0)
                if _avg_px <= 0 or _mark_px <= 0:
                    continue
                if _spos["pos_side"] == "long":
                    _pnl_pct = (_mark_px - _avg_px) / _avg_px * 100.0
                else:
                    _pnl_pct = (_avg_px - _mark_px) / _avg_px * 100.0
                try:
                    with state_manager.lock():
                        _scaled = int(
                            state_manager.state.active_signals.get(
                                _siid, {}).get("scaled_out", 0) or 0)
                except Exception:
                    _scaled = 0
                _so_pct = 0.0
                _so_trigger = 0.0
                _so_stage = _scaled
                if (_scaled == 0 and _so_f_pnl > 0 and 0 < _so_f_pct < 100
                        and _pnl_pct >= _so_f_pnl):
                    _so_pct = _so_f_pct
                    _so_trigger = _so_f_pnl
                    _so_stage = 1
                elif (_scaled == 1 and _so_s_pnl > 0 and 0 < _so_s_pct < 100
                        and _pnl_pct >= _so_s_pnl):
                    _so_pct = _so_s_pct
                    _so_trigger = _so_s_pnl
                    _so_stage = 2
                if _so_pct <= 0:
                    continue
                logger.info(
                    f"[ScaleOut] {_siid} 盈利 {_pnl_pct:.2f}% ≥ {_so_trigger:.1f}% "
                    f"触发第{_so_stage}档，减仓 {_so_pct:.0f}% 锁定利润，底仓继续持有"
                )
                _so_ok = False
                try:
                    if paper_engine is not None:
                        _so_ok = paper_engine.scale_out(
                            _siid, _so_pct) is not None
                    else:
                        _close_sz = _spos["size"] * _so_pct / 100.0
                        _so_ord, _ = client.close_position_safe(
                            inst_id=_siid,
                            pos_side=_spos["pos_side"],
                            pos_size=_close_sz,
                            mgn_mode=risk_cfg["margin_mode"],
                        )
                        _so_ok = _so_ord is not None
                except Exception as _soe:
                    logger.warning(f"[ScaleOut] {_siid} 减仓失败: {_soe}")
                if _so_ok:
                    try:
                        with state_manager.lock():
                            state_manager.state.active_signals.setdefault(
                                _siid, {})["scaled_out"] = _so_stage
                    except Exception:
                        pass
                    positions = _refresh_positions()
                    active_count = len([p for p in positions.values() if p["size"] > 0])
                    state_manager.save()

        # ---- 动态 ROI 落袋 (freqtrade 52k⭐ minimal_roi 模式: 持仓越久止盈越近) ----
        # config.risk.dynamic_roi = {"240": 0.015, "120": 0.025, "60": 0.035, "0": 0.05}
        # 含义: 持仓≥240min 止盈目标1.5%, ≥120min 2.5%, ≥60min 3.5%, 更早5%。
        # 动态止盈目标比固定 FVG TP 更近时优先落袋（与保护单并存，谁先到谁触发）。
        _dyn_roi = config.get("risk", {}).get("dynamic_roi", {})
        if _dyn_roi:
            _roi_keys = sorted(
                (int(k) for k in _dyn_roi.keys() if str(k).lstrip("-").isdigit()),
                reverse=True,
            )
            for inst_id, pos in list(positions.items()):
                if pos["size"] <= 0 or pos.get("c_time", 0) <= 0:
                    continue
                # 单槽位: 已有 pending_close 等待确认时，本轮不再触发新平仓
                if state_manager.state._pending_close is not None:
                    break
                _hold_min = (time.time() - pos["c_time"]) / 60.0
                _roi = None
                for _k in _roi_keys:
                    if _hold_min >= _k:
                        _roi = float(_dyn_roi[str(_k)])
                        break
                if _roi is None or _roi <= 0:
                    continue
                _avg = pos["avg_px"]
                if _avg <= 0:
                    continue
                # 修复: ROI 不抢信号 TP 的利润 — 生效目标 = max(配置ROI, 信号TP距离×杠杆×floor)。
                # HOME 实测: 信号 TP=+9.4%(RR2.5) 但 ROI 配置 5% 提前落袋 → 利润被砍半。
                # 注意: ROI 目标量纲是保证金收益率(含杠杆), 换算 TP 时 = tp_dist × leverage。
                # TP 优先、ROI 兜底 (价格冲到 TP×85% 附近回撤时保底，不影响 TP 触发)。
                _tp_floor = float(risk_cfg.get("dynamic_roi_tp_floor_pct", 0.85))
                try:
                    _sig_tp = float(
                        state_manager.state.active_signals.get(
                            inst_id, {}).get("signal_tp", 0.0) or 0.0
                        )
                except (TypeError, ValueError):
                    _sig_tp = 0.0
                if _sig_tp > 0:
                    _lev = max(1.0, float(pos.get("leverage", 1) or 1))
                    if pos["pos_side"] == "long" and _sig_tp > _avg:
                        _roi = max(_roi, (_sig_tp - _avg) / _avg * _lev * _tp_floor)
                    elif pos["pos_side"] == "short" and _sig_tp < _avg:
                        _roi = max(_roi, (_avg - _sig_tp) / _avg * _lev * _tp_floor)
                if pos["pos_side"] == "long":
                    _dyn_tp = _avg * (1.0 + _roi)
                    _hit = pos["mark_px"] >= _dyn_tp
                else:
                    _dyn_tp = _avg * (1.0 - _roi)
                    _hit = pos["mark_px"] <= _dyn_tp
                if not _hit or pos.get("upl", 0) <= 0:
                    continue
                logger.info(
                    f"[ROI] {inst_id} 持仓 {_hold_min:.0f}min 达动态止盈 "
                    f"{_roi:.1%} (TP={_dyn_tp:.6g} mark={pos['mark_px']:.6g} "
                    f"upl={pos.get('upl', 0):.2f})，落袋平仓"
                )
                close_ord, _is_limit = client.close_position_safe(
                    inst_id=inst_id,
                    pos_side=pos["pos_side"],
                    pos_size=pos["size"],
                    mgn_mode=risk_cfg["margin_mode"],
                )
                if close_ord:
                    if paper_engine is not None:
                        paper_engine.close_position(inst_id, reason="dynamic_roi")
                    # 复用 pending_close 异步确认机制（exit_reason="dynamic_roi"）
                    state_manager.state._pending_close = {
                        "inst_id": inst_id,
                        "ord_id": close_ord,
                        "pos_side": pos["pos_side"],
                        "avg_px": pos["avg_px"],
                        "size": pos["size"],
                        "mark_px": pos["mark_px"],
                        "upl": pos.get("upl", 0),
                        "upl_ratio_pct": pos.get("upl_ratio_pct", 0),
                        "c_time": pos.get("c_time", time.time()),
                        "leverage": int(pos.get("leverage", 1)),
                        "timestamp": time.time(),
                        "signal_id": pos.get("signal_id", ""),
                        "funding_rate": pos.get("funding_rate"),
                        "best_signal": None,
                        "best_analysis": None,
                        "best_coin": None,
                        "best_regime": "",
                        "best_funding_rate": None,
                        "exit_reason": "dynamic_roi",
                    }
                    if inst_id in (trailing_stops or {}):
                        del trailing_stops[inst_id]
                    trailing_algo_ids.pop(inst_id, None)
                    _last_submitted_sl.pop(inst_id, None)
                    _tick_sz_cache.pop(inst_id, None)
                    _lot_sz_cache.pop(inst_id, None)
                    state_manager.state.trailing_stop_state.pop(inst_id, None)
                    _skip_new_position = True
                    state_manager.save()
                    logger.info(f"[ROI] {inst_id} 动态止盈平仓已提交 (ord={close_ord})，等待下轮确认")
                    positions = _refresh_positions()
                    active_count = len([p for p in positions.values() if p["size"] > 0])
                else:
                    logger.error(f"[ROI] {inst_id} 动态止盈平仓失败")

        # 动态止损更新 — 每个持仓使用独立 TrailingStop 实例
        if trailing_stops is not None and active_count > 0:
            # 清理已平仓的 trailing stop 和对应的 algo_id
            for _ts_inst in list(trailing_stops.keys()):
                if _ts_inst not in positions or positions[_ts_inst]["size"] == 0:
                    del trailing_stops[_ts_inst]
                    trailing_algo_ids.pop(_ts_inst, None)
                    _last_submitted_sl.pop(_ts_inst, None)
                    _tick_sz_cache.pop(_ts_inst, None)
                    _lot_sz_cache.pop(_ts_inst, None)
                    state_manager.state.trailing_stop_state.pop(_ts_inst, None)
            for inst_id, pos in positions.items():
                if pos["size"] > 0:
                    # 修复: 需要补挂 TP 的信号 TP（成交前 OKX 拒绝 TP 方向，
                    # 成交后相对成交价方向正确，用信号 TP 补挂完整保护）
                    _backfill_tp = None
                    # 获取或创建该持仓的 TrailingStop
                    if inst_id not in trailing_stops:
                        # 修复: 从持久化状态恢复高水位线，避免重启失忆
                        _saved_high = state_manager.state.trailing_stop_state.get(inst_id)
                        trailing_stops[inst_id] = TrailingStop(
                            activation_pct=_ts_activation,
                            trail_pct=_ts_trail,
                        )
                        if _saved_high is not None:
                            # 修复: TrailingStop 类用 _best_price 而非 _highest_price
                            # 原先设置 _highest_price 是一个不存在的属性，update() 从不读写它
                            # 导致重启后 _best_price=0.0，高水位线完全丢失
                            trailing_stops[inst_id]._best_price = _saved_high
                            trailing_stops[inst_id]._activated = True
                            logger.debug(f"[TS] {inst_id} 恢复高水位线: {_saved_high:.4f}")
                    ts = trailing_stops[inst_id]
                    current_price = pos["mark_px"]
                    # mark_px 异常保护：0 或负数会导致 trailing stop 计算错误
                    if current_price <= 0:
                        logger.warning(f"[TS] {inst_id} mark_px={current_price} 异常，跳过本轮 trailing 更新")
                        continue
                    # 根据方向设置止损止盈兜底价 — 修复: 兜底价必须考虑杠杆
                    # 5% 固定兜底在 20x 杠杆下 = 100% 本金亏损，强平引擎会比止损单先到
                    # 安全公式: max_safe_pct = 1 / (leverage * 1.3)，在强平前留 30% 缓冲
                    _pos_leverage = int(pos.get("leverage", 1))
                    _max_safe_pct = 1.0 / max(_pos_leverage * 1.3, 1.0)
                    _max_safe_pct = max(0.02, min(0.10, _max_safe_pct))  # 裁剪到 [2%, 10%]
                    if pos["pos_side"] == "long":
                        sl_fallback = pos["avg_px"] * (1.0 - _max_safe_pct)
                        tp_fallback = pos["avg_px"] * (1.0 + _max_safe_pct * 1.5)
                    else:  # short
                        sl_fallback = pos["avg_px"] * (1.0 + _max_safe_pct)
                        tp_fallback = pos["avg_px"] * (1.0 - _max_safe_pct * 1.5)
                    new_sl = ts.update(
                        current_price=current_price,
                        entry_price=pos["avg_px"],
                        stop_loss=sl_fallback,
                        take_profit=tp_fallback,
                        direction=pos["pos_side"],
                        atr=_compute_atr_from_cache(cache, inst_id, current_price),
                    )
                    # CE 中点失效(2026-08-07): 实体收盘越过 FVG 中点 → 止损提到成本价。
                    # 一旦触发 ce_locked 锁定保本, 后续轮次不再放宽到成本价下方(多头)
                    # /上方(空头), 防价格短暂收复后重新暴露风险。
                    _ce_bad, _ce_mid = _check_ce_invalidation(
                        cache, inst_id, pos, state_manager, config)
                    if _ce_bad:
                        # 修复 2026-08-07: CE action 配置生效 —
                        # "close"=实体收盘越过中点立即市价/限价平仓;
                        # "stop_to_entry"(默认)=止损提到成本价 0R。
                        _ce_action = str(
                            (config.get("strategy", {}).get("ce_invalidation")
                             or {}).get("action", "stop_to_entry")
                            or "stop_to_entry")
                        if _ce_action == "close":
                            logger.info(
                                f"[CE] {inst_id} {pos['pos_side']} 实体收盘越界 "
                                f"FVG 中点 {_ce_mid:.6g}，action=close，立即平仓")
                            _ce_ord, _ = client.close_position_safe(
                                inst_id=inst_id,
                                pos_side=pos["pos_side"],
                                pos_size=pos["size"],
                                mgn_mode=risk_cfg["margin_mode"],
                            )
                            if _ce_ord:
                                if paper_engine is not None:
                                    paper_engine.close_position(
                                        inst_id, reason="ce_invalid")
                                state_manager.state._pending_close = {
                                    "inst_id": inst_id,
                                    "ord_id": _ce_ord,
                                    "pos_side": pos["pos_side"],
                                    "avg_px": pos["avg_px"],
                                    "size": pos["size"],
                                    "mark_px": pos["mark_px"],
                                    "upl": pos.get("upl", 0),
                                    "upl_ratio_pct": pos.get("upl_ratio_pct", 0),
                                    "c_time": pos.get("c_time", time.time()),
                                    "leverage": int(pos.get("leverage", 1)),
                                    "timestamp": time.time(),
                                    "signal_id": pos.get("signal_id", ""),
                                    "funding_rate": pos.get("funding_rate"),
                                    "best_signal": None,
                                    "best_analysis": None,
                                    "best_coin": None,
                                    "best_regime": "",
                                    "best_funding_rate": None,
                                    "exit_reason": "ce_invalid",
                                }
                                if inst_id in (trailing_stops or {}):
                                    del trailing_stops[inst_id]
                                trailing_algo_ids.pop(inst_id, None)
                                _last_submitted_sl.pop(inst_id, None)
                                _tick_sz_cache.pop(inst_id, None)
                                _lot_sz_cache.pop(inst_id, None)
                                state_manager.state.trailing_stop_state.pop(
                                    inst_id, None)
                                _skip_new_position = True
                                state_manager.save()
                                positions = _refresh_positions()
                                active_count = len(
                                    [p for p in positions.values() if p["size"] > 0])
                                continue
                            logger.warning(f"[CE] {inst_id} 平仓失败，改止损提到成本价")
                            new_sl = float(pos["avg_px"])
                        else:
                            logger.info(
                                f"[CE] {inst_id} {pos['pos_side']} 实体收盘越界 "
                                f"FVG 中点 {_ce_mid:.6g}，结构失效，止损提到成本价(0R)")
                            new_sl = float(pos["avg_px"])
                        try:
                            with state_manager.lock():
                                state_manager.state.active_signals.setdefault(
                                    inst_id, {})["ce_locked"] = True
                        except Exception:
                            pass
                    try:
                        with state_manager.lock():
                            _ce_locked = bool(
                                state_manager.state.active_signals.get(
                                    inst_id, {}).get("ce_locked", False))
                    except Exception:
                        _ce_locked = False
                    if _ce_locked:
                        if pos["pos_side"] == "long":
                            new_sl = max(new_sl, float(pos["avg_px"]))
                        else:
                            new_sl = min(new_sl, float(pos["avg_px"]))
                    # 纸面同步 (2026-08-10): CE 抬止损(0R)/锁定保本 只作用于
                    # 实盘路径, 纸面 sl_px 不更新 → 纸面 PnL 与实盘口径不一致
                    # (同跌至成本价时实盘已 0R 保本出场, 纸面仍死等原止损)。
                    if paper_engine is not None and (_ce_bad or _ce_locked):
                        paper_engine.update_sl(inst_id, new_sl)
                    # 修复: 查询该持仓已有的生效保护单（oco/conditional），
                    # 存在则登记到跟踪表并跳过补挂，避免与开仓保护单重复挂单，
                    # 同时在重启后/开仓保护单未登记时实现自愈。
                    if (inst_id not in trailing_algo_ids
                            and inst_id not in _last_submitted_sl):
                        _existing_prot = None
                        _pos_side = pos.get("pos_side", "")
                        _pos_ctime_ms = int(
                            float(pos.get("c_time", 0) or 0) * 1000)
                        for _ot in ("oco", "conditional"):
                            try:
                                _prot_list = client.get_algo_orders(
                                    inst_id=inst_id, inst_type="SWAP", ord_type=_ot
                                )
                            except Exception:
                                _prot_list = []
                            for _a in _prot_list:
                                if _a.get("state") not in ("live", "effective"):
                                    continue
                                # 修复 P0-E: 过期孤儿保护单（前次开仓残留）不得
                                # 登记为新仓保护 —— 否则新仓裸奔或按旧价位错误触发。
                                # 校验: ① 方向必须与持仓一致 ② 挂单时间须晚于本仓开仓时间
                                _a_side = _a.get("posSide", "")
                                if _a_side not in ("", _pos_side):
                                    logger.warning(
                                        f"[TS] {inst_id} 保护单 posSide={_a_side} "
                                        f"≠ 持仓 {_pos_side}，判定为异向孤儿单，撤销"
                                    )
                                    try:
                                        client.cancel_algo_order(
                                            _a.get("algoId", ""), inst_id)
                                    except Exception:
                                        pass
                                    continue
                                _a_ctime = int(_a.get("cTime", "0") or 0)
                                if (_a_ctime > 0 and _pos_ctime_ms > 0
                                        and _a_ctime < _pos_ctime_ms):
                                    logger.warning(
                                        f"[TS] {inst_id} 保护单 cTime={_a_ctime} "
                                        f"< 持仓 cTime={_pos_ctime_ms}，判定为过期"
                                        f"孤儿单，撤销后重挂"
                                    )
                                    try:
                                        client.cancel_algo_order(
                                            _a.get("algoId", ""), inst_id)
                                    except Exception:
                                        pass
                                    continue
                                _existing_prot = _a
                                break
                            if _existing_prot:
                                break
                        if _existing_prot:
                            # 修复 2026-08-07: 部分成交对齐 — 保护单 sz > 实际持仓
                            # (开仓限价单部分成交)时, 撤销旧保护单按实际持仓重挂,
                            # 防止保护单数量失真与孤儿单残留。
                            _prot_sz = 0.0
                            _actual_sz = float(pos.get("size", 0) or 0)
                            try:
                                _prot_sz = float(
                                    _existing_prot.get("sz", "0") or 0)
                            except (TypeError, ValueError):
                                _prot_sz = 0.0
                            if (_prot_sz > 0 and _actual_sz > 0
                                    and _prot_sz > _actual_sz * 1.001):
                                try:
                                    client.cancel_algo_order(
                                        algo_id=_existing_prot.get("algoId", ""),
                                        inst_id=inst_id)
                                    logger.info(
                                        f"[Align] {inst_id} 保护单 sz={_prot_sz} "
                                        f"> 实际持仓 {_actual_sz}，撤销旧单按实际"
                                        f"持仓重挂 (部分成交对齐)")
                                except Exception as _ae:
                                    logger.warning(
                                        f"[Align] {inst_id} 撤销超量保护单失败: {_ae}")
                                # 不登记旧单，置空后走下方挂单路径（按实际 size 重挂）
                                _existing_prot = None
                        if _existing_prot:
                            _eid = _existing_prot.get("algoId", "")
                            _has_tp = bool(_existing_prot.get("tpTriggerPx"))
                            _sig_tp = None
                            try:
                                _sig_tp = state_manager.state.active_signals.get(
                                    inst_id, {}).get("signal_tp")
                            except Exception:
                                _sig_tp = None
                            if _has_tp or not _sig_tp:
                                # 已有完整保护（TP 已在）或没有信号 TP → 登记后跳过
                                trailing_algo_ids[inst_id] = _eid
                                _last_submitted_sl[inst_id] = new_sl
                                logger.info(
                                    f"[TS] {inst_id} 已有生效保护单 {_eid} "
                                    f"(TP={_existing_prot.get('tpTriggerPx')} "
                                    f"SL={_existing_prot.get('slTriggerPx')})，"
                                    f"登记后跳过本轮补挂"
                                )
                                continue
                            # 修复: 已有 SL-only 保护单且持有信号 TP →
                            # 登记旧单，置 _backfill_tp 走下方补挂路径，用
                            # 信号 TP 重挂 OCO 完整保护（FVG 成交后 TP 方向
                            # 相对成交价必然正确）。
                            trailing_algo_ids[inst_id] = _eid
                            _last_submitted_sl[inst_id] = new_sl
                            _backfill_tp = float(_sig_tp)
                            logger.info(
                                f"[TS] {inst_id} 已有保护单 {_eid} 缺 TP "
                                f"(TP={_existing_prot.get('tpTriggerPx')})，"
                                f"用信号 TP={_backfill_tp} 补挂完整保护"
                            )
                    # 修复: 已激活的移动止损正常更新；新持仓（无已挂止损记录）也补挂
                    # 初始止损 — 覆盖限价主单成交后止损单未挂上的缺口（execute_signal
                    # 独立挂单在限价单未成交时挂不上，成交后由这里补上）
                    if _backfill_tp is not None or ts._activated or (
                            inst_id not in trailing_algo_ids
                            and inst_id not in _last_submitted_sl):
                        # 修复: 按 tickSz 精度对齐止损价，防止 OKX API 拒单
                        if inst_id not in _tick_sz_cache:
                            _info = client.get_instrument_info(inst_id)
                            if _info is None:
                                logger.warning(f"Cannot get instrument info for {inst_id}, skipping trailing stop update")
                                continue
                            _tick_sz_cache[inst_id] = float(_info.get("tickSz", "0.1"))
                            _lot_sz_cache[inst_id] = float(_info.get("lotSz", "1"))
                        _tick = _tick_sz_cache[inst_id]
                        _lot = _lot_sz_cache.get(inst_id, 1.0)
                        new_sl_rounded = _round_to_tick(new_sl, _tick)

                        # 修复: 防抖阈值 — 止损价变化 < 2x tickSz 时跳过更新
                        # 避免几分钟内疯狂生成成百上千次撤单和重挂单请求
                        # 补挂 TP 场景 (_backfill_tp) 不受防抖限制 — 必须立即
                        # 用信号 TP 挂上完整保护。
                        _prev_sl = _last_submitted_sl.get(inst_id)
                        if (_backfill_tp is None and _prev_sl is not None
                                and abs(new_sl_rounded - _prev_sl) < _tick * 2):
                            logger.debug(f"[TS] {inst_id} 止损价变化 {abs(new_sl_rounded - _prev_sl):.4f} "
                                         f"< 防抖阈值 {_tick*2:.4f}，跳过更新")
                            continue

                        _sl_price_str = _format_price_for_exchange(new_sl_rounded, _tick)

                        # 修复: sz 按 lotSz 步长对齐，防止 OKX API 拒单
                        # 浮点数直接格式化 (如 f"{1.23456:g}") 会输出不符合合约张数步长的值
                        _sz_raw = pos["size"]
                        _sz_rounded = math.floor(_sz_raw / _lot) * _lot if _lot > 0 else _sz_raw
                        try:
                            _lot_decimals = max(0, -int(decimal.Decimal(str(_lot)).as_tuple().exponent))
                        except Exception:
                            _lot_decimals = 0
                        _sz_str = f"{_sz_rounded:.{_lot_decimals}f}"

                        # 修复: 使用限价止损并带 ±0.5% 滑点保护，避免市价止损在插针时产生过大滑点
                        if pos["pos_side"] == "long":
                            _sl_limit_px = new_sl_rounded * (1 - 0.005)
                        else:
                            _sl_limit_px = new_sl_rounded * (1 + 0.005)
                        _sl_limit_px_str = _format_price_for_exchange(_sl_limit_px, _tick)

                        logger.debug(f"[TS] {inst_id} trailing stop: {new_sl:.2f} → "
                                     f"tickSz={_tick} → {_sl_price_str}, "
                                     f"sz={_sz_raw} → lotSz={_lot} → {_sz_str}, "
                                     f"limit_px={_sl_limit_px_str}")

                        # 先提交新止损单，成功后再撤销旧单（防损丢失）
                        # 修复: 保留原 TP — TP 来源优先级:
                        #   1) _backfill_tp: 成交后补挂的信号 TP（FVG 目标，
                        #      相对成交价方向必然正确）
                        #   2) 旧保护单上的 TP
                        #   3) active_signals.signal_tp (新仓首挂保护单的 TP 来源)
                        #   4) tp_fallback 兜底
                        # 避免撤旧单时把 TP 一并撤掉，导致移动止损后持仓变成
                        # "只亏不赚"的裸 SL 单。
                        _tp_keep = None
                        if _backfill_tp is not None:
                            _tp_keep = _format_price_for_exchange(_backfill_tp, _tick)
                        else:
                            _old_algo_id = trailing_algo_ids.get(inst_id)
                            if _old_algo_id:
                                _old_detail = client.get_algo_order_details(_old_algo_id)
                                if _old_detail:
                                    _tp_keep = _old_detail.get("tpTriggerPx") or None
                        if not _tp_keep:
                            # 修复: 新仓首挂保护单必须优先用信号 TP。原逻辑首挂
                            # 恒用 tp_fallback (10x 下 ≈11.5% 距离过远), 信号 TP
                            # (≈4.7%) 从不落单 → 实盘止盈形同虚设, 只亏不赚。
                            try:
                                _sig_tp_px = state_manager.state.active_signals.get(
                                    inst_id, {}).get("signal_tp")
                                if _sig_tp_px:
                                    _tp_keep = _format_price_for_exchange(
                                        float(_sig_tp_px), _tick)
                            except (TypeError, ValueError):
                                _tp_keep = None
                        if not _tp_keep:
                            _tp_keep = _format_price_for_exchange(tp_fallback, _tick)
                        try:
                            new_ord = client.place_algo_order(
                                inst_id=inst_id,
                                td_mode=risk_cfg["margin_mode"],
                                side="sell" if pos["pos_side"] == "long" else "buy",
                                pos_side=pos["pos_side"],
                                sz=_sz_str,
                                ord_type="conditional",
                                tp_trigger_px=_tp_keep,
                                tp_trigger_px_type="last",
                                sl_trigger_px=_sl_price_str,
                                sl_trigger_px_type="last",
                                sl_ord_px=_sl_limit_px_str,
                            )
                            if new_ord:
                                # 新止损生效后撤销旧单（使用记录的 algo_id）
                                old_algo_id = trailing_algo_ids.get(inst_id)
                                if old_algo_id:
                                    try:
                                        client.cancel_algo_order(
                                            algo_id=old_algo_id, inst_id=inst_id
                                        )
                                    except (ConnectionError, TimeoutError, OSError) as cancel_err:
                                        logger.debug(f"[TS] {inst_id} 撤销旧止损单 {old_algo_id} 失败: {cancel_err}")
                                # 保存新 algo_id 供下次更新时撤销
                                trailing_algo_ids[inst_id] = new_ord
                                _last_submitted_sl[inst_id] = new_sl_rounded
                                # 修复: 持久化高水位线，防止重启失忆
                                # 修复: TrailingStop 用 _best_price，不是 _highest_price
                                state_manager.state.trailing_stop_state[inst_id] = ts._best_price
                                state_manager.save()
                                logger.debug(f"[TS] {inst_id} 追踪止损已更新至: {_sl_price_str} "
                                             f"(TP={_tp_keep} algo_id={new_ord})")
                            else:
                                logger.warning(f"[TS] {inst_id} 新止损提交失败，保留旧止损不变")
                        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
                            logger.error(f"[TS] {inst_id} 追踪止损更新异常: {e}")

        # ---- 步骤 2: 提现检查 ----
        # 纸面模拟模式下跳过: 虚拟余额提现提醒是纯噪音
        wd_pct = risk_cfg["profit_withdrawal_pct"]
        if paper_engine is None and state_manager.check_withdrawal(equity, wd_pct):
            withdrawal_amount = equity * wd_pct / 100.0
            logger.info(
                f"\n{'#'*60}\n"
                f"  !!! 提现提醒 !!!\n"
                f"  钱包达到提现阈值！当前权益: {equity:.2f} USDT\n"
                f"  建议提现: {withdrawal_amount:.2f} USDT ({wd_pct}%)\n"
                f"  剩余继续: {equity - withdrawal_amount:.2f} USDT\n"
                f"{'#'*60}"
            )
            state_manager.record_withdrawal(equity, wd_pct)

        # ---- 步骤 3: 自适应参数调整 ----
        if adaptive_tuner and edge_analyzer.trades:
            edge_stats = edge_analyzer.analyze(opt_cfg.get("edge_lookback", 100),
                                               equity_baseline=equity)
            adaptive_tuner.adapt(edge_stats, current_equity=equity)

            if adaptive_tuner.trading_paused:
                logger.warning("自适应调参器暂停交易，等待恢复条件")
                state_manager.save()
                if once:
                    break
                if tracker:
                    tracker.resume()
                time.sleep(scan_interval)
                continue

            eff_lev, eff_risk, eff_min_score = adaptive_tuner.get_effective_params()
            eff_min_conf = max(0.0, min(1.0, eff_min_score))
            # 修复: 低流动性时段的阈值提升不应被自适应调参覆盖
            if _low_liquidity:
                eff_min_conf = min(0.95, eff_min_conf * 1.2)
            _active_thresholds["min_confidence"] = eff_min_conf
            logger.debug(f"Adaptive params: leverage={eff_lev}x, "
                         f"risk={eff_risk:.1f}%, min_score={eff_min_score:.2f}")

        # ---- 步骤 4: 风控门禁 ----
        passed, reason = risk_gate(client, state_manager, equity, config, active_count=active_count)
        if not passed:
            logger.info(f"Risk gate blocked: {reason}")
            state_manager.save()
            if once:
                break
            if tracker:
                tracker.resume()
            time.sleep(scan_interval)
            continue

        # ---- 步骤 5: 扫描标的 (优先缓存) ----
        all_signals: List[Signal] = []
        best_analysis: Optional[MasterAnalysis] = None
        best_coin = None
        best_debate_result: Optional[Any] = None
        best_regime = "NEUTRAL"
        best_funding_rate: Optional[float] = None
        best_candles_1h: List[Candle] = []
        best_candles_4h: List[Candle] = []  # HTF 方向门用 (2026-08-07)
        # 修复: 按币种记录信号配对的完整分析数据，杜绝 best_signal 与
        # best_analysis/best_coin 等跨币种错配
        signal_analysis_map: Dict[str, Dict[str, Any]] = {}

        # 优先从后台缓存读取预研究结果
        cache_hit = False
        if cache:
            # 诊断：打印缓存统计
            stats = cache.stats()
            logger.debug(f"[Cache] 统计: total={stats['total']} fresh={stats['fresh']} "
                         f"with_signals={stats['with_signals']} with_analysis={stats['with_analysis']}")
            # 诊断：仅在 DEBUG 级别打印缓存条目详情，避免每轮遍历 200+ 条目
            if logger.isEnabledFor(logging.DEBUG):
                for entry in cache.get_all_entries():
                    if entry.analysis:
                        a = entry.analysis
                        logger.debug(f"[Cache] {entry.inst_id}: conf={a.final_confidence:.0%} "
                                     f"agree={a.channel_agreement:.0%} "
                                     f"fresh={entry.is_fresh(120)}")
            cached_signals = cache.get_fresh_signals(
                min_confidence=_active_thresholds["min_confidence"],
                # 修复 Bug 24: 缓存预筛选必须使用挡位阈值本身（不再放宽 0.4 倍），
                # 否则低质量信号绕过保守挡位过滤直接进入下单
                min_agreement=_active_thresholds["min_agreement"],
            )
            if cached_signals:
                # 价格偏离检查：丢弃偏离超过 0.5% 的缓存条目（FVG 时效性强）
                filtered = []
                for entry, coin_info in cached_signals:
                    current_price = coin_info.get("last", 0)
                    cached_price = entry.current_price if entry.current_price else current_price
                    if current_price > 0 and cached_price > 0:
                        deviation = abs(current_price - cached_price) / cached_price
                        if deviation > config.get("cache_price_deviation_threshold", 0.005):
                            logger.debug(f"[Cache] {entry.inst_id} 价格偏离 {deviation:.2%}，丢弃")
                            continue
                    # 修复 Bug 25: 缓存中的 last 字段若为 0（API 异常/数据延迟），丢弃
                    if current_price <= 0:
                        logger.debug(f"[Cache] {entry.inst_id} last 价格异常 ({current_price})，丢弃")
                        continue
                    filtered.append((entry, coin_info))
                logger.info(f"[Cache] 命中 {len(filtered)} 个币种（偏离 <0.5%, last>0）"
                            f" | positions: {active_count}/{risk_cfg['max_positions']}")

                # 修复: 缓存池过小不短路实时扫描 — 置信度修复后缓存池诚实变窄
                # (单通道信号 conf≈0.26)，若只要 ≥1 条缓存就跳过 100 币实时扫描，
                # 直接扫描路径会被饿死 (实测候选池 64→4, ROUND 2 起几乎无候选)。
                # 缓存条目仍并入候选，同时回退实时扫描补充，确保每轮都有充分候选。
                _min_cache_pool = int(config.get("agent", {}).get("min_cache_pool_for_hit", 8))
                cache_hit = len(filtered) >= _min_cache_pool
                if not cache_hit:
                    logger.warning(
                        f"[Cache] 缓存池仅 {len(filtered)} < {_min_cache_pool}，"
                        f"回退实时扫描补充候选")

                for entry, coin_info in filtered:
                    inst_id = entry.inst_id
                    if inst_id in positions:
                        continue

                    # 修复 Bug 23: 缓存路径也必须应用挡位 threshold 二次过滤，
                    # 防止后台线程在阈值放宽时记录了低于当前挡位的信号
                    if entry.analysis:
                        _min_factor = _active_thresholds.get("min_factor_score", 40) / 100.0
                        _max_signal_score = max((s.score for s in entry.signals), default=0.0) if entry.signals else 0.0
                        if (entry.analysis.final_confidence < _active_thresholds["min_confidence"] or
                                entry.analysis.channel_agreement < _active_thresholds["min_agreement"] or
                                _max_signal_score < _min_factor):
                            logger.debug(
                                f"[Cache] {inst_id} 二次过滤未达挡位阈值: "
                                f"conf={entry.analysis.final_confidence:.0%} "
                                f"agree={entry.analysis.channel_agreement:.0%} "
                                f"factor={_max_signal_score:.2f}"
                            )
                            continue

                    all_signals.extend(entry.signals)
                    signal_analysis_map[inst_id] = {
                        "analysis": entry.analysis,
                        "coin": coin_info,
                        "regime": entry.detected_regime,
                        "funding_rate": entry.funding_rate,
                        "candles_1h": entry.candles_by_tf.get("1H", []),
                        "candles_4h": entry.candles_by_tf.get("4H", []),
                        "debate_result": None,
                    }

                    if entry.analysis and (best_analysis is None or
                            entry.analysis.final_confidence > best_analysis.final_confidence):
                        best_analysis = entry.analysis
                        best_coin = coin_info
                        best_debate_result = None  # cached path 不保留辩论结果
                        best_regime = entry.detected_regime
                        best_funding_rate = entry.funding_rate
                        best_candles_1h = entry.candles_by_tf.get("1H", [])

                    logger.debug(f"  [Cache] {inst_id}: "
                                 f"score={entry.analysis.final_score:+.2f} "
                                 f"conf={entry.analysis.final_confidence:.0%} "
                                 f"regime={entry.detected_regime}")

        # 缓存未命中，回退到 WebSocket 实时缓存扫描
        if not cache_hit:
            # 使用 WebSocket 缓存替代 REST get_tickers() 轮询
            ws_coins = ws_cache.get_top_by_volume(100)
            # 修复: WS 缓存为空 (订阅失败/断连) 时回退 REST get_tickers，
            # 避免主循环扫描 0 个币种静默空转 (曾空转 90 分钟无信号)
            if not ws_coins:
                try:
                    _raw_ticks = client.get_tickers(inst_type="SWAP")
                    if _raw_ticks:
                        # REST 返回字符串字段，需转 float 保持与 WS 缓存同构，
                        # 否则后续 vol24h 过滤 (str < float) 会抛 TypeError
                        def _f(v, d=0.0):
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                return d
                        ws_coins = [{
                            "instId": t.get("instId", ""),
                            "last": _f(t.get("last")),
                            "bidPx": _f(t.get("bidPx")),
                            "askPx": _f(t.get("askPx")),
                            "high24h": _f(t.get("high24h")),
                            "low24h": _f(t.get("low24h")),
                            "vol24h": _f(t.get("vol24h")),
                            "ts": int(t.get("ts") or 0),
                        } for t in _raw_ticks]
                        logger.warning(
                            "[Scan] WS 缓存为空，回退 REST get_tickers "
                            f"({len(ws_coins)} 个 ticker)"
                        )
                except Exception as _ws_e:
                    logger.warning(f"[Scan] REST get_tickers 回退失败: {_ws_e}")
                    ws_coins = []
            # 过滤: 仅 USDT 本位永续合约 + 成交量阈值
            coins = []
            min_vol = config["strategy"]["min_volume_24h_usd"]
            coin_limit = config["agent"].get("coin_scan_limit", 100)
            for c in ws_coins:
                if not c.get("instId", "").endswith("-USDT-SWAP"):
                    continue
                if c.get("vol24h", 0) < min_vol:
                    continue
                coins.append(c)
            coins = coins[:coin_limit]
            logger.info(f"[Scan] WebSocket 缓存扫描 {len(coins)} 个币种 "
                        f"(positions: {active_count}/{risk_cfg['max_positions']})")

            for coin in coins:
                if coin["instId"] in positions:
                    continue

                try:
                    inst_id = coin["instId"]
                    current_price = coin["last"]

                    # 使用 scan_round 完成 K 线获取 + FVG 扫描
                    scan_result = scan_round(client, coin, scan_config, ws_cache=ws_cache)
                    if not scan_result.signals:
                        continue

                    signals = scan_result.signals
                    candles_by_tf = scan_result.candles_by_tf
                    candles_1h = candles_by_tf.get("1H", [])
                    candles_4h = candles_by_tf.get("4H", [])
                    funding_rate = scan_result.funding_rate
                    spread = scan_result.spread

                    # ---- 多通道分析 ----
                    if mc_cfg.get("enabled", True) and expert_engine:
                        if _is_budget_tight():
                            logger.debug("Budget tight, skipping multi-channel analysis")
                            # 修复 H-2: 预算紧张时裸信号也需经过阈值过滤，
                            # 否则未经多通道/体制/辩论分析的裸信号可能直接进入下单
                            _min_factor = _active_thresholds.get("min_factor_score", 40) / 100.0
                            _passed = [s for s in signals
                                       if (s.score or 0.0) >= _min_factor]
                            if _passed:
                                all_signals.extend(_passed)
                                signal_analysis_map[inst_id] = {
                                    "analysis": None,
                                    "coin": coin,
                                    "regime": "NEUTRAL",
                                    "funding_rate": funding_rate,
                                    "candles_1h": candles_1h,
                                    "debate_result": None,
                                }
                                logger.debug(f"[Budget] {inst_id}: {len(_passed)}/{len(signals)} "
                                             f"signals passed factor filter (≥{_min_factor:.2f})")
                            continue
                        analysis = full_multi_channel_analysis(
                            client=client,
                            inst_id=inst_id,
                            current_price=current_price,
                            candles_1h=candles_1h,
                            candles_4h=candles_4h,
                            fvg_signals=signals,
                            config=scan_config,
                            engine=expert_engine,
                        )

                        # ---- 相关性体制检测 (Vibe-Trading CausalHysteresisRegime) ----
                        detected_regime = "NEUTRAL"
                        if regime_detector and len(candles_1h) >= 20 and len(candles_4h) >= 20:
                            ret_1h = [math.log(candles_1h[i].close / candles_1h[i-1].close)
                                      for i in range(1, len(candles_1h))
                                      if candles_1h[i].close > 0 and candles_1h[i-1].close > 0]
                            ret_4h = [math.log(candles_4h[i].close / candles_4h[i-1].close)
                                      for i in range(1, len(candles_4h))
                                      if candles_4h[i].close > 0 and candles_4h[i-1].close > 0]
                            # 计算趋势方向
                            trend_1h_sign = float(np.sign(np.mean(ret_1h))) if ret_1h else 0.0
                            trend_4h_sign = float(np.sign(np.mean(ret_4h))) if ret_4h else 0.0
                            # 计算 4 阶滞后自相关 — 衡量 1H 动量持续性
                            # (1H 与 4H 收益率跨时间尺度直接相关无数学意义，
                            #  改用 corr(r_t, r_{t-4}) 检测动量/反转体制)
                            corr = 0.0
                            if len(ret_1h) > 5:
                                try:
                                    ret_arr = np.array(ret_1h)
                                    corr = float(np.corrcoef(ret_arr[4:], ret_arr[:-4])[0, 1])
                                    if math.isnan(corr):
                                        corr = 0.0
                                except (ValueError, TypeError, IndexError, RuntimeError):
                                    corr = 0.0
                            regime_state = regime_detector_map.setdefault(
                                inst_id,
                                CausalHysteresisRegime(
                                    hysteresis_threshold=az_cfg.get(
                                        "hysteresis_threshold", 0.15),
                                    min_regime_duration=az_cfg.get(
                                        "min_regime_duration", 5),
                                    enter_threshold=az_cfg.get(
                                        "regime_enter_threshold", 0.65),
                                    exit_threshold=az_cfg.get(
                                        "regime_exit_threshold", 0.45),
                                    corr_window=az_cfg.get(
                                        "regime_corr_window", 60),
                                    smooth_window=az_cfg.get(
                                        "regime_smooth_window", 5),
                                    symbol=inst_id,
                                ),
                            ).update(
                                correlation=corr,
                                trend_1h_sign=trend_1h_sign,
                                trend_4h_sign=trend_4h_sign,
                            )
                            detected_regime = regime_state.value
                            logger.debug(f"[Regime] {inst_id}: {detected_regime} "
                                         f"(duration={regime_detector_map[inst_id].state.regime_duration})")

                        # ---- 多空辩论 (TradingAgents 86k⭐) ----
                        debate_result = None  # 默认值，防止 UnboundLocalError
                        if debate_cfg.get("enabled", True) and debate_engine:
                            if _is_budget_tight():
                                logger.debug("Budget tight, skipping debate engine")
                            else:
                                # 将 ChannelReport 传给辩论引擎进行完整辩论（带超时保护）
                                debate_result = debate_engine.conduct_debate_with_timeout(
                                    symbol=inst_id,
                                    channel_reports=analysis.channels,
                                    fvg_signals=signals,
                                    current_price=current_price,
                                    regime=detected_regime,
                                    timeout_sec=debate_cfg.get("debate_timeout", 30.0),
                                )
                                logger.debug(f"[Debate] {inst_id}: {debate_result.winner} "
                                             f"score={debate_result.final_score:+.2f} "
                                             f"conf={debate_result.confidence:.0%} "
                                             f"verdict={debate_result.action_recommendation[:50]}")

                                # 辩论结果影响分析置信度 — 一致增强、矛盾惩罚、平局折损
                                if debate_result.winner == "tie":
                                    analysis.final_confidence *= 0.9  # 温和折扣，减少震荡市过度淘汰
                                elif debate_result.winner == "bullish" and analysis.final_score > 0:
                                    analysis.final_confidence *= 1.1
                                elif debate_result.winner == "bearish" and analysis.final_score < 0:
                                    analysis.final_confidence *= 1.1
                                elif debate_result.winner in ("bullish", "bearish"):
                                    # 辩论方向与分析分数矛盾 — 降低置信度
                                    analysis.final_confidence *= 0.85
                                analysis.final_confidence = max(0.0, min(0.95, analysis.final_confidence))

                        # 检查是否满足最低置信度 + 因子评分
                        _min_factor = _active_thresholds.get("min_factor_score", 40) / 100.0
                        _max_signal_score = max((s.score for s in signals), default=0.0)
                        below_threshold = (
                            analysis.final_confidence < _active_thresholds["min_confidence"] or
                            analysis.channel_agreement < _active_thresholds["min_agreement"] or
                            _max_signal_score < _min_factor
                        )

                        if below_threshold:
                            logger.debug(f"[Filter] {inst_id} "
                                         f"confidence={analysis.final_confidence:.0%} "
                                         f"agreement={analysis.channel_agreement:.0%} "
                                         f"factor={_max_signal_score:.2f} — filtered "
                                         f"(threshold: conf≥{_active_thresholds['min_confidence']:.0%}, "
                                         f"agree≥{_active_thresholds['min_agreement']:.0%}, "
                                         f"factor≥{_min_factor:.2f})")
                            continue

                        # 记录通过阈值的最佳分析
                        if best_analysis is None or analysis.final_confidence > best_analysis.final_confidence:
                            best_analysis = analysis
                            best_coin = coin
                            best_debate_result = debate_result
                            best_regime = detected_regime
                            best_funding_rate = funding_rate
                            best_candles_1h = candles_1h
                            best_candles_4h = candles_by_tf.get("4H", [])

                        logger.debug(f"  {inst_id}: {len(signals)} FVG signals, "
                                     f"master={analysis.final_score:+.2f} "
                                     f"conf={analysis.final_confidence:.0%}")

                        all_signals.extend(signals)
                        signal_analysis_map[inst_id] = {
                            "analysis": analysis,
                            "coin": coin,
                            "regime": detected_regime,
                            "funding_rate": funding_rate,
                            "candles_1h": candles_1h,
                            "candles_4h": candles_by_tf.get("4H", []),
                            "debate_result": debate_result,
                        }

                    else:
                        # When expert_engine is None, create a minimal fallback
                        all_signals.extend(signals)
                        if best_coin is None:
                            best_coin = coin
                            best_analysis = None  # will be handled downstream

                except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, KeyError) as e:
                    logger.error(f"Error scanning {coin.get('instId', '?')}: {e}")

        # 修复 H-1: 恢复原始 FVG 配置，防止低流动性乘数指数膨胀
        if isinstance(_orig_fvg_cfg, dict):
            scan_config["strategy"]["min_fvg_width_pct"] = _orig_fvg_cfg

        # ---- 打印最佳分析报告 ----
        if best_analysis:
            report = format_analysis_report(best_analysis)
            logger.info(report)

        # ---- 步骤 6: 信号排序 & 执行 ----
        all_signals.sort(key=lambda s: s.score or 0.0, reverse=True)

        # ---- 汇流确认 (任务3, 多因素汇流评分过滤) ----
        # 排序后、执行前对所有信号做汇流确认: 低于 min_score 的信号被过滤。
        # 每个信号记录 confluence_score / confluence_details / entry_quality。
        # 异常时放行（不阻塞主循环），保持与 ML 过滤一致的最小侵入策略。
        if confluence_checker is not None and all_signals:
            _con_threshold = float(con_cfg.get("min_score", 0.5))
            _kept_signals = []
            for _sig in all_signals:
                _sig_map = signal_analysis_map.get(_sig.inst_id, {})
                try:
                    _cr = confluence_checker.check(
                        from_legacy_fvg(_sig.fvg),
                        _sig_map.get("candles_1h", []) or [],
                        _sig_map.get("candles_4h", []) or [],
                        {
                            "current_price": _sig.entry_price,
                            "funding_rate": _sig_map.get("funding_rate"),
                            "spread_pct": getattr(_sig, "spread_pct", 0.0),
                        },
                    )
                except Exception as _ce:
                    logger.warning(
                        f"[Confluence] {_sig.inst_id} 汇流检查异常，放行: {_ce}")
                    _kept_signals.append(_sig)
                    continue
                _sig.confluence_score = float(_cr.get("confluence_score", 0.0))
                _sig.confluence_details = _cr
                _sig.entry_quality = str(_cr.get("entry_quality", "poor"))
                # 门控审计上下文: 每个信号的完整判定指标，供排查胜率低时对照
                # 哪些信号、被哪条规则、以什么理由拦截/放行。
                _sig_dir = str(getattr(_sig, "position_side", "?"))
                _sig_score = float(getattr(_sig, "score", 0.0) or 0.0)
                _sig_width = float(getattr(_sig.fvg, "width_pct", 0.0) or 0.0)
                _sig_tf = str(getattr(_sig.fvg, "timeframe", "1H") or "1H")
                _sig_ml = float(getattr(_sig, "ml_score", 0.0) or 0.0)
                _audit_ctx = (
                    f"[GateAudit] {_sig.inst_id} {_sig_dir} {_sig_tf} "
                    f"score={_sig_score:.2f} width={_sig_width:.2f}% "
                    f"confl={_sig.confluence_score:.2f} qual={_sig.entry_quality} "
                    f"ml={_sig_ml:.3f}"
                )
                _strat_cfg = config.get("strategy", {}) if isinstance(config, dict) else {}
                _min_width_1h = float(_strat_cfg.get("min_fvg_width_pct", {}).get("1H", 1.5))
                _min_width_4h = float(_strat_cfg.get("min_fvg_width_pct", {}).get("4H", 3.0))
                _sig_min_width = _min_width_4h if _sig_tf == "4H" else _min_width_1h
                if _sig_width < _sig_min_width:
                    logger.info(
                        f"{_audit_ctx} → [WidthGate] 拦截: FVG 宽度 {_sig_width:.2f}% "
                        f"< {_sig_tf} 下限 {_sig_min_width:.2f}%，窄缺口方向性弱"
                    )
                    continue
                # FVG 前置硬门(2026-08-07): 孤立 FVG(无 sweep 也无 MSS 确认)不交易。
                # 独立于 confluence_reject_poor 的硬门——即使质量拦截被关闭，
                # 没有任何汇流确认的裸 FVG 仍被拒（研究: 无汇流 FVG=抛硬币）。
                _hard_gate = bool(
                    con_cfg.get("require_sweep_or_structure", True))
                if _hard_gate:
                    _dt = _cr.get("details") or {}
                    _sweep_met = bool(
                        (_dt.get("liquidity_sweep") or {}).get("met"))
                    _mss_met = bool(
                        (_dt.get("structure_break") or {}).get("met"))
                    if not (_sweep_met or _mss_met):
                        logger.info(
                            f"{_audit_ctx} → [FVG硬门] 拦截: 孤立 FVG "
                            f"(sweep={_sweep_met} mss={_mss_met}，均未确认)，不交易"
                        )
                        continue
                # 教训固化(2026-08-07, GALA 复盘): quality=poor 拦截 —
                # GALA 汇流 0.58(poor) 仅凭分数过 0.50 阈值进入执行，核心证据链
                # 不足(无 bias_alignment 支撑)。汇流分数可能被次要条件拉高，
                # 但 poor 质量说明方向性证据不足，直接拦截(默认开启, 可配置)。
                _reject_poor = bool(
                    _strat_cfg.get("confluence_reject_poor", True))
                if _reject_poor and _sig.entry_quality == "poor":
                    logger.info(
                        f"{_audit_ctx} → [QualityGate] 拦截: 汇流质量 "
                        f"{_sig.entry_quality}(得分 {_sig.confluence_score:.2f}) "
                        f"核心证据不足，条件: {_cr.get('conditions_met')}"
                    )
                    continue
                if _sig.confluence_score < _con_threshold:
                    logger.info(
                        f"{_audit_ctx} → [Confluence] 拦截: 汇流得分 "
                        f"{_sig.confluence_score:.2f} < {_con_threshold:.2f} "
                        f"(quality={_sig.entry_quality})"
                    )
                    continue
                logger.info(
                    f"{_audit_ctx} → [Confluence] 放行: 汇流得分 "
                    f"{_sig.confluence_score:.2f} ≥ {_con_threshold:.2f} "
                    f"条件: {_cr.get('conditions_met')} quality={_sig.entry_quality}"
                )
                _kept_signals.append(_sig)
            if len(_kept_signals) < len(all_signals):
                logger.info(
                    f"[Confluence] 汇流过滤: {len(all_signals)} → "
                    f"{len(_kept_signals)} 信号")
            # 统计: 每轮门控后剩余信号，便于对比最终执行信号与全部候选
            logger.info(
                f"[GateAudit] 本轮候选 {len(all_signals)} → 汇流/宽度/质量门控后 "
                f"{len(_kept_signals)} 信号"
            )
            all_signals = _kept_signals

        if not all_signals:
            logger.info("No signals passed confluence filter this round")
            state_manager.save()
            print_summary(state_manager.state, equity)
            if once:
                break
            if tracker:
                tracker.resume()
            elapsed = time.time() - round_start
            time.sleep(max(0, scan_interval - elapsed))
            continue

        # 修复: 顺势加仓 (Pyramiding) — 不再粗暴地 all_signals=[] 扼杀同币种加仓机会
        # 当最强信号指向已有持仓时，如果满足加仓条件（浮盈 + 高置信度），
        # 使用减半仓位追加，利用浮盈扩大战果
        _pyramiding = False
        _pyramiding_signal = None
        _reverse = False  # 同品种反手（平多开空 / 平空开多）
        _reverse_target = None

        if _skip_new_position:
            all_signals = []

        if active_count > 0 and all_signals:
            best_signal = all_signals[0]
            current_inst_id = None
            current_pos = None

            # 修复: 多持仓架构 — 只有当 active_count >= max_positions 时才需要换仓
            # 如果 active_count < max_positions，仍有仓位空间，应该直接开新仓
            _need_switch = active_count >= risk_cfg["max_positions"]

            # 修复 H3: 多持仓时选择 worst upl 持仓替换，而非随机取最后一个
            _worst_upl = float("inf")
            for inst_id, pos in positions.items():
                if pos["size"] > 0:
                    if best_signal.inst_id == inst_id:
                        # 已有该币种持仓
                        _upl = pos.get("upl", 0)
                        _upl_pct = pos.get("upl_ratio_pct", 0)

                        # 修复: 同品种反手检测 — 信号方向与当前持仓相反
                        if best_signal.position_side != pos["pos_side"]:
                            # 趋势反转信号：平掉现有仓位，反向开仓
                            _reverse = True
                            _reverse_target = best_signal
                            current_inst_id = inst_id
                            current_pos = pos
                            logger.warning(
                                f"[Reverse] {inst_id} 检测到趋势反转信号！"
                                f"当前: {pos['pos_side']}, 信号: {best_signal.position_side}, "
                                f"将平仓后反向开仓"
                            )
                            break

                        # 浮盈 + 信号置信度高于加仓阈值 → 金字塔加仓
                        # 修复 C-2: 金字塔加仓应使用多通道置信度（而非 FVG score）
                        # FVG score 衡量价格缺口质量，confidence 衡量多通道一致性
                        # 两者量纲不同，混用会导致加仓条件失真
                        _pyramid_min_conf = _active_thresholds.get("min_confidence", 0.40) * 1.2
                        _signal_confidence = (
                            best_analysis.final_confidence
                            if best_analysis and best_analysis.final_confidence > 0
                            else (best_signal.score or 0.0)
                        )
                        if _upl > 0 and _upl_pct is not None and _upl_pct > 0:
                            if _signal_confidence >= _pyramid_min_conf:
                                _pyramiding = True
                                _pyramiding_signal = best_signal
                                logger.info(
                                    f"[Pyramiding] {inst_id} 浮盈 {_upl_pct:.2%}，"
                                    f"信号置信度 {_signal_confidence:.2f} ≥ {_pyramid_min_conf:.2f}，"
                                    f"触发金字塔加仓"
                                )
                        current_inst_id = inst_id
                        current_pos = pos
                        break
                    _upl = pos.get("upl", 0)
                    if _upl < _worst_upl:
                        _worst_upl = _upl
                        current_inst_id = inst_id
                        current_pos = pos

            # ---- 同品种反手 ----
            if _reverse and _reverse_target:
                if paper_engine is not None:
                    paper_engine.close_position(current_inst_id, reason="reverse")
                close_ord, _is_limit = client.close_position_safe(
                    inst_id=current_inst_id,
                    pos_side=current_pos["pos_side"],
                    pos_size=current_pos["size"],
                    mgn_mode=risk_cfg["margin_mode"],
                )
                if close_ord:
                    _pending_close = {
                        "inst_id": current_inst_id,
                        "ord_id": close_ord,
                        "pos_side": current_pos["pos_side"],
                        "avg_px": current_pos["avg_px"],
                        "size": current_pos["size"],
                        "mark_px": current_pos["mark_px"],
                        "upl": current_pos.get("upl", 0),
                        "upl_ratio_pct": current_pos.get("upl_ratio_pct", 0),
                        "c_time": current_pos.get("c_time", time.time()),
                        "leverage": int(current_pos.get("leverage", 1)),
                        "timestamp": time.time(),
                        "signal_id": current_pos.get("signal_id", ""),
                        "funding_rate": current_pos.get("funding_rate", best_funding_rate),
                        "best_signal": _reverse_target,
                        "best_analysis": best_analysis,
                        "best_coin": best_coin,
                        "best_regime": best_regime,
                        "best_funding_rate": best_funding_rate,
                        "candles_1h": best_candles_1h,
                        "candles_4h": best_candles_4h,
                    }
                    state_manager.state._pending_close = _pending_close
                    state_manager.save()
                    all_signals = []
                    logger.info(f"[Reverse] 平仓已提交 {current_inst_id} (ord={close_ord})，"
                                f"等待下一轮确认后反向开仓 {_reverse_target.position_side}")
                if tracker:
                    tracker.resume()
                elapsed = time.time() - round_start
                time.sleep(max(0, scan_interval - elapsed))
                continue

            # ---- 金字塔加仓 ----
            if _pyramiding and _pyramiding_signal:
                if _pyramiding_signal.inst_id not in positions:
                    _pyramiding_signal = None  # 持仓已变化，放弃加仓
                else:
                    # 修复 P1-6: 加仓前检查聚合敞口上限（fail-closed） —
                    # 此前多轮 0.5× 加仓无聚合限制，名义敞口可数倍于上限。
                    # 持仓查询失败或敞口已达上限时一律放弃加仓。
                    _pyramid_cap_ok = _exposure_cap_allows_add(
                        client, equity, config)
                    if not _pyramid_cap_ok:
                        logger.warning(
                            f"[Pyramiding] {_pyramiding_signal.inst_id} "
                            f"聚合敞口已达上限(或持仓查询失败)，放弃加仓"
                        )
                    if _pyramid_cap_ok:
                        _info = client.get_instrument_info(_pyramiding_signal.inst_id)
                        if _info is None:
                            logger.warning(
                                f"[Pyramiding] {_pyramiding_signal.inst_id} "
                                f"无法获取合约信息，放弃加仓"
                            )
                            _pyramid_cap_ok = False
                    if _pyramid_cap_ok:
                        # 减半风险比例用于加仓
                        _pyramid_config = copy.deepcopy(config)
                        _pyramid_config["risk"]["risk_per_trade_pct"] = \
                            _pyramid_config["risk"].get("risk_per_trade_pct", 1.0) * 0.5
                        _pyramid_ord = _execute_signal_with_quant_enhancements(
                            client=client,
                            signal=_pyramiding_signal,
                            equity=equity,
                            config=_pyramid_config,
                            instrument_info=_info,
                            state_manager=state_manager,
                            risk_committee=risk_committee,
                            market_guard=market_guard,
                            market_state=market_state,
                            signal_tracker=signal_tracker,
                            analysis=best_analysis,
                            debate_result=best_debate_result,
                            candles_1h=best_candles_1h,
                            funding_rate=best_funding_rate,
                            regime=best_regime,
                            paper_engine=paper_engine,
                            candles_htf=best_candles_4h,
                        )
                        if _pyramid_ord:
                            logger.info(
                                f"[Pyramiding] {_pyramiding_signal.inst_id} "
                                f"加仓完成 (ord={_pyramid_ord})"
                            )
                        else:
                            logger.warning(
                                f"[Pyramiding] {_pyramiding_signal.inst_id} "
                                f"加仓被拦截或下单失败"
                            )
                    # 刷新 positions
                    positions = _refresh_positions()
                    active_count = len([p for p in positions.values() if p["size"] > 0])
                    state_manager.save()
                    # 修复: 加仓后清空信号，避免 fall-through 对本轮信号重复开仓
                    all_signals = []

            # ---- 仍有仓位空间，直接开新仓（无需换仓） ----
            elif not _need_switch and best_signal.inst_id not in positions:
                # active_count < max_positions，有空间容纳新仓位
                # 直接开仓，无需关闭现有仓位
                # 修复: 开仓前价格校验 — 信号基于缓存，实时价脱节时直接跳过，
                # 避免"开单→撤单"空转消耗手续费。
                _fresh_ok, _live_px = check_signal_price_fresh(
                    client, best_signal,
                    max_dev_pct=config["strategy"].get("max_signal_price_deviation_pct", 3.0),
                )
                if not _fresh_ok:
                    all_signals = []
                    state_manager.save()
                    if tracker:
                        tracker.resume()
                    elapsed = time.time() - round_start
                    time.sleep(max(0, scan_interval - elapsed))
                    continue
                logger.info(
                    f"[Multi] 仓位空间可用 ({active_count}/{risk_cfg['max_positions']})，"
                    f"直接开新仓: {best_signal.inst_id}"
                )
                _info = client.get_instrument_info(best_signal.inst_id)
                if _info is None:
                    logger.warning(f"Cannot get instrument info for {best_signal.inst_id}, "
                                   f"跳过开新仓")
                else:
                    _new_ord = _execute_signal_with_quant_enhancements(
                        client=client,
                        signal=best_signal,
                        equity=equity,
                        config=config,
                        instrument_info=_info,
                        state_manager=state_manager,
                        risk_committee=risk_committee,
                        market_guard=market_guard,
                        market_state=market_state,
                        signal_tracker=signal_tracker,
                        analysis=best_analysis,
                        debate_result=best_debate_result,
                        candles_1h=best_candles_1h,
                        funding_rate=best_funding_rate,
                        regime=best_regime,
                        paper_engine=paper_engine,
                        candles_htf=best_candles_4h,
                    )
                    if _new_ord:
                        logger.info(f"[Multi] 新仓已开: {best_signal.inst_id} (ord={_new_ord})")
                    else:
                        logger.warning(f"[Multi] 新仓被拦截或下单失败: {best_signal.inst_id}")
                positions = _refresh_positions()
                active_count = len([p for p in positions.values() if p["size"] > 0])
                state_manager.save()
                # 修复: 已开新仓，清空信号避免 fall-through 重复开仓
                all_signals = []

            # ---- 仓位已满，需要换仓 ----
            elif _need_switch and current_inst_id and best_signal.inst_id != current_inst_id:
                # 修复: 换仓前价格校验 — 异常波动币信号基于缓存，实时价已脱节
                # 时先平仓会白吃手续费（开单→撤单空转）。价格偏离超阈值直接跳过。
                _fresh_ok, _live_px = check_signal_price_fresh(
                    client, best_signal,
                    max_dev_pct=config["strategy"].get("max_signal_price_deviation_pct", 3.0),
                )
                if not _fresh_ok:
                    all_signals = []
                    state_manager.save()
                    if tracker:
                        tracker.resume()
                    elapsed = time.time() - round_start
                    time.sleep(max(0, scan_interval - elapsed))
                    continue

                # 统一换仓守卫: 最小持仓时长 + 评分门槛(含往返成本) +
                # 资金费双守卫 + 相关性。杜绝换仓绞肉机（实测 TRIA 持仓
                # 5 分钟就被换掉, 11 笔平仓 100% 是 signal_switch 白吃手续费）。
                _cur_score = 0.0
                try:
                    _cur_score = float(
                        state_manager.state.active_signals.get(
                            current_inst_id, {}).get("master_score", 0.0) or 0.0
                    )
                except (TypeError, ValueError):
                    _cur_score = 0.0
                _new_score = (
                    best_analysis.final_score
                    if best_analysis and best_analysis.final_score
                    else (best_signal.score or 0.0)
                )
                _sw_ok, _sw_reason = _switch_guards(
                    config=config,
                    client=client,
                    cache=cache,
                    cur_inst_id=current_inst_id,
                    cur_score=_cur_score,
                    cur_c_time=float(current_pos.get("c_time", 0) or 0),
                    new_inst_id=best_signal.inst_id,
                    new_score=_new_score,
                    new_side=best_signal.position_side,
                    new_funding_rate=best_funding_rate,
                )
                if not _sw_ok:
                    logger.info(_sw_reason)
                    all_signals = []
                    state_manager.save()
                    if tracker:
                        tracker.resume()
                    elapsed = time.time() - round_start
                    time.sleep(max(0, scan_interval - elapsed))
                    continue

                logger.info(f"持有 {current_inst_id}，检测到新信号 {best_signal.inst_id} "
                            f"(score={best_signal.score:.2f})，先平仓")

                current_pos = positions[current_inst_id]
                if paper_engine is not None:
                    paper_engine.close_position(current_inst_id, reason="switch")
                close_ord, _is_limit = client.close_position_safe(
                    inst_id=current_inst_id,
                    pos_side=current_pos["pos_side"],
                    pos_size=current_pos["size"],
                    mgn_mode=risk_cfg["margin_mode"],
                )
                if not close_ord:
                    logger.error(f"平仓失败 {current_inst_id}，跳过本轮开新仓")
                    state_manager.save()
                    if tracker:
                        tracker.resume()
                    elapsed = time.time() - round_start
                    time.sleep(max(0, scan_interval - elapsed))
                    continue

                # 修复: 非阻塞平仓确认 — 不阻塞主循环
                # 记录 pending_close 状态，下一轮确认后再开新仓
                # 这样主循环的追踪止损、风控门禁在此期间仍然有效
                _pending_close = {
                    "inst_id": current_inst_id,
                    "ord_id": close_ord,
                    "pos_side": current_pos["pos_side"],
                    "avg_px": current_pos["avg_px"],
                    "size": current_pos["size"],
                    "mark_px": current_pos["mark_px"],
                    "upl": current_pos.get("upl", 0),
                    "upl_ratio_pct": current_pos.get("upl_ratio_pct", 0),
                    "c_time": current_pos.get("c_time", time.time()),
                    "leverage": int(current_pos.get("leverage", 1)),
                    "timestamp": time.time(),
                    "signal_id": current_pos.get("signal_id", ""),
                    "funding_rate": current_pos.get("funding_rate", best_funding_rate),
                    "best_signal": best_signal,
                    "best_analysis": best_analysis,
                    "best_coin": best_coin,
                    "best_regime": best_regime,
                    "best_funding_rate": best_funding_rate,
                    "candles_1h": best_candles_1h,
                    "candles_4h": best_candles_4h,
                }
                # 记录到 state_manager 供跨轮次访问
                state_manager.state._pending_close = _pending_close
                state_manager.save()
                all_signals = []
                logger.info(f"平仓已提交 {current_inst_id} (ord={close_ord})，"
                            f"等待下一轮确认后开新仓 {best_signal.inst_id}")

                # 本轮不再继续，等待下一轮确认
                if tracker:
                    tracker.resume()
                elapsed = time.time() - round_start
                time.sleep(max(0, scan_interval - elapsed))
                continue
            else:
                # 修复: 兜底 — best_signal 属于已持仓币种且不满足反手/加仓
                # 条件时禁止重复开仓，清空信号跳过本轮
                logger.info(
                    f"[Multi] {best_signal.inst_id} 已持仓且不满足反手/加仓条件，"
                    f"跳过本轮开仓"
                )
                all_signals = []

        if _skip_new_position:
            all_signals = []

        if not all_signals:
            logger.info("No new signals this round")
            state_manager.save()
            print_summary(state_manager.state, equity)

            # Edge 统计
            if edge_analyzer.trades:
                stats = edge_analyzer.analyze(opt_cfg.get("edge_lookback", 100),
                                              equity_baseline=equity)
                logger.info(f"[Edge] 胜率={stats.win_rate:.1%} "
                            f"盈亏比={stats.profit_factor:.2f} "
                            f"期望值={stats.expectancy:+.2f} "
                            f"连亏={stats.consecutive_losses}")

            if once:
                break
            if tracker:
                tracker.resume()
            elapsed = time.time() - round_start
            time.sleep(max(0, scan_interval - elapsed))
            continue

        best_signal = all_signals[0]
        # 成单率漏斗: 本轮通过门控的信号数
        _FILL_FUNNEL["signals"] += len(all_signals)
        # 修复: 配对 best_signal 与同币种的分析数据，杜绝跨币种错配
        # （best_analysis/best_coin 按 final_confidence 选择，可能与按 score
        #  排序第一的 best_signal 属于不同币种，导致风控/记录/特征使用错数据）
        _paired = signal_analysis_map.get(best_signal.inst_id)
        if _paired and _paired.get("analysis") is not None:
            best_analysis = _paired["analysis"]
            best_coin = _paired["coin"]
            best_regime = _paired["regime"]
            best_funding_rate = _paired["funding_rate"]
            best_candles_1h = _paired["candles_1h"]
            best_candles_4h = _paired.get("candles_4h", []) or []
            best_debate_result = _paired["debate_result"]
        # 修复 Bug 25: best_coin.last 异常保护（API 返回 0 时不能用于下单和偏离检测）
        _mkt_price = best_coin.get("last", 0) if best_coin else 0
        if _mkt_price <= 0:
            _mkt_price = best_signal.entry_price
        # entry_price 兜底：信号对象自身的价格也必须 > 0
        if _mkt_price <= 0 or best_signal.entry_price <= 0:
            logger.warning(
                f"[Guard] {best_signal.inst_id} 市场价/入场价异常 "
                f"(mkt={_mkt_price}, entry={best_signal.entry_price})，跳过本轮"
            )
            state_manager.save()
            if once:
                break
            if tracker:
                tracker.resume()
            elapsed = time.time() - round_start
            time.sleep(max(0, scan_interval - elapsed))
            continue
        _cancelled, _should_skip = manage_pending_orders(client, _mkt_price, signal=best_signal)
        if _should_skip:
            logger.info(f"Pending order exists for {best_signal.inst_id}, skip new signal")
            state_manager.save()
            if once:
                break
            if tracker:
                tracker.resume()
            elapsed = time.time() - round_start
            time.sleep(max(0, scan_interval - elapsed))
            continue

        # ---- 自适应评分过滤 ----
        if adaptive_tuner:
            _, _, min_score = adaptive_tuner.get_effective_params()
            if (best_signal.score or 0.0) < min_score:
                logger.info(f"Signal score {best_signal.score:.2f} < adaptive min {min_score:.2f}, skip")
                state_manager.save()
                if once:
                    break
                if tracker:
                    tracker.resume()
                elapsed = time.time() - round_start
                time.sleep(max(0, scan_interval - elapsed))
                continue

        # ---- FreqAI 在线预测过滤 (freqtrade 52k⭐) ----
        if freqai and best_analysis:
            if _is_budget_tight():
                logger.debug("Budget tight, skipping FreqAI prediction")
            else:
                # 构建特征向量
                features = {
                    "fvg_score": best_signal.score,
                    "master_score": best_analysis.final_confidence,
                    "leverage": float(best_signal.leverage),
                    "is_long": 1.0 if best_signal.position_side == "long" else 0.0,
                }
                predicted_pnl = freqai.predict(features)
                # 修复 C1: 使用 _active_thresholds 而非 agg_thresholds
                if predicted_pnl < _active_thresholds["min_prediction_confidence"]:
                    logger.info(f"[FreqAI] Predicted PnL {predicted_pnl:+.4f} below threshold, skip")
                    state_manager.save()
                    if once:
                        break
                    if tracker:
                        tracker.resume()
                    elapsed = time.time() - round_start
                    time.sleep(max(0, scan_interval - elapsed))
                    continue
                logger.debug(f"[FreqAI] {best_signal.inst_id} predicted_pnl={predicted_pnl:+.4f}")

        # ---- FVG ML 二次评分过滤 (第二阶段 ML 增强, 可插拔) ----
        # 在风控门禁/FreqAI 之后、执行之前: ML 模型对 FVG 信号二次评分,
        # 低于动态阈值直接跳过。模型缺失/异常时放行（不阻塞主循环）。
        if ml_ranker is not None:
            try:
                # 按信号时间框架匹配 K 线（4H 信号用 4H K 线，特征窗口才一致）
                _ml_tf = getattr(best_signal.fvg, "timeframe", "1H") or "1H"
                _ml_candles = (_paired or {}).get(
                    "candles_4h" if _ml_tf == "4H" else "candles_1h", [])
                if not _ml_candles:
                    _ml_candles = best_candles_1h or []
                # from_legacy_fvg 适配 + 注入 BTC K 线 → corr_with_btc 特征
                # （获取失败返回 None，corr_with_btc 回落中性 0，不阻塞）
                _ml_detected = from_legacy_fvg(best_signal.fvg)
                _ml_detected._btc_candles = _btc_candles_cached(client, _ml_tf)
                _ml_feats = _ml_detector.compute_features(_ml_detected, _ml_candles)
                _ml_score = ml_ranker.predict(_ml_feats)
                _ml_threshold = float(_ml_cfg.get("min_ml_score", 0.6))
                # 教训固化(2026-08-07, GALA 复盘): ML 过线余量校验 —
                # GALA ML=0.504 仅高于阈值 0.50 的 0.004 即被放行，卡线放行
                # 等同于未过滤。要求 ML ≥ min_ml_score + ml_score_margin。
                _ml_margin = float(_ml_cfg.get("ml_score_margin", 0.05))
                _ml_effective_threshold = _ml_threshold + _ml_margin
                # ML 门控审计上下文: 与汇流门控同格式，便于横向对比同一信号
                # 在各级门控的判定结果，定位"胜率低"是哪个环节放行了劣质信号。
                _ml_ctx = (
                    f"[GateAudit] {best_signal.inst_id} "
                    f"{getattr(best_signal, 'position_side', '?')} "
                    f"{_ml_tf} score={getattr(best_signal, 'score', 0.0) or 0.0:.2f} "
                    f"width={getattr(best_signal.fvg, 'width_pct', 0.0) or 0.0:.2f}% "
                    f"confl={best_signal.confluence_score:.2f} "
                    f"qual={getattr(best_signal, 'entry_quality', '?')} "
                    f"ml={_ml_score:.3f}"
                )
                if _ml_score < _ml_effective_threshold:
                    logger.info(
                        f"{_ml_ctx} → [MLGate] 拦截: ML分数 {_ml_score:.3f} "
                        f"< {_ml_effective_threshold:.2f} (阈值{_ml_threshold:.2f}"
                        f"+余量{_ml_margin:.2f})，ML二次评分否决，跳过本轮"
                    )
                    state_manager.save()
                    if once:
                        break
                    if tracker:
                        tracker.resume()
                    elapsed = time.time() - round_start
                    time.sleep(max(0, scan_interval - elapsed))
                    continue
                best_signal.ml_score = _ml_score  # 记录供后续分析
                logger.info(
                    f"{_ml_ctx} → [MLGate] 放行: ML分数 {_ml_score:.3f} "
                    f"≥ {_ml_effective_threshold:.2f}，执行前门控全部通过"
                )
            except Exception as _ml_e:
                logger.warning(f"[ML] {best_signal.inst_id} ML过滤异常，放行: {_ml_e}")

        # 修复 2026-08-08: 纸面模式源头去重 — 该币种已有纸面挂单/持仓时,
        # 跳过本轮执行。dry-run 下 get_pending_orders 恒为空, manage_pending
        # _orders 去重失效, 同一信号每轮重复走 execute_signal 假单路径 →
        # positions_opened 虚增 + 重复假单噪音; 纸面引擎内部虽有去重, 但
        # 在入口拦截才能保住记账与日志干净。
        if paper_engine is not None:
            try:
                if paper_engine.has_position(best_signal.inst_id):
                    logger.info(
                        f"[Paper] {best_signal.inst_id} 已有纸面挂单/持仓，"
                        f"跳过重复执行")
                    state_manager.save()
                    if once:
                        break
                    if tracker:
                        tracker.resume()
                    elapsed = time.time() - round_start
                    time.sleep(max(0, scan_interval - elapsed))
                    continue
            except Exception:
                pass

        inst_info = client.get_instrument_info(best_signal.inst_id)
        ord_id = None  # 修复: 显式初始化，避免 inst_info 为 None 时下方 if ord_id 抛 UnboundLocalError
        if inst_info is None:
            logger.warning(f"Cannot get instrument info for {best_signal.inst_id}")
        else:
            # 使用自适应杠杆和风险比例
            _orig_leverage = best_signal.leverage
            effective_leverage = best_signal.leverage  # default
            _exec_config = config
            try:
                eff_risk = float(risk_cfg["risk_per_trade_pct"])
                if adaptive_tuner:
                    eff_lev, _eff_risk, _ = adaptive_tuner.get_effective_params()
                    # 临时应用自适应杠杆；finally 恢复，避免共享对象长期污染
                    effective_leverage = min(eff_lev, best_signal.leverage)
                    best_signal.leverage = effective_leverage
                    eff_risk = _eff_risk

                # ---- 滚动 Kelly 风险上限 (v3.3 / PRO: 探索→利用) ----
                # eff_risk = min(自适应风险, 滚动分数Kelly)。开源核心版
                # 自动跳过；样本不足(返回 None)时保持原值不引入噪声。
                if _PRO is not None:
                    try:
                        with state_manager.lock():
                            _rk_pnl = list(state_manager.state.recent_pnl or [])
                        _kelly_cap, _kdiag = _PRO.rolling_kelly_risk_pct(
                            _rk_pnl, float(risk_cfg["risk_per_trade_pct"]), risk_cfg)
                    except Exception as _rk_e:
                        _kelly_cap, _kdiag = None, {"error": str(_rk_e)}
                        logger.debug(f"[RollingKelly] 计算失败(忽略): {_rk_e}")
                    if _kelly_cap is not None and _kelly_cap < eff_risk:
                        _ewma = _kdiag.get("ewma_lambda") or 0
                        logger.info(
                            f"[RollingKelly] 风险上限 {eff_risk:.1f}% → "
                            f"{_kelly_cap:.1f}% (f*={_kdiag.get('kelly_f')} "
                            f"样本={_kdiag.get('samples')} 胜率={_kdiag.get('win_rate')} "
                            f"档位={_kdiag.get('tier')}"
                            + (f" ewmaλ={_ewma}" if _ewma else "") + ")")
                        eff_risk = _kelly_cap

                # 使用配置副本应用风险比例，避免修改共享 config
                if eff_risk != float(risk_cfg["risk_per_trade_pct"]):
                    _exec_config = copy.deepcopy(config)
                    _exec_config["risk"]["risk_per_trade_pct"] = eff_risk

                ord_id = _execute_signal_with_quant_enhancements(
                    client=client,
                    signal=best_signal,
                    equity=equity,
                    config=_exec_config,
                    instrument_info=inst_info,
                    state_manager=state_manager,
                    risk_committee=risk_committee,
                    market_guard=market_guard,
                    market_state=market_state,
                    signal_tracker=signal_tracker,
                    analysis=best_analysis,
                    debate_result=best_debate_result,
                    candles_1h=best_candles_1h,
                    funding_rate=best_funding_rate,
                    regime=best_regime,
                    paper_engine=paper_engine,
                    candles_htf=best_candles_4h,
                )
            finally:
                # 恢复原始杠杆
                best_signal.leverage = _orig_leverage
            if ord_id:
                state_manager.state.positions_opened += 1
                trade_count_since_reflection += 1
                logger.info(f"Signal executed: {best_signal.inst_id} "
                            f"ordId={ord_id} score={best_signal.score:.2f} "
                            f"leverage={effective_leverage}x")

                # 记录入场决策
                if memory and best_analysis:
                    memory.log_decision(DecisionLog(
                        timestamp=time.time(),
                        symbol=best_signal.inst_id,
                        direction=best_signal.position_side,
                        entry_price=best_signal.entry_price,
                        master_score=best_analysis.final_confidence,
                        fvg_score=best_signal.score,
                        fvg_timeframe=best_signal.fvg.timeframe,
                        channel_scores={
                            ch.channel_name: ch.net_score
                            for ch in best_analysis.channels
                        },
                        regime=best_regime,
                        red_flags=best_analysis.key_risks,
                        expert_verdict=(best_analysis.expert_verdict or "")[:200],
                    ))

        # ---- 步骤 7: 定期反思（后台异步执行，不阻塞主循环） ----
        # 修复: generate_reflection 可能调用 LLM API (10-30s 阻塞)，
        # 必须剥离到独立线程，避免主循环追踪止损和风控在此期间完全失效
        if memory and trade_count_since_reflection >= mem_cfg.get("reflection_interval_trades", 10):
            recent_decisions = memory.get_recent_decisions(20)
            logger.info("\n📝 生成交易反思报告（后台线程）...")
            _reflection_thread = threading.Thread(
                target=_run_reflection_safe,
                args=(memory, recent_decisions),
                daemon=True,
                name="reflection-worker",
            )
            _reflection_thread.start()
            trade_count_since_reflection = 0
            # 修复 2026-08-07: 记忆生命周期接线 — decay_all/prune_archive
            # 原从未被调用, 记忆衰减/归档功能形同虚设。随反思周期同步维护。
            try:
                memory.maintain_lifecycle()
            except Exception as _ml:
                logger.warning(f"[Memory] 生命周期维护失败: {_ml}")

        # ---- 步骤 8: Edge 统计 ----
        if edge_analyzer.trades:
            stats = edge_analyzer.analyze(opt_cfg.get("edge_lookback", 100),
                                          equity_baseline=equity)
            logger.info(f"[Edge] 胜率={stats.win_rate:.1%} "
                        f"盈亏比={stats.profit_factor:.2f} "
                        f"期望值={stats.expectancy:+.2f} "
                        f"连亏={stats.consecutive_losses}")

        # ---- 步骤 9: Kelly 仓位分析 (freqtrade 52k⭐) ----
        # 修复 P3-9: Kelly 公式对样本量极度敏感，< 50 笔交易统计噪声远大于信号
        # 业界标准: 至少 50 笔同向交易才用 Kelly；不足时用固定比例
        if hyperopt_enabled and edge_analyzer.trades and len(edge_analyzer.trades) >= 50:
            kelly = compute_kelly(edge_analyzer.trades)
            logger.info(f"[Kelly] f*={kelly.kelly_fraction:.4f} "
                        f"1/2K={kelly.half_kelly:.4f} "
                        f"推荐风险={kelly.recommended_risk_pct:.2f}% "
                        f"期望增长={kelly.expected_growth_rate:+.4f}")
        elif hyperopt_enabled and edge_analyzer.trades:
            logger.debug(f"[Kelly] 样本不足 ({len(edge_analyzer.trades)} < 50)，使用固定风险比例")

        # ---- 步骤 10: 定期 Hyperopt 优化 (后台异步，不阻塞主循环) ----
        # 修复: Hyperopt 贝叶斯优化可能耗时数分钟，不能在主线程同步运行
        # 否则主循环追踪止损、风控、紧急平仓全部失效，等同于无人驾驶汽车在高速上关掉传感器
        if (hyperopt_enabled and edge_analyzer.trades
                and len(edge_analyzer.trades) >= 20
                and round_count % hyperopt_interval == 0):
            logger.info("\n🔧 启动 Hyperopt 参数优化（后台线程）...")
            _opt_trades = list(edge_analyzer.trades)  # 快照拷贝，防止并发修改
            _opt_equity = state_manager.state.initial_equity or equity
            _opt_cfg = copy.deepcopy(config)  # 修复 2026-08-07: 传 config, hyperopt 参数生效
            _opt_thread = threading.Thread(
                target=_run_hyperopt_safe,
                args=(_opt_trades, _opt_equity, _opt_cfg),
                daemon=True,
                name="hyperopt-worker",
            )
            _opt_thread.start()

        # ---- 步骤 11: FreqAI 在线更新 (防数据中毒) ----
        # 修复: 数据中毒防护 — 仅在新增交易时才更新 FreqAI
        # 原逻辑每轮都喂入相同数据，导致模型严重过拟合
        if freqai and edge_analyzer.trades:
            _current_trade_count = len(edge_analyzer.trades)
            _last_update_count = freqai._last_update_trade_count
            _last_update_time = freqai._last_update_time
            _min_retrain_interval = fa_cfg.get("retrain_interval", 10)  # 最少间隔轮数
            _new_trades_since_update = _current_trade_count - _last_update_count

            if _new_trades_since_update > 0 and (round_count - _last_update_time) >= _min_retrain_interval:
                recent_trades = edge_analyzer.trades[-fa_cfg.get("feature_window", 50):]
                freqai.update(recent_trades)
                freqai._last_update_trade_count = _current_trade_count
                freqai._last_update_time = round_count
                logger.debug(f"[FreqAI] 模型在线更新完成 (新增 {_new_trades_since_update} 笔交易)")
            else:
                logger.debug(f"[FreqAI] 跳过更新: 无新交易或未到更新间隔 "
                             f"(新增={_new_trades_since_update}, "
                             f"距上次={round_count - _last_update_time:.0f}轮)")

        # ---- 步骤 12: 因子选择（后台异步，降低过拟合） ----
        _fs_interval = config.get("factor_selector", {}).get("reselect_interval_rounds", 100)
        if (factor_selector and factor_zoo_adapter and cache and quant_db
                and round_count - _last_factor_select_round >= _fs_interval):
            logger.info("\n🔬 启动因子选择（后台线程）...")
            _fs_thread = threading.Thread(
                target=_run_factor_selection_safe,
                args=(factor_selector, factor_zoo_adapter, cache, quant_db),
                daemon=True,
                name="factor-selector-worker",
            )
            _fs_thread.start()
            _last_factor_select_round = round_count

        # ---- 步骤 13: Walk Forward 验证（后台异步，检测过拟合） ----
        _wf_interval = config.get("walk_forward", {}).get("interval_rounds", hyperopt_interval * 2)
        if (walk_forward and backtest_runner
                and round_count - _last_walk_forward_round >= _wf_interval):
            logger.info("\n📈 启动 Walk Forward 验证（后台线程）...")
            _wf_thread = threading.Thread(
                target=_run_walk_forward_safe,
                args=(walk_forward, backtest_runner, client, config),
                daemon=True,
                name="walk-forward-worker",
            )
            _wf_thread.start()
            _last_walk_forward_round = round_count

        # ---- 步骤 14: 每日量化报告 ----
        if quant_report:
            _report_hour = config.get("quant_report", {}).get("generate_hour_utc", 0)
            _today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if (datetime.now(timezone.utc).hour == _report_hour
                    and _today_utc != _last_quant_report_date):
                logger.info("\n📊 生成每日量化报告（后台线程）...")
                _qr_thread = threading.Thread(
                    target=_generate_daily_report_safe,
                    args=(quant_report,),
                    kwargs={"equity_curve": None},
                    daemon=True,
                    name="quant-report-worker",
                )
                _qr_thread.start()
                _last_quant_report_date = _today_utc

        # ---- 保存状态 ----
        # 修复 L-4: 每 100 轮清理一次缓存的 tickSz/lotSz/sl 记录，
        # 防止长期运行后缓存无限增长（虽然实际受限于交易币种数）
        if round_count % 100 == 0:
            _active_positions = set(positions.keys())
            _total_stale = 0
            for _cache_dict in (_tick_sz_cache, _lot_sz_cache, _last_submitted_sl):
                _stale = [k for k in _cache_dict if k not in _active_positions]
                for _k in _stale:
                    _cache_dict.pop(_k, None)
                _total_stale += len(_stale)
            if _total_stale > 0:
                logger.debug(f"[CacheCleanup] 清理 {_total_stale} 条过期缓存记录")

        state_manager.save()
        print_summary(state_manager.state, equity)

        if once:
            break

        # 恢复后台追踪
        if tracker:
            tracker.resume()

        elapsed = time.time() - round_start
        sleep_time = max(0, scan_interval - elapsed)
        logger.info(f"Round completed in {elapsed:.1f}s, "
                    f"sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)

    ws_cache.stop()
    logger.info("Agent stopped.")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FVG KILLER（公允价值缺口杀手）— 基于 Fair Value Gap 的 OKX 合约交易 Agent"
    )
    parser.add_argument(
        "-c", "--config", "--配置文件",
        dest="config",
        default=os.path.join(os.path.dirname(__file__), "config.json"),
        help="配置文件路径，默认: config.json"
    )
    parser.add_argument(
        "-d", "--dry-run", "--演练",
        dest="dry_run",
        action="store_true",
        help="演练模式，只分析不下单"
    )
    parser.add_argument(
        "--paper", "--模拟建仓",
        dest="paper",
        action="store_true",
        help="纸面交易模式（模拟建仓）：dry_run + 虚拟余额 + 实时行情模拟成交/止盈止损/盈亏"
    )
    parser.add_argument(
        "-o", "--once", "--单轮",
        dest="once",
        action="store_true",
        help="只跑一轮后退出"
    )
    parser.add_argument(
        "--log-level", "--日志级别",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "-r", "--rounds", "--轮次",
        dest="rounds",
        type=int, default=0,
        help="最大运行轮次，0 表示无限制"
    )
    parser.add_argument(
        "--scan-interval", "--扫描间隔",
        dest="scan_interval",
        type=int, default=None,
        help="扫描间隔秒数，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--coin-limit", "--币种上限",
        dest="coin_limit",
        type=int, default=None,
        help="扫描币种数量上限，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "-a", "--aggressiveness", "--挡位",
        dest="aggressiveness",
        type=int, choices=[1, 2, 3],
        default=None,
        help="研判挡位: 1=激进(每天必找一币), 2=均衡(2-3天一笔), 3=保守(默认)"
    )
    parser.add_argument(
        "--auto",
        dest="auto_confirm",
        action="store_true",
        help="自动确认所有操作（无人值守模式）"
    )
    parser.add_argument(
        "--analyze-trades", "--分析交易",
        dest="analyze_trades",
        action="store_true",
        help="启动时先扫描所有历史交易并进行超详细分析"
    )
    parser.add_argument(
        "--analyze-days", "--分析天数",
        dest="analyze_days",
        type=int, default=90,
        help="历史交易分析的回溯天数，默认 90"
    )
    parser.add_argument(
        "--analyze-max", "--分析最大笔数",
        dest="analyze_max_trades",
        type=int, default=100,
        help="历史交易分析的最大笔数，默认 100"
    )

    args = parser.parse_args()

    # 加载配置
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # 命令行参数覆盖
    if args.dry_run:
        config["agent"]["dry_run"] = True
    if getattr(args, "paper", False):
        # 纸面交易 = 演练模式（绝不下真实单）+ 虚拟余额引擎
        config["agent"]["dry_run"] = True
        config.setdefault("paper", {})["enabled"] = True
    if args.log_level:
        config["agent"]["log_level"] = args.log_level
    if args.scan_interval is not None:
        config["agent"]["scan_interval_seconds"] = args.scan_interval
    if args.coin_limit is not None:
        config["agent"]["coin_scan_limit"] = args.coin_limit
    if args.aggressiveness is not None:
        config["agent"]["aggressiveness"] = args.aggressiveness
    if args.auto_confirm:
        config["agent"]["auto_confirm"] = True

    # 配置日志
    setup_logging(config)

    logger = logging.getLogger("main")

    # 安全检查
    if not config["agent"]["dry_run"]:
        if config["okx"]["api_key"] in ("YOUR_API_KEY", ""):
            logger.error("请先在 config.json 中设置 OKX API 密钥！")
            sys.exit(1)
        logger.warning("=" * 60)
        logger.warning("  ⚠ 实盘模式 — 将产生真实交易订单！")
        logger.warning("  按 Ctrl+C 可在 5 秒内取消...")
        logger.warning("=" * 60)
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("用户取消运行。")
            sys.exit(0)

    # 运行
    # 修复: 主循环崩溃保护 — 不再因单次网络异常直接 sys.exit(1)
    # 使用指数退避重试策略，确保 7x24 无人值守运行
    _restart_count = 0
    _max_restarts = 10
    _base_cooldown = 30  # 基础冷却时间（秒）
    while True:
        try:
            # ---- 历史交易分析（可选） ----
            if args.analyze_trades and _restart_count == 0:
                logger.info("=" * 60)
                logger.info("  启动历史交易超详细分析...")
                logger.info("=" * 60)
                try:
                    # 先初始化客户端用于分析
                    analyze_client = OKXClient(config)
                    run_trade_analysis(
                        client=analyze_client,
                        config=config,
                        days_back=args.analyze_days,
                        max_trades=args.analyze_max_trades,
                        upload=True,
                    )
                except (ConnectionError, TimeoutError, OSError, ValueError, TypeError) as e:
                    logger.error(f"历史交易分析失败: {e}")
                logger.info("=" * 60)
                logger.info("  历史交易分析完成，继续启动主循环...")
                logger.info("=" * 60)

            main_loop(config, once=args.once, max_rounds=args.rounds)
            break  # 正常退出
        except KeyboardInterrupt:
            logger.info("用户中断。正在保存状态...")
            break
        except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, OKXQueryError) as e:
            _restart_count += 1
            if _restart_count > _max_restarts:
                logger.critical(
                    f"主循环连续崩溃 {_max_restarts} 次，放弃重试。"
                    f"请检查日志并手动排查。"
                )
                break
            # Cleanup before restart — prevent resource leaks
            logger.error(f"Main loop crashed: {e}")
            # 修复 P0-D: 崩溃-重启路径进程未退出、atexit 不触发，
            # 必须先落盘（否则最后一轮内存变更最长丢失一个扫描周期）
            _sm = _cleanup_registry.get("state_manager")
            if _sm is not None:
                try:
                    _sm.save()
                except Exception as _se:
                    logger.error(f"崩溃前保存状态失败: {_se}")
            _ws = _cleanup_registry.get("ws_cache")
            if _ws:
                try:
                    _ws.stop()
                except Exception:
                    pass
            _trk = _cleanup_registry.get("tracker")
            if _trk:
                try:
                    _trk.stop()
                except Exception:
                    pass
            # 指数退避: 30s, 60s, 120s, 240s, ...
            _cooldown = _base_cooldown * (2 ** (_restart_count - 1))
            _cooldown = min(_cooldown, 600)  # 最多等 10 分钟
            logger.exception(
                f"主循环崩溃 (第 {_restart_count}/{_max_restarts} 次)，"
                f"{_cooldown}s 后自动重启..."
            )
            time.sleep(_cooldown)


if __name__ == "__main__":
    main()