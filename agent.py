"""
OKX FVG 交易 Agent — 主循环 (v3.1 全模块融合版)。

融合 GitHub Top 3 开源项目精华：
  - freqtrade (52k ⭐): Hyperopt 参数优化 + Edge 分析 + Trailing Stop + FreqAI + Kelly
  - TradingAgents (86k ⭐): 多 Agent 辩论引擎 + 分析师信誉 + 决策反思 + 跨品种经验迁移
  - Vibe-Trading (23.6k ⭐): Alpha Zoo 因子库 + 因果滞后体制检测 + 记忆生命周期

运行流程:
  1. 扫描 Top N 合约标的
  2. 五通道数据采集 + 超级交易专家分析
  3. 多空辩论引擎 (TradingAgents) + 因果滞后体制检测 (Vibe-Trading)
  4. Alpha 因子分析 (Vibe-Trading) + FreqAI 在线预测 (freqtrade)
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
import json
import logging
import math
import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np

from okx_client import OKXClient
from strategy import (
    Candle, FVG, Signal,
    candles_from_raw,
    scan_fvg_all_timeframes,
)
from executor import (
    AgentState, StateManager,
    execute_signal,
    monitor_positions,
    manage_pending_orders,
    get_tradable_coins,
    calculate_spread,
    print_summary,
)
from multi_channel import (
    ChannelReport, MasterAnalysis,
    MasterTraderEngine,
    full_multi_channel_analysis,
    format_analysis_report,
    analyze_price_action,
    analyze_market_structure,
    analyze_capital_flow,
    analyze_market_sentiment,
    analyze_macro_context,
)
from optimization import (
    TradeRecord, EdgeStats, EdgeAnalyzer,
    AdaptiveParameterTuner, TrailingStop,
    PortfolioRisk, assess_portfolio_risk,
)
from memory import (
    DecisionLog, MemoryManager,
)
from debate_engine import (
    TradingAgentsDebateEngine,
    run_enhanced_debate,
    SimpleDebateResult,
    inject_past_reflections,
    format_debate_report,
)
from hyperopt import (
    BayesianHyperopt,
    ParamSpace,
    compute_kelly,
    FreqAIPipeline,
    generate_performance_dashboard,
    run_full_optimization,
)
from alpha_zoo import (
    AlphaZoo,
    CausalHysteresisRegime,
    EnhancedMemoryLifecycle,
    FactorAnalyzer,
    MarketRegime,
)
from coin_tracker import (
    CoinResearchCache, CoinTracker, CoinResearchEntry,
    warmup_research,
)
from report import (
    SessionReporter, generate_and_send_report,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

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

    # 控制台
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # 文件
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
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
            "force_trade": True,            # 无信号时强制选最优币种
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
            "force_trade": False,
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
            "force_trade": False,
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
    import copy
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

    return cfg


# ---------------------------------------------------------------------------
# 单轮扫描结果
# ---------------------------------------------------------------------------

from dataclasses import dataclass

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
) -> ScanResult:
    """对单个合约执行一轮完整扫描。

    Returns:
        ScanResult: 信号列表 + K 线数据 + 资金费率 + 价差，供后续多通道分析复用
    """
    inst_id = coin["instId"]
    current_price = coin["last"]
    bid_px = coin.get("bidPx", 0)
    ask_px = coin.get("askPx", 0)

    logger.debug(f"Scanning {inst_id} @ {current_price}")

    # ---- 获取 K 线 ----
    candles_by_tf: Dict[str, List[Candle]] = {}
    for tf in config["strategy"]["timeframes"]:
        # 需要足够的历史数据用于异常检测 (lookback=50) + FVG 检测 (3 蜡烛)
        raw = client.get_candles(inst_id, bar=tf, limit=200)
        if raw:
            candles_by_tf[tf] = candles_from_raw(raw)
        else:
            logger.warning(f"No candle data for {inst_id} {tf}")

    if not candles_by_tf:
        return ScanResult(signals=[], candles_by_tf={}, funding_rate=None, spread=0)

    # ---- 获取资金费率 ----
    funding_rate = client.get_funding_rate(inst_id)

    # ---- 计算价差 ----
    spread = calculate_spread(bid_px, ask_px)

    # ---- 扫描信号 ----
    signals = scan_fvg_all_timeframes(
        inst_id=inst_id,
        candles_by_tf=candles_by_tf,
        current_price=current_price,
        config=config,
        funding_rate=funding_rate,
        spread_pct=spread,
    )

    return ScanResult(
        signals=signals,
        candles_by_tf=candles_by_tf,
        funding_rate=funding_rate,
        spread=spread,
    )


# ---------------------------------------------------------------------------
# 风控门禁
# ---------------------------------------------------------------------------

def risk_gate(
    client: OKXClient,
    state_manager: StateManager,
    equity: float,
    config: dict,
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

    # 持仓数检查
    active_count = len(state.active_signals)
    if active_count >= risk_cfg["max_positions"]:
        return False, f"Max positions reached ({active_count}/{risk_cfg['max_positions']})"

    # 每日亏损检查
    max_daily_loss = equity * risk_cfg["max_daily_loss_pct"] / 100.0
    if abs(state.daily_loss) >= max_daily_loss and max_daily_loss > 0:
        return False, (f"Daily loss limit reached: "
                       f"{state.daily_loss:.2f} >= {max_daily_loss:.2f}")

    # 挂单数检查（避免挂单堆积）
    pending = client.get_pending_orders()
    if len(pending) >= risk_cfg["max_positions"] * 2:
        return False, f"Too many pending orders ({len(pending)})"

    return True, "OK"


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def main_loop(config: dict, once: bool = False, max_rounds: int = 0):
    """Agent v2.0 主循环 — 多通道融合 + 自适应优化 + 记忆反思。"""
    logger = logging.getLogger("agent")
    client = OKXClient(config)

    # ---- 初始化模块 ----
    state_path = os.path.join(os.path.dirname(__file__), "agent_state.json")
    state_manager = StateManager(state_path)

    # 多通道专家引擎
    mc_cfg = config.get("multi_channel", {})
    expert_engine = MasterTraderEngine(
        weights=mc_cfg.get("channel_weights")
    ) if mc_cfg.get("enabled", True) else None

    # 优化模块
    opt_cfg = config.get("optimization", {})
    edge_analyzer = EdgeAnalyzer()
    adaptive_tuner = AdaptiveParameterTuner(config) if opt_cfg.get("adaptive_enabled", True) else None
    trailing_stop = TrailingStop(
        activation_pct=opt_cfg.get("trailing_stop_activation_pct", 0.50),
        trail_pct=opt_cfg.get("trailing_stop_trail_pct", 0.30),
    ) if opt_cfg.get("trailing_stop_enabled", True) else None

    # 记忆模块
    mem_cfg = config.get("memory", {})
    memory = MemoryManager(
        memory_dir=mem_cfg.get("memory_dir", "memory")
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
    alpha_zoo = AlphaZoo() if az_cfg.get("enabled", True) else None

    # 因果滞后体制检测 (Vibe-Trading)
    regime_detector = CausalHysteresisRegime(
        hysteresis_threshold=az_cfg.get("hysteresis_threshold", 0.15),
        min_regime_duration=az_cfg.get("min_regime_duration", 5),
    ) if az_cfg.get("enabled", True) else None

    # FreqAI 在线学习 (freqtrade 52k⭐)
    fa_cfg = config.get("freqai", {})
    freqai = FreqAIPipeline(
        feature_window=fa_cfg.get("feature_window", 50),
        retrain_interval=fa_cfg.get("retrain_interval", 10),
    ) if fa_cfg.get("enabled", True) else None

    # Hyperopt 优化器 (freqtrade) — 定期触发
    hp_cfg = config.get("hyperopt", {})
    hyperopt_enabled = hp_cfg.get("enabled", True)
    hyperopt_interval = hp_cfg.get("optimize_interval_rounds", 50)

    # ---- 时段报告 ----
    session_reporter = SessionReporter(config) if config.get("report", {}).get("enabled", True) else None
    if session_reporter:
        logger.info(f"  Session Report: enabled (times: {session_reporter.session_times})")

    scan_interval = config["agent"]["scan_interval_seconds"]
    risk_cfg = config["risk"]

    # ---- 研判挡位（必须在 tracker/warmup 之前计算，供其使用 scan_config） ----
    agg_mode = config["agent"].get("aggressiveness", 3)
    agg_thresholds = get_aggressiveness_thresholds(agg_mode)
    logger.info(f"  Aggressiveness: {agg_mode} ({agg_thresholds['label']})")
    # 应用挡位到策略配置（影响 FVG 检测宽严度）
    scan_config = apply_aggressiveness_to_config(config, agg_thresholds)

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
        )
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
    logger.info("  OKX FVG Trading Agent v3.0 — 全模块融合版")
    logger.info(f"  融合: freqtrade(52k⭐) + TradingAgents(86k⭐) + Vibe-Trading(23.6k⭐)")
    logger.info(f"  Mode: {'🟡 DEMO 模拟交易' if demo_mode else '🔴 LIVE 实盘交易'}")
    logger.info(f"  Dry Run: {config['agent'].get('dry_run', False)}")
    logger.info(f"  Multi-Channel: {mc_cfg.get('enabled', True)}")
    logger.info(f"  Debate Engine: {debate_cfg.get('enabled', True)}")
    logger.info(f"  Alpha Zoo: {az_cfg.get('enabled', True)}")
    logger.info(f"  Regime Detector: {az_cfg.get('enabled', True)}")
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

    # ---- 初始权益 ----
    equity = client.get_total_equity()
    if equity is None:
        logger.error("Cannot get account equity. Check API credentials.")
        return
    state_manager.update_equity(equity)
    logger.info(f"Initial equity: {equity:.2f} USDT")

    round_count = 0
    trade_count_since_reflection = 0

    while True:
        round_count += 1
        round_start = time.time()
        logger.info(f"\n{'─'*50}\n  ROUND {round_count}  "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'─'*50}")

        # 轮次上限检查
        if max_rounds > 0 and round_count > max_rounds:
            logger.info(f"达到最大轮次 {max_rounds}，停止")
            if tracker:
                tracker.stop()
            break

        # 暂停后台追踪（主循环执行期间避免 API 冲突）
        if tracker:
            tracker.pause()

        # ---- 时段报告: 检测是否到达亚洲/欧美开盘转换点 ----
        if session_reporter and cache:
            try:
                generate_and_send_report(cache, config)
            except Exception as e:
                logger.error(f"时段报告生成失败: {e}")

        # ---- 步骤 0: 状态维护 ----
        state_manager.reset_daily_if_new_day()
        equity = client.get_total_equity()
        if equity is None:
            logger.error("Cannot get equity, skipping round")
            if tracker:
                tracker.resume()
            time.sleep(scan_interval)
            continue
        state_manager.update_equity(equity)

        # ---- 步骤 1: 监控持仓 + Trailing Stop ----
        positions = monitor_positions(client, state_manager, config)
        active_count = len([p for p in positions.values() if p["size"] > 0])

        # 动态止损更新
        if trailing_stop and active_count > 0:
            for inst_id, pos in positions.items():
                if pos["size"] > 0:
                    current_price = pos["mark_px"]
                    new_sl = trailing_stop.update(
                        current_price=current_price,
                        entry_price=pos["avg_px"],
                        stop_loss=pos["avg_px"] * 0.95,  # 兜底
                        take_profit=pos["avg_px"] * 1.05,  # 兜底
                        direction=pos["pos_side"],
                    )
                    if trailing_stop._activated:
                        logger.debug(f"[TS] {inst_id} trailing stop: {new_sl:.2f}")
                        # 撤销旧止损单并提交新追踪止损
                        try:
                            client.cancel_algo_order(inst_id=inst_id, ord_type="conditional")
                            client.place_algo_order(
                                inst_id=inst_id,
                                td_mode=risk_cfg["margin_mode"],
                                side="sell" if pos["pos_side"] == "long" else "buy",
                                pos_side=pos["pos_side"],
                                sz=str(pos["size"]),
                                ord_type="conditional",
                                sl_trigger_px=str(new_sl),
                                sl_trigger_px_type="last",
                                sl_ord_px="-1",
                            )
                            logger.debug(f"[TS] {inst_id} 追踪止损已提交至交易所: {new_sl:.2f}")
                        except Exception as e:
                            logger.error(f"[TS] {inst_id} 追踪止损提交失败: {e}")

        # ---- 步骤 2: 提现检查 ----
        if state_manager.check_withdrawal(equity, risk_cfg["profit_withdrawal_pct"]):
            withdrawal_amount = equity * risk_cfg["profit_withdrawal_pct"] / 100.0
            logger.info(
                f"\n{'#'*60}\n"
                f"  !!! 提现提醒 !!!\n"
                f"  钱包已翻倍！当前权益: {equity:.2f} USDT\n"
                f"  建议提现: {withdrawal_amount:.2f} USDT ({risk_cfg['profit_withdrawal_pct']}%)\n"
                f"  剩余继续: {equity - withdrawal_amount:.2f} USDT\n"
                f"{'#'*60}"
            )
            state_manager.record_withdrawal(equity, risk_cfg["profit_withdrawal_pct"])

        # ---- 步骤 3: 自适应参数调整 ----
        if adaptive_tuner and edge_analyzer.trades:
            edge_stats = edge_analyzer.analyze(opt_cfg.get("edge_lookback", 100))
            adaptive_tuner.adapt(edge_stats)

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
            logger.debug(f"Adaptive params: leverage={eff_lev}x, "
                         f"risk={eff_risk:.1f}%, min_score={eff_min_score:.2f}")

        # ---- 步骤 4: 风控门禁 ----
        passed, reason = risk_gate(client, state_manager, equity, config)
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

        # 优先从后台缓存读取预研究结果
        cache_hit = False
        if cache:
            # 诊断：打印缓存统计
            stats = cache.stats()
            logger.debug(f"[Cache] 统计: total={stats['total']} fresh={stats['fresh']} "
                         f"with_signals={stats['with_signals']} with_analysis={stats['with_analysis']}")
            # 诊断：打印每条 entry 的过滤原因
            for entry in cache.get_all_entries():
                if entry.analysis:
                    a = entry.analysis
                    logger.debug(f"[Cache] {entry.inst_id}: conf={a.final_confidence:.0%} "
                                 f"agree={a.channel_agreement:.0%} "
                                 f"fresh={entry.is_fresh(120)}")
            cached_signals = cache.get_fresh_signals(
                min_confidence=agg_thresholds["min_confidence"],
                min_agreement=agg_thresholds["min_agreement"] * 0.4,  # 缓存预筛选比主筛选宽松
            )
            if cached_signals:
                cache_hit = True
                logger.info(f"[Cache] 命中 {len(cached_signals)} 个预研究币种 "
                            f"(positions: {active_count}/{risk_cfg['max_positions']})")

                for entry, coin_info in cached_signals:
                    inst_id = entry.inst_id
                    if inst_id in positions:
                        continue

                    all_signals.extend(entry.signals)

                    if entry.analysis and (best_analysis is None or
                            entry.analysis.final_confidence > best_analysis.final_confidence):
                        best_analysis = entry.analysis
                        best_coin = coin_info

                    logger.debug(f"  [Cache] {inst_id}: "
                                 f"score={entry.analysis.final_score:+.2f} "
                                 f"conf={entry.analysis.final_confidence:.0%} "
                                 f"regime={entry.detected_regime}")

        # 缓存未命中，回退到传统实时扫描
        if not cache_hit:
            coins = get_tradable_coins(client, config)
            logger.info(f"[Scan] 缓存未命中，实时扫描 {len(coins)} 个币种 "
                        f"(positions: {active_count}/{risk_cfg['max_positions']})")

            for coin in coins:
                if coin["instId"] in positions:
                    continue

                try:
                    inst_id = coin["instId"]
                    current_price = coin["last"]

                    # 使用 scan_round 完成 K 线获取 + FVG 扫描
                    scan_result = scan_round(client, coin, scan_config)
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
                                      for i in range(1, len(candles_1h))]
                            ret_4h = [math.log(candles_4h[i].close / candles_4h[i-1].close)
                                      for i in range(1, len(candles_4h))]
                            # 计算趋势方向和相关性
                            trend_1h_sign = float(np.sign(np.mean(ret_1h))) if ret_1h else 0.0
                            trend_4h_sign = float(np.sign(np.mean(ret_4h))) if ret_4h else 0.0
                            # 对齐数组长度避免 np.corrcoef 维度不匹配
                            min_len = min(len(ret_1h), len(ret_4h))
                            corr = 0.0
                            if min_len > 1:
                                try:
                                    corr = float(np.corrcoef(ret_1h[:min_len], ret_4h[:min_len])[0, 1])
                                except Exception:
                                    corr = 0.0
                            regime_state = regime_detector.update(
                                correlation=corr,
                                trend_1h_sign=trend_1h_sign,
                                trend_4h_sign=trend_4h_sign,
                            )
                            detected_regime = regime_state.value
                            logger.debug(f"[Regime] {inst_id}: {detected_regime} "
                                         f"(duration={regime_detector.state.regime_duration})")

                        # ---- 多空辩论 (TradingAgents 86k⭐) ----
                        if debate_cfg.get("enabled", True) and debate_engine:
                            # 将 ChannelReport 传给辩论引擎进行完整辩论
                            debate_result = debate_engine.conduct_debate(
                                symbol=inst_id,
                                channel_reports=analysis.channels,
                                fvg_signals=signals,
                                current_price=current_price,
                                regime=detected_regime,
                            )
                            logger.debug(f"[Debate] {inst_id}: {debate_result.winner} "
                                         f"score={debate_result.final_score:+.2f} "
                                         f"conf={debate_result.confidence:.0%} "
                                         f"verdict={debate_result.action_recommendation[:50]}")

                            # 辩论结果影响分析置信度
                            if debate_result.winner == "tie":
                                analysis.final_confidence *= 0.8  # 平局降低置信度
                            elif debate_result.winner == "bullish" and analysis.final_score > 0:
                                analysis.final_confidence *= 1.1  # 多方一致增强
                            elif debate_result.winner == "bearish" and analysis.final_score < 0:
                                analysis.final_confidence *= 1.1
                            analysis.final_confidence = min(0.95, analysis.final_confidence)

                        # ---- Alpha 因子分析 (Vibe-Trading 23.6k⭐) ----
                        if alpha_zoo and signals:
                            # 提取 FVG 信号作为因子值
                            fvg_scores = np.array([s.score for s in signals])
                            # 记录因子值用于后续回测
                            logger.debug(f"[AlphaZoo] {inst_id}: {len(signals)} signals, "
                                         f"max_score={fvg_scores.max():.2f}")

                        # 检查是否满足最低置信度
                        below_threshold = (
                            analysis.final_confidence < agg_thresholds["min_confidence"] or
                            analysis.channel_agreement < agg_thresholds["min_agreement"]
                        )

                        # 记录所有分析（挡位 1 激进模式需要回退用）
                        if best_analysis is None or analysis.final_confidence > best_analysis.final_confidence:
                            best_analysis = analysis
                            best_coin = coin

                        if below_threshold:
                            logger.debug(f"[Filter] {inst_id} "
                                         f"confidence={analysis.final_confidence:.0%} "
                                         f"agreement={analysis.channel_agreement:.0%} — filtered "
                                         f"(threshold: conf>={agg_thresholds['min_confidence']:.0%}, "
                                         f"agree>={agg_thresholds['min_agreement']:.0%})")
                            continue

                        logger.debug(f"  {inst_id}: {len(signals)} FVG signals, "
                                     f"master={analysis.final_score:+.2f} "
                                     f"conf={analysis.final_confidence:.0%}")

                    all_signals.extend(signals)

                except Exception as e:
                    logger.error(f"Error scanning {coin['instId']}: {e}")

        # ---- 打印最佳分析报告 ----
        if best_analysis:
            report = format_analysis_report(best_analysis)
            logger.info(report)

        # ---- 步骤 6: 信号排序 & 执行 ----
        all_signals.sort(key=lambda s: s.score, reverse=True)

        if active_count > 0 and all_signals:
            best_signal = all_signals[0]
            current_inst_id = next(iter(positions.keys()), None)

            if current_inst_id and best_signal.inst_id != current_inst_id:
                logger.info(f"持有 {current_inst_id}，检测到新信号 {best_signal.inst_id} "
                            f"(score={best_signal.score:.2f})，先平仓")

                # 记录平仓交易
                current_pos = positions[current_inst_id]
                if current_pos["size"] > 0:
                    trade_record = TradeRecord(
                        symbol=current_inst_id,
                        direction=current_pos["pos_side"],
                        entry_time=time.time(),
                        exit_time=time.time(),
                        entry_price=current_pos["avg_px"],
                        exit_price=current_pos["mark_px"],
                        quantity=current_pos["size"],
                        leverage=int(current_pos.get("leverage", 1)),
                        pnl=current_pos.get("upl", 0),
                        pnl_pct=current_pos.get("upl_ratio_pct", 0),
                        is_win=current_pos.get("upl", 0) > 0,
                        exit_reason="signal_switch",
                        fvg_score=0,
                        master_score=best_analysis.final_score if best_analysis else 0,
                    )
                    edge_analyzer.add_trade(trade_record)

                    # 记录决策日志
                    if memory:
                        memory.log_decision(DecisionLog(
                            timestamp=time.time(),
                            symbol=current_inst_id,
                            direction=current_pos["pos_side"],
                            entry_price=current_pos["avg_px"],
                            exit_price=current_pos["mark_px"],
                            pnl=current_pos.get("upl", 0),
                            pnl_pct=current_pos.get("upl_ratio_pct", 0),
                            is_win=current_pos.get("upl", 0) > 0,
                            exit_reason="signal_switch",
                            master_score=best_analysis.final_score if best_analysis else 0,
                        ))
                        trade_count_since_reflection += 1

                client.close_position(
                    inst_id=current_inst_id,
                    pos_side=current_pos["pos_side"],
                    mgn_mode=risk_cfg["margin_mode"],
                )
                time.sleep(2)
            elif current_inst_id and best_signal.inst_id == current_inst_id:
                logger.debug(f"已有相同标的 {current_inst_id} 持仓，跳过")
                all_signals = []

        if not all_signals:
            # ---- 挡位 1 激进模式: 无信号时强制选最优币种 ----
            if agg_thresholds.get("force_trade") and best_analysis and best_coin:
                logger.warning(
                    f"[ForceTrade] 激进模式: 无币种通过阈值，强制选择最优 "
                    f"{best_coin['instId']} (conf={best_analysis.final_confidence:.0%}, "
                    f"agree={best_analysis.channel_agreement:.0%})"
                )
                # 用最优分析生成信号（即使低于阈值也执行）
                inst_id = best_coin["instId"]
                raw = client.get_candles(inst_id, bar="1H", limit=200)
                if raw:
                    candles_1h = candles_from_raw(raw)
                    fvgs = scan_fvg_all_timeframes(
                        inst_id=inst_id,
                        candles_by_tf={"1H": candles_1h},
                        current_price=best_coin["last"],
                        config=scan_config,
                        funding_rate=client.get_funding_rate(inst_id),
                        spread_pct=calculate_spread(
                            best_coin.get("bidPx", 0), best_coin.get("askPx", 0)
                        ),
                    )
                    all_signals = fvgs
                if not all_signals:
                    # 极端情况: 无 FVG 信号，构造一个最小信号
                    direction = "long" if best_analysis.final_score > 0 else "short"
                    minimal_fvg = FVG(
                        direction=direction,
                        top=best_coin["last"] * 1.005,
                        bottom=best_coin["last"] * 0.995,
                        width_pct=1.0,
                        candle_ts=int(time.time() * 1000),
                        timeframe="1H",
                        impulse_candle=Candle(
                            timestamp=int(time.time() * 1000),
                            open=best_coin["last"],
                            high=best_coin["last"] * 1.005,
                            low=best_coin["last"] * 0.995,
                            close=best_coin["last"],
                            volume=0,
                        ),
                        is_abnormal=False,
                        sigma=0.0,
                        volume_ratio=1.0,
                    )
                    all_signals = [Signal(
                        inst_id=inst_id,
                        fvg=minimal_fvg,
                        score=best_analysis.final_score * 0.5,
                        position_side=direction,
                        entry_price=best_coin["last"],
                        stop_loss=best_coin["last"] * 0.95,
                        take_profit=best_coin["last"] * 1.05,
                        leverage=min(risk_cfg["max_leverage"], 3),
                        reason=f"Force trade fallback (conf={best_analysis.final_confidence:.0%})",
                    )]

                # ---- 用户确认建仓 ----
                s = all_signals[0]
                direction = "做多" if s.position_side == "long" else "做空"
                print()
                print(f"  ╔══════════════════════════════════════════════╗")
                print(f"  ║  [激进模式] 强制建仓候选                      ║")
                print(f"  ╠══════════════════════════════════════════════╣")
                print(f"  ║  币种: {s.inst_id:<38s} ║")
                print(f"  ║  方向: {direction:<38s} ║")
                print(f"  ║  入场: {s.entry_price:<38.6f} ║")
                print(f"  ║  止损: {s.stop_loss:<38.6f} ║")
                print(f"  ║  止盈: {s.take_profit:<38.6f} ║")
                print(f"  ║  杠杆: {s.leverage}x{'':>36s} ║")
                print(f"  ║  置信度: {best_analysis.final_confidence:<35.0%} ║")
                print(f"  ║  一致性: {best_analysis.channel_agreement:<35.0%} ║")
                print(f"  ╚══════════════════════════════════════════════╝")
                print()

                if client.dry_run:
                    answer = input("  [Dry Run] 确认下单? (y/n): ").strip().lower()
                else:
                    answer = input("  ⚠️  实盘下单! 确认? (y/n): ").strip().lower()

                if answer != "y":
                    logger.info("[ForceTrade] 用户取消建仓，跳过本轮")
                    all_signals = []
                else:
                    logger.info(f"[ForceTrade] 用户确认建仓: {s.inst_id} {direction}")

            else:
                logger.info("No new signals this round")
                state_manager.save()
                print_summary(state_manager.state, equity)

                # Edge 统计
                if edge_analyzer.trades:
                    stats = edge_analyzer.analyze(opt_cfg.get("edge_lookback", 100))
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
        manage_pending_orders(client, best_signal.entry_price, signal=best_signal)

        # ---- 自适应评分过滤 ----
        if adaptive_tuner:
            _, _, min_score = adaptive_tuner.get_effective_params()
            if best_signal.score < min_score:
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
            # 构建特征向量
            features = {
                "fvg_score": best_signal.score,
                "master_score": best_analysis.final_confidence,
                "leverage": float(best_signal.leverage),
                "is_long": 1.0 if best_signal.position_side == "long" else 0.0,
            }
            predicted_pnl = freqai.predict(features)
            if predicted_pnl < agg_thresholds["min_prediction_confidence"]:
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

        inst_info = client.get_instrument_info(best_signal.inst_id)
        if inst_info is None:
            logger.warning(f"Cannot get instrument info for {best_signal.inst_id}")
        else:
            # 使用自适应杠杆 — 修改 best_signal.leverage 使其在 execute_signal 中生效
            if adaptive_tuner:
                eff_lev, _, _ = adaptive_tuner.get_effective_params()
                best_signal.leverage = min(eff_lev, best_signal.leverage)
            effective_leverage = best_signal.leverage

            ord_id = execute_signal(
                client=client,
                signal=best_signal,
                equity=equity,
                config=config,
                instrument_info=inst_info,
            )
            if ord_id:
                state_manager.state.positions_opened += 1
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
                        regime=regime_detector.state.current_regime.value if regime_detector else "",
                        red_flags=best_analysis.key_risks,
                        expert_verdict=best_analysis.expert_verdict[:200],
                    ))

        # ---- 步骤 7: 定期反思 ----
        if memory and trade_count_since_reflection >= mem_cfg.get("reflection_interval_trades", 10):
            recent_decisions = memory.get_recent_decisions(20)
            logger.info("\n📝 生成交易反思报告...")
            memory.generate_reflection(recent_decisions)
            trade_count_since_reflection = 0

        # ---- 步骤 8: Edge 统计 ----
        if edge_analyzer.trades:
            stats = edge_analyzer.analyze(opt_cfg.get("edge_lookback", 100))
            logger.info(f"[Edge] 胜率={stats.win_rate:.1%} "
                        f"盈亏比={stats.profit_factor:.2f} "
                        f"期望值={stats.expectancy:+.2f} "
                        f"连亏={stats.consecutive_losses}")

        # ---- 步骤 9: Kelly 仓位分析 (freqtrade 52k⭐) ----
        if hyperopt_enabled and edge_analyzer.trades and len(edge_analyzer.trades) >= 10:
            kelly = compute_kelly(edge_analyzer.trades)
            logger.info(f"[Kelly] f*={kelly.kelly_fraction:.4f} "
                        f"1/2K={kelly.half_kelly:.4f} "
                        f"推荐风险={kelly.recommended_risk_pct:.2f}% "
                        f"期望增长={kelly.expected_growth_rate:+.4f}")

        # ---- 步骤 10: 定期 Hyperopt 优化 (freqtrade 52k⭐) ----
        if (hyperopt_enabled and edge_analyzer.trades
                and len(edge_analyzer.trades) >= 20
                and round_count % hyperopt_interval == 0):
            logger.info("\n🔧 运行 Hyperopt 参数优化...")
            try:
                opt_result = run_full_optimization(
                    trades=edge_analyzer.trades,
                    initial_equity=state_manager.state.initial_equity or equity,
                )
                if opt_result:
                    logger.info(f"[Hyperopt] best_score={opt_result['metrics'].composite_score:.1f}")
                    logger.info(f"[Hyperopt] best_params={opt_result['hyperopt'].best_params}")
                    logger.info(f"[Hyperopt] is_overfitting={opt_result['walk_forward'].is_overfitting}")
                    if opt_result.get("dashboard"):
                        logger.info(opt_result["dashboard"])
            except Exception as e:
                logger.error(f"Hyperopt failed: {e}")

        # ---- 步骤 11: FreqAI 在线更新 ----
        if freqai and edge_analyzer.trades:
            recent_trades = edge_analyzer.trades[-fa_cfg.get("feature_window", 50):]
            freqai.update(recent_trades)

        # ---- 保存状态 ----
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

    logger.info("Agent stopped.")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OKX FVG 交易机器人 — 基于 Fair Value Gap 的合约交易 Agent"
    )
    parser.add_argument(
        "--配置文件", "-c",
        dest="config",
        default=os.path.join(os.path.dirname(__file__), "config.json"),
        help="配置文件路径，默认: config.json"
    )
    parser.add_argument(
        "--演练", "-d",
        dest="dry_run",
        action="store_true",
        help="演练模式，只分析不下单"
    )
    parser.add_argument(
        "--单轮", "-o",
        dest="once",
        action="store_true",
        help="只跑一轮后退出"
    )
    parser.add_argument(
        "--日志级别",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--轮次", "-r",
        dest="rounds",
        type=int, default=0,
        help="最大运行轮次，0 表示无限制"
    )
    parser.add_argument(
        "--扫描间隔",
        dest="scan_interval",
        type=int, default=None,
        help="扫描间隔秒数，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--币种上限",
        dest="coin_limit",
        type=int, default=None,
        help="扫描币种数量上限，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--挡位", "-a",
        dest="aggressiveness",
        type=int, choices=[1, 2, 3],
        default=None,
        help="研判挡位: 1=激进(每天必找一币), 2=均衡(2-3天一笔), 3=保守(默认)"
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
    if args.log_level:
        config["agent"]["log_level"] = args.log_level
    if args.scan_interval is not None:
        config["agent"]["scan_interval_seconds"] = args.scan_interval
    if args.coin_limit is not None:
        config["agent"]["coin_scan_limit"] = args.coin_limit
    if args.aggressiveness is not None:
        config["agent"]["aggressiveness"] = args.aggressiveness

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
    try:
        main_loop(config, once=args.once, max_rounds=args.rounds)
    except KeyboardInterrupt:
        logger.info("用户中断。正在保存状态...")
    except Exception as e:
        logger.exception(f"致命错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()