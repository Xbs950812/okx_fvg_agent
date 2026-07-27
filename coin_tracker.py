"""
币种追踪研究模块 — 后台持续研究，主循环直接取用缓存。

架构:
  CoinResearchCache — 线程安全的研究结果缓存 (TTL 过期机制)
  CoinTracker — 后台线程，持续拉取 K 线、计算 FVG、多通道分析、体制检测

设计原则:
  - 闲置时间（300s 扫描间隔）持续研究，不等建仓信号才去拉数据
  - 高成交量币种优先研究，确保热门标的缓存最新
  - 主循环从缓存读取，命中率越高，延迟越低
  - 缓存过期自动刷新，保证数据新鲜度
"""

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from strategy import Candle, Signal, candles_from_raw, scan_fvg_all_timeframes
from multi_channel import (
    MasterAnalysis, MasterTraderEngine,
    full_multi_channel_analysis,
)
from executor import calculate_spread, get_tradable_coins


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 缓存数据结构
# ---------------------------------------------------------------------------

@dataclass
class CoinResearchEntry:
    """单个币种的预计算研究结果。"""
    inst_id: str
    # 基础数据
    current_price: float = 0.0
    bid_px: float = 0.0
    ask_px: float = 0.0
    vol24h: float = 0.0
    # K 线
    candles_by_tf: Dict[str, List[Candle]] = field(default_factory=dict)
    # FVG 信号
    signals: List[Signal] = field(default_factory=list)
    # 多通道分析
    analysis: Optional[MasterAnalysis] = None
    # 体制检测
    detected_regime: str = "NEUTRAL"
    # 资金费率
    funding_rate: Optional[float] = None
    spread_pct: float = 0.0
    # 元数据
    researched_at: float = 0.0          # 研究完成时间戳
    candle_fetched_at: float = 0.0       # K 线拉取时间戳
    error_count: int = 0                 # 连续错误计数
    has_signals: bool = False            # 是否有有效 FVG 信号
    has_analysis: bool = False           # 是否完成多通道分析

    def is_fresh(self, max_age_seconds: float = 120) -> bool:
        """检查研究结果是否仍新鲜。"""
        if self.researched_at <= 0:
            return False
        return (time.time() - self.researched_at) < max_age_seconds

    def is_candle_fresh(self, max_age_seconds: float = 60) -> bool:
        """检查 K 线数据是否仍新鲜。"""
        if self.candle_fetched_at <= 0:
            return False
        return (time.time() - self.candle_fetched_at) < max_age_seconds


# ---------------------------------------------------------------------------
# 线程安全缓存
# ---------------------------------------------------------------------------

class CoinResearchCache:
    """线程安全的研究结果缓存。

    特性:
      - TTL 过期自动失效
      - LRU 淘汰 (保留最近访问的)
      - 读写锁保护
    """

    def __init__(self, max_entries: int = 200, candle_ttl: float = 60.0,
                 research_ttl: float = 120.0):
        self._lock = threading.RLock()
        self._entries: Dict[str, CoinResearchEntry] = OrderedDict()
        self.max_entries = max_entries
        self.candle_ttl = candle_ttl
        self.research_ttl = research_ttl

    def get(self, inst_id: str) -> Optional[CoinResearchEntry]:
        """获取缓存条目（线程安全）。"""
        with self._lock:
            entry = self._entries.get(inst_id)
            if entry is None:
                return None
            # 移动到 LRU 末尾（最近访问）
            self._entries.move_to_end(inst_id)
            return entry

    def put(self, inst_id: str, entry: CoinResearchEntry):
        """写入缓存条目（线程安全）。"""
        with self._lock:
            self._entries[inst_id] = entry
            self._entries.move_to_end(inst_id)
            # LRU 淘汰
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def get_or_create(self, inst_id: str) -> CoinResearchEntry:
        """获取或创建空条目（线程安全）。"""
        with self._lock:
            if inst_id in self._entries:
                self._entries.move_to_end(inst_id)
                return self._entries[inst_id]
            entry = CoinResearchEntry(inst_id=inst_id)
            self._entries[inst_id] = entry
            return entry

    def needs_candle_refresh(self, inst_id: str) -> bool:
        """判断是否需要刷新 K 线。"""
        entry = self.get(inst_id)
        if entry is None:
            return True
        return not entry.is_candle_fresh(self.candle_ttl)

    def needs_research(self, inst_id: str) -> bool:
        """判断是否需要完整研究。"""
        entry = self.get(inst_id)
        if entry is None:
            return True
        return not entry.is_fresh(self.research_ttl)

    def get_all_entries(self) -> List[CoinResearchEntry]:
        """获取所有条目（用于扫描）。"""
        with self._lock:
            return list(self._entries.values())

    def get_fresh_signals(self, min_confidence: float = 0.40,
                          min_agreement: float = 0.50) -> List[Tuple[CoinResearchEntry, dict]]:
        """获取所有新鲜且满足置信度要求的信号条目。

        Returns:
            [(entry, coin_info_dict), ...] 按置信度降序
        """
        with self._lock:
            results = []
            now = time.time()
            for entry in self._entries.values():
                if not entry.is_fresh(self.research_ttl):
                    continue
                if not entry.has_analysis or entry.analysis is None:
                    continue
                a = entry.analysis
                if (a.final_confidence < min_confidence or
                        a.channel_agreement < min_agreement):
                    continue
                coin_info = {
                    "instId": entry.inst_id,
                    "last": entry.current_price,
                    "vol24h": entry.vol24h,
                    "bidPx": entry.bid_px,
                    "askPx": entry.ask_px,
                }
                results.append((entry, coin_info))
            # 按置信度降序
            results.sort(key=lambda x: x[0].analysis.final_confidence, reverse=True)
            return results

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息。"""
        with self._lock:
            total = len(self._entries)
            fresh = sum(1 for e in self._entries.values()
                       if e.is_fresh(self.research_ttl))
            with_signals = sum(1 for e in self._entries.values() if e.has_signals)
            with_analysis = sum(1 for e in self._entries.values() if e.has_analysis)
            return {
                "total": total,
                "fresh": fresh,
                "stale": total - fresh,
                "with_signals": with_signals,
                "with_analysis": with_analysis,
            }


# ---------------------------------------------------------------------------
# 后台追踪线程
# ---------------------------------------------------------------------------

class CoinTracker(threading.Thread):
    """后台币种追踪研究线程。

    在闲置时间持续研究币种，主循环读取缓存。

    工作流程:
      1. 每 refresh_interval 秒拉取可交易币种列表
      2. 按成交量排序，高成交量优先研究
      3. 对每个币种：拉 K 线 → 计算 FVG → 多通道分析 → 体制检测 → 存入缓存
      4. 已研究的币种按 TTL 自动刷新
      5. 主循环通过 pause/resume 控制执行时机

    用法:
      tracker = CoinTracker(client, config, cache, expert_engine, regime_detector)
      tracker.start()   # 开始后台研究
      ...
      tracker.pause()   # 主循环执行时暂停
      # 主循环从 cache 读取
      tracker.resume()  # 主循环完成后恢复
    """

    def __init__(
        self,
        client,           # OKXClient
        config: dict,
        cache: CoinResearchCache,
        scan_config: Optional[dict] = None,
        expert_engine: Optional[MasterTraderEngine] = None,
        regime_detector=None,  # CausalHysteresisRegime
        debate_engine=None,    # TradingAgentsDebateEngine
    ):
        super().__init__(daemon=True, name="CoinTracker")
        self.client = client
        self.config = config
        self.scan_config = scan_config if scan_config is not None else config
        self.cache = cache
        self.expert_engine = expert_engine
        self.regime_detector = regime_detector
        self.debate_engine = debate_engine

        # 控制信号
        self._pause_event = threading.Event()
        self._pause_event.set()  # 初始为运行状态
        self._stop_event = threading.Event()

        # 配置
        tracker_cfg = config.get("coin_tracker", {})
        self.refresh_interval = tracker_cfg.get("refresh_interval_seconds", 120)
        self.candle_refresh_interval = tracker_cfg.get("candle_refresh_seconds", 60)
        self.coin_list_refresh_interval = tracker_cfg.get("coin_list_refresh_seconds", 300)
        self.research_batch_size = tracker_cfg.get("research_batch_size", 5)
        self.batch_cooldown_seconds = tracker_cfg.get("batch_cooldown_seconds", 2.0)

        self._coin_list: List[dict] = []
        self._coin_list_fetched_at: float = 0.0
        self._research_cursor: int = 0    # 轮询指针
        self._total_researched: int = 0
        self._total_errors: int = 0

    # ---- 控制接口 ----

    def pause(self):
        """暂停后台研究（主循环执行前调用）。"""
        self._pause_event.clear()
        logger.debug("CoinTracker paused")

    def resume(self):
        """恢复后台研究（主循环完成后调用）。"""
        self._pause_event.set()
        logger.debug("CoinTracker resumed")

    def stop(self):
        """停止后台线程。"""
        self._stop_event.set()
        self._pause_event.set()  # 确保不卡在等待

    # ---- 线程主循环 ----

    def run(self):
        logger.info(f"[CoinTracker] 后台研究线程启动 "
                    f"(refresh={self.refresh_interval}s, "
                    f"candle={self.candle_refresh_interval}s, "
                    f"batch={self.research_batch_size})")

        while not self._stop_event.is_set():
            # 等待运行信号
            self._pause_event.wait()

            try:
                self._research_cycle()
            except Exception as e:
                logger.error(f"[CoinTracker] 研究周期异常: {e}")
                self._total_errors += 1
                time.sleep(5)

            # 短暂休眠避免 CPU 空转
            time.sleep(1)

        logger.info(f"[CoinTracker] 后台线程停止 "
                    f"(研究: {self._total_researched}, 错误: {self._total_errors})")

    def _research_cycle(self):
        """单次研究周期：拉取币种列表，逐批研究。"""
        # ---- 刷新币种列表 ----
        now = time.time()
        if (not self._coin_list or
                now - self._coin_list_fetched_at > self.coin_list_refresh_interval):
            try:
                self._coin_list = get_tradable_coins(self.client, self.config)
                self._coin_list_fetched_at = now
                self._research_cursor = 0
                logger.debug(f"[CoinTracker] 刷新币种列表: {len(self._coin_list)} 个")
            except Exception as e:
                logger.warning(f"[CoinTracker] 获取币种列表失败: {e}")
                return

        if not self._coin_list:
            return

        # ---- 逐批研究 ----
        mc_cfg = self.config.get("multi_channel", {})
        debate_cfg = self.config.get("debate_engine", {})
        researched_this_cycle = 0

        for _ in range(self.research_batch_size):
            if self._stop_event.is_set() or not self._pause_event.is_set():
                break

            # 轮询选择下一个币种
            coin = self._coin_list[self._research_cursor % len(self._coin_list)]
            self._research_cursor += 1

            inst_id = coin["instId"]

            # 检查是否需要研究
            if not self.cache.needs_research(inst_id):
                continue

            try:
                self._research_single_coin(coin, mc_cfg, debate_cfg)
                researched_this_cycle += 1
                self._total_researched += 1
            except Exception as e:
                logger.warning(f"[CoinTracker] 研究 {inst_id} 失败: {e}")
                # 记录错误，但不阻塞
                entry = self.cache.get_or_create(inst_id)
                entry.error_count += 1

        if researched_this_cycle > 0:
            stats = self.cache.stats()
            logger.debug(f"[CoinTracker] 本轮研究 {researched_this_cycle} 个币种 "
                         f"(缓存: {stats['fresh']}/{stats['total']} 新鲜)")

        # 批次间冷却
        if researched_this_cycle > 0:
            time.sleep(self.batch_cooldown_seconds)

    def _research_single_coin(self, coin: dict, mc_cfg: dict, debate_cfg: dict):
        """研究单个币种：拉 K 线 → FVG → 多通道 → 体制 → 辩论 → 缓存。"""
        inst_id = coin["instId"]
        current_price = coin["last"]
        entry = self.cache.get_or_create(inst_id)

        # 更新基础信息
        entry.current_price = current_price
        entry.bid_px = coin.get("bidPx", 0)
        entry.ask_px = coin.get("askPx", 0)
        entry.vol24h = coin.get("vol24h", 0)

        # ---- 步骤 1: 拉 K 线 ----
        if self.cache.needs_candle_refresh(inst_id):
            candles_by_tf: Dict[str, List[Candle]] = {}
            for tf in self.config["strategy"]["timeframes"]:
                raw = self.client.get_candles(inst_id, bar=tf, limit=200)
                if raw:
                    candles_by_tf[tf] = candles_from_raw(raw)
            if not candles_by_tf:
                logger.debug(f"[CoinTracker] {inst_id} 无 K 线数据")
                return
            entry.candles_by_tf = candles_by_tf
            entry.candle_fetched_at = time.time()
        else:
            candles_by_tf = entry.candles_by_tf

        # ---- 步骤 2: 资金费率 & 价差 ----
        funding_rate = self.client.get_funding_rate(inst_id)
        spread = calculate_spread(coin.get("bidPx", 0), coin.get("askPx", 0))
        entry.funding_rate = funding_rate
        entry.spread_pct = spread

        # ---- 步骤 3: FVG 扫描 ----
        signals = scan_fvg_all_timeframes(
            inst_id=inst_id,
            candles_by_tf=candles_by_tf,
            current_price=current_price,
            config=self.scan_config,
            funding_rate=funding_rate,
            spread_pct=spread,
        )
        entry.signals = signals
        entry.has_signals = len(signals) > 0

        # ---- 步骤 4: 多通道分析 ----
        candles_1h = candles_by_tf.get("1H", [])
        candles_4h = candles_by_tf.get("4H", [])

        if mc_cfg.get("enabled", True) and self.expert_engine:
            analysis = full_multi_channel_analysis(
                client=self.client,
                inst_id=inst_id,
                current_price=current_price,
                candles_1h=candles_1h,
                candles_4h=candles_4h,
                fvg_signals=signals,
                config=self.config,
                engine=self.expert_engine,
            )
            entry.analysis = analysis
            entry.has_analysis = True

            # ---- 步骤 5: 体制检测 ----
            if self.regime_detector and len(candles_1h) >= 20 and len(candles_4h) >= 20:
                ret_1h = [math.log(candles_1h[i].close / candles_1h[i-1].close)
                          for i in range(1, len(candles_1h))]
                ret_4h = [math.log(candles_4h[i].close / candles_4h[i-1].close)
                          for i in range(1, len(candles_4h))]
                trend_1h_sign = float(np.sign(np.mean(ret_1h))) if ret_1h else 0.0
                trend_4h_sign = float(np.sign(np.mean(ret_4h))) if ret_4h else 0.0
                min_len = min(len(ret_1h), len(ret_4h))
                corr = 0.0
                if min_len > 1:
                    try:
                        corr = float(np.corrcoef(ret_1h[:min_len], ret_4h[:min_len])[0, 1])
                    except Exception:
                        corr = 0.0
                regime_state = self.regime_detector.update(
                    correlation=corr,
                    trend_1h_sign=trend_1h_sign,
                    trend_4h_sign=trend_4h_sign,
                )
                entry.detected_regime = regime_state.value

            # ---- 步骤 6: 多空辩论 ----
            if debate_cfg.get("enabled", True) and self.debate_engine:
                debate_result = self.debate_engine.conduct_debate(
                    symbol=inst_id,
                    channel_reports=analysis.channels,
                    fvg_signals=signals,
                    current_price=current_price,
                    regime=entry.detected_regime,
                )
                # 辩论结果影响分析置信度
                if debate_result.winner == "tie":
                    analysis.final_confidence *= 0.8
                elif debate_result.winner == "bullish" and analysis.final_score > 0:
                    analysis.final_confidence *= 1.1
                elif debate_result.winner == "bearish" and analysis.final_score < 0:
                    analysis.final_confidence *= 1.1
                analysis.final_confidence = min(0.95, analysis.final_confidence)

        # ---- 标记研究完成 ----
        entry.researched_at = time.time()
        entry.error_count = 0

        # 存入缓存
        self.cache.put(inst_id, entry)

        # 如果有信号，打印摘要
        if entry.has_signals and entry.has_analysis:
            a = entry.analysis
            logger.debug(
                f"[CoinTracker] {inst_id}: "
                f"score={a.final_score:+.2f} conf={a.final_confidence:.0%} "
                f"regime={entry.detected_regime} "
                f"signals={len(entry.signals)}"
            )


# ---------------------------------------------------------------------------
# 批量预热：启动时快速研究 Top N 币种
# ---------------------------------------------------------------------------

def warmup_research(
    client,
    config: dict,
    cache: CoinResearchCache,
    scan_config: Optional[dict] = None,
    expert_engine: Optional[MasterTraderEngine] = None,
    regime_detector=None,
    debate_engine=None,
    top_n: int = 10,
) -> int:
    """启动时对 Top N 币种进行预热研究。

    在主循环启动前调用，确保第一批扫描有缓存可用。

    Returns:
        成功研究的币种数
    """
    logger.info(f"[Warmup] 预热研究 Top {top_n} 币种...")
    coins = get_tradable_coins(client, config)[:top_n]

    mc_cfg = config.get("multi_channel", {})
    debate_cfg = config.get("debate_engine", {})
    success = 0
    _scan_cfg = scan_config if scan_config is not None else config

    for coin in coins:
        inst_id = coin["instId"]
        try:
            # 获取 K 线
            candles_by_tf: Dict[str, List[Candle]] = {}
            for tf in config["strategy"]["timeframes"]:
                raw = client.get_candles(inst_id, bar=tf, limit=200)
                if raw:
                    candles_by_tf[tf] = candles_from_raw(raw)
            if not candles_by_tf:
                continue

            current_price = coin["last"]
            funding_rate = client.get_funding_rate(inst_id)
            spread = calculate_spread(coin.get("bidPx", 0), coin.get("askPx", 0))

            # FVG 扫描
            signals = scan_fvg_all_timeframes(
                inst_id=inst_id,
                candles_by_tf=candles_by_tf,
                current_price=current_price,
                config=_scan_cfg,
                funding_rate=funding_rate,
                spread_pct=spread,
            )

            # 多通道分析
            analysis = None
            if mc_cfg.get("enabled", True) and expert_engine and signals:
                analysis = full_multi_channel_analysis(
                    client=client,
                    inst_id=inst_id,
                    current_price=current_price,
                    candles_1h=candles_by_tf.get("1H", []),
                    candles_4h=candles_by_tf.get("4H", []),
                    fvg_signals=signals,
                    config=config,
                    engine=expert_engine,
                )

            # 体制检测
            detected_regime = "NEUTRAL"
            if regime_detector:
                c1h = candles_by_tf.get("1H", [])
                c4h = candles_by_tf.get("4H", [])
                if len(c1h) >= 20 and len(c4h) >= 20:
                    ret_1h = [math.log(c1h[i].close / c1h[i-1].close)
                              for i in range(1, len(c1h))]
                    ret_4h = [math.log(c4h[i].close / c4h[i-1].close)
                              for i in range(1, len(c4h))]
                    trend_1h_sign = float(np.sign(np.mean(ret_1h))) if ret_1h else 0.0
                    trend_4h_sign = float(np.sign(np.mean(ret_4h))) if ret_4h else 0.0
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

            # 辩论
            if debate_cfg.get("enabled", True) and debate_engine and analysis:
                debate_result = debate_engine.conduct_debate(
                    symbol=inst_id,
                    channel_reports=analysis.channels,
                    fvg_signals=signals,
                    current_price=current_price,
                    regime=detected_regime,
                )
                if debate_result.winner == "tie":
                    analysis.final_confidence *= 0.8
                elif debate_result.winner == "bullish" and analysis.final_score > 0:
                    analysis.final_confidence *= 1.1
                elif debate_result.winner == "bearish" and analysis.final_score < 0:
                    analysis.final_confidence *= 1.1
                analysis.final_confidence = min(0.95, analysis.final_confidence)

            # 存入缓存
            entry = CoinResearchEntry(
                inst_id=inst_id,
                current_price=current_price,
                bid_px=coin.get("bidPx", 0),
                ask_px=coin.get("askPx", 0),
                vol24h=coin.get("vol24h", 0),
                candles_by_tf=candles_by_tf,
                signals=signals,
                analysis=analysis,
                detected_regime=detected_regime,
                funding_rate=funding_rate,
                spread_pct=spread,
                researched_at=time.time(),
                candle_fetched_at=time.time(),
                has_signals=len(signals) > 0,
                has_analysis=analysis is not None,
            )
            cache.put(inst_id, entry)
            success += 1

            if analysis and analysis.final_confidence >= 0.40:
                logger.info(f"[Warmup] {inst_id}: "
                            f"score={analysis.final_score:+.2f} "
                            f"conf={analysis.final_confidence:.0%} "
                            f"regime={detected_regime}")

        except Exception as e:
            logger.warning(f"[Warmup] {inst_id} 失败: {e}")

    logger.info(f"[Warmup] 完成: {success}/{len(coins)} 个币种已预热")
    return success