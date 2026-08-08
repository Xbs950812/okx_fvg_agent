"""
记忆与反思系统 — 融合 TradingAgents 决策日志 + Vibe-Trading 记忆生命周期。

借鉴来源：
  - TradingAgents (46k ⭐): 决策日志 + 反思学习 + 跨品种经验迁移
  - Vibe-Trading (25k ⭐): 记忆生命周期 (质量评分 + 衰减 + 归档)

核心能力：
  - 每笔交易记录决策日志
  - 定期反思：什么有效、什么无效
  - 跨品种经验迁移
  - 记忆质量评分 + 衰减
  - 体制记忆（不同市场体制下的最佳参数）
"""

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

import numpy as np


logger = logging.getLogger(__name__)


def _write_atomic_json(path: str, data: dict) -> None:
    """原子写 JSON（.tmp + fsync + os.replace），防崩溃损坏记忆文件。"""
    try:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")


# ===========================================================================
# 记忆条目
# ===========================================================================

@dataclass
class DecisionLog:
    """单次决策日志 — 借鉴 TradingAgents。"""
    timestamp: float
    symbol: str
    direction: str              # "long" | "short" | "neutral"
    entry_price: float
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_win: bool = False
    exit_reason: str = ""       # tp / sl / manual / signal

    # 入场时的分析快照
    master_score: float = 0.0
    fvg_score: float = 0.0
    fvg_timeframe: str = ""
    channel_scores: Dict[str, float] = field(default_factory=dict)
    regime: str = ""            # 市场体制
    red_flags: List[str] = field(default_factory=list)
    expert_verdict: str = ""    # 入场时的专家结论

    # 反思
    reflection: str = ""        # 事后反思
    lessons: List[str] = field(default_factory=list)


@dataclass
class MemoryEntry:
    """记忆条目 — 带质量评分和衰减。

    借鉴 Vibe-Trading 记忆生命周期：
      - quality_score: 记忆质量 (0-1)
      - access_count: 访问次数
      - last_accessed: 最后访问时间
      - decay_factor: 衰减因子 (基于 Ebbinghaus 遗忘曲线)
    """
    key: str                    # 唯一标识
    category: str               # "regime" | "lesson" | "pattern" | "stat"
    content: Any
    created_at: float
    quality_score: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    decay_factor: float = 1.0   # 1.0 = 新鲜, 0.0 = 完全遗忘

    def access(self):
        self.access_count += 1
        self.last_accessed = time.time()
        # 访问重置记忆为新鲜状态（Ebbinghaus 遗忘曲线）
        self.decay_factor = 1.0

    def decay(self, half_life_days: float = 30.0):
        """Ebbinghaus 遗忘曲线衰减。"""
        # M-18: 修复未访问记忆的衰减计算，统一使用 0.5 ** (days / half_life)
        if self.last_accessed == 0.0:
            # 从未访问 — 使用创建时间计算衰减
            days_since = (datetime.now(timezone.utc) - datetime.fromtimestamp(self.created_at, tz=timezone.utc)).days
            self.decay_factor = 0.5 ** (days_since / half_life_days)
        else:
            days_since = (datetime.now(timezone.utc) - datetime.fromtimestamp(self.last_accessed, tz=timezone.utc)).days
            self.decay_factor = 0.5 ** (days_since / half_life_days)


# ===========================================================================
# 记忆管理器
# ===========================================================================

class MemoryManager:
    """记忆管理器 — 融合 TradingAgents 决策日志 + Vibe-Trading 记忆生命周期。

    文件结构:
      memory/
        decision_log.jsonl    # 决策日志（追加模式）
        regime_memory.json    # 体制记忆
        lessons.json          # 经验教训
        reflection.md         # 定期反思报告
    """

    def __init__(self, memory_dir: str = "memory",
                 half_life_days: float = 30.0,
                 archive_threshold: float = 0.1):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        # 修复 2026-08-07: 记忆生命周期参数生效 — 原构造只收 memory_dir,
        # decay_all/prune_archive 永远用默认值, config 的 half_life_days/
        # archive_threshold 配置未生效。
        self.half_life_days: float = half_life_days
        self.archive_threshold: float = archive_threshold

        self.decision_log_path = os.path.join(memory_dir, "decision_log.jsonl")
        self.regime_memory_path = os.path.join(memory_dir, "regime_memory.json")
        self.lessons_path = os.path.join(memory_dir, "lessons.json")
        self.entries_path = os.path.join(memory_dir, "entries.json")
        self.reflection_path = os.path.join(memory_dir, "reflection.md")

        self._entries: Dict[str, MemoryEntry] = {}
        self._regime_memory: Dict[str, Dict] = {}
        self._lessons: List[Dict] = []

        self._load()

    def _load(self):
        """加载持久化记忆。"""
        # 加载体制记忆
        if os.path.exists(self.regime_memory_path):
            try:
                with open(self.regime_memory_path, "r", encoding="utf-8") as f:
                    self._regime_memory = json.load(f)
            except Exception:
                pass

        # 加载经验教训
        if os.path.exists(self.lessons_path):
            try:
                with open(self.lessons_path, "r", encoding="utf-8") as f:
                    self._lessons = json.load(f)
            except Exception:
                pass

        # 加载记忆条目
        if os.path.exists(self.entries_path):
            try:
                with open(self.entries_path, "r", encoding="utf-8") as f:
                    entries_data = json.load(f)
                for key, data in entries_data.items():
                    entry = MemoryEntry(
                        key=data.get("key", key),
                        category=data.get("category", "general"),
                        content=data.get("content"),
                        created_at=data.get("created_at", time.time()),
                        quality_score=data.get("quality_score", 0.5),
                        access_count=data.get("access_count", 0),
                        last_accessed=data.get("last_accessed", 0.0),
                        decay_factor=data.get("decay_factor", 1.0),
                    )
                    self._entries[key] = entry
            except Exception:
                pass

    def save(self):
        """持久化所有记忆。

        修复 P2-9: 此前直接 json.dump 到目标文件（非原子写），崩溃可能损坏
        整个记忆文件；改为 .tmp + fsync + os.replace 原子写。
        """
        _write_atomic_json(self.regime_memory_path, self._regime_memory)
        _write_atomic_json(self.lessons_path, self._lessons)

        try:
            entries_data = {
                k: {
                    "key": e.key,
                    "category": e.category,
                    "content": e.content,
                    "created_at": e.created_at,
                    "quality_score": e.quality_score,
                    "access_count": e.access_count,
                    "last_accessed": e.last_accessed,
                    "decay_factor": e.decay_factor,
                }
                for k, e in self._entries.items()
            }
            _write_atomic_json(self.entries_path, entries_data)
        except Exception as e:
            logger.error(f"Failed to save entries: {e}")

    # ------------------------------------------------------------------
    # 决策日志
    # ------------------------------------------------------------------

    def log_decision(self, log: DecisionLog):
        """追加决策日志。"""
        try:
            with open(self.decision_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(log), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")

    def get_recent_decisions(self, n: int = 20) -> List[DecisionLog]:
        """获取最近 N 条决策。"""
        decisions = deque(maxlen=n)
        try:
            if os.path.exists(self.decision_log_path):
                with open(self.decision_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            decisions.append(DecisionLog(**json.loads(line.strip())))
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"Failed to read decision log: {e}")
        return list(decisions)

    # ------------------------------------------------------------------
    # 反思引擎（借鉴 TradingAgents）
    # ------------------------------------------------------------------

    def generate_reflection(self, recent_trades: List[DecisionLog]) -> str:
        """生成反思报告 — 分析最近的交易，提炼经验教训。

        借鉴 TradingAgents 的决策后反思机制：
          - 什么做对了？
          - 什么做错了？
          - 市场体制是否发生了变化？
          - 参数是否需要调整？
        """
        if not recent_trades:
            return "暂无交易记录，无可反思。\n"

        # 修复 Bug 53: 反思报告胜率统一以 pnl 符号为准，保本交易单独显示
        wins = [t for t in recent_trades if t.pnl > 0]
        losses = [t for t in recent_trades if t.pnl < 0]
        break_evens = [t for t in recent_trades if t.pnl == 0]
        total = len(recent_trades)
        decisive = len(wins) + len(losses)

        lines = [
            f"# 交易反思报告",
            f"生成时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"分析范围: 最近 {total} 笔交易",
            "",
            f"## 总体表现",
            f"- 总交易: {total} 笔 (W:{len(wins)} L:{len(losses)} BE:{len(break_evens)})",
            f"- 胜率: {len(wins)/decisive*100:.1f}% (BE不计入)" if decisive > 0 else "- 胜率: N/A",
            f"- 总盈亏: {sum(t.pnl for t in recent_trades):+.2f} USDT",
            f"- 平均盈亏: {np.mean([t.pnl for t in recent_trades]):+.2f} USDT" if recent_trades else "",
            "",
        ]

        # 盈亏比 (profit factor)
        if wins and losses:
            avg_win = np.mean([abs(t.pnl) for t in wins])
            avg_loss = np.mean([abs(t.pnl) for t in losses])
            if avg_loss <= 0:
                avg_loss = 0.01  # 防止除零
            profit_factor = avg_win / avg_loss
            lines.append(f"- 盈亏比 (Profit Factor): {profit_factor:.2f}")
            lines.append("")

        # 分析盈利交易
        if wins:
            lines.append("## 盈利交易分析")
            avg_score = np.mean([t.master_score for t in wins])
            lines.append(f"- 平均入场评分: {avg_score:.2f}")
            # 按体制分组
            regime_wins = {}
            for t in wins:
                r = t.regime or "unknown"
                regime_wins[r] = regime_wins.get(r, 0) + 1
            lines.append(f"- 盈利体制分布: {regime_wins}")
            # 常见止盈原因
            reasons = {}
            for t in wins:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            lines.append(f"- 止盈原因: {reasons}")
            lines.append("")

        # 分析亏损交易
        if losses:
            lines.append("## 亏损交易分析")
            avg_score = np.mean([t.master_score for t in losses])
            lines.append(f"- 平均入场评分: {avg_score:.2f}")
            regime_losses = {}
            for t in losses:
                r = t.regime or "unknown"
                regime_losses[r] = regime_losses.get(r, 0) + 1
            lines.append(f"- 亏损体制分布: {regime_losses}")
            reasons = {}
            for t in losses:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            lines.append(f"- 止损原因: {reasons}")

            # 红旗分析
            red_flags_count = {}
            for t in losses:
                for rf in t.red_flags:
                    red_flags_count[rf] = red_flags_count.get(rf, 0) + 1
            if red_flags_count:
                top_reds = sorted(red_flags_count.items(), key=lambda x: x[1], reverse=True)[:3]
                lines.append(f"- 最常被忽略的红旗: {top_reds}")
            lines.append("")

        # 经验教训
        lines.append("## 经验教训")
        if losses:
            if any("资金费率" in rf or "funding" in rf.lower()
                   for t in losses for rf in t.red_flags):
                lines.append("- ⚠ 资金费率异常时仍强行入场，导致亏损。极高资金费率是反向信号。")
            if any("多周期" in rf or "一致" in rf
                   for t in losses for rf in t.red_flags):
                lines.append("- ⚠ 多周期趋势不一致时入场，需等待共振确认。")
            if any("价差" in rf or "spread" in rf.lower()
                   for t in losses for rf in t.red_flags):
                lines.append("- ⚠ 价差过大时入场，滑点吃掉利润。")

        if wins and losses:
            win_score = np.nanmean([t.master_score for t in wins])
            loss_score = np.nanmean([t.master_score for t in losses])
            if np.isnan(loss_score):
                loss_score = 0.0
            if np.isnan(win_score):
                win_score = 0.0
            # L-18: 综合评分 NaN 防护
            score = np.nanmean([win_score, loss_score])
            if np.isnan(score):
                score = 0.5
            if win_score > loss_score:
                lines.append(f"- ✓ 高评分信号 (平均 {win_score:.2f}) 确实更优，低评分 (平均 {loss_score:.2f}) 需更谨慎")
            # 经验规则: 建议最低入场评分上调至略高于亏损平均评分
            # 注: +0.05 为工程经验值，非数学推导；可根据回测结果调整
            lines.append(f"- 📊 建议最低入场评分上调至 {max(0.5, loss_score + 0.05):.2f}")

        # 反思建议
        lines.append("")
        lines.append("## 调整建议")
        if len(losses) >= 3:
            lines.append("- 连亏后应降低杠杆或暂停交易")
        if total >= 10:
            lines.append("- 积累足够数据，可运行参数优化")
        lines.append("- 关注市场体制变化，不同体制适用不同策略参数")

        reflection = "\n".join(lines)

        # 保存反思报告
        try:
            with open(self.reflection_path, "w", encoding="utf-8") as f:
                f.write(reflection)
        except Exception as e:
            logger.error(f"Failed to save reflection: {e}")

        return reflection

    # ------------------------------------------------------------------
    # 体制记忆（借鉴 Vibe-Trading 相关性体制检测）
    # ------------------------------------------------------------------

    def update_regime_memory(self, regime: str, params: Dict[str, float]):
        """更新体制记忆。"""
        if regime not in self._regime_memory:
            self._regime_memory[regime] = {
                "count": 0,
                "best_params": {},
                "avg_win_rate": 0.0,
                "avg_pnl": 0.0,
                "last_seen": time.time(),
            }

        entry = self._regime_memory[regime]
        entry["count"] += 1
        entry["last_seen"] = time.time()

        # 更新最优参数
        for k, v in params.items():
            if k not in entry["best_params"]:
                entry["best_params"][k] = v
            else:
                entry["best_params"][k] = entry["best_params"][k] * 0.7 + v * 0.3

    def get_regime_params(self, regime: str) -> Optional[Dict]:
        """获取指定体制的历史最优参数。"""
        return self._regime_memory.get(regime, {}).get("best_params")

    def get_regime_stats(self, regime: str) -> Optional[Dict]:
        """获取指定体制的统计信息。"""
        return self._regime_memory.get(regime)

    # ------------------------------------------------------------------
    # 经验教训
    # ------------------------------------------------------------------

    def add_lesson(self, lesson: str, category: str = "general", importance: float = 0.5):
        """添加经验教训。"""
        self._lessons.append({
            "lesson": lesson,
            "category": category,
            "importance": importance,
            "timestamp": time.time(),
        })
        # 按重要性排序，保留前 100 条
        self._lessons.sort(key=lambda x: x["importance"], reverse=True)
        self._lessons = self._lessons[:100]

    def get_lessons(self, category: Optional[str] = None, top_n: int = 10) -> List[Dict]:
        """获取经验教训。"""
        if category:
            filtered = [l for l in self._lessons if l["category"] == category]
        else:
            filtered = self._lessons
        filtered.sort(key=lambda x: x["importance"], reverse=True)
        return filtered[:top_n]

    # ------------------------------------------------------------------
    # 跨品种经验迁移
    # ------------------------------------------------------------------

    def get_cross_symbol_insight(
        self,
        symbol: str,
        current_regime: Optional[str] = None,
    ) -> Optional[str]:
        """获取跨品种经验迁移。

        修复 Bug 41: 优先使用调用方传入的 current_regime；
        若未传入，则回退到当前币种（symbol）历史最近一个有记录的 regime，
        避免用其他币种的最后一笔 regime 错配。
        """
        decisions = self.get_recent_decisions(50)
        if not decisions:
            return None

        regime = current_regime
        if not regime:
            # 只取当前币种历史决策中的最后一个 regime
            for d in reversed(decisions):
                if d.symbol == symbol and d.regime:
                    regime = d.regime
                    break

        if not regime:
            return None

        same_regime = [d for d in decisions
                       if d.regime == regime and d.symbol != symbol]
        if not same_regime:
            return None

        # 胜率以 pnl 符号为准，保本不计入分母，保持与 StateManager 一致
        wins = sum(1 for d in same_regime if d.pnl > 0)
        losses = sum(1 for d in same_regime if d.pnl < 0)
        decisive = wins + losses
        if decisive == 0:
            return None
        win_rate = wins / decisive
        return (f"当前体制 ({regime}) 下，"
                f"其他品种胜率 {win_rate:.0%} ({wins}W/{losses}L/{len(same_regime)}笔)")

    # ------------------------------------------------------------------
    # 记忆生命周期（借鉴 Vibe-Trading）
    # ------------------------------------------------------------------

    def store_entry(self, key: str, category: str, content: Any,
                    quality: float = 0.5):
        """存储记忆条目。"""
        entry = MemoryEntry(
            key=key,
            category=category,
            content=content,
            created_at=time.time(),
            quality_score=quality,
            last_accessed=time.time(),
        )
        self._entries[key] = entry

    def retrieve_entry(self, key: str) -> Optional[MemoryEntry]:
        """检索记忆条目并更新访问计数。"""
        entry = self._entries.get(key)
        if entry:
            entry.access()
        return entry

    def decay_all(self, half_life_days: Optional[float] = None):
        """对所有记忆条目执行衰减。half_life_days=None 用构造参数。"""
        _hl = half_life_days if half_life_days is not None else self.half_life_days
        for entry in self._entries.values():
            entry.decay(_hl)

    def prune_archive(self, threshold: Optional[float] = None):
        """归档低质量记忆（质量评分 < 阈值）。threshold=None 用构造参数。"""
        _th = threshold if threshold is not None else self.archive_threshold
        to_remove = [k for k, e in self._entries.items()
                     if e.quality_score < _th and e.decay_factor < _th]
        for k in to_remove:
            logger.debug(f"Archiving memory: {k}")
            del self._entries[k]
        return len(to_remove)

    def maintain_lifecycle(self):
        """周期维护: 衰减 + 归档 + 持久化（修复 2026-08-07: 原 decay_all/
        prune_archive 从未被调用, 记忆生命周期功能未接线）。"""
        self.decay_all()
        _archived = self.prune_archive()
        if _archived:
            logger.info(f"[Memory] 归档 {_archived} 条低质量记忆")
        self.save()


# ===========================================================================
# 相关性体制检测（借鉴 Vibe-Trading）
# ===========================================================================

def detect_correlation_regime(
    returns_1h: List[float],
    returns_4h: List[float],
    window: int = 20,
) -> str:
    """检测市场相关性体制。

    借鉴 Vibe-Trading 的 correlation-regime 技能 (PR #557, #756)：
      - FUSED (融合): 高相关性 + 同向 → 市场共振，分散化失效
      - DIVERGENT (背离): 低相关性 + 反向 → 分歧，信号可靠性降低
      - NEUTRAL (中性): 中等相关性

    行业标准做法 (Freqtrade, Vibe-Trading):
      将高频数据聚合到低频，确保两序列覆盖相同时间范围后再计算 Pearson 相关系数。
      1H→4H 聚合: 每 4 根 1H 对数收益率求和 = 1 根 4H 等效收益率。
    """
    if len(returns_1h) < window or len(returns_4h) < window:
        return "NEUTRAL"

    # 修复 M-1: 时间对齐 — 将 1H 收益率聚合为 4H 等效收益率
    # 每 4 根 1H 对应 1 根 4H，对数收益率可加
    n_4h = min(len(returns_4h), len(returns_1h) // 4, window)
    if n_4h < 3:
        return "NEUTRAL"

    r4 = returns_4h[-n_4h:]
    # 1H → 4H 聚合（对数收益率直接求和）
    r1_agg = [
        sum(returns_1h[-(i + 1) * 4: -i * 4] if i > 0 else returns_1h[-4:])
        for i in range(n_4h)
    ][::-1]  # 反转回正序，使最旧在前、最新在后

    # Pearson 相关系数
    correlation = np.corrcoef(r1_agg, r4)[0, 1] if len(r1_agg) > 1 else 0
    if np.isnan(correlation):
        correlation = 0.0

    # 趋势方向（基于聚合后的等时间窗口）
    trend_1h = np.sign(np.mean(r1_agg))
    trend_4h = np.sign(np.mean(r4))

    if abs(correlation) > 0.7 and trend_1h == trend_4h:
        return "FUSED"      # 共振 — 趋势确定的信号更可靠，但分散化失效
    elif abs(correlation) < 0.3:
        return "DIVERGENT"  # 背离 — 信号可靠性降低
    elif trend_1h != trend_4h and abs(correlation) < 0.5:
        return "DIVERGENT"
    else:
        return "NEUTRAL"


# ===========================================================================
# 多空辩论引擎已迁移至 debate_engine.py (TradingAgentsDebateEngine)
# 此处保留空区域以维护模块结构一致性
# ===========================================================================