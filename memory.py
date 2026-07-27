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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import numpy as np


logger = logging.getLogger(__name__)


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
        # 访问强化记忆
        self.decay_factor = min(1.0, self.decay_factor + 0.05)

    def decay(self, half_life_days: float = 30.0):
        """Ebbinghaus 遗忘曲线衰减。"""
        elapsed_days = (time.time() - self.last_accessed) / 86400.0
        if elapsed_days > 0:
            self.decay_factor = math.exp(-math.log(2) * elapsed_days / half_life_days)


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

    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        self.decision_log_path = os.path.join(memory_dir, "decision_log.jsonl")
        self.regime_memory_path = os.path.join(memory_dir, "regime_memory.json")
        self.lessons_path = os.path.join(memory_dir, "lessons.json")
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

    def save(self):
        """持久化所有记忆。"""
        try:
            with open(self.regime_memory_path, "w", encoding="utf-8") as f:
                json.dump(self._regime_memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save regime memory: {e}")

        try:
            with open(self.lessons_path, "w", encoding="utf-8") as f:
                json.dump(self._lessons, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save lessons: {e}")

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
        decisions = []
        try:
            if os.path.exists(self.decision_log_path):
                with open(self.decision_log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines[-n:]:
                    try:
                        decisions.append(DecisionLog(**json.loads(line.strip())))
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Failed to read decision log: {e}")
        return decisions

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

        wins = [t for t in recent_trades if t.is_win]
        losses = [t for t in recent_trades if not t.is_win]
        total = len(recent_trades)

        lines = [
            f"# 交易反思报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"分析范围: 最近 {total} 笔交易",
            "",
            f"## 总体表现",
            f"- 总交易: {total} 笔",
            f"- 胜率: {len(wins)/total*100:.1f}% ({len(wins)}W/{len(losses)}L)" if total > 0 else "",
            f"- 总盈亏: {sum(t.pnl for t in recent_trades):+.2f} USDT",
            f"- 平均盈亏: {np.mean([t.pnl for t in recent_trades]):+.2f} USDT" if recent_trades else "",
            "",
        ]

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
            win_score = np.mean([t.master_score for t in wins])
            loss_score = np.mean([t.master_score for t in losses])
            if win_score > loss_score:
                lines.append(f"- ✓ 高评分信号 (平均 {win_score:.2f}) 确实更优，低评分 (平均 {loss_score:.2f}) 需更谨慎")
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
        # 只保留最近 100 条
        self._lessons = self._lessons[-100:]

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

    def get_cross_symbol_insight(self, symbol: str) -> Optional[str]:
        """获取跨品种经验迁移。

        如果某个品种在类似体制下表现良好，提供参考。
        """
        decisions = self.get_recent_decisions(50)
        if not decisions:
            return None

        # 寻找同方向同体制的其他品种表现
        current_regime = ""
        for d in reversed(decisions):
            if d.regime:
                current_regime = d.regime
                break

        if not current_regime:
            return None

        same_regime = [d for d in decisions
                       if d.regime == current_regime and d.symbol != symbol]
        if not same_regime:
            return None

        win_rate = sum(1 for d in same_regime if d.is_win) / len(same_regime)
        return (f"当前体制 ({current_regime}) 下，"
                f"其他品种胜率 {win_rate:.0%} ({len(same_regime)} 笔)")

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

    def decay_all(self, half_life_days: float = 30.0):
        """对所有记忆条目执行衰减。"""
        for entry in self._entries.values():
            entry.decay(half_life_days)

    def prune_archive(self, threshold: float = 0.1):
        """归档低质量记忆（质量评分 < 阈值）。"""
        to_remove = [k for k, e in self._entries.items()
                     if e.quality_score < threshold and e.decay_factor < threshold]
        for k in to_remove:
            logger.debug(f"Archiving memory: {k}")
            del self._entries[k]
        return len(to_remove)


# ===========================================================================
# 相关性体制检测（借鉴 Vibe-Trading）
# ===========================================================================

def detect_correlation_regime(
    returns_1h: List[float],
    returns_4h: List[float],
    window: int = 20,
) -> str:
    """检测市场相关性体制。

    借鉴 Vibe-Trading 的 correlation-regime 技能：
      - FUSED (融合): 高相关性 + 同向 → 市场共振
      - DIVERGENT (背离): 低相关性 + 反向 → 分歧
      - NEUTRAL (中性): 中等相关性

    简化版：通过 1H 和 4H 收益率的滚动相关性来判断。
    """
    if len(returns_1h) < window or len(returns_4h) < window:
        return "NEUTRAL"

    # 对齐长度
    min_len = min(len(returns_1h), len(returns_4h), window)
    r1 = returns_1h[-min_len:]
    r4 = returns_4h[-min_len:]

    if len(r1) < 3:
        return "NEUTRAL"

    # 滚动相关性
    correlation = np.corrcoef(r1, r4)[0, 1] if len(r1) > 1 else 0

    # 趋势方向
    trend_1h = np.sign(np.mean(r1))
    trend_4h = np.sign(np.mean(r4))

    if abs(correlation) > 0.7 and trend_1h == trend_4h:
        return "FUSED"      # 共振 — 趋势确定的信号更可靠
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