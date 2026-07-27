"""
多空辩论引擎 — 借鉴 TradingAgents (86k ⭐) 的多 Agent 辩论架构。

TradingAgents 核心设计：
  - Analyst Team: 基本面 / 情绪 / 新闻 / 技术分析师
  - Researcher Team: 多头研究员 + 空头研究员，结构化辩论
  - Trader Agent: 综合研判 + 下单决策
  - Risk Management + Portfolio Manager: 风控 + 组合管理

本模块实现：
  1. Multi-Agent Analyst Team — 模拟分析师团队的多维度研判
  2. Structured Debate Protocol — 多轮结构化辩论 + 交叉质询
  3. Analyst Reputation System — 分析师信誉评分，动态调整权重
  4. Checkpoint Resume — 辩论状态持久化，支持中断恢复
  5. Decision Log Injection — 历史决策反思注入未来研判
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any, Set

import numpy as np


logger = logging.getLogger(__name__)


# ===========================================================================
# 分析师角色定义
# ===========================================================================

@dataclass
class AnalystProfile:
    """分析师档案 — 模拟 TradingAgents 的专业分析师团队。"""
    name: str                       # 分析师名称
    role: str                       # 角色：technical / fundamental / sentiment / macro / risk
    expertise: List[str]            # 专长领域
    bias: str = "neutral"           # 偏向：bullish / bearish / neutral
    reputation: float = 0.5         # 信誉评分 (0-1)
    total_analyses: int = 0         # 总分析次数
    correct_calls: int = 0          # 正确预测次数
    weight: float = 0.20            # 当前权重


# 预设分析师团队
DEFAULT_ANALYST_TEAM = [
    AnalystProfile(
        name="技术分析师-A",
        role="technical",
        expertise=["FVG", "趋势线", "支撑阻力", "多周期共振", "成交量分析"],
        bias="neutral",
        weight=0.25,
    ),
    AnalystProfile(
        name="市场结构分析师",
        role="structure",
        expertise=["订单簿深度", "流动性墙", "买卖失衡", "价差分析"],
        bias="neutral",
        weight=0.20,
    ),
    AnalystProfile(
        name="资金流分析师",
        role="capital_flow",
        expertise=["资金费率", "OI变化", "主动量", "大单追踪"],
        bias="neutral",
        weight=0.20,
    ),
    AnalystProfile(
        name="情绪分析师",
        role="sentiment",
        expertise=["多空比", "恐慌贪婪", "社交情绪", "持仓分布"],
        bias="contrarian",  # 反向指标
        weight=0.15,
    ),
    AnalystProfile(
        name="宏观分析师",
        role="macro",
        expertise=["BTC主导", "板块轮动", "相关性", "宏观经济"],
        bias="neutral",
        weight=0.10,
    ),
    AnalystProfile(
        name="风控官",
        role="risk",
        expertise=["回撤评估", "波动率", "VaR", "敞口管理"],
        bias="conservative",
        weight=0.10,
    ),
]


# ===========================================================================
# 辩论数据类
# ===========================================================================

@dataclass
class AnalystOpinion:
    """单个分析师的观点。"""
    analyst_name: str
    role: str
    direction: str                 # "bullish" | "bearish" | "neutral"
    conviction: float              # 确信度 (0-1)
    reasoning: List[str]           # 推理要点
    evidence: List[str]            # 支撑证据
    red_flags: List[str]           # 风险提示
    score: float = 0.0             # 综合评分 (-1 ~ +1)
    weight: float = 0.20           # 分析师权重


@dataclass
class DebateRound:
    """单轮辩论记录。"""
    round_number: int
    bullish_arguments: List[str]
    bearish_arguments: List[str]
    cross_examination: List[str]   # 交叉质询
    round_score: float             # 本轮评分 (-1 ~ +1, 正=多方占优)
    key_insight: str = ""          # 本轮关键洞察


@dataclass
class DebateResult:
    """完整辩论结果。"""
    symbol: str
    timestamp: float

    # 分析师意见
    opinions: List[AnalystOpinion] = field(default_factory=list)

    # 辩论过程
    rounds: List[DebateRound] = field(default_factory=list)
    total_rounds: int = 0

    # 最终结论
    winner: str = ""               # "bullish" | "bearish" | "tie"
    final_score: float = 0.0       # 最终综合评分 (-1 ~ +1)
    confidence: float = 0.0         # 综合置信度

    # 关键分歧
    disagreements: List[str] = field(default_factory=list)
    consensus_points: List[str] = field(default_factory=list)

    # 元数据
    verdict: str = ""              # 最终研判文本
    risk_assessment: str = ""      # 风险评估
    action_recommendation: str = ""  # 行动建议


# ===========================================================================
# 辩论引擎
# ===========================================================================

class TradingAgentsDebateEngine:
    """多空辩论引擎 — 模拟 TradingAgents 的多 Agent 辩论流程。

    流程：
      1. 分析师独立研判 → 各自给出观点
      2. 多头研究员汇总看多理由
      3. 空头研究员汇总看空理由
      4. 结构化辩论 (多轮交叉)
      5. 风控官评估风险
      6. 综合研判 → 输出结论
    """

    def __init__(
        self,
        analysts: Optional[List[AnalystProfile]] = None,
        debate_rounds: int = 2,
        min_agreement: float = 0.50,
        checkpoint_dir: Optional[str] = None,
    ):
        self.analysts = analysts or DEFAULT_ANALYST_TEAM
        self.debate_rounds = debate_rounds
        self.min_agreement = min_agreement
        self.checkpoint_dir = checkpoint_dir

        # 分析师索引
        self._analyst_index: Dict[str, AnalystProfile] = {
            a.name: a for a in self.analysts
        }

    def conduct_debate(
        self,
        symbol: str,
        channel_reports: List[Any],    # ChannelReport 列表
        fvg_signals: List[Any],        # Signal 列表
        current_price: float,
        regime: str = "NEUTRAL",
    ) -> DebateResult:
        """执行完整的辩论流程。

        Args:
            symbol: 交易对
            channel_reports: 五通道分析报告
            fvg_signals: FVG 信号
            current_price: 当前价格
            regime: 市场体制

        Returns:
            DebateResult 完整辩论结果
        """
        result = DebateResult(
            symbol=symbol,
            timestamp=time.time(),
        )

        # ---- Phase 1: 分析师独立研判 ----
        result.opinions = self._phase_analyst_analysis(
            symbol, channel_reports, fvg_signals, current_price, regime
        )

        # ---- Phase 2: 结构化辩论 ----
        result.rounds, result.disagreements, result.consensus_points = \
            self._phase_structured_debate(result.opinions, regime)

        result.total_rounds = len(result.rounds)

        # ---- Phase 3: 综合研判 ----
        result.final_score, result.confidence, result.winner = \
            self._phase_synthesis(result)

        # ---- Phase 4: 生成结论 ----
        result.verdict, result.risk_assessment, result.action_recommendation = \
            self._phase_verdict(result, regime)

        return result

    # ------------------------------------------------------------------
    # Phase 1: 分析师独立研判
    # ------------------------------------------------------------------

    def _phase_analyst_analysis(
        self,
        symbol: str,
        channel_reports: List[Any],
        fvg_signals: List[Any],
        current_price: float,
        regime: str,
    ) -> List[AnalystOpinion]:
        """Phase 1: 每个分析师基于自己的专长独立研判。

        模拟 TradingAgents 的 Analyst Team：
          - 技术分析师关注 FVG + 趋势
          - 市场结构分析师关注订单簿
          - 资金流分析师关注资金费率 + OI
          - 情绪分析师关注多空比 + 恐慌贪婪
          - 宏观分析师关注 BTC 主导 + 相关性
          - 风控官关注风险因素
        """
        opinions = []

        # 从 channel_reports 提取各通道数据
        ch_data = {}
        for ch in channel_reports:
            ch_data[ch.channel_name] = ch

        for analyst in self.analysts:
            opinion = self._generate_analyst_opinion(
                analyst, ch_data, fvg_signals, current_price, regime
            )
            opinions.append(opinion)

        return opinions

    def _generate_analyst_opinion(
        self,
        analyst: AnalystProfile,
        ch_data: Dict[str, Any],
        fvg_signals: List[Any],
        current_price: float,
        regime: str,
    ) -> AnalystOpinion:
        """根据分析师角色生成观点 — 基于通道数据的量化分析。"""

        reasoning = []
        evidence = []
        red_flags = []
        bullish_score = 0.0
        bearish_score = 0.0

        if analyst.role == "technical":
            # 技术分析师：关注价格行为通道
            ch = ch_data.get("价格行为")
            if ch:
                bullish_score = ch.bullish_score * 0.8
                bearish_score = ch.bearish_score * 0.8
                reasoning.extend(ch.observations[:3])
                red_flags.extend(ch.red_flags)

                # FVG 信号分析
                if fvg_signals:
                    best_fvg = fvg_signals[0]
                    reasoning.append(
                        f"最优FVG: {best_fvg.position_side.upper()} "
                        f"{best_fvg.fvg.timeframe} "
                        f"宽度={best_fvg.fvg.width_pct:.2f}% "
                        f"评分={best_fvg.score:.2f}"
                    )
                    evidence.append(f"FVG信号评分: {best_fvg.score:.2f}")

            # 补充趋势分析
            if ch and hasattr(ch, 'raw_data'):
                trend_1h = ch.raw_data.get("trend_1h", "unknown")
                trend_4h = ch.raw_data.get("trend_4h", "unknown")
                if trend_1h == trend_4h and trend_1h != "unknown":
                    evidence.append(f"1H/4H趋势共振: {trend_1h}")

        elif analyst.role == "structure":
            # 市场结构分析师
            ch = ch_data.get("市场结构")
            if ch:
                bullish_score = ch.bullish_score * 0.9
                bearish_score = ch.bearish_score * 0.9
                reasoning.extend(ch.observations[:3])
                red_flags.extend(ch.red_flags)

                if hasattr(ch, 'raw_data'):
                    imb = ch.raw_data.get("imbalance", 0)
                    if abs(imb) > 0.2:
                        evidence.append(f"订单簿失衡度: {imb:.1%}")

        elif analyst.role == "capital_flow":
            # 资金流分析师
            ch = ch_data.get("资金流向")
            if ch:
                bullish_score = ch.bullish_score * 0.85
                bearish_score = ch.bearish_score * 0.85
                reasoning.extend(ch.observations[:3])
                red_flags.extend(ch.red_flags)

                if hasattr(ch, 'raw_data'):
                    fr = ch.raw_data.get("funding_rate")
                    if fr is not None:
                        evidence.append(f"资金费率: {fr:.4%}")

        elif analyst.role == "sentiment":
            # 情绪分析师 (反向指标)
            ch = ch_data.get("市场情绪")
            if ch:
                # 情绪分析师偏向逆向思维
                if ch.bullish_score > 0.5:
                    # 过度看多 → 警惕
                    bullish_score = ch.bullish_score * 0.4
                    bearish_score = ch.bullish_score * 0.6
                    red_flags.append("市场情绪过度乐观，警惕反转")
                elif ch.bearish_score > 0.5:
                    bullish_score = ch.bearish_score * 0.6
                    bearish_score = ch.bearish_score * 0.4
                else:
                    bullish_score = ch.bullish_score * 0.7
                    bearish_score = ch.bearish_score * 0.7
                reasoning.extend(ch.observations[:3])
                red_flags.extend(ch.red_flags)

        elif analyst.role == "macro":
            # 宏观分析师
            ch = ch_data.get("宏观背景")
            if ch:
                bullish_score = ch.bullish_score * 0.7
                bearish_score = ch.bearish_score * 0.7
                reasoning.extend(ch.observations[:3])
                red_flags.extend(ch.red_flags)

        elif analyst.role == "risk":
            # 风控官：关注所有通道的红旗
            all_reds = []
            for ch_name, ch in ch_data.items():
                all_reds.extend(ch.red_flags)

            if all_reds:
                reasoning.append(f"发现 {len(all_reds)} 个风险信号")
                red_flags = all_reds[:5]
                # 风控官偏向保守
                bearish_score += 0.1 * len(all_reds)

            if regime == "DIVERGENT":
                reasoning.append("市场体制处于背离状态，风险偏高")
                red_flags.append("相关性体制背离")
                bearish_score += 0.15

            # 风控官不轻易给出方向
            bullish_score = max(0, bullish_score - 0.1)
            bearish_score = min(1.0, bearish_score)

        # 归一化
        total = bullish_score + bearish_score
        if total > 0:
            score = (bullish_score - bearish_score) / total
        else:
            score = 0.0

        # 方向判断
        if score > 0.15:
            direction = "bullish"
        elif score < -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        # 确信度
        conviction = min(0.95, abs(score) * 1.5)

        return AnalystOpinion(
            analyst_name=analyst.name,
            role=analyst.role,
            direction=direction,
            conviction=round(conviction, 3),
            reasoning=reasoning if reasoning else ["无足够数据"],
            evidence=evidence,
            red_flags=red_flags,
            score=round(score, 3),
            weight=analyst.weight,
        )

    # ------------------------------------------------------------------
    # Phase 2: 结构化辩论
    # ------------------------------------------------------------------

    def _phase_structured_debate(
        self,
        opinions: List[AnalystOpinion],
        regime: str,
    ) -> Tuple[List[DebateRound], List[str], List[str]]:
        """Phase 2: 结构化辩论。

        模拟 TradingAgents 的研究员辩论：
          - Round 1: 各方陈述核心论点
          - Round 2-N: 交叉质询 + 反驳
          - 每轮后更新评分
        """
        rounds = []
        disagreements = []
        consensus_points = []

        # 分离多空观点
        bulls = [o for o in opinions if o.direction == "bullish"]
        bears = [o for o in opinions if o.direction == "bearish"]
        neutrals = [o for o in opinions if o.direction == "neutral"]

        # Round 1: 初始陈述
        bull_args = []
        for o in bulls:
            bull_args.extend([f"[{o.analyst_name}] {r}" for r in o.reasoning[:2]])
        bear_args = []
        for o in bears:
            bear_args.extend([f"[{o.analyst_name}] {r}" for r in o.reasoning[:2]])

        # 计算 Round 1 评分
        bull_weight = sum(o.weight * o.conviction for o in bulls)
        bear_weight = sum(o.weight * o.conviction for o in bears)
        total_weight = bull_weight + bear_weight
        if total_weight > 0:
            round1_score = (bull_weight - bear_weight) / total_weight
        else:
            round1_score = 0.0

        rounds.append(DebateRound(
            round_number=1,
            bullish_arguments=bull_args if bull_args else ["无明确看多理由"],
            bearish_arguments=bear_args if bear_args else ["无明确看空理由"],
            cross_examination=[],
            round_score=round(round1_score, 3),
            key_insight=f"初始陈述: 多头{bull_weight:.2f} vs 空头{bear_weight:.2f}",
        ))

        # Round 2: 交叉质询
        cross_exam = []
        if bulls and bears:
            # 空头对多头的质疑
            for o in bulls:
                for r in o.red_flags[:1]:
                    cross_exam.append(f"对[{o.analyst_name}]的质疑: {r}")
            # 多头对空头的质疑
            for o in bears:
                for r in o.red_flags[:1]:
                    cross_exam.append(f"对[{o.analyst_name}]的质疑: {r}")

        if cross_exam:
            # 交叉质询后调整评分
            if regime == "FUSED":
                # 共振体制下，多数派得分更高
                if len(bulls) > len(bears):
                    round2_score = round1_score * 1.2
                elif len(bears) > len(bulls):
                    round2_score = round1_score * 1.2
                else:
                    round2_score = round1_score
            elif regime == "DIVERGENT":
                # 背离体制下，降低置信度
                round2_score = round1_score * 0.7
            else:
                round2_score = round1_score

            rounds.append(DebateRound(
                round_number=2,
                bullish_arguments=[f"反驳: {a}" for a in bear_args[:3]] if bulls else [],
                bearish_arguments=[f"反驳: {a}" for a in bull_args[:3]] if bears else [],
                cross_examination=cross_exam,
                round_score=round(round2_score, 3),
                key_insight=f"交叉质询后: regime={regime}, score调整={round2_score:+.2f}",
            ))

        # 识别分歧和共识
        all_reds = set()
        all_evidence = set()
        for o in opinions:
            all_reds.update(o.red_flags)
            all_evidence.update(o.evidence)

        # 分歧点：正反双方都提到的红旗
        disagreements = list(all_reds)[:5]

        # 共识点：多数分析师同意的方向
        if len(bulls) >= len(opinions) * 0.6:
            consensus_points = ["多数分析师看多"]
        elif len(bears) >= len(opinions) * 0.6:
            consensus_points = ["多数分析师看空"]
        elif len(neutrals) >= len(opinions) * 0.6:
            consensus_points = ["多数分析师认为方向不明"]
        else:
            consensus_points = ["分析师意见分歧"]

        return rounds, disagreements, consensus_points

    # ------------------------------------------------------------------
    # Phase 3: 综合研判
    # ------------------------------------------------------------------

    def _phase_synthesis(
        self,
        result: DebateResult,
    ) -> Tuple[float, float, str]:
        """Phase 3: 综合所有分析师意见和辩论结果。

        Returns:
            (final_score, confidence, winner)
        """
        # 加权综合所有分析师评分
        weighted_sum = 0.0
        total_weight = 0.0

        for o in result.opinions:
            w = o.weight * o.conviction
            weighted_sum += o.score * w
            total_weight += w

        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = 0.0

        # 辩论轮次调整
        if result.rounds:
            last_round_score = result.rounds[-1].round_score
            final_score = final_score * 0.6 + last_round_score * 0.4

        # 置信度
        # 基于分析师一致性
        directions = [o.direction for o in result.opinions]
        if directions:
            most_common = max(set(directions), key=directions.count)
            agreement_ratio = directions.count(most_common) / len(directions)
        else:
            agreement_ratio = 0.5

        # 最终置信度
        confidence = min(0.95, agreement_ratio * 0.7 + abs(final_score) * 0.3)

        # 确定胜者
        if final_score > 0.2:
            winner = "bullish"
        elif final_score < -0.2:
            winner = "bearish"
        else:
            winner = "tie"

        return round(final_score, 3), round(confidence, 3), winner

    # ------------------------------------------------------------------
    # Phase 4: 生成结论
    # ------------------------------------------------------------------

    def _phase_verdict(
        self,
        result: DebateResult,
        regime: str,
    ) -> Tuple[str, str, str]:
        """Phase 4: 生成最终研判文本。

        Returns:
            (verdict, risk_assessment, action_recommendation)
        """
        # 分析师总结
        bull_count = sum(1 for o in result.opinions if o.direction == "bullish")
        bear_count = sum(1 for o in result.opinions if o.direction == "bearish")
        neutral_count = sum(1 for o in result.opinions if o.direction == "neutral")

        # 最终研判
        if result.winner == "bullish":
            if result.confidence > 0.7:
                verdict = (
                    f"【辩论结论】多方证据链充分，{bull_count}/{len(result.opinions)} 位分析师看多。"
                    f"辩论评分 {result.final_score:+.2f}，置信度 {result.confidence:.0%}。"
                    f"多轮辩论后多方论点未被有效反驳。"
                )
            else:
                verdict = (
                    f"【辩论结论】多方略占优势，{bull_count}/{len(result.opinions)} 位分析师看多。"
                    f"辩论评分 {result.final_score:+.2f}，置信度 {result.confidence:.0%}。"
                    f"但空方提出了值得关注的风险点。"
                )
        elif result.winner == "bearish":
            if result.confidence > 0.7:
                verdict = (
                    f"【辩论结论】空方证据链充分，{bear_count}/{len(result.opinions)} 位分析师看空。"
                    f"辩论评分 {result.final_score:+.2f}，置信度 {result.confidence:.0%}。"
                    f"多轮辩论后空方论点未被有效反驳。"
                )
            else:
                verdict = (
                    f"【辩论结论】空方略占优势，{bear_count}/{len(result.opinions)} 位分析师看空。"
                    f"辩论评分 {result.final_score:+.2f}，置信度 {result.confidence:.0%}。"
                    f"但多方仍有支撑论据。"
                )
        else:
            verdict = (
                f"【辩论结论】多空力量均衡，{bull_count}看多/{bear_count}看空/{neutral_count}中性。"
                f"辩论评分 {result.final_score:+.2f}，置信度 {result.confidence:.0%}。"
                f"当前市场方向不明，建议观望。"
            )

        # 风险评估
        risk_parts = []
        if result.disagreements:
            risk_parts.append(f"分歧点: {', '.join(result.disagreements[:3])}")
        if regime == "DIVERGENT":
            risk_parts.append("市场体制背离，信号可靠性降低")
        elif regime == "FUSED":
            risk_parts.append("市场共振，趋势信号可靠性较高")
        else:
            risk_parts.append("市场体制中性，信号正常")

        risk_assessment = " | ".join(risk_parts)

        # 行动建议
        if result.winner == "tie" or result.confidence < 0.4:
            action = "建议观望。辩论未能达成明确方向，不做也是一种交易。"
        elif abs(result.final_score) < 0.3:
            action = "建议轻仓试探。信号方向明确但强度不足，控制仓位。"
        elif result.confidence > 0.6:
            if result.winner == "bullish":
                action = "建议正常做多入场。辩论形成明确多头共识，按策略信号执行。"
            else:
                action = "建议正常做空入场。辩论形成明确空头共识，按策略信号执行。"
        else:
            action = "建议谨慎入场。方向明确但存在分歧，注意风险控制。"

        return verdict, risk_assessment, action

    # ------------------------------------------------------------------
    # 分析师信誉管理
    # ------------------------------------------------------------------

    def update_analyst_reputation(
        self,
        analyst_name: str,
        was_correct: bool,
        learning_rate: float = 0.05,
    ):
        """更新分析师信誉 — 正确预测加分，错误减分。"""
        analyst = self._analyst_index.get(analyst_name)
        if not analyst:
            return

        analyst.total_analyses += 1
        if was_correct:
            analyst.correct_calls += 1

        # 更新信誉
        if analyst.total_analyses > 0:
            raw_rep = analyst.correct_calls / analyst.total_analyses
            # 贝叶斯平滑，避免小样本极端
            analyst.reputation = (raw_rep * analyst.total_analyses + 0.5 * 10) / \
                                 (analyst.total_analyses + 10)

        # 调整权重
        analyst.weight = 0.10 + analyst.reputation * 0.20

    # ------------------------------------------------------------------
    # Checkpoint 持久化
    # ------------------------------------------------------------------

    def save_checkpoint(self, debate_id: str, result: DebateResult):
        """保存辩论检查点 — 借鉴 LangGraph checkpoint。"""
        if not self.checkpoint_dir:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{debate_id}.json")

        try:
            data = {
                "debate_id": debate_id,
                "symbol": result.symbol,
                "timestamp": result.timestamp,
                "final_score": result.final_score,
                "confidence": result.confidence,
                "winner": result.winner,
                "total_rounds": result.total_rounds,
                "verdict": result.verdict,
                "risk_assessment": result.risk_assessment,
                "action_recommendation": result.action_recommendation,
                "analyst_reputations": {
                    a.name: {
                        "reputation": a.reputation,
                        "total_analyses": a.total_analyses,
                        "correct_calls": a.correct_calls,
                        "weight": a.weight,
                    }
                    for a in self.analysts
                },
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, debate_id: str) -> Optional[Dict]:
        """加载辩论检查点。"""
        if not self.checkpoint_dir:
            return None

        checkpoint_path = os.path.join(self.checkpoint_dir, f"{debate_id}.json")
        if not os.path.exists(checkpoint_path):
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 恢复分析师信誉
            if "analyst_reputations" in data:
                for name, rep in data["analyst_reputations"].items():
                    if name in self._analyst_index:
                        a = self._analyst_index[name]
                        a.reputation = rep["reputation"]
                        a.total_analyses = rep["total_analyses"]
                        a.correct_calls = rep["correct_calls"]
                        a.weight = rep["weight"]

            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


# ===========================================================================
# 决策日志注入（借鉴 TradingAgents 的 reflection injection）
# ===========================================================================

def inject_past_reflections(
    symbol: str,
    memory_manager: Any,
    max_reflections: int = 3,
) -> str:
    """将历史决策反思注入当前研判 — 借鉴 TradingAgents 的决策日志机制。

    TradingAgents 在每次分析时，会获取：
      - 同品种最近决策的反思
      - 跨品种的经验教训
      - 实现收益率（与 SPY 对比的 alpha）
    """
    recent_decisions = memory_manager.get_recent_decisions(50) if memory_manager else []

    if not recent_decisions:
        return ""

    parts = []

    # 1. 同品种最近决策
    same_symbol = [d for d in recent_decisions if d.symbol == symbol]
    if same_symbol:
        recent = same_symbol[-max_reflections:]
        wins = sum(1 for d in recent if d.is_win)
        total = len(recent)
        parts.append(
            f"【历史参考】{symbol} 最近 {total} 次决策: "
            f"胜率 {wins/total:.0%} ({wins}W/{total-wins}L)"
        )
        if recent:
            last = recent[-1]
            if last.reflection:
                parts.append(f"上次反思: {last.reflection[:200]}")

    # 2. 跨品种经验
    cross_insight = memory_manager.get_cross_symbol_insight(symbol) if memory_manager else None
    if cross_insight:
        parts.append(f"【跨品种参考】{cross_insight}")

    # 3. 经验教训
    lessons = memory_manager.get_lessons(top_n=3) if memory_manager else []
    if lessons:
        lesson_text = " | ".join([l["lesson"][:80] for l in lessons[:2]])
        parts.append(f"【经验教训】{lesson_text}")

    if parts:
        return "\n".join(parts)
    return ""


# ===========================================================================
# 简化版辩论接口（兼容旧代码）
# ===========================================================================

@dataclass
class SimpleDebateResult:
    """简化版辩论结果 — 兼容旧 run_debate 接口。"""
    bullish_points: List[str]
    bearish_points: List[str]
    winner: str                     # "bullish" | "bearish" | "tie"
    score_margin: float             # 正=多方胜, 负=空方胜
    net_verdict: str                # 综合结论
    confidence: float


def run_enhanced_debate(
    bullish_evidence: List[Tuple[str, float]],
    bearish_evidence: List[Tuple[str, float]],
    debate_rounds: int = 2,
    regime: str = "NEUTRAL",
    past_reflection: str = "",
) -> SimpleDebateResult:
    """增强版辩论 — 兼容旧 run_debate 接口，增加体制感知和反思注入。

    Args:
        bullish_evidence: [(理由, 权重), ...]
        bearish_evidence: [(理由, 权重), ...]
        debate_rounds: 辩论轮次
        regime: 市场体制
        past_reflection: 历史反思文本

    Returns:
        SimpleDebateResult
    """
    # 加权计分
    bullish_score = sum(w for _, w in bullish_evidence)
    bearish_score = sum(w for _, w in bearish_evidence)

    # 体制调整
    if regime == "FUSED":
        # 共振体制：增强多数派
        if bullish_score > bearish_score:
            bullish_score *= 1.2
        elif bearish_score > bullish_score:
            bearish_score *= 1.2
    elif regime == "DIVERGENT":
        # 背离体制：降低所有得分
        bullish_score *= 0.7
        bearish_score *= 0.7

    # 反思注入调整
    if past_reflection:
        # 如果历史反思提示风险，增加保守倾向
        if "亏损" in past_reflection or "风险" in past_reflection:
            bullish_score *= 0.9
            bearish_score *= 0.9

    # 计算分差
    total = bullish_score + bearish_score
    if total > 0:
        score_margin = (bullish_score - bearish_score) / total
    else:
        score_margin = 0.0

    # 辩论结果
    if score_margin > 0.3:
        winner = "bullish"
    elif score_margin < -0.3:
        winner = "bearish"
    else:
        winner = "tie"

    # 置信度
    confidence = min(0.95, abs(score_margin) * 2.0)

    # 综合结论
    if winner == "bullish":
        net_verdict = f"多方证据占优 (优势 {score_margin:.1%})，但需注意空方警告"
    elif winner == "bearish":
        net_verdict = f"空方证据占优 (优势 {abs(score_margin):.1%})，但多方仍有支撑"
    else:
        net_verdict = "多空力量均衡，市场方向不明，建议观望"

    return SimpleDebateResult(
        bullish_points=[p for p, _ in bullish_evidence],
        bearish_points=[p for p, _ in bearish_evidence],
        winner=winner,
        score_margin=round(score_margin, 3),
        net_verdict=net_verdict,
        confidence=round(confidence, 3),
    )


# ===========================================================================
# 格式化输出
# ===========================================================================

def format_debate_report(result: DebateResult) -> str:
    """格式化辩论报告。"""
    lines = [
        "",
        "┌" + "─" * 58 + "┐",
        f"│  ⚖  TradingAgents 辩论引擎 · 综合研判报告".ljust(61) + "│",
        f"│  标的: {result.symbol:<50}│",
        "├" + "─" * 58 + "┤",
    ]

    # 分析师投票
    lines.append("│  【分析师投票】".ljust(61) + "│")
    for o in result.opinions:
        icon = "🔴" if o.direction == "bullish" else ("🟢" if o.direction == "bearish" else "⚪")
        lines.append(
            f"│    {icon} [{o.analyst_name}] {o.direction.upper():7s} "
            f"score={o.score:+.2f} conv={o.conviction:.0%}".ljust(61) + "│"
        )

    lines.append("├" + "─" * 58 + "┤")

    # 辩论过程
    for r in result.rounds:
        lines.append(f"│  Round {r.round_number}: {r.key_insight[:45]}".ljust(61) + "│")

    lines.append("├" + "─" * 58 + "┤")

    # 最终结论
    winner_text = "多方胜出" if result.winner == "bullish" else (
        "空方胜出" if result.winner == "bearish" else "平局"
    )
    lines.append(
        f"│  结果: {winner_text} | "
        f"评分: {result.final_score:+.2f} | "
        f"置信度: {result.confidence:.0%}".ljust(61) + "│"
    )

    lines.append("├" + "─" * 58 + "┤")

    # 研判
    for vline in result.verdict.split("\n"):
        lines.append(f"│  {vline}".ljust(61) + "│")

    lines.append("├" + "─" * 58 + "┤")

    # 风险评估
    lines.append(f"│  ⚠ 风险: {result.risk_assessment[:50]}".ljust(61) + "│")
    lines.append(f"│  📋 {result.action_recommendation[:50]}".ljust(61) + "│")

    lines.append("└" + "─" * 58 + "┘")

    return "\n".join(lines)