"""
Risk Committee：AI 风险委员会。

不直接输出 BUY/SELL，而是基于市场数据、技术指标、资金流、新闻情绪输出 0-100 的 Risk Score。
量化规则仍做最终交易决策，但会根据 Risk Score 缩放仓位或禁止交易。

当前实现提供两种模式：
  1. 使用现有 debate_engine 的辩论结果转换为风险分。
  2. 当 debate_engine 不可用时，使用基于市场微观结构/波动率的规则化风险分兜底。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """风险评估结果。"""
    risk_score: int          # 0-100
    allow_trade: bool
    position_factor: float   # 仓位缩放系数
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "allow_trade": self.allow_trade,
            "position_factor": self.position_factor,
            "reasons": self.reasons,
        }


class RiskCommittee:
    """AI 风险委员会。"""

    def __init__(
        self,
        debate_engine: Optional[Any] = None,
        low_risk_threshold: int = 30,
        high_risk_threshold: int = 70,
        crisis_threshold: int = 95,
    ):
        """
        Args:
            debate_engine: 可选，外部 debate_engine 实例
            low_risk_threshold: <= 此值允许正常仓位
            high_risk_threshold: > 此值降低仓位
            crisis_threshold: >= 此值禁止交易
        """
        self.debate_engine = debate_engine
        self.low_risk_threshold = low_risk_threshold
        self.high_risk_threshold = high_risk_threshold
        self.crisis_threshold = crisis_threshold

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def assess(
        self,
        market_data: Dict[str, Any],
        technical_indicators: Optional[Dict[str, Any]] = None,
        flow_data: Optional[Dict[str, Any]] = None,
        sentiment: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """输出 Risk Score 与仓位建议。

        Args:
            market_data: 至少包含 price, returns_24h, volatility, funding_rate
            technical_indicators: 如 rsi, macd, trend_strength
            flow_data: 资金流数据，如 net_inflow
            sentiment: 情绪数据，如 news_sentiment
        """
        reasons: List[str] = []

        # 模式 1：使用 debate_engine 输出转换
        if self.debate_engine is not None:
            try:
                score = self._from_debate_engine(
                    market_data, technical_indicators, flow_data, sentiment
                )
                reasons.append("基于 TradingAgents 辩论结果")
            except Exception as e:
                logger.warning(f"[RiskCommittee] debate engine failed: {e}, fallback to rule-based")
                score, rule_reasons = self._rule_based(
                    market_data, technical_indicators, flow_data, sentiment
                )
                reasons.extend(rule_reasons)
        else:
            score, rule_reasons = self._rule_based(
                market_data, technical_indicators, flow_data, sentiment
            )
            reasons.extend(rule_reasons)

        return self._build_assessment(score, reasons)

    # ------------------------------------------------------------------
    # 直接从 debate_result 生成 Risk Score
    # ------------------------------------------------------------------
    def assess_from_debate(
        self,
        debate_result: Any,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """直接传入 TradingAgents 辩论结果，输出 Risk Score。

        Args:
            debate_result: DebateResult 或 SimpleDebateResult 实例
            market_data: 可选的市场数据，用于与规则化风险分融合

        Returns:
            RiskAssessment
        """
        reasons: List[str] = []
        score = self._score_from_debate_result(debate_result, reasons)

        # 与市场微观结构融合：若 debate_result 未包含足够风险信号，补充规则化风险分
        if market_data:
            rule_score, rule_reasons = self._rule_based(market_data, None, None, None)
            # 加权：debate 60% + rule 40%
            score = int(0.6 * score + 0.4 * rule_score)
            reasons.extend([r for r in rule_reasons if r not in reasons])

        return self._build_assessment(score, reasons)

    def _score_from_debate_result(self, debate_result: Any, reasons: List[str]) -> int:
        """把 debate_result 映射为 0-100 风险分。"""
        winner = ""
        confidence = 0.0
        final_score = 0.0
        verdict = ""

        # 统一取值
        if isinstance(debate_result, dict):
            winner = debate_result.get("winner", "")
            confidence = debate_result.get("confidence", 0.0)
            final_score = debate_result.get("final_score", debate_result.get("score_margin", 0.0))
            verdict = debate_result.get("verdict", "")
        else:
            winner = getattr(debate_result, "winner", "")
            confidence = getattr(debate_result, "confidence", 0.0)
            final_score = getattr(
                debate_result, "final_score",
                getattr(debate_result, "score_margin", 0.0)
            )
            verdict = getattr(debate_result, "verdict", "")
            if not verdict:
                verdict = getattr(debate_result, "action_recommendation", "")

        winner = (winner or "").lower()
        verdict = (verdict or "").upper()

        # 基础分：50（中性）
        base = 50

        # 方向调整
        if winner == "bullish":
            base -= 20
            reasons.append(f"辩论结果看多 (confidence={confidence:.0%})，风险降低")
        elif winner == "bearish":
            base += 20
            reasons.append(f"辩论结果看空 (confidence={confidence:.0%})，风险上升")
        elif winner == "tie":
            base += 10
            reasons.append(f"辩论平局 (confidence={confidence:.0%})，风险略升")

        # 置信度调整：高置信度观点会放大方向判断
        conf_adjustment = int((confidence - 0.5) * 40)
        if winner in ("bullish", "bearish"):
            if winner == "bullish":
                base -= conf_adjustment
            else:
                base += conf_adjustment

        # final_score 反映多空强度 (-1 ~ +1)
        score_adjustment = int(-final_score * 20)
        base += score_adjustment

        #  verdict 关键词兜底
        if "CRISIS" in verdict or "AVOID" in verdict or "SELL" in verdict:
            base = max(base, 90)
            reasons.append("辩论 verdict 提示危机/规避")
        elif "CAUTION" in verdict or "HOLD" in verdict:
            base = max(base, 70)
            reasons.append("辩论 verdict 提示谨慎/观望")
        elif "BUY" in verdict or "BULLISH" in verdict:
            base = min(base, 30)
            reasons.append("辩论 verdict 提示买入/看多")

        return int(np.clip(base, 0, 100))

    # ------------------------------------------------------------------
    # 构建统一评估结果
    # ------------------------------------------------------------------
    def _build_assessment(self, score: int, reasons: List[str]) -> RiskAssessment:
        """根据分数生成 RiskAssessment。"""
        score = int(np.clip(score, 0, 100))
        allow_trade = score < self.crisis_threshold

        if score <= self.low_risk_threshold:
            factor = 1.0
            reasons.append(f"风险分 {score} <= {self.low_risk_threshold}，允许正常仓位")
        elif score <= self.high_risk_threshold:
            factor = 0.5
            reasons.append(f"风险分 {score} 处于中等风险，仓位减半")
        else:
            factor = 0.0 if not allow_trade else 0.25
            if allow_trade:
                reasons.append(f"风险分 {score} 极高，仅允许 25% 仓位")
            else:
                reasons.append(f"风险分 {score} >= {self.crisis_threshold}，禁止交易")

        return RiskAssessment(
            risk_score=score,
            allow_trade=allow_trade,
            position_factor=factor,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # 模式 1：debate_engine 结果转换
    # ------------------------------------------------------------------
    def _from_debate_engine(
        self,
        market_data: Dict[str, Any],
        technical_indicators: Optional[Dict[str, Any]],
        flow_data: Optional[Dict[str, Any]],
        sentiment: Optional[Dict[str, Any]],
    ) -> int:
        """调用 debate_engine 并将输出映射为风险分。"""
        result = self.debate_engine.conduct_debate(
            symbol=market_data.get("symbol", "UNKNOWN"),
            channel_reports=[],  # 修复: 不可为 None，debate_engine 需要可迭代对象
            fvg_signals=[],
            current_price=market_data.get("price", 0.0),
            regime=market_data.get("regime", "NEUTRAL"),
        )
        # 适配 debate 输出为 dict
        verdict = result.winner if hasattr(result, "winner") else "tie"
        verdict = verdict.upper()
        if "CRISIS" in verdict or "AVOID" in verdict or "SELL" in verdict:
            return 95
        if "CAUTION" in verdict or "HOLD" in verdict:
            return 70
        if "BUY" in verdict or "BULLISH" in verdict:
            return 30
        # 若 debate 输出已有 risk_score 或 confidence，映射为风险分
        if hasattr(result, "risk_score"):
            return int(result.risk_score)
        if hasattr(result, "confidence"):
            return int((1.0 - result.confidence) * 100)
        return 50

    # ------------------------------------------------------------------
    # 模式 2：规则化兜底
    # ------------------------------------------------------------------
    def _rule_based(
        self,
        market_data: Dict[str, Any],
        technical_indicators: Optional[Dict[str, Any]],
        flow_data: Optional[Dict[str, Any]],
        sentiment: Optional[Dict[str, Any]],
    ) -> tuple:
        score = 20  # 基础分
        reasons: List[str] = []

        ret = market_data.get("returns_24h", 0.0) * 100
        vol = market_data.get("volatility", 0.0)
        funding = market_data.get("funding_rate", 0.0)

        # 24h 跌幅风险
        if ret < -8:
            score += 30
            reasons.append(f"24h 跌幅 {ret:.1f}% 过大")
        elif ret < -4:
            score += 15
            reasons.append(f"24h 跌幅 {ret:.1f}%")
        elif ret > 8:
            score += 10
            reasons.append(f"24h 涨幅 {ret:.1f}% 过快")

        # 波动率风险
        if vol > 0.05:  # 5% 小时波动
            score += 25
            reasons.append(f"波动率 {vol:.2%} 过高")
        elif vol > 0.03:
            score += 10
            reasons.append(f"波动率 {vol:.2%} 偏高")

        # 资金费率极端
        if abs(funding) > 0.01:
            score += 15
            reasons.append(f"资金费率 {funding:.3%} 极端")

        # 技术指标
        if technical_indicators:
            rsi = technical_indicators.get("rsi")
            if rsi is not None:
                if rsi > 75:
                    score += 10
                    reasons.append(f"RSI {rsi:.1f} 超买")
                elif rsi < 25:
                    score += 10
                    reasons.append(f"RSI {rsi:.1f} 超卖")

            trend = technical_indicators.get("trend_strength", 0)
            if abs(trend) > 0.8:
                score += 5
                reasons.append("趋势强度极端")

        # 资金流
        if flow_data:
            inflow = flow_data.get("net_inflow", 0)
            if inflow < -0.2:
                score += 10
                reasons.append("资金大幅流出")

        # 情绪
        if sentiment:
            news = sentiment.get("news_sentiment", 0)
            if news < -0.5:
                score += 10
                reasons.append("新闻情绪极度负面")

        if not reasons:
            reasons.append("无显著风险信号")

        return int(np.clip(score, 0, 100)), reasons
