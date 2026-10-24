"""Configurable reward / penalty engine (Section 18.3).

Design notes
------------
* Returns are **risk-normalised**: ``net_pnl`` is divided by the capital at risk
  (``notional * stop_distance``).  A bigger position therefore buys no bigger
  reward, satisfying requirement 18.3.7.
* Every component is recorded so a reward is fully explainable and reproducible.
* Extreme values are clipped so one outlier cannot dominate training.
* A profitable trade that violated risk limits is gated to ``<= 0`` — it can
  never receive an unconditional positive reward (requirement 18.3.4).
* Leverage breaches, oversized positions, recovery trading, excessive
  frequency and repeated entries in unfavourable conditions are penalised and
  never rewarded (requirement 18.3.9).
* Ordinary losses from a correctly followed strategy are recorded as mild
  negatives; avoidable decision/execution errors attract much larger penalties.
"""

from __future__ import annotations

from typing import Optional

from core.config import RewardConfig
from core.types import OutcomeCategory, RewardBreakdown, Trade


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(min(max(x, lo), hi))


class RewardEngine:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        trade: Trade,
        classification: Optional[OutcomeCategory] = None,
    ) -> RewardBreakdown:
        cfg = self.config

        risk_budget = trade.risk_budget(cfg.default_stop_distance)
        net_pnl = trade.gross_pnl - trade.total_costs
        if trade.net_pnl and abs(trade.net_pnl - net_pnl) < 1e-9:
            net_pnl = trade.net_pnl
        risk_adjusted_return = net_pnl / risk_budget

        # ---- 1. P&L component (risk-normalised) ----------------------------
        pnl_component = _clip(risk_adjusted_return / cfg.pnl_scale)

        # ---- 2. Execution-quality component --------------------------------
        cost_drag = trade.total_costs / risk_budget
        execution_quality = _clip(1.0 - cost_drag / cfg.cost_scale, -1.0, 1.0)
        if not trade.execution_matched:
            execution_quality = -1.0
        if trade.excessive_cost:
            execution_quality = _clip(execution_quality - 0.5)

        # ---- 3. Risk-management component -----------------------------------
        mae_pnl = abs(trade.mae) * abs(trade.quantity)
        honors_stop = mae_pnl <= risk_budget * 1.05
        risk_component = 0.0
        if honors_stop:
            risk_component += 0.5
        if not trade.leverage_breach and not trade.oversized_position:
            risk_component += 0.5
        if trade.risk_violation:
            risk_component -= 1.0
        risk_component = _clip(risk_component)

        # ---- 4. Strategy-compliance component ------------------------------
        if trade.strategy_followed and trade.exit_planned:
            compliance = 1.0
        elif trade.strategy_followed and not trade.exit_planned:
            compliance = -0.5  # premature exit
        else:
            compliance = -1.0

        # ---- Penalties -----------------------------------------------------
        excessive_risk_penalty = 0.0
        if trade.risk_violation:
            excessive_risk_penalty += 1.5
        if trade.leverage_breach:
            excessive_risk_penalty += 0.5
        if trade.oversized_position:
            excessive_risk_penalty += 0.5
        if trade.recovery_trade:
            excessive_risk_penalty += 0.3  # chasing losses is penalised
        if trade.excessive_frequency:
            excessive_risk_penalty += 0.5

        avoidable_cost_penalty = 0.0
        if trade.excessive_cost:
            avoidable_cost_penalty += 0.6
        if trade.slippage_estimate > risk_budget * cfg.excessive_cost_ratio:
            avoidable_cost_penalty += 0.3

        repeated_entry_penalty = 0.8 if trade.repeated_entry_unfavorable else 0.0

        decision_error_penalty = 0.0
        if classification in (
            OutcomeCategory.POOR_ENTRY,
            OutcomeCategory.FALSE_BREAKOUT,
            OutcomeCategory.WRONG_REGIME,
            OutcomeCategory.EXECUTION_ERROR,
        ):
            decision_error_penalty += 1.0
        if trade.poor_entry or trade.false_breakout or trade.wrong_regime:
            decision_error_penalty = max(decision_error_penalty, 1.0)

        # ---- Weighted total ------------------------------------------------
        raw_components = {
            "pnl_component": pnl_component,
            "risk_component": risk_component,
            "execution_component": execution_quality,
            "compliance_component": compliance,
            "excessive_risk_penalty": excessive_risk_penalty,
            "avoidable_cost_penalty": avoidable_cost_penalty,
            "repeated_entry_penalty": repeated_entry_penalty,
            "decision_error_penalty": decision_error_penalty,
        }
        weights = {
            "pnl_component": cfg.w_pnl,
            "risk_component": cfg.w_risk,
            "execution_component": cfg.w_execution,
            "compliance_component": cfg.w_compliance,
            "excessive_risk_penalty": cfg.w_excess_risk,
            "avoidable_cost_penalty": cfg.w_cost,
            "repeated_entry_penalty": cfg.w_repeated_entry,
            "decision_error_penalty": cfg.w_decision_error,
        }

        total = (
            cfg.w_pnl * pnl_component
            + cfg.w_risk * risk_component
            + cfg.w_execution * execution_quality
            + cfg.w_compliance * compliance
            - cfg.w_excess_risk * excessive_risk_penalty
            - cfg.w_cost * avoidable_cost_penalty
            - cfg.w_repeated_entry * repeated_entry_penalty
            - cfg.w_decision_error * decision_error_penalty
        )

        # Outcome sign safety + risk-violation gate.
        if trade.risk_violation:
            total = min(total, 0.0)          # never reward a breach unconditionally
        elif net_pnl > 0:
            total = max(total, 1e-6)         # a winning trade is never negative
        elif net_pnl < 0:
            total = min(total, -1e-6)        # a losing trade is never positive
        else:
            total = min(total, 0.0)          # flat / missed setups stay neutral

        weighted = {k: weights[k] * raw_components[k] for k in raw_components}

        # Clip extreme values (requirement 18.3.7).
        clipped = _clip(total, -cfg.reward_clip, cfg.reward_clip)
        clipped_flag = abs(total - clipped) > 1e-9

        explanation = self._explain(trade, raw_components, weighted, total, clipped, clipped_flag)

        return RewardBreakdown(
            trade_id=trade.trade_id,
            total_reward=float(total),
            clipped_reward=float(clipped),
            clipped=clipped_flag,
            risk_budget=float(risk_budget),
            risk_adjusted_return=float(risk_adjusted_return),
            raw_components=raw_components,
            weighted_components=weighted,
            weights=weights,
            explanation=explanation,
            classification=classification.value if classification else None,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _explain(trade, raw, weighted, total, clipped, clipped_flag) -> str:
        parts = [f"net_pnl={trade.net_pnl:.4f}", f"risk_budget={trade.risk_budget():.4f}"]
        for key in ("pnl_component", "risk_component", "execution_component", "compliance_component"):
            if abs(weighted[key]) > 1e-9:
                parts.append(f"{key}={weighted[key]:+.3f}")
        for key in ("excessive_risk_penalty", "avoidable_cost_penalty",
                    "repeated_entry_penalty", "decision_error_penalty"):
            if abs(weighted[key]) > 1e-9:
                parts.append(f"-{key}={weighted[key]:+.3f}")
        parts.append(f"total={total:+.3f}")
        if clipped_flag:
            parts.append(f"clipped={clipped:+.3f}")
        if trade.risk_violation:
            parts.append("gated<=0 due to risk violation")
        return "; ".join(parts)

    @staticmethod
    def bucket(reward: float, neutral_band: float = 0.10) -> str:
        if reward > neutral_band:
            return "positive"
        if reward < -neutral_band:
            return "negative"
        return "neutral"
