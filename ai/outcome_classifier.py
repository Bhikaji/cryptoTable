"""Outcome / error taxonomy (Section 18.4).

A single generic "losing trade" penalty is explicitly *not* used.  Every outcome
is mapped to one of the eleven categories, with a plain-language explanation.

Learning penalties are applied primarily to **avoidable** decision and
execution errors.  Ordinary market losses from a sound, followed strategy are
recorded as ``VALID_SETUP_LOSS`` and are not treated as proof the strategy is
wrong (requirement 18.3.5).

Risk violations and data-quality failures additionally flag safety controls,
which are enforced independently of the reward score.
"""

from __future__ import annotations

from typing import Optional

from core.config import RewardConfig
from core.types import (
    AVOIDABLE_ERROR_CATEGORIES,
    SAFETY_CRITICAL_CATEGORIES,
    DataQuality,
    OutcomeCategory,
    OutcomeClassification,
    RewardBreakdown,
    Trade,
)


class OutcomeClassifier:
    def __init__(self, reward_config: RewardConfig | None = None) -> None:
        self.reward_config = reward_config or RewardConfig()

    # ------------------------------------------------------------------ #
    def classify(
        self,
        trade: Trade,
        reward: Optional[RewardBreakdown] = None,
    ) -> OutcomeClassification:
        # 1. Safety-critical, highest precedence.
        if trade.risk_violation or not trade.risk_manager_approved:
            return self._make(trade, OutcomeCategory.RISK_VIOLATION,
                               "trade breached a configured risk constraint")

        if trade.data_quality != DataQuality.OK:
            return self._make(trade, OutcomeCategory.DATA_QUALITY_FAILURE,
                               f"market data quality: {trade.data_quality.value}")

        if not trade.execution_matched:
            return self._make(trade, OutcomeCategory.EXECUTION_ERROR,
                               "execution did not match the approved proposal")

        # 2. Cost erosion.
        cost_ratio = trade.total_costs / max(trade.risk_budget(self.reward_config.default_stop_distance), 1e-9)
        if trade.excessive_cost or cost_ratio > self.reward_config.excessive_cost_ratio:
            return self._make(trade, OutcomeCategory.EXCESSIVE_COST,
                               f"costs consumed the edge (cost/risk={cost_ratio:.2f})")

        # 3. Setup / regime errors.
        if trade.false_breakout:
            return self._make(trade, OutcomeCategory.FALSE_BREAKOUT,
                               "breakout failed against a predefined rule")
        if trade.wrong_regime:
            return self._make(trade, OutcomeCategory.WRONG_REGIME,
                               "market regime was misclassified")

        # 4. Exit quality.
        if trade.net_pnl < 0:
            if trade.poor_entry or not trade.strategy_followed:
                return self._make(trade, OutcomeCategory.POOR_ENTRY,
                                   "entry conditions were weak or timing poor")
            if not trade.exit_planned:
                return self._make(trade, OutcomeCategory.PREMATURE_EXIT,
                                   "exit violated the strategy without valid reason")
            return self._make(trade, OutcomeCategory.VALID_SETUP_LOSS,
                               "strategy followed correctly; ordinary market loss")

        if trade.net_pnl > 0:
            return self._make(trade, OutcomeCategory.GOOD_EXIT,
                               "profitable exit consistent with the plan")

        # 5. Essentially flat.
        if trade.strategy_followed:
            return self._make(trade, OutcomeCategory.VALID_SETUP_LOSS,
                               "no material move; strategy followed")
        return self._make(trade, OutcomeCategory.POOR_ENTRY,
                           "no material move with weak entry conditions")

    # ------------------------------------------------------------------ #
    def classify_rejection(self, trade: Trade) -> OutcomeClassification:
        """Classify a trade that was *rejected* by the risk manager."""
        valid_setup = trade.strategy_followed and all(
            trade.strategy_conditions.values()
        ) if trade.strategy_conditions else trade.strategy_followed
        if valid_setup and not trade.risk_manager_approved:
            return self._make(trade, OutcomeCategory.MISSED_VALID_SETUP,
                              "a valid setup was rejected (recorded separately from losses)")
        return self._make(trade, OutcomeCategory.RISK_VIOLATION,
                          "proposed trade violated configured risk constraints")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make(trade: Trade, category: OutcomeCategory, explanation: str) -> OutcomeClassification:
        return OutcomeClassification(
            trade_id=trade.trade_id,
            category=category,
            explanation=explanation,
            is_avoidable_error=category in AVOIDABLE_ERROR_CATEGORIES,
            triggers_safety_control=category in SAFETY_CRITICAL_CATEGORIES,
        )
