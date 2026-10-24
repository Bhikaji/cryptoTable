"""Independent risk manager.

This is a **safety control**, not a learned component.  It receives only the
risk-relevant fields of a proposal (size, leverage, exposure, stops, timing) and
never sees a reward score, model version or outcome label.  That independence is
what guarantees "the AI must never learn to override these controls".
"""

from __future__ import annotations

from typing import List

from core.config import RiskConfig
from core.types import RiskDecision, TradeProposal


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    # ------------------------------------------------------------------ #
    def evaluate(self, proposal: TradeProposal) -> RiskDecision:
        cfg = self.config
        reasons: List[str] = []
        max_notional = proposal.equity * cfg.max_position_pct
        leverage_cap = cfg.max_leverage
        cooldown = False

        # 1. Position size --------------------------------------------------
        notional = abs(proposal.quantity) * proposal.entry_price
        if notional > max_notional + 1e-9:
            reasons.append(
                f"position_notional {notional:.2f} exceeds cap {max_notional:.2f}"
            )

        # 2. Leverage -------------------------------------------------------
        if proposal.leverage > leverage_cap + 1e-9:
            reasons.append(f"leverage {proposal.leverage} exceeds cap {leverage_cap}")

        # 3. Total exposure -------------------------------------------------
        exposure_after = proposal.open_exposure + notional
        if exposure_after > proposal.equity * cfg.max_total_exposure_pct + 1e-9:
            reasons.append("total exposure would breach the account cap")

        # 4. Stop distance --------------------------------------------------
        if cfg.require_stop_loss and proposal.stop_distance is None:
            reasons.append("missing stop loss")
        elif proposal.stop_distance is not None:
            if proposal.stop_distance > cfg.max_stop_distance_pct + 1e-9:
                reasons.append(
                    f"stop distance {proposal.stop_distance:.4f} too wide "
                    f"(cap {cfg.max_stop_distance_pct})"
                )
            if proposal.stop_distance < cfg.min_stop_distance_pct - 1e-9:
                reasons.append("stop distance too tight")

        # 5. Cooldown -------------------------------------------------------
        if cfg.cooldown_seconds > 0 and proposal.now - proposal.last_entry_time < cfg.cooldown_seconds:
            cooldown = True
            reasons.append("entry cooldown active")

        # 6. Daily loss halt ------------------------------------------------
        if proposal.daily_pnl <= -proposal.equity * cfg.max_daily_loss_pct:
            reasons.append("daily loss limit breached; trading halted")

        # 7. Entry frequency ------------------------------------------------
        if proposal.entries_today >= cfg.max_entries_per_day:
            reasons.append("max entries per day reached")

        approved = not reasons
        return RiskDecision(
            approved=approved,
            reasons=reasons,
            position_size_cap=max_notional,
            leverage_cap=leverage_cap,
            exposure_used=exposure_after,
            cooldown_active=cooldown,
        )
