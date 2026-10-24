"""Domain types shared across the trading core and the AI subsystem.

Nothing here imports the ``ai`` package, which keeps the risk manager and the
trading engine free of any reward coupling.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    UNKNOWN = "UNKNOWN"


class ExitReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    MANUAL = "MANUAL"
    SIGNAL = "SIGNAL"
    RISK_STOP = "RISK_STOP"
    END_OF_DATA = "END_OF_DATA"
    LIQUIDATION = "LIQUIDATION"


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExecutionKind(str, Enum):
    """Never mix hypothetical fills with real exchange fills."""

    LIVE = "LIVE"
    PAPER = "PAPER"


class DataQuality(str, Enum):
    OK = "OK"
    STALE = "STALE"
    INCOMPLETE = "INCOMPLETE"
    INCONSISTENT = "INCONSISTENT"


class OutcomeCategory(str, Enum):
    """Error / outcome taxonomy from Section 18.4."""

    VALID_SETUP_LOSS = "VALID_SETUP_LOSS"
    POOR_ENTRY = "POOR_ENTRY"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    WRONG_REGIME = "WRONG_REGIME"
    EXCESSIVE_COST = "EXCESSIVE_COST"
    RISK_VIOLATION = "RISK_VIOLATION"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
    GOOD_EXIT = "GOOD_EXIT"
    PREMATURE_EXIT = "PREMATURE_EXIT"
    MISSED_VALID_SETUP = "MISSED_VALID_SETUP"
    PROFITABLE = "PROFITABLE"


# Categories that represent *avoidable* decision / execution errors and should
# therefore attract learning penalties (Section 18.4).
AVOIDABLE_ERROR_CATEGORIES = frozenset(
    {
        OutcomeCategory.POOR_ENTRY,
        OutcomeCategory.FALSE_BREAKOUT,
        OutcomeCategory.WRONG_REGIME,
        OutcomeCategory.EXECUTION_ERROR,
        OutcomeCategory.EXCESSIVE_COST,
    }
)

# Categories that must trigger safety controls independently of the reward.
SAFETY_CRITICAL_CATEGORIES = frozenset(
    {
        OutcomeCategory.RISK_VIOLATION,
        OutcomeCategory.DATA_QUALITY_FAILURE,
        OutcomeCategory.EXECUTION_ERROR,
    }
)


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
@dataclass
class MarketBar:
    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
@dataclass
class RiskDecision:
    approved: bool
    reasons: List[str] = field(default_factory=list)
    position_size_cap: float = 0.0
    leverage_cap: float = 0.0
    exposure_used: float = 0.0
    cooldown_active: bool = False


@dataclass
class TradeProposal:
    """A proposed trade submitted to the independent risk manager."""

    symbol: str
    side: TradeSide
    entry_price: float
    quantity: float
    leverage: float = 1.0
    stop_distance: Optional[float] = None
    expected_holding: float = 0.0
    equity: float = 1.0
    open_exposure: float = 0.0
    last_entry_time: float = 0.0
    now: float = 0.0
    entries_today: int = 0
    daily_pnl: float = 0.0


# --------------------------------------------------------------------------- #
# Trade / experience record (Section 18.2)
# --------------------------------------------------------------------------- #
def new_trade_id() -> str:
    return f"T-{uuid.uuid4().hex[:12]}"


@dataclass
class Trade:
    # --- identity / provenance -------------------------------------------------
    trade_id: str
    symbol: str
    timeframe: str
    side: TradeSide
    model_version: str
    execution_kind: ExecutionKind = ExecutionKind.PAPER
    entry_time: float = 0.0
    entry_price: float = 0.0

    # --- exit ----------------------------------------------------------------
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    exit_planned: bool = True

    # --- position ------------------------------------------------------------
    quantity: float = 0.0
    position_size: float = 0.0            # notional exposure
    leverage: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    stop_distance: Optional[float] = None  # fraction of entry price
    holding_duration: float = 0.0

    # --- market / decision context -------------------------------------------
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    features: Dict[str, float] = field(default_factory=dict)
    prediction_probabilities: Dict[str, float] = field(default_factory=dict)
    strategy_conditions: Dict[str, bool] = field(default_factory=dict)
    strategy_followed: bool = True

    # --- execution / costs ---------------------------------------------------
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    funding_cost: float = 0.0
    slippage_estimate: float = 0.0
    execution_matched: bool = True
    data_quality: DataQuality = DataQuality.OK

    # --- P&L / excursions ----------------------------------------------------
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    mfe: float = 0.0          # maximum favourable excursion (price terms)
    mae: float = 0.0          # maximum adverse excursion (price terms)

    # --- risk manager --------------------------------------------------------
    risk_manager_approved: bool = True
    risk_manager_reasons: List[str] = field(default_factory=list)

    # --- decision quality flags (used by reward + classifier) ----------------
    risk_violation: bool = False
    poor_entry: bool = False
    false_breakout: bool = False
    wrong_regime: bool = False
    excessive_cost: bool = False
    repeated_entry_unfavorable: bool = False
    recovery_trade: bool = False
    leverage_breach: bool = False
    oversized_position: bool = False
    excessive_frequency: bool = False

    notes: str = ""

    # --- derived convenience -------------------------------------------------
    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None

    @property
    def total_costs(self) -> float:
        return (
            self.entry_fee
            + self.exit_fee
            + self.funding_cost
            + self.slippage_estimate
        )

    @property
    def notional(self) -> float:
        if self.position_size:
            return self.position_size
        return abs(self.quantity) * self.entry_price

    def risk_budget(self, default_stop_distance: float = 0.02) -> float:
        """Capital at risk, used to normalise returns (Section 18.3.7)."""
        sd = self.stop_distance if self.stop_distance else default_stop_distance
        return max(self.notional * sd, 1e-9)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side.value,
            "model_version": self.model_version,
            "execution_kind": self.execution_kind.value,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "exit_planned": self.exit_planned,
            "quantity": self.quantity,
            "position_size": self.position_size,
            "leverage": self.leverage,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "stop_distance": self.stop_distance,
            "holding_duration": self.holding_duration,
            "market_regime": self.market_regime.value,
            "features": self.features,
            "prediction_probabilities": self.prediction_probabilities,
            "strategy_conditions": self.strategy_conditions,
            "strategy_followed": self.strategy_followed,
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "funding_cost": self.funding_cost,
            "slippage_estimate": self.slippage_estimate,
            "execution_matched": self.execution_matched,
            "data_quality": self.data_quality.value,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "mfe": self.mfe,
            "mae": self.mae,
            "risk_manager_approved": self.risk_manager_approved,
            "risk_manager_reasons": self.risk_manager_reasons,
            "risk_violation": self.risk_violation,
            "poor_entry": self.poor_entry,
            "false_breakout": self.false_breakout,
            "wrong_regime": self.wrong_regime,
            "excessive_cost": self.excessive_cost,
            "repeated_entry_unfavorable": self.repeated_entry_unfavorable,
            "recovery_trade": self.recovery_trade,
            "leverage_breach": self.leverage_breach,
            "oversized_position": self.oversized_position,
            "excessive_frequency": self.excessive_frequency,
            "notes": self.notes,
        }
        return d


@dataclass
class OutcomeClassification:
    trade_id: str
    category: OutcomeCategory
    explanation: str
    is_avoidable_error: bool = False
    triggers_safety_control: bool = False


@dataclass
class RewardBreakdown:
    trade_id: str
    total_reward: float
    clipped_reward: float
    clipped: bool
    risk_budget: float
    risk_adjusted_return: float
    raw_components: Dict[str, float]
    weighted_components: Dict[str, float]
    weights: Dict[str, float]
    explanation: str
    classification: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "total_reward": self.total_reward,
            "clipped_reward": self.clipped_reward,
            "clipped": self.clipped,
            "risk_budget": self.risk_budget,
            "risk_adjusted_return": self.risk_adjusted_return,
            "raw_components": self.raw_components,
            "weighted_components": self.weighted_components,
            "weights": self.weights,
            "explanation": self.explanation,
            "classification": self.classification,
        }
