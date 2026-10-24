"""Central, JSON-serialisable configuration for every subsystem.

Every weight and threshold that the reward / penalty engine uses lives here so
that a reward can always be explained and reproduced from its recorded weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict


def _as_dict(obj) -> Dict:
    return asdict(obj)


# --------------------------------------------------------------------------- #
# Reward engine (Section 18.3)
# --------------------------------------------------------------------------- #
@dataclass
class RewardConfig:
    # Component weights (each component is normalised to roughly [-1, 1]).
    # The P&L term dominates so the sign of the outcome is preserved; quality
    # terms modulate magnitude and penalties punish avoidable errors.
    w_pnl: float = 2.0
    w_risk: float = 0.20
    w_execution: float = 0.20
    w_compliance: float = 0.20
    w_excess_risk: float = 2.0
    w_cost: float = 0.80
    w_repeated_entry: float = 1.0
    w_decision_error: float = 1.0

    # Normalisation / clipping.
    pnl_scale: float = 1.0
    cost_scale: float = 1.0
    reward_clip: float = 5.0

    # Default stop distance (fraction of entry price) used when a trade does
    # not carry one, so that ``risk_budget`` is always well defined.
    default_stop_distance: float = 0.02

    # Cost ratio above which a trade is flagged as execution-quality problem.
    excessive_cost_ratio: float = 0.50

    # Neutral band: rewards inside [-neutral_band, +neutral_band] are "neutral".
    neutral_band: float = 0.10

    def to_dict(self) -> Dict:
        return _as_dict(self)


# --------------------------------------------------------------------------- #
# Risk manager (independent safety control)
# --------------------------------------------------------------------------- #
@dataclass
class RiskConfig:
    max_position_pct: float = 0.20          # max notional per trade / equity
    max_leverage: float = 3.0
    max_total_exposure_pct: float = 1.00    # sum of open notional / equity
    max_daily_loss_pct: float = 0.05        # halt trading for the day beyond this
    max_stop_distance_pct: float = 0.05     # reject setups with too-wide stops
    min_stop_distance_pct: float = 0.0025
    cooldown_seconds: float = 60.0          # min gap between entries
    max_entries_per_day: int = 20
    require_stop_loss: bool = True

    def to_dict(self) -> Dict:
        return _as_dict(self)


# --------------------------------------------------------------------------- #
# Supervised training (Stage 1)
# --------------------------------------------------------------------------- #
@dataclass
class SupervisedTrainingConfig:
    label_horizon: int = 12                 # bars into the future used for label
    label_threshold: float = 0.0010         # min move to count as directional
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    # test_fraction is implied (1 - train - val) and is never touched until the
    # final out-of-sample evaluation.
    min_train_samples: int = 60
    random_state: int = 42

    def to_dict(self) -> Dict:
        return _as_dict(self)


# --------------------------------------------------------------------------- #
# Model promotion / continuous training (Section 18.6)
# --------------------------------------------------------------------------- #
@dataclass
class ModelPromotionConfig:
    min_validation_samples: int = 100
    min_experience_samples: int = 50
    min_expectancy_improvement: float = 0.0005   # net expectancy per trade
    min_win_rate_improvement: float = 0.0        # absolute
    max_drawdown_worsening_pct: float = 0.0      # candidate must not worsen DD
    require_safety_pass: bool = True
    require_no_drift: bool = True
    max_promotion_interval_trades: int = 1        # batch size before re-eval
    min_retrain_interval_trades: int = 50        # ignore tiny sample noise

    def to_dict(self) -> Dict:
        return _as_dict(self)


@dataclass
class ContinuousTrainingConfig:
    # Retrain a candidate only after this many new experiences accumulate.
    batch_size: int = 50
    # Never react to a single trade.
    single_trade_can_promote: bool = False
    use_chronological_split: bool = True
    separate_out_of_sample: bool = True

    def to_dict(self) -> Dict:
        return _as_dict(self)


# --------------------------------------------------------------------------- #
# Drift detection (Section 18.6)
# --------------------------------------------------------------------------- #
@dataclass
class DriftConfig:
    psi_threshold: float = 0.20
    ks_pvalue_threshold: float = 0.01
    min_reference_samples: int = 100
    window_size: int = 100

    def to_dict(self) -> Dict:
        return _as_dict(self)


# --------------------------------------------------------------------------- #
# Paper trading gate (Section 18.7)
# --------------------------------------------------------------------------- #
@dataclass
class PaperTradingConfig:
    min_paper_trades: int = 200
    min_net_expectancy: float = 0.0
    max_drawdown_pct: float = 0.15
    require_multi_regime: bool = True
    require_safety_tests: bool = True
    safety_max_risk_violations: int = 0

    def to_dict(self) -> Dict:
        return _as_dict(self)
