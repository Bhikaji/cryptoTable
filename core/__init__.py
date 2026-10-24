"""Core trading domain: market data, orders, risk, and paper-trading primitives.

These modules are deliberately independent of the ``ai`` learning subsystem so
that the risk manager can never be influenced by reward signals.
"""

from core.config import (
    ContinuousTrainingConfig,
    DriftConfig,
    ModelPromotionConfig,
    PaperTradingConfig,
    RewardConfig,
    RiskConfig,
    SupervisedTrainingConfig,
)

__all__ = [
    "ContinuousTrainingConfig",
    "DriftConfig",
    "ModelPromotionConfig",
    "PaperTradingConfig",
    "RewardConfig",
    "RiskConfig",
    "SupervisedTrainingConfig",
]
