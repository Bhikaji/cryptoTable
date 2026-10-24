"""Acceptance tests for safe continuous training (Section 18.6 / 18.9)."""

from __future__ import annotations

from ai.continuous_training import ContinuousTrainer
from ai.drift_detector import DriftDetector
from ai.features import FEATURE_COLUMNS
from ai.model_registry import ModelRegistry, PromotionDecision
from ai.outcome_classifier import OutcomeClassifier
from ai.policy_evaluator import PolicyEvaluator
from ai.reward_engine import RewardEngine
from ai.train import SupervisedTrainer
from core.config import (
    ContinuousTrainingConfig,
    DriftConfig,
    ModelPromotionConfig,
    SupervisedTrainingConfig,
)
from tests.conftest import make_features, make_trade


def build_ct(store, tmp_path, batch_size, min_retrain):
    trainer = SupervisedTrainer(SupervisedTrainingConfig(min_train_samples=20))
    reg = ModelRegistry(
        store,
        ModelPromotionConfig(min_validation_samples=20, min_retrain_interval_trades=min_retrain),
    )
    reg.set_champion(
        "base-1",
        {"net_expectancy": 0.0, "win_rate": 0.5, "max_drawdown": 0.2},
        artifact_path="",
        feature_columns=FEATURE_COLUMNS,
    )
    ct = ContinuousTrainer(
        store=store,
        reward_engine=RewardEngine(),
        classifier=OutcomeClassifier(),
        trainer=trainer,
        registry=reg,
        evaluator=PolicyEvaluator(reg.config),
        drift_detector=DriftDetector(DriftConfig()),
        artifact_dir=str(tmp_path / "models"),
        continuous_config=ContinuousTrainingConfig(
            batch_size=batch_size, single_trade_can_promote=False
        ),
    )
    return ct, reg


def record_experiences(ct, n, seed=0):
    for i in range(n):
        pnl = (i % 3) * 10.0 - 10.0  # mix of wins/losses
        t = make_trade(
            trade_id=f"T{i}",
            features=make_features(seed + i),
            gross_pnl=pnl,
            net_pnl=pnl,
            entry_fee=0.0,
            exit_fee=0.0,
        )
        ct.record_trade(t)


def test_record_trade_stores_reward_and_classification(store, tmp_path):
    ct, reg = build_ct(store, tmp_path, 50, 50)
    t = make_trade(trade_id="T-x", features=make_features(1))
    ct.record_trade(t)
    row = store.get_experience("T-x")
    assert row["reward"] is not None
    assert row["category"] == "GOOD_EXIT"


def test_single_trade_cannot_promote(store, tmp_path):
    ct, reg = build_ct(store, tmp_path, batch_size=50, min_retrain=50)
    ct.record_trade(make_trade(trade_id="T-a", features=make_features(0)))
    assert ct.maybe_retrain() is None
    assert reg.current_champion()["version"] == "base-1"
    # the experience is still pending (no dataset version assigned)
    assert len(ct.pending_experiences()) == 1


def test_batch_triggers_retrain_and_promotes_gate(store, tmp_path):
    ct, reg = build_ct(store, tmp_path, batch_size=10, min_retrain=10)
    record_experiences(ct, 70)
    assert len(ct.pending_experiences()) == 70

    decision = ct.maybe_retrain()
    assert isinstance(decision, PromotionDecision)
    # batch consumed -> no longer pending
    assert len(ct.pending_experiences()) == 0
    # a candidate was registered even if promotion was rejected
    assert len(reg.list_versions()) >= 2


def test_promotion_requires_improvement(store, tmp_path):
    ct, reg = build_ct(store, tmp_path, batch_size=10, min_retrain=10)
    # mixed labels but negative expectancy -> candidate must not promote
    for i in range(70):
        pnl = -5.0 if i % 2 else 1.0
        ct.record_trade(make_trade(trade_id=f"T{i}", features=make_features(i),
                                   gross_pnl=pnl, net_pnl=pnl, entry_fee=0.0, exit_fee=0.0))
    decision = ct.maybe_retrain()
    assert decision is not None
    assert decision.promoted is False
    assert reg.current_champion()["version"] == "base-1"
