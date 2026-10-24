"""Acceptance tests for the model registry / promotion gate (Section 18.6)."""

from __future__ import annotations

import pytest

from ai.model_registry import ModelRegistry
from ai.policy_evaluator import PolicyEvaluation
from core.config import ModelPromotionConfig


def make_evaluation(**kw) -> PolicyEvaluation:
    base = dict(
        candidate_version="cand-1",
        champion_version="base-1",
        n_samples=200,
        candidate_metrics={},
        champion_metrics={},
        candidate_expectancy=0.010,
        champion_expectancy=0.000,
        candidate_win_rate=0.55,
        champion_win_rate=0.50,
        candidate_max_drawdown=0.05,
        champion_max_drawdown=0.20,
        safety_pass=True,
        drift_drifted=False,
    )
    base.update(kw)
    ev = PolicyEvaluation(**base)
    ev.check(ModelPromotionConfig())
    return ev


def setup_registry(store) -> ModelRegistry:
    reg = ModelRegistry(store, ModelPromotionConfig(min_validation_samples=20))
    reg.set_champion("base-1", {"net_expectancy": 0.0, "win_rate": 0.5, "max_drawdown": 0.2},
                     artifact_path="/tmp/base.joblib", feature_columns=["a", "b"])
    return reg


def test_candidate_is_not_champion_until_promoted(store):
    reg = setup_registry(store)
    reg.register_candidate("cand-1", "cand-1", "/tmp/c.joblib", {}, feature_columns=["a", "b"])
    assert reg.current_champion()["version"] == "base-1"


def test_unvalidated_or_failing_candidate_cannot_replace_champion(store):
    reg = setup_registry(store)
    # failing evaluation: expectancy actually worse
    bad = make_evaluation(candidate_expectancy=-0.01, candidate_win_rate=0.4,
                          candidate_max_drawdown=0.30)
    decision = reg.promote(bad)
    assert decision.promoted is False
    assert reg.current_champion()["version"] == "base-1"
    assert any(p["decision"] == "retained" for p in reg.promotion_log())


def test_promoted_candidate_becomes_champion(store):
    reg = setup_registry(store)
    good = make_evaluation()
    decision = reg.promote(good)
    assert decision.promoted is True
    assert reg.current_champion()["version"] == "cand-1"
    assert any(p["decision"] == "promoted" for p in reg.promotion_log())


def test_safety_failure_blocks_promotion(store):
    reg = setup_registry(store)
    unsafe = make_evaluation(safety_pass=False)
    assert reg.promote(unsafe).promoted is False
    assert reg.current_champion()["version"] == "base-1"


def test_drift_blocks_promotion(store):
    reg = setup_registry(store)
    drifted = make_evaluation(drift_drifted=True)
    assert reg.promote(drifted).promoted is False


def test_rollback_restores_previous_model(store):
    reg = setup_registry(store)
    reg.promote(make_evaluation())
    assert reg.current_champion()["version"] == "cand-1"
    reg.rollback("base-1")
    assert reg.current_champion()["version"] == "base-1"


def test_promotion_decision_is_auditable(store):
    reg = setup_registry(store)
    reg.promote(make_evaluation())
    log = reg.promotion_log()
    assert log
    entry = log[0]
    assert entry["candidate_version"] == "cand-1"
    assert entry["reason"]
    assert entry["metrics"]


def test_minimum_sample_size_gate(store):
    reg = setup_registry(store)
    small = make_evaluation(n_samples=5)
    assert reg.promote(small).promoted is False
