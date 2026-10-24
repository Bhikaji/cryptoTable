"""Acceptance tests for the experience store (Sections 18.2, 18.7, 18.8)."""

from __future__ import annotations

from ai.experience_store import ExperienceStore
from ai.reward_engine import RewardEngine
from core.types import ExecutionKind, ExitReason, MarketRegime, TradeSide
from tests.conftest import make_trade


def test_paper_and_live_fills_are_labelled_separately(store):
    paper = make_trade(trade_id="P1", execution_kind=ExecutionKind.PAPER, gross_pnl=5.0, net_pnl=4.5)
    live = make_trade(trade_id="L1", execution_kind=ExecutionKind.LIVE, gross_pnl=5.0, net_pnl=4.5)
    store.record_experience(paper, RewardEngine().evaluate(paper))
    store.record_experience(live, RewardEngine().evaluate(live))

    assert store.count_experiences("PAPER") == 1
    assert store.count_experiences("LIVE") == 1
    assert store.count_experiences() == 2

    p = store.get_experience("P1")
    l = store.get_experience("L1")
    assert p["payload"]["execution_kind"] == "PAPER"
    assert l["payload"]["execution_kind"] == "LIVE"


def test_record_roundtrip_preserves_payload(store):
    t = make_trade(trade_id="T-rt", market_regime=MarketRegime.RANGING,
                   exit_reason=ExitReason.STOP_LOSS)
    reward = RewardEngine().evaluate(t)
    store.record_experience(t, reward)
    row = store.get_experience("T-rt")
    assert row["payload"]["symbol"] == "BTCUSDT"
    assert row["payload"]["market_regime"] == "RANGING"
    assert row["reward"] == reward.total_reward
    assert row["category"] is None  # no classification supplied


def test_versioned_training_sets(store):
    ids = [f"T{i}" for i in range(3)]
    for tid in ids:
        store.record_experience(make_trade(trade_id=tid))
    ver = store.add_to_training_set(ids, source="continuous")
    versions = store.training_set_versions()
    assert any(v["version"] == ver for v in versions)
    assert len(store.training_experiences(ver)) == 3


def test_reward_and_error_analytics(store):
    for i, (gross, net) in enumerate([(10, 9), (-10, -9), (1, 0.5)]):
        t = make_trade(trade_id=f"T{i}", gross_pnl=gross, net_pnl=net)
        store.record_experience(t, RewardEngine().evaluate(t))
    summary = store.reward_summary()
    assert summary["total"] == 3
    assert summary["positive"] >= 1
    exp = store.net_expectancy("PAPER")
    assert exp["count"] == 3


def test_model_version_and_promotion_audit(store):
    store.record_model_version("v1", "v1", "champion", "/x", {}, feature_columns=["a"])
    store.record_promotion("v2", "v1", "retained", "reason", {})
    versions = store.list_model_versions()
    assert versions[0]["feature_columns"] == ["a"]
    assert store.promotion_log()[0]["decision"] == "retained"
