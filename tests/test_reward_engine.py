"""Acceptance tests for the reward engine (Section 18.3)."""

from __future__ import annotations

from ai.reward_engine import RewardEngine
from core.config import RewardConfig
from core.types import ExecutionKind, ExitReason, MarketRegime, OutcomeCategory, TradeSide
from tests.conftest import make_trade


def engine(**kw) -> RewardEngine:
    return RewardEngine(RewardConfig(**kw))


def test_winning_trade_is_positive_and_uses_net_pnl():
    t = make_trade(gross_pnl=10.0, net_pnl=9.0, entry_fee=0.5, exit_fee=0.5)
    r = engine().evaluate(t)
    assert r.total_reward > 0
    # risk-adjusted return must be based on net (after costs), not gross
    assert abs(r.risk_adjusted_return - (9.0 / t.risk_budget())) < 1e-9


def test_losing_trade_is_negative():
    t = make_trade(gross_pnl=-10.0, net_pnl=-9.0, entry_fee=0.5, exit_fee=0.5)
    r = engine().evaluate(t)
    assert r.total_reward < 0


def test_transaction_costs_included_in_outcome():
    base = make_trade(gross_pnl=10.0, net_pnl=9.0, entry_fee=0.5, exit_fee=0.5)
    high_cost = make_trade(gross_pnl=10.0, net_pnl=0.5, entry_fee=5.0, exit_fee=4.5)
    losing_after_cost = make_trade(gross_pnl=10.0, net_pnl=-0.5, entry_fee=5.25, exit_fee=5.25)

    r_base = engine().evaluate(base)
    r_cost = engine().evaluate(high_cost)
    r_lose = engine().evaluate(losing_after_cost)

    assert r_cost.total_reward < r_base.total_reward       # costs erode reward
    assert r_lose.total_reward < 0                         # gross profit but net loss


def test_sound_strategy_loss_is_distinguished_from_poor_decision():
    ordinary = make_trade(gross_pnl=-10.0, net_pnl=-9.0, entry_fee=0.5, exit_fee=0.5)
    poor = make_trade(
        gross_pnl=-10.0, net_pnl=-9.0, entry_fee=0.5, exit_fee=0.5,
        poor_entry=True, strategy_followed=False,
    )
    eng = engine()
    r_ordinary = eng.evaluate(ordinary, OutcomeCategory.VALID_SETUP_LOSS)
    r_poor = eng.evaluate(poor, OutcomeCategory.POOR_ENTRY)

    assert r_ordinary.total_reward < 0
    assert r_poor.total_reward < r_ordinary.total_reward   # avoidable error punished harder


def test_profitable_trade_with_risk_violation_not_unconditionally_approved():
    t = make_trade(
        gross_pnl=10.0, net_pnl=9.0, risk_violation=True,
        risk_manager_approved=False,
    )
    eng = engine()
    r = eng.evaluate(t, OutcomeCategory.RISK_VIOLATION)
    assert r.total_reward <= 0


def test_large_outlier_reward_is_clipped():
    t = make_trade(gross_pnl=1_000_000.0, net_pnl=1_000_000.0, entry_fee=0.0, exit_fee=0.0)
    eng = engine(reward_clip=1.0)
    r = eng.evaluate(t)
    assert r.clipped is True
    assert abs(r.clipped_reward) <= 1.0


def test_risk_normalisation_prevents_position_size_inflation():
    small = make_trade(position_size=1000.0, gross_pnl=20.0, net_pnl=20.0,
                       entry_fee=0.0, exit_fee=0.0)
    large = make_trade(trade_id="T-2", position_size=10_000.0, quantity=100.0,
                       gross_pnl=200.0, net_pnl=200.0, entry_fee=0.0, exit_fee=0.0)
    eng = engine()
    rs = eng.evaluate(small)
    rl = eng.evaluate(large)
    # identical return-on-risk => identical pnl component regardless of size
    assert abs(rs.raw_components["pnl_component"] - rl.raw_components["pnl_component"]) < 1e-9


def test_leverage_oversized_recovery_frequency_are_penalised():
    base = make_trade(gross_pnl=10.0, net_pnl=9.0)
    levered = make_trade(trade_id="T-2", gross_pnl=10.0, net_pnl=9.0, leverage_breach=True)
    oversized = make_trade(trade_id="T-3", gross_pnl=10.0, net_pnl=9.0, oversized_position=True)
    recovery = make_trade(trade_id="T-4", gross_pnl=10.0, net_pnl=9.0, recovery_trade=True)
    frequent = make_trade(trade_id="T-5", gross_pnl=10.0, net_pnl=9.0, excessive_frequency=True)

    eng = engine()
    base_r = eng.evaluate(base).total_reward
    assert eng.evaluate(levered).total_reward < base_r
    assert eng.evaluate(oversized).total_reward < base_r
    assert eng.evaluate(recovery).total_reward < base_r
    assert eng.evaluate(frequent).total_reward < base_r


def test_repeated_entry_in_unfavourable_conditions_penalised():
    base = make_trade(gross_pnl=10.0, net_pnl=9.0)
    repeated = make_trade(trade_id="T-2", gross_pnl=10.0, net_pnl=9.0,
                          repeated_entry_unfavorable=True)
    eng = engine()
    assert eng.evaluate(repeated).total_reward < eng.evaluate(base).total_reward


def test_reward_breakdown_is_explainable():
    t = make_trade()
    r = engine().evaluate(t)
    assert set(r.weighted_components) == set(r.raw_components)
    assert r.explanation
    assert abs(sum(r.weighted_components.values()) - r.total_reward) < 1e-9


def test_reward_buckets():
    assert RewardEngine.bucket(2.0) == "positive"
    assert RewardEngine.bucket(-2.0) == "negative"
    assert RewardEngine.bucket(0.0) == "neutral"
