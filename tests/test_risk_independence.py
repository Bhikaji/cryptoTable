"""Acceptance tests: the risk manager is independent of AI rewards (18.4, 18.9)."""

from __future__ import annotations

import inspect

from core.config import RiskConfig
from core.risk_manager import RiskManager
from core.types import TradeProposal, TradeSide


def test_risk_manager_takes_no_reward_input():
    sig = inspect.signature(RiskManager.evaluate)
    assert "reward" not in sig.parameters
    assert "classification" not in sig.parameters
    assert "outcome" not in sig.parameters


def test_risk_manager_decision_is_deterministic():
    rm = RiskManager(RiskConfig())
    p = TradeProposal(symbol="BTCUSDT", side=TradeSide.LONG, entry_price=100,
                      quantity=5, equity=1000, stop_distance=0.02)
    d1 = rm.evaluate(p)
    d2 = rm.evaluate(p)
    assert d1.approved == d2.approved
    assert d1.reasons == d2.reasons


def test_oversized_position_is_rejected_regardless_of_outcome():
    rm = RiskManager(RiskConfig(max_position_pct=0.2))
    p = TradeProposal(symbol="BTCUSDT", side=TradeSide.LONG, entry_price=100,
                      quantity=50, equity=1000, stop_distance=0.02)  # notional 5000 > 200
    assert rm.evaluate(p).approved is False


def test_excessive_leverage_is_rejected():
    rm = RiskManager(RiskConfig(max_leverage=3))
    p = TradeProposal(symbol="BTCUSDT", side=TradeSide.LONG, entry_price=100,
                      quantity=5, leverage=5, equity=1000, stop_distance=0.02)
    assert rm.evaluate(p).approved is False


def test_missing_stop_loss_is_rejected():
    rm = RiskManager(RiskConfig())
    p = TradeProposal(symbol="BTCUSDT", side=TradeSide.LONG, entry_price=100,
                      quantity=5, equity=1000, stop_distance=None)
    assert rm.evaluate(p).approved is False


def test_risk_violation_gates_reward_independently():
    """A reward score can never unlock a blocked risk decision."""
    from ai.reward_engine import RewardEngine
    from tests.conftest import make_trade

    rm = RiskManager(RiskConfig())
    big = TradeProposal(symbol="BTCUSDT", side=TradeSide.LONG, entry_price=100,
                        quantity=50, equity=1000, stop_distance=0.02)
    assert rm.evaluate(big).approved is False  # risk manager blocks it

    # Even a hugely "profitable" record is gated to <= 0.
    t = make_trade(gross_pnl=1000.0, net_pnl=1000.0, risk_violation=True,
                   risk_manager_approved=False)
    assert RewardEngine().evaluate(t).total_reward <= 0
