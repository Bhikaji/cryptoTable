"""Acceptance tests for the outcome / error classifier (Section 18.4)."""

from __future__ import annotations

from ai.outcome_classifier import OutcomeClassifier
from core.types import DataQuality, ExitReason, OutcomeCategory
from tests.conftest import make_trade

clf = OutcomeClassifier()


def test_valid_setup_loss():
    t = make_trade(gross_pnl=-10.0, net_pnl=-9.0, entry_fee=0.5, exit_fee=0.5)
    c = clf.classify(t)
    assert c.category == OutcomeCategory.VALID_SETUP_LOSS
    assert c.is_avoidable_error is False
    assert c.triggers_safety_control is False


def test_poor_entry():
    t = make_trade(gross_pnl=-10.0, net_pnl=-9.0, entry_fee=0.5, exit_fee=0.5,
                   poor_entry=True, strategy_followed=False)
    assert clf.classify(t).category == OutcomeCategory.POOR_ENTRY


def test_false_breakout():
    t = make_trade(gross_pnl=-10.0, net_pnl=-9.0, false_breakout=True)
    assert clf.classify(t).category == OutcomeCategory.FALSE_BREAKOUT


def test_wrong_regime():
    t = make_trade(gross_pnl=-10.0, net_pnl=-9.0, wrong_regime=True)
    assert clf.classify(t).category == OutcomeCategory.WRONG_REGIME


def test_excessive_cost():
    t = make_trade(gross_pnl=-10.0, net_pnl=-9.0, excessive_cost=True)
    assert clf.classify(t).category == OutcomeCategory.EXCESSIVE_COST


def test_risk_violation():
    t = make_trade(risk_violation=True, risk_manager_approved=False)
    c = clf.classify(t)
    assert c.category == OutcomeCategory.RISK_VIOLATION
    assert c.triggers_safety_control is True


def test_execution_error():
    t = make_trade(execution_matched=False)
    c = clf.classify(t)
    assert c.category == OutcomeCategory.EXECUTION_ERROR
    assert c.triggers_safety_control is True


def test_data_quality_failure():
    t = make_trade(data_quality=DataQuality.STALE)
    c = clf.classify(t)
    assert c.category == OutcomeCategory.DATA_QUALITY_FAILURE
    assert c.triggers_safety_control is True


def test_premature_exit():
    t = make_trade(gross_pnl=-10.0, net_pnl=-9.0, exit_planned=False, exit_reason=ExitReason.MANUAL)
    assert clf.classify(t).category == OutcomeCategory.PREMATURE_EXIT


def test_good_exit():
    t = make_trade(gross_pnl=10.0, net_pnl=9.0)
    assert clf.classify(t).category == OutcomeCategory.GOOD_EXIT


def test_missed_valid_setup_recorded_separately():
    t = make_trade(risk_manager_approved=False, strategy_followed=True,
                   strategy_conditions={"a": True, "b": True})
    c = clf.classify_rejection(t)
    assert c.category == OutcomeCategory.MISSED_VALID_SETUP


def test_invalid_rejection_is_risk_violation():
    t = make_trade(risk_manager_approved=False, strategy_followed=False,
                   strategy_conditions={"a": False})
    assert clf.classify_rejection(t).category == OutcomeCategory.RISK_VIOLATION


def test_avoidable_error_flag():
    t = make_trade(gross_pnl=-10.0, net_pnl=-9.0, false_breakout=True)
    assert clf.classify(t).is_avoidable_error is True
