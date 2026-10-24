"""Acceptance tests for the Stage 4 RL research agent (Section 18.5)."""

from __future__ import annotations

from ai.rl_agent import (
    ACTION_LONG,
    RLConfig,
    RLEnvironment,
    train_rl_agent,
)
from core.config import RiskConfig
from core.risk_manager import RiskManager


def test_rl_agent_cannot_bypass_risk_manager(market_df):
    # Stop distance cap below the agent's proposal => every entry rejected.
    rm = RiskManager(RiskConfig(max_stop_distance_pct=0.001))
    env = RLEnvironment(df=market_df, risk_manager=rm, config=RLConfig())
    env.reset()
    _, reward, _, info = env.step(ACTION_LONG)
    assert info["risk_violation"] is True
    assert env.position == 0
    assert reward < 0  # penalised, not rewarded


def test_rl_training_runs_and_learns(market_df):
    rm = RiskManager(RiskConfig())
    agent, env, history = train_rl_agent(market_df, rm, RLConfig(seed=1), episodes=8)
    assert len(history) == 8
    assert agent.W.shape == (env.state_dim, 4)
