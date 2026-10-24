"""Stage 4 — reinforcement-learning research agent (optional).

This is a genuine policy-gradient (REINFORCE) agent operating in a simulated
trading environment.  It is **not** the deployed trading policy; it exists to
research learning from net returns, drawdown, risk violations and transaction
costs.  The environment consults the independent ``RiskManager`` before allowing
any entry, so the agent can never bypass the safety control.

State  : market features, current position, available risk budget, cost estimate
Actions: WAIT(0) | LONG(1) | SHORT(2) | CLOSE(3)
Reward : net return - drawdown penalty - risk-violation penalty - transaction cost
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ai.features import FEATURE_COLUMNS, compute_indicators
from core.risk_manager import RiskManager
from core.types import TradeProposal, TradeSide

ACTION_WAIT, ACTION_LONG, ACTION_SHORT, ACTION_CLOSE = 0, 1, 2, 3
ACTIONS = [ACTION_WAIT, ACTION_LONG, ACTION_SHORT, ACTION_CLOSE]


@dataclass
class RLConfig:
    gamma: float = 0.99
    lr: float = 0.05
    episodes: int = 150
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0002
    risk_penalty: float = 2.0
    drawdown_penalty: float = 0.5
    seed: int = 0


@dataclass
class RLEnvironment:
    df: pd.DataFrame
    risk_manager: RiskManager
    config: RLConfig = field(default_factory=RLConfig)
    equity: float = 10_000.0

    def __post_init__(self) -> None:
        self.ind = compute_indicators(self.df)
        self.n = len(self.ind)
        self.state_dim = len(FEATURE_COLUMNS) + 3  # features + position + risk + unrealized
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.position = 0          # 0 none, 1 long, -1 short
        self.entry_price = 0.0
        self.quantity = 0.0
        self.equity = 10_000.0
        self.peak_equity = self.equity
        self.max_drawdown = 0.0
        return self._state()

    def _state(self) -> np.ndarray:
        row = self.ind.iloc[self.t]
        feats = np.array(
            [float(row[c]) if pd.notna(row[c]) else 0.0 for c in FEATURE_COLUMNS],
            dtype=float,
        )
        feats = np.nan_to_num(feats)
        feats = np.clip(feats / (np.abs(feats) + 1e-6), -50, 50)
        unrealized = self._unrealized_return()
        return np.concatenate([feats, [float(self.position), 1.0, unrealized]])

    def _unrealized_return(self) -> float:
        if self.position == 0:
            return 0.0
        close = float(self.ind.iloc[self.t]["close"])
        return (close - self.entry_price) * self.quantity * self.position / self.equity

    def _advance_equity(self) -> None:
        if self.t == 0:
            return
        prev = float(self.ind.iloc[self.t - 1]["close"])
        cur = float(self.ind.iloc[self.t]["close"])
        self.equity += (cur - prev) * self.quantity * self.position
        self.peak_equity = max(self.peak_equity, self.equity)
        self.max_drawdown = max(self.max_drawdown, self.peak_equity - self.equity)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        cfg = self.config
        info = {"risk_violation": False}
        equity_before = self.equity
        position_before = self.position
        costs = 0.0
        penalty = 0.0

        if action in (ACTION_LONG, ACTION_SHORT):
            side = TradeSide.LONG if action == ACTION_LONG else TradeSide.SHORT
            close = float(self.ind.iloc[self.t]["close"])
            notional = self.equity * self.risk_manager.config.max_position_pct
            qty = notional / close
            proposal = TradeProposal(
                symbol="RL", side=side, entry_price=close, quantity=qty,
                leverage=1.0, stop_distance=0.02, equity=self.equity,
                open_exposure=0.0, last_entry_time=-1e18, now=float(self.t),
                entries_today=0, daily_pnl=0.0,
            )
            decision = self.risk_manager.evaluate(proposal)
            if decision.approved and self.position == 0:
                self.position = 1 if side == TradeSide.LONG else -1
                self.entry_price = close
                self.quantity = qty
                costs += notional * cfg.fee_rate
            else:
                info["risk_violation"] = True
                penalty += cfg.risk_penalty
        elif action == ACTION_CLOSE and self.position != 0:
            costs += abs(self.quantity) * self.entry_price * cfg.fee_rate
            self.position = 0
            self.quantity = 0.0

        # Advance one bar. If the position was already open, equity moves with price.
        if self.position == position_before and self.position != 0:
            self._advance_equity()

        self.equity -= costs
        reward = (self.equity - equity_before) / max(equity_before, 1e-9)
        reward -= penalty
        reward -= cfg.drawdown_penalty * self.max_drawdown / max(self.equity, 1e-9)

        self.t += 1
        done = self.t >= self.n - 1
        return self._state(), float(reward), done, info

    def close_all(self) -> None:
        self.position = 0
        self.quantity = 0.0


class REINFORCEAgent:
    """Softmax linear policy trained with vanilla policy-gradient."""

    def __init__(self, state_dim: int, n_actions: int = 4, config: RLConfig | None = None,
                 seed: int = 0) -> None:
        self.config = config or RLConfig()
        self.rng = np.random.default_rng(seed or self.config.seed)
        self.W = self.rng.normal(scale=0.01, size=(state_dim, n_actions))

    def _logits(self, state: np.ndarray) -> np.ndarray:
        return self.W.T @ state

    def _probs(self, state: np.ndarray) -> np.ndarray:
        z = self._logits(state) - self._logits(state).max()
        e = np.exp(z)
        return e / e.sum()

    def act(self, state: np.ndarray, explore: float = 1.0) -> int:
        p = self._probs(state)
        if explore > 0 and self.rng.random() < explore:
            return int(self.rng.choice(len(p), p=p))
        return int(np.argmax(p))

    def update(self, states: List[np.ndarray], actions: List[int], rewards: List[float]) -> float:
        """REINFORCE gradient update; returns the mean episode return."""
        disc = np.array([self.config.gamma ** i for i in range(len(rewards))])
        G = np.array(rewards) * disc
        G = (G - G.mean()) / (G.std() + 1e-9)  # baseline / normalisation
        logp_sum = np.zeros(self.W.shape)
        for state, a, g in zip(states, actions, G):
            one_hot = np.zeros(self.W.shape[1])
            one_hot[a] = 1.0
            probs = self._probs(state)
            logp = np.log(probs[a] + 1e-12)
            logp_sum += np.outer(state, (one_hot - probs) * logp * g)
        self.W += self.config.lr * logp_sum
        return float(np.mean(rewards))


def train_rl_agent(
    df: pd.DataFrame,
    risk_manager: RiskManager,
    config: RLConfig | None = None,
    episodes: Optional[int] = None,
) -> Tuple[REINFORCEAgent, RLEnvironment, List[float]]:
    config = config or RLConfig()
    episodes = episodes or config.episodes
    env = RLEnvironment(df=df, risk_manager=risk_manager, config=config)
    agent = REINFORCEAgent(env.state_dim, len(ACTIONS), config)
    history: List[float] = []
    for ep in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        done = False
        while not done:
            a = agent.act(state, explore=max(0.1, 1.0 - ep / episodes))
            ns, r, done, _ = env.step(a)
            states.append(state)
            actions.append(a)
            rewards.append(r)
            state = ns
        env.close_all()
        mean_ret = agent.update(states, actions, rewards)
        history.append(mean_ret)
    return agent, env, history
