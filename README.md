# INFYNEX QUANT AI

An adaptive **AI reinforcement-learning subsystem** for a crypto scalping
desktop application. It evaluates every completed trade, assigns a transparent
**reward or penalty**, classifies *why* an outcome happened, and uses
accumulated experience to improve future decisions — without ever letting a
single trade change live behaviour or letting the AI override the independent
risk manager.

This repository implements **Section 18: AI Reinforcement Learning — Reward,
Penalty, and Continuous Improvement**.

> The application *does* implement a real learning algorithm (a supervised
> baseline, outcome-feedback retraining, offline policy evaluation, and a
> genuine REINFORCE agent). It is **not** described as "RL-trained" merely
> because it computes a reward score.

---

## Architecture

```
core/                      trading core (reward-free, safety-critical)
├── config.py              all weights / thresholds (JSON-serialisable)
├── types.py               domain enums + Trade / RewardBreakdown dataclasses
├── market_data.py         regime-switching synthetic OHLCV generator
├── risk_manager.py        INDEPENDENT risk control (never sees a reward)
├── backtest.py            paper-trading engine (labelled PAPER fills)
└── gates.py               paper-trading → live deployment gate (18.7)

ai/                        learning subsystem
├── features.py            causal feature engineering + leakage-safe labels
├── train.py               Stage 1 — supervised baseline (chronological split)
├── predict.py             inference from a registered model version
├── reward_engine.py       Section 18.3 — configurable reward / penalty
├── outcome_classifier.py  Section 18.4 — 11-category error taxonomy
├── experience_store.py    SQLite store + versioned datasets + audit trail
├── policy_evaluator.py    Stage 3 — offline policy evaluation
├── drift_detector.py      PSI / KS drift detection (18.6)
├── model_registry.py      champion/candidate promotion + rollback gate
├── continuous_training.py Section 18.6 — the safe training loop
└── rl_agent.py            Stage 4 — REINFORCE research agent (optional)

app/                       desktop application
├── main.py                FastAPI backend + dashboard API
└── static/                self-contained AI Learning dashboard

scripts/run_pipeline.py    end-to-end demonstration (data → train → paper → promote)
tests/                     acceptance tests (Section 18.9)
```

## Install & run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) end-to-end demonstration (writes data/learning.db + artifacts/)
python -m scripts.run_pipeline

# 2) start the AI Learning dashboard
uvicorn app.main:app --host 0.0.0.0 --port 8000
#    open http://localhost:8000

# 3) run the acceptance test suite
pytest -q
```

## How it satisfies Section 18

### 18.2 Trade outcome evaluation
Every closed trade becomes a training record (`Trade` / `ExperienceStore`)
capturing identity, regime, entry-time features, prediction probabilities,
strategy conditions, fills, costs, funding, slippage, gross/net P&L, MFE/MAE,
exit reason, risk-manager decisions and compliance. Actual fills and
simulated paper fills are **never mixed** — each carries `execution_kind`.

### 18.3 Reward function
`ai/reward_engine.py` computes a risk-normalised, clipped, fully explainable
reward:

```
reward = pnl + risk + execution + compliance − risk_penalty − cost_penalty
         − repeated_entry_penalty − decision_error_penalty
```

* Returns are divided by capital at risk, so **position size cannot inflate the
  reward**.
* A profitable trade that **violated risk limits is gated to ≤ 0** — never
  unconditionally approved.
* A sound strategy followed by an ordinary loss stays mildly negative and is
  **not** labelled a bad decision.
* Leverage breaches, oversized positions, recovery trading, excessive frequency
  and repeated entries in unfavourable conditions are penalised.
* Extreme values are clipped; every component is recorded for audit.

### 18.4 Penalty & error classification
`ai/outcome_classifier.py` maps every outcome to one of 11 categories
(`VALID_SETUP_LOSS`, `POOR_ENTRY`, `FALSE_BREAKOUT`, `WRONG_REGIME`,
`EXCESSIVE_COST`, `RISK_VIOLATION`, `EXECUTION_ERROR`, `DATA_QUALITY_FAILURE`,
`GOOD_EXIT`, `PREMATURE_EXIT`, `MISSED_VALID_SETUP`). Learning penalties fall
primarily on **avoidable** errors. Risk violations and data-quality failures
**trigger safety controls independently of the reward**.

### 18.5 Learning architecture
Four stages: supervised baseline (`train.py`, chronological split, no leakage),
outcome feedback (`experience_store.py`), offline policy optimisation
(`policy_evaluator.py`, separate out-of-sample test), and an optional REINFORCE
agent (`rl_agent.py`) whose environment is hard-wired to the risk manager.

### 18.6 Safe continuous training
`ai/continuous_training.py` implements the exact 10-step workflow: record →
reward → store → versioned dataset → retrain candidate in isolation →
validate chronologically → test out-of-sample → compare to champion → promote
only if every criterion passes → keep the old model for rollback. **A single
trade can never promote a model.**

### 18.7 Paper trading before live
`core/gates.py` enforces pre-defined criteria (minimum paper trades, positive
net expectancy after costs, drawdown bound, multi-regime stability, safety
tests) before any live deployment.

### 18.8 Dashboard
The **AI Learning** page shows evaluated trades, reward distribution, expectancy
before/after costs, reward history, error frequencies, model versions,
candidate-vs-champion results, drift warnings and promotion reasons. Clicking a
trade reveals its exact reward breakdown.

### 18.9 Acceptance tests
`tests/` covers all required guarantees: net-P&L-after-costs evaluation, loss
recording, sound-loss vs poor-decision distinction, risk-violation gating,
outlier clipping, no future leakage, paper/live separation, one-trade promotion
block, unvalidated-candidate retention, risk-manager independence, rollback on
failed validation, and full auditability.

## Design principles

* **Capital preservation over trade frequency** — recovery trading and
  over-trading are penalised.
* **Safety is never learned away** — the risk manager takes no reward input.
* **Every decision is auditable and reproducible** — weights, breakdowns and
  promotion logs are persisted in SQLite.
