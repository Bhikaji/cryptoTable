"""End-to-end demonstration pipeline.

    python -m scripts.run_pipeline

Steps
-----
1. Generate synthetic, regime-aware market data.
2. Stage 1 — train a supervised baseline (chronological split, no leakage).
3. Register the baseline as the initial champion.
4. Stage 2 — run a paper backtest, recording every fill as PAPER.
5. Reward + classify every trade and store it in SQLite.
6. Stage 3 / 18.6 — batched continuous training: retrain a candidate, validate,
   compare to champion, promote only if criteria pass.
7. Stage 4 — (optional) train a REINFORCE research agent.

Run the dashboard afterwards with ``infynex-quant`` or
``uvicorn app.main:app --port 8000``.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (  # noqa: E402
    ContinuousTrainingConfig,
    DriftConfig,
    ModelPromotionConfig,
    PaperTradingConfig,
    RiskConfig,
    SupervisedTrainingConfig,
)
from core.gates import PaperTradingGate  # noqa: E402
from core.market_data import SyntheticMarketGenerator  # noqa: E402
from core.risk_manager import RiskManager  # noqa: E402
from ai.continuous_training import ContinuousTrainer  # noqa: E402
from ai.drift_detector import DriftDetector  # noqa: E402
from ai.experience_store import ExperienceStore  # noqa: E402
from ai.model_registry import ModelRegistry  # noqa: E402
from ai.outcome_classifier import OutcomeClassifier  # noqa: E402
from ai.policy_evaluator import PolicyEvaluator  # noqa: E402
from ai.predict import Predictor  # noqa: E402
from ai.reward_engine import RewardEngine  # noqa: E402
from ai.train import SupervisedTrainer  # noqa: E402
from core.backtest import BacktestConfig, PaperTradingEngine  # noqa: E402


def main() -> None:
    artifact_dir = "artifacts/models"
    db_path = "data/learning.db"
    os.makedirs("data", exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    print("[1] generating market data ...")
    gen = SyntheticMarketGenerator(symbols=["BTCUSDT", "ETHUSDT"], n_bars=900, seed=11)
    frames = gen.generate()
    regimes = gen.regime_labels(frames)

    print("[2] Stage 1 — supervised baseline training ...")
    trainer = SupervisedTrainer(SupervisedTrainingConfig())
    store = ExperienceStore(db_path)
    registry = ModelRegistry(store, ModelPromotionConfig(min_validation_samples=20,
                                                         min_retrain_interval_trades=10))
    champion = None
    for sym, df in frames.items():
        model, _ = trainer.train_from_frame(df, sym, artifact_dir, version=f"base-{sym}")
        registry.set_champion(model.version, model.metrics,
                              artifact_path=model.artifact_path,
                              feature_columns=model.feature_columns)
        champion = model
        print(f"    {sym}: train acc={model.metrics['accuracy']:.3f} "
              f"val acc={model.metrics['validation']['accuracy']:.3f}")

    print("[3] Stage 2 — paper backtest on a held-out tail ...")
    predictor = Predictor.from_registry(store)
    risk_manager = RiskManager(RiskConfig())
    bt = PaperTradingEngine(predictor, risk_manager, BacktestConfig(),
                            model_version=champion.version)
    result = bt.run(frames["BTCUSDT"], "BTCUSDT", regimes["BTCUSDT"])
    ct = ContinuousTrainer(
        store=store,
        reward_engine=RewardEngine(),
        classifier=OutcomeClassifier(),
        trainer=trainer,
        registry=registry,
        evaluator=PolicyEvaluator(registry.config),
        drift_detector=DriftDetector(DriftConfig()),
        artifact_dir=artifact_dir,
        continuous_config=ContinuousTrainingConfig(batch_size=5, single_trade_can_promote=False),
    )
    for trade in result.closed_trades:
        ct.record_trade(trade)
    # also record rejected setups as missed-setups
    from ai.outcome_classifier import OutcomeClassifier as OC
    cls = OC()
    for trade in result.rejected:
        c = cls.classify_rejection(trade)
        reward = RewardEngine().evaluate(trade, c.category)
        store.record_experience(trade, reward, c, training_eligible=False)

    print(f"    recorded {len(result.closed_trades)} paper trades, "
          f"{len(result.rejected)} rejected setups")
    print("[4] Stage 3 — batched continuous training & promotion gate ...")
    decision = ct.maybe_retrain()
    if decision:
        print(f"    promotion decision: promoted={decision.promoted} reason={decision.reason}")
    else:
        print("    no promotion (insufficient new experiences / criteria not met)")

    print("[5] Stage 4 — REINFORCE research agent ...")
    from ai.rl_agent import RLConfig, train_rl_agent

    agent, env, history = train_rl_agent(frames["ETHUSDT"], risk_manager,
                                         RLConfig(episodes=40, seed=3))
    print(f"    RL mean return first={history[0]:.4f} last={history[-1]:.4f}")

    print("[6] Section 18.7 — paper-trading deployment gate ...")
    paper_experiences = [
        r["payload"] for r in store.list_experiences(execution_kind="PAPER", limit=5000)
    ]
    gate = PaperTradingGate(
        PaperTradingConfig(min_paper_trades=5, max_drawdown_pct=1e9,
                           require_multi_regime=False, safety_max_risk_violations=0)
    )
    gate_result = gate.evaluate(paper_experiences)
    print(f"    live deployment allowed: {gate_result.passed} "
          f"(reasons: {gate_result.reasons or 'none'})")

    summary = {
        "generated_at": time.time(),
        "paper_trades": len(result.closed_trades),
        "rejected_setups": len(result.rejected),
        "reward_summary": store.reward_summary(),
        "expectancy": store.net_expectancy("PAPER"),
        "errors": store.error_category_counts(),
        "models": [m["version"] for m in store.list_model_versions()],
        "promotions": [p["decision"] for p in store.promotion_log()],
        "paper_gate": gate_result.to_dict(),
    }
    with open("data/pipeline_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[done] summary -> data/pipeline_summary.json")
    print("       start dashboard: uvicorn app.main:app --port 8000")
    store.close()


if __name__ == "__main__":
    main()
