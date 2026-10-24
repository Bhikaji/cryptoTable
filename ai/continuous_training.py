"""Safe continuous training loop (Section 18.6).

Workflow implemented here:

    1. record a completed trade + outcome
    2. calculate reward and penalty components
    3. store the experience in SQLite
    4. add eligible records to a versioned training dataset
    5. retrain a candidate in isolation (never touching the champion)
    6. validate on chronological validation data
    7. evaluate on untouched out-of-sample data
    8. compare candidate vs champion
    9. promote only if every performance + safety criterion passes
   10. keep the previous model available for rollback

A single winning/losing trade can never promote a model: retraining is batched
and the minimum retrain interval is enforced.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ai.drift_detector import DriftDetector, DriftReport
from ai.experience_store import ExperienceStore
from ai.features import FEATURE_COLUMNS
from ai.model_registry import ModelRegistry, PromotionDecision
from ai.outcome_classifier import OutcomeClassifier
from ai.policy_evaluator import PolicyEvaluator, max_drawdown
from ai.reward_engine import RewardEngine
from ai.train import SupervisedModel, SupervisedTrainer
from core.config import ContinuousTrainingConfig, ModelPromotionConfig


class ContinuousTrainer:
    def __init__(
        self,
        store: ExperienceStore,
        reward_engine: RewardEngine,
        classifier: OutcomeClassifier,
        trainer: SupervisedTrainer,
        registry: ModelRegistry,
        evaluator: PolicyEvaluator,
        drift_detector: DriftDetector,
        artifact_dir: str = "artifacts/models",
        continuous_config: ContinuousTrainingConfig | None = None,
    ) -> None:
        self.store = store
        self.reward_engine = reward_engine
        self.classifier = classifier
        self.trainer = trainer
        self.registry = registry
        self.evaluator = evaluator
        self.drift_detector = drift_detector
        self.artifact_dir = artifact_dir
        self.config = continuous_config or ContinuousTrainingConfig()

    # ------------------------------------------------------------------ #
    # 1-3. Record, reward, classify, store
    # ------------------------------------------------------------------ #
    def record_trade(self, trade) -> Dict[str, Any]:
        classification = self.classifier.classify(trade)
        reward = self.reward_engine.evaluate(trade, classification.category)
        self.store.record_experience(trade, reward, classification, training_eligible=True)
        return {
            "trade_id": trade.trade_id,
            "reward": reward.total_reward,
            "clipped_reward": reward.clipped_reward,
            "classification": classification.category.value,
            "explanation": reward.explanation,
        }

    # ------------------------------------------------------------------ #
    # 4-10. Batched retrain / validate / promote
    # ------------------------------------------------------------------ #
    def pending_experiences(self) -> List[Dict[str, Any]]:
        rows = self.store.list_experiences(limit=10_000)
        return [r for r in rows if r["dataset_version"] is None]

    def maybe_retrain(self) -> Optional[PromotionDecision]:
        pending = self.pending_experiences()
        needed = max(self.config.batch_size, 1)

        # Rule: a single trade must never trigger a promotion.
        if len(pending) < needed and not self.config.single_trade_can_promote:
            return None
        if len(pending) < self.registry.config.min_retrain_interval_trades and not self.config.single_trade_can_promote:
            return None

        # Add eligible records to a versioned training dataset.
        set_version = self.store.add_to_training_set(
            [r["trade_id"] for r in pending], source="continuous"
        )

        # Build candidate data from entry-time features + realized outcomes.
        frame, experiences = self._build_candidate(pending)
        if frame is None or len(frame) < self.trainer.config.min_train_samples:
            self.store.record_promotion(
                candidate_version="(none)",
                champion_version=(self.registry.current_champion() or {}).get("version"),
                decision="retained",
                reason=f"insufficient candidate samples ({0 if frame is None else len(frame)})",
                metrics={},
            )
            return None

        # Chronological split (validation + untouched out-of-sample test).
        n = len(frame)
        n_tr = int(n * self.trainer.config.train_fraction)
        n_va = int(n * self.trainer.config.val_fraction)
        x_tr, y_tr = frame.iloc[:n_tr, :-1], frame.iloc[:n_tr, -1]
        x_va, y_va = frame.iloc[n_tr : n_tr + n_va, :-1], frame.iloc[n_tr : n_tr + n_va, -1]
        x_te, y_te = frame.iloc[n_tr + n_va :, :-1], frame.iloc[n_tr + n_va :, -1]

        version = f"c-{uuid.uuid4().hex[:8]}"
        candidate = self.trainer.train(
            x_tr, y_tr, self.artifact_dir, version=version,
            parent_version=(self.registry.current_champion() or {}).get("version"),
        )
        candidate.metrics["validation"] = self.trainer.evaluate(candidate.load(), x_va, y_va)
        candidate.metrics["test"] = self.trainer.evaluate(candidate.load(), x_te, y_te)
        self.registry.register_candidate(
            version, name=version, artifact_path=candidate.artifact_path,
            metrics=candidate.metrics,
            feature_columns=candidate.feature_columns,
            parent_version=(self.registry.current_champion() or {}).get("version"),
        )

        # Drift check: older reference window vs recent window.
        drift_report = self._drift_check(frame)

        # Evaluation window = validation experience payloads, in order.
        val_experiences = experiences[n_tr : n_tr + n_va]
        for e in val_experiences:
            e["_safety_flag"] = e.get("risk_violation", False)

        champion = self.registry.current_champion()
        champion_metrics = (champion or {}).get("metrics", {})
        if not champion_metrics:
            champion_metrics = self._champion_metrics_from_store()

        evaluation = self.evaluator.evaluate(
            candidate_version=version,
            candidate_metrics=candidate.metrics,
            n_samples=len(val_experiences),
            experiences=val_experiences,
            champion_version=(champion or {}).get("version"),
            champion_metrics=champion_metrics,
            drift_report=drift_report,
        )

        decision = self.registry.promote(evaluation)
        return decision

    # ------------------------------------------------------------------ #
    def _build_candidate(self, experiences: List[Dict[str, Any]]):
        if not experiences:
            return None, []
        champion = self.registry.current_champion()
        feature_columns = (champion or {}).get("feature_columns") or FEATURE_COLUMNS
        rows = []
        labels = []
        for e in experiences:
            payload = e.get("payload", e)
            feats = payload.get("features", {}) or {}
            rows.append({c: float(feats.get(c, 0.0)) for c in feature_columns})
            pnl = float(payload.get("net_pnl", 0.0))
            labels.append(1 if pnl > 0 else (-1 if pnl < 0 else 0))
        frame = pd.DataFrame(rows)
        frame["label"] = labels
        payloads = [e.get("payload", e) for e in experiences]
        return frame, payloads

    def _drift_check(self, frame: pd.DataFrame) -> DriftReport:
        features = [c for c in frame.columns if c != "label"]
        if len(frame) < 4:
            return DriftReport(drifted=False, features=[], warnings=[])
        window = min(self.drift_detector.config.window_size, len(frame) // 3)
        reference = frame[features].iloc[: len(frame) - window]
        current = frame[features].iloc[-window:]
        try:
            return self.drift_detector.detect(reference, current)
        except Exception:
            return DriftReport(drifted=False, features=[], warnings=[])

    def _champion_metrics_from_store(self) -> Dict[str, Any]:
        exps = self.store.list_experiences(limit=10_000)
        pnls = [float(e["net_pnl"] or 0.0) for e in exps]
        if not pnls:
            return {"net_expectancy": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}
        return {
            "net_expectancy": sum(pnls) / len(pnls),
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
            "max_drawdown": max_drawdown(pnls),
        }
