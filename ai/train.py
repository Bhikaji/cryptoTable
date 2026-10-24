"""Stage 1 — supervised baseline learning (Section 18.5).

Trains a directional classifier on causal features with a strictly chronological
train/validation/test split.  Labels come from ``ai.features.build_dataset`` and
are therefore free of future information leakage.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ai.features import build_dataset, chronological_split, assert_no_leakage
from core.config import SupervisedTrainingConfig


@dataclass
class SupervisedModel:
    """A persisted, versioned classifier artifact."""

    version: str
    artifact_path: str
    feature_columns: list[str]
    metrics: Dict[str, Any]
    trained_at: float
    parent_version: Optional[str] = None

    def load(self) -> GradientBoostingClassifier:
        return joblib.load(self.artifact_path)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        model = self.load()
        X = X[self.feature_columns]
        return model.predict_proba(X)


@dataclass
class SupervisedTrainer:
    config: SupervisedTrainingConfig = field(default_factory=SupervisedTrainingConfig)

    # ------------------------------------------------------------------ #
    def prepare(self, df: pd.DataFrame, symbol: str = "SYMBOL"):
        ds = build_dataset(df, self.config, symbol=symbol)
        assert_no_leakage(ds)
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = chronological_split(
            ds, self.config.train_fraction, self.config.val_fraction
        )
        return ds, (x_tr, y_tr), (x_va, y_va), (x_te, y_te)

    # ------------------------------------------------------------------ #
    def _sample_weights(self, y: pd.Series) -> np.ndarray:
        classes, counts = np.unique(y.values, return_counts=True)
        weight_map = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
        return np.array([weight_map[c] for c in y.values], dtype=float)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        artifact_dir: str,
        version: Optional[str] = None,
        parent_version: Optional[str] = None,
    ) -> SupervisedModel:
        if len(X) < self.config.min_train_samples:
            raise ValueError(
                f"insufficient samples: {len(X)} < min {self.config.min_train_samples}"
            )
        version = version or f"m-{uuid.uuid4().hex[:8]}"
        os.makedirs(artifact_dir, exist_ok=True)
        model = GradientBoostingClassifier(random_state=self.config.random_state)
        model.fit(X, y.values, sample_weight=self._sample_weights(y))

        path = os.path.join(artifact_dir, f"{version}.joblib")
        joblib.dump(model, path)

        metrics = self.evaluate(model, X, y)
        return SupervisedModel(
            version=version,
            artifact_path=path,
            feature_columns=list(X.columns),
            metrics=metrics,
            trained_at=time.time(),
            parent_version=parent_version,
        )

    # ------------------------------------------------------------------ #
    def evaluate(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        y_pred = model.predict(X)
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            auc = roc_auc_score(y, proba[:, 1])
        else:
            auc = float("nan")
        return {
            "n": int(len(X)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "roc_auc": float(auc),
            "label_distribution": {str(k): int(v) for k, v in zip(*np.unique(y.values, return_counts=True))},
        }

    # ------------------------------------------------------------------ #
    def train_from_frame(
        self,
        df: pd.DataFrame,
        symbol: str,
        artifact_dir: str,
        version: Optional[str] = None,
        parent_version: Optional[str] = None,
    ) -> Tuple[SupervisedModel, Dict[str, Any]]:
        ds, (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = self.prepare(df, symbol)
        model = self.train(x_tr, y_tr, artifact_dir, version, parent_version)
        model.metrics["validation"] = self.evaluate(model.load(), x_va, y_va)
        model.metrics["test"] = self.evaluate(model.load(), x_te, y_te)
        return model, {"train": model.metrics, "val": model.metrics["validation"]}
