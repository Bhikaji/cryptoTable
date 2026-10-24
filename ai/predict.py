"""Inference — load a registered model version and produce probabilities."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from ai.experience_store import ExperienceStore


class Predictor:
    """Predicts outcome probabilities for a single feature row."""

    def __init__(
        self,
        artifact_path: str,
        feature_columns: list[str],
        version: str = "unknown",
    ) -> None:
        self.artifact_path = artifact_path
        self.feature_columns = feature_columns
        self.version = version
        self._model = None

    @classmethod
    def from_registry(cls, store: ExperienceStore, version: Optional[str] = None) -> "Predictor":
        record = None
        if version:
            record = store.get_model_version(version)
        else:
            record = store.get_champion()
        if record is None:
            raise ValueError("no model version available in registry")
        return cls(
            artifact_path=record["artifact_path"],
            feature_columns=record.get("feature_columns", []),
            version=record["version"],
        )

    @property
    def model(self):
        if self._model is None:
            self._model = joblib.load(self.artifact_path)
        return self._model

    def predict_proba(self, row: Dict[str, float]) -> Dict[str, float]:
        X = pd.DataFrame([row])
        X = X.reindex(columns=self.feature_columns)
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        return {str(c): float(p) for c, p in zip(classes, proba)}

    def up_probability(self, row: Dict[str, float]) -> float:
        proba = self.predict_proba(row)
        if "1" in proba:
            return proba["1"]
        # classes may be strings
        for k, v in proba.items():
            if str(k) in ("1", "1.0", "up"):
                return v
        return 0.0
