"""Drift detection (Section 18.6).

Compares a recent window of feature distributions against a reference window
using the Population Stability Index (PSI) and a Kolmogorov–Smirnov test.  Drift
warnings feed the promotion gate so a degraded candidate is not promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from core.config import DriftConfig


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    direction: str = "stable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": round(float(self.psi), 4),
            "ks_stat": round(float(self.ks_stat), 4),
            "ks_pvalue": round(float(self.ks_pvalue), 6),
            "direction": self.direction,
        }


@dataclass
class DriftReport:
    drifted: bool
    features: List[FeatureDrift]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drifted": self.drifted,
            "features": [f.to_dict() for f in self.features],
            "warnings": self.warnings,
        }


class DriftDetector:
    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        # Quantile-based binning from the reference distribution.
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        exp_counts = np.histogram(expected, bins=edges)[0].astype(float) + 1e-6
        act_counts = np.histogram(actual, bins=edges)[0].astype(float) + 1e-6
        exp_pct = exp_counts / exp_counts.sum()
        act_pct = act_counts / act_counts.sum()
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    def detect(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        cfg = self.config
        features: List[FeatureDrift] = []
        warnings: List[str] = []

        for col in reference.columns:
            if col not in current.columns:
                continue
            exp = reference[col].dropna().values
            act = current[col].dropna().values
            if len(exp) < 2 or len(act) < 2:
                continue
            psi = self._psi(exp, act)
            ks_stat, ks_p = stats.ks_2samp(exp, act)
            direction = "stable"
            if psi > cfg.psi_threshold:
                direction = "drift"
                warnings.append(f"feature '{col}' PSI={psi:.2f} > {cfg.psi_threshold}")
            if ks_p < cfg.ks_pvalue_threshold:
                direction = "drift"
                warnings.append(
                    f"feature '{col}' KS p={ks_p:.2e} < {cfg.ks_pvalue_threshold}"
                )
            features.append(
                FeatureDrift(col, psi, float(ks_stat), float(ks_p), direction)
            )

        return DriftReport(drifted=len(warnings) > 0, features=features, warnings=warnings)
