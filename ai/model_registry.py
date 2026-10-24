"""Model registry with a promotion / rollback gate (Section 18.6).

The registry never promotes a candidate on a single trade.  Promotion requires a
validated ``PolicyEvaluation`` that satisfies every configured safety and
performance criterion.  The previous champion is retained for rollback and every
decision is written to an immutable audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.experience_store import ExperienceStore
from ai.policy_evaluator import PolicyEvaluation
from core.config import ModelPromotionConfig


@dataclass
class PromotionDecision:
    promoted: bool
    reason: str
    candidate_version: str
    champion_version: Optional[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    criteria: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "candidate_version": self.candidate_version,
            "champion_version": self.champion_version,
            "metrics": self.metrics,
            "criteria": self.criteria,
        }


class ModelRegistry:
    def __init__(
        self,
        store: ExperienceStore,
        config: ModelPromotionConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or ModelPromotionConfig()

    # ------------------------------------------------------------------ #
    def register_candidate(
        self,
        version: str,
        name: str,
        artifact_path: str,
        metrics: Dict[str, Any],
        feature_columns: Optional[list[str]] = None,
        parent_version: Optional[str] = None,
    ) -> None:
        self.store.record_model_version(
            version=version,
            name=name,
            kind="candidate",
            artifact_path=artifact_path,
            metrics=metrics,
            feature_columns=feature_columns,
            parent_version=parent_version,
            status="candidate",
        )

    def set_champion(
        self,
        version: str,
        metrics: Dict[str, Any],
        artifact_path: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
    ) -> None:
        if artifact_path is None:
            existing = self.store.get_model_version(version)
            artifact_path = existing["artifact_path"] if existing else ""
        if feature_columns is None:
            existing = self.store.get_model_version(version)
            feature_columns = existing.get("feature_columns", []) if existing else []
        self.store.record_model_version(
            version=version,
            name=version,
            kind="champion",
            artifact_path=artifact_path,
            metrics=metrics,
            feature_columns=feature_columns,
            parent_version=None,
            status="champion",
        )
        # demote any existing champion
        for v in self.store.list_model_versions():
            if v["version"] != version and (v["kind"] == "champion" or v["status"] == "champion"):
                self.store.set_model_status(v["version"], "retired")

    def current_champion(self) -> Optional[Dict[str, Any]]:
        return self.store.get_champion()

    def list_versions(self) -> List[Dict[str, Any]]:
        return self.store.list_model_versions()

    def promotion_log(self) -> List[Dict[str, Any]]:
        return self.store.promotion_log()

    # ------------------------------------------------------------------ #
    def promote(self, evaluation: PolicyEvaluation) -> PromotionDecision:
        cfg = self.config
        # Single source of truth for the criteria lives on the evaluation.
        promoted = evaluation.ok
        criteria = evaluation.criteria
        reasons = evaluation.reasons
        reason = "promotion criteria satisfied" if promoted else "; ".join(reasons)
        champion_version = evaluation.champion_version

        self.store.record_promotion(
            candidate_version=evaluation.candidate_version,
            champion_version=champion_version,
            decision="promoted" if promoted else "retained",
            reason=reason,
            metrics={
                "candidate_expectancy": evaluation.candidate_expectancy,
                "champion_expectancy": evaluation.champion_expectancy,
                "candidate_win_rate": evaluation.candidate_win_rate,
                "champion_win_rate": evaluation.champion_win_rate,
            },
        )

        if promoted:
            self.set_champion(evaluation.candidate_version, evaluation.candidate_metrics)

        return PromotionDecision(
            promoted=promoted,
            reason=reason,
            candidate_version=evaluation.candidate_version,
            champion_version=champion_version,
            metrics=evaluation.candidate_metrics,
            criteria=criteria,
        )

    # ------------------------------------------------------------------ #
    def rollback(self, version: str) -> None:
        """Restore a previous model as champion (rollback path)."""
        rec = self.store.get_model_version(version)
        if rec is None:
            raise ValueError(f"unknown model version {version}")
        champion = self.current_champion()
        if champion and champion["version"] == version:
            return
        self.store.record_promotion(
            candidate_version=version,
            champion_version=champion["version"] if champion else None,
            decision="rolled_back",
            reason=f"rollback to {version}",
            metrics={},
        )
        # Demote every current champion, then promote the requested version.
        for v in self.store.list_model_versions():
            if v["version"] != version and (
                v["kind"] == "champion" or v["status"] == "champion"
            ):
                self.store.set_model_status(v["version"], "retired")
        self.store.set_model_status(version, "champion")
