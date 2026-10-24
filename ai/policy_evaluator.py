"""Stage 3 — offline policy evaluation (Section 18.5).

Evaluates a candidate model/strategy against historical data and stored
experiences using chronological validation and a separate untouched out-of-sample
window.  Produces the metrics the promotion gate compares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.config import ModelPromotionConfig


def max_drawdown(pnls: List[float]) -> float:
    """Maximum peak-to-trough drawdown of a cumulative P&L equity curve."""
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)
    return float(mdd)


@dataclass
class PolicyEvaluation:
    candidate_version: str
    champion_version: Optional[str]
    n_samples: int
    candidate_metrics: Dict[str, Any]
    champion_metrics: Dict[str, Any]
    candidate_expectancy: float
    champion_expectancy: float
    candidate_win_rate: float
    champion_win_rate: float
    candidate_max_drawdown: float
    champion_max_drawdown: float
    safety_pass: bool
    drift_drifted: bool
    risks_to_check: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    criteria: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.safety_pass and not self.drift_drifted and all(
            v.get("passed", False) for v in self.criteria.values()
        ) if self.criteria else (self.safety_pass and not self.drift_drifted)

    def check(self, config: ModelPromotionConfig):
        """Run the promotion criteria and populate ``criteria``/``reasons``."""
        criteria: Dict[str, Any] = {}
        reasons: List[str] = []
        ok = True

        if self.n_samples < config.min_validation_samples:
            ok = False
            reasons.append(f"insufficient samples {self.n_samples} < {config.min_validation_samples}")
        criteria["min_validation_samples"] = {
            "required": config.min_validation_samples,
            "actual": self.n_samples,
            "passed": self.n_samples >= config.min_validation_samples,
        }

        exp_gain = self.candidate_expectancy - self.champion_expectancy
        criteria["expectancy_improvement"] = {
            "required": config.min_expectancy_improvement,
            "actual": exp_gain,
            "passed": exp_gain >= config.min_expectancy_improvement,
        }
        if exp_gain < config.min_expectancy_improvement:
            ok = False
            reasons.append(f"expectancy improvement {exp_gain:.5f} too small")

        win_gain = self.candidate_win_rate - self.champion_win_rate
        criteria["win_rate_improvement"] = {
            "required": config.min_win_rate_improvement,
            "actual": win_gain,
            "passed": win_gain >= config.min_win_rate_improvement,
        }
        if win_gain < config.min_win_rate_improvement:
            ok = False
            reasons.append(f"win rate change {win_gain:.4f} too small")

        dd_ok = self.candidate_max_drawdown <= self.champion_max_drawdown * (
            1 + config.max_drawdown_worsening_pct
        )
        criteria["drawdown"] = {
            "candidate": self.candidate_max_drawdown,
            "champion": self.champion_max_drawdown,
            "passed": dd_ok,
        }
        if not dd_ok:
            ok = False
            reasons.append("drawdown worsened beyond tolerance")

        criteria["safety_pass"] = {
            "required": config.require_safety_pass,
            "actual": self.safety_pass,
            "passed": (not config.require_safety_pass) or self.safety_pass,
        }
        if config.require_safety_pass and not self.safety_pass:
            ok = False
            reasons.append("safety checks failed")

        criteria["no_drift"] = {
            "required": config.require_no_drift,
            "actual": not self.drift_drifted,
            "passed": (not config.require_no_drift) or (not self.drift_drifted),
        }
        if config.require_no_drift and self.drift_drifted:
            ok = False
            reasons.append("feature drift detected")

        self.criteria = criteria
        self.reasons = reasons
        self._ok = ok
        return ok

    @property
    def ok(self) -> bool:
        return getattr(self, "_ok", False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_version": self.candidate_version,
            "champion_version": self.champion_version,
            "n_samples": self.n_samples,
            "candidate_expectancy": self.candidate_expectancy,
            "champion_expectancy": self.champion_expectancy,
            "candidate_win_rate": self.candidate_win_rate,
            "champion_win_rate": self.champion_win_rate,
            "candidate_max_drawdown": self.candidate_max_drawdown,
            "champion_max_drawdown": self.champion_max_drawdown,
            "safety_pass": self.safety_pass,
            "drift_drifted": self.drift_drifted,
            "passed": self.ok,
            "reasons": self.reasons,
            "criteria": self.criteria,
        }


class PolicyEvaluator:
    def __init__(self, config: ModelPromotionConfig | None = None) -> None:
        self.config = config or ModelPromotionConfig()

    def evaluate(
        self,
        candidate_version: str,
        candidate_metrics: Dict[str, Any],
        n_samples: int,
        experiences: List[Dict[str, Any]],
        champion_version: Optional[str] = None,
        champion_metrics: Optional[Dict[str, Any]] = None,
        drift_report=None,
    ) -> PolicyEvaluation:
        champion_metrics = champion_metrics or {}
        pnls = [float(e.get("net_pnl", 0.0)) for e in experiences]
        expectancy = sum(pnls) / len(pnls) if pnls else 0.0
        win_rate = (sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else 0.0
        mdd = max_drawdown(pnls)

        safety_pass = all(not e.get("risk_violation", False) for e in experiences)
        # Treat stored safety flags on the evaluation window as disqualifying.
        safety_pass = safety_pass and not any(
            bool(e.get("_safety_flag")) for e in experiences
        )

        ev = PolicyEvaluation(
            candidate_version=candidate_version,
            champion_version=champion_version,
            n_samples=n_samples,
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
            candidate_expectancy=expectancy,
            champion_expectancy=float(champion_metrics.get("net_expectancy", 0.0)),
            candidate_win_rate=win_rate,
            champion_win_rate=float(champion_metrics.get("win_rate", 0.0)),
            candidate_max_drawdown=mdd,
            champion_max_drawdown=float(champion_metrics.get("max_drawdown", 0.0)),
            safety_pass=safety_pass,
            drift_drifted=bool(drift_report.drifted) if drift_report is not None else False,
        )
        ev.check(self.config)
        return ev
