"""Deployment gates — paper-trading readiness checks (Section 18.7).

All learning and policy experiments run in historical simulation / paper trading
first.  Before any live deployment the system must pass a configurable set of
criteria *defined before evaluation*: minimum paper trades, positive net
expectancy after costs, acceptable drawdown, stable results across market
regimes, and passing safety tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from core.config import PaperTradingConfig


@dataclass
class GateResult:
    passed: bool
    criteria: Dict[str, Dict]
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"passed": self.passed, "criteria": self.criteria, "reasons": self.reasons}


def max_drawdown(pnls: List[float]) -> float:
    equity = peak = mdd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)
    return mdd


class PaperTradingGate:
    def __init__(self, config: PaperTradingConfig | None = None) -> None:
        self.config = config or PaperTradingConfig()

    def evaluate(self, paper_experiences: List[Dict]) -> GateResult:
        cfg = self.config
        criteria: Dict[str, Dict] = {}
        reasons: List[str] = []
        ok = True

        n = len(paper_experiences)
        criteria["min_paper_trades"] = {
            "required": cfg.min_paper_trades, "actual": n,
            "passed": n >= cfg.min_paper_trades,
        }
        if n < cfg.min_paper_trades:
            ok = False
            reasons.append(f"insufficient paper trades: {n} < {cfg.min_paper_trades}")

        pnls = [float(e.get("net_pnl", 0.0)) for e in paper_experiences]
        expectancy = sum(pnls) / len(pnls) if pnls else 0.0
        criteria["net_expectancy"] = {
            "required": cfg.min_net_expectancy, "actual": expectancy,
            "passed": expectancy >= cfg.min_net_expectancy,
        }
        if expectancy < cfg.min_net_expectancy:
            ok = False
            reasons.append(f"net expectancy {expectancy:.4f} below required {cfg.min_net_expectancy}")

        mdd = max_drawdown(pnls)
        criteria["max_drawdown"] = {
            "required": cfg.max_drawdown_pct, "actual": mdd,
            "passed": mdd <= cfg.max_drawdown_pct,
        }
        if mdd > cfg.max_drawdown_pct:
            ok = False
            reasons.append(f"drawdown {mdd:.4f} exceeds {cfg.max_drawdown_pct}")

        regimes = {e.get("market_regime") for e in paper_experiences}
        criteria["multi_regime"] = {
            "required": cfg.require_multi_regime, "actual": sorted(r for r in regimes if r),
            "passed": (not cfg.require_multi_regime) or len(regimes) >= 2,
        }
        if cfg.require_multi_regime and len(regimes) < 2:
            ok = False
            reasons.append("results not stable across multiple market regimes")

        violations = sum(1 for e in paper_experiences if e.get("risk_violation"))
        criteria["safety_tests"] = {
            "required": cfg.safety_max_risk_violations, "actual": violations,
            "passed": violations <= cfg.safety_max_risk_violations,
        }
        if violations > cfg.safety_max_risk_violations:
            ok = False
            reasons.append(f"{violations} risk violations during paper trading")

        return GateResult(passed=ok, criteria=criteria, reasons=reasons)
