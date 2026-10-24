"""SQLite-backed experience store (Sections 18.2 and 18.6).

Stores every completed trade together with its full reward breakdown and
classification, keeps paper fills labelled separately from live fills, and
manages versioned training datasets plus a model / promotion audit trail.

The store is the single source of truth for the learning loop: record -> reward
-> store -> versioned dataset -> retrain candidate -> validate -> promote.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from core.types import (
    ExecutionKind,
    OutcomeCategory,
    OutcomeClassification,
    RewardBreakdown,
    Trade,
)


@dataclass
class ExperienceRecord:
    trade_id: str
    payload: Dict[str, Any]
    reward: Optional[RewardBreakdown] = None
    classification: Optional[OutcomeClassification] = None
    training_eligible: bool = True
    dataset_version: Optional[str] = None
    created_at: float = 0.0


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    trade_id            TEXT PRIMARY KEY,
    symbol              TEXT,
    timeframe           TEXT,
    execution_kind      TEXT,
    model_version       TEXT,
    market_regime       TEXT,
    net_pnl             REAL,
    gross_pnl           REAL,
    total_costs         REAL,
    reward              REAL,
    clipped_reward      REAL,
    category            TEXT,
    is_avoidable_error  INTEGER,
    triggers_safety     INTEGER,
    training_eligible   INTEGER,
    dataset_version     TEXT,
    payload             TEXT,
    created_at          REAL
);

CREATE INDEX IF NOT EXISTS idx_exp_symbol ON experiences(symbol);
CREATE INDEX IF NOT EXISTS idx_exp_kind ON experiences(execution_kind);
CREATE INDEX IF NOT EXISTS idx_exp_regime ON experiences(market_regime);
CREATE INDEX IF NOT EXISTS idx_exp_created ON experiences(created_at);

CREATE TABLE IF NOT EXISTS model_versions (
    version          TEXT PRIMARY KEY,
    name             TEXT,
    kind             TEXT,
    artifact_path   TEXT,
    feature_columns  TEXT,
    metrics          TEXT,
    trained_at       REAL,
    parent_version   TEXT,
    status           TEXT
);

CREATE TABLE IF NOT EXISTS promotions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_version TEXT,
    champion_version  TEXT,
    decision         TEXT,
    reason           TEXT,
    metrics          TEXT,
    created_at       REAL
);

CREATE TABLE IF NOT EXISTS training_sets (
    version     TEXT PRIMARY KEY,
    n_samples   INTEGER,
    source      TEXT,
    created_at  REAL
);
"""


class ExperienceStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Experiences
    # ------------------------------------------------------------------ #
    def record_experience(
        self,
        trade: Trade,
        reward: Optional[RewardBreakdown] = None,
        classification: Optional[OutcomeClassification] = None,
        training_eligible: bool = True,
    ) -> str:
        payload = trade.to_dict()
        now = time.time()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO experiences (
                trade_id, symbol, timeframe, execution_kind, model_version,
                market_regime, net_pnl, gross_pnl, total_costs, reward,
                clipped_reward, category, is_avoidable_error, triggers_safety,
                training_eligible, dataset_version, payload, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade.trade_id,
                trade.symbol,
                trade.timeframe,
                trade.execution_kind.value,
                trade.model_version,
                trade.market_regime.value,
                trade.net_pnl,
                trade.gross_pnl,
                trade.total_costs,
                reward.total_reward if reward else None,
                reward.clipped_reward if reward else None,
                classification.category.value if classification else None,
                int(classification.is_avoidable_error) if classification else 0,
                int(classification.triggers_safety_control) if classification else 0,
                int(training_eligible),
                None,
                json.dumps(payload),
                now,
            ),
        )
        self._conn.commit()
        return trade.trade_id

    def get_experience(self, trade_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM experiences WHERE trade_id=?", (trade_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_experience(row)

    def list_experiences(
        self,
        symbol: Optional[str] = None,
        execution_kind: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM experiences"
        clauses, params = [], []
        if symbol:
            clauses.append("symbol=?")
            params.append(symbol)
        if execution_kind:
            clauses.append("execution_kind=?")
            params.append(execution_kind)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at ASC LIMIT ? OFFSET ?"
        params += [limit, offset]
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_experience(r) for r in rows]

    def count_experiences(self, execution_kind: Optional[str] = None) -> int:
        if execution_kind:
            row = self._conn.execute(
                "SELECT COUNT(*) c FROM experiences WHERE execution_kind=?",
                (execution_kind,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) c FROM experiences").fetchone()
        return int(row["c"])

    # ------------------------------------------------------------------ #
    # Versioned training sets
    # ------------------------------------------------------------------ #
    def add_to_training_set(
        self,
        trade_ids: Iterable[str],
        set_version: Optional[str] = None,
        source: str = "continuous",
    ) -> str:
        trade_ids = list(trade_ids)
        if not trade_ids:
            raise ValueError("no trade ids supplied")
        if set_version is None:
            set_version = f"ds-{int(time.time() * 1000)}"
        now = time.time()
        # Mark eligible records as belonging to this dataset version.
        placeholders = ",".join("?" for _ in trade_ids)
        self._conn.execute(
            f"UPDATE experiences SET training_eligible=1, dataset_version=? "
            f"WHERE trade_id IN ({placeholders})",
            [set_version, *trade_ids],
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO training_sets (version, n_samples, source, created_at) "
            "VALUES (?,?,?,?)",
            (set_version, len(trade_ids), source, now),
        )
        self._conn.commit()
        return set_version

    def training_set_versions(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM training_sets ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def training_experiences(self, set_version: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT payload FROM experiences WHERE dataset_version=? ORDER BY created_at ASC",
            (set_version,),
        ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    # ------------------------------------------------------------------ #
    # Analytics for the dashboard (Section 18.8)
    # ------------------------------------------------------------------ #
    def reward_summary(self) -> Dict[str, Any]:
        total = self.count_experiences()
        if total == 0:
            return {"total": 0, "positive": 0, "negative": 0, "neutral": 0,
                    "avg_reward": 0.0, "avg_reward_clipped": 0.0}
        row = self._conn.execute(
            "SELECT COUNT(*) c, AVG(reward) avg_r, AVG(clipped_reward) avg_rc, "
            "SUM(CASE WHEN reward > 0.1 THEN 1 ELSE 0 END) pos, "
            "SUM(CASE WHEN reward < -0.1 THEN 1 ELSE 0 END) neg, "
            "SUM(CASE WHEN reward > -0.1 AND reward <= 0.1 THEN 1 ELSE 0 END) neu "
            "FROM experiences"
        ).fetchone()
        return {
            "total": int(row["c"]),
            "positive": int(row["pos"] or 0),
            "negative": int(row["neg"] or 0),
            "neutral": int(row["neu"] or 0),
            "avg_reward": float(row["avg_r"] or 0.0),
            "avg_reward_clipped": float(row["avg_rc"] or 0.0),
        }

    def reward_by_group(self, field: str, limit: int = 50) -> List[Dict[str, Any]]:
        allowed = {"symbol", "market_regime", "execution_kind", "category", "model_version"}
        if field not in allowed:
            raise ValueError(f"field must be one of {allowed}")
        rows = self._conn.execute(
            f"SELECT {field} k, COUNT(*) n, AVG(reward) avg_reward, AVG(net_pnl) avg_pnl "
            f"FROM experiences GROUP BY {field} ORDER BY avg_reward DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"group": r["k"], "count": int(r["n"]),
             "avg_reward": float(r["avg_reward"] or 0.0),
             "avg_pnl": float(r["avg_pnl"] or 0.0)}
            for r in rows
        ]

    def reward_history(self, buckets: int = 30) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT created_at, reward, net_pnl FROM experiences ORDER BY created_at ASC"
        ).fetchall()
        if not rows:
            return []
        bucket_size = max(1, len(rows) // buckets)
        out, cur = [], []
        for r in rows:
            cur.append(r)
            if len(cur) >= bucket_size:
                out.append(
                    {"ts": cur[-1]["created_at"],
                     "avg_reward": sum(x["reward"] or 0 for x in cur) / len(cur),
                     "avg_pnl": sum(x["net_pnl"] or 0 for x in cur) / len(cur),
                     "count": len(cur)}
                )
                cur = []
        if cur:
            out.append(
                {"ts": cur[-1]["created_at"],
                 "avg_reward": sum(x["reward"] or 0 for x in cur) / len(cur),
                 "avg_pnl": sum(x["net_pnl"] or 0 for x in cur) / len(cur),
                 "count": len(cur)}
            )
        return out

    def error_category_counts(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT category, COUNT(*) n FROM experiences "
            "WHERE category IN (?,?,?,?,?,?,?) GROUP BY category ORDER BY n DESC",
            (OutcomeCategory.POOR_ENTRY.value, OutcomeCategory.FALSE_BREAKOUT.value,
             OutcomeCategory.WRONG_REGIME.value, OutcomeCategory.EXECUTION_ERROR.value,
             OutcomeCategory.EXCESSIVE_COST.value, OutcomeCategory.RISK_VIOLATION.value,
             OutcomeCategory.DATA_QUALITY_FAILURE.value),
        ).fetchall()
        return [{"category": r["category"], "count": int(r["n"])} for r in rows]

    def net_expectancy(self, execution_kind: Optional[str] = None) -> Dict[str, Any]:
        q = "SELECT COUNT(*) n, AVG(net_pnl) exp_pnl, SUM(net_pnl) total, AVG(gross_pnl) exp_gross, AVG(total_costs) exp_cost FROM experiences"
        params: List[Any] = []
        if execution_kind:
            q += " WHERE execution_kind=?"
            params.append(execution_kind)
        row = self._conn.execute(q, params).fetchone()
        n = int(row["n"] or 0)
        return {
            "count": n,
            "expectancy": float(row["exp_pnl"] or 0.0),
            "total_pnl": float(row["total"] or 0.0),
            "expectancy_gross": float(row["exp_gross"] or 0.0),
            "expectancy_cost": float(row["exp_cost"] or 0.0),
        }

    # ------------------------------------------------------------------ #
    # Model registry audit
    # ------------------------------------------------------------------ #
    def record_model_version(
        self,
        version: str,
        name: str,
        kind: str,
        artifact_path: str,
        metrics: Dict[str, Any],
        feature_columns: Optional[list[str]] = None,
        trained_at: Optional[float] = None,
        parent_version: Optional[str] = None,
        status: str = "candidate",
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?,?,?,?)",
            (version, name, kind, artifact_path, json.dumps(feature_columns or []),
             json.dumps(metrics), trained_at or time.time(), parent_version, status),
        )
        self._conn.commit()

    def set_model_status(self, version: str, status: str) -> None:
        self._conn.execute(
            "UPDATE model_versions SET status=? WHERE version=?", (status, version)
        )
        self._conn.commit()

    def get_model_version(self, version: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM model_versions WHERE version=?", (version,)
        ).fetchone()
        return self._model_row(row) if row else None

    def list_model_versions(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM model_versions ORDER BY trained_at DESC"
        ).fetchall()
        return [self._model_row(r) for r in rows]

    def get_champion(self) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM model_versions WHERE status='champion' "
            "ORDER BY trained_at DESC LIMIT 1"
        ).fetchone()
        return self._model_row(row) if row else None

    def record_promotion(
        self,
        candidate_version: str,
        champion_version: Optional[str],
        decision: str,
        reason: str,
        metrics: Dict[str, Any],
    ) -> None:
        self._conn.execute(
            "INSERT INTO promotions (candidate_version, champion_version, decision, reason, metrics, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (candidate_version, champion_version, decision, reason,
             json.dumps(metrics), time.time()),
        )
        self._conn.commit()

    def promotion_log(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM promotions ORDER BY created_at DESC"
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d["metrics"] or "{}")
            out.append(d)
        return out

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_experience(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["payload"] = json.loads(d["payload"] or "{}")
        return d

    @staticmethod
    def _model_row(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["metrics"] = json.loads(d["metrics"] or "{}")
        d["feature_columns"] = json.loads(d.get("feature_columns") or "[]")
        return d
