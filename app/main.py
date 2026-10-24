"""FastAPI application exposing the AI Learning dashboard (Section 18.8).

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000

The dashboard reads from the SQLite experience store and lets the user inspect
every reward calculation and model promotion in full detail.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai.drift_detector import DriftDetector
from ai.experience_store import ExperienceStore
from core.config import DriftConfig

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DEFAULT_DB = os.environ.get("INFYNEX_DB", os.path.join("data", "learning.db"))

app = FastAPI(title="INFYNEX QUANT AI — Learning Dashboard", version="0.1.0")


def get_store(db_path: Optional[str] = None) -> ExperienceStore:
    return ExperienceStore(db_path or DEFAULT_DB)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# --------------------------------------------------------------------------- #
# Dashboard API
# --------------------------------------------------------------------------- #
@app.get("/api/learning/summary")
def summary(db: Optional[str] = None) -> Dict[str, Any]:
    store = get_store(db)
    data: Dict[str, Any] = {
        "reward": store.reward_summary(),
        "expectancy": store.net_expectancy(),
        "counts": {"total": store.count_experiences()},
        "paper": store.count_experiences("PAPER"),
        "live": store.count_experiences("LIVE"),
        "training_sets": store.training_set_versions(),
    }
    store.close()
    return data


@app.get("/api/learning/experiences")
def experiences(
    symbol: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = Query(100, le=5000),
    offset: int = 0,
    db: Optional[str] = None,
) -> List[Dict[str, Any]]:
    store = get_store(db)
    rows = store.list_experiences(symbol=symbol, execution_kind=kind, limit=limit, offset=offset)
    store.close()
    return rows


@app.get("/api/learning/experiences/{trade_id}")
def experience(trade_id: str, db: Optional[str] = None) -> Dict[str, Any]:
    store = get_store(db)
    row = store.get_experience(trade_id)
    store.close()
    if row is None:
        raise HTTPException(status_code=404, detail="experience not found")
    return row


@app.get("/api/learning/reward-history")
def reward_history(buckets: int = 30, db: Optional[str] = None) -> List[Dict[str, Any]]:
    store = get_store(db)
    data = store.reward_history(buckets=buckets)
    store.close()
    return data


@app.get("/api/learning/by-symbol")
def by_symbol(limit: int = 50, db: Optional[str] = None) -> List[Dict[str, Any]]:
    store = get_store(db)
    data = store.reward_by_group("symbol", limit=limit)
    store.close()
    return data


@app.get("/api/learning/by-regime")
def by_regime(limit: int = 50, db: Optional[str] = None) -> List[Dict[str, Any]]:
    store = get_store(db)
    data = store.reward_by_group("market_regime", limit=limit)
    store.close()
    return data


@app.get("/api/learning/errors")
def errors(db: Optional[str] = None) -> List[Dict[str, Any]]:
    store = get_store(db)
    data = store.error_category_counts()
    store.close()
    return data


@app.get("/api/learning/models")
def models(db: Optional[str] = None) -> List[Dict[str, Any]]:
    store = get_store(db)
    data = store.list_model_versions()
    store.close()
    return data


@app.get("/api/learning/promotions")
def promotions(db: Optional[str] = None) -> List[Dict[str, Any]]:
    store = get_store(db)
    data = store.promotion_log()
    store.close()
    return data


@app.get("/api/learning/drift")
def drift(db: Optional[str] = None) -> Dict[str, Any]:
    """Compute feature drift between the older half and recent window."""
    import pandas as pd

    store = get_store(db)
    rows = store.list_experiences(limit=5000)
    store.close()
    if len(rows) < 10:
        return {"drifted": False, "features": [], "warnings": ["insufficient data"]}

    from ai.features import FEATURE_COLUMNS

    frame = pd.DataFrame([r["payload"].get("features", {}) for r in rows])
    frame = frame.reindex(columns=FEATURE_COLUMNS).fillna(0.0)
    detector = DriftDetector(DriftConfig())
    window = max(1, len(frame) // 3)
    report = detector.detect(frame.iloc[: len(frame) - window], frame.iloc[-window:])
    return report.to_dict()


@app.get("/api/learning/expected")
def expected(db: Optional[str] = None) -> Dict[str, Any]:
    store = get_store(db)
    data = {
        "paper": store.net_expectancy("PAPER"),
        "live": store.net_expectancy("LIVE"),
        "overall": store.net_expectancy(),
    }
    store.close()
    return data


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    run()
