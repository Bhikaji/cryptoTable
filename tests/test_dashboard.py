"""Acceptance tests for the AI Learning dashboard (Section 18.8)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ai.experience_store import ExperienceStore
from ai.reward_engine import RewardEngine
from app.main import app
from tests.conftest import make_features, make_trade

client = TestClient(app)


def populate(db_path) -> None:
    store = ExperienceStore(db_path)
    for i in range(5):
        t = make_trade(trade_id=f"T{i}", features=make_features(i),
                       gross_pnl=(i % 2) * 10.0 - 5.0,
                       net_pnl=(i % 2) * 9.0 - 5.0)
        store.record_experience(t, RewardEngine().evaluate(t))
    store.close()


def test_index_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "INFYNEX" in r.text


def test_summary_and_experiences(tmp_path):
    db = str(tmp_path / "dash.db")
    populate(db)
    s = client.get("/api/learning/summary", params={"db": db}).json()
    assert s["reward"]["total"] == 5
    assert s["counts"]["total"] == 5

    rows = client.get("/api/learning/experiences", params={"db": db}).json()
    assert len(rows) == 5

    one = client.get("/api/learning/experiences/T0", params={"db": db}).json()
    assert one["payload"]["symbol"] == "BTCUSDT"


def test_missing_experience_returns_404(tmp_path):
    db = str(tmp_path / "dash2.db")
    populate(db)
    assert client.get("/api/learning/experiences/nope", params={"db": db}).status_code == 404


def test_analytics_endpoints(tmp_path):
    db = str(tmp_path / "dash3.db")
    populate(db)
    for path in (
        "/api/learning/errors",
        "/api/learning/models",
        "/api/learning/promotions",
        "/api/learning/drift",
        "/api/learning/by-symbol",
        "/api/learning/by-regime",
        "/api/learning/reward-history",
        "/api/learning/expected",
    ):
        assert client.get(path, params={"db": db}).status_code == 200
