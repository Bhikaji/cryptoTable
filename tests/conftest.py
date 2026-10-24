"""Shared fixtures for the acceptance test suite."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from ai.features import FEATURE_COLUMNS
from ai.experience_store import ExperienceStore
from core.market_data import SyntheticMarketGenerator
from core.types import (
    DataQuality,
    ExecutionKind,
    ExitReason,
    MarketRegime,
    Trade,
    TradeSide,
)


def make_trade(**kwargs: Any) -> Trade:
    """A clean, profitable, strategy-compliant paper trade by default."""
    base: Dict[str, Any] = dict(
        trade_id="T-1",
        symbol="BTCUSDT",
        timeframe="1m",
        side=TradeSide.LONG,
        model_version="m1",
        execution_kind=ExecutionKind.PAPER,
        entry_time=0.0,
        entry_price=100.0,
        exit_time=10.0,
        exit_price=101.0,
        exit_reason=ExitReason.TAKE_PROFIT,
        exit_planned=True,
        quantity=10.0,
        position_size=1000.0,
        leverage=1.0,
        stop_distance=0.02,
        holding_duration=10.0,
        market_regime=MarketRegime.TRENDING_UP,
        strategy_followed=True,
        execution_matched=True,
        data_quality=DataQuality.OK,
        gross_pnl=10.0,
        net_pnl=9.0,
        entry_fee=0.5,
        exit_fee=0.5,
    )
    base.update(kwargs)
    return Trade(**base)


def make_features(seed: int = 0) -> Dict[str, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    return {c: float(rng.normal(0, 0.01)) for c in FEATURE_COLUMNS}


@pytest.fixture
def make_trade_fixture():
    return make_trade


@pytest.fixture
def market_df():
    gen = SyntheticMarketGenerator(symbols=["BTCUSDT"], n_bars=400, seed=5)
    return gen.generate()["BTCUSDT"]


@pytest.fixture
def store(tmp_path):
    s = ExperienceStore(str(tmp_path / "test.db"))
    yield s
    s.close()
