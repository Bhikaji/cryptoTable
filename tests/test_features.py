"""Acceptance tests: no future information in features (Section 18.5 / 18.9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.features import (
    FEATURE_COLUMNS,
    assert_no_leakage,
    build_dataset,
    chronological_split,
    compute_indicators,
)
from core.config import SupervisedTrainingConfig


def test_label_is_not_a_feature_column():
    df = pd.DataFrame(
        np.cumsum(np.random.default_rng(0).normal(0, 1, 200)) + 100,
        columns=["close"],
    )
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"] = df[["open", "close"]].max(axis=1) + 0.5
    df["low"] = df[["open", "close"]].min(axis=1) - 0.5
    df["volume"] = 1000.0
    ds = build_dataset(df, SupervisedTrainingConfig())
    assert "label" not in ds.features.columns
    for col in ("future_return", "future_ret", "label_horizon_return", "target"):
        assert col not in ds.features.columns
    assert_no_leakage(ds)


def test_features_are_causal(market_df):
    full = compute_indicators(market_df)
    cut = len(market_df) - 60
    truncated = compute_indicators(market_df.iloc[:cut])
    # Rows well inside the truncation must be identical whether or not the
    # future bars exist.
    lo, hi = 60, cut - 40
    a = full.iloc[lo:hi][FEATURE_COLUMNS]
    b = truncated.iloc[lo:hi][FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(a.fillna(-999), b.fillna(-999), check_exact=False, atol=1e-12)


def test_label_depends_on_the_future(market_df):
    cfg = SupervisedTrainingConfig(label_horizon=12)
    ds_a = build_dataset(market_df, cfg)

    # Perturb a *future* bar only.
    df2 = market_df.copy()
    j = len(df2) - 30
    df2.loc[df2.index[j], "close"] *= 1.05
    df2.loc[df2.index[j], "high"] = max(df2.loc[df2.index[j], "high"], df2.loc[df2.index[j], "close"])
    ds_b = build_dataset(df2, cfg)

    # Early features are unchanged (no look-forward), but labels differ.
    pd.testing.assert_frame_equal(
        ds_a.features.iloc[:50], ds_b.features.iloc[:50], atol=1e-12
    )
    common = min(len(ds_a.labels), len(ds_b.labels))
    assert not ds_a.labels.iloc[:common].equals(ds_b.labels.iloc[:common])


def test_chronological_split_is_time_ordered(market_df):
    ds = build_dataset(market_df, SupervisedTrainingConfig())
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = chronological_split(ds, 0.6, 0.2)
    # No shuffling: the test block starts after the validation block.
    assert len(x_tr) + len(x_va) + len(x_te) == len(ds)
    assert x_te.index.min() > x_va.index.max() >= x_tr.index.max()


def test_feature_count_matches_schema(market_df):
    ds = build_dataset(market_df, SupervisedTrainingConfig())
    assert list(ds.features.columns) == FEATURE_COLUMNS
    assert len(ds.features) == len(ds.labels)
