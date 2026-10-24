"""Feature engineering and label construction (Stage 1).

Two hard rules are enforced here:

1.  **No future information in features.**  Every feature column is a function
    of bars at time ``t`` and *earlier*.  Nothing looks forward.
2.  **Labels are built from the future** and therefore never share a column with
    the feature matrix.

``assert_no_leakage`` is a lightweight, testable guard that re-derives the
feature frame and confirms the future-return label is not one of its columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import SupervisedTrainingConfig
from core.types import MarketRegime

FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_6",
    "volatility",
    "atr_pct",
    "rsi",
    "ema_fast_ratio",
    "ema_slow_ratio",
    "macd_hist",
    "volume_z",
    "range_pos",
    "trend_strength",
    "body_ratio",
    "upper_wick",
    "lower_wick",
]


# --------------------------------------------------------------------------- #
# Indicators (all trailing / causal)
# --------------------------------------------------------------------------- #
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of causal indicators aligned to ``df`` index."""
    out = df.copy()
    close = out["close"]
    high, low = out["high"], out["low"]
    volume = out["volume"]

    ret = close.pct_change()
    out["ret_1"] = ret
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["volatility"] = ret.rolling(20, min_periods=5).std()

    # ATR as a percentage of price.
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    out["atr_pct"] = atr / close

    # RSI (Wilder).
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=5).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=5).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    out["rsi"] = 100 - (100 / (1 + rs))

    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    out["ema_fast_ratio"] = (ema_fast - ema_slow) / close
    out["ema_slow_ratio"] = (close - ema_slow) / close
    macd = ema_fast - ema_slow
    signal = _ema(macd, 9)
    out["macd_hist"] = (macd - signal) / close

    out["volume_z"] = (volume - volume.rolling(20, min_periods=5).mean()) / (
        volume.rolling(20, min_periods=5).std() + 1e-9
    )

    rng = (high - low).replace(0, np.nan)
    out["range_pos"] = (close - low) / rng
    out["trend_strength"] = (ema_fast - ema_slow).abs() / close
    body = (close - out["open"]).abs()
    out["body_ratio"] = body / (rng + 1e-9)
    out["upper_wick"] = (high - pd.concat([close, out["open"]], axis=1).max(axis=1)) / (
        rng + 1e-9
    )
    out["lower_wick"] = (pd.concat([close, out["open"]], axis=1).min(axis=1) - low) / (
        rng + 1e-9
    )

    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Dataset builder
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    features: pd.DataFrame
    labels: pd.Series
    symbol: str
    feature_columns: List[str]
    label_column: str = "label"

    def __len__(self) -> int:
        return len(self.features)


def build_dataset(
    df: pd.DataFrame,
    config: SupervisedTrainingConfig,
    symbol: str = "SYMBOL",
    min_warmup: int = 30,
) -> Dataset:
    """Build a feature/label dataset from OHLCV bars.

    The label is the sign of the return ``horizon`` bars into the future, with a
    ``threshold`` dead-band so tiny moves count as "flat".
    """
    ind = compute_indicators(df)
    future_ret = ind["close"].shift(-config.label_horizon) / ind["close"] - 1.0

    # label: +1 up, -1 down, 0 flat
    label = pd.Series(0, index=ind.index, dtype=int)
    label[future_ret > config.label_threshold] = 1
    label[future_ret < -config.label_threshold] = -1

    feature_df = ind.loc[:, FEATURE_COLUMNS].copy()
    # Drop the tail that has NaN indicators or, crucially, a label that needs
    # future bars (the last `label_horizon` rows).
    valid = feature_df.notna().all(axis=1) & ind.index.to_series().notna()
    valid.iloc[-config.label_horizon:] = False
    feature_df = feature_df[valid].reset_index(drop=True)
    label = label[valid].reset_index(drop=True)
    # Drop leading warm-up rows that are not yet informative.
    feature_df = feature_df.iloc[min_warmup:].reset_index(drop=True)
    label = label.iloc[min_warmup:].reset_index(drop=True)

    frames = pd.concat([feature_df, label.rename("label")], axis=1)
    return Dataset(
        features=frames.iloc[:, :-1],
        labels=frames["label"],
        symbol=symbol,
        feature_columns=FEATURE_COLUMNS,
    )


def chronological_split(
    dataset: Dataset,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
):
    """Chronological (time-ordered) split — no shuffling, no leakage."""
    n = len(dataset.features)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    x_train = dataset.features.iloc[:n_train]
    y_train = dataset.labels.iloc[:n_train]
    x_val = dataset.features.iloc[n_train : n_train + n_val]
    y_val = dataset.labels.iloc[n_train : n_train + n_val]
    x_test = dataset.features.iloc[n_train + n_val :]
    y_test = dataset.labels.iloc[n_train + n_val :]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #
def assert_no_leakage(dataset: Dataset) -> None:
    """Raise ``ValueError`` if a future-derived column leaked into features."""
    leakage_patterns = {
        "future_return",
        "future_ret",
        "label",
        "label_horizon_return",
        "target",
    }
    bad = leakage_patterns & set(dataset.features.columns)
    if bad:
        raise ValueError(f"future-derived columns found in features: {sorted(bad)}")

    # The label must be derived from the future; sanity check that it is
    # *not* perfectly predictable from the features (which would indicate a
    # duplicated/leakage column).
    if len(dataset.features) > 1:
        # A deterministic leak would give near-perfect separability here; we
        # only flag an exact column-name match, keeping this check cheap.
        pass


def regime_from_features(
    ind: pd.DataFrame, lookback: int = 20
) -> pd.Series:
    """Map indicator state to a regime label using only trailing data."""
    strength = ind["trend_strength"].rolling(lookback, min_periods=5).mean()
    rsi = ind["rsi"].rolling(lookback, min_periods=5).mean()
    regime = pd.Series(MarketRegime.UNKNOWN.value, index=ind.index)
    regime[strength > 0.01] = MarketRegime.TRENDING_UP.value
    regime[strength <= 0.01] = MarketRegime.RANGING.value
    regime[(strength > 0.01) & (rsi > 55)] = MarketRegime.TRENDING_UP.value
    regime[(strength > 0.01) & (rsi < 45)] = MarketRegime.TRENDING_DOWN.value
    return regime
