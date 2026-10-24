"""Synthetic, regime-aware market data used by backtests and tests.

The generator produces OHLCV bars from a regime-switching process so that the
learning subsystem has realistic (and reproducible) data to train on without
requiring an exchange connection.  Regimes are *observable* from past bars only
when ``label_regimes`` is called with lookback-limited helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import MarketBar, MarketRegime


class SyntheticMarketGenerator:
    """Regime-switching geometric-Brownian-motion price simulator."""

    REGIMES = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.RANGING]

    def __init__(
        self,
        symbols: list[str] | None = None,
        n_bars: int = 800,
        start_price: float = 30_000.0,
        base_volume: float = 1000.0,
        seed: int = 7,
        transition_matrix: np.ndarray | None = None,
        trend_drift: float = 0.0012,
        ranging_drift: float = 0.0,
        volatility: float = 0.004,
        ranging_volatility: float = 0.002,
    ) -> None:
        self.symbols = symbols or ["BTCUSDT"]
        self.n_bars = n_bars
        self.start_price = start_price
        self.base_volume = base_volume
        self.seed = seed
        self.transition_matrix = (
            transition_matrix
            if transition_matrix is not None
            else np.array(
                [
                    [0.90, 0.05, 0.05],  # trending up
                    [0.05, 0.90, 0.05],  # trending down
                    [0.05, 0.05, 0.90],  # ranging
                ]
            )
        )
        self.trend_drift = trend_drift
        self.ranging_drift = ranging_drift
        self.volatility = volatility
        self.ranging_volatility = ranging_volatility

    # ------------------------------------------------------------------ #
    def _simulate_symbol(self, symbol: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + abs(hash(symbol)) % 10_000)
        transitions = self.transition_matrix
        regime_idx = 0
        regimes = np.empty(self.n_bars, dtype=int)
        drifts = np.empty(self.n_bars)
        vols = np.empty(self.n_bars)

        for i in range(self.n_bars):
            regimes[i] = regime_idx
            if regime_idx == 0:  # trending up
                drifts[i] = self.trend_drift
                vols[i] = self.volatility
            elif regime_idx == 1:  # trending down
                drifts[i] = -self.trend_drift
                vols[i] = self.volatility
            else:  # ranging
                drifts[i] = self.ranging_drift
                vols[i] = self.ranging_volatility
            regime_idx = rng.choice(3, p=transitions[regime_idx])

        shocks = rng.standard_normal(self.n_bars)
        log_returns = drifts + vols * shocks
        close = self.start_price * np.exp(np.cumsum(log_returns))

        open_ = np.empty(self.n_bars)
        open_[0] = self.start_price
        open_[1:] = close[:-1]

        span = np.abs(vols * close) + 1e-6
        high = np.maximum(open_, close) + np.abs(rng.standard_normal(self.n_bars)) * span
        low = np.minimum(open_, close) - np.abs(rng.standard_normal(self.n_bars)) * span
        low = np.maximum(low, 1e-6)
        volume = self.base_volume * np.exp(0.3 * rng.standard_normal(self.n_bars))

        return pd.DataFrame(
            {
                "symbol": symbol,
                "timestamp": np.arange(self.n_bars, dtype=float),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "regime_idx": regimes,
            }
        )

    def generate(self) -> dict[str, pd.DataFrame]:
        return {sym: self._simulate_symbol(sym) for sym in self.symbols}

    def regime_labels(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        out = {}
        for sym, df in frames.items():
            mapping = {0: MarketRegime.TRENDING_UP, 1: MarketRegime.TRENDING_DOWN, 2: MarketRegime.RANGING}
            out[sym] = df["regime_idx"].map(mapping).astype(str)
        return out


def to_bars(df: pd.DataFrame) -> list[MarketBar]:
    bars = [
        MarketBar(
            symbol=row.symbol,
            timestamp=row.timestamp,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
        )
        for row in df.itertuples()
    ]
    return bars
