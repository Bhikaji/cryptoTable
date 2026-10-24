"""Historical / paper-trading engine (Stage 2 data source, Section 18.7).

Walks OHLCV bars, uses the current model + the independent risk manager to
decide entries, and records every fill as ``ExecutionKind.PAPER``.  Paper fills
are never mixed with live fills: the engine stamps ``execution_kind`` on every
trade it produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from core.config import RiskConfig
from core.market_data import to_bars  # noqa: F401  (re-export convenience)
from core.risk_manager import RiskManager
from core.types import (
    DataQuality,
    ExecutionKind,
    ExitReason,
    MarketBar,
    MarketRegime,
    Trade,
    TradeProposal,
    TradeSide,
    new_trade_id,
)


@dataclass
class BacktestConfig:
    fee_rate: float = 0.0005          # per side, fraction of notional
    slippage_rate: float = 0.0002     # fraction of notional
    tp_r_multiple: float = 1.5
    entry_probability_threshold: float = 0.55
    exit_probability_threshold: float = 0.45
    initial_equity: float = 10_000.0
    default_stop_distance: float = 0.02
    timeframe: str = "1m"


@dataclass
class PaperTradingResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    rejected: List[Trade] = field(default_factory=list)

    @property
    def closed_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.is_closed]


class PaperTradingEngine:
    def __init__(
        self,
        predictor,
        risk_manager: RiskManager,
        config: BacktestConfig | None = None,
        model_version: str = "unknown",
    ) -> None:
        self.predictor = predictor
        self.risk_manager = risk_manager
        self.config = config or BacktestConfig()
        self.model_version = model_version

    # ------------------------------------------------------------------ #
    def run(self, df: pd.DataFrame, symbol: str, regime_series: Optional[pd.Series] = None) -> PaperTradingResult:
        cfg = self.config
        from ai.features import compute_indicators, FEATURE_COLUMNS

        ind = compute_indicators(df)
        equity = cfg.initial_equity
        open_trade: Optional[Trade] = None
        result = PaperTradingResult()
        equity_curve: List[float] = []
        last_entry_time = -1e18
        entries_today = 0

        for i in range(len(ind)):
            row = ind.iloc[i]
            ts = float(ind.index[i])
            close = float(row["close"])
            atr_pct = float(row["atr_pct"]) if not np.isnan(row["atr_pct"]) else cfg.default_stop_distance
            stop_distance = float(np.clip(atr_pct, 0.002, 0.05))
            regime = self._regime(ind, i, regime_series)
            feats = {c: float(row[c]) if not np.isnan(row[c]) else 0.0 for c in FEATURE_COLUMNS}

            if open_trade is None:
                # ---- entry logic ----
                proba = self._safe_proba(feats)
                side = self._side(proba, regime)
                if side is None:
                    equity_curve.append(equity)
                    continue
                notional = equity * self.risk_manager.config.max_position_pct
                quantity = notional / close
                proposal = TradeProposal(
                    symbol=symbol,
                    side=side,
                    entry_price=close,
                    quantity=quantity,
                    leverage=1.0,
                    stop_distance=stop_distance,
                    expected_holding=cfg.default_stop_distance,
                    equity=equity,
                    open_exposure=0.0,
                    last_entry_time=last_entry_time,
                    now=ts,
                    entries_today=entries_today,
                    daily_pnl=0.0,
                )
                decision = self.risk_manager.evaluate(proposal)
                if decision.approved:
                    open_trade = Trade(
                        trade_id=new_trade_id(),
                        symbol=symbol,
                        timeframe=cfg.timeframe,
                        side=side,
                        model_version=self.model_version,
                        execution_kind=ExecutionKind.PAPER,
                        entry_time=ts,
                        entry_price=close,
                        quantity=quantity,
                        position_size=notional,
                        leverage=1.0,
                        stop_distance=stop_distance,
                        stop_loss=close * (1 - stop_distance) if side == TradeSide.LONG else close * (1 + stop_distance),
                        take_profit=close * (1 + cfg.tp_r_multiple * stop_distance) if side == TradeSide.LONG else close * (1 - cfg.tp_r_multiple * stop_distance),
                        market_regime=regime,
                        features=feats,
                        prediction_probabilities={"up": proba, "down": 1 - proba},
                        strategy_conditions={"regime_ok": regime != MarketRegime.UNKNOWN, "proba_ok": True},
                        strategy_followed=True,
                        risk_manager_approved=True,
                        risk_manager_reasons=[],
                        data_quality=DataQuality.OK,
                    )
                    entries_today += 1
                    last_entry_time = ts
                else:
                    # Record a rejected (missed) setup separately from losses.
                    rejected = Trade(
                        trade_id=new_trade_id(),
                        symbol=symbol,
                        timeframe=cfg.timeframe,
                        side=side,
                        model_version=self.model_version,
                        execution_kind=ExecutionKind.PAPER,
                        entry_time=ts,
                        entry_price=close,
                        quantity=quantity,
                        position_size=notional,
                        stop_distance=stop_distance,
                        market_regime=regime,
                        features=feats,
                        strategy_followed=True,
                        strategy_conditions={"regime_ok": regime != MarketRegime.UNKNOWN},
                        risk_manager_approved=False,
                        risk_manager_reasons=list(decision.reasons),
                    )
                    result.rejected.append(rejected)
                equity_curve.append(equity)
                continue

            # ---- manage open position ----
            high, low = float(row["high"]), float(row["low"])
            if open_trade.side == TradeSide.LONG:
                open_trade.mfe = max(open_trade.mfe, high - open_trade.entry_price)
                open_trade.mae = max(open_trade.mae, open_trade.entry_price - low)
            else:
                open_trade.mfe = max(open_trade.mfe, open_trade.entry_price - low)
                open_trade.mae = max(open_trade.mae, high - open_trade.entry_price)

            exit_price, exit_reason, exit_planned = None, None, True
            if open_trade.side == TradeSide.LONG:
                if low <= open_trade.stop_loss:
                    exit_price, exit_reason = open_trade.stop_loss, ExitReason.STOP_LOSS
                elif high >= open_trade.take_profit:
                    exit_price, exit_reason = open_trade.take_profit, ExitReason.TAKE_PROFIT
                elif self._safe_proba(feats) < cfg.exit_probability_threshold:
                    exit_price, exit_reason = close, ExitReason.SIGNAL
            else:
                if high >= open_trade.stop_loss:
                    exit_price, exit_reason = open_trade.stop_loss, ExitReason.STOP_LOSS
                elif low <= open_trade.take_profit:
                    exit_price, exit_reason = open_trade.take_profit, ExitReason.TAKE_PROFIT
                elif self._safe_proba(feats) > cfg.exit_probability_threshold:
                    exit_price, exit_reason = close, ExitReason.SIGNAL

            if exit_price is None and i == len(ind) - 1:
                exit_price, exit_reason = close, ExitReason.END_OF_DATA

            if exit_price is not None:
                self._close(open_trade, exit_price, ts, exit_reason, exit_planned, equity)
                equity += open_trade.net_pnl
                result.trades.append(open_trade)
                open_trade = None
                entries_today = 0
            equity_curve.append(equity)

        return result

    # ------------------------------------------------------------------ #
    def _close(self, trade: Trade, exit_price: float, exit_time: float,
               reason: ExitReason, planned: bool, equity: float) -> None:
        cfg = self.config
        direction = 1.0 if trade.side == TradeSide.LONG else -1.0
        gross = (exit_price - trade.entry_price) * trade.quantity * direction
        entry_fee = trade.notional * cfg.fee_rate
        exit_fee = trade.notional * cfg.fee_rate
        slippage = trade.notional * cfg.slippage_rate
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.exit_planned = planned
        trade.holding_duration = exit_time - trade.entry_time
        trade.entry_fee = entry_fee
        trade.exit_fee = exit_fee
        trade.slippage_estimate = slippage
        trade.gross_pnl = gross
        trade.net_pnl = gross - entry_fee - exit_fee - slippage
        trade.strategy_followed = planned

    # ------------------------------------------------------------------ #
    def _side(self, proba: float, regime: MarketRegime) -> Optional[TradeSide]:
        trending = (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN)
        long_ok = proba >= self.config.entry_probability_threshold and regime in trending
        short_ok = (1 - proba) >= self.config.entry_probability_threshold and regime in trending
        if long_ok:
            return TradeSide.LONG
        if short_ok:
            return TradeSide.SHORT
        return None

    def _safe_proba(self, feats: dict) -> float:
        try:
            return float(self.predictor.up_probability(feats))
        except Exception:
            return 0.5

    @staticmethod
    def _regime(ind: pd.DataFrame, i: int, regime_series: Optional[pd.Series]) -> MarketRegime:
        if regime_series is not None and i < len(regime_series):
            r = regime_series.iloc[i]
            return MarketRegime(r) if r in MarketRegime.__members__ else MarketRegime.UNKNOWN
        strength = float(ind["trend_strength"].iloc[i]) if i >= 0 else 0.0
        rsi = float(ind["rsi"].iloc[i]) if i >= 0 else 50.0
        if strength > 0.01:
            return MarketRegime.TRENDING_UP if rsi >= 50 else MarketRegime.TRENDING_DOWN
        return MarketRegime.RANGING
