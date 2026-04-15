"""Donchian Channel (price channel) entry strategy.

Classic trend-following entry: enter when price makes a new N-day high/low.
Unlike HAB box breakout, this does NOT require volatility compression —
it only requires price reaching a new extreme. Entries come later in the
trend (worse entry price) but signals are more reliable.

R definition:
  Long:  initial_stop = donchian_low  - stop_atr_mult * atr_ref
  Short: initial_stop = donchian_high + stop_atr_mult * atr_ref
  R = |close - initial_stop|
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DonchianEntryConfig:
    donchian_period: int = 20
    initial_stop_atr_mult: float = 2.0
    allow_short: bool = False
    eps: float = 1e-12


class DonchianEntryStrategy:
    """Donchian channel breakout entry."""

    def __init__(self, config: Optional[DonchianEntryConfig] = None) -> None:
        self.config = config or DonchianEntryConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Donchian channel breakout signals.

        Expects ``atr``, ``atr_ref``, ``next_trade_date`` pre-computed by engine.
        """
        cfg = self.config
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        atr_ref = df["atr_ref"]

        # Only look at completed bars (shift by 1)
        donchian_high = high.shift(1).rolling(cfg.donchian_period, min_periods=cfg.donchian_period).max()
        donchian_low = low.shift(1).rolling(cfg.donchian_period, min_periods=cfg.donchian_period).min()

        # Long: close makes a new N-day high
        long_trigger = close > donchian_high
        # Short: close makes a new N-day low
        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = close < donchian_low

        entry_trigger_pass = (long_trigger | short_trigger).fillna(False)
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger.fillna(False)] = 1
        entry_direction[short_trigger.fillna(False) & ~long_trigger.fillna(False)] = -1

        # Initial stop: opposite channel boundary with ATR buffer
        initial_stop_long = donchian_low - cfg.initial_stop_atr_mult * atr_ref
        initial_stop_short = donchian_high + cfg.initial_stop_atr_mult * atr_ref
        initial_stop = initial_stop_long.copy()
        short_mask = entry_direction == -1
        initial_stop[short_mask] = initial_stop_short[short_mask]

        out = df.copy()
        out["donchian_high"] = donchian_high
        out["donchian_low"] = donchian_low
        out["entry_trigger_pass"] = entry_trigger_pass
        out["entry_direction"] = entry_direction
        out["initial_stop"] = initial_stop
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "donchian_high": float(row["donchian_high"]) if pd.notna(row.get("donchian_high")) else np.nan,
            "donchian_low": float(row["donchian_low"]) if pd.notna(row.get("donchian_low")) else np.nan,
        }
