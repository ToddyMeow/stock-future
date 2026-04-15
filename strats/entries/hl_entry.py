"""HL(N) Channel Breakout entry strategy.

Classic trend-following entry: enter when price makes a new N-bar
high/low (High-Low channel). Unlike the Donchian variant, this uses
the raw channel boundaries as stops without an ATR buffer.

R definition:
  Long:  initial_stop = channel_low   (no ATR buffer)
  Short: initial_stop = channel_high  (no ATR buffer)
  R = |close - initial_stop|
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HLEntryConfig:
    period: int = 20
    allow_short: bool = False


class HLEntryStrategy:
    """HL channel breakout entry."""

    def __init__(self, config: Optional[HLEntryConfig] = None) -> None:
        self.config = config or HLEntryConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute HL channel breakout signals.

        Expects ``atr``, ``atr_ref``, ``next_trade_date`` pre-computed by engine.
        """
        cfg = self.config
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        # Only look at completed bars (shift by 1)
        channel_high = high.shift(1).rolling(cfg.period, min_periods=cfg.period).max()
        channel_low = low.shift(1).rolling(cfg.period, min_periods=cfg.period).min()

        # Long: close makes a new N-bar high
        long_trigger = close > channel_high
        # Short: close makes a new N-bar low
        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = close < channel_low

        entry_trigger_pass = (long_trigger | short_trigger).fillna(False)
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger.fillna(False)] = 1
        entry_direction[short_trigger.fillna(False) & ~long_trigger.fillna(False)] = -1

        # Initial stop: opposite channel boundary, NO ATR buffer
        initial_stop_long = channel_low.copy()
        initial_stop_short = channel_high.copy()
        initial_stop = initial_stop_long.copy()
        short_mask = entry_direction == -1
        initial_stop[short_mask] = initial_stop_short[short_mask]

        out = df.copy()
        out["channel_high"] = channel_high
        out["channel_low"] = channel_low
        out["entry_trigger_pass"] = entry_trigger_pass
        out["entry_direction"] = entry_direction
        out["initial_stop"] = initial_stop
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "channel_high": float(row["channel_high"]) if pd.notna(row.get("channel_high")) else np.nan,
            "channel_low": float(row["channel_low"]) if pd.notna(row.get("channel_low")) else np.nan,
        }
