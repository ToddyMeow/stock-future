"""Double Moving Average Crossover entry strategy.

Enter on golden cross (fast MA crosses above slow MA) or death cross
(fast MA crosses below slow MA). Uses the slow MA as the initial stop.

R definition:
  Long:  initial_stop = ma_slow
  Short: initial_stop = ma_slow
  R = |close - initial_stop|
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DoubleMaEntryConfig:
    fast: int = 13
    slow: int = 34
    allow_short: bool = False


class DoubleMaEntryStrategy:
    """Double MA crossover entry."""

    def __init__(self, config: Optional[DoubleMaEntryConfig] = None) -> None:
        self.config = config or DoubleMaEntryConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute double MA crossover signals.

        Expects ``atr``, ``atr_ref``, ``next_trade_date`` pre-computed by engine.
        All indicators use shift(1) for completed bars only.
        """
        cfg = self.config
        close = df["close"].astype(float)

        close_shifted = close.shift(1)
        ma_fast = close_shifted.rolling(cfg.fast, min_periods=cfg.fast).mean()
        ma_slow = close_shifted.rolling(cfg.slow, min_periods=cfg.slow).mean()
        ma_fast_prev = ma_fast.shift(1)
        ma_slow_prev = ma_slow.shift(1)

        # Golden cross: fast crosses above slow
        long_trigger = (ma_fast > ma_slow) & (ma_fast_prev <= ma_slow_prev)
        # Death cross: fast crosses below slow
        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = (ma_fast < ma_slow) & (ma_fast_prev >= ma_slow_prev)

        entry_trigger_pass = (long_trigger | short_trigger).fillna(False)
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger.fillna(False)] = 1
        entry_direction[short_trigger.fillna(False) & ~long_trigger.fillna(False)] = -1

        # Initial stop: slow MA as support/resistance
        initial_stop = ma_slow.copy()

        out = df.copy()
        out["ma_fast"] = ma_fast
        out["ma_slow"] = ma_slow
        out["entry_trigger_pass"] = entry_trigger_pass
        out["entry_direction"] = entry_direction
        out["initial_stop"] = initial_stop
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "ma_fast": float(row["ma_fast"]) if pd.notna(row.get("ma_fast")) else np.nan,
            "ma_slow": float(row["ma_slow"]) if pd.notna(row.get("ma_slow")) else np.nan,
        }
