"""Random entry strategy (benchmark / null hypothesis).

Enters long or short based on a deterministic pseudo-random sequence.
Useful as a baseline to compare real strategies against.

R definition:
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RandEntryConfig:
    seed: int = 42
    allow_short: bool = False

class RandEntryStrategy:
    """Random entry for benchmarking."""

    def __init__(self, config: Optional[RandEntryConfig] = None) -> None:
        self.config = config or RandEntryConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute random entry signals.

        Expects ``atr``, ``atr_ref``, ``next_trade_date`` pre-computed by engine.
        """
        cfg = self.config
        close = df["close"].astype(float)
        atr_ref = df["atr_ref"]

        rng = np.random.RandomState(cfg.seed)
        signal = rng.uniform(0, 1, size=len(df))
        rand_signal = pd.Series(signal, index=df.index)

        # Long: signal > 0.5
        long_trigger = rand_signal > 0.5
        # Short (if allow_short): signal <= 0.5
        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = rand_signal <= 0.5

        entry_trigger_pass = (long_trigger | short_trigger).fillna(False)
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger] = 1
        entry_direction[short_trigger & ~long_trigger] = -1

        # If not allow_short, only long signals
        if not cfg.allow_short:
            entry_trigger_pass = long_trigger.fillna(False)
            entry_direction[~long_trigger] = 0

        # Initial stop: close +/- ATR buffer
        short_mask = entry_direction == -1

        out = df.copy()
        out["rand_signal"] = rand_signal
        out["entry_trigger_pass"] = entry_trigger_pass
        out["entry_direction"] = entry_direction
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "rand_signal": float(row["rand_signal"]) if pd.notna(row.get("rand_signal")) else np.nan,
        }
