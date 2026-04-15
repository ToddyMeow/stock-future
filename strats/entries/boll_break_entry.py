"""Bollinger Band Breakout entry strategy.

Enter when price closes outside the Bollinger Bands. Uses the opposite
band as the initial stop.

R definition:
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class BollBreakEntryConfig:
    period: int = 22
    k: float = 2.0
    allow_short: bool = False

class BollBreakEntryStrategy:
    """Bollinger Band breakout entry."""

    def __init__(self, config: Optional[BollBreakEntryConfig] = None) -> None:
        self.config = config or BollBreakEntryConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Bollinger Band breakout signals.

        Expects ``atr``, ``atr_ref``, ``next_trade_date`` pre-computed by engine.
        All indicators use shift(1) for completed bars only.
        """
        cfg = self.config
        close = df["close"].astype(float)

        # Bollinger Bands on completed bars
        close_shifted = close.shift(1)
        ma = close_shifted.rolling(cfg.period, min_periods=cfg.period).mean()
        std = close_shifted.rolling(cfg.period, min_periods=cfg.period).std(ddof=0)
        upper = ma + cfg.k * std
        lower = ma - cfg.k * std

        # Long: close breaks above upper band
        long_trigger = close > upper
        # Short: close breaks below lower band
        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = close < lower

        entry_trigger_pass = (long_trigger | short_trigger).fillna(False)
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger.fillna(False)] = 1
        entry_direction[short_trigger.fillna(False) & ~long_trigger.fillna(False)] = -1

        # Initial stop: opposite band
        short_mask = entry_direction == -1

        out = df.copy()
        out["boll_upper"] = upper
        out["boll_lower"] = lower
        out["boll_ma"] = ma
        out["entry_trigger_pass"] = entry_trigger_pass
        out["entry_direction"] = entry_direction
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "boll_upper": float(row["boll_upper"]) if pd.notna(row.get("boll_upper")) else np.nan,
            "boll_lower": float(row["boll_lower"]) if pd.notna(row.get("boll_lower")) else np.nan,
            "boll_ma": float(row["boll_ma"]) if pd.notna(row.get("boll_ma")) else np.nan,
        }
