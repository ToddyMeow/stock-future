"""HL(N) channel breakout entry."""

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
    """Enter when close breaks N-bar high (long) or low (short)."""

    def __init__(self, config: Optional[HLEntryConfig] = None) -> None:
        self.config = config or HLEntryConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        channel_high = high.shift(1).rolling(cfg.period, min_periods=cfg.period).max()
        channel_low = low.shift(1).rolling(cfg.period, min_periods=cfg.period).min()

        long_trigger = close > channel_high
        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = close < channel_low

        entry_trigger_pass = (long_trigger | short_trigger).fillna(False)
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger.fillna(False)] = 1
        entry_direction[short_trigger.fillna(False) & ~long_trigger.fillna(False)] = -1

        out = df.copy()
        out["channel_high"] = channel_high
        out["channel_low"] = channel_low
        out["entry_trigger_pass"] = entry_trigger_pass
        out["entry_direction"] = entry_direction
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "channel_high": float(row["channel_high"]) if pd.notna(row.get("channel_high")) else np.nan,
            "channel_low": float(row["channel_low"]) if pd.notna(row.get("channel_low")) else np.nan,
        }
