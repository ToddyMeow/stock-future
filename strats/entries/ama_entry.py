"""Kaufman Adaptive Moving Average (AMA) entry strategy.

Enter when price crosses above/below the AMA while the AMA itself
is trending in the same direction. Uses an ATR-based stop.

R definition:
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class AmaEntryConfig:
    n: int = 10
    fast_period: int = 2
    slow_period: int = 30
    allow_short: bool = False
    eps: float = 1e-12

class AmaEntryStrategy:
    """Kaufman AMA trend entry."""

    def __init__(self, config: Optional[AmaEntryConfig] = None) -> None:
        self.config = config or AmaEntryConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute AMA entry signals.

        Expects ``atr``, ``atr_ref``, ``next_trade_date`` pre-computed by engine.
        """
        cfg = self.config
        close = df["close"].astype(float)
        atr_ref = df["atr_ref"]

        close_shifted = close.shift(1)

        # Efficiency Ratio components
        direction = close_shifted - close.shift(1 + cfg.n)
        volatility = close.diff().abs().shift(1).rolling(cfg.n).sum()
        er = direction.abs() / volatility.where(volatility > cfg.eps, np.nan)

        # Smoothing constant
        fast_sc = 2.0 / (cfg.fast_period + 1)
        slow_sc = 2.0 / (cfg.slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # AMA recursive computation (must be a loop)
        sc_arr = sc.to_numpy(dtype=float)
        cs_arr = close_shifted.to_numpy(dtype=float)
        ama_arr = np.full(len(close), np.nan, dtype=float)

        # Find first valid index where both sc and close_shifted are available
        first = None
        for i in range(len(close)):
            if not np.isnan(sc_arr[i]) and not np.isnan(cs_arr[i]):
                first = i
                break

        if first is not None:
            ama_arr[first] = cs_arr[first]
            for i in range(first + 1, len(close)):
                if np.isnan(sc_arr[i]):
                    ama_arr[i] = ama_arr[i - 1]
                    continue
                ama_arr[i] = ama_arr[i - 1] + sc_arr[i] * (cs_arr[i] - ama_arr[i - 1])

        ama = pd.Series(ama_arr, index=df.index)
        ama_prev = ama.shift(1)
        er_series = er

        # Long: close > ama AND ama trending up
        long_trigger = (close > ama) & (ama > ama_prev)
        # Short: close < ama AND ama trending down
        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = (close < ama) & (ama < ama_prev)

        entry_trigger_pass = (long_trigger | short_trigger).fillna(False)
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger.fillna(False)] = 1
        entry_direction[short_trigger.fillna(False) & ~long_trigger.fillna(False)] = -1

        # Initial stop: AMA +/- ATR buffer
        short_mask = entry_direction == -1

        out = df.copy()
        out["ama"] = ama
        out["er"] = er_series
        out["entry_trigger_pass"] = entry_trigger_pass
        out["entry_direction"] = entry_direction
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "ama": float(row["ama"]) if pd.notna(row.get("ama")) else np.nan,
            "er": float(row["er"]) if pd.notna(row.get("er")) else np.nan,
        }
