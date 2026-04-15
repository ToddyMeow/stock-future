"""HL(N) channel trailing exit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from strats.helpers import favorable_excursion, adverse_excursion


@dataclass(frozen=True)
class HLExitConfig:
    period: int = 21


class HLExitStrategy:
    """Exit when price crosses the N-bar trailing channel boundary."""

    def __init__(self, config: Optional[HLExitConfig] = None) -> None:
        self.config = config or HLExitConfig()

    def process_close_phase(self, position: Any, row: pd.Series, next_trade_date: Any) -> None:
        cfg = self.config
        d = position.direction
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        date = pd.Timestamp(row.get("date", row.name))

        position.highest_high_since_entry = max(position.highest_high_since_entry, high_price)
        position.lowest_low_since_entry = min(position.lowest_low_since_entry, low_price)
        position.completed_bars += 1

        fav = position.highest_high_since_entry if d == 1 else position.lowest_low_since_entry
        adv = position.lowest_low_since_entry if d == 1 else position.highest_high_since_entry
        position.mfe_price = max(position.mfe_price, favorable_excursion(fav, position.entry_fill, d))
        position.mae_price = max(position.mae_price, adverse_excursion(adv, position.entry_fill, d))

        # Track rolling window in metadata
        highs = position.metadata.setdefault("_hl_highs", [])
        lows = position.metadata.setdefault("_hl_lows", [])
        highs.append(high_price)
        lows.append(low_price)
        if len(highs) > cfg.period:
            highs[:] = highs[-cfg.period:]
            lows[:] = lows[-cfg.period:]

        trailing_stop_candidate: Optional[float] = None
        active_stop_before = position.active_stop

        if len(lows) >= cfg.period and position.pending_exit_reason is None and pd.notna(next_trade_date):
            if d == 1:
                trailing_low = min(lows)
                if close_price <= trailing_low:
                    position.pending_exit_reason = "HL_EXIT"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)
                trailing_stop_candidate = trailing_low
                position.active_stop = max(position.active_stop, trailing_low)
            else:
                trailing_high = max(highs)
                if close_price >= trailing_high:
                    position.pending_exit_reason = "HL_EXIT"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)
                trailing_stop_candidate = trailing_high
                position.active_stop = min(position.active_stop, trailing_high)

        position.active_stop_series.append({
            "computed_on": date.strftime("%Y-%m-%d"),
            "effective_from": pd.Timestamp(next_trade_date).strftime("%Y-%m-%d") if pd.notna(next_trade_date) else None,
            "phase": "close_update",
            "active_stop_before": active_stop_before,
            "active_stop_after": position.active_stop,
            "trailing_stop_candidate": trailing_stop_candidate,
            "atr_used": None,
            "highest_high_since_entry": position.highest_high_since_entry,
        })
