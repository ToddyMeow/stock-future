"""Pure ATR trailing stop exit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from strats.helpers import favorable_excursion, adverse_excursion


@dataclass(frozen=True)
class AtrTrailExitConfig:
    atr_mult: float = 2.0


class AtrTrailExitStrategy:
    """ATR trailing stop only — no pending exit signals, engine intraday stop handles exit."""

    def __init__(self, config: Optional[AtrTrailExitConfig] = None) -> None:
        self.config = config or AtrTrailExitConfig()

    def process_close_phase(self, position: Any, row: pd.Series, next_trade_date: Any) -> None:
        cfg = self.config
        d = position.direction
        high_price = float(row["high"])
        low_price = float(row["low"])
        date = pd.Timestamp(row.get("date", row.name))

        position.highest_high_since_entry = max(position.highest_high_since_entry, high_price)
        position.lowest_low_since_entry = min(position.lowest_low_since_entry, low_price)
        position.completed_bars += 1

        fav = position.highest_high_since_entry if d == 1 else position.lowest_low_since_entry
        adv = position.lowest_low_since_entry if d == 1 else position.highest_high_since_entry
        position.mfe_price = max(position.mfe_price, favorable_excursion(fav, position.entry_fill, d))
        position.mae_price = max(position.mae_price, adverse_excursion(adv, position.entry_fill, d))

        atr_today = float(row["atr"]) if pd.notna(row.get("atr")) else np.nan
        trailing_stop_candidate: Optional[float] = None
        active_stop_before = position.active_stop

        if np.isfinite(atr_today):
            if d == 1:
                trailing_stop_candidate = position.highest_high_since_entry - cfg.atr_mult * atr_today
                position.active_stop = max(position.active_stop, trailing_stop_candidate)
            else:
                trailing_stop_candidate = position.lowest_low_since_entry + cfg.atr_mult * atr_today
                position.active_stop = min(position.active_stop, trailing_stop_candidate)

        position.active_stop_series.append({
            "computed_on": date.strftime("%Y-%m-%d"),
            "effective_from": pd.Timestamp(next_trade_date).strftime("%Y-%m-%d") if pd.notna(next_trade_date) else None,
            "phase": "close_update",
            "active_stop_before": active_stop_before,
            "active_stop_after": position.active_stop,
            "trailing_stop_candidate": trailing_stop_candidate,
            "atr_used": atr_today if np.isfinite(atr_today) else None,
            "highest_high_since_entry": position.highest_high_since_entry,
        })
