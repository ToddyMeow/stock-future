"""Fixed-term exit: mandatory exit by max_bars, optional early exit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from strats.helpers import favorable_excursion, adverse_excursion


@dataclass(frozen=True)
class TermExitConfig:
    min_bars: int = 2
    max_bars: int = 13
    min_target_r: float = 1.0  # early exit requires this R


class TermExitStrategy:
    """Time-based exit: must exit by max_bars, can exit early after min_bars if profitable."""

    def __init__(self, config: Optional[TermExitConfig] = None) -> None:
        self.config = config or TermExitConfig()

    def process_close_phase(self, position: Any, row: pd.Series, next_trade_date: Any) -> None:
        cfg = self.config
        d = position.direction
        high_price = float(row["high"])
        low_price = float(row["low"])
        date = pd.Timestamp(row.get("date", row.name))

        position.highest_high_since_entry = max(position.highest_high_since_entry, high_price)
        position.lowest_low_since_entry = min(position.lowest_low_since_entry, low_price)
        position.completed_bars += 1

        fav_extreme = position.highest_high_since_entry if d == 1 else position.lowest_low_since_entry
        adv_extreme = position.lowest_low_since_entry if d == 1 else position.highest_high_since_entry
        position.mfe_price = max(position.mfe_price, favorable_excursion(fav_extreme, position.entry_fill, d))
        position.mae_price = max(position.mae_price, adverse_excursion(adv_extreme, position.entry_fill, d))

        active_stop_before = position.active_stop

        if position.pending_exit_reason is None and pd.notna(next_trade_date):
            if position.completed_bars >= cfg.max_bars:
                position.pending_exit_reason = "TERM_MAX"
                position.pending_exit_date = pd.Timestamp(next_trade_date)
            elif position.completed_bars >= cfg.min_bars:
                fav_exc = favorable_excursion(fav_extreme, position.entry_fill, d)
                if fav_exc >= cfg.min_target_r * position.r_price:
                    position.pending_exit_reason = "TERM_EARLY"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)

        position.active_stop_series.append({
            "computed_on": date.strftime("%Y-%m-%d"),
            "effective_from": pd.Timestamp(next_trade_date).strftime("%Y-%m-%d") if pd.notna(next_trade_date) else None,
            "phase": "close_update",
            "active_stop_before": active_stop_before,
            "active_stop_after": position.active_stop,
            "trailing_stop_candidate": None,
            "atr_used": None,
            "highest_high_since_entry": position.highest_high_since_entry,
        })
