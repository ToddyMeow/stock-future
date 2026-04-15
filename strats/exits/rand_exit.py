"""Random exit — benchmark for measuring entry strategy alpha."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from strats.helpers import favorable_excursion, adverse_excursion


@dataclass(frozen=True)
class RandExitConfig:
    seed: int = 42
    exit_probability: float = 0.1  # ~10% chance per bar to exit
    min_bars: int = 1  # don't exit before this many bars held


class RandExitStrategy:
    """Random exit with fixed seed for reproducibility.

    Each bar, rolls a uniform random number. If > (1 - exit_probability)
    and position held >= min_bars, sets pending exit for next open.
    """

    def __init__(self, config: Optional[RandExitConfig] = None) -> None:
        self.config = config or RandExitConfig()

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

        active_stop_before = position.active_stop

        if position.pending_exit_reason is None and pd.notna(next_trade_date):
            if position.completed_bars >= cfg.min_bars:
                # Deterministic PRNG seeded from config seed + position identity
                rng = np.random.RandomState(
                    cfg.seed + hash((position.symbol, str(position.entry_date))) % (2**31)
                    + position.completed_bars
                )
                if rng.uniform() < cfg.exit_probability:
                    position.pending_exit_reason = "RAND_EXIT"
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
