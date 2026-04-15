"""Double MA crossover exit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from strats.helpers import favorable_excursion, adverse_excursion


@dataclass(frozen=True)
class DoubleMaExitConfig:
    fast: int = 13
    slow: int = 34


class DoubleMaExitStrategy:
    """Exit on MA death cross (long) or golden cross (short)."""

    def __init__(self, config: Optional[DoubleMaExitConfig] = None) -> None:
        self.config = config or DoubleMaExitConfig()

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

        closes = position.metadata.setdefault("_dma_closes", [])
        closes.append(close_price)

        active_stop_before = position.active_stop

        if len(closes) >= cfg.slow and position.pending_exit_reason is None and pd.notna(next_trade_date):
            ma_fast = np.mean(closes[-cfg.fast:])
            ma_slow = np.mean(closes[-cfg.slow:])

            if d == 1 and ma_fast < ma_slow:
                position.pending_exit_reason = "DMA_EXIT"
                position.pending_exit_date = pd.Timestamp(next_trade_date)
            elif d == -1 and ma_fast > ma_slow:
                position.pending_exit_reason = "DMA_EXIT"
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
