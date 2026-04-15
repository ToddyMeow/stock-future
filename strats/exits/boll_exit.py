"""Bollinger Band exit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from strats.helpers import favorable_excursion, adverse_excursion


@dataclass(frozen=True)
class BollExitConfig:
    period: int = 22
    k: float = 2.0


class BollExitStrategy:
    """Exit when price crosses the opposite Bollinger band."""

    def __init__(self, config: Optional[BollExitConfig] = None) -> None:
        self.config = config or BollExitConfig()

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

        closes = position.metadata.setdefault("_boll_closes", [])
        closes.append(close_price)

        trailing_stop_candidate: Optional[float] = None
        active_stop_before = position.active_stop

        if len(closes) >= cfg.period and position.pending_exit_reason is None and pd.notna(next_trade_date):
            window = closes[-cfg.period:]
            ma = np.mean(window)
            std = np.std(window, ddof=0)
            upper = ma + cfg.k * std
            lower = ma - cfg.k * std

            if d == 1:
                if close_price < lower:
                    position.pending_exit_reason = "BOLL_EXIT"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)
                trailing_stop_candidate = lower
                position.active_stop = max(position.active_stop, lower)
            else:
                if close_price > upper:
                    position.pending_exit_reason = "BOLL_EXIT"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)
                trailing_stop_candidate = upper
                position.active_stop = min(position.active_stop, upper)

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
