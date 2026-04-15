"""AMA (Kaufman Adaptive Moving Average) reversal exit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from strats.helpers import favorable_excursion, adverse_excursion


@dataclass(frozen=True)
class AmaExitConfig:
    n: int = 10
    fast_period: int = 2
    slow_period: int = 30
    eps: float = 1e-12


class AmaExitStrategy:
    """Exit when AMA reverses direction and price crosses it."""

    def __init__(self, config: Optional[AmaExitConfig] = None) -> None:
        self.config = config or AmaExitConfig()

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

        # Incremental AMA computation
        closes = position.metadata.setdefault("_ama_closes", [])
        closes.append(close_price)
        prev_ama = position.metadata.get("_ama_value")

        fast_sc = 2.0 / (cfg.fast_period + 1)
        slow_sc = 2.0 / (cfg.slow_period + 1)

        ama_value = prev_ama
        if len(closes) > cfg.n:
            direction_val = closes[-1] - closes[-(cfg.n + 1)]
            volatility = sum(abs(closes[i] - closes[i - 1]) for i in range(max(1, len(closes) - cfg.n), len(closes)))
            er = abs(direction_val) / volatility if volatility > cfg.eps else 0.0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

            if prev_ama is None:
                ama_value = closes[-1]
            else:
                ama_value = prev_ama + sc * (closes[-1] - prev_ama)
        elif prev_ama is None and len(closes) >= cfg.n:
            ama_value = closes[-1]

        position.metadata["_ama_prev"] = prev_ama
        position.metadata["_ama_value"] = ama_value

        trailing_stop_candidate: Optional[float] = None
        active_stop_before = position.active_stop

        if ama_value is not None and prev_ama is not None and position.pending_exit_reason is None and pd.notna(next_trade_date):
            if d == 1:
                if close_price < ama_value and ama_value < prev_ama:
                    position.pending_exit_reason = "AMA_EXIT"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)
                trailing_stop_candidate = ama_value
                position.active_stop = max(position.active_stop, ama_value)
            else:
                if close_price > ama_value and ama_value > prev_ama:
                    position.pending_exit_reason = "AMA_EXIT"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)
                trailing_stop_candidate = ama_value
                position.active_stop = min(position.active_stop, ama_value)

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
