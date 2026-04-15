"""HAB exit strategy: structure fail, time fail, ATR trailing stop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from strats.helpers import favorable_excursion, adverse_excursion


@dataclass(frozen=True)
class HABExitConfig:
    structure_fail_bars: int = 15
    structure_fail_mode: Literal[
        "CLOSE_BELOW_BOX", "CLOSE_BELOW_BOX_MINUS_ATR", "CONSECUTIVE_CLOSE"
    ] = "CLOSE_BELOW_BOX"
    structure_fail_atr_buffer: float = 0.5
    structure_fail_consecutive: int = 2
    time_fail_bars: int = 5
    time_fail_target_r: float = 0.5
    trail_activate_r: float = 1.0
    trail_atr_mult: float = 2.0


class HABExitStrategy:
    """HAB exit: struct fail + time fail + ATR trailing stop."""

    def __init__(self, config: Optional[HABExitConfig] = None) -> None:
        self.config = config or HABExitConfig()

    def process_close_phase(
        self,
        position: Any,
        row: pd.Series,
        next_trade_date: Any,
    ) -> None:
        cfg = self.config
        d = position.direction
        high_price = float(row["high"]) if "high" in row.index else float(row.get("high", 0))
        low_price = float(row["low"]) if "low" in row.index else float(row.get("low", 0))
        close_price = float(row["close"]) if "close" in row.index else float(row.get("close", 0))
        date = pd.Timestamp(row.get("date", row.name))

        # 1) Update extremes
        position.highest_high_since_entry = max(position.highest_high_since_entry, high_price)
        position.lowest_low_since_entry = min(position.lowest_low_since_entry, low_price)
        position.completed_bars += 1

        # 2) MFE / MAE
        fav = position.highest_high_since_entry if d == 1 else position.lowest_low_since_entry
        adv = position.lowest_low_since_entry if d == 1 else position.highest_high_since_entry
        position.mfe_price = max(position.mfe_price, favorable_excursion(fav, position.entry_fill, d))
        position.mae_price = max(position.mae_price, adverse_excursion(adv, position.entry_fill, d))

        if position.pending_exit_reason is None and pd.notna(next_trade_date):
            # 3) Structure fail
            atr_now = float(row["atr"]) if pd.notna(row.get("atr")) else 0.0
            if position.completed_bars <= cfg.structure_fail_bars:
                box_high = position.metadata.get("box_high", 0.0)
                box_low = position.metadata.get("box_low", 0.0)
                ref = box_high if d == 1 else box_low
                mode = cfg.structure_fail_mode
                buf = cfg.structure_fail_atr_buffer * atr_now

                if mode == "CLOSE_BELOW_BOX_MINUS_ATR":
                    threshold = ref - buf if d == 1 else ref + buf
                else:
                    threshold = ref
                bar_fail = close_price <= threshold if d == 1 else close_price >= threshold

                if mode == "CONSECUTIVE_CLOSE":
                    position.consecutive_fail_count = position.consecutive_fail_count + 1 if bar_fail else 0
                    struct_trigger = position.consecutive_fail_count >= cfg.structure_fail_consecutive
                else:
                    struct_trigger = bar_fail

                if struct_trigger:
                    position.pending_exit_reason = "STRUCT_FAIL"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)

            # 4) Time fail
            if position.pending_exit_reason is None and position.completed_bars == cfg.time_fail_bars:
                fav_exc = favorable_excursion(
                    position.highest_high_since_entry if d == 1 else position.lowest_low_since_entry,
                    position.entry_fill, d,
                )
                if fav_exc < cfg.time_fail_target_r * position.r_price:
                    position.pending_exit_reason = "TIME_FAIL"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)

        # 5) Trailing stop
        atr_today = float(row["atr"]) if pd.notna(row.get("atr")) else np.nan
        trailing_stop_candidate: Optional[float] = None
        active_stop_before = position.active_stop

        if np.isfinite(atr_today):
            if d == 1:
                trail_activated = position.highest_high_since_entry >= position.entry_fill + cfg.trail_activate_r * position.r_price
                if trail_activated:
                    trailing_stop_candidate = position.highest_high_since_entry - cfg.trail_atr_mult * atr_today
                    position.active_stop = max(position.active_stop, trailing_stop_candidate)
            else:
                trail_activated = position.entry_fill - position.lowest_low_since_entry >= cfg.trail_activate_r * position.r_price
                if trail_activated:
                    trailing_stop_candidate = position.lowest_low_since_entry + cfg.trail_atr_mult * atr_today
                    position.active_stop = min(position.active_stop, trailing_stop_candidate)

        position.active_stop_series.append(
            {
                "computed_on": date.strftime("%Y-%m-%d"),
                "effective_from": (
                    pd.Timestamp(next_trade_date).strftime("%Y-%m-%d")
                    if pd.notna(next_trade_date)
                    else None
                ),
                "phase": "close_update",
                "active_stop_before": active_stop_before,
                "active_stop_after": position.active_stop,
                "trailing_stop_candidate": trailing_stop_candidate,
                "atr_used": atr_today if np.isfinite(atr_today) else None,
                "highest_high_since_entry": position.highest_high_since_entry,
            }
        )
