"""HAB entry strategy: horizontal accumulation box breakout with BB compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from strats.helpers import (
    detect_hlh_pattern,
    detect_lhl_pattern,
    rolling_last_value_percentile,
)


@dataclass(frozen=True)
class HABEntryConfig:
    bb_period: int = 20
    bb_std: float = 2.0
    bb_percentile_lookback: int = 60
    bb_percentile_threshold: float = 0.30
    box_lookback: int = 7
    box_width_atr_mult: float = 1.5
    tol_atr_mult: float = 0.25
    breakout_atr_mult: float = 0.5
    upper_shadow_ratio_max: float = 0.25
    initial_stop_atr_mult: float = 0.4
    allow_short: bool = False
    eps: float = 1e-12


class HABEntryStrategy:
    """HAB entry: box detection + BB compression + breakout confirmation."""

    def __init__(self, config: Optional[HABEntryConfig] = None) -> None:
        self.config = config or HABEntryConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute HAB entry signals.

        Expects ``atr`` and ``atr_ref`` columns already present (computed by engine).
        Adds: BB indicators, box detection, pattern flags, entry_trigger_pass,
        entry_direction, initial_stop, plus diagnostic columns.
        """
        cfg = self.config
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        atr = df["atr"]
        atr_ref = df["atr_ref"]

        # BB bands
        bb_mid = close.rolling(cfg.bb_period, min_periods=cfg.bb_period).mean()
        bb_std_val = close.rolling(cfg.bb_period, min_periods=cfg.bb_period).std(ddof=0)
        bb_upper = bb_mid + cfg.bb_std * bb_std_val
        bb_lower = bb_mid - cfg.bb_std * bb_std_val

        # Bandwidth normalized by ATR (shift-invariant on back-adjusted futures)
        bandwidth_denom = atr.where(atr > cfg.eps, np.nan)
        bandwidth = (bb_upper - bb_lower) / bandwidth_denom
        bb_percentile = rolling_last_value_percentile(
            bandwidth.shift(1), cfg.bb_percentile_lookback,
        )

        # Box detection
        box_high = high.shift(1).rolling(cfg.box_lookback, min_periods=cfg.box_lookback).max()
        box_low = low.shift(1).rolling(cfg.box_lookback, min_periods=cfg.box_lookback).min()
        box_width = box_high - box_low
        tol = cfg.tol_atr_mult * atr_ref

        # Pattern detection arrays
        has_upper_test_1 = np.zeros(len(df), dtype=bool)
        has_lower_confirm = np.zeros(len(df), dtype=bool)
        has_upper_test_2 = np.zeros(len(df), dtype=bool)
        h_l_h_valid = np.zeros(len(df), dtype=bool)
        has_lower_test_1 = np.zeros(len(df), dtype=bool)
        has_upper_confirm = np.zeros(len(df), dtype=bool)
        has_lower_test_2 = np.zeros(len(df), dtype=bool)
        l_h_l_valid = np.zeros(len(df), dtype=bool)

        for i in range(len(df)):
            if i < cfg.box_lookback:
                continue
            if pd.isna(box_high.iloc[i]) or pd.isna(box_low.iloc[i]) or pd.isna(tol.iloc[i]):
                continue

            win_h = high.iloc[i - cfg.box_lookback : i].tolist()
            win_l = low.iloc[i - cfg.box_lookback : i].tolist()
            bh = float(box_high.iloc[i])
            bl = float(box_low.iloc[i])
            t = float(tol.iloc[i])

            valid, flag_u1, flag_l1, flag_u2 = detect_hlh_pattern(
                high_window=win_h, low_window=win_l, box_high=bh, box_low=bl, tol=t,
            )
            h_l_h_valid[i] = valid
            has_upper_test_1[i] = flag_u1
            has_lower_confirm[i] = flag_l1
            has_upper_test_2[i] = flag_u2

            if cfg.allow_short:
                s_valid, s_lt1, s_uc, s_lt2 = detect_lhl_pattern(
                    high_window=win_h, low_window=win_l, box_high=bh, box_low=bl, tol=t,
                )
                l_h_l_valid[i] = s_valid
                has_lower_test_1[i] = s_lt1
                has_upper_confirm[i] = s_uc
                has_lower_test_2[i] = s_lt2

        box_width_pass = box_width <= cfg.box_width_atr_mult * atr_ref
        is_box_long = box_width_pass & h_l_h_valid
        is_box_short = box_width_pass & l_h_l_valid

        spread = (high - low).abs()
        upper_shadow_ratio = (high - close) / np.maximum(spread, cfg.eps)
        lower_shadow_ratio = (close - low) / np.maximum(spread, cfg.eps)
        bb_filter_pass = bb_percentile <= cfg.bb_percentile_threshold

        long_trigger = (
            is_box_long & bb_filter_pass
            & (close > box_high + cfg.breakout_atr_mult * atr_ref)
            & (upper_shadow_ratio <= cfg.upper_shadow_ratio_max)
        )

        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = (
                is_box_short & bb_filter_pass
                & (close < box_low - cfg.breakout_atr_mult * atr_ref)
                & (lower_shadow_ratio <= cfg.upper_shadow_ratio_max)
            )

        entry_trigger_pass = long_trigger | short_trigger
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger] = 1
        entry_direction[short_trigger & ~long_trigger] = -1

        initial_stop_long = box_low - cfg.initial_stop_atr_mult * atr_ref
        initial_stop_short = box_high + cfg.initial_stop_atr_mult * atr_ref
        initial_stop = initial_stop_long.copy()
        short_mask = entry_direction == -1
        initial_stop[short_mask] = initial_stop_short[short_mask]

        # Write all columns to the output frame
        out = df.copy()
        out["bb_mid"] = bb_mid
        out["bb_upper"] = bb_upper
        out["bb_lower"] = bb_lower
        out["bandwidth_denom"] = bandwidth_denom
        out["bandwidth"] = bandwidth
        out["bb_percentile"] = bb_percentile
        out["box_high"] = box_high
        out["box_low"] = box_low
        out["box_width"] = box_width
        out["tol"] = tol
        out["box_width_pass"] = box_width_pass.fillna(False)
        out["is_box"] = (is_box_long | is_box_short).fillna(False)
        out["has_upper_test_1"] = has_upper_test_1
        out["has_lower_confirm"] = has_lower_confirm
        out["has_upper_test_2"] = has_upper_test_2
        out["has_lower_test_1"] = has_lower_test_1
        out["has_upper_confirm"] = has_upper_confirm
        out["has_lower_test_2"] = has_lower_test_2
        out["shadow_ratio"] = upper_shadow_ratio
        out["bb_filter_pass"] = bb_filter_pass.fillna(False)
        out["entry_trigger_pass"] = entry_trigger_pass.fillna(False)
        out["entry_direction"] = entry_direction
        out["initial_stop"] = initial_stop
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract HAB-specific fields for PendingEntry.metadata."""
        return {
            "box_high": float(row["box_high"]),
            "box_low": float(row["box_low"]),
            "bb_percentile": float(row["bb_percentile"]),
            "bandwidth": float(row["bandwidth"]),
            "shadow_ratio": float(row["shadow_ratio"]),
            "tol": float(row["tol"]),
        }
