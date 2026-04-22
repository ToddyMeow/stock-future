"""v5 — Adaptive HL(N) channel breakout entry.

The lookback `period` scales with the current volatility regime:
  effective_period(t) = round( base_period × [min_ratio + (max_ratio - min_ratio) × atr_rank(t)] )

where `atr_rank(t)` is the rolling percentile of ATR (on completed bars) over
the last `adapt_lookback` days (default 60). The rationale:

- Low-volatility regime → short period → faster break signal on emerging trends
- High-volatility regime → long period → more stable breakout level, filters noise

Direction semantics (long/short) mirror HLEntryStrategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdaptiveHLEntryConfig:
    base_period: int = 20
    adapt_lookback: int = 60
    min_period_ratio: float = 0.5   # low-vol → 0.5 × base
    max_period_ratio: float = 1.5   # high-vol → 1.5 × base
    allow_short: bool = False


class AdaptiveHLEntryStrategy:
    """Volatility-adaptive Donchian/HL channel breakout."""

    def __init__(self, config: Optional[AdaptiveHLEntryConfig] = None) -> None:
        self.config = config or AdaptiveHLEntryConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if df.empty:
            out = df.copy()
            out["entry_trigger_pass"] = False
            out["entry_direction"] = 0
            return out

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        atr = df["atr"].astype(float) if "atr" in df.columns else pd.Series(np.nan, index=df.index)

        # ATR rank in [0, 1] — use completed bars only (shift(1))
        atr_shift = atr.shift(1)
        atr_rank = atr_shift.rolling(cfg.adapt_lookback, min_periods=10).rank(pct=True)
        atr_rank = atr_rank.fillna(0.5)  # default to neutral when warmup incomplete

        # Effective period bounds
        min_p = max(2, int(round(cfg.base_period * cfg.min_period_ratio)))
        max_p = max(min_p + 1, int(round(cfg.base_period * cfg.max_period_ratio)))

        eff_period = (
            min_p + (max_p - min_p) * atr_rank
        ).round().clip(lower=min_p, upper=max_p).astype(int)

        # Precompute rolling max/min for each candidate period (shifted — completed bars only)
        high_shift = high.shift(1)
        low_shift = low.shift(1)
        chan_highs: Dict[int, np.ndarray] = {}
        chan_lows: Dict[int, np.ndarray] = {}
        for p in range(min_p, max_p + 1):
            chan_highs[p] = high_shift.rolling(p, min_periods=p).max().to_numpy()
            chan_lows[p] = low_shift.rolling(p, min_periods=p).min().to_numpy()

        # Select per bar
        n = len(df)
        channel_high = np.full(n, np.nan)
        channel_low = np.full(n, np.nan)
        eff_arr = eff_period.to_numpy()
        for i in range(n):
            p = int(eff_arr[i])
            if p in chan_highs:
                channel_high[i] = chan_highs[p][i]
                channel_low[i] = chan_lows[p][i]

        channel_high_s = pd.Series(channel_high, index=df.index)
        channel_low_s = pd.Series(channel_low, index=df.index)

        long_trigger = close > channel_high_s
        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = close < channel_low_s

        entry_trigger_pass = (long_trigger | short_trigger).fillna(False)
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger.fillna(False)] = 1
        entry_direction[short_trigger.fillna(False) & ~long_trigger.fillna(False)] = -1

        out = df.copy()
        out["channel_high"] = channel_high_s
        out["channel_low"] = channel_low_s
        out["adaptive_eff_period"] = eff_period
        out["atr_rank"] = atr_rank
        out["entry_trigger_pass"] = entry_trigger_pass
        out["entry_direction"] = entry_direction
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "channel_high": float(row["channel_high"]) if pd.notna(row.get("channel_high")) else np.nan,
            "channel_low": float(row["channel_low"]) if pd.notna(row.get("channel_low")) else np.nan,
            "adaptive_eff_period": int(row["adaptive_eff_period"]) if pd.notna(row.get("adaptive_eff_period")) else 0,
            "atr_rank": float(row["atr_rank"]) if pd.notna(row.get("atr_rank")) else np.nan,
        }
