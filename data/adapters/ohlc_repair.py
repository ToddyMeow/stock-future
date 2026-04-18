"""OHLC bar repair — method B (envelope expansion) from 1.4.

China futures limit-lock days and settle-alignment produce bars where
`close` or `settle` sits outside the published [low, high] range. The
reported high/low are the actual *traded* extremes; the close is often
administratively set to the settlement (limit-up) price. This module
expands [low, high] to envelope the enveloped columns so every row
satisfies `low <= each_price <= high`, without modifying the
administrative prices.

Pure functions — no I/O, no ordering assumptions. Callers (adapter,
enhanced-bars builder) invoke on the relevant column sets.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def repair_ohlc_envelope(
    df: pd.DataFrame,
    *,
    high_col: str,
    low_col: str,
    enveloped_cols: Sequence[str],
) -> int:
    """Expand [low, high] so every enveloped column fits within the range.

    For each row:
        high := max(high, *enveloped_cols)
        low  := min(low,  *enveloped_cols)

    Does NOT repair h<l (real corruption — caller should fail loudly).
    Returns the count of rows where high or low was modified.

    Columns missing from `df` are silently skipped (lets callers pass
    optional columns like `settle` that may not exist in legacy frames).
    """
    if df.empty:
        return 0
    if high_col not in df.columns or low_col not in df.columns:
        return 0

    h = df[high_col].astype(float).to_numpy(copy=True)
    l = df[low_col].astype(float).to_numpy(copy=True)
    orig_h = h.copy()
    orig_l = l.copy()

    for col in enveloped_cols:
        if col not in df.columns:
            continue
        v = df[col].astype(float).to_numpy()
        h = np.maximum(h, v)
        l = np.minimum(l, v)

    changed_mask = (h != orig_h) | (l != orig_l)
    changes = int(changed_mask.sum())

    if changes > 0:
        df[high_col] = h
        df[low_col] = l

    return changes
