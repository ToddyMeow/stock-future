"""Extract roll events from enhanced bars.

A roll event occurs when the dominant contract (`order_book_id`) changes
from one trading day to the next. This module provides pure utility
functions on the enhanced bars DataFrame — no I/O.
"""

from __future__ import annotations

from typing import Dict, Set

import pandas as pd


def extract_roll_dates_per_symbol(
    bars: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    contract_col: str = "order_book_id",
) -> Dict[str, Set[pd.Timestamp]]:
    """Return {symbol: {roll_date, ...}} where roll_date is the first day on
    the NEW contract (the day the contract code first differs from yesterday).

    The first row per symbol is NOT a roll (there's no yesterday to compare).
    """
    required = {symbol_col, date_col, contract_col}
    missing = required - set(bars.columns)
    if missing:
        raise KeyError(f"bars missing columns: {sorted(missing)}")

    rolls: Dict[str, Set[pd.Timestamp]] = {}
    for symbol, group in bars.groupby(symbol_col, sort=False):
        g = group.sort_values(date_col)
        changed = g[contract_col].ne(g[contract_col].shift())
        # Drop the first row per symbol (changed=True but not a real roll).
        changed.iloc[0] = False
        roll_dates = g.loc[changed, date_col]
        rolls[str(symbol)] = {pd.Timestamp(d) for d in roll_dates}
    return rolls


def build_roll_events_frame(
    bars: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    contract_col: str = "order_book_id",
    close_raw_col: str = "close_raw",
    open_raw_col: str = "open_raw",
) -> pd.DataFrame:
    """Long-form roll events table for audit/diagnostics.

    One row per (symbol, roll_date) with columns:
        symbol, roll_date, old_contract, new_contract,
        old_close (yesterday's raw close on old contract),
        new_open (today's raw open on new contract)
    """
    required = {symbol_col, date_col, contract_col, close_raw_col, open_raw_col}
    missing = required - set(bars.columns)
    if missing:
        raise KeyError(f"bars missing columns: {sorted(missing)}")

    rows = []
    for symbol, group in bars.groupby(symbol_col, sort=False):
        g = group.sort_values(date_col).reset_index(drop=True)
        changed = g[contract_col].ne(g[contract_col].shift())
        changed.iloc[0] = False
        for idx in g.index[changed]:
            if idx == 0:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "roll_date": g.at[idx, date_col],
                    "old_contract": g.at[idx - 1, contract_col],
                    "new_contract": g.at[idx, contract_col],
                    "old_close": float(g.at[idx - 1, close_raw_col]),
                    "new_open": float(g.at[idx, open_raw_col]),
                }
            )
    return pd.DataFrame(rows, columns=[
        "symbol", "roll_date", "old_contract", "new_contract",
        "old_close", "new_open",
    ])
