"""Shared research helpers used by CLI scripts.

These helpers are stable library functions; CLI entrypoints should import
from here instead of from other scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

GROUPS = [
    "chem_energy",
    "rubber_fiber",
    "metals",
    "black_steel",
    "agri",
    "building",
    "livestock",
]

IND_GROUPS = [
    "ind_CJ",
    "ind_EC",
    "ind_FB",
    "ind_LU",
    "ind_RR",
    "ind_WH",
]

ALL_GROUPS = GROUPS + IND_GROUPS


def load_hab_bars() -> pd.DataFrame:
    path = ROOT / "data" / "cache" / "normalized" / "hab_bars.csv"
    bars = pd.read_csv(path)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["high"] = bars[["high", "open", "close"]].max(axis=1)
    bars["low"] = bars[["low", "open", "close"]].min(axis=1)
    return bars


def filter_group_bars(bars: pd.DataFrame, group: str) -> pd.DataFrame:
    return bars[bars["group_name"] == group].copy()


def yearly_stats_from_trades(
    trades: pd.DataFrame,
    portfolio_daily: pd.DataFrame,
    initial_capital: float,
) -> List[Dict[str, Any]]:
    del initial_capital
    if trades.empty:
        return []

    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year
    years = sorted(trades["year"].unique())

    pdf = portfolio_daily.copy()
    if not pdf.empty:
        pdf["year"] = pd.to_datetime(pdf["date"]).dt.year

    rows = []
    for yr in years:
        yr_trades = trades[trades["year"] == yr]
        n = len(yr_trades)
        if n == 0:
            continue

        wins = yr_trades[yr_trades["net_pnl"] > 0]
        losses = yr_trades[yr_trades["net_pnl"] <= 0]
        win_rate = len(wins) / n
        pf = (
            wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())
            if len(losses) > 0 and losses["net_pnl"].sum() != 0
            else float("inf") if len(wins) > 0 else 0.0
        )
        avg_r = yr_trades["r_multiple"].mean()

        sharpe = 0.0
        cagr = 0.0
        max_dd = 0.0
        if not pdf.empty:
            yr_eq = pdf[pdf["year"] == yr]
            if len(yr_eq) > 1:
                daily_ret = yr_eq["equity"].pct_change().dropna()
                if daily_ret.std() > 0:
                    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                eq_start = yr_eq["equity"].iloc[0]
                eq_end = yr_eq["equity"].iloc[-1]
                days = (yr_eq["date"].iloc[-1] - yr_eq["date"].iloc[0]).days
                if eq_start > 0 and days > 0:
                    cagr = (eq_end / eq_start) ** (365.25 / max(days, 1)) - 1.0
                peak = yr_eq["equity"].cummax()
                dd = (yr_eq["equity"] - peak) / peak.where(peak > 0, np.nan)
                max_dd = dd.min() if not dd.isna().all() else 0.0

        rows.append(
            {
                "year": yr,
                "trades": n,
                "sharpe": round(sharpe, 3),
                "cagr": round(cagr, 4),
                "profit_factor": round(pf, 3) if pf != float("inf") else 999.0,
                "win_rate": round(win_rate, 3),
                "avg_r": round(avg_r, 3),
                "max_dd_pct": round(max_dd, 4),
                "long_trades": int((yr_trades["direction"] == 1).sum()),
                "short_trades": int((yr_trades["direction"] == -1).sum()),
                "net_pnl": round(yr_trades["net_pnl"].sum(), 2),
                "exit_reasons": json.dumps(
                    yr_trades["exit_reason"].value_counts().to_dict(),
                    ensure_ascii=False,
                ),
            }
        )
    return rows
