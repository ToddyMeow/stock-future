"""BacktestResult — pulled out of engine.py.

Re-exported from strats.engine for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily_status: pd.DataFrame
    portfolio_daily: pd.DataFrame
    open_positions: pd.DataFrame
    prepared_data: pd.DataFrame
    cancelled_entries: pd.DataFrame
    # Per-symbol data-quality stats computed from raw input bars (1.4). Columns:
    # symbol, n_bars, lock_pct, near_zero_range_pct, zero_volume_pct, ohlc_anomaly_count.
    data_quality_report: pd.DataFrame = field(default_factory=pd.DataFrame)
