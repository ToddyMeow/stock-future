"""BacktestResult — pulled out of engine.py.

Re-exported from strats.engine for backward compat.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict

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

    # Live-trading support (P1a): every day's newly-produced pending entries
    # collapsed into one DataFrame. Columns:
    #   generated_date  — date the signal was evaluated (trading date of bar T)
    #   symbol          — underlying symbol key (e.g. "RB")
    #   contract_code   — order_book_id when dual-stream bars carry it,
    #                     otherwise a stable alias of symbol
    #   group_name      — bars group (e.g. "black_steel")
    #   strategy_id     — slot key
    #   action          — one of {"open", "close", "add", "reduce"}
    #   direction       — "long" or "short"
    #   target_qty      — integer contracts to work (always > 0)
    #   entry_price_ref — engine's best estimate of fill price (Panama)
    #   stop_loss_ref   — engine-computed initial stop price (Panama)
    #   entry_date      — calendar date the order should trade on
    pending_entries: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Engine terminal state — JSON-compatible dict used for incremental
    # resume (signal_service's "load yesterday → run today → save" loop).
    engine_state: Dict[str, Any] = field(default_factory=dict)

    # ----- state helpers -------------------------------------------------

    def save_state(self) -> Dict[str, Any]:
        """Return a JSON-compatible snapshot of the engine's terminal state.

        The returned dict is a deep copy — callers may mutate it freely
        without affecting this BacktestResult.
        """
        return copy.deepcopy(self.engine_state)

    @staticmethod
    def load_state(d: Dict[str, Any]) -> Dict[str, Any]:
        """Rehydrate a state dict.

        Accepts either a plain dict (what save_state returned) or a JSON
        string. Returns a dict shaped identically to save_state's output
        so it can be passed directly to engine.run(initial_state=...).
        """
        if isinstance(d, str):
            d = json.loads(d)
        if not isinstance(d, dict):
            raise TypeError(
                f"BacktestResult.load_state expects dict or JSON str, got {type(d)!r}"
            )
        return copy.deepcopy(d)
