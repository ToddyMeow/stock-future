"""Strategy protocols for composable entry/exit modules.

Any class implementing the correct methods works — no need to inherit.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class EntryStrategy(Protocol):
    """Produces entry signals from bar data."""

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-symbol signal computation.

        Receives a single-symbol DataFrame sorted by date, with at minimum
        the standard OHLCV columns plus ``atr``, ``atr_ref``, and
        ``next_trade_date`` pre-computed by the engine.

        Must return the same DataFrame with AT LEAST these columns added:
          - ``entry_trigger_pass`` (bool): True on bars that fire an entry
          - ``entry_direction``    (int):  1=long, -1=short, 0=no signal
          - ``initial_stop``       (float): stop price for the signal

        R definition contract:
          R (unit risk) = |entry_price - initial_stop|.
          The engine computes this from ``close`` (signal bar) and ``initial_stop``.
          Different entry strategies produce different R by setting ``initial_stop``
          differently (box low - ATR buffer, N-day low, MA distance, etc.).
          The exit strategy and risk management read R as ``position.r_price``
          and ``position.r_money`` — they never recompute it from strategy-specific
          fields, so any entry strategy that provides a valid ``initial_stop``
          integrates seamlessly.
        """
        ...

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract strategy-specific fields from a triggered signal row.

        Called by the engine for each row where ``entry_trigger_pass`` is True.
        Returns a dict of extra fields stored on ``PendingEntry.metadata``
        and later flattened into the trade record.
        """
        ...


@runtime_checkable
class ExitStrategy(Protocol):
    """Manages position exits during the close phase."""

    def process_close_phase(
        self,
        position: Any,
        row: pd.Series,
        next_trade_date: Any,
    ) -> None:
        """Evaluate exit conditions and update position state.

        Called once per surviving position per bar, after open-phase exits
        and pending fills have been processed.

        Responsibilities:
          - Update ``position.highest_high_since_entry`` / ``lowest_low_since_entry``
          - Increment ``position.completed_bars``
          - Update ``position.mfe_price`` / ``mae_price``
          - Optionally set ``position.pending_exit_reason`` / ``pending_exit_date``
          - Update ``position.active_stop`` (trailing)
          - Append to ``position.active_stop_series``

        The engine handles actual exit execution (gap stop, pending exit fill,
        intraday stop) on the next bar.
        """
        ...
