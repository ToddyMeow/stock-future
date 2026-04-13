
# HorizontalAccumulationBreakout_V1

A compact daily-bar research backtester for the long-only strategy described in the spec.

## What is implemented

- Daily OHLC backtest
- Long-only, one position per symbol, no pyramiding
- Multi-symbol portfolio with three-layer risk controls
  - single trade risk: 2% of equity
  - portfolio risk cap: 12% of equity
  - group risk cap: 6% of equity
- Box detection using the prior 7 completed bars only
- Mandatory H-L-H boundary confirmation
- Bollinger bandwidth percentile compression filter
- Entry generated on signal-day close, filled on next open
- Quantity frozen from signal-day close estimate; never back-solved from next open
- Open-invalidated entries are cancelled, not force-counted as trades
  - if next-open entry fill is `<= initial_stop + eps`, the pending order is cancelled with reason `OPEN_INVALIDATES_STOP`
  - the order is written to `cancelled_entries` and does not enter the trade log
- Exit priority implemented in the required order
  1. gap stop / intraday stop
  2. structure fail
  3. time fail
  4. trailing-stop update for subsequent bars
- Trade log output with risk-audit fields
- Per-symbol daily status table output
- Cancelled pending-entry log output
- Unit tests for box detection, structure fail, time fail, existing-position gap stop, entry-day intraday stop, quantity freeze from signal close, entry cancellation, tomorrow-risk netting, fill-risk blowout logging, positive bandwidth normalization, and invalid OHLC input

## Assumptions kept explicit

- `commission` is treated as **single-side fixed commission per contract**.
- `slippage` is treated as **adverse price slippage per fill** in price units.
  - long entry fill = `open + slippage`
  - long exit fill = `open - slippage` or `stop - slippage`
- Existing portfolio risk is re-marked daily as:
  - `max(current_close - active_stop, 0) * multiplier * qty`
- Signal-day risk budgeting is evaluated against the candidate order's **actual next entry date**.
  - positions with `pending_exit_date <= candidate_entry_date` are excluded from effective open risk because execution order is "exit first, then enter"
  - already accepted pending entries with `entry_date <= candidate_entry_date` are still counted conservatively in pending-order risk
- When multiple same-day signals compete for remaining portfolio/group risk budget, they are processed by **entry date first**, then **lexicographic symbol order**, so the outcome is deterministic.
- Trailing stop is updated **after** the daily bar closes and only applies from the next bar onward. This avoids same-bar high/low sequencing look-ahead.
- The engine records post-fill risk blowout honestly but does **not** shrink or cancel the trade after a valid fill just because the next-open gap worsened actual initial risk. Portfolio caps remain signal-close risk caps, not physical gap-proof caps.
- The engine does **not** force-close open positions at the end of the dataset. Remaining live positions are returned in `open_positions`.

## Indicator details

- ATR uses Wilder smoothing.
- Bollinger bandwidth is normalized by `abs(bb_mid)` with an epsilon floor.
  - this is deliberate so near-zero or negative price regimes cannot produce negative bandwidth and corrupt percentile-based compression filtering

## Active stop audit semantics

`active_stop_series` is intentionally phase-aware.

- `phase="signal_init"`
  - the initial stop was computed on `signal_date`
  - it becomes effective from `entry_date`
- `phase="close_update"`
  - the active stop was recomputed on that bar's close
  - it becomes effective from the next trade date
- each record includes:
  - `computed_on`
  - `effective_from`
  - `active_stop_before`
  - `active_stop_after`
  - `trailing_stop_candidate`
  - `atr_used`
  - `highest_high_since_entry`

This avoids the common audit mistake of reading a close-updated trailing stop as if it had already been live during that same bar.

## Input validation

`prepare_data()` rejects rows that violate basic market-data invariants, including:

- missing required columns
- duplicate `(symbol, date)` rows
- non-finite numeric values
- `high < low`
- `high < max(open, close)`
- `low > min(open, close)`
- `contract_multiplier <= 0`
- `commission < 0`
- `slippage < 0`
- `volume < 0`
- `open_interest < 0`

## Outputs

`BacktestResult` returns:

- `trades`: closed-trade log including
  - `group_name`
  - `atr_ref`
  - `bandwidth`
  - `shadow_ratio`
  - `tol`
  - `volume`
  - `open_interest`
  - `estimated_initial_risk`
  - `estimated_order_risk`
  - `actual_initial_risk`
  - `actual_order_risk`
  - `risk_blowout_vs_estimate`
  - `risk_blowout_ratio`
  - `active_stop_series`
- `daily_status`: per-symbol daily state table including
  - `is_box`
  - `has_upper_test_1`
  - `has_lower_confirm`
  - `has_upper_test_2`
  - `bb_filter_pass`
  - `entry_trigger_pass`
  - `risk_reject_reason`
  - `bandwidth`
  - `shadow_ratio`
  - `tol`
  - `volume`
  - `open_interest`
- `portfolio_daily`: portfolio/equity snapshot by date
- `open_positions`: positions still open at end of sample, with the same risk-audit fields as live position state
- `prepared_data`: full enriched symbol/day frame
- `cancelled_entries`: pending orders cancelled before fill, including the attempted open/fill context and cancellation reason

## Quick usage

```python
import pandas as pd
from horizontal_accumulation_breakout_v1 import HABConfig, HorizontalAccumulationBreakoutV1

bars = pd.read_parquet("daily_bars.parquet")
engine = HorizontalAccumulationBreakoutV1(HABConfig(initial_capital=1_000_000))
result = engine.run(bars)

trades = result.trades
status = result.daily_status
portfolio = result.portfolio_daily
cancelled = result.cancelled_entries
```

## Test command

When the three files sit in the same working directory:

```bash
pytest -q test_horizontal_accumulation_breakout_v1.py
```

For the delivered artifact layout in this workspace:

```bash
PYTHONPATH=/mnt/data/hab_v1 pytest -q /mnt/data/hab_v1/test_horizontal_accumulation_breakout_v1.py
```
