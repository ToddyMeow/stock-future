# Architecture Snapshot

Static snapshot of the repo's layered structure. For change history + per-issue
audit trail, see [HANDOFF.md](HANDOFF.md). For runnable how-to, see the usage
section at the top of [strats/engine.py](strats/engine.py).

```
┌─────────────────────────────────────────────────────────────────────┐
│ Data layer                                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  scripts/download_rqdata_futures.py                                 │
│    ↓ pulls OHLC + settlement for 91 symbols × dominant series       │
│    → data/cache/raw_rqdata/dominant_{none,pre}/<SYMBOL>.csv         │
│                                                                     │
│  scripts/download_dominant_contracts.py (1.2)                       │
│    → data/cache/dominant_contracts/<SYMBOL>.csv   (date, ob_id)     │
│                                                                     │
│  scripts/download_limit_prices.py (1.8)                             │
│    → data/cache/limit_prices/<SYMBOL>.csv  (limit_up, limit_down)   │
│                                                                     │
│  scripts/build_trading_calendar.py (1.1)                            │
│    → data/cache/calendar/cn_futures_trading_days.csv                │
│                                                                     │
│  scripts/fetch_commission_specs.py (5.2)                            │
│    → data/cache/commission_specs.json  (by_money vs by_volume)      │
│                                                                     │
│  data/adapters/rqdata_futures_adatpter.py                           │
│    normalize_one(): OHLC repair (1.4) + calendar validation (1.1)   │
│    + per-row commission by spec (5.2) + output columns              │
│                                                                     │
│  scripts/build_enhanced_bars.py  ← main merge                       │
│    reads dominant_none / dominant_contracts / limit_prices          │
│    → data/cache/normalized/hab_bars.csv (22 cols)                   │
│                                                                     │
│  scripts/apply_commissions.py (5.2 one-shot patch)                  │
│    re-computes commission column from specs.json                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼  hab_bars.csv (22 columns)
                           │  date, symbol, [open|high|low|close|settle]
                           │  volume, open_interest, contract_multiplier
                           │  commission, slippage, group_name, margin_rate
                           │  [open|high|low|close|settle]_raw
                           │  order_book_id  ← per-day dominant contract
                           │  limit_up, limit_down
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Engine layer (strats/)                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  engine_config.py        EngineConfig + StrategySlot (dataclasses)  │
│  position.py             PendingEntry + Position dataclasses        │
│  result.py               BacktestResult dataclass                   │
│  protocols.py            EntryStrategy + ExitStrategy Protocols     │
│  helpers.py              wilder_atr / adx / PortfolioAnalyzer       │
│  config_loader.py        YAML → EngineConfig                        │
│                                                                     │
│  engine.py (StrategyEngine) — main orchestrator:                    │
│    prepare_data / _prepare_all_strategies                           │
│      → _normalize_and_validate_bars (shared 1.1+1.3+1.4 validation) │
│      → _prepare_symbol_base (ATR, ADX, _bar_index, next_trade_date) │
│      → each EntryStrategy.prepare_signals                           │
│                                                                     │
│    run(bars) main loop, per trading date:                           │
│      0. _check_and_apply_roll (1.2)                                 │
│      1. _process_open_and_intraday_for_existing_position            │
│         → _cannot_fill_side (1.8 + 1.10 limit-lock / gap-open)      │
│         → _close_position (segment P&L + roll cost + commissions)   │
│      2. fill pending entries                                        │
│         → _fill_pending_entry / _build_cancelled_entry              │
│      3. each ExitStrategy.process_close_phase                       │
│      4. update last_close_by_symbol / last_raw_close_by_symbol      │
│      5. mark equity via _compute_equity_close (settle, 1.3)         │
│      6. signal eval loop with gates (ordered):                      │
│         WARMUP_INSUFFICIENT (1.7)                                   │
│         NO_NEXT_TRADE_DATE / ALREADY_IN_POSITION                    │
│         ATR_BELOW_FLOOR (1.4)                                       │
│         LIMIT_LOCK_ENTRY (1.8/1.10)                                 │
│         NON_POSITIVE_RISK / QTY_LT_1                                │
│         PORTFOLIO_RISK_CAP / GROUP_RISK_CAP / INDEPENDENT_SOFT_CAP  │
│         LEVERAGE_CAP                                                │
│         MARGIN_CAP (4.1)                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼  BacktestResult
                           │  trades / daily_status / portfolio_daily
                           │  open_positions / prepared_data
                           │  cancelled_entries / data_quality_report
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Analysis layer (strats/helpers.py::PortfolioAnalyzer)               │
├─────────────────────────────────────────────────────────────────────┤
│  equity_curve(), drawdown_episodes()                                │
│  group_contribution(), risk_utilization(), signal_density()         │
│  periodic_returns(freq="M"), summary_stats()                        │
│    → Sharpe (√252), Sortino, max DD, profit factor, expectancy      │
└─────────────────────────────────────────────────────────────────────┘
```

## Key design decisions (cross-reference HANDOFF.md for details)

- **1.1 Trading calendar** — every `date` must be a trading day; enforced at
  adapter ingestion. [data/adapters/trading_calendar.py](data/adapters/trading_calendar.py)
- **1.2 Continuous contract** — Panama-adjusted for indicators (single-stream
  default); dual-stream via `enable_dual_stream=True` switches entry/exit
  fills to raw contract prices with segment-accounted P&L and explicit roll
  cost. `_check_and_apply_roll` detects contract changes, closes old
  segment, opens new at raw open.
- **1.3 Settle vs close** — daily mark-to-market uses `settle_col` (falls
  back to `close_col` for legacy bars without settle); indicators keep
  reading `close`.
- **1.4 OHLC repair** — method B (envelope `[low, high]` to include
  `open/close/settle`) in the adapter. Data-quality report per symbol in
  `BacktestResult.data_quality_report`. ATR floor = 0.25% × close to guard
  against limit-lock indicator pollution.
- **1.5/1.6 Execution timing** — signal at T close, fill at T+1 open
  (= 21:00 night session). No look-ahead; invariant tests in
  [strats/test_execution_policy.py](strats/test_execution_policy.py).
- **1.7 Warmup** — `EngineConfig.warmup_bars` gates first N bars per symbol
  to avoid unconverged Wilder-smoothed ATR/ADX. Default 0 (disabled).
- **1.8/1.10 Limit lock / gap open** — `_cannot_fill_side` blocks entries
  and open-fill exits when `raw_open` hits the adverse limit. Sizing floor
  via `max_limit_days` uses worst-case consecutive-lock loss. Intraday
  stop fills still proceed (market traded within [low, high]).
- **4.1 Margin cap** — optional `max_margin_utilization` + tiered
  `margin_tier_schedule` keyed by months-to-delivery (`_months_to_delivery`
  parses YYMM suffix). Signals rejected as `MARGIN_CAP` when over cap.
- **5.2 Real commissions** — per-symbol specs in
  `data/cache/commission_specs.json`; adapter emits per-row yuan (by_volume
  constant, by_money = rate × |close| × multiplier).
- **5.3 Daily P&L formula** — `(settle_today − settle_yesterday)` is the
  math-equivalent increment of our unrealized diff; `entry_fill` cancels
  between consecutive days, so no code change was needed on top of 1.3.

## Deliberately out of scope

- **5.1 平今/平昨** — 2.06% of trades are same-day in our daily-bar
  backtest; lot-level tracking not justified. Revisit if moving to
  intraday frequency.
- **4.2 Delivery-month hard check** — active-product dominant roll handles
  it; residual exposure concentrated on BB/PM/RI (already filtered by ATR
  floor + limit-lock gates). Explicit check deferred until live.
- **Ratio-adjusted continuous series** — Panama is equivalent for our
  point-value indicator set (ATR/Donchian/Bollinger); ratio would
  introduce negative-price artifacts on deep-offset symbols.
- **Forced-close on margin shortage** — `max_margin_utilization` is an
  entry-time cap, not a live liquidation model. Add if/when backtesting
  black-swan drawdown scenarios.

## Test suite

- `data/adapters/` — calendar, roll events, OHLC repair, adapter normalizer
- `strats/` — engine core, execution policy, dual stream, data quality,
  limit lock, margin cap, warmup, HAB strategy, donchian entry
- `scripts/` — download_rqdata_futures smoke
- **Total: 138 passing**. Run with `python -m pytest -q`.
