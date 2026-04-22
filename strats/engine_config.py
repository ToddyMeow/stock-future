"""EngineConfig + StrategySlot — pulled out of engine.py for navigation.

Re-exported from strats.engine for backward compat. External callers should
continue `from strats.engine import EngineConfig, StrategySlot`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Literal


@dataclass(frozen=True)
class EngineConfig:
    """Orchestrator-level config: columns, capital, risk, shared params.

    Execution policy (1.5):
      - Signals are evaluated at bar T's close (using T's OHLC and indicators
        computed from bars ≤ T; stops use atr_ref = ATR(T-1) which is strictly
        pre-signal).
      - Fills occur at bar T+1's open (the next trading day per the calendar).
      - Because our daily bars span the Chinese futures night→day session,
        T+1's open is the 21:00 night-session open on calendar evening T,
        NOT the 09:00 day-session open of calendar-day T+1. In other words:
        NEXT_BAR_OPEN ≡ NIGHT_SESSION_OPEN for this project.
      - NEXT_DAY_OPEN (09:00) is not supported — it would require intraday
        data (frequency="1m") which we don't pull.
      - This is enforced structurally via `next_trade_date = date.shift(-1)`
        and the "pending entry fills on entry_date" loop; see
        strats/test_execution_policy.py for the no-look-ahead invariant test.
    """

    # Column names
    date_col: str = "date"
    symbol_col: str = "symbol"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    open_interest_col: str = "open_interest"
    multiplier_col: str = "contract_multiplier"
    commission_col: str = "commission"
    slippage_col: str = "slippage"
    group_col: str = "group_name"
    margin_rate_col: str = "margin_rate"

    # Dual-stream P&L. When enabled, bars must carry
    # per-contract raw prices + order_book_id; entries/exits fill on raw, P&L
    # is segment-accounted across rolls, roll_cost applied explicitly.
    # Indicators and stops remain in Panama space (the `open`/`high`/`low`/
    # `close` columns) — only the economics are on raw.
    enable_dual_stream: bool = False
    raw_open_col: str = "open_raw"
    raw_high_col: str = "high_raw"
    raw_low_col: str = "low_raw"
    raw_close_col: str = "close_raw"
    contract_col: str = "order_book_id"

    # Settle price. China futures settle = daily VWAP
    # (or last hour VWAP on CFFEX), used for mark-to-market, margin, and risk
    # caps. Technical indicators + signal still read `close` (last trade).
    # If the column is absent from bars, the engine silently falls back to
    # `close_col` so legacy synthetic test fixtures keep working.
    settle_col: str = "settle"
    settle_raw_col: str = "settle_raw"

    # Indicator warmup (1.7). Reject entry signals on the first N bars of each
    # symbol — Wilder-smoothed ATR/ADX take up to 10× their period to converge
    # (ADX is double-smoothed). Default 0 = disabled (keeps short-fixture tests
    # working). Production recommendation for atr_period=adx_period=20: 200.
    warmup_bars: int = 0

    # Limit-lock defense (1.8). When > 0 and bars carry `limit_up`/`limit_down`
    # columns, sizing uses a worst-case floor of
    #   price × limit_pct × max_limit_days × multiplier
    # to bound qty against consecutive-lock scenarios (e.g. China futures
    # ±4–10% limits held for 2 days). 0 disables this branch.
    max_limit_days: int = 0
    limit_up_col: str = "limit_up"
    limit_down_col: str = "limit_down"

    # Margin utilization cap (4.1). Effective margin rate per contract =
    # base_margin_rate (from bars `margin_rate_col`) + broker_margin_addon +
    # tier_addon (from margin_tier_schedule keyed by months-to-delivery
    # derived from order_book_id's YYMM suffix). Total occupied margin across
    # open positions + new candidate must satisfy
    #   occupied <= equity × max_margin_utilization
    # or the signal is rejected as `MARGIN_CAP`. 0 disables the cap (default).
    max_margin_utilization: float = 0.0
    broker_margin_addon: float = 0.0
    margin_tier_schedule: Dict[int, float] = field(default_factory=dict)

    # Data quality defenses (1.4). Reject a signal when atr_ref is degenerately
    # small relative to close — this prevents qty explosion on limit-lock bars
    # where TR → 0. 0.25% × close is a conservative floor: typical daily range
    # on liquid futures is > 0.5% × close, so normal bars pass; fully locked
    # limit-up/down days (OHLC all equal) are blocked. Set to 0 to disable
    # (legacy behaviour — NaN / 0 ATR still caught by NON_POSITIVE_RISK).
    min_atr_pct: float = 0.0025

    # Capital and risk
    initial_capital: float = 1_000_000.0
    risk_per_trade: float = 0.02
    portfolio_risk_cap: float = 0.12
    group_risk_cap: Dict[str, float] = field(default_factory=lambda: {
        "equity_index": 0.04, "bond": 0.04,
        "chem_energy": 0.06, "rubber_fiber": 0.06, "metals": 0.06,
        "black_steel": 0.06, "agri": 0.06, "building": 0.05, "livestock": 0.04,
    })
    default_group_risk_cap: float = 0.02
    independent_group_soft_cap: float = 0.08
    max_portfolio_leverage: float = 3.0
    default_margin_rate: float = 0.10

    # Shared technical
    atr_period: int = 20
    adx_period: int = 20
    adx_scale: float = 30.0      # ADX / adx_scale = trend_score (before clip)
    adx_floor: float = 0.2       # minimum trend_score (never fully zero out)

    # R definition: R = stop_atr_mult × ATR(atr_period)
    # initial_stop = entry_price ∓ stop_atr_mult × atr_ref
    stop_atr_mult: float = 2.0

    # Fill mechanics
    risk_blowout_cap: float = 1.5
    risk_blowout_action: Literal["SHRINK", "CANCEL"] = "SHRINK"
    allow_short: bool = False

    # Symbol exclusion
    exclude_symbols: FrozenSet[str] = frozenset()

    eps: float = 1e-12


@dataclass
class StrategySlot:
    """Pairs one entry strategy with one exit strategy under a unique ID."""

    strategy_id: str
    entry_strategy: Any
    exit_strategy: Any
