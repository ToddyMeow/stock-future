"""Look-ahead invariant check for the execution policy (1.5).

Our policy: signal at bar T close → fill at bar T+1 open (NEXT_BAR_OPEN,
which for daily futures bars is the 21:00 night-session open, equivalent
to NIGHT_SESSION_OPEN).

This test runs the full HAB-on-synthetic-bars fixture used by
test_horizontal_accumulation_breakout_v1 and asserts every trade's
`signal_date < entry_date`. Nothing may fill on the same bar it signalled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from strats.engine import EngineConfig, StrategyEngine
from strats.entries.hl_entry import HLEntryConfig, HLEntryStrategy
from strats.exits.atr_trail_exit import AtrTrailExitConfig, AtrTrailExitStrategy


REPO_ROOT = Path(__file__).resolve().parents[1]
HAB_BARS = REPO_ROOT / "data" / "cache" / "normalized" / "hab_bars.csv"


def test_prepared_next_trade_date_is_strictly_later_than_date() -> None:
    """The engine pre-computes next_trade_date via shift(-1); each row's
    next_trade_date must be strictly later than its own date (or NaT on the
    last bar).
    """
    # Minimal synthetic bars, enough for ATR warmup.
    rows = [(f"2024-01-{d:02d}", 100.0, 101.0, 99.0, 100.0, 1.0)
            for d in [2, 3, 4, 5, 8, 9, 10, 11, 12, 15]]
    bars = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    bars["date"] = pd.to_datetime(bars["date"])
    bars["symbol"] = "X"
    bars["group_name"] = "G"
    bars["open_interest"] = 100.0
    bars["contract_multiplier"] = 10.0
    bars["commission"] = 5.0
    bars["slippage"] = 1.0

    cfg = EngineConfig(atr_period=2, adx_period=2)
    engine = StrategyEngine(
        config=cfg,
        entry_strategy=HLEntryStrategy(HLEntryConfig(period=3)),
        exit_strategy=AtrTrailExitStrategy(AtrTrailExitConfig(atr_mult=3.0)),
    )
    r = engine.run(bars)
    prep = r.prepared_data[["date", "next_trade_date"]].copy()
    prep["date"] = pd.to_datetime(prep["date"])
    non_last = prep.dropna(subset=["next_trade_date"]).copy()
    non_last["next_trade_date"] = pd.to_datetime(non_last["next_trade_date"])
    assert (non_last["next_trade_date"] > non_last["date"]).all(), (
        "next_trade_date must be strictly after date — same-bar fill would be look-ahead."
    )


def test_entry_and_exit_strategies_do_not_reference_settle() -> None:
    """Look-ahead invariant (1.6): settle is published after close, so any
    signal logic referencing it is look-ahead. Enforce by static scan — the
    entry and exit strategy directories must not contain the tokens `settle`
    or `settlement`.
    """
    for subdir in ["entries", "exits"]:
        d = REPO_ROOT / "strats" / subdir
        offenders: list[str] = []
        for py in d.glob("*.py"):
            if py.name.startswith("test_"):
                continue
            txt = py.read_text(encoding="utf-8")
            for token in ("settle", "settlement"):
                if token in txt:
                    offenders.append(f"{py.name}: '{token}'")
        assert not offenders, (
            f"signal/exit strategies must not reference settle "
            f"(look-ahead — settle is end-of-day data): {offenders}"
        )


@pytest.mark.skipif(not HAB_BARS.exists(), reason="hab_bars.csv not available")
def test_no_same_bar_fill_in_real_backtest() -> None:
    """Regression: run HL breakout on one liquid symbol; verify every trade
    has signal_date strictly before entry_date. A failure here would mean
    some code path lets a signal fill on its own bar (look-ahead).
    """
    bars = pd.read_csv(HAB_BARS, parse_dates=["date"])
    rb = bars[bars["symbol"] == "RB"].reset_index(drop=True)

    cfg = EngineConfig(
        initial_capital=1_000_000, risk_per_trade=0.01,
        atr_period=20, adx_period=20, stop_atr_mult=2.0,
        allow_short=True,
    )
    engine = StrategyEngine(
        config=cfg,
        entry_strategy=HLEntryStrategy(HLEntryConfig(period=21, allow_short=True)),
        exit_strategy=AtrTrailExitStrategy(AtrTrailExitConfig(atr_mult=3.0)),
    )
    r = engine.run(rb)
    if r.trades.empty:
        pytest.skip("no trades fired; can't check invariant")
    sig = pd.to_datetime(r.trades["signal_date"])
    ent = pd.to_datetime(r.trades["entry_date"])
    exi = pd.to_datetime(r.trades["exit_date"])
    assert (sig < ent).all(), "signal_date must strictly precede entry_date"
    assert (ent <= exi).all(), "entry_date must precede or equal exit_date"
