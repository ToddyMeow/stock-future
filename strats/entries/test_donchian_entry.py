"""Tests for DonchianEntryStrategy."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure strats/ is importable as a package
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from strats.entries.donchian_entry import DonchianEntryConfig, DonchianEntryStrategy
from strats.helpers import wilder_atr


def _make_df(bars, start="2025-01-01"):
    """Build a single-symbol DataFrame with atr/atr_ref/next_trade_date."""
    dates = pd.date_range(start, periods=len(bars), freq="B")
    rows = [
        {"date": d, "symbol": "A", "open": o, "high": h, "low": l, "close": c,
         "volume": 1000, "open_interest": 100, "contract_multiplier": 1,
         "commission": 0.0, "slippage": 0.0, "group_name": "G1"}
        for d, (o, h, l, c) in zip(dates, bars)
    ]
    df = pd.DataFrame(rows)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    df["atr"] = wilder_atr(high, low, close, period=3)
    df["atr_ref"] = df["atr"].shift(1)
    df["next_trade_date"] = df["date"].shift(-1)
    return df


# Bars that stay in a channel for 5 bars, then break out up
CHANNEL_BARS = [
    (100.0, 102.0, 98.0, 101.0),
    (101.0, 103.0, 99.0, 100.5),
    (100.5, 102.5, 98.5, 101.5),
    (101.0, 103.0, 99.5, 100.0),
    (100.0, 102.0, 98.0, 101.0),
]


def test_donchian_long_signal_on_new_high():
    """Close above N-day high triggers a long signal."""
    bars = CHANNEL_BARS + [
        (101.0, 105.0, 100.5, 104.0),  # close=104 > donchian_high(5)=103
    ]
    df = _make_df(bars)
    strategy = DonchianEntryStrategy(DonchianEntryConfig(donchian_period=5))
    result = strategy.prepare_signals(df)

    signal_bar = result.iloc[-1]
    assert signal_bar["entry_trigger_pass"] == True
    assert signal_bar["entry_direction"] == 1
    assert pd.notna(signal_bar["initial_stop"])
    # Stop should be below donchian_low
    assert signal_bar["initial_stop"] < signal_bar["donchian_low"]


def test_donchian_short_signal_on_new_low():
    """Close below N-day low triggers a short signal when allow_short=True."""
    bars = CHANNEL_BARS + [
        (99.0, 99.5, 95.0, 96.0),  # close=96 < donchian_low(5)=98
    ]
    df = _make_df(bars)
    strategy = DonchianEntryStrategy(DonchianEntryConfig(donchian_period=5, allow_short=True))
    result = strategy.prepare_signals(df)

    signal_bar = result.iloc[-1]
    assert signal_bar["entry_trigger_pass"] == True
    assert signal_bar["entry_direction"] == -1
    # Stop should be above donchian_high
    assert signal_bar["initial_stop"] > signal_bar["donchian_high"]


def test_donchian_no_signal_within_channel():
    """Price within the channel produces no signal."""
    bars = CHANNEL_BARS + [
        (100.0, 101.0, 99.0, 100.5),  # stays within 98-103 channel
    ]
    df = _make_df(bars)
    strategy = DonchianEntryStrategy(DonchianEntryConfig(donchian_period=5))
    result = strategy.prepare_signals(df)

    signal_bar = result.iloc[-1]
    assert signal_bar["entry_trigger_pass"] == False
    assert signal_bar["entry_direction"] == 0


def test_donchian_no_short_when_disabled():
    """New low does not produce short signal when allow_short=False."""
    bars = CHANNEL_BARS + [
        (99.0, 99.5, 95.0, 96.0),
    ]
    df = _make_df(bars)
    strategy = DonchianEntryStrategy(DonchianEntryConfig(donchian_period=5, allow_short=False))
    result = strategy.prepare_signals(df)

    signal_bar = result.iloc[-1]
    assert signal_bar["entry_trigger_pass"] == False


def test_donchian_stop_uses_atr_buffer():
    """Initial stop is donchian_low - stop_mult * atr_ref for long."""
    bars = CHANNEL_BARS + [
        (101.0, 105.0, 100.5, 104.0),
    ]
    df = _make_df(bars)
    strategy = DonchianEntryStrategy(DonchianEntryConfig(
        donchian_period=5, initial_stop_atr_mult=2.0,
    ))
    result = strategy.prepare_signals(df)

    signal_bar = result.iloc[-1]
    expected_stop = signal_bar["donchian_low"] - 2.0 * signal_bar["atr_ref"]
    assert signal_bar["initial_stop"] == pytest.approx(expected_stop)


def test_donchian_metadata():
    """build_pending_entry_metadata returns donchian_high and donchian_low."""
    bars = CHANNEL_BARS + [
        (101.0, 105.0, 100.5, 104.0),
    ]
    df = _make_df(bars)
    strategy = DonchianEntryStrategy(DonchianEntryConfig(donchian_period=5))
    result = strategy.prepare_signals(df)

    signal_bar = result.iloc[-1]
    meta = strategy.build_pending_entry_metadata(signal_bar)
    assert "donchian_high" in meta
    assert "donchian_low" in meta
    assert meta["donchian_high"] == pytest.approx(signal_bar["donchian_high"])


def test_donchian_with_engine_end_to_end():
    """Full StrategyEngine run with Donchian entry + HAB exit."""
    from strats.engine import EngineConfig, StrategyEngine
    from strats.exits.hab_exit import HABExitConfig, HABExitStrategy

    # Need enough bars for donchian period + signal + entry + exit
    bars = CHANNEL_BARS + [
        (101.0, 105.0, 100.5, 104.0),  # signal day: breakout
        (104.0, 105.0, 103.5, 104.5),  # entry day
        (104.5, 105.0, 100.0, 100.5),  # close back in channel → struct fail
        (100.5, 101.0, 100.0, 100.8),  # exit day
        (100.8, 101.0, 100.5, 100.9),
    ]
    dates = pd.date_range("2025-01-01", periods=len(bars), freq="B")
    rows = [
        {"date": d, "symbol": "A", "open": o, "high": h, "low": l, "close": c,
         "volume": 1000, "open_interest": 100, "contract_multiplier": 1,
         "commission": 0.0, "slippage": 0.0, "group_name": "G1"}
        for d, (o, h, l, c) in zip(dates, bars)
    ]
    df = pd.DataFrame(rows)

    engine = StrategyEngine(
        config=EngineConfig(
            initial_capital=100_000, atr_period=3,
            risk_blowout_cap=float("inf"),
            portfolio_risk_cap=1.0, group_risk_cap=1.0,
        ),
        entry_strategy=DonchianEntryStrategy(DonchianEntryConfig(donchian_period=5)),
        exit_strategy=HABExitStrategy(HABExitConfig(structure_fail_bars=15)),
    )
    result = engine.run(df)

    total = len(result.trades) + len(result.open_positions)
    assert total >= 1, "Expected at least one trade or open position from Donchian + HABExit"

    # Verify R is set correctly (from Donchian stop, not HAB stop)
    if len(result.trades) > 0:
        trade = result.trades.iloc[0]
        assert trade["direction"] == 1
        assert trade["r_price"] > 0
        # Donchian metadata should be in the trade record
        assert "donchian_high" in trade.index
    elif len(result.open_positions) > 0:
        pos = result.open_positions.iloc[0]
        assert pos["direction"] == 1
        assert pos["r_price"] > 0
