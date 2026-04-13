
import json
import math

import pandas as pd
import pytest

from horizontal_accumulation_breakout_v1 import (
    HABConfig,
    HorizontalAccumulationBreakoutV1,
    detect_hlh_pattern,
)


BASE_BARS = [
    (99.5, 101.0, 98.5, 100.0),
    (100.0, 100.6, 99.0, 99.5),
    (99.6, 100.8, 98.8, 100.0),
    (100.0, 101.0, 99.1, 100.2),
    (100.1, 100.9, 98.7, 100.0),
    (100.0, 100.7, 99.0, 99.9),
    (99.8, 100.95, 98.9, 100.1),
    (100.5, 103.2, 100.4, 103.1),  # signal day
]


def make_test_config(**overrides) -> HABConfig:
    base = dict(
        initial_capital=100_000,
        atr_period=3,
        bb_period=3,
        bb_percentile_lookback=5,
        bb_percentile_threshold=1.0,
        portfolio_risk_cap=1.0,
        group_risk_cap=1.0,
    )
    base.update(overrides)
    return HABConfig(**base)


def make_symbol_frame(bars, symbol="A", group="G1", start="2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(bars), freq="B")
    rows = []
    for date, (o, h, l, c) in zip(dates, bars):
        rows.append(
            {
                "date": date,
                "symbol": symbol,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1_000,
                "open_interest": 100,
                "contract_multiplier": 1,
                "commission": 0.0,
                "slippage": 0.0,
                "group_name": group,
            }
        )
    return pd.DataFrame(rows)


def make_frame(extra_bars, start="2025-01-01", symbol="A", group="G1") -> pd.DataFrame:
    return make_symbol_frame(BASE_BARS + list(extra_bars), symbol=symbol, group=group, start=start)


def test_hlh_box_detection_requires_high_low_high_round_trip() -> None:
    valid = detect_hlh_pattern(
        high_window=[101.0, 100.5, 100.8, 101.0, 100.6, 100.7, 100.9],
        low_window=[98.5, 99.0, 98.8, 99.1, 98.7, 99.0, 98.9],
        box_high=101.0,
        box_low=98.5,
        tol=0.5,
    )
    invalid = detect_hlh_pattern(
        high_window=[101.0, 100.6, 100.7, 100.8, 100.7, 100.8, 100.85],
        low_window=[99.0, 99.1, 99.0, 98.5, 98.8, 98.9, 99.0],
        box_high=101.0,
        box_low=98.5,
        tol=0.1,
    )

    assert valid == (True, True, True, True)
    assert invalid == (False, True, True, False)


def test_bandwidth_uses_positive_denominator_when_bb_mid_is_negative() -> None:
    bars = [
        (-10.0, -9.5, -10.5, -10.1),
        (-10.1, -9.8, -10.4, -10.0),
        (-10.0, -9.6, -10.3, -9.8),
        (-9.9, -9.4, -10.2, -9.7),
        (-9.8, -9.3, -10.1, -9.6),
        (-9.7, -9.2, -10.0, -9.5),
    ]
    df = make_symbol_frame(bars, symbol="NEG", group="GN", start="2025-01-01")
    engine = HorizontalAccumulationBreakoutV1(
        make_test_config(atr_period=3, bb_period=3, bb_percentile_lookback=3)
    )
    prepared = engine.prepare_data(df)

    finite_denom = prepared["bandwidth_denom"].dropna()
    finite_bandwidth = prepared["bandwidth"].dropna()

    assert not finite_denom.empty
    assert (finite_denom > 0).all()
    assert (finite_bandwidth >= 0).all()


def test_structure_fail_exits_next_open_after_close_back_into_box() -> None:
    df = make_frame(
        [
            (103.3, 104.0, 103.0, 103.6),  # entry day
            (103.5, 104.2, 100.4, 100.8),  # close back into box
            (100.6, 101.2, 100.1, 100.9),  # next-open exit
            (101.0, 101.5, 100.8, 101.3),
        ]
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config())
    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "STRUCT_FAIL"
    assert trade["exit_date"] == df.loc[10, "date"]
    assert trade["exit_fill"] == pytest.approx(df.loc[10, "open"])
    assert trade["entry_date"] == df.loc[8, "date"]


def test_time_fail_exits_next_open_after_five_completed_bars_without_half_r() -> None:
    df = make_frame(
        [
            (103.3, 104.0, 102.9, 103.4),
            (103.4, 104.5, 103.0, 103.5),
            (103.5, 105.2, 103.2, 103.8),
            (103.8, 105.4, 103.5, 104.0),
            (104.0, 105.5, 103.8, 104.2),  # 5th completed bar, still < 0.5R target
            (104.1, 104.4, 103.9, 104.0),  # next-open exit
            (104.0, 104.2, 103.7, 103.9),
        ],
        start="2025-02-03",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config())
    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "TIME_FAIL"
    assert trade["exit_date"] == df.loc[13, "date"]
    assert trade["exit_fill"] == pytest.approx(df.loc[13, "open"])


def test_gap_open_below_stop_exits_at_open_price_not_at_stop() -> None:
    df = make_frame(
        [
            (103.3, 104.0, 102.9, 103.4),
            (96.0, 97.0, 95.0, 96.5),  # existing position gaps below active stop
            (96.6, 97.2, 96.0, 96.9),
            (97.0, 97.5, 96.8, 97.2),
        ],
        start="2025-03-03",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config())
    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "STOP_GAP"
    assert trade["exit_date"] == df.loc[9, "date"]
    assert trade["exit_fill"] == pytest.approx(df.loc[9, "open"])
    assert trade["exit_fill"] < trade["initial_stop"]


def test_position_size_is_frozen_from_signal_close_not_next_open() -> None:
    cfg = make_test_config()
    df = make_frame(
        [
            (110.0, 110.5, 109.5, 109.8),
            (109.8, 110.1, 109.0, 109.3),
        ],
        start="2025-04-01",
    )
    engine = HorizontalAccumulationBreakoutV1(cfg)
    result = engine.run(df)

    assert result.trades.empty
    assert len(result.open_positions) == 1

    position = result.open_positions.iloc[0]
    signal_close = df.loc[7, "close"]
    next_open = df.loc[8, "open"]
    initial_stop = position["initial_stop"]

    expected_qty_from_signal_close = math.floor(
        cfg.initial_capital * cfg.risk_per_trade / (signal_close - initial_stop)
    )
    hypothetical_qty_from_next_open = math.floor(
        cfg.initial_capital * cfg.risk_per_trade / (next_open - initial_stop)
    )

    assert position["qty"] == expected_qty_from_signal_close
    assert position["qty"] != hypothetical_qty_from_next_open


def test_active_stop_series_records_computed_and_effective_dates() -> None:
    df = make_frame(
        [
            (110.0, 110.5, 109.5, 109.8),
            (109.8, 110.1, 109.0, 109.3),
        ],
        start="2025-04-21",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config())
    result = engine.run(df)

    position = result.open_positions.iloc[0]
    stop_series = json.loads(position["active_stop_series"])

    assert stop_series[0]["phase"] == "signal_init"
    assert stop_series[0]["computed_on"] == df.loc[7, "date"].strftime("%Y-%m-%d")
    assert stop_series[0]["effective_from"] == df.loc[8, "date"].strftime("%Y-%m-%d")

    assert stop_series[1]["phase"] == "close_update"
    assert stop_series[1]["computed_on"] == df.loc[8, "date"].strftime("%Y-%m-%d")
    assert stop_series[1]["effective_from"] == df.loc[9, "date"].strftime("%Y-%m-%d")


def test_cancel_entry_when_open_below_initial_stop() -> None:
    df = make_frame(
        [
            (96.0, 97.0, 95.0, 96.5),  # next open invalidates initial stop
            (96.6, 97.2, 96.0, 96.9),
        ],
        start="2025-05-01",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config())
    result = engine.run(df)

    assert result.trades.empty
    assert result.open_positions.empty
    assert len(result.cancelled_entries) == 1

    cancelled = result.cancelled_entries.iloc[0]
    assert cancelled["cancel_reason"] == "OPEN_INVALIDATES_STOP"
    assert cancelled["cancel_date"] == df.loc[8, "date"]
    assert cancelled["attempted_entry_fill"] == pytest.approx(df.loc[8, "open"])
    assert cancelled["attempted_entry_fill"] <= cancelled["initial_stop"]


def test_pending_next_open_exit_releases_risk_budget() -> None:
    cfg = make_test_config(portfolio_risk_cap=0.025, group_risk_cap=1.0)

    symbol_a = make_frame(
        [
            (103.3, 104.0, 103.0, 103.6),  # entry day
            (103.5, 104.2, 100.4, 100.8),  # structure fail -> exits next open
            (100.6, 101.2, 100.1, 100.9),
        ],
        start="2025-06-02",
        symbol="A",
        group="GA",
    )

    symbol_b_bars = [
        (99.8, 100.6, 99.1, 100.0),
        (100.0, 100.7, 99.3, 100.1),
        *BASE_BARS[:7],
        BASE_BARS[7],  # signal day aligns with A's structure-fail day
        (103.3, 104.0, 103.0, 103.6),  # entry day
    ]
    symbol_b = make_symbol_frame(symbol_b_bars, symbol="B", group="GB", start="2025-06-02")

    df = pd.concat([symbol_a, symbol_b], ignore_index=True)
    engine = HorizontalAccumulationBreakoutV1(cfg)
    result = engine.run(df)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["symbol"] == "A"
    assert result.trades.iloc[0]["exit_reason"] == "STRUCT_FAIL"

    assert len(result.open_positions) == 1
    b_position = result.open_positions.iloc[0]
    assert b_position["symbol"] == "B"

    b_signal_date = symbol_b.loc[9, "date"]
    b_entry_date = symbol_b.loc[10, "date"]
    b_signal_status = result.daily_status[
        (result.daily_status["symbol"] == "B")
        & (result.daily_status["date"] == b_signal_date)
    ].iloc[0]
    assert pd.isna(b_signal_status["risk_reject_reason"])

    a_trade = result.trades.iloc[0]
    a_close_back_date = symbol_a.loc[9, "date"]
    portfolio_row = result.portfolio_daily[result.portfolio_daily["date"] == a_close_back_date].iloc[0]
    a_risk_if_counted = max(symbol_a.loc[9, "close"] - a_trade["initial_stop"], 0.0) * a_trade["qty"]

    assert b_position["entry_date"] == b_entry_date
    assert b_position["estimated_order_risk"] <= portfolio_row["portfolio_risk_cap"]
    assert a_risk_if_counted + b_position["estimated_order_risk"] > portfolio_row["portfolio_risk_cap"]


def test_actual_fill_risk_blowout_is_logged() -> None:
    df = make_frame(
        [
            (108.0, 109.0, 107.0, 108.5),  # entry day gaps up but stays above stop
            (97.0, 98.0, 96.5, 97.5),  # later stop gap to close the trade
            (97.2, 97.8, 96.9, 97.4),
        ],
        start="2025-07-01",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config())
    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["actual_initial_risk"] == pytest.approx(trade["entry_fill"] - trade["initial_stop"])
    assert trade["estimated_order_risk"] < trade["actual_order_risk"]
    assert trade["risk_blowout_vs_estimate"] == pytest.approx(
        trade["actual_order_risk"] - trade["estimated_order_risk"]
    )
    assert trade["risk_blowout_ratio"] == pytest.approx(
        trade["actual_order_risk"] / trade["estimated_order_risk"]
    )


def test_invalid_ohlc_input_raises() -> None:
    df = make_frame([], start="2025-08-01")
    df.loc[0, "high"] = 99.0  # below open=99.5 and close=100.0

    engine = HorizontalAccumulationBreakoutV1(make_test_config())
    with pytest.raises(ValueError):
        engine.prepare_data(df)


def test_entry_bar_intraday_stop_after_fill() -> None:
    df = make_frame(
        [
            (103.3, 104.0, 97.0, 103.4),  # open above stop, intraday low hits stop
            (103.2, 103.6, 102.9, 103.1),
        ],
        start="2025-09-01",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config())
    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_date"] == df.loc[8, "date"]
    assert trade["exit_date"] == df.loc[8, "date"]
    assert trade["exit_reason"] == "STOP_INTRADAY"
    assert trade["exit_fill"] == pytest.approx(trade["initial_stop"])
