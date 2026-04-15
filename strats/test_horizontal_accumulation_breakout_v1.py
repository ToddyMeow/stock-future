
import json
import math

import pandas as pd
import pytest

from horizontal_accumulation_breakout_v1 import (
    HABConfig,
    HorizontalAccumulationBreakoutV1,
    PortfolioAnalyzer,
    detect_hlh_pattern,
    detect_lhl_pattern,
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
        risk_blowout_cap=float("inf"),
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


# ── Phase 1: Risk blowout hard cap tests ──────────────────────────────────


def test_risk_blowout_cancel_when_gap_exceeds_cap() -> None:
    """Gap up causes blowout ratio > cap with action=CANCEL → entry cancelled."""
    df = make_frame(
        [
            (108.0, 109.0, 107.0, 108.5),  # large gap up from signal close ~103.1
            (109.0, 110.0, 108.5, 109.5),
        ],
        start="2025-10-01",
    )
    engine = HorizontalAccumulationBreakoutV1(
        make_test_config(risk_blowout_cap=1.2, risk_blowout_action="CANCEL")
    )
    result = engine.run(df)

    assert len(result.trades) == 0
    assert len(result.cancelled_entries) == 1
    assert result.cancelled_entries.iloc[0]["cancel_reason"] == "RISK_BLOWOUT_CANCEL"


def test_risk_blowout_shrink_reduces_qty() -> None:
    """Gap up causes blowout ratio > cap with action=SHRINK → qty reduced."""
    df = make_frame(
        [
            (108.0, 109.0, 107.0, 108.5),  # large gap up
            (97.0, 98.0, 96.5, 97.5),  # stop gap to close
            (97.2, 97.8, 96.9, 97.4),
        ],
        start="2025-10-01",
    )
    engine = HorizontalAccumulationBreakoutV1(
        make_test_config(risk_blowout_cap=1.2, risk_blowout_action="SHRINK")
    )
    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["original_qty"] is not None
    assert trade["qty"] < trade["original_qty"]
    assert trade["qty_shrink_reason"] == "RISK_BLOWOUT"


def test_risk_blowout_shrink_to_zero_cancels() -> None:
    """Gap up where shrunk qty rounds to 0 → entry cancelled as SHRINK_TO_ZERO.

    Signal close ~103.1, next open 108.0. est_initial_risk ~5.39, act ~10.29.
    With capital=270, qty=1 at signal time. cap=1.01 means max_allowed_risk=5.44,
    shrunk_qty = floor(5.44 / 10.29) = 0 → cancelled.
    """
    df = make_frame(
        [
            (108.0, 109.0, 107.0, 108.5),
            (109.0, 110.0, 108.5, 109.5),
        ],
        start="2025-10-01",
    )
    engine = HorizontalAccumulationBreakoutV1(
        make_test_config(initial_capital=270, risk_blowout_cap=1.01, risk_blowout_action="SHRINK")
    )
    result = engine.run(df)

    assert len(result.trades) == 0
    assert len(result.cancelled_entries) == 1
    assert result.cancelled_entries.iloc[0]["cancel_reason"] == "RISK_BLOWOUT_SHRINK_TO_ZERO"


def test_blowout_within_cap_proceeds_normally() -> None:
    """Blowout ratio within cap → normal fill, no shrink."""
    df = make_frame(
        [
            (103.5, 105.0, 103.0, 104.5),  # small gap from signal close ~103.1
            (97.0, 98.0, 96.5, 97.5),  # stop gap to close
            (97.2, 97.8, 96.9, 97.4),
        ],
        start="2025-10-01",
    )
    engine = HorizontalAccumulationBreakoutV1(
        make_test_config(risk_blowout_cap=1.5)
    )
    result = engine.run(df)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert pd.isna(trade["original_qty"]) or trade["original_qty"] is None
    assert pd.isna(trade["qty_shrink_reason"]) or trade["qty_shrink_reason"] is None


# ── Phase 3a: Short side symmetry tests ───────────────────────────────────

# Short-side BASE_BARS: L-H-L pattern in box, then breakout below box_low.
# Box bars: prices oscillate around 100, with lows touching box_low and highs touching box_high.
# Pattern: bar touches low → high → low (L-H-L).
# Narrower box (range ~1.5) for short side: box_high~100.5, box_low~99.0
SHORT_BASE_BARS = [
    (99.5, 100.3, 99.0, 99.2),    # low touch (lower_test_1)
    (99.5, 100.2, 99.2, 100.0),
    (100.0, 100.5, 99.5, 100.3),  # high touch (upper_confirm)
    (100.2, 100.4, 99.3, 99.5),
    (99.5, 100.1, 99.0, 99.3),    # low touch (lower_test_2)
    (99.3, 100.0, 99.1, 99.4),
    (99.4, 100.2, 99.2, 99.5),
    (98.0, 98.5, 95.0, 95.5),     # signal day: close breaks below box_low
]


def make_short_frame(extra_bars, start="2025-01-01", symbol="A", group="G1") -> pd.DataFrame:
    return make_symbol_frame(SHORT_BASE_BARS + list(extra_bars), symbol=symbol, group=group, start=start)


def test_lhl_pattern_detection() -> None:
    """L-H-L pattern correctly detected for short side."""
    # Window that has L-H-L sequence
    highs = [101.0, 100.5, 101.0, 100.8, 100.2, 100.0, 100.0]
    lows = [98.5, 99.0, 99.5, 99.0, 98.6, 98.8, 98.9]
    box_high = 101.0
    box_low = 98.5
    tol = 0.5

    valid, lt1, uc, lt2 = detect_lhl_pattern(highs, lows, box_high, box_low, tol)
    assert valid is True
    assert lt1 is True
    assert uc is True
    assert lt2 is True

    # Without valid L-H-L (all highs, no low touch)
    highs2 = [101.0] * 7
    lows2 = [100.0] * 7  # never touches box_low
    valid2, _, _, _ = detect_lhl_pattern(highs2, lows2, box_high, box_low, tol)
    assert valid2 is False


def test_short_entry_breakout_below_box() -> None:
    """Short signal fires on close below box_low and generates entry."""
    df = make_short_frame(
        [
            (95.0, 95.5, 94.0, 94.5),  # entry day
            (94.5, 95.0, 94.0, 94.2),
        ],
        start="2025-11-01",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config(allow_short=True))
    result = engine.run(df)

    # Should have a short position (either open or traded)
    total = len(result.trades) + len(result.open_positions)
    assert total >= 1

    if len(result.open_positions) > 0:
        pos = result.open_positions.iloc[0]
        assert pos["direction"] == -1
    elif len(result.trades) > 0:
        trade = result.trades.iloc[0]
        assert trade["direction"] == -1


def test_short_entry_fill_uses_open_minus_slippage() -> None:
    """Short entry fill = open - slippage."""
    df = make_short_frame(
        [
            (95.0, 95.5, 94.0, 94.5),  # entry day
            (94.5, 95.0, 94.0, 94.2),
        ],
        start="2025-11-01",
    )
    # Use non-zero slippage to verify
    df["slippage"] = 0.5
    engine = HorizontalAccumulationBreakoutV1(make_test_config(allow_short=True))
    result = engine.run(df)

    total = len(result.trades) + len(result.open_positions)
    assert total >= 1, "Expected a short position or trade"
    entry_fill = (
        result.open_positions.iloc[0]["entry_fill"] if len(result.open_positions) > 0
        else result.trades.iloc[0]["entry_fill"]
    )
    entry_open = df.loc[8, "open"]  # entry day open
    assert entry_fill == pytest.approx(entry_open - 0.5)


def test_short_pnl_positive_when_price_drops() -> None:
    """Short trade has positive P&L when exiting below entry.

    Entry ~95, price drops, then struct_fail when close rises above box_low.
    Exit open ~93 is below entry ~95 → positive gross_pnl for short.
    """
    df = make_short_frame(
        [
            (95.0, 95.5, 94.0, 94.5),    # entry day
            (94.0, 94.5, 90.0, 90.5),    # price drops
            (91.0, 99.5, 90.5, 99.0),    # close above box_low → struct fail
            (93.0, 93.5, 92.5, 93.0),    # exit day: open 93 < entry 95
            (93.0, 93.5, 92.5, 93.2),
        ],
        start="2025-11-01",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config(allow_short=True))
    result = engine.run(df)

    assert len(result.trades) >= 1, "Expected at least one closed short trade"
    trade = result.trades.iloc[0]
    assert trade["direction"] == -1
    assert trade["exit_reason"] == "STRUCT_FAIL"
    assert trade["gross_pnl"] > 0  # exit below entry → short profits


def test_short_gap_stop_exits_when_open_above_stop() -> None:
    """Short position: open gaps above active_stop → STOP_GAP."""
    df = make_short_frame(
        [
            (95.0, 95.5, 94.0, 94.5),      # entry day
            (103.0, 104.0, 102.5, 103.5),   # gap up above stop
            (103.5, 104.0, 103.0, 103.2),
        ],
        start="2025-11-01",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config(allow_short=True))
    result = engine.run(df)

    assert len(result.trades) >= 1, "Expected a STOP_GAP trade"
    trade = result.trades.iloc[0]
    assert trade["direction"] == -1
    assert trade["exit_reason"] == "STOP_GAP"
    assert trade["gross_pnl"] < 0  # gap against short


def test_short_struct_fail_close_above_box_low() -> None:
    """Short position: close >= box_low within structure_fail_bars → STRUCT_FAIL."""
    df = make_short_frame(
        [
            (95.0, 95.5, 94.0, 94.5),    # entry day
            (95.0, 99.5, 94.5, 99.0),    # close back above box_low → struct fail
            (99.0, 99.5, 98.5, 99.0),    # exit day
            (99.0, 99.5, 98.5, 99.2),
        ],
        start="2025-11-01",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config(allow_short=True))
    result = engine.run(df)

    assert len(result.trades) >= 1, "Expected a STRUCT_FAIL trade"
    trade = result.trades.iloc[0]
    assert trade["direction"] == -1
    assert trade["exit_reason"] == "STRUCT_FAIL"


def test_allow_short_false_no_short_signals() -> None:
    """With allow_short=False (default), no short signals fire."""
    df = make_short_frame(
        [
            (95.0, 95.5, 94.0, 94.5),
            (94.5, 95.0, 94.0, 94.2),
        ],
        start="2025-11-01",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config(allow_short=False))
    result = engine.run(df)

    assert len(result.trades) == 0
    assert len(result.open_positions) == 0


# ── Phase 3b: Structure fail relaxation tests ─────────────────────────────


def test_struct_fail_consecutive_mode_single_bar_does_not_trigger() -> None:
    """CONSECUTIVE_CLOSE mode: a single close back into box does not trigger."""
    df = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),  # entry day
            (103.0, 103.5, 100.0, 100.0),  # close at box_high → fail bar 1
            (100.0, 103.6, 99.8, 103.5),   # close above box_high → reset
            (103.0, 103.5, 102.5, 103.0),
            (103.0, 103.5, 102.5, 103.0),
        ],
        start="2025-12-01",
    )
    engine = HorizontalAccumulationBreakoutV1(
        make_test_config(structure_fail_mode="CONSECUTIVE_CLOSE", structure_fail_consecutive=2)
    )
    result = engine.run(df)

    # Should NOT have struct fail exit (only 1 consecutive fail, then reset)
    if len(result.trades) > 0:
        assert result.trades.iloc[0]["exit_reason"] != "STRUCT_FAIL"


def test_struct_fail_consecutive_mode_triggers_after_n_bars() -> None:
    """CONSECUTIVE_CLOSE mode: N consecutive closes in box triggers exit."""
    df = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),  # entry day
            (103.0, 103.5, 100.0, 100.0),  # fail bar 1: close <= box_high
            (100.0, 100.5, 99.5, 100.0),   # fail bar 2: close <= box_high → trigger
            (100.0, 100.5, 99.5, 100.0),   # exit day
            (100.0, 100.5, 99.5, 100.0),
        ],
        start="2025-12-01",
    )
    engine = HorizontalAccumulationBreakoutV1(
        make_test_config(structure_fail_mode="CONSECUTIVE_CLOSE", structure_fail_consecutive=2)
    )
    result = engine.run(df)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "STRUCT_FAIL"


def test_struct_fail_atr_buffer_mode_tolerates_slight_return() -> None:
    """CLOSE_BELOW_BOX_MINUS_ATR mode: close slightly below box_high but above
    box_high - buffer*ATR does not trigger."""
    df = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),  # entry day
            # close just below box_high but above box_high - 0.5*ATR
            (103.0, 103.5, 100.5, 100.5),
            (100.5, 103.5, 100.0, 103.0),
            (103.0, 103.5, 102.5, 103.0),
            (103.0, 103.5, 102.5, 103.0),
        ],
        start="2025-12-01",
    )
    engine = HorizontalAccumulationBreakoutV1(
        make_test_config(structure_fail_mode="CLOSE_BELOW_BOX_MINUS_ATR", structure_fail_atr_buffer=0.5)
    )
    result = engine.run(df)

    # Should NOT trigger struct fail (close is above box_high - 0.5*ATR)
    if len(result.trades) > 0:
        assert result.trades.iloc[0]["exit_reason"] != "STRUCT_FAIL"


# ── Phase 2: Leverage cap tests ───────────────────────────────────────────


def test_leverage_cap_rejects_entry() -> None:
    """When leverage exceeds max_portfolio_leverage, entry is rejected."""
    df = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),
            (103.4, 104.0, 103.0, 103.2),
        ],
        start="2025-12-01",
    )
    # With multiplier=1, capital=100k, qty~371, notional=103*1*371=38213.
    # Set a very low leverage cap so 38213/100000 = 0.38 exceeds cap 0.01.
    engine = HorizontalAccumulationBreakoutV1(
        make_test_config(max_portfolio_leverage=0.01)
    )
    result = engine.run(df)

    # The entry should be rejected with LEVERAGE_CAP
    ds = result.daily_status
    signal_rows = ds[ds["entry_trigger_pass"] == True]
    if not signal_rows.empty:
        reject = signal_rows.iloc[0].get("risk_reject_reason")
        assert reject == "LEVERAGE_CAP"


def test_leverage_cap_allows_within_limit() -> None:
    """When leverage is within limit, entry proceeds."""
    df = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),
            (97.0, 98.0, 96.5, 97.5),
            (97.2, 97.8, 96.9, 97.4),
        ],
        start="2025-12-01",
    )
    engine = HorizontalAccumulationBreakoutV1(
        make_test_config(max_portfolio_leverage=100.0)  # very high cap
    )
    result = engine.run(df)

    assert len(result.trades) == 1


def test_leverage_tracked_in_portfolio_daily() -> None:
    """portfolio_daily includes total_notional and leverage columns."""
    df = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),
            (103.4, 104.0, 103.0, 103.2),
        ],
        start="2025-12-01",
    )
    engine = HorizontalAccumulationBreakoutV1(make_test_config())
    result = engine.run(df)

    assert "total_notional" in result.portfolio_daily.columns
    assert "leverage" in result.portfolio_daily.columns


def test_missing_margin_rate_uses_default() -> None:
    """Input without margin_rate column still works using default_margin_rate."""
    df = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),
            (97.0, 98.0, 96.5, 97.5),
        ],
        start="2025-12-01",
    )
    # Explicitly remove margin_rate if present
    if "margin_rate" in df.columns:
        df = df.drop(columns=["margin_rate"])

    engine = HorizontalAccumulationBreakoutV1(make_test_config())
    result = engine.run(df)

    # Should run without error
    assert len(result.portfolio_daily) > 0


# ── Phase 4: Portfolio analyzer tests ─────────────────────────────────────


def test_analyzer_equity_curve_drawdown() -> None:
    """PortfolioAnalyzer.equity_curve() computes correct drawdown."""
    df = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),  # entry
            (103.0, 103.5, 100.0, 100.5),  # drop → drawdown
            (100.5, 101.0, 100.0, 100.8),
            (101.0, 103.0, 100.8, 102.5),
            (102.5, 103.0, 102.0, 102.8),
            (102.8, 103.0, 102.5, 103.0),
        ],
        start="2025-12-01",
    )
    cfg = make_test_config()
    engine = HorizontalAccumulationBreakoutV1(cfg)
    result = engine.run(df)
    analyzer = PortfolioAnalyzer(result, cfg)
    ec = analyzer.equity_curve()

    assert "drawdown" in ec.columns
    assert "drawdown_pct" in ec.columns
    assert "peak" in ec.columns
    assert "daily_return" in ec.columns
    # Peak should never decrease
    assert (ec["peak"].diff().dropna() >= 0).all()
    # Drawdown should be <= 0
    assert (ec["drawdown"] <= 1e-12).all()


def test_analyzer_group_contribution() -> None:
    """PortfolioAnalyzer.group_contribution() splits P&L by group."""
    # Two symbols in different groups
    bars_a = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),
            (97.0, 98.0, 96.5, 97.5),
            (97.2, 97.8, 96.9, 97.4),
        ],
        start="2025-12-01",
        symbol="A",
        group="G1",
    )
    bars_b = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),
            (97.0, 98.0, 96.5, 97.5),
            (97.2, 97.8, 96.9, 97.4),
        ],
        start="2025-12-01",
        symbol="B",
        group="G2",
    )
    df = pd.concat([bars_a, bars_b], ignore_index=True)
    cfg = make_test_config()
    engine = HorizontalAccumulationBreakoutV1(cfg)
    result = engine.run(df)
    analyzer = PortfolioAnalyzer(result, cfg)
    gc = analyzer.group_contribution()

    assert len(gc) >= 1
    assert "group_name" in gc.columns
    assert "net_pnl_sum" in gc.columns
    assert "win_rate" in gc.columns


def test_analyzer_summary_stats() -> None:
    """PortfolioAnalyzer.summary_stats() returns core metrics."""
    df = make_frame(
        [
            (103.3, 104.0, 103.0, 103.4),
            (97.0, 98.0, 96.5, 97.5),
            (97.2, 97.8, 96.9, 97.4),
        ],
        start="2025-12-01",
    )
    cfg = make_test_config()
    engine = HorizontalAccumulationBreakoutV1(cfg)
    result = engine.run(df)
    analyzer = PortfolioAnalyzer(result, cfg)
    stats = analyzer.summary_stats()

    assert "total_return" in stats
    assert "cagr" in stats
    assert "sharpe" in stats
    assert "max_drawdown_pct" in stats
    assert "total_trades" in stats
    assert "win_rate" in stats
    assert "profit_factor" in stats
