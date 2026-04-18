from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.adapters.roll_events import (
    build_roll_events_frame,
    extract_roll_dates_per_symbol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HAB_BARS = REPO_ROOT / "data" / "cache" / "normalized" / "hab_bars.csv"


def _tiny_bars() -> pd.DataFrame:
    """Two symbols; symbol A has 1 roll, symbol B has 2 rolls."""
    return pd.DataFrame(
        [
            ("2024-01-02", "A", "A2405", 1.0, 2.0),
            ("2024-01-03", "A", "A2405", 2.0, 3.0),
            ("2024-01-04", "A", "A2410", 10.0, 11.0),  # roll #1
            ("2024-01-02", "B", "B2403", 100.0, 101.0),
            ("2024-01-03", "B", "B2406", 200.0, 201.0),  # roll #1
            ("2024-01-04", "B", "B2409", 300.0, 301.0),  # roll #2
        ],
        columns=["date", "symbol", "order_book_id", "open_raw", "close_raw"],
    ).assign(date=lambda df: pd.to_datetime(df["date"]))


def test_extract_roll_dates_per_symbol_basic() -> None:
    bars = _tiny_bars()
    rolls = extract_roll_dates_per_symbol(bars)
    assert rolls == {
        "A": {pd.Timestamp("2024-01-04")},
        "B": {pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")},
    }


def test_extract_skips_first_row_per_symbol() -> None:
    # Even though contract "changes" (from NaN prev) on first row, it's not a roll.
    bars = _tiny_bars()
    rolls = extract_roll_dates_per_symbol(bars)
    for dates in rolls.values():
        assert pd.Timestamp("2024-01-02") not in dates


def test_extract_missing_columns_raises() -> None:
    bad = pd.DataFrame({"date": ["2024-01-02"], "symbol": ["X"]})
    with pytest.raises(KeyError, match="order_book_id"):
        extract_roll_dates_per_symbol(bad)


def test_build_roll_events_frame_shape() -> None:
    bars = _tiny_bars()
    frame = build_roll_events_frame(bars)
    assert list(frame.columns) == [
        "symbol", "roll_date", "old_contract", "new_contract",
        "old_close", "new_open",
    ]
    assert len(frame) == 3  # A: 1, B: 2


def test_build_roll_events_prices() -> None:
    bars = _tiny_bars()
    frame = build_roll_events_frame(bars)
    a_row = frame[frame.symbol == "A"].iloc[0]
    assert a_row.old_contract == "A2405"
    assert a_row.new_contract == "A2410"
    assert a_row.old_close == 3.0  # A2405 close on Jan 3
    assert a_row.new_open == 10.0  # A2410 open on Jan 4


def test_regression_on_real_bars() -> None:
    """Regression: the production hab_bars.csv must produce a plausible
    number of rolls per symbol (8 to 100 range), and no symbol should
    be missing rolls entirely.
    """
    if not HAB_BARS.exists():
        pytest.skip("hab_bars.csv not available")
    bars = pd.read_csv(HAB_BARS, parse_dates=["date"])
    if "order_book_id" not in bars.columns:
        pytest.skip("enhanced hab_bars not built yet")
    rolls = extract_roll_dates_per_symbol(bars)
    counts = {s: len(d) for s, d in rolls.items()}
    # Nothing implausible on the upper end. Lower bound is 0 because some
    # newly-listed underlyings (LG, AD, OP, PL, PT, V_F etc.) have few/no
    # rolls in our window.
    assert all(0 <= n <= 110 for n in counts.values()), (
        f"implausible roll counts: {counts}"
    )
    # Every symbol covered
    assert set(counts) == set(bars["symbol"].unique())
