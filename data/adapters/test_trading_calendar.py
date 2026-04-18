from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.adapters.trading_calendar import TradingCalendar


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALENDAR_CSV = (
    REPO_ROOT / "data" / "cache" / "calendar" / "cn_futures_trading_days.csv"
)
HAB_BARS_CSV = REPO_ROOT / "data" / "cache" / "normalized" / "hab_bars.csv"


def _small_cal() -> TradingCalendar:
    """Hand-crafted calendar covering 2024-01-02..2024-03-01 (trading days only).

    Includes the 2024 Spring Festival boundary: last trading day 2024-02-08,
    first day back 2024-02-19.
    """
    return TradingCalendar(
        [
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
            date(2024, 1, 8),
            date(2024, 1, 9),
            date(2024, 2, 7),
            date(2024, 2, 8),
            # Spring Festival gap here: 2024-02-09..2024-02-18
            date(2024, 2, 19),
            date(2024, 2, 20),
            date(2024, 2, 21),
            date(2024, 2, 22),
            date(2024, 2, 23),
            date(2024, 2, 26),
            date(2024, 2, 27),
            date(2024, 2, 28),
            date(2024, 2, 29),
            date(2024, 3, 1),
        ]
    )


# --- construction ---


def test_empty_trading_days_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TradingCalendar([])


def test_non_ascending_raises() -> None:
    with pytest.raises(ValueError, match="strictly ascending"):
        TradingCalendar([date(2024, 1, 3), date(2024, 1, 2)])


def test_duplicates_raise() -> None:
    with pytest.raises(ValueError, match="strictly ascending"):
        TradingCalendar([date(2024, 1, 2), date(2024, 1, 2)])


def test_accepts_timestamps_and_datetimes() -> None:
    cal = TradingCalendar([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
    assert cal.first_day == date(2024, 1, 2)
    assert cal.last_day == date(2024, 1, 3)


# --- is_trading_day ---


def test_is_trading_day_basic() -> None:
    cal = _small_cal()
    assert cal.is_trading_day(date(2024, 1, 2)) is True
    assert cal.is_trading_day(date(2024, 2, 8)) is True
    assert cal.is_trading_day(date(2024, 2, 19)) is True


def test_is_trading_day_weekend_false() -> None:
    cal = _small_cal()
    assert cal.is_trading_day(date(2024, 1, 6)) is False  # Saturday
    assert cal.is_trading_day(date(2024, 1, 7)) is False  # Sunday


def test_is_trading_day_spring_festival_false() -> None:
    cal = _small_cal()
    for d in [
        date(2024, 2, 9),
        date(2024, 2, 12),
        date(2024, 2, 13),
        date(2024, 2, 15),
        date(2024, 2, 18),
    ]:
        assert cal.is_trading_day(d) is False, f"{d} should be non-trading"


def test_is_trading_day_out_of_range_returns_false() -> None:
    cal = _small_cal()
    assert cal.is_trading_day(date(2020, 1, 1)) is False
    assert cal.is_trading_day(date(2030, 12, 31)) is False


def test_is_trading_day_accepts_timestamp() -> None:
    cal = _small_cal()
    assert cal.is_trading_day(pd.Timestamp("2024-01-08")) is True
    assert cal.is_trading_day(pd.Timestamp("2024-01-07")) is False


# --- next / prev trading day ---


def test_next_trading_day_from_trading_day() -> None:
    cal = _small_cal()
    assert cal.next_trading_day(date(2024, 1, 2)) == date(2024, 1, 3)


def test_next_trading_day_skips_weekend() -> None:
    cal = _small_cal()
    assert cal.next_trading_day(date(2024, 1, 5)) == date(2024, 1, 8)


def test_next_trading_day_skips_spring_festival() -> None:
    cal = _small_cal()
    assert cal.next_trading_day(date(2024, 2, 8)) == date(2024, 2, 19)
    # From inside the holiday gap — still lands on 02-19.
    assert cal.next_trading_day(date(2024, 2, 14)) == date(2024, 2, 19)


def test_next_trading_day_beyond_last_raises() -> None:
    cal = _small_cal()
    with pytest.raises(ValueError, match="exceeds calendar last_day"):
        cal.next_trading_day(date(2024, 3, 1))


def test_prev_trading_day_from_trading_day() -> None:
    cal = _small_cal()
    assert cal.prev_trading_day(date(2024, 1, 3)) == date(2024, 1, 2)


def test_prev_trading_day_skips_spring_festival() -> None:
    cal = _small_cal()
    assert cal.prev_trading_day(date(2024, 2, 19)) == date(2024, 2, 8)


def test_prev_trading_day_before_first_raises() -> None:
    cal = _small_cal()
    with pytest.raises(ValueError, match="precedes calendar first_day"):
        cal.prev_trading_day(date(2024, 1, 2))


# --- trading_days_between ---


def test_trading_days_between_across_spring_festival() -> None:
    cal = _small_cal()
    days = cal.trading_days_between(date(2024, 2, 1), date(2024, 2, 22))
    assert days == [
        date(2024, 2, 7),
        date(2024, 2, 8),
        date(2024, 2, 19),
        date(2024, 2, 20),
        date(2024, 2, 21),
        date(2024, 2, 22),
    ]


def test_trading_days_between_empty_range() -> None:
    cal = _small_cal()
    assert cal.trading_days_between(date(2024, 3, 1), date(2024, 2, 1)) == []


def test_trading_days_between_endpoints_inclusive() -> None:
    cal = _small_cal()
    days = cal.trading_days_between(date(2024, 1, 2), date(2024, 1, 2))
    assert days == [date(2024, 1, 2)]


# --- validate_trading_days ---


def test_validate_empty_series_noop() -> None:
    cal = _small_cal()
    cal.validate_trading_days(pd.Series([], dtype="datetime64[ns]"))


def test_validate_all_trading_days_passes() -> None:
    cal = _small_cal()
    s = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-08", "2024-02-19"]))
    cal.validate_trading_days(s, context="unit")


def test_validate_weekend_raises() -> None:
    cal = _small_cal()
    s = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-06"]))  # Saturday
    with pytest.raises(ValueError) as exc:
        cal.validate_trading_days(s, context="symbol=TEST")
    assert "non-trading-day" in str(exc.value)
    assert "2024-01-06" in str(exc.value)
    assert "symbol=TEST" in str(exc.value)


def test_validate_spring_festival_date_raises() -> None:
    cal = _small_cal()
    s = pd.Series(pd.to_datetime(["2024-02-13"]))
    with pytest.raises(ValueError, match="2024-02-13"):
        cal.validate_trading_days(s)


def test_validate_nan_raises() -> None:
    cal = _small_cal()
    s = pd.Series([pd.Timestamp("2024-01-02"), pd.NaT])
    with pytest.raises(ValueError, match="NaN/NaT"):
        cal.validate_trading_days(s)


def test_validate_out_of_range_raises() -> None:
    cal = _small_cal()
    s = pd.Series(pd.to_datetime(["2020-01-02"]))
    with pytest.raises(ValueError, match="out of calendar range"):
        cal.validate_trading_days(s)


def test_validate_sample_caps_at_10() -> None:
    cal = _small_cal()
    bad = pd.date_range("2024-01-13", periods=15, freq="D")  # many weekends
    s = pd.Series(bad)
    with pytest.raises(ValueError) as exc:
        cal.validate_trading_days(s)
    # Error message should not include ALL bad dates — only a sample.
    msg = str(exc.value)
    assert "sample" in msg


# --- assign_trading_date (stub) ---


def test_assign_trading_date_not_implemented() -> None:
    cal = _small_cal()
    with pytest.raises(NotImplementedError, match="minute-level"):
        cal.assign_trading_date(pd.Timestamp("2024-01-02 21:30"))


# --- from_csv / default ---


@pytest.mark.skipif(
    not DEFAULT_CALENDAR_CSV.exists(),
    reason="default calendar CSV not built",
)
def test_from_csv_loads_default_calendar() -> None:
    cal = TradingCalendar.from_csv(DEFAULT_CALENDAR_CSV)
    assert len(cal) > 1000
    assert cal.first_day <= date(2018, 1, 2)
    assert cal.last_day >= date(2025, 12, 31)


@pytest.mark.skipif(
    not DEFAULT_CALENDAR_CSV.exists(),
    reason="default calendar CSV not built",
)
def test_default_factory_loads_bundled_csv() -> None:
    cal = TradingCalendar.default()
    assert cal.is_trading_day(date(2024, 2, 8)) is True
    assert cal.is_trading_day(date(2024, 2, 12)) is False  # Spring Festival
    assert cal.is_trading_day(date(2024, 10, 1)) is False  # National Day
    assert cal.is_trading_day(date(2024, 1, 6)) is False  # Saturday


# --- regression: existing hab_bars.csv must validate cleanly ---


@pytest.mark.skipif(
    not (HAB_BARS_CSV.exists() and DEFAULT_CALENDAR_CSV.exists()),
    reason="requires both hab_bars.csv and default calendar",
)
def test_hab_bars_csv_all_trading_days() -> None:
    """Regression: every date in hab_bars.csv must be a valid trading day.

    Validates the core assumption that RQData's daily bars already carry
    trading_date semantic (night-session correctly attributed to next day).
    """
    cal = TradingCalendar.default()
    bars = pd.read_csv(HAB_BARS_CSV, parse_dates=["date"], usecols=["date"])
    cal.validate_trading_days(bars["date"], context="hab_bars.csv regression")
