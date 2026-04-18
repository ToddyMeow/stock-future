"""China futures trading calendar.

Enforces the invariant: every `date` entering the backtest pipeline must be a
valid trading day (trading_date semantic). In particular, this makes the
night-session attribution assumption explicit — a row stamped "2024-02-09" is
Friday's trading day, and any night session bars opening at 21:00 on Feb 9
must already be aggregated into the next trading day (2024-02-19) by the data
source.
"""

from __future__ import annotations

import bisect
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence, Union

import pandas as pd

DateLike = Union[date, datetime, pd.Timestamp]


def _coerce_date(value: DateLike) -> date:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ValueError("NaT is not a valid date")
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"Unsupported date type: {type(value).__name__}")


class TradingCalendar:
    """Immutable trading calendar for China futures.

    Backed by a sorted list of `date` objects plus a set for O(1) membership.
    All public methods accept `date | datetime | pd.Timestamp` and coerce
    internally; NaT/NaN is rejected explicitly.
    """

    def __init__(self, trading_days: Sequence[DateLike]) -> None:
        if len(trading_days) == 0:
            raise ValueError("trading_days must not be empty")
        coerced = [_coerce_date(d) for d in trading_days]
        # Enforce: strictly ascending, unique.
        for prev, curr in zip(coerced, coerced[1:]):
            if curr <= prev:
                raise ValueError(
                    f"trading_days must be strictly ascending and unique; "
                    f"saw {prev} then {curr}"
                )
        self._days: list[date] = coerced
        self._day_set: frozenset[date] = frozenset(coerced)
        self._first: date = coerced[0]
        self._last: date = coerced[-1]

    @property
    def first_day(self) -> date:
        return self._first

    @property
    def last_day(self) -> date:
        return self._last

    def __len__(self) -> int:
        return len(self._days)

    @classmethod
    def from_csv(cls, path: Path | str) -> "TradingCalendar":
        """Load from a CSV with a single column `trading_date` (YYYY-MM-DD)."""
        path = Path(path)
        df = pd.read_csv(path, parse_dates=["trading_date"])
        return cls(df["trading_date"].tolist())

    @classmethod
    def default(cls) -> "TradingCalendar":
        """Load the bundled calendar at data/cache/calendar/cn_futures_trading_days.csv.

        Resolves the path relative to the repo root (two levels up from this
        file), so it works regardless of CWD.
        """
        repo_root = Path(__file__).resolve().parents[2]
        default_path = repo_root / "data" / "cache" / "calendar" / "cn_futures_trading_days.csv"
        if not default_path.exists():
            raise FileNotFoundError(
                f"Default trading calendar not found at {default_path}. "
                f"Run scripts/build_trading_calendar.py to generate it."
            )
        return cls.from_csv(default_path)

    @classmethod
    def from_rqdata(cls, start: str, end: str) -> "TradingCalendar":
        """Fetch trading days from RQData. Requires rqdatac + credentials.

        Credentials are read from env vars RQDATAC_USER / RQDATAC_PASSWORD.
        Only intended for scripts/build_trading_calendar.py — the regular
        backtest path loads from the cached CSV via `default()`.
        """
        import os
        import rqdatac

        user = os.getenv("RQDATAC_USER")
        password = os.getenv("RQDATAC_PASSWORD")
        if not user or not password:
            raise RuntimeError(
                "Missing RQData credentials. Set RQDATAC_USER and "
                "RQDATAC_PASSWORD (e.g., via the repo .env file)."
            )
        rqdatac.init(user, password)
        days = rqdatac.get_trading_dates(start, end, market="cn")
        if len(days) == 0:
            raise RuntimeError(f"RQData returned no trading days for {start}..{end}")
        return cls(list(days))

    def is_trading_day(self, d: DateLike) -> bool:
        """True iff d is within [first_day, last_day] and is a trading day.

        Out-of-range dates return False (not raise) — use this as a pure
        predicate. For strict validation use validate_trading_days().
        """
        try:
            coerced = _coerce_date(d)
        except (TypeError, ValueError):
            return False
        return coerced in self._day_set

    def _require_in_range(self, d: date) -> None:
        if d < self._first or d > self._last:
            raise ValueError(
                f"date {d} is out of calendar range "
                f"[{self._first}, {self._last}]"
            )

    def next_trading_day(self, d: DateLike) -> date:
        """Strictly next trading day after d.

        If d is a trading day, returns the following trading day. If d falls
        on a weekend/holiday, returns the first trading day after d.
        Raises if the result would fall beyond the calendar's last_day.
        """
        coerced = _coerce_date(d)
        # bisect_right finds the insertion point past any equal element,
        # so self._days[idx] is strictly > d.
        idx = bisect.bisect_right(self._days, coerced)
        if idx >= len(self._days):
            raise ValueError(
                f"next_trading_day({coerced}) exceeds calendar last_day "
                f"{self._last}"
            )
        return self._days[idx]

    def prev_trading_day(self, d: DateLike) -> date:
        """Strictly previous trading day before d."""
        coerced = _coerce_date(d)
        idx = bisect.bisect_left(self._days, coerced)
        if idx == 0:
            raise ValueError(
                f"prev_trading_day({coerced}) precedes calendar first_day "
                f"{self._first}"
            )
        return self._days[idx - 1]

    def trading_days_between(self, start: DateLike, end: DateLike) -> list[date]:
        """Trading days in the closed interval [start, end]."""
        s = _coerce_date(start)
        e = _coerce_date(end)
        if e < s:
            return []
        lo = bisect.bisect_left(self._days, s)
        hi = bisect.bisect_right(self._days, e)
        return self._days[lo:hi]

    def validate_trading_days(
        self,
        dates: pd.Series | Iterable[DateLike],
        *,
        context: str = "",
    ) -> None:
        """Raise ValueError if any date is not a trading day.

        Empty input is a no-op. NaN/NaT raises with a dedicated message.
        For non-trading-day violations, the error message lists up to 10
        offending values plus the count. `context` is appended to the
        message to aid debugging (e.g., "symbol=RB").
        """
        if isinstance(dates, pd.Series):
            if dates.empty:
                return
            if dates.isna().any():
                nan_count = int(dates.isna().sum())
                raise ValueError(
                    self._build_err(
                        f"{nan_count} NaN/NaT value(s) in dates", context
                    )
                )
            iterable = dates.tolist()
        else:
            iterable = list(dates)
            if not iterable:
                return
            # Manual NaN/NaT check for non-Series input.
            nan_samples = [v for v in iterable if _is_nan(v)]
            if nan_samples:
                raise ValueError(
                    self._build_err(
                        f"{len(nan_samples)} NaN/NaT value(s) in dates",
                        context,
                    )
                )

        violations: list[date] = []
        out_of_range: list[date] = []
        for raw in iterable:
            d = _coerce_date(raw)
            if d < self._first or d > self._last:
                out_of_range.append(d)
            elif d not in self._day_set:
                violations.append(d)

        if out_of_range:
            sample = [d.isoformat() for d in sorted(set(out_of_range))[:10]]
            raise ValueError(
                self._build_err(
                    f"{len(out_of_range)} date(s) out of calendar range "
                    f"[{self._first.isoformat()}, {self._last.isoformat()}]; "
                    f"sample: {sample}",
                    context,
                )
            )
        if violations:
            sample = [d.isoformat() for d in sorted(set(violations))[:10]]
            raise ValueError(
                self._build_err(
                    f"{len(violations)} non-trading-day value(s); "
                    f"sample: {sample}",
                    context,
                )
            )

    def assign_trading_date(self, ts: pd.Timestamp) -> date:
        """Map a minute-level timestamp to its owning trading_date.

        Rule: ts.hour >= 20 → next_trading_day(ts.date()); else ts.date()
        (which must itself be a trading day).

        Not implemented in this iteration — daily bars from RQData are
        already pre-aggregated with correct trading_date semantic, so this
        function has no caller. Ship the stub now so the signature is
        locked; implement when minute-level backtesting lands.
        """
        raise NotImplementedError(
            "assign_trading_date is reserved for minute-level backtesting; "
            "daily bars should already carry trading_date semantic at ingestion."
        )

    @staticmethod
    def _build_err(body: str, context: str) -> str:
        if context:
            return f"{body} [context: {context}]"
        return body


def _is_nan(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    if isinstance(value, pd.Timestamp) and pd.isna(value):
        return True
    return False
