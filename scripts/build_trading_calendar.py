"""Fetch China futures trading days from RQData and write the cached CSV.

Run this once (and whenever the calendar needs extending) to regenerate
`data/cache/calendar/cn_futures_trading_days.csv`. The regular backtest
path does NOT call RQData — it loads the bundled CSV via
`TradingCalendar.default()`.

Usage:
    python scripts/build_trading_calendar.py            # 2015-2030 default
    python scripts/build_trading_calendar.py --start 2015-01-01 --end 2030-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.adapters.trading_calendar import TradingCalendar
from scripts.download_rqdata_futures import load_env_file


DEFAULT_START = "2015-01-01"
DEFAULT_END = "2030-12-31"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "cache" / "calendar" / "cn_futures_trading_days.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="YYYY-MM-DD")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file()
    print(f"Fetching trading days from RQData: {args.start} to {args.end}")
    try:
        cal = TradingCalendar.from_rqdata(args.start, args.end)
    except ImportError:
        print(
            "ERROR: rqdatac is not installed. Install it and set credentials "
            "before running this script.",
            file=sys.stderr,
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"trading_date": [d.isoformat() for d in _iter_days(cal)]})
    df.to_csv(args.output, index=False)

    print(f"Wrote {len(df)} trading days to {args.output}")
    print(f"Range: {cal.first_day.isoformat()} to {cal.last_day.isoformat()}")

    # Spot-check around known holidays for manual sanity.
    _print_holiday_boundary(cal, "Spring Festival 2024", "2024-02-07", "2024-02-22")
    _print_holiday_boundary(cal, "National Day 2024", "2024-09-27", "2024-10-10")
    return 0


def _iter_days(cal: TradingCalendar):
    for d in cal.trading_days_between(cal.first_day, cal.last_day):
        yield d


def _print_holiday_boundary(cal: TradingCalendar, label: str, start: str, end: str) -> None:
    try:
        days = cal.trading_days_between(pd.Timestamp(start), pd.Timestamp(end))
    except ValueError:
        return
    if days:
        print(f"  {label}: {[d.isoformat() for d in days]}")


if __name__ == "__main__":
    sys.exit(main())
