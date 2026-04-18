"""Fetch dominant contract codes per (symbol, date) from RQData.

Output: `data/cache/dominant_contracts/<SYMBOL>.csv` with columns
`date, order_book_id`. One file per underlying.

The data is required for the dual-stream P&L pipeline:
- Panama-adjusted series (data/cache/normalized/hab_bars.csv) drives indicators
- Raw dominant prices (data/cache/raw_rqdata/dominant_none/) drive fills
- This script adds the missing piece: which specific contract each day's
  dominant price actually belongs to. That enables precise roll detection
  and per-segment P&L accounting across roll days.

Uses the same 'dominant' rule as rqfutures.get_dominant_price() so the
output aligns with the existing dominant_none CSVs on disk.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_rqdata_futures import load_env_file


DEFAULT_START = "2015-01-01"
DEFAULT_END = "2030-12-31"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cache" / "dominant_contracts"
DOMINANT_NONE_DIR = REPO_ROOT / "data" / "cache" / "raw_rqdata" / "dominant_none"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output dir (default {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--symbols",
        nargs="*",
        help="Explicit symbol list. Default: infer from dominant_none dir.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def discover_symbols(dominant_none_dir: Path) -> List[str]:
    if not dominant_none_dir.exists():
        raise FileNotFoundError(
            f"dominant_none dir missing: {dominant_none_dir}. "
            f"Run scripts/download_rqdata_futures.py first."
        )
    return sorted(p.stem for p in dominant_none_dir.glob("*.csv"))


def fetch_one(
    rqfutures,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    series = rqfutures.get_dominant(symbol, start_date=start, end_date=end)
    if series is None or len(series) == 0:
        return pd.DataFrame(columns=["date", "order_book_id"])
    df = series.reset_index()
    df.columns = ["date", "order_book_id"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> int:
    args = parse_args()
    load_env_file()
    import rqdatac
    from rqdatac import futures as rqfutures

    user = os.getenv("RQDATAC_USER")
    password = os.getenv("RQDATAC_PASSWORD")
    if not user or not password:
        print("ERROR: RQDATAC_USER / RQDATAC_PASSWORD not set.", file=sys.stderr)
        return 2
    rqdatac.init(user, password)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    symbols = args.symbols or discover_symbols(DOMINANT_NONE_DIR)
    print(f"Fetching dominant contracts for {len(symbols)} symbols: {args.start} to {args.end}")

    skipped: List[str] = []
    written: List[str] = []
    failed: List[str] = []

    for sym in symbols:
        out_path = args.output_dir / f"{sym}.csv"
        if out_path.exists() and not args.overwrite:
            skipped.append(sym)
            continue
        try:
            df = fetch_one(rqfutures, sym, args.start, args.end)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{sym}: {type(e).__name__}: {e}")
            continue
        if df.empty:
            failed.append(f"{sym}: empty result")
            continue
        df.to_csv(out_path, index=False)
        written.append(sym)

    print(f"Written: {len(written)}; Skipped (exists): {len(skipped)}; Failed: {len(failed)}")
    if failed:
        print("Failures:")
        for f in failed:
            print(f"  {f}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
