"""Fetch daily limit_up / limit_down per dominant series from RQData.

Sidecar script — does NOT replace the main OHLCV download. Writes one CSV
per underlying to `data/cache/limit_prices/<SYMBOL>.csv` with columns
`date, limit_up, limit_down`.

`scripts/build_enhanced_bars.py` merges these into `hab_bars.csv` as
`limit_up, limit_down` columns used by the engine for:
  - execution-layer lock detection (skip exits on opposing-locked bars)
  - sizing-layer worst-case floor (see EngineConfig.max_limit_days)
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


DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cache" / "limit_prices"
DOMINANT_NONE_DIR = REPO_ROOT / "data" / "cache" / "raw_rqdata" / "dominant_none"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--symbols", nargs="*", help="Default: all symbols in dominant_none/")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def discover_symbols() -> List[str]:
    return sorted(p.stem for p in DOMINANT_NONE_DIR.glob("*.csv"))


def fetch_one(rqfutures, symbol: str, start: str, end: str) -> pd.DataFrame:
    df = rqfutures.get_dominant_price(
        underlying_symbols=symbol,
        start_date=start, end_date=end,
        frequency="1d",
        fields=["limit_up", "limit_down"],
        adjust_type="none",
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "limit_up", "limit_down"])
    # MultiIndex (underlying_symbol, date) → flatten to (date, limit_up, limit_down).
    df = df.reset_index()
    cols = ["date", "limit_up", "limit_down"]
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[1]: "date"})
    df = df[cols]
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
    symbols = args.symbols or discover_symbols()
    print(f"Fetching limit prices for {len(symbols)} symbols: {args.start} to {args.end}")

    written: List[str] = []
    skipped: List[str] = []
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
    for f in failed:
        print(f"  {f}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
