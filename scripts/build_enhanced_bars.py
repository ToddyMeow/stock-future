"""Merge Panama-adjusted bars with raw dominant prices + contract codes.

Enhances `data/cache/normalized/hab_bars.csv` with columns required for
dual-stream P&L:
  - open_raw, high_raw, low_raw, close_raw, settle_raw  (from dominant_none/*.csv)
  - order_book_id                                        (from dominant_contracts/*.csv)

Panama columns (`open, high, low, close, settle`) are preserved unchanged.
Existing callers that only read the Panama columns continue to work.
`settle_raw` is optional — falls back to close_raw if settlement wasn't
pulled for this symbol.

Run order:
  1. scripts/download_rqdata_futures.py      -> produces dominant_none + dominant_pre
  2. scripts/download_dominant_contracts.py  -> produces dominant_contracts
  3. scripts/build_enhanced_bars.py          -> THIS — merges everything
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.adapters.ohlc_repair import repair_ohlc_envelope
from data.adapters.trading_calendar import TradingCalendar


HAB_BARS_CSV = REPO_ROOT / "data" / "cache" / "normalized" / "hab_bars.csv"
DOMINANT_NONE_DIR = REPO_ROOT / "data" / "cache" / "raw_rqdata" / "dominant_none"
DOMINANT_CONTRACTS_DIR = REPO_ROOT / "data" / "cache" / "dominant_contracts"
LIMIT_PRICES_DIR = REPO_ROOT / "data" / "cache" / "limit_prices"

RAW_OHLC_COLUMNS = ["open_raw", "high_raw", "low_raw", "close_raw"]
RAW_SETTLE_COLUMN = "settle_raw"
CONTRACT_COLUMN = "order_book_id"
LIMIT_COLUMNS = ["limit_up", "limit_down"]


def load_raw_dominant(symbol: str) -> pd.DataFrame:
    path = DOMINANT_NONE_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dominant_none file: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    cols = ["date", "open", "high", "low", "close"]
    rename = {
        "open": "open_raw", "high": "high_raw",
        "low": "low_raw", "close": "close_raw",
    }
    if "settlement" in df.columns:
        cols.append("settlement")
        rename["settlement"] = "settle_raw"
    elif "settle" in df.columns:
        cols.append("settle")
        rename["settle"] = "settle_raw"
    out = df[cols].copy().rename(columns=rename)
    if "settle_raw" not in out.columns:
        out["settle_raw"] = out["close_raw"]
    return out


def load_contract_codes(symbol: str) -> pd.DataFrame:
    path = DOMINANT_CONTRACTS_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dominant contract file: {path}. "
            f"Run scripts/download_dominant_contracts.py"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df[["date", "order_book_id"]]


def load_limit_prices(symbol: str) -> pd.DataFrame:
    path = LIMIT_PRICES_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "limit_up", "limit_down"])
    df = pd.read_csv(path, parse_dates=["date"])
    return df[["date", "limit_up", "limit_down"]]


def main() -> int:
    if not HAB_BARS_CSV.exists():
        print(f"ERROR: {HAB_BARS_CSV} does not exist", file=sys.stderr)
        return 1

    bars = pd.read_csv(HAB_BARS_CSV, parse_dates=["date"])
    print(f"Loaded {HAB_BARS_CSV}: {len(bars)} rows, {bars['symbol'].nunique()} symbols")

    # Drop any previously-added columns (idempotent re-run).
    for col in RAW_OHLC_COLUMNS + [RAW_SETTLE_COLUMN, CONTRACT_COLUMN] + LIMIT_COLUMNS:
        if col in bars.columns:
            bars = bars.drop(columns=[col])

    enriched_frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for symbol, sym_bars in bars.groupby("symbol", sort=False):
        try:
            raw = load_raw_dominant(symbol)
            codes = load_contract_codes(symbol)
        except FileNotFoundError as e:
            missing.append(f"{symbol}: {e}")
            continue

        merged = sym_bars.merge(raw, on="date", how="left")
        merged = merged.merge(codes, on="date", how="left")
        limits = load_limit_prices(symbol)
        if not limits.empty:
            merged = merged.merge(limits, on="date", how="left")
        else:
            merged["limit_up"] = pd.NA
            merged["limit_down"] = pd.NA
        enriched_frames.append(merged)

    if missing:
        print("WARNING: missing source data for some symbols:")
        for m in missing:
            print(f"  {m}")
        # Don't silently drop; raise so user re-runs upstream downloads.
        if len(missing) == bars["symbol"].nunique():
            print("ERROR: every symbol missing source data", file=sys.stderr)
            return 2

    enriched = pd.concat(enriched_frames, axis=0, ignore_index=True)
    enriched = enriched.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Sanity: every row must have raw OHLC and contract code.
    for col in RAW_OHLC_COLUMNS + [RAW_SETTLE_COLUMN, CONTRACT_COLUMN]:
        na_count = enriched[col].isna().sum()
        if na_count > 0:
            sample = enriched[enriched[col].isna()][["date", "symbol"]].head(5)
            print(
                f"ERROR: {na_count} rows missing '{col}'. Sample:\n{sample.to_string(index=False)}",
                file=sys.stderr,
            )
            return 3

    # OHLC repair (1.4 method B): envelope [low, high] around open/close/settle
    # on both Panama and raw price streams. Idempotent on clean data.
    panama_changes = repair_ohlc_envelope(
        enriched, high_col="high", low_col="low",
        enveloped_cols=["open", "close", "settle"],
    )
    raw_changes = repair_ohlc_envelope(
        enriched, high_col="high_raw", low_col="low_raw",
        enveloped_cols=["open_raw", "close_raw", "settle_raw"],
    )
    if panama_changes or raw_changes:
        print(f"OHLC repair (method B): panama={panama_changes} rows, raw={raw_changes} rows")

    # Calendar invariant (inherited from 1.1).
    cal = TradingCalendar.default()
    cal.validate_trading_days(enriched["date"], context="build_enhanced_bars")

    enriched.to_csv(HAB_BARS_CSV, index=False)
    print(f"Wrote {len(enriched)} rows to {HAB_BARS_CSV}")
    print(f"New columns: {RAW_OHLC_COLUMNS + [RAW_SETTLE_COLUMN, CONTRACT_COLUMN]}")
    # Summary: count of rolls per symbol
    rolls_per_symbol = enriched.groupby("symbol", sort=False).apply(
        lambda g: g["order_book_id"].ne(g["order_book_id"].shift()).sum() - 1
    )
    print(f"Rolls per symbol — min: {rolls_per_symbol.min()}, "
          f"max: {rolls_per_symbol.max()}, median: {int(rolls_per_symbol.median())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
