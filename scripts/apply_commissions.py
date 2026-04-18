"""Recompute the `commission` column in hab_bars.csv using commission_specs.json.

One-shot patch for existing data. Future adapter runs already emit the
right values (see rqdata_futures_adatpter.py + FuturesMeta changes).

Formula:
  by_volume → commission = rate (yuan/lot, constant per row)
  by_money  → commission = rate × close × multiplier (per-row, scales with price)

Run after `scripts/fetch_commission_specs.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
HAB_BARS = REPO_ROOT / "data" / "cache" / "normalized" / "hab_bars.csv"
SPECS_PATH = REPO_ROOT / "data" / "cache" / "commission_specs.json"


def main() -> int:
    if not HAB_BARS.exists() or not SPECS_PATH.exists():
        print(f"ERROR: missing {HAB_BARS} or {SPECS_PATH}", file=sys.stderr)
        return 2

    specs = json.loads(SPECS_PATH.read_text())
    bars = pd.read_csv(HAB_BARS, parse_dates=["date"])
    old_median_by_symbol = bars.groupby("symbol")["commission"].median()

    unmatched = []
    for sym, sym_bars in bars.groupby("symbol", sort=False):
        spec = specs.get(sym)
        if spec is None:
            unmatched.append(sym)
            continue
        rate = float(spec["rate"])
        if spec["type"] == "by_money":
            # Use close_raw (actual contract price, always positive) rather
            # than Panama close which can drift negative on deep-offset symbols
            # (I, P, LU, EC). Fall back to |close| if close_raw absent.
            price_col = "close_raw" if "close_raw" in sym_bars.columns else "close"
            price = sym_bars[price_col].abs()
            bars.loc[sym_bars.index, "commission"] = (
                rate * price * sym_bars["contract_multiplier"]
            )
        else:  # by_volume
            bars.loc[sym_bars.index, "commission"] = rate

    if unmatched:
        print(f"WARNING: no spec for {len(unmatched)} symbols, left untouched: {unmatched[:10]}...")

    bars.to_csv(HAB_BARS, index=False)
    new_median_by_symbol = bars.groupby("symbol")["commission"].median()

    print(f"Updated commission for {bars['symbol'].nunique() - len(unmatched)} symbols.")
    print("Top 10 biggest shifts (median commission before vs after):")
    diff = (new_median_by_symbol - old_median_by_symbol).abs().sort_values(ascending=False).head(10)
    for sym in diff.index:
        print(f"  {sym}: {old_median_by_symbol[sym]:.4f} → {new_median_by_symbol[sym]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
