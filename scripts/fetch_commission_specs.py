"""Fetch commission specs per symbol from RQData.

Writes `data/cache/commission_specs.json` with one entry per underlying:
  {"RB": {"type": "by_money", "rate": 0.000110},
   "AU": {"type": "by_volume", "rate": 10.0}, ...}

Uses one representative dominant contract per symbol (e.g. RB2410) to probe
`rqfutures.get_commission_margin`. Symbol-level specs are stable across
contracts within the same underlying.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_rqdata_futures import load_env_file


DOMINANT_CONTRACTS_DIR = REPO_ROOT / "data" / "cache" / "dominant_contracts"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "cache" / "commission_specs.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--symbols", nargs="*", help="Default: all symbols in dominant_contracts/")
    return p.parse_args()


def discover_symbols() -> List[str]:
    return sorted(p.stem for p in DOMINANT_CONTRACTS_DIR.glob("*.csv"))


def latest_contract_for_symbol(symbol: str) -> str:
    """Return the last order_book_id observed in the dominant series."""
    path = DOMINANT_CONTRACTS_DIR / f"{symbol}.csv"
    df = pd.read_csv(path)
    return str(df["order_book_id"].iloc[-1])


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
    warnings.filterwarnings("ignore")

    symbols = args.symbols or discover_symbols()
    specs: Dict[str, Dict[str, object]] = {}
    failed: List[str] = []
    for sym in symbols:
        try:
            contract = latest_contract_for_symbol(sym)
            df = rqfutures.get_commission_margin(contract)
            if df is None or df.empty:
                failed.append(f"{sym}: empty")
                continue
            row = df.iloc[0]
            # Use open_commission_ratio as the base rate; close_commission_ratio
            # is typically identical (平今 / close_today is a separate concern
            # handled by 5.1 if/when that lands).
            specs[sym] = {
                "type": str(row["commission_type"]),
                "rate": float(row["open_commission_ratio"]),
                "probe_contract": contract,
            }
        except Exception as e:  # noqa: BLE001
            failed.append(f"{sym}: {type(e).__name__}: {e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(specs, indent=2, sort_keys=True))
    print(f"Wrote {len(specs)} symbol specs to {args.output}")
    if failed:
        print("Failures:")
        for f in failed:
            print(f"  {f}")
    # Summary
    by_type: Dict[str, int] = {}
    for s in specs.values():
        by_type[str(s["type"])] = by_type.get(str(s["type"]), 0) + 1
    print(f"Type breakdown: {by_type}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
