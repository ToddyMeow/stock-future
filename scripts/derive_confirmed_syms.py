#!/usr/bin/env python
"""
Helper: derive confirmed_syms JSON from Phase 0 tradeable universe ∩ Phase 2 tradeability scoring.

Inputs:
  --phase0-json   Phase 0 tradeable_symbols.json (per capital tier)
  --phase2-csv    Phase 2 per_symbol.csv with columns [symbol, group, tradeability_score]
  --min-score     Phase 2 score threshold (default 50)
  --output-json   Output JSON with keys 'symbols' and 'groups'

Output JSON schema:
  { "symbols": ["AO", "AP", ...], "groups": ["building", "ind_AP", ...] }

Designed for feeding into scripts/run_phase3_combo_selection.py --confirmed-syms-json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase0-json", required=True, help="Phase 0 tradeable_symbols.json")
    ap.add_argument("--phase2-csv", required=True,
                    help="Phase 2 phase2_per_symbol.csv (or phase2_cross_prob.csv)")
    ap.add_argument("--min-score", type=int, default=50,
                    help="Minimum Phase 2 tradeability_score to include (default 50)")
    ap.add_argument("--output-json", required=True, help="Output confirmed_syms JSON path")
    args = ap.parse_args()

    with open(args.phase0_json) as f:
        p0 = json.load(f)
    if isinstance(p0, dict) and "tradeable_symbols" in p0:
        tradeable = set(p0["tradeable_symbols"])
    elif isinstance(p0, list):
        tradeable = set(p0)
    else:
        raise ValueError(f"Unexpected Phase 0 JSON shape: {type(p0)}")

    p2 = pd.read_csv(args.phase2_csv)
    score_col = "tradeability_score" if "tradeability_score" in p2.columns else "score"
    passed = p2[p2[score_col] >= args.min_score]

    # Intersection: must be in both Phase 0 tradeable and Phase 2 score-passed
    final = passed[passed["symbol"].isin(tradeable)].copy()
    symbols = sorted(final["symbol"].unique().tolist())
    groups = sorted(final["group"].unique().tolist())

    out = {"symbols": symbols, "groups": groups,
           "source_phase0": str(args.phase0_json),
           "source_phase2": str(args.phase2_csv),
           "min_score": args.min_score,
           "n_tradeable_phase0": len(tradeable),
           "n_passed_phase2": len(passed),
           "n_final": len(symbols)}

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Phase 0 tradeable: {len(tradeable)} symbols")
    print(f"Phase 2 passed (score>={args.min_score}): {len(passed)} symbols")
    print(f"Intersection → confirmed: {len(symbols)} symbols / {len(groups)} groups")
    print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
