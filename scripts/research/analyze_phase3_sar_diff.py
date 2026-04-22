"""Phase 3 SAR diff — side-by-side compare of non-SAR baseline vs SAR variant.

Inputs (both produced by run_phase3_combo_selection.py under same risk config):
  data/runs/phase3/combo_grid_<baseline_tag>.csv
  data/runs/phase3/combo_grid_<sar_tag>.csv
  data/runs/phase3/trades_<sar_tag>.csv        — to compute SAR trigger stats

Output:
  data/runs/phase3/sar_diff_<sar_tag>.csv
    per (group, entry, exit): baseline vs SAR metrics + deltas,
    SAR trigger count/pct, average reverse chain depth.

Usage:
  python scripts/research/analyze_phase3_sar_diff.py \\
      --baseline-tag risk3cap8_baseline --sar-tag risk3cap8_sar
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PHASE3_DIR = ROOT / "data" / "runs" / "phase3"


def compute_sar_attribution(sar_trades_path: Path) -> pd.DataFrame:
    """Aggregate SAR trigger stats per (group, entry, exit) from trades CSV."""
    if not sar_trades_path.exists():
        print(f"WARNING: {sar_trades_path} not found — SAR attribution will be empty")
        return pd.DataFrame(columns=["group", "entry", "exit",
                                     "sar_trigger_count", "sar_trigger_pct",
                                     "avg_chain_depth", "max_chain_depth"])

    t = pd.read_csv(sar_trades_path)
    # entry_type comes through metadata spill; NaN for regular signal entries
    t["_is_sar"] = (t.get("entry_type", pd.Series(dtype=object)) == "SAR_REVERSE")
    t["_chain"] = pd.to_numeric(t.get("reverse_leg_count"), errors="coerce")

    g = t.groupby(["group", "entry_id", "exit_id"]).agg(
        total_trades=("_is_sar", "size"),
        sar_trigger_count=("_is_sar", "sum"),
        avg_chain_depth=("_chain", "mean"),
        max_chain_depth=("_chain", "max"),
    ).reset_index().rename(columns={"entry_id": "entry", "exit_id": "exit"})
    g["sar_trigger_pct"] = g["sar_trigger_count"] / g["total_trades"].clip(lower=1)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-tag", required=True,
                    help="Non-SAR baseline tag (e.g. 'risk3cap8_baseline')")
    ap.add_argument("--sar-tag", required=True,
                    help="SAR variant tag (e.g. 'risk3cap8_sar')")
    args = ap.parse_args()

    baseline_path = PHASE3_DIR / f"combo_grid_{args.baseline_tag}.csv"
    sar_path = PHASE3_DIR / f"combo_grid_{args.sar_tag}.csv"
    trades_sar_path = PHASE3_DIR / f"trades_{args.sar_tag}.csv"

    for p in (baseline_path, sar_path):
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    base = pd.read_csv(baseline_path)
    sar = pd.read_csv(sar_path)
    print(f"Loaded baseline: {len(base)} rows from {baseline_path.name}")
    print(f"Loaded SAR    : {len(sar)} rows from {sar_path.name}")

    # Merge on combo key
    merged = base.merge(
        sar, on=["group", "entry", "exit"], suffixes=("_base", "_sar"), how="outer",
        indicator=True,
    )
    n_both = (merged["_merge"] == "both").sum()
    print(f"Joined        : {n_both}/{len(merged)} rows present in both tags")
    if (merged["_merge"] != "both").any():
        missing = merged[merged["_merge"] != "both"][["group", "entry", "exit", "_merge"]]
        print("Missing rows (likely incomplete run):")
        print(missing.head(20).to_string(index=False))
    merged = merged[merged["_merge"] == "both"].drop(columns=["_merge"])

    # Deltas
    for col in ["trades", "net_pnl", "expectancy", "sharpe", "cagr",
                "total_return", "max_dd_pct", "profit_factor",
                "profitable_years", "win_rate"]:
        b_col, s_col = f"{col}_base", f"{col}_sar"
        if b_col in merged.columns and s_col in merged.columns:
            merged[f"delta_{col}"] = merged[s_col] - merged[b_col]

    # SAR attribution from trades
    attr = compute_sar_attribution(trades_sar_path)
    merged = merged.merge(
        attr[["group", "entry", "exit", "sar_trigger_count",
              "sar_trigger_pct", "avg_chain_depth", "max_chain_depth"]],
        on=["group", "entry", "exit"], how="left",
    )

    # Order output columns
    col_order = [
        "group", "entry", "exit",
        "sar_trigger_count", "sar_trigger_pct",
        "avg_chain_depth", "max_chain_depth",
        "trades_base", "trades_sar", "delta_trades",
        "expectancy_base", "expectancy_sar", "delta_expectancy",
        "sharpe_base", "sharpe_sar", "delta_sharpe",
        "cagr_base", "cagr_sar", "delta_cagr",
        "total_return_base", "total_return_sar", "delta_total_return",
        "max_dd_pct_base", "max_dd_pct_sar", "delta_max_dd_pct",
        "profit_factor_base", "profit_factor_sar",
        "profitable_years_base", "profitable_years_sar",
        "win_rate_base", "win_rate_sar",
        "net_pnl_base", "net_pnl_sar", "delta_net_pnl",
    ]
    out = merged[[c for c in col_order if c in merged.columns]]
    out = out.sort_values("delta_expectancy", ascending=False).reset_index(drop=True)

    out_path = PHASE3_DIR / f"sar_diff_{args.sar_tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(out)} rows)")

    # Print top / bottom winners
    print("\n===== Top 10 combos where SAR helped most (delta_expectancy ↑) =====")
    show_cols = ["group", "entry", "exit",
                 "expectancy_base", "expectancy_sar", "delta_expectancy",
                 "sharpe_base", "sharpe_sar", "sar_trigger_pct",
                 "avg_chain_depth"]
    print(out.head(10)[[c for c in show_cols if c in out.columns]].to_string(index=False))

    print("\n===== Bottom 10 combos where SAR hurt most =====")
    print(out.tail(10)[[c for c in show_cols if c in out.columns]].to_string(index=False))

    # Summary
    n_helped = (out["delta_expectancy"] > 0).sum()
    n_hurt = (out["delta_expectancy"] < 0).sum()
    print("\n===== Summary =====")
    print(f"  Total combos     : {len(out)}")
    print(f"  SAR helped       : {n_helped} ({100*n_helped/len(out):.1f}%)")
    print(f"  SAR hurt         : {n_hurt} ({100*n_hurt/len(out):.1f}%)")
    print(f"  Mean delta_exp   : {out['delta_expectancy'].mean():.1f}")
    print(f"  Median delta_exp : {out['delta_expectancy'].median():.1f}")
    print(f"  Mean delta_sharpe: {out['delta_sharpe'].mean():.3f}")


if __name__ == "__main__":
    main()
