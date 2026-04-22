"""Phase 3 analyzer — composite ranking + per-group best combo.

For each group in combo_grid_<tag>.csv:
  1. Excess expectancy  = combo.expectancy − Phase1_A_baseline[group]
     (Phase 1 A baseline = avg of A_mean across 3 exit_probs per group)
  2. Profitable count   = # of 30 combos with excess > 0
  3. Filter rule: profitable_count / 30 < 0.4  →  skip whole group
  4. Composite rank (lower = better), sum of independent ranks on:
     - excess_expectancy (desc)
     - profit_factor (desc)
     - max_dd_pct (asc, less negative = better)
     - profitable_years (desc)
  5. Best combo per group = lowest composite rank

Outputs:
  data/runs/phase3/best_combos_<tag>.csv
  data/runs/phase3/A_baseline.csv  (one-time write)

Usage:
  python scripts/research/analyze_phase3.py --tag risk3cap6
  python scripts/research/analyze_phase3.py --tag risk3cap8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PHASE3_DIR = ROOT / "data" / "runs" / "phase3"
ABENCH = ROOT / "data" / "runs" / "alpha_benchmark"


def compute_phase1_baseline() -> dict:
    """Average of A_mean across 3 exit_prob variants, per group."""
    baselines = []
    for prob in ["005", "007", "010"]:
        p = ABENCH / f"phase1_labels_ep{prob}.csv"
        if not p.exists():
            print(f"WARNING: {p} missing; skipping")
            continue
        df = pd.read_csv(p)
        baselines.append(df[["group", "A_mean"]].rename(columns={"A_mean": f"A_mean_ep{prob}"}))

    if not baselines:
        print("ERROR: no Phase 1 baseline data found")
        sys.exit(1)

    merged = baselines[0]
    for b in baselines[1:]:
        merged = merged.merge(b, on="group", how="outer")
    cols_mean = [c for c in merged.columns if c.startswith("A_mean")]
    merged["A_baseline"] = merged[cols_mean].mean(axis=1)
    merged.to_csv(PHASE3_DIR / "A_baseline.csv", index=False)
    return dict(zip(merged["group"], merged["A_baseline"]))


def rank_combos(group_df: pd.DataFrame, baseline: float) -> pd.DataFrame:
    g = group_df.copy()
    g["excess_expectancy"] = g["expectancy"] - baseline
    # Handle inf/NaN in profit_factor
    g["profit_factor_clipped"] = g["profit_factor"].clip(upper=20.0)

    g["rank_excess"] = g["excess_expectancy"].rank(ascending=False, method="min")
    g["rank_pf"] = g["profit_factor_clipped"].rank(ascending=False, method="min")
    g["rank_dd"] = g["max_dd_pct"].rank(ascending=False, method="min")  # less negative → higher rank value
    g["rank_profy"] = g["profitable_years"].rank(ascending=False, method="min")

    g["composite_rank"] = (
        g["rank_excess"] + g["rank_pf"] + g["rank_dd"] + g["rank_profy"]
    )
    return g.sort_values("composite_rank").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="Grid tag, e.g. 'risk3cap6'")
    ap.add_argument("--filter-ratio", type=float, default=0.4,
                    help="Skip group if profitable_count/n_combos < ratio (default 0.4)")
    args = ap.parse_args()

    grid_path = PHASE3_DIR / f"combo_grid_{args.tag}.csv"
    if not grid_path.exists():
        print(f"ERROR: {grid_path} not found")
        sys.exit(1)
    grid = pd.read_csv(grid_path)
    print(f"Loaded {len(grid)} rows ({grid['group'].nunique()} groups × "
          f"{grid['entry'].nunique()} entries × {grid['exit'].nunique()} exits)")

    baseline_map = compute_phase1_baseline()
    print(f"\nPhase 1 A baseline (avg across 3 exit_probs), per group:")
    for g, v in sorted(baseline_map.items()):
        print(f"  {g:15s} {v:+.1f}")

    all_rows = []
    best_rows = []

    n_combos = grid["entry"].nunique() * grid["exit"].nunique()
    filter_threshold = int(np.ceil(args.filter_ratio * n_combos))
    print(f"\nFilter threshold: need ≥ {filter_threshold}/{n_combos} combos profitable to keep group\n")

    for group, gdf in grid.groupby("group"):
        if group not in baseline_map:
            print(f"  [{group}] no baseline, skipping")
            continue
        baseline = baseline_map[group]
        ranked = rank_combos(gdf, baseline)
        ranked["group"] = group
        ranked["A_baseline"] = baseline

        profitable_count = int((ranked["excess_expectancy"] > 0).sum())
        filter_status = "kept" if profitable_count >= filter_threshold else "filtered_out"

        all_rows.extend(ranked.to_dict("records"))

        if filter_status == "kept":
            best = ranked.iloc[0]
            best_rows.append({
                "group": group,
                "best_combo": f"{best['entry']}+{best['exit']}",
                "trades": int(best["trades"]),
                "expectancy": round(float(best["expectancy"]), 1),
                "excess_expectancy": round(float(best["excess_expectancy"]), 1),
                "profit_factor": round(float(best["profit_factor_clipped"]), 3),
                "max_dd_pct": round(float(best["max_dd_pct"]), 4),
                "profitable_years": int(best["profitable_years"]),
                "n_years": int(best["n_years"]),
                "total_return": round(float(best["total_return"]), 4),
                "sharpe_IS": round(float(best["sharpe"]), 3),
                "composite_rank": int(best["composite_rank"]),
                "profitable_count": profitable_count,
                "n_combos": n_combos,
                "A_baseline": round(baseline, 1),
                "filter_status": filter_status,
            })
        else:
            best_rows.append({
                "group": group, "best_combo": None, "trades": 0,
                "expectancy": None, "excess_expectancy": None,
                "profit_factor": None, "max_dd_pct": None,
                "profitable_years": 0, "n_years": 0,
                "total_return": None, "sharpe_IS": None,
                "composite_rank": None,
                "profitable_count": profitable_count,
                "n_combos": n_combos,
                "A_baseline": round(baseline, 1),
                "filter_status": filter_status,
            })

    # Save full ranked grid + best combos
    pd.DataFrame(all_rows).to_csv(PHASE3_DIR / f"combo_ranked_{args.tag}.csv", index=False)
    best_df = pd.DataFrame(best_rows).sort_values(
        ["filter_status", "composite_rank"], ascending=[True, True]
    ).reset_index(drop=True)
    best_df.to_csv(PHASE3_DIR / f"best_combos_{args.tag}.csv", index=False)

    print("===== Best combo per group (tag={}) =====".format(args.tag))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(best_df.to_string(index=False))

    print("\n===== Summary =====")
    print(f"  Groups kept:       {(best_df['filter_status'] == 'kept').sum()}")
    print(f"  Groups filtered:   {(best_df['filter_status'] == 'filtered_out').sum()}")
    print(f"  Total best combos: {best_df['best_combo'].notna().sum()}")
    print(f"\n[saved] {PHASE3_DIR}/best_combos_{args.tag}.csv, combo_ranked_{args.tag}.csv")


if __name__ == "__main__":
    main()
