"""Alpha attribution analysis from strategy-layer CSV.

Question: is each alpha (entry / exit) truly useful? And is it useful
universally or only on specific groups/symbols?

Reads `data/runs/<suffix>/backtest_strategy_layer.csv` (output of
run_three_layer_backtest.py) which has rows (group, combo, year, ...).
combo = "<entry>+<exit>".

Produces four views:
  1. Per-entry marginal value (averaged over all exits)
  2. Per-exit marginal value (averaged over all entries)
  3. Per-group × per-entry heatmap (group-specific contribution)
  4. Per-group × per-exit heatmap
  5. "Top-combo dominance": how often each alpha appears in the best combo per group

Usage:
  python scripts/analyze_alpha_attribution.py --suffix v2
  python scripts/analyze_alpha_attribution.py --suffix v2 --metric sharpe  # default
  python scripts/analyze_alpha_attribution.py --suffix v2 --min-trades 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_layer1(suffix: str) -> pd.DataFrame:
    path = ROOT / "data" / "runs" / suffix / "backtest_strategy_layer.csv"
    df = pd.read_csv(path)
    df = df[df["year"] > 0].copy()  # drop zero-trade placeholder rows
    df[["entry", "exit"]] = df["combo"].str.split("+", expand=True)
    return df


def aggregate_by_combo(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    """Aggregate (group, combo) across years."""
    agg = df.groupby(["group", "combo", "entry", "exit"]).agg(
        trades=("trades", "sum"),
        pnl=("net_pnl", "sum"),
        sharpe=("sharpe", "mean"),
        pf=("profit_factor", "mean"),
        worst_dd=("max_dd_pct", "min"),
        avg_r=("avg_r", "mean"),
        win_rate=("win_rate", "mean"),
    ).reset_index()
    agg = agg[agg["trades"] >= min_trades]
    return agg


def marginal_view(agg: pd.DataFrame, alpha_col: str) -> pd.DataFrame:
    """Average metric across all pairings of the OTHER side.

    alpha_col: 'entry' or 'exit'. Marginalizes over the complement.
    """
    rows = []
    for name, sub in agg.groupby(alpha_col):
        rows.append({
            alpha_col: name,
            "n_pairings": len(sub),
            "total_trades": int(sub["trades"].sum()),
            "total_pnl": round(sub["pnl"].sum(), 0),
            "mean_sharpe": round(sub["sharpe"].mean(), 3),
            "median_sharpe": round(sub["sharpe"].median(), 3),
            "pos_sharpe_rate": round((sub["sharpe"] > 0).mean(), 3),
            "mean_pf": round(sub["pf"].mean(), 3),
            "mean_avg_r": round(sub["avg_r"].mean(), 3),
            "worst_dd": round(sub["worst_dd"].min(), 3),
        })
    out = pd.DataFrame(rows).sort_values("mean_sharpe", ascending=False)
    return out


def per_group_heatmap(agg: pd.DataFrame, alpha_col: str, metric: str) -> pd.DataFrame:
    """Pivot: rows=alpha, cols=group, values=mean metric across complement side."""
    pvt = agg.groupby(["group", alpha_col])[metric].mean().unstack(0)
    # Order alphas by overall mean
    pvt["_mean"] = pvt.mean(axis=1)
    pvt = pvt.sort_values("_mean", ascending=False).drop(columns="_mean")
    return pvt


def top_combo_dominance(agg: pd.DataFrame) -> pd.DataFrame:
    """For each group, pick top-1 by Sharpe rank + PF rank + worst_dd rank
    (same logic as pick_best_combo). Count how often each entry/exit wins.
    """
    rows = []
    for group, sub in agg.groupby("group"):
        if sub.empty:
            continue
        s = sub.copy()
        s["r_sh"] = s["sharpe"].rank(ascending=False)
        s["r_pf"] = s["pf"].rank(ascending=False)
        s["r_dd"] = s["worst_dd"].rank(ascending=False)
        s["r_sum"] = s["r_sh"] + s["r_pf"] + s["r_dd"]
        best = s.sort_values("r_sum").iloc[0]
        rows.append({
            "group": group,
            "best_combo": best["combo"],
            "entry": best["entry"],
            "exit": best["exit"],
            "sharpe": round(best["sharpe"], 3),
            "pf": round(best["pf"], 3),
            "trades": int(best["trades"]),
        })
    return pd.DataFrame(rows)


def dead_alpha_report(agg: pd.DataFrame, dominance: pd.DataFrame) -> dict:
    """Identify entries/exits that never win (top-1) in any group."""
    all_entries = set(agg["entry"].unique())
    all_exits = set(agg["exit"].unique())
    winning_entries = set(dominance["entry"].unique())
    winning_exits = set(dominance["exit"].unique())
    return {
        "dead_entries": sorted(all_entries - winning_entries),
        "dead_exits": sorted(all_exits - winning_exits),
        "winning_entries_count": dominance["entry"].value_counts().to_dict(),
        "winning_exits_count": dominance["exit"].value_counts().to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="v2", help="CSV suffix (e.g. v2)")
    ap.add_argument("--min-trades", type=int, default=30,
                    help="Min total trades per (group, combo) to count; relaxed for ind_*")
    ap.add_argument("--metric", default="sharpe",
                    choices=["sharpe", "pf", "pnl", "avg_r"])
    ap.add_argument("--out", default=None,
                    help="Output prefix for CSVs (default: data/alpha_attrib_<suffix>)")
    args = ap.parse_args()

    df = load_layer1(args.suffix)
    print(f"Loaded {len(df)} rows, {df['group'].nunique()} groups, "
          f"{df['combo'].nunique()} combos, "
          f"entries={sorted(df['entry'].unique())}, "
          f"exits={sorted(df['exit'].unique())}")

    # Relax min_trades for ind_* single-symbol groups
    multi = df[~df["group"].str.startswith("ind_")]
    ind = df[df["group"].str.startswith("ind_")]
    agg_multi = aggregate_by_combo(multi, args.min_trades)
    agg_ind = aggregate_by_combo(ind, max(args.min_trades // 3, 10))
    agg = pd.concat([agg_multi, agg_ind], ignore_index=True)

    # View 1 & 2: marginal
    entry_marg = marginal_view(agg, "entry")
    exit_marg = marginal_view(agg, "exit")

    print("\n===== ENTRY marginal (averaged over exits) =====")
    print(entry_marg.to_string(index=False))

    print("\n===== EXIT marginal (averaged over entries) =====")
    print(exit_marg.to_string(index=False))

    # View 3 & 4: heatmaps
    metric_col = {"sharpe": "sharpe", "pf": "pf", "pnl": "pnl", "avg_r": "avg_r"}[args.metric]
    entry_heat = per_group_heatmap(agg, "entry", metric_col)
    exit_heat = per_group_heatmap(agg, "exit", metric_col)

    print(f"\n===== ENTRY × GROUP heatmap ({args.metric}, mean over exits) =====")
    print(entry_heat.round(2).to_string())

    print(f"\n===== EXIT × GROUP heatmap ({args.metric}, mean over entries) =====")
    print(exit_heat.round(2).to_string())

    # View 5: dominance
    dom = top_combo_dominance(agg)
    dead = dead_alpha_report(agg, dom)
    print("\n===== BEST combo per group (composite rank) =====")
    print(dom.to_string(index=False))

    print("\n===== ALPHA DOMINANCE SUMMARY =====")
    print("Winning entries (top-1 in group count):")
    for k, v in sorted(dead["winning_entries_count"].items(), key=lambda x: -x[1]):
        print(f"  {k:<15s} {v} group(s)")
    print("Winning exits (top-1 in group count):")
    for k, v in sorted(dead["winning_exits_count"].items(), key=lambda x: -x[1]):
        print(f"  {k:<15s} {v} group(s)")
    if dead["dead_entries"]:
        print(f"DEAD entries (never top-1): {dead['dead_entries']}")
    if dead["dead_exits"]:
        print(f"DEAD exits (never top-1): {dead['dead_exits']}")

    # Save outputs
    out_prefix = args.out or str(ROOT / "data" / "runs" / args.suffix / "alpha_attrib")
    entry_marg.to_csv(f"{out_prefix}_entry_marginal.csv", index=False)
    exit_marg.to_csv(f"{out_prefix}_exit_marginal.csv", index=False)
    entry_heat.to_csv(f"{out_prefix}_entry_group_heatmap.csv")
    exit_heat.to_csv(f"{out_prefix}_exit_group_heatmap.csv")
    dom.to_csv(f"{out_prefix}_dominance.csv", index=False)
    print(f"\n[saved] {out_prefix}_*.csv")


if __name__ == "__main__":
    main()
