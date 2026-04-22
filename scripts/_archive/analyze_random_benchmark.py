"""Four-experiment alpha attribution analyzer.

Reads:
  data/runs/alpha_benchmark/exp_A.csv  (rand × rand)
  data/runs/alpha_benchmark/exp_B.csv  (rand × real)
  data/runs/alpha_benchmark/exp_C.csv  (real × rand)
  data/runs/<baseline>/backtest_strategy_layer.csv  (real × real, default: v2)

Aggregates: trade-weighted Expectancy = sum(net_pnl) / sum(trades),
across all years and seeds (robust against year-level noise).

Decision rules (per group):
  A > 0     → beta/drift present (adjust for this before reading D as "alpha")
  B_best > A → exit has alpha
  C_best > A → entry has alpha
  D > B_best → entry adds value on top of exit
  D > C_best → exit adds value on top of entry
  D > A + (B_best - A) + (C_best - A) → positive entry×exit synergy

Produces four CSVs under data/runs/alpha_benchmark/:
  summary.csv          — group × experiment expectancy matrix + verdict
  exit_attribution.csv — group × exit delta-expectancy (B_<exit> - A)
  entry_attribution.csv — group × entry delta-expectancy (C_<entry> - A)
  verdict.csv          — per-group label

Usage:
  python scripts/analyze_random_benchmark.py
  python scripts/analyze_random_benchmark.py --baseline v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ABENCH = ROOT / "data" / "runs" / "alpha_benchmark"


def weighted_expectancy(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Trade-weighted expectancy: sum(net_pnl) / sum(trades)."""
    agg = df.groupby(group_cols, as_index=False).agg(
        trades=("trades", "sum"),
        net_pnl=("net_pnl", "sum"),
    )
    agg["expectancy"] = np.where(
        agg["trades"] > 0, agg["net_pnl"] / agg["trades"], 0.0
    )
    return agg


def per_seed_expectancy(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Per-seed weighted expectancy — used for CI."""
    agg = df.groupby(group_cols + ["seed"], as_index=False).agg(
        trades=("trades", "sum"),
        net_pnl=("net_pnl", "sum"),
    )
    agg["expectancy"] = np.where(
        agg["trades"] > 0, agg["net_pnl"] / agg["trades"], 0.0
    )
    return agg


def load_exp(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: {path} missing — skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["year"] > 0].copy()
    return df


def load_baseline_D(run: str) -> pd.DataFrame:
    """Real × real baseline from existing layer1 CSV."""
    path = ROOT / "data" / "runs" / run / "backtest_strategy_layer.csv"
    df = pd.read_csv(path)
    df = df[df["year"] > 0].copy()
    df[["entry", "exit"]] = df["combo"].str.split("+", expand=True)
    df["seed"] = -1  # no seeds in D
    return df


def a_vs_b_per_exit(expA: pd.DataFrame, expB: pd.DataFrame) -> pd.DataFrame:
    """ΔExpectancy for each (group, exit) vs A's rand+rand.

    B has rand entry × real exit; aggregate across the rand entry seeds.
    Returns rows (group, exit, exp_B, exp_A, delta).
    """
    a_exp = weighted_expectancy(expA, ["group"]).rename(columns={"expectancy": "exp_A"})
    b_by_exit = weighted_expectancy(expB, ["group", "exit"]).rename(
        columns={"expectancy": "exp_B"}
    )
    merged = b_by_exit.merge(a_exp[["group", "exp_A"]], on="group", how="left")
    merged["delta"] = merged["exp_B"] - merged["exp_A"]
    return merged[["group", "exit", "exp_A", "exp_B", "delta", "trades"]]


def a_vs_c_per_entry(expA: pd.DataFrame, expC: pd.DataFrame) -> pd.DataFrame:
    a_exp = weighted_expectancy(expA, ["group"]).rename(columns={"expectancy": "exp_A"})
    c_by_entry = weighted_expectancy(expC, ["group", "entry"]).rename(
        columns={"expectancy": "exp_C"}
    )
    merged = c_by_entry.merge(a_exp[["group", "exp_A"]], on="group", how="left")
    merged["delta"] = merged["exp_C"] - merged["exp_A"]
    return merged[["group", "entry", "exp_A", "exp_C", "delta", "trades"]]


def summary_table(
    expA: pd.DataFrame, expB: pd.DataFrame, expC: pd.DataFrame, expD: pd.DataFrame
) -> pd.DataFrame:
    """One row per group. A with ±CI (across 3 seeds), B_best, C_best, D_best."""
    rows = []
    groups = sorted(set(expA["group"]) | set(expB["group"]) | set(expC["group"]))

    a_seed = per_seed_expectancy(expA, ["group"])
    a_all = weighted_expectancy(expA, ["group"])
    b_by_exit = weighted_expectancy(expB, ["group", "exit"])
    c_by_entry = weighted_expectancy(expC, ["group", "entry"])
    d_by_combo = weighted_expectancy(expD, ["group", "combo"])

    for g in groups:
        # A (pooled across seeds) + per-seed std/CI
        a_row = a_all[a_all["group"] == g]
        a_exp = float(a_row["expectancy"].iloc[0]) if not a_row.empty else np.nan
        a_trades = int(a_row["trades"].iloc[0]) if not a_row.empty else 0
        a_seeds = a_seed[a_seed["group"] == g]["expectancy"].values
        a_std = float(np.std(a_seeds, ddof=1)) if len(a_seeds) > 1 else np.nan

        # B: best exit (highest expectancy), plus its delta vs A
        b_sub = b_by_exit[b_by_exit["group"] == g].sort_values("expectancy", ascending=False)
        b_best_row = b_sub.iloc[0] if not b_sub.empty else None

        # C: best entry
        c_sub = c_by_entry[c_by_entry["group"] == g].sort_values("expectancy", ascending=False)
        c_best_row = c_sub.iloc[0] if not c_sub.empty else None

        # D: best real×real combo
        d_sub = d_by_combo[d_by_combo["group"] == g].sort_values("expectancy", ascending=False)
        d_best_row = d_sub.iloc[0] if not d_sub.empty else None

        rows.append({
            "group": g,
            "A_exp": round(a_exp, 1),
            "A_std": round(a_std, 1) if not np.isnan(a_std) else None,
            "A_trades": a_trades,
            "B_best_exit": b_best_row["exit"] if b_best_row is not None else None,
            "B_best_exp": round(float(b_best_row["expectancy"]), 1) if b_best_row is not None else None,
            "B_best_trades": int(b_best_row["trades"]) if b_best_row is not None else 0,
            "C_best_entry": c_best_row["entry"] if c_best_row is not None else None,
            "C_best_exp": round(float(c_best_row["expectancy"]), 1) if c_best_row is not None else None,
            "C_best_trades": int(c_best_row["trades"]) if c_best_row is not None else 0,
            "D_best_combo": d_best_row["combo"] if d_best_row is not None else None,
            "D_best_exp": round(float(d_best_row["expectancy"]), 1) if d_best_row is not None else None,
            "D_best_trades": int(d_best_row["trades"]) if d_best_row is not None else 0,
        })
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def compute_verdict(row: pd.Series) -> Dict[str, str]:
    """Label each group by alpha pattern.

    Rules (in CNY per trade):
      A > BETA_THR      → beta-driven (drift unhedged by allow_short)
      D − B > SIG_THR   → entry has MARGINAL value over best-exit-alone
      D − C > SIG_THR   → exit has MARGINAL value over best-entry-alone
      Both above        → alpha-clear (both sides contribute)
      Only D−B          → entry-marginal-only (exit is the floor)
      Only D−C          → exit-marginal-only (entry is the floor)
      Neither           → dead-pairing (combo doesn't beat best individual)

    Standalone tests (B−A, C−A) reported separately as 'exit-standalone' and
    'entry-standalone' tags so we can spot alphas that only work in pair.
    """
    a = row["A_exp"]
    b = row["B_best_exp"]
    c = row["C_best_exp"]
    d = row["D_best_exp"]
    a_ci = (row["A_std"] or 0) * 1.96 / np.sqrt(3) if row["A_std"] else 0
    BETA_THR = 50.0
    SIG_THR = max(a_ci, 30.0)

    labels = []
    if a > BETA_THR:
        labels.append("beta-driven")

    has_d_over_b = (d is not None) and (b is not None) and (d - b > SIG_THR)
    has_d_over_c = (d is not None) and (c is not None) and (d - c > SIG_THR)

    if has_d_over_b and has_d_over_c:
        labels.append("alpha-clear")
    elif has_d_over_b:
        labels.append("entry-marginal-only")
    elif has_d_over_c:
        labels.append("exit-marginal-only")
    else:
        labels.append("dead-pairing")

    # Standalone capability tags (B − A, C − A): does the alpha work even
    # when paired with random on the other side?
    if (b is not None) and (b - a > SIG_THR):
        labels.append("exit-standalone+")
    if (c is not None) and (c - a > SIG_THR):
        labels.append("entry-standalone+")
    return ";".join(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="v2",
                    help="Run name for D experiment (real×real). Default: v2")
    args = ap.parse_args()

    expA = load_exp(ABENCH / "exp_A.csv")
    expB = load_exp(ABENCH / "exp_B.csv")
    expC = load_exp(ABENCH / "exp_C.csv")
    expD = load_baseline_D(args.baseline)

    print(f"Loaded:  A={len(expA)}  B={len(expB)}  C={len(expC)}  D={len(expD)}")

    if expA.empty or expB.empty or expC.empty:
        print("Missing experiment data; run runs/run_random_benchmark.py first.")
        sys.exit(1)

    # === Main summary table ===
    summary = summary_table(expA, expB, expC, expD)
    summary["D_minus_B"] = summary["D_best_exp"] - summary["B_best_exp"]
    summary["D_minus_C"] = summary["D_best_exp"] - summary["C_best_exp"]
    summary["verdict"] = summary.apply(compute_verdict, axis=1)
    summary.to_csv(ABENCH / "summary.csv", index=False)
    print("\n===== Table 1 — Group × Experiment Expectancy (CNY per trade) =====")
    print(summary.to_string(index=False))

    # === Exit attribution ===
    exit_attrib = a_vs_b_per_exit(expA, expB)
    exit_attrib.to_csv(ABENCH / "exit_attribution.csv", index=False)
    # Pivot for readability
    pvt_exit = exit_attrib.pivot(index="group", columns="exit", values="delta").round(1)
    print("\n===== Table 2 — Exit Alpha Δ (B_exit − A, CNY per trade) =====")
    print(pvt_exit.to_string())

    # === Entry attribution ===
    entry_attrib = a_vs_c_per_entry(expA, expC)
    entry_attrib.to_csv(ABENCH / "entry_attribution.csv", index=False)
    pvt_entry = entry_attrib.pivot(index="group", columns="entry", values="delta").round(1)
    print("\n===== Table 3 — Entry Alpha Δ (C_entry − A, CNY per trade) =====")
    print(pvt_entry.to_string())

    # === Verdict summary ===
    verdicts = summary[["group", "verdict"]].copy()
    verdicts.to_csv(ABENCH / "verdict.csv", index=False)
    print("\n===== Table 4 — Verdict per group =====")
    print(verdicts.to_string(index=False))

    # === Aggregate stats ===
    print("\n===== Verdict distribution =====")
    all_labels = []
    for v in verdicts["verdict"]:
        all_labels.extend(v.split(";"))
    print(pd.Series(all_labels).value_counts().to_string())

    print(f"\n[saved] {ABENCH}/summary.csv, exit_attribution.csv, entry_attribution.csv, verdict.csv")


if __name__ == "__main__":
    main()
