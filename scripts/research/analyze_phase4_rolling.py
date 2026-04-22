"""Phase 4 — Rolling Window Stability.

Takes Phase 3 best_combos (excluding bond), evaluates each on 6 overlapping
3-year rolling windows within the IS period (2018-2023):

  W1: 2018-01 → 2020-12
  W2: 2018-07 → 2021-06
  W3: 2019-01 → 2021-12
  W4: 2019-07 → 2022-06
  W5: 2020-01 → 2022-12
  W6: 2021-01 → 2023-12

For each window:
  - combo expectancy = per-trade mean in window (from Phase 3 trades CSV)
  - A baseline      = Phase 1 A per-trade mean in window (across 3 exit_probs)
  - excess_window  = combo_exp − A_exp
  - pass = excess > 0

Filter rule: combo must pass ≥ 4/6 windows → 'stable'.
Fewer: 'unstable' (candidate for drop or replacement).

Output:
  data/runs/phase3/phase4_rolling_<tag>.csv
  data/runs/phase3/best_combos_stable_<tag>.csv
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

WINDOWS = [
    ("W1", "2018-01-01", "2020-12-31"),
    ("W2", "2018-07-01", "2021-06-30"),
    ("W3", "2019-01-01", "2021-12-31"),
    ("W4", "2019-07-01", "2022-06-30"),
    ("W5", "2020-01-01", "2022-12-31"),
    ("W6", "2021-01-01", "2023-12-31"),
]

EXCLUDED_GROUPS = ["bond"]  # per user instruction
MIN_WINDOWS_PASS = 4  # out of 6
MIN_TRADES_PER_WINDOW = 5  # drop window with too few trades to evaluate


def load_phase1_A_trades() -> pd.DataFrame:
    """Pool Phase 1 A trades across 3 exit_probs (averaged baseline)."""
    parts = []
    for prob in ["005", "007", "010"]:
        p = ABENCH / f"trades_A_ep{prob}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["_ep"] = prob
            parts.append(df)
    if not parts:
        print("ERROR: no Phase 1 A trades found")
        sys.exit(1)
    return pd.concat(parts, ignore_index=True)


def window_stats(trades: pd.DataFrame, start, end, date_col="entry_date") -> dict:
    """Slice trades by window date range; return (n, mean, std)."""
    t = trades.copy()
    t[date_col] = pd.to_datetime(t[date_col])
    mask = (t[date_col] >= start) & (t[date_col] <= end)
    sub = t[mask]
    if len(sub) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan}
    return {
        "n": len(sub),
        "mean": float(sub["net_pnl"].mean()),
        "std": float(sub["net_pnl"].std(ddof=1)) if len(sub) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="Phase 3 tag, e.g. risk3cap6")
    args = ap.parse_args()

    best_path = PHASE3_DIR / f"best_combos_{args.tag}.csv"
    if not best_path.exists():
        print(f"ERROR: {best_path} not found")
        sys.exit(1)
    best = pd.read_csv(best_path)
    best = best[best["filter_status"] == "kept"]
    before = len(best)
    best = best[~best["group"].isin(EXCLUDED_GROUPS)]
    print(f"Loaded {before} kept combos; {len(best)} after excluding {EXCLUDED_GROUPS}")

    trades_path = PHASE3_DIR / f"trades_{args.tag}.csv"
    if not trades_path.exists():
        print(f"ERROR: {trades_path} not found")
        sys.exit(1)
    phase3_trades = pd.read_csv(trades_path)
    phase3_trades["entry_date"] = pd.to_datetime(phase3_trades["entry_date"])
    print(f"Phase 3 trades: {len(phase3_trades)} rows")

    # Phase 1 A baselines (pooled 3 probs)
    phase1_A = load_phase1_A_trades()
    phase1_A["entry_date"] = pd.to_datetime(phase1_A["entry_date"])
    print(f"Phase 1 A trades: {len(phase1_A)} rows (pooled 3 exit_probs)")

    # Print window grid
    print("\n=== 6 Rolling Windows (3-year, 6-month step) ===")
    for wname, ws, we in WINDOWS:
        print(f"  {wname}: {ws} → {we}")

    # Evaluate each best combo on each window
    rolling_rows = []
    stable_rows = []
    print("\n=== Per-combo window evaluation ===")

    for _, row in best.iterrows():
        group = row["group"]
        combo = row["best_combo"]
        entry_id, exit_id = combo.split("+")

        # Phase 3 trades for this group × combo
        ct = phase3_trades[
            (phase3_trades["group"] == group)
            & (phase3_trades["entry_id"] == entry_id)
            & (phase3_trades["exit_id"] == exit_id)
        ]
        # Phase 1 A trades for this group (pooled 3 probs)
        at = phase1_A[phase1_A["group"] == group]

        window_results = []
        for wname, ws, we in WINDOWS:
            cstats = window_stats(ct, ws, we)
            astats = window_stats(at, ws, we)

            excess = (
                cstats["mean"] - astats["mean"]
                if cstats["n"] >= MIN_TRADES_PER_WINDOW and not np.isnan(astats["mean"])
                else np.nan
            )
            passes = (excess > 0) if not np.isnan(excess) else False

            window_results.append({
                "group": group, "combo": combo, "window": wname,
                "ws": ws, "we": we,
                "combo_trades": cstats["n"], "combo_exp": round(cstats["mean"], 1) if not np.isnan(cstats["mean"]) else None,
                "A_trades": astats["n"], "A_exp": round(astats["mean"], 1) if not np.isnan(astats["mean"]) else None,
                "excess": round(excess, 1) if not np.isnan(excess) else None,
                "pass": passes,
            })
        rolling_rows.extend(window_results)

        pass_count = sum(1 for w in window_results if w["pass"])
        absolute_profitable = sum(
            1 for w in window_results
            if w["combo_exp"] is not None and w["combo_exp"] > 0
        )
        stable = pass_count >= MIN_WINDOWS_PASS
        status = "stable" if stable else "unstable"

        excess_values = [w["excess"] for w in window_results if w["excess"] is not None]
        combo_exp_values = [w["combo_exp"] for w in window_results if w["combo_exp"] is not None]

        stable_rows.append({
            "group": group, "best_combo": combo,
            "IS_expectancy": row["expectancy"],
            "IS_excess": row["excess_expectancy"],
            "windows_pass": pass_count,
            "windows_total": len(WINDOWS),
            "windows_absolute_profitable": absolute_profitable,
            "excess_mean": round(np.mean(excess_values), 1) if excess_values else None,
            "excess_min": round(min(excess_values), 1) if excess_values else None,
            "excess_max": round(max(excess_values), 1) if excess_values else None,
            "combo_exp_mean": round(np.mean(combo_exp_values), 1) if combo_exp_values else None,
            "combo_exp_min": round(min(combo_exp_values), 1) if combo_exp_values else None,
            "stability_status": status,
        })

        print(f"  {group:15s} {combo:30s} "
              f"pass={pass_count}/6, abs_profit={absolute_profitable}/6, "
              f"status={status}")

    # Save outputs
    rolling_df = pd.DataFrame(rolling_rows)
    rolling_df.to_csv(PHASE3_DIR / f"phase4_rolling_{args.tag}.csv", index=False)
    stable_df = pd.DataFrame(stable_rows).sort_values(
        ["stability_status", "windows_pass"], ascending=[True, False]
    ).reset_index(drop=True)
    stable_df.to_csv(PHASE3_DIR / f"best_combos_stable_{args.tag}.csv", index=False)

    print("\n===== Stability summary =====")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(stable_df.to_string(index=False))

    print("\n===== Counts =====")
    print(stable_df["stability_status"].value_counts().to_string())

    print(f"\n[saved] {PHASE3_DIR}/phase4_rolling_{args.tag}.csv, best_combos_stable_{args.tag}.csv")


if __name__ == "__main__":
    main()
