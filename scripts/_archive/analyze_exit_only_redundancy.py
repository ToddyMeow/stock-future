"""Test 3: prove the entry signal is redundant on exit-only groups.

For each exit-only group (agri, chem_energy, metals), compare:
  - B = rand_entry × best_exit  (from data/runs/alpha_benchmark/exp_B.csv)
  - D = best real_entry × best real_exit  (from data/runs/v2/backtest_strategy_layer.csv)

Headline metrics: per-trade Expectancy + Sharpe + win_rate + profit factor.

Read this output as: if B's per-trade behavior matches D, the entry is
contributing nothing on these groups; we could replace it with a random
trigger of similar frequency without losing Sharpe.

Caveats:
  - Random entry fires far more often → total trades and CAGR are NOT
    directly comparable (annual exposure differs). Expectancy and per-trade
    win_rate / wl_ratio ARE comparable.
  - Sharpe in B is per-year-mean from yearly_stats; treat it as an indicator
    not a portfolio-grade number.

Usage:
  python scripts/analyze_exit_only_redundancy.py
  python scripts/analyze_exit_only_redundancy.py --groups agri,chem_energy,metals
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_B(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["year"] > 0].copy()


def load_D(run: str = "v2") -> pd.DataFrame:
    p = ROOT / "data" / "runs" / run / "backtest_strategy_layer.csv"
    df = pd.read_csv(p)
    df = df[df["year"] > 0].copy()
    df[["entry", "exit"]] = df["combo"].str.split("+", expand=True)
    return df


def trade_weighted(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    g = df.groupby(group_cols, as_index=False).agg(
        trades=("trades", "sum"),
        net_pnl=("net_pnl", "sum"),
        wins_proxy=("win_rate", lambda x: (x.values * df.loc[x.index, "trades"].values).sum()),
        sharpe_mean=("sharpe", "mean"),
        pf_mean=("profit_factor", "mean"),
    )
    g["expectancy"] = np.where(g["trades"] > 0, g["net_pnl"] / g["trades"], 0.0)
    g["win_rate"] = np.where(g["trades"] > 0, g["wins_proxy"] / g["trades"], 0.0)
    return g.drop(columns=["wins_proxy"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", default="agri,chem_energy,metals")
    ap.add_argument("--baseline", default="v2", help="Run name for D (real×real). Default v2")
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",")]
    print(f"Exit-only groups under test: {groups}")
    print(f"D baseline run: {args.baseline}")

    B = load_B(ROOT / "data" / "runs" / "alpha_benchmark" / "exp_B.csv")
    D = load_D(args.baseline)
    B = B[B["group"].isin(groups)]
    D = D[D["group"].isin(groups)]

    B_per_exit = trade_weighted(B, ["group", "exit"])
    D_per_combo = trade_weighted(D, ["group", "combo", "entry", "exit"])

    rows = []
    for g in groups:
        # B's best exit by expectancy
        b_g = B_per_exit[B_per_exit["group"] == g].sort_values("expectancy", ascending=False)
        if b_g.empty:
            continue
        b_best = b_g.iloc[0]

        # D's best combo by expectancy
        d_g = D_per_combo[D_per_combo["group"] == g].sort_values("expectancy", ascending=False)
        if d_g.empty:
            continue
        d_best = d_g.iloc[0]

        # D's best combo using THE SAME exit as B (so we isolate the entry's contribution)
        d_same_exit = D_per_combo[
            (D_per_combo["group"] == g) & (D_per_combo["exit"] == b_best["exit"])
        ].sort_values("expectancy", ascending=False)
        d_same = d_same_exit.iloc[0] if not d_same_exit.empty else None

        rows.append({
            "group": g,
            "B_alpha": f"rand+{b_best['exit']}",
            "B_trades": int(b_best["trades"]),
            "B_exp": round(b_best["expectancy"], 1),
            "B_sharpe": round(b_best["sharpe_mean"], 2),
            "B_pf": round(b_best["pf_mean"], 2),
            "B_wr": round(b_best["win_rate"], 3),
            "D_best_combo": d_best["combo"],
            "D_trades": int(d_best["trades"]),
            "D_exp": round(d_best["expectancy"], 1),
            "D_sharpe": round(d_best["sharpe_mean"], 2),
            "D_pf": round(d_best["pf_mean"], 2),
            "D_wr": round(d_best["win_rate"], 3),
            "D_same_exit_combo": d_same["combo"] if d_same is not None else None,
            "D_same_exit_exp": round(d_same["expectancy"], 1) if d_same is not None else None,
            "D_same_exit_trades": int(d_same["trades"]) if d_same is not None else 0,
            "delta_D_minus_B": (
                round(d_same["expectancy"] - b_best["expectancy"], 1) if d_same is not None else None
            ),
        })

    out = pd.DataFrame(rows)
    print("\n===== Test 3 — Exit-only group redundancy check =====")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(out.to_string(index=False))

    print("\nReading guide:")
    print("  delta_D_minus_B is the per-trade expectancy gain from using")
    print("  REAL entry instead of RAND entry, keeping the SAME exit.")
    print("  - If positive and large: entry adds value")
    print("  - If near zero or negative: entry is redundant on this group")

    out_path = ROOT / "data" / "runs" / "alpha_benchmark" / "exit_only_redundancy.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
