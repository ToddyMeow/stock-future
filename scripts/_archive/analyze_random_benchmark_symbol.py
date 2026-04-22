"""Phase 2 per-symbol drill-down for alpha-clear groups (修正版).

Reads:
  data/runs/alpha_benchmark/exp_A_symbol.csv  (随机入场 × 随机出场)
  data/runs/alpha_benchmark/exp_B_symbol.csv  (随机入场 × 真实出场)
  data/runs/alpha_benchmark/exp_C_symbol.csv  (真实入场 × 随机出场)
  data/runs/v4/backtest_strategy_layer.csv     (真实入场 × 真实出场, group=symbol)

Per the user's guidance, at per-symbol granularity we DO NOT use expectancy
as primary metric (too few trades per symbol → wide CI). Instead:
  - win_rate = wins / trades       (descriptive, robust)
  - wl_ratio = -avg_winner / avg_loser
             = profit_factor × (1 - win_rate) / win_rate

Marginal alpha (the correct test):
  - 入场边际 = wl_ratio(真实×真实 best) − wl_ratio(随机×真实 best)
  - 出场边际 = wl_ratio(真实×真实 best) − wl_ratio(真实×随机 best)

Verdict per symbol:
  - alpha-clear: both marginals > 0.10
  - entry-marginal-only: 入场边际 > 0.10, 出场边际 ≤ 0.10
  - exit-marginal-only:  出场边际 > 0.10, 入场边际 ≤ 0.10
  - dead-pairing: neither > 0.10
  + 'tiny-sample' if A_trades < 30
  + 'beta-driven' if A_exp > 30 (CNY per trade)

Outputs:
  data/runs/alpha_benchmark/symbol_summary.csv
  data/runs/alpha_benchmark/symbol_marginal.csv

Usage:
  python scripts/analyze_random_benchmark_symbol.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ABENCH = ROOT / "data" / "runs" / "alpha_benchmark"

# Phase 2 spans all 13 active groups now (was just 4 alpha-clear initially).
# We concat both the original symbol files and the _remaining ones.
TARGET_GROUPS = [
    "black_steel", "building", "livestock", "rubber_fiber",  # 22 syms (orig)
    "agri", "chem_energy", "metals",                          # 32 syms (remaining)
    "ind_CJ", "ind_EC", "ind_FB", "ind_LU", "ind_RR", "ind_WH",  # 6 syms (remaining)
]


def load_random_exp(name: str) -> pd.DataFrame:
    """Concat symbol-scope CSVs (original + _remaining). Drops zero-trade rows."""
    parts = []
    for tag in ("", "_remaining"):
        p = ABENCH / f"exp_{name}_symbol{tag}.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df[df["year"] > 0].copy()
    return df


def load_real_real_per_symbol(target_syms: set) -> pd.DataFrame:
    """Load v4 per-symbol layer-1 (real×real). Filter to target syms.

    Adds derived wl_ratio = pf × (1 − wr) / wr (since v4 lacks gross_win).
    """
    p = ROOT / "data" / "runs" / "v4" / "backtest_strategy_layer.csv"
    df = pd.read_csv(p)
    df = df[df["year"] > 0].copy()
    df = df[df["symbol"].isin(target_syms)].copy()
    df[["entry", "exit"]] = df["combo"].str.split("+", expand=True)
    return df


def aggregate_random(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Pool trades across years + seeds; compute win_rate and wl_ratio."""
    agg = df.groupby(group_cols, as_index=False).agg(
        trades=("trades", "sum"),
        wins=("wins", "sum"),
        losses=("losses", "sum"),
        net_pnl=("net_pnl", "sum"),
        gross_win=("gross_win", "sum"),
        gross_loss=("gross_loss", "sum"),
    )
    agg["win_rate"] = np.where(agg["trades"] > 0, agg["wins"] / agg["trades"], 0.0)
    agg["avg_winner"] = np.where(agg["wins"] > 0, agg["gross_win"] / agg["wins"], 0.0)
    agg["avg_loser"] = np.where(agg["losses"] > 0, agg["gross_loss"] / agg["losses"], 0.0)
    agg["wl_ratio"] = np.where(
        agg["avg_loser"] < 0, -agg["avg_winner"] / agg["avg_loser"], 0.0
    )
    agg["expectancy"] = np.where(agg["trades"] > 0, agg["net_pnl"] / agg["trades"], 0.0)
    return agg


def aggregate_real_real(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Pool trades across years; derive gross_win/gross_loss per yearly row,
    then sum and compute wl_ratio.

    Per yearly row, given net_pnl and profit_factor (PF):
      net_pnl = gross_win + gross_loss   (gross_loss ≤ 0)
      PF      = gross_win / |gross_loss|
      → |gross_loss| = net_pnl / (PF − 1)   (works for PF > 1 OR PF < 1)
      → gross_win   = PF × |gross_loss|
    Edge case: PF = 999 (placeholder for inf, all-wins year) → gross_loss=0,
               gross_win = net_pnl.
    Edge case: PF ≈ 1 → ambiguous; treat as gross_win ≈ |gross_loss| ≈ net_pnl/2.
    """
    df = df.copy()
    pf = df["profit_factor"].clip(lower=0.001, upper=999.0).astype(float)
    net = df["net_pnl"].astype(float)
    delta = pf - 1.0
    safe_delta = np.where(np.abs(delta) > 1e-3, delta, 1e-3)
    # gross_loss = -net_pnl / (pf - 1), signed negative
    gross_loss = np.where(pf >= 999.0, 0.0, -net / safe_delta)
    gross_win = np.where(pf >= 999.0, net, pf * np.abs(gross_loss))
    df["gw_yr"] = gross_win
    df["gl_yr"] = gross_loss
    df["wins_yr"] = df["win_rate"] * df["trades"]
    df["losses_yr"] = df["trades"] - df["wins_yr"]

    agg = df.groupby(group_cols, as_index=False).agg(
        trades=("trades", "sum"),
        net_pnl=("net_pnl", "sum"),
        wins=("wins_yr", "sum"),
        losses=("losses_yr", "sum"),
        gross_win=("gw_yr", "sum"),
        gross_loss=("gl_yr", "sum"),
    )
    agg["win_rate"] = np.where(agg["trades"] > 0, agg["wins"] / agg["trades"], 0.0)
    agg["avg_winner"] = np.where(agg["wins"] > 0, agg["gross_win"] / agg["wins"], 0.0)
    agg["avg_loser"] = np.where(agg["losses"] > 0, agg["gross_loss"] / agg["losses"], 0.0)
    # Cap wl_ratio at 50 (anything above is dominated by pf=999 edge cases)
    raw_wl = np.where(
        agg["avg_loser"] < 0, -agg["avg_winner"] / agg["avg_loser"], 0.0
    )
    agg["wl_ratio"] = np.clip(raw_wl, 0.0, 50.0)
    agg["expectancy"] = np.where(agg["trades"] > 0, agg["net_pnl"] / agg["trades"], 0.0)
    return agg


def best_per_symbol(df: pd.DataFrame, alpha_col: str, min_trades: int = 20):
    """For each (group, symbol), pick the alpha with the highest wl_ratio
    (ties broken by expectancy).
    """
    rows = []
    for (grp, sym), sub in df.groupby(["group", "symbol"]):
        sub = sub[sub["trades"] >= min_trades]
        if sub.empty:
            continue
        best = sub.sort_values(["wl_ratio", "expectancy"], ascending=False).iloc[0]
        rows.append({
            "group": grp, "symbol": sym, "best_alpha": best[alpha_col],
            "trades": int(best["trades"]),
            "win_rate": round(float(best["win_rate"]), 3),
            "wl_ratio": round(float(best["wl_ratio"]), 3),
            "expectancy": round(float(best["expectancy"]), 0),
        })
    return pd.DataFrame(rows)


def kelly_edge(wr: float, wl: float) -> float:
    """Kelly edge = wr × (wl + 1) − 1.

    Combines win_rate and wl_ratio into a dimensionless 'edge' number:
      edge > 0  →  positive expectancy per bet (in fractions of risk unit)
      edge = 0  →  break-even
      edge < 0  →  losing edge

    This avoids the wl/wr trade-off paradox: real exits drop win_rate but
    boost wl_ratio. Kelly answers "did the trade-off net out positive?"
    """
    return wr * (wl + 1.0) - 1.0


def label_symbol(a, b, c, d, edge_sig=0.05) -> str:
    """Verdict: each alpha (B, C, D) compared against A's Kelly edge.

    Per user guidance:
      - Per-symbol baseline = A (随机入场×随机出场), NOT v2
      - Use win_rate + wl_ratio (not expectancy) — combined via Kelly edge
      - threshold edge_sig=0.05 ≈ 5pt of bet size positive edge over random

    Three independent flags:
      - exit-useful  : (随机×真实出场) edge > A edge + edge_sig
      - entry-useful : (真实×随机出场) edge > A edge + edge_sig
      - combo-useful : (真实×真实)     edge > A edge + edge_sig
    """
    labels = []
    if a is None or a["trades"] < 30:
        return "tiny-sample"

    if a["expectancy"] > 30:
        labels.append("beta-driven")

    a_edge = kelly_edge(a["win_rate"], a["wl_ratio"])

    def _useful(side):
        if side is None:
            return False
        return kelly_edge(side["win_rate"], side["wl_ratio"]) - a_edge > edge_sig

    has_exit = _useful(b)
    has_entry = _useful(c)
    has_combo = _useful(d)

    flags = []
    if has_entry:
        flags.append("entry-useful")
    if has_exit:
        flags.append("exit-useful")
    if has_combo:
        flags.append("combo-useful")
    if not flags:
        labels.append("dead-symbol")
    else:
        labels.extend(flags)
    return ";".join(labels)


def main() -> None:
    expA = load_random_exp("A")
    expB = load_random_exp("B")
    expC = load_random_exp("C")
    print(f"Loaded random experiments:  A={len(expA)}  B={len(expB)}  C={len(expC)}")
    if expA.empty or expB.empty or expC.empty:
        print("Missing per-symbol experiment data; run runner with --scope symbol first.")
        sys.exit(1)

    # Filter to TARGET_GROUPS only (defensive — symbol files might contain more)
    expA = expA[expA["group"].isin(TARGET_GROUPS)]
    expB = expB[expB["group"].isin(TARGET_GROUPS)]
    expC = expC[expC["group"].isin(TARGET_GROUPS)]
    target_syms = set(expA["symbol"].unique())
    print(f"Target groups: {TARGET_GROUPS},  syms = {sorted(target_syms)}")

    expD = load_real_real_per_symbol(target_syms)
    print(f"Loaded real×real (v4) baseline:  {len(expD)} rows for {expD['symbol'].nunique()} syms")

    # Aggregate
    A_agg = aggregate_random(expA, ["group", "symbol"])
    B_agg = aggregate_random(expB, ["group", "symbol", "exit"])
    C_agg = aggregate_random(expC, ["group", "symbol", "entry"])
    D_agg = aggregate_real_real(expD, ["group_name", "symbol", "combo", "entry", "exit"]) \
        if "group_name" in expD.columns else None

    # v4 CSV uses 'symbol' but no 'group' column — map back from current bars
    if "group_name" not in expD.columns:
        # Map symbol → group from expA (ground truth for our 22 syms)
        sym_to_group = dict(zip(expA["symbol"], expA["group"]))
        expD["group"] = expD["symbol"].map(sym_to_group)
        D_agg = aggregate_real_real(expD, ["group", "symbol", "combo", "entry", "exit"])

    # Best per symbol
    B_best = best_per_symbol(B_agg, "exit")
    C_best = best_per_symbol(C_agg, "entry")
    D_best = best_per_symbol(D_agg, "combo")

    # Build summary
    rows = []
    for _, a_row in A_agg.iterrows():
        sym, grp = a_row["symbol"], a_row["group"]
        b = B_best[(B_best["symbol"] == sym) & (B_best["group"] == grp)]
        c = C_best[(C_best["symbol"] == sym) & (C_best["group"] == grp)]
        d = D_best[(D_best["symbol"] == sym) & (D_best["group"] == grp)]
        b_row = b.iloc[0] if not b.empty else None
        c_row = c.iloc[0] if not c.empty else None
        d_row = d.iloc[0] if not d.empty else None
        verdict = label_symbol(a_row, b_row, c_row, d_row)
        rows.append({
            "group": grp,
            "symbol": sym,
            "A_trades": int(a_row["trades"]),
            "A_wr": round(a_row["win_rate"], 3),
            "A_wl": round(a_row["wl_ratio"], 2),
            "A_exp": round(a_row["expectancy"], 0),
            "best_B_exit": b_row["best_alpha"] if b_row is not None else None,
            "B_trades": b_row["trades"] if b_row is not None else 0,
            "B_wr": b_row["win_rate"] if b_row is not None else None,
            "B_wl": b_row["wl_ratio"] if b_row is not None else None,
            "B_exp": b_row["expectancy"] if b_row is not None else None,
            "best_C_entry": c_row["best_alpha"] if c_row is not None else None,
            "C_trades": c_row["trades"] if c_row is not None else 0,
            "C_wr": c_row["win_rate"] if c_row is not None else None,
            "C_wl": c_row["wl_ratio"] if c_row is not None else None,
            "C_exp": c_row["expectancy"] if c_row is not None else None,
            "D_best_combo": d_row["best_alpha"] if d_row is not None else None,
            "D_trades": d_row["trades"] if d_row is not None else 0,
            "D_wr": d_row["win_rate"] if d_row is not None else None,
            "D_wl": d_row["wl_ratio"] if d_row is not None else None,
            "D_exp": d_row["expectancy"] if d_row is not None else None,
            # All comparisons against A (随机×随机), the zero baseline:
            "B_minus_A_wl": (
                round(b_row["wl_ratio"] - a_row["wl_ratio"], 2) if b_row is not None else None
            ),
            "B_minus_A_wr": (
                round(b_row["win_rate"] - a_row["win_rate"], 3) if b_row is not None else None
            ),
            "C_minus_A_wl": (
                round(c_row["wl_ratio"] - a_row["wl_ratio"], 2) if c_row is not None else None
            ),
            "C_minus_A_wr": (
                round(c_row["win_rate"] - a_row["win_rate"], 3) if c_row is not None else None
            ),
            "D_minus_A_wl": (
                round(d_row["wl_ratio"] - a_row["wl_ratio"], 2) if d_row is not None else None
            ),
            "D_minus_A_wr": (
                round(d_row["win_rate"] - a_row["win_rate"], 3) if d_row is not None else None
            ),
            "verdict": verdict,
        })
    summary = pd.DataFrame(rows).sort_values(["group", "symbol"]).reset_index(drop=True)
    summary.to_csv(ABENCH / "symbol_summary.csv", index=False)

    print("\n===== Per-symbol summary (alpha-clear groups, 22 syms) =====")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(summary.to_string(index=False))

    # Verdict breakdown
    counts = []
    for v in summary["verdict"]:
        counts.extend(v.split(";"))
    print("\n===== Verdict counts =====")
    print(pd.Series(counts).value_counts().to_string())

    # Long-form: each (symbol, alpha-side) pair with Kelly edge vs A
    long_rows = []
    for _, r in summary.iterrows():
        for kind, wl_col, wr_col in [
            ("B_vs_A", "B_minus_A_wl", "B_minus_A_wr"),
            ("C_vs_A", "C_minus_A_wl", "C_minus_A_wr"),
            ("D_vs_A", "D_minus_A_wl", "D_minus_A_wr"),
        ]:
            long_rows.append({
                "group": r["group"], "symbol": r["symbol"], "kind": kind,
                "delta_wl": r[wl_col], "delta_wr": r[wr_col],
            })
    pd.DataFrame(long_rows).to_csv(ABENCH / "symbol_marginal.csv", index=False)
    print(f"\n[saved] {ABENCH}/symbol_summary.csv, symbol_marginal.csv")


if __name__ == "__main__":
    main()
