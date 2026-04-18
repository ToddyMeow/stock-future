"""Overfitting diagnostics — DSR + PBO (Step 7 of the backtest plan).

**DSR (Deflated Sharpe Ratio)**: Bailey & López de Prado 2014. Given
n_trials strategy variants were evaluated and the best selected, DSR is
the probability the selected variant's TRUE Sharpe > 0 after multi-testing
correction. Reads the baseline portfolio equity CSV directly — fast.

**PBO (Probability of Backtest Overfitting)**: Bailey-Borwein-López-Zhu 2014
via CSCV. Requires per-combo daily returns. We rerun the full 30 × 9 = 270
grid to build a (T, N) returns matrix, then call `pbo_cscv`. Skippable
with --skip-pbo for a fast DSR-only run.

Usage:
    python scripts/diagnose_overfitting.py --suffix baseline --skip-pbo   # ~5s
    python scripts/diagnose_overfitting.py --suffix baseline              # ~15-25 min
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_three_layer_backtest import (
    GROUPS, build_entries, build_exits, filter_group, load_bars, make_engine_config,
)
from strats.engine import StrategyEngine, StrategySlot
from strats.helpers import deflated_sharpe, pbo_cscv


def compute_dsr(portfolio_csv: Path, n_trials: int) -> float:
    df = pd.read_csv(portfolio_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    returns = df["equity"].pct_change().dropna()
    return deflated_sharpe(returns, n_trials)


def build_returns_matrix(bars, engine_cfg, entries, exits) -> pd.DataFrame:
    """Rerun all 35 x 9 combos. Return a (T, N) DataFrame of daily returns.

    Key = "<group>|<entry>+<exit>"; columns aligned on date; missing = 0.
    """
    combos = [(eid, xid) for eid in entries for xid in exits]
    series: dict[str, pd.Series] = {}
    total = len(GROUPS) * len(combos)
    done = 0
    for group in GROUPS:
        group_bars = filter_group(bars, group)
        if group_bars.empty:
            continue
        for eid, xid in combos:
            done += 1
            key = f"{group}|{eid}+{xid}"
            t0 = time.time()
            try:
                slot = StrategySlot(
                    strategy_id=f"{eid}+{xid}",
                    entry_strategy=entries[eid],
                    exit_strategy=exits[xid],
                )
                engine = StrategyEngine(config=engine_cfg, strategies=[slot])
                result = engine.run(group_bars)
                pdf = result.portfolio_daily
                if pdf.empty:
                    continue
                r = pdf.set_index("date")["equity"].pct_change().dropna()
                if len(r) >= 30:
                    series[key] = r
                print(f"  [{done}/{total}] {key:38s}  T={len(r)}  ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  [{done}/{total}] {key} — ERROR: {e}")
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).fillna(0.0)


def interpret_dsr(dsr: float) -> str:
    if pd.isna(dsr):
        return "INCONCLUSIVE (too few samples)"
    if dsr > 0.95:
        return "STRONG evidence of edge"
    if dsr > 0.80:
        return "MODERATE evidence"
    if dsr > 0.50:
        return "WEAK / inconclusive"
    return "WARNING: indistinguishable from overfitting"


def interpret_pbo(pbo: float) -> str:
    if pd.isna(pbo):
        return "INCONCLUSIVE"
    if pbo < 0.25:
        return "HEALTHY (low overfitting probability)"
    if pbo < 0.50:
        return "MODERATE overfitting probability"
    return "WARNING: severe overfitting"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="baseline")
    parser.add_argument("--n-trials", type=int, default=5 * 6 * 9,
                        help="Number of variants explored (5 entries x 6 exits x 9 groups = 270).")
    parser.add_argument("--skip-pbo", action="store_true",
                        help="Skip PBO (DSR only; ~5 sec).")
    parser.add_argument("--n-splits", type=int, default=16,
                        help="CSCV splits for PBO (must be even, default 16).")
    parser.add_argument("--adx-off", action="store_true",
                        help="Mirror --adx-off of the baseline run when reconstructing.")
    args = parser.parse_args()

    file_suffix = f"_{args.suffix}" if args.suffix else ""
    portfolio_csv = ROOT / "data" / f"backtest_portfolio_layer{file_suffix}.csv"
    if not portfolio_csv.exists():
        print(f"ERROR: {portfolio_csv} not found — run baseline first", file=sys.stderr)
        sys.exit(1)

    # DSR (fast)
    print("=" * 60)
    print("DSR — Deflated Sharpe Ratio")
    print("=" * 60)
    dsr = compute_dsr(portfolio_csv, args.n_trials)
    print(f"  DSR      = {dsr:.4f}")
    print(f"  n_trials = {args.n_trials}")
    print(f"  verdict  = {interpret_dsr(dsr)}")

    results: dict = {"dsr": float(dsr), "n_trials": args.n_trials}

    # PBO (slow)
    if args.skip_pbo:
        print("\n[PBO skipped — --skip-pbo]")
    else:
        print()
        print("=" * 60)
        print(f"PBO — rerunning {args.n_trials} combos for daily returns")
        print("=" * 60)
        bars = load_bars()
        engine_cfg = make_engine_config(adx_off=args.adx_off)
        entries = build_entries()
        exits = build_exits()
        t0 = time.time()
        matrix = build_returns_matrix(bars, engine_cfg, entries, exits)
        if matrix.empty:
            print("  No valid returns — aborting PBO.")
            pbo_val = float("nan")
        else:
            mat_out = ROOT / "data" / f"combo_returns_matrix{file_suffix}.csv"
            matrix.to_csv(mat_out)
            print(f"  Saved returns matrix: {mat_out}  shape={matrix.shape}")
            pbo_val = pbo_cscv(matrix, n_splits=args.n_splits)
        print(f"\n  PBO      = {pbo_val:.4f}")
        print(f"  n_splits = {args.n_splits}")
        print(f"  verdict  = {interpret_pbo(pbo_val)}")
        print(f"  elapsed  = {time.time()-t0:.0f}s")
        results.update({
            "pbo": float(pbo_val),
            "n_splits": args.n_splits,
            "n_variants": int(matrix.shape[1]) if not matrix.empty else 0,
        })

    out = ROOT / "data" / f"overfitting_diagnostics{file_suffix}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
