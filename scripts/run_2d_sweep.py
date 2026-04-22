"""2D parameter sweep: (risk_per_trade, portfolio_risk_cap) grid.

P0 follow-up to Step 5: the 1-D sweeps showed both knobs move Sharpe
significantly. This script does a 3x3 grid that brackets the best 1-D
points so we can see if they combine or interact.

Also reruns DSR at the best point with multiple n_trials hypotheses,
since the baseline DSR (n_trials=270) is the right number only for
Layer 1 combo selection — the parameter search we did was only ~9-28
trials, which is much less punitive.

Usage:
    python scripts/run_2d_sweep.py --suffix baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.research.report_baseline import derive_best_combos
from strats.engine import StrategyEngine, StrategySlot
from strats.factory import build_engine_config, build_entries, build_exits
from strats.helpers import PortfolioAnalyzer, deflated_sharpe
from strats.research_support import load_hab_bars


RISK_GRID = [0.005, 0.010, 0.015]
CAP_GRID  = [0.12, 0.16, 0.20]


def run_engine(bars, engine_cfg, entries, exits, best_combos):
    slots = [
        StrategySlot(
            strategy_id=f"{g}_{c}",
            entry_strategy=entries[c.split("+")[0]],
            exit_strategy=exits[c.split("+")[1]],
        )
        for g, c in best_combos.items()
    ]
    included = set(best_combos.keys())
    portfolio_bars = bars[bars["group_name"].isin(included)].copy()
    engine = StrategyEngine(config=engine_cfg, strategies=slots)
    return engine.run(portfolio_bars)


def interpret_dsr(dsr: float) -> str:
    if pd.isna(dsr):
        return "NaN"
    if dsr > 0.95:
        return "STRONG edge"
    if dsr > 0.80:
        return "MODERATE"
    if dsr > 0.50:
        return "WEAK"
    return "overfitting"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="baseline")
    args = parser.parse_args()

    run_name = args.suffix or "default"
    group_csv = ROOT / "data" / "runs" / run_name / "backtest_group_layer.csv"
    best_combos = derive_best_combos(group_csv)
    print(f"Best combos ({len(best_combos)} groups) loaded.\n")

    sens_dir = ROOT / "data" / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)
    best_point_dir = ROOT / "data" / "runs" / "best_point"
    best_point_dir.mkdir(parents=True, exist_ok=True)
    best_2d_dir = ROOT / "data" / "runs" / "best_2d"
    best_2d_dir.mkdir(parents=True, exist_ok=True)

    bars = load_hab_bars()
    entries = build_entries()
    exits = build_exits()

    rows: list[dict] = []
    for r in RISK_GRID:
        for cap in CAP_GRID:
            t0 = time.time()
            cfg = replace(
                build_engine_config(profile="research"),
                risk_per_trade=r,
                portfolio_risk_cap=cap,
            )
            try:
                result = run_engine(bars, cfg, entries, exits, best_combos)
                stats = PortfolioAnalyzer(result, cfg).summary_stats()
                pdf = result.portfolio_daily
                daily_ret = pdf["equity"].pct_change().dropna() if not pdf.empty else pd.Series(dtype=float)
                rows.append({
                    "risk_per_trade": r,
                    "portfolio_risk_cap": cap,
                    "sharpe": stats.get("sharpe", 0.0),
                    "cagr": stats.get("cagr", 0.0),
                    "max_dd": stats.get("max_drawdown_pct", 0.0),
                    "total_return": stats.get("total_return", 0.0),
                    "pf": stats.get("profit_factor", 0.0),
                    "trades": stats.get("total_trades", 0),
                    "_daily_returns": daily_ret,
                    "_portfolio_daily": pdf,
                })
                print(
                    f"  risk={r*100:5.2f}%  cap={cap*100:4.1f}%  "
                    f"Sharpe={stats.get('sharpe',0):+.3f}  "
                    f"CAGR={stats.get('cagr',0)*100:+.2f}%  "
                    f"MaxDD={stats.get('max_drawdown_pct',0)*100:+.2f}%  "
                    f"trades={stats.get('total_trades',0):4d}  "
                    f"PF={stats.get('profit_factor',0):.3f}  "
                    f"({time.time()-t0:.0f}s)"
                )
            except Exception as e:
                print(f"  risk={r*100:.2f}% cap={cap*100:.1f}% ERROR: {e}")

    if not rows:
        print("No results — aborting.")
        return

    # Save grid CSV (strip internal fields)
    df = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    out = sens_dir / "sensitivity_2d_risk_x_cap.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved grid: {out}")

    # Heatmap-style pivot for eyeballing
    print("\nSharpe heatmap (rows=risk_per_trade, cols=portfolio_risk_cap):")
    print(df.pivot(index="risk_per_trade", columns="portfolio_risk_cap", values="sharpe").round(3).to_string())
    print("\nCAGR heatmap:")
    print((df.pivot(index="risk_per_trade", columns="portfolio_risk_cap", values="cagr") * 100).round(2).to_string())
    print("\nMaxDD heatmap:")
    print((df.pivot(index="risk_per_trade", columns="portfolio_risk_cap", values="max_dd") * 100).round(2).to_string())

    # Identify best by Sharpe
    best_idx = int(df["sharpe"].idxmax())
    best = df.iloc[best_idx]
    best_row = rows[best_idx]
    print("\n" + "=" * 60)
    print(f"Best point: risk={best['risk_per_trade']*100:.2f}%  cap={best['portfolio_risk_cap']*100:.1f}%")
    print("=" * 60)
    print(f"  Sharpe  = {best['sharpe']:+.3f}")
    print(f"  CAGR    = {best['cagr']*100:+.2f}%")
    print(f"  MaxDD   = {best['max_dd']*100:+.2f}%")
    print(f"  PF      = {best['pf']:.3f}")
    print(f"  Trades  = {best['trades']}")

    # DSR at best point under multiple n_trials
    daily = best_row["_daily_returns"]
    print("\n=== DSR at best point (varying n_trials hypothesis) ===")
    dsr_results: dict[int, float] = {}
    for n in [9, 19, 28, 270, 298]:
        dsr = deflated_sharpe(daily, n)
        print(f"  n_trials={n:4d}  DSR={dsr:.4f}  [{interpret_dsr(dsr)}]")
        dsr_results[n] = float(dsr)

    # Save equity curve for best point
    pdf_best = best_row["_portfolio_daily"]
    eq_out = best_2d_dir / "backtest_portfolio_layer.csv"
    if not pdf_best.empty:
        # Recompute drawdown cols for the plot script
        pdf_best = pdf_best.sort_values("date").reset_index(drop=True)
        pdf_best["daily_return"] = pdf_best["equity"].pct_change()
        peak = pdf_best["equity"].cummax()
        pdf_best["drawdown_pct"] = (pdf_best["equity"] - peak) / peak.where(peak > 0)
        pdf_best.to_csv(eq_out, index=False)
        print(f"\nSaved best-point equity curve: {eq_out}")

    # Save JSON summary
    summary = {
        "best_risk_per_trade": float(best["risk_per_trade"]),
        "best_portfolio_risk_cap": float(best["portfolio_risk_cap"]),
        "sharpe": float(best["sharpe"]),
        "cagr": float(best["cagr"]),
        "max_dd_pct": float(best["max_dd"]),
        "total_return": float(best["total_return"]),
        "profit_factor": float(best["pf"]),
        "total_trades": int(best["trades"]),
        "dsr_by_n_trials": {str(k): v for k, v in dsr_results.items()},
    }
    bp_out = best_point_dir / "summary.json"
    bp_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {bp_out}")


if __name__ == "__main__":
    main()
