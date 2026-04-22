"""Random-benchmark runner: four-experiment alpha attribution at group level.

Experiments (all pair rand_entry with allow_short=True for unbiased zero line):
  A: rand × rand            — zero baseline (drift + costs)
  B: rand × {real exits}    — exit alpha
  C: {real entries} × rand  — entry alpha
  D: {real entries} × {real exits} — combined alpha (30 combos)

Phase 1 granularity: group (19 groups = 7 main + 12 ind_*).
Phase 2 (not in this runner) drills per-symbol on alpha-clear groups.

Metric: Expectancy = net_pnl / trades (per-trade, frequency-invariant),
which is the right comparison unit — random entry trades much more often
than real entries, so Sharpe/CAGR are not comparable across experiments.

Usage:
  python scripts/run_random_benchmark.py --experiment A --seeds 42,43,44
  python scripts/run_random_benchmark.py --experiment B --seeds 42,43,44
  python scripts/run_random_benchmark.py --experiment C --seeds 42,43,44
  python scripts/run_random_benchmark.py --experiment A --groups chem_energy --seeds 42    # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strats.engine import StrategyEngine, StrategySlot
from strats.entries.rand_entry import RandEntryConfig, RandEntryStrategy
from strats.exits.rand_exit import RandExitConfig, RandExitStrategy
from strats.factory import build_engine_config, build_entries, build_exits
from strats.research_support import (
    ALL_GROUPS,
    filter_group_bars as filter_group,
    load_hab_bars,
    yearly_stats_from_trades,
)


OUT_DIR = ROOT / "data" / "runs" / "alpha_benchmark"


def build_rand_entry(seed: int) -> Dict[str, Any]:
    return {"rand": RandEntryStrategy(RandEntryConfig(seed=seed, allow_short=True))}


def build_rand_exit(seed: int, exit_prob: float = 0.1) -> Dict[str, Any]:
    return {
        "rand": RandExitStrategy(
            RandExitConfig(exit_probability=exit_prob, min_bars=1, seed=seed)
        )
    }


def run_one(
    bars: pd.DataFrame,
    engine_cfg,
    entry_id: str,
    entry_strategy: Any,
    exit_id: str,
    exit_strategy: Any,
    group: str,
    seed: int,
    symbol: str = "ALL",
) -> tuple:
    """Returns (aggregated_rows, per_trade_rows). per_trade_rows has tags
    (group, symbol, entry, exit, seed) + net_pnl + entry_date per trade,
    for downstream bootstrap significance testing."""
    slot = StrategySlot(
        strategy_id=f"{entry_id}+{exit_id}",
        entry_strategy=entry_strategy,
        exit_strategy=exit_strategy,
    )
    engine = StrategyEngine(config=engine_cfg, strategies=[slot])
    result = engine.run(bars)
    yr_stats = yearly_stats_from_trades(
        result.trades, result.portfolio_daily, engine_cfg.initial_capital
    )

    # Pre-compute per-year wins/losses gross sums for richer Phase-2 metrics
    extra_per_year: Dict[int, Dict[str, float]] = {}
    per_trade_rows: List[Dict[str, Any]] = []
    if not result.trades.empty:
        td = result.trades.copy()
        td["year"] = pd.to_datetime(td["entry_date"]).dt.year
        for yr_val, sub in td.groupby("year"):
            wins = sub[sub["net_pnl"] > 0]
            losses = sub[sub["net_pnl"] <= 0]
            extra_per_year[int(yr_val)] = {
                "wins": int(len(wins)),
                "losses": int(len(losses)),
                "gross_win": float(wins["net_pnl"].sum()),
                "gross_loss": float(losses["net_pnl"].sum()),
            }
        # Rich per-trade record. Keep actual trade symbol (may differ from scope
        # unit tag for group-scope runs with multi-sym groups).
        keep_cols = [
            "symbol", "direction", "signal_date", "entry_date", "exit_date",
            "entry_fill", "exit_fill", "qty", "exit_reason",
            "mfe", "mae", "r_multiple",
            "gross_pnl", "net_pnl",
            "entry_commission_total", "exit_commission_total",
        ]
        sub_td = td[[c for c in keep_cols if c in td.columns]].copy()
        sub_td["group"] = group
        sub_td["scope_unit"] = symbol  # "ALL" for group scope, sym for symbol scope
        sub_td["entry"] = entry_id
        sub_td["exit"] = exit_id
        sub_td["seed"] = seed
        per_trade_rows.extend(sub_td.to_dict("records"))

    rows: List[Dict[str, Any]] = []
    if not yr_stats:
        rows.append(_empty_row(group, entry_id, exit_id, seed, year=0, symbol=symbol))
    else:
        for yr in yr_stats:
            extra = extra_per_year.get(int(yr["year"]), {
                "wins": 0, "losses": 0, "gross_win": 0.0, "gross_loss": 0.0,
            })
            rows.append({
                "group": group,
                "symbol": symbol,
                "entry": entry_id,
                "exit": exit_id,
                "seed": seed,
                **yr,
                **extra,
                "expectancy": (yr["net_pnl"] / yr["trades"]) if yr["trades"] else 0.0,
            })
    return rows, per_trade_rows


def _empty_row(
    group: str, entry_id: str, exit_id: str, seed: int, year: int,
    symbol: str = "ALL",
) -> Dict[str, Any]:
    return {
        "group": group, "symbol": symbol, "entry": entry_id, "exit": exit_id,
        "seed": seed, "year": year, "trades": 0, "sharpe": 0.0, "cagr": 0.0,
        "profit_factor": 0.0, "win_rate": 0.0, "avg_r": 0.0, "max_dd_pct": 0.0,
        "long_trades": 0, "short_trades": 0, "net_pnl": 0.0, "exit_reasons": "{}",
        "wins": 0, "losses": 0, "gross_win": 0.0, "gross_loss": 0.0,
        "expectancy": 0.0,
    }


def enumerate_combos(experiment: str, seed: int, exit_prob: float = 0.10):
    """Yield (entry_id, entry, exit_id, exit) combos for a given experiment+seed."""
    if experiment == "A":
        entries = build_rand_entry(seed)
        exits = build_rand_exit(seed, exit_prob=exit_prob)
    elif experiment == "B":
        entries = build_rand_entry(seed)
        exits = build_exits()
    elif experiment == "C":
        entries = build_entries(include_adaptive=False)
        exits = build_rand_exit(seed, exit_prob=exit_prob)
    elif experiment == "D":
        entries = build_entries(include_adaptive=False)
        exits = build_exits()
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    for eid, estr in entries.items():
        for xid, xstr in exits.items():
            yield eid, estr, xid, xstr


def loose_cap_engine_config():
    """Phase 1 config: real qty (risk=3%) but no gating from caps or ADX.
    All rejection gates wide-open so every signal trades its target qty.
    """
    known_groups = [
        "equity_index", "bond", "chem_energy", "rubber_fiber", "metals",
        "black_steel", "agri", "building", "livestock", "commodity", "index",
    ]
    return build_engine_config(profile="research", overrides={
        "risk_per_trade": 0.03,
        "portfolio_risk_cap": 10.0,
        "group_risk_cap": {g: 10.0 for g in known_groups},
        "default_group_risk_cap": 10.0,
        "independent_group_soft_cap": 10.0,
        "max_portfolio_leverage": 100.0,
        "adx_floor": 1.0,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--loose-cap", action="store_true",
                    help="Use Phase 1 loose-cap config (no gating, real qty at 3% risk, ADX off)")
    ap.add_argument("--seeds", default="42,43,44",
                    help="Comma-separated seeds (default: 42,43,44)")
    ap.add_argument("--groups", default=None,
                    help="Comma-separated groups (default: all ALL_GROUPS)")
    ap.add_argument("--scope", default="group", choices=["group", "symbol"],
                    help="group: aggregate within group; symbol: one engine run per symbol")
    ap.add_argument("--output-tag", default="",
                    help="Suffix for output file (e.g. '_remaining' → exp_A_symbol_remaining.csv)")
    ap.add_argument("--exit-prob", type=float, default=0.10,
                    help="Probability of rand exit firing per bar (default 0.10). "
                         "Only affects experiments A and C (which use rand exit).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    groups = (
        [g.strip() for g in args.groups.split(",")] if args.groups else list(ALL_GROUPS)
    )

    print(f"=== Random benchmark: experiment {args.experiment} ===")
    print(f"  scope = {args.scope}")
    print(f"  seeds = {seeds}")
    print(f"  groups = {groups}")

    bars = load_hab_bars()
    engine_cfg = (
        loose_cap_engine_config()
        if args.loose_cap
        else build_engine_config(profile="research")
    )
    print(f"  engine: risk={engine_cfg.risk_per_trade}, "
          f"portfolio_cap={engine_cfg.portfolio_risk_cap}, "
          f"default_group_cap={engine_cfg.default_group_risk_cap}, "
          f"adx_floor={engine_cfg.adx_floor}")

    # Build scope units: group → list of (unit_id, unit_bars, group_name, symbol_name)
    units: List[tuple] = []
    for group in groups:
        gb = filter_group(bars, group)
        if gb.empty:
            print(f"  [{group}] SKIP — no data")
            continue
        if args.scope == "group":
            units.append((group, gb, group, "ALL"))
        else:
            for sym in sorted(gb["symbol"].unique()):
                sb = gb[gb["symbol"] == sym]
                units.append((f"{group}/{sym}", sb, group, sym))
    print(f"  units = {len(units)} ({args.scope} scope)")

    all_rows: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []
    t0 = time.time()

    n_combos_by_exp = {"A": 1, "B": 6, "C": 5, "D": 30}
    # D is deterministic — collapse seeds to a single pass
    seeds_for_exp = [seeds[0]] if args.experiment == "D" else seeds
    total = len(units) * n_combos_by_exp[args.experiment] * len(seeds_for_exp)
    done = 0

    for seed in seeds_for_exp:
        for unit_tag, unit_bars, group, symbol in units:
            for eid, estr, xid, xstr in enumerate_combos(args.experiment, seed, args.exit_prob):
                done += 1
                tag = f"[{done}/{total}] exp={args.experiment} seed={seed} {unit_tag} {eid}+{xid}"
                try:
                    rows, trade_rows = run_one(
                        unit_bars, engine_cfg, eid, estr, xid, xstr,
                        group=group, seed=seed, symbol=symbol,
                    )
                    all_rows.extend(rows)
                    all_trades.extend(trade_rows)
                    n_tr = sum(r["trades"] for r in rows)
                    print(f"  {tag} — {n_tr} trades, {len(rows)} year-rows")
                except Exception as e:
                    print(f"  {tag} — ERROR: {e}")
                    traceback.print_exc()
                    all_rows.append(_empty_row(group, eid, xid, seed, 0, symbol=symbol))

    df = pd.DataFrame(all_rows)
    cols = [
        "group", "symbol", "entry", "exit", "seed", "year", "trades", "net_pnl", "expectancy",
        "wins", "losses", "gross_win", "gross_loss",
        "sharpe", "cagr", "profit_factor", "win_rate", "avg_r", "max_dd_pct",
        "long_trades", "short_trades", "exit_reasons",
    ]
    df = df[[c for c in cols if c in df.columns]]
    suffix = "_symbol" if args.scope == "symbol" else ""
    out_path = OUT_DIR / f"exp_{args.experiment}{suffix}{args.output_tag}.csv"
    df.to_csv(out_path, index=False)

    # Save per-trade dump for bootstrap significance testing
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_path = OUT_DIR / f"trades_{args.experiment}{suffix}{args.output_tag}.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"Saved trades {trades_path}  ({len(trades_df)} trade rows)")

    elapsed = time.time() - t0
    print(f"\nSaved {out_path}  ({len(df)} rows, {elapsed:.0f}s / {elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
