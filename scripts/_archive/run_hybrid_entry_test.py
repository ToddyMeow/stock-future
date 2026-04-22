"""Hybrid portfolio test: replace real entry with rand_entry on 8
exit-marginal-only symbols, keep the rest as v2. Compare metrics.

The 8 symbols (from Phase 2 per-symbol drill): J, JM, RB, SF (black_steel),
AO, FG (building), CF, RU (rubber_fiber). On these, Phase 2 showed
真实入场×真实出场 − 随机入场×真实出场 ≤ 0.10 wl_ratio, i.e. the real entry
contributes nothing marginal over best-exit-alone.

This script builds two 60-symbol portfolios:
  - baseline: per-symbol slots with v2 group-best (entry, exit) each
  - hybrid:   same, but rand_entry replaces the real entry on the 8 syms

Run the engine on each. Report per-trade, NOT Sharpe/CAGR (per user's
explicit instruction: "不要看 Sharpe 不要看 Sharpe").

Metrics reported:
  - total_trades
  - total_net_pnl
  - expectancy (net_pnl / trades)
  - win_rate
  - wl_ratio (avg_winner / |avg_loser|)
  - max_drawdown_pct
  - per-group net_pnl breakdown
  - diff between baseline and hybrid

Output:
  data/runs/alpha_benchmark/hybrid_entry_test.csv
  data/runs/alpha_benchmark/hybrid_per_group.csv

Usage:
  python scripts/run_hybrid_entry_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strats.engine import StrategyEngine, StrategySlot
from strats.entries.rand_entry import RandEntryConfig, RandEntryStrategy

from scripts.run_three_layer_backtest import (
    build_entries,
    build_exits,
    load_bars,
    make_engine_config,
    ALL_GROUPS,
)
from scripts.run_per_symbol_backtest import SymbolFilteredEntry


OUT_DIR = ROOT / "data" / "runs" / "alpha_benchmark"

# 8 exit-marginal-only symbols from Phase 2 per-symbol drill
RAND_ENTRY_SYMS: Set[str] = {
    "J", "JM", "RB", "SF",   # black_steel
    "AO", "FG",              # building
    "CF", "RU",              # rubber_fiber
}


def load_v2_best_combos() -> Dict[str, str]:
    """Load v2's group → best_combo map."""
    df = pd.read_csv(ROOT / "data" / "runs" / "v2" / "backtest_group_layer.csv")
    return df.groupby("group")["best_combo"].first().to_dict()


def build_slots(
    bars: pd.DataFrame,
    best_combos: Dict[str, str],
    rand_syms: Set[str],
    real_entries: Dict[str, Any],
    real_exits: Dict[str, Any],
    seed: int = 42,
) -> List[StrategySlot]:
    """Per-symbol slot list. Entry swapped to rand for syms in rand_syms."""
    # Map symbol → group
    sym_to_group = bars.groupby("symbol")["group_name"].first().to_dict()

    slots: List[StrategySlot] = []
    rand_cfg = RandEntryConfig(seed=seed, allow_short=True)
    for sym, grp in sym_to_group.items():
        if grp not in best_combos:
            continue
        entry_id, exit_id = best_combos[grp].split("+")
        if entry_id not in real_entries or exit_id not in real_exits:
            continue

        if sym in rand_syms:
            entry_strategy = RandEntryStrategy(rand_cfg)
            slot_entry_id = "rand"
        else:
            entry_strategy = real_entries[entry_id]
            slot_entry_id = entry_id

        slot = StrategySlot(
            strategy_id=f"{sym}:{slot_entry_id}+{exit_id}",
            entry_strategy=SymbolFilteredEntry(entry_strategy, sym),
            exit_strategy=real_exits[exit_id],
        )
        slots.append(slot)
    return slots


def compute_metrics(trades: pd.DataFrame, portfolio_daily: pd.DataFrame) -> Dict[str, Any]:
    """Skip Sharpe and CAGR. Report per-trade and drawdown."""
    if trades.empty:
        return {
            "total_trades": 0, "total_net_pnl": 0.0, "expectancy": 0.0,
            "win_rate": 0.0, "wl_ratio": 0.0, "max_drawdown_pct": 0.0,
        }
    n = len(trades)
    net = float(trades["net_pnl"].sum())
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    avg_winner = float(wins["net_pnl"].mean()) if len(wins) > 0 else 0.0
    avg_loser = float(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
    wl_ratio = (-avg_winner / avg_loser) if avg_loser < 0 else 0.0
    win_rate = len(wins) / n

    # Max drawdown from equity curve
    if not portfolio_daily.empty:
        eq = portfolio_daily["equity"]
        peak = eq.cummax()
        dd_pct = (eq - peak) / peak.where(peak > 0, np.nan)
        max_dd_pct = float(dd_pct.min()) if not dd_pct.isna().all() else 0.0
    else:
        max_dd_pct = 0.0

    return {
        "total_trades": n,
        "total_net_pnl": round(net, 0),
        "expectancy": round(net / n, 1),
        "win_rate": round(win_rate, 3),
        "wl_ratio": round(wl_ratio, 3),
        "avg_winner": round(avg_winner, 0),
        "avg_loser": round(avg_loser, 0),
        "max_drawdown_pct": round(max_dd_pct, 3),
    }


def per_group_breakdown(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Per-group PnL, trades, expectancy, win_rate, wl_ratio."""
    if trades.empty:
        return pd.DataFrame()
    sym_to_group = bars.groupby("symbol")["group_name"].first().to_dict()
    trades = trades.copy()
    trades["group"] = trades["symbol"].map(sym_to_group)

    rows = []
    for g, sub in trades.groupby("group"):
        n = len(sub)
        wins = sub[sub["net_pnl"] > 0]
        losses = sub[sub["net_pnl"] <= 0]
        avg_w = float(wins["net_pnl"].mean()) if len(wins) > 0 else 0.0
        avg_l = float(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
        rows.append({
            "group": g,
            "trades": n,
            "net_pnl": round(float(sub["net_pnl"].sum()), 0),
            "expectancy": round(float(sub["net_pnl"].sum()) / n, 1),
            "win_rate": round(len(wins) / n, 3),
            "wl_ratio": round(-avg_w / avg_l, 3) if avg_l < 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = load_bars()

    # Exclude groups v2 dropped (ind_AP/BB/JR/LR/PM/RI + commodity/index/bond)
    from data.adapters.futures_static_meta import EXCLUDED_SYMBOLS
    bars = bars[~bars["symbol"].isin(EXCLUDED_SYMBOLS)]
    excluded_groups = {"commodity", "index", "bond", "equity_index",
                       "ind_AP", "ind_BB", "ind_JR", "ind_LR", "ind_PM", "ind_RI"}
    bars = bars[~bars["group_name"].isin(excluded_groups)]
    print(f"Portfolio: {bars['symbol'].nunique()} syms, "
          f"{bars['group_name'].nunique()} groups")

    best_combos = load_v2_best_combos()
    engine_cfg = make_engine_config()
    real_entries = build_entries(include_adaptive=False)
    real_exits = build_exits()

    print(f"\nRand-entry swap set: {sorted(RAND_ENTRY_SYMS)}")

    results = {}
    per_group = {}

    for variant, rand_set in [("baseline", set()), ("hybrid", RAND_ENTRY_SYMS)]:
        print(f"\n=== Running {variant} ===")
        t0 = time.time()
        slots = build_slots(bars, best_combos, rand_set, real_entries, real_exits)
        print(f"  slots: {len(slots)}")
        engine = StrategyEngine(config=engine_cfg, strategies=slots)
        result = engine.run(bars)
        elapsed = time.time() - t0
        print(f"  done ({elapsed:.0f}s)  trades={len(result.trades)}")
        results[variant] = compute_metrics(result.trades, result.portfolio_daily)
        per_group[variant] = per_group_breakdown(result.trades, bars)

    # Build comparison table
    cmp_rows = []
    for k in results["baseline"]:
        v_base = results["baseline"][k]
        v_hyb = results["hybrid"][k]
        delta = (v_hyb - v_base) if isinstance(v_base, (int, float)) else None
        cmp_rows.append({
            "metric": k,
            "baseline": v_base,
            "hybrid": v_hyb,
            "delta": round(delta, 1) if delta is not None else None,
        })
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT_DIR / "hybrid_entry_test.csv", index=False)

    print("\n===== Portfolio comparison (NO Sharpe, NO CAGR) =====")
    pd.set_option("display.width", 180)
    print(cmp_df.to_string(index=False))

    # Per-group breakdown side-by-side
    if not per_group["baseline"].empty and not per_group["hybrid"].empty:
        base = per_group["baseline"].rename(
            columns={c: f"base_{c}" for c in per_group["baseline"].columns if c != "group"}
        )
        hyb = per_group["hybrid"].rename(
            columns={c: f"hyb_{c}" for c in per_group["hybrid"].columns if c != "group"}
        )
        merged = base.merge(hyb, on="group", how="outer")
        for metric in ["trades", "net_pnl", "expectancy", "win_rate", "wl_ratio"]:
            merged[f"delta_{metric}"] = merged[f"hyb_{metric}"] - merged[f"base_{metric}"]
        # Highlight the 3 groups with rand-swap symbols
        targeted_groups = {"black_steel", "building", "rubber_fiber"}
        merged["has_rand_syms"] = merged["group"].isin(targeted_groups)
        merged = merged.sort_values(["has_rand_syms", "group"], ascending=[False, True])
        merged.to_csv(OUT_DIR / "hybrid_per_group.csv", index=False)

        print("\n===== Per-group delta (hybrid − baseline), rand-syms groups on top =====")
        cols_show = ["group", "has_rand_syms",
                     "base_trades", "hyb_trades", "delta_trades",
                     "base_net_pnl", "hyb_net_pnl", "delta_net_pnl",
                     "base_expectancy", "hyb_expectancy", "delta_expectancy",
                     "base_win_rate", "hyb_win_rate", "delta_win_rate",
                     "base_wl_ratio", "hyb_wl_ratio", "delta_wl_ratio"]
        print(merged[cols_show].to_string(index=False))

    # Persist summary for downstream reference
    (OUT_DIR / "hybrid_summary.json").write_text(json.dumps({
        "rand_entry_syms": sorted(RAND_ENTRY_SYMS),
        "baseline": results["baseline"],
        "hybrid": results["hybrid"],
    }, indent=2))
    print(f"\n[saved] {OUT_DIR}/hybrid_entry_test.csv, hybrid_per_group.csv, hybrid_summary.json")


if __name__ == "__main__":
    main()
