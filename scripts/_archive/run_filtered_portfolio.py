"""Run a portfolio of the 47 symbols that showed alpha in the random-benchmark
per-symbol analysis, using each group's historical best combo.

Dropped 13 'nothing-works' symbols (both entry-useful and exit-useful flags
false, via Kelly edge vs 随机入场×随机出场 baseline):
  agri:         A, B, CS
  black_steel:  J, RB
  building:     SH
  chem_energy:  EG, FU, PF, PP
  ind_CJ:       CJ       (whole group dropped — no other symbols)
  metals:       SS
  rubber_fiber: SR

Kept 47 syms across 12 groups (ind_CJ dropped entirely).

Risk params per user:
  risk_per_trade        = 0.03  (3% per trade)
  group_risk_cap        = 0.08  (8% uniform, replacing v2's 4-6% differentiated)
  portfolio_risk_cap    = 0.20  (20%)

Each kept group runs with its historically best combo from
data/runs/v2/backtest_group_layer.csv.

Output: data/runs/filtered47/  (backtest_portfolio_layer.csv + metrics.json)
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
from strats.helpers import PortfolioAnalyzer

from scripts.run_three_layer_backtest import (
    build_entries,
    build_exits,
    load_bars,
    make_engine_config,
)

OUT_DIR = ROOT / "data" / "runs" / "filtered47"

# 13 symbols where 随机入场×随机出场 was NOT meaningfully worse than any real
# alpha (Kelly edge delta < 0.05 for entry, exit, AND combo sides).
DROP_SYMS: Set[str] = {
    "A", "B", "CS",             # agri
    "J", "RB",                  # black_steel
    "SH",                       # building
    "EG", "FU", "PF", "PP",     # chem_energy
    "CJ",                       # ind_CJ (whole group)
    "SS",                       # metals
    "SR",                       # rubber_fiber
}


def load_group_best_combos() -> Dict[str, str]:
    """Pull each group's historical best combo from backtest_group_layer.csv.

    Groups dropped entirely (ind_CJ) are skipped at runtime via the bars filter.
    """
    df = pd.read_csv(ROOT / "data" / "runs" / "v2" / "backtest_group_layer.csv")
    return df.groupby("group")["best_combo"].first().to_dict()


def build_group_slots(
    best_combos: Dict[str, str],
    active_groups: Set[str],
    real_entries: Dict[str, Any],
    real_exits: Dict[str, Any],
) -> List[StrategySlot]:
    """One slot per active group, using that group's best combo."""
    slots: List[StrategySlot] = []
    for g, combo in best_combos.items():
        if g not in active_groups:
            continue
        entry_id, exit_id = combo.split("+")
        if entry_id not in real_entries or exit_id not in real_exits:
            continue
        slots.append(StrategySlot(
            strategy_id=f"{g}_{combo}",
            entry_strategy=real_entries[entry_id],
            exit_strategy=real_exits[exit_id],
        ))
    return slots


def per_group_pnl(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    sym_to_group = bars.groupby("symbol")["group_name"].first().to_dict()
    t = trades.copy()
    t["group"] = t["symbol"].map(sym_to_group)
    rows = []
    for g, sub in t.groupby("group"):
        wins = sub[sub["net_pnl"] > 0]
        losses = sub[sub["net_pnl"] <= 0]
        avg_w = float(wins["net_pnl"].mean()) if len(wins) > 0 else 0.0
        avg_l = float(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
        n = len(sub)
        rows.append({
            "group": g,
            "trades": n,
            "net_pnl": round(float(sub["net_pnl"].sum()), 0),
            "expectancy": round(float(sub["net_pnl"].sum()) / n, 1),
            "win_rate": round(len(wins) / n, 3),
            "wl_ratio": round(-avg_w / avg_l, 3) if avg_l < 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = load_bars()

    # Exclude v3b-dropped ind_* and structural junk groups
    from data.adapters.futures_static_meta import EXCLUDED_SYMBOLS
    bars = bars[~bars["symbol"].isin(EXCLUDED_SYMBOLS)]
    exc_groups = {"commodity", "index", "bond", "equity_index",
                  "ind_AP", "ind_BB", "ind_JR", "ind_LR", "ind_PM", "ind_RI"}
    bars = bars[~bars["group_name"].isin(exc_groups)]

    # Now apply the 13-sym dead-list filter
    before = bars["symbol"].nunique()
    bars = bars[~bars["symbol"].isin(DROP_SYMS)].copy()
    after = bars["symbol"].nunique()
    print(f"Symbol filter: {before} → {after} ({before - after} dropped)")
    print(f"Dropped: {sorted(DROP_SYMS)}")
    active_groups = set(bars["group_name"].unique())
    print(f"Active groups: {len(active_groups)} ({sorted(active_groups)})")

    # Build engine config with user's risk params
    cap = 0.08
    engine_cfg = make_engine_config(risk_overrides={
        "risk_per_trade": 0.03,
        "portfolio_risk_cap": 0.20,
        "group_risk_cap": {g: cap for g in active_groups},
    })
    print(f"\nEngine: risk_per_trade=0.03, group_risk_cap=0.08 (uniform), "
          f"portfolio_risk_cap=0.20")

    best_combos = load_group_best_combos()
    print(f"\nPer-group best combos (v2 archive):")
    for g in sorted(active_groups):
        if g in best_combos:
            print(f"  {g:15s} {best_combos[g]}")

    real_entries = build_entries(include_adaptive=False)
    real_exits = build_exits()
    slots = build_group_slots(best_combos, active_groups, real_entries, real_exits)
    print(f"\nSlots: {len(slots)}")

    t0 = time.time()
    engine = StrategyEngine(config=engine_cfg, strategies=slots)
    result = engine.run(bars)
    elapsed = time.time() - t0
    print(f"\nEngine run: {elapsed:.0f}s, trades={len(result.trades)}")

    # Portfolio-level metrics (we're NOT looking at Sharpe per user guidance;
    # still include it for the CSV but don't feature it in headline)
    analyzer = PortfolioAnalyzer(result, engine_cfg)
    stats = analyzer.summary_stats()

    trades = result.trades
    if not trades.empty:
        n = len(trades)
        net = float(trades["net_pnl"].sum())
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] <= 0]
        avg_w = float(wins["net_pnl"].mean()) if len(wins) > 0 else 0.0
        avg_l = float(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
        headline = {
            "total_trades": n,
            "total_net_pnl": round(net, 0),
            "expectancy": round(net / n, 1),
            "win_rate": round(len(wins) / n, 3),
            "wl_ratio": round(-avg_w / avg_l, 3) if avg_l < 0 else 0.0,
            "avg_winner": round(avg_w, 0),
            "avg_loser": round(avg_l, 0),
            "max_drawdown_pct": round(stats.get("max_drawdown_pct", 0.0), 3),
            "total_return": round(stats.get("total_return", 0.0), 3),
            "profit_factor": round(stats.get("profit_factor", 0.0), 2),
            "cagr": round(stats.get("cagr", 0.0), 3),
        }
    else:
        headline = {}

    print("\n===== Portfolio headline (NO Sharpe focus) =====")
    for k, v in headline.items():
        print(f"  {k:20s} {v}")

    pg = per_group_pnl(trades, bars)
    if not pg.empty:
        print("\n===== Per-group PnL breakdown =====")
        pd.set_option("display.width", 180)
        print(pg.to_string(index=False))
        pg.to_csv(OUT_DIR / "per_group.csv", index=False)

    # Save equity curve
    if not result.portfolio_daily.empty:
        eq = result.portfolio_daily.copy().sort_values("date")
        eq["daily_return"] = eq["equity"].pct_change()
        peak = eq["equity"].cummax()
        eq["drawdown_pct"] = (eq["equity"] - peak) / peak.where(peak > 0, np.nan)
        eq.to_csv(OUT_DIR / "backtest_portfolio_layer.csv", index=False)

    # Save metrics + metadata
    metadata = {
        "active_groups": sorted(active_groups),
        "n_symbols": int(bars["symbol"].nunique()),
        "dropped_symbols": sorted(DROP_SYMS),
        "risk_per_trade": 0.03,
        "group_risk_cap": 0.08,
        "portfolio_risk_cap": 0.20,
        "best_combos": {g: best_combos[g] for g in sorted(active_groups) if g in best_combos},
        "headline": headline,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n[saved] {OUT_DIR}/backtest_portfolio_layer.csv, per_group.csv, metrics.json")


if __name__ == "__main__":
    main()
