"""v4 per-symbol + top-3 backtest (no group cap, first-fire-wins optional).

Architecture:
  - Each SYMBOL is a standalone unit (groups collapsed away).
  - For each symbol, Layer 1 runs all 30 entry×exit combos on single-symbol bars.
  - Layer 2 picks top-3 combos per symbol by composite rank (Sharpe + PF + MaxDD).
  - Layer 3 Portfolio: build 3 × N_symbols slots, each wrapped with
    SymbolFilteredEntry so it only fires on its target symbol.
    - variant "v4a": symbol_position_lock=True (first-fire wins per symbol)
    - variant "v4b": symbol_position_lock=False (up to 3 parallel positions)

Engine flags:
  - use_group_risk_cap=False (group cap fully disabled)
  - portfolio_risk_cap still enforced (20% default)

Usage:
  python scripts/run_per_symbol_backtest.py --suffix v4 \\
      --risk-per-trade 0.03 --portfolio-risk-cap 0.20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strats.engine import EngineConfig, StrategyEngine, StrategySlot
from strats.factory import build_engine_config, build_entries, build_exits
from strats.helpers import PortfolioAnalyzer, deflated_sharpe
from strats.research_support import load_hab_bars, yearly_stats_from_trades


# ---------- Symbol-filtered wrapper ----------


class SymbolFilteredEntry:
    """Delegate to inner entry, but only fire signals on the target symbol.

    When prepare_signals is called with bars for a different symbol, return
    the DataFrame with entry_trigger_pass=False everywhere (skip inner compute).
    This keeps the engine efficient when 180 slots each only care about 1/60 symbols.
    """

    def __init__(self, inner: Any, target_symbol: str) -> None:
        self.inner = inner
        self.target_symbol = target_symbol

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return self.inner.prepare_signals(df)
        # DataFrame received is single-symbol (engine groups by symbol before calling).
        sym = df["symbol"].iloc[0]
        if sym != self.target_symbol:
            return df.assign(entry_trigger_pass=False, entry_direction=0)
        return self.inner.prepare_signals(df)

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return self.inner.build_pending_entry_metadata(row)


# ---------- Layer 1 per-symbol ----------


def run_layer1_per_symbol(
    bars: pd.DataFrame,
    engine_cfg: EngineConfig,
    entries: Dict[str, Any],
    exits: Dict[str, Any],
) -> pd.DataFrame:
    """For each (symbol, combo) pair, run a single-symbol engine.

    Returns a DataFrame with columns:
      symbol, combo, year, trades, sharpe, cagr, profit_factor, win_rate,
      avg_r, max_dd_pct, long_trades, short_trades, net_pnl, exit_reasons
    """
    all_rows: List[Dict[str, Any]] = []
    combos = [(eid, xid) for eid in entries for xid in exits]
    symbols = sorted(bars["symbol"].unique())
    total = len(symbols) * len(combos)
    done = 0

    # Disable group cap for per-symbol runs (per-symbol engine sees 1 group anyway,
    # but we want cap behaviour identical to the final portfolio run for comparability).
    layer1_cfg = replace(engine_cfg, use_group_risk_cap=False)

    for sym in symbols:
        sym_bars = bars[bars["symbol"] == sym].copy()
        if sym_bars.empty:
            continue

        for eid, xid in combos:
            combo_id = f"{eid}+{xid}"
            done += 1
            tag = f"[{done}/{total}] {sym} / {combo_id}"
            try:
                slot = StrategySlot(
                    strategy_id=combo_id,
                    entry_strategy=entries[eid],   # no symbol filter needed — single-symbol bars
                    exit_strategy=exits[xid],
                )
                engine = StrategyEngine(config=layer1_cfg, strategies=[slot])
                result = engine.run(sym_bars)
                yr_stats = yearly_stats_from_trades(
                    result.trades,
                    result.portfolio_daily,
                    layer1_cfg.initial_capital,
                )
                if not yr_stats:
                    all_rows.append({
                        "symbol": sym, "combo": combo_id, "year": 0,
                        "trades": 0, "sharpe": 0, "cagr": 0, "profit_factor": 0,
                        "win_rate": 0, "avg_r": 0, "max_dd_pct": 0,
                        "long_trades": 0, "short_trades": 0, "net_pnl": 0,
                        "exit_reasons": "{}",
                    })
                else:
                    for row in yr_stats:
                        row["symbol"] = sym
                        row["combo"] = combo_id
                        all_rows.append(row)
                    total_trades = sum(r["trades"] for r in yr_stats)
                    if done % 50 == 0 or done == total:
                        print(f"  {tag} — {total_trades} trades  ({done}/{total})")
            except Exception as e:
                print(f"  {tag} — ERROR: {e}")
                traceback.print_exc()

    df = pd.DataFrame(all_rows)
    cols = ["symbol", "combo", "year", "trades", "sharpe", "cagr",
            "profit_factor", "win_rate", "avg_r", "max_dd_pct",
            "long_trades", "short_trades", "net_pnl", "exit_reasons"]
    return df[[c for c in cols if c in df.columns]]


# ---------- Top-3 selection ----------


def pick_top3_per_symbol(
    layer1: pd.DataFrame, symbol: str, min_primary: int = 15, min_fallback: int = 5,
) -> List[str]:
    """Per-symbol top-3 combos by composite rank (Sharpe + PF + MaxDD)."""
    sdf = layer1[(layer1["symbol"] == symbol) & (layer1["year"] > 0)]
    if sdf.empty:
        return []

    agg = sdf.groupby("combo").agg(
        total_trades=("trades", "sum"),
        mean_sharpe=("sharpe", "mean"),
        mean_pf=("profit_factor", "mean"),
        worst_dd=("max_dd_pct", "min"),
    ).reset_index()

    filt = agg[agg["total_trades"] >= min_primary]
    if filt.empty:
        filt = agg[agg["total_trades"] >= min_fallback]
    if filt.empty:
        return []

    filt = filt.assign(
        rank_sharpe=filt["mean_sharpe"].rank(ascending=False),
        rank_pf=filt["mean_pf"].rank(ascending=False),
        rank_dd=filt["worst_dd"].rank(ascending=False),  # less-negative = better
    )
    filt["rank_sum"] = filt["rank_sharpe"] + filt["rank_pf"] + filt["rank_dd"]
    ranked = filt.sort_values("rank_sum")
    return ranked.head(3)["combo"].tolist()


def select_top3_all_symbols(layer1: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for sym in sorted(layer1["symbol"].unique()):
        top3 = pick_top3_per_symbol(layer1, sym)
        if top3:
            out[sym] = top3
    return out


# ---------- Portfolio build + run ----------


def build_portfolio_slots(
    top3_by_symbol: Dict[str, List[str]],
    entries: Dict[str, Any],
    exits: Dict[str, Any],
) -> List[StrategySlot]:
    slots: List[StrategySlot] = []
    for sym, top3 in top3_by_symbol.items():
        for rank, combo_id in enumerate(top3, start=1):
            eid, xid = combo_id.split("+")
            slot = StrategySlot(
                strategy_id=f"{sym}__{combo_id}__r{rank}",
                entry_strategy=SymbolFilteredEntry(entries[eid], sym),
                exit_strategy=exits[xid],  # exits only see positions they opened → no wrap needed
            )
            slots.append(slot)
    return slots


def run_portfolio(
    bars: pd.DataFrame,
    engine_cfg: EngineConfig,
    slots: List[StrategySlot],
    variant: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """Run the 180-slot engine. variant='v4a' locks symbols; 'v4b' allows parallel."""
    assert variant in ("v4a", "v4b"), variant
    cfg = replace(
        engine_cfg,
        use_group_risk_cap=False,
        symbol_position_lock=(variant == "v4a"),
    )

    # Portfolio bars = all symbols that have at least one slot.
    included_symbols = {s.entry_strategy.target_symbol for s in slots
                        if isinstance(s.entry_strategy, SymbolFilteredEntry)}
    portfolio_bars = bars[bars["symbol"].isin(included_symbols)].copy()

    print(f"  [{variant}] slots={len(slots)}  symbols={len(included_symbols)}  "
          f"use_group_cap={cfg.use_group_risk_cap}  symbol_lock={cfg.symbol_position_lock}")

    engine = StrategyEngine(config=cfg, strategies=slots)
    result = engine.run(portfolio_bars)
    analyzer = PortfolioAnalyzer(result, cfg)
    stats = analyzer.summary_stats()
    equity = analyzer.equity_curve()

    print(f"\n  [{variant}] Portfolio Summary:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
    return equity, stats, result


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="v4")
    parser.add_argument("--risk-per-trade", type=float, default=0.03)
    parser.add_argument("--portfolio-risk-cap", type=float, default=0.20)
    parser.add_argument("--layer1-only", action="store_true",
                        help="Run only Layer 1 (per-symbol × combo) + top-3 pick, skip portfolio.")
    parser.add_argument("--skip-layer1", action="store_true",
                        help="Skip Layer 1, read existing strategy_layer CSV from data/.")
    args = parser.parse_args()

    suffix = args.suffix
    run_dir = ROOT / "data" / "runs" / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    bars = load_hab_bars()
    symbols_before = sorted(bars["symbol"].unique())
    print(f"Loaded {len(bars)} rows across {len(symbols_before)} symbols.")

    # Exclude problematic groups / symbols that would NEVER be tradeable.
    # Keep only symbols present in non-excluded groups.
    from data.adapters.futures_static_meta import EXCLUDED_SYMBOLS
    bars = bars[~bars["symbol"].isin(EXCLUDED_SYMBOLS)]
    # Also drop the "commodity" catch-all and "index" and "bond" and "equity_index"
    excluded_groups = {"commodity", "index", "bond", "equity_index"}
    bars = bars[~bars["group_name"].isin(excluded_groups)]
    print(f"After exclusions: {bars['symbol'].nunique()} symbols, "
          f"{bars['group_name'].nunique()} groups.")

    risk_overrides = {
        "risk_per_trade": args.risk_per_trade,
        "portfolio_risk_cap": args.portfolio_risk_cap,
    }
    engine_cfg = build_engine_config(profile="research", overrides=risk_overrides)
    entries = build_entries()
    exits = build_exits()

    # ---- Layer 1 ----
    layer1_path = run_dir / "backtest_strategy_layer.csv"
    if args.skip_layer1 and layer1_path.exists():
        print(f"\n--- Skipping Layer 1 — loading {layer1_path} ---")
        layer1 = pd.read_csv(layer1_path)
    else:
        print(f"\n=== LAYER 1: per-symbol × combo ===")
        print(f"  {bars['symbol'].nunique()} symbols × {len(entries) * len(exits)} combos = "
              f"{bars['symbol'].nunique() * len(entries) * len(exits)} runs")
        t0 = time.time()
        layer1 = run_layer1_per_symbol(bars, engine_cfg, entries, exits)
        layer1.to_csv(layer1_path, index=False)
        print(f"  Saved: {layer1_path} ({len(layer1)} rows, {time.time()-t0:.0f}s)")

    # ---- Top-3 selection ----
    print(f"\n=== TOP-3 SELECTION ===")
    top3 = select_top3_all_symbols(layer1)
    top3_path = run_dir / "backtest_top3.json"
    top3_path.write_text(json.dumps(top3, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Symbols with viable top-3: {len(top3)} / {bars['symbol'].nunique()}")
    print(f"  Saved: {top3_path}")
    for sym, combos in sorted(top3.items())[:5]:
        print(f"    {sym}: {combos}")
    if len(top3) > 5:
        print(f"    ... (+{len(top3)-5} more)")

    if args.layer1_only:
        print("\n[--layer1-only: stopping after top-3 selection]")
        return

    # ---- Portfolio runs (v4a + v4b) ----
    slots = build_portfolio_slots(top3, entries, exits)
    print(f"\n=== LAYER 3: portfolio (slots={len(slots)}) ===")

    for variant in ("v4a", "v4b"):
        variant_dir = run_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        equity, stats, result = run_portfolio(bars, engine_cfg, slots, variant)
        out_path = variant_dir / "backtest_portfolio_layer.csv"
        if not equity.empty:
            equity.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}  ({len(equity)} rows, {time.time()-t0:.0f}s)")
        # DSR at per-symbol n_trials hypothesis
        daily_ret = equity["equity"].pct_change().dropna() if not equity.empty else pd.Series(dtype=float)
        dsr_sym = deflated_sharpe(daily_ret, len(top3))
        dsr_combo = deflated_sharpe(daily_ret, len(top3) * 30)  # every (sym, combo) pair
        print(f"  DSR(n={len(top3)}, per-symbol selection) = {dsr_sym:.4f}")
        print(f"  DSR(n={len(top3)*30}, full per-symbol × combo) = {dsr_combo:.4f}")

        summary_path = variant_dir / "summary.json"
        summary = {
            "variant": variant,
            "n_slots": len(slots),
            "n_symbols": len(top3),
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in stats.items()},
            "dsr_n_symbols": float(dsr_sym),
            "dsr_n_full": float(dsr_combo),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  Saved: {summary_path}")


if __name__ == "__main__":
    main()
