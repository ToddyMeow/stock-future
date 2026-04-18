"""Three-layer backtest: strategy layer x group layer x portfolio layer.

Usage:
    python scripts/run_three_layer_backtest.py

Settings: allow_short=True, ADX ON (adx_scale=30, adx_floor=0.2)
Combo matrix: 5 entries x 7 exits = 35
Groups: 9 (equity_index, bond, chem_energy, rubber_fiber, metals,
            black_steel, agri, building, livestock)

Output:
    data/backtest_strategy_layer.csv   — group x combo x year
    data/backtest_group_layer.csv      — group x year (best combo per group)
    data/backtest_portfolio_layer.csv  — daily equity curve (all groups combined)
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strats.config_loader import load_config, build_engine_config
from strats.engine import EngineConfig, StrategyEngine, StrategySlot
from strats.helpers import PortfolioAnalyzer, adx as compute_adx, wilder_atr

# ── Entries ──────────────────────────────────────────────────────────────
from strats.entries.hl_entry import HLEntryConfig, HLEntryStrategy
from strats.entries.boll_break_entry import BollBreakEntryConfig, BollBreakEntryStrategy
from strats.entries.ama_entry import AmaEntryConfig, AmaEntryStrategy
from strats.entries.double_ma_entry import DoubleMaEntryConfig, DoubleMaEntryStrategy
# from strats.entries.adaptive_macd_entry import AdaptiveMacdEntryConfig, AdaptiveMacdEntryStrategy

# ── Exits ────────────────────────────────────────────────────────────────
from strats.exits.hl_exit import HLExitConfig, HLExitStrategy
from strats.exits.boll_exit import BollExitConfig, BollExitStrategy
from strats.exits.ama_exit import AmaExitConfig, AmaExitStrategy
from strats.exits.atr_trail_exit import AtrTrailExitConfig, AtrTrailExitStrategy
from strats.exits.term_exit import TermExitConfig, TermExitStrategy
from strats.exits.double_ma_exit import DoubleMaExitConfig, DoubleMaExitStrategy
# from strats.exits.adaptive_trail_exit import AdaptiveTrailExitConfig, AdaptiveTrailExitStrategy

# ── Constants ────────────────────────────────────────────────────────────

GROUPS = [
    # v3b: revert chem_energy split (v3 experiment showed it hurt performance)
    "chem_energy", "rubber_fiber", "metals", "black_steel",
    "agri", "building", "livestock",
]

# v3b: dropped ind_AP (8% win rate) + five high-lock-rate symbols
# (ind_PM 94.7%, ind_RI 83.5%, ind_LR 80.0%, ind_BB 75.5%, ind_JR 69.3% limit-lock days)
IND_GROUPS = [
    "ind_CJ", "ind_EC", "ind_FB", "ind_LU", "ind_RR", "ind_WH",
]

ALL_GROUPS = GROUPS + IND_GROUPS


def build_entries() -> Dict[str, Any]:
    """Return {entry_id: entry_strategy} with allow_short=True."""
    return {
        "hl_9": HLEntryStrategy(HLEntryConfig(period=9, allow_short=True)),
        "hl_21": HLEntryStrategy(HLEntryConfig(period=21, allow_short=True)),
        "boll": BollBreakEntryStrategy(BollBreakEntryConfig(period=22, k=2.0, allow_short=True)),
        "ama": AmaEntryStrategy(AmaEntryConfig(n=10, fast_period=2, slow_period=30, allow_short=True)),
        "double_ma": DoubleMaEntryStrategy(DoubleMaEntryConfig(fast=13, slow=34, allow_short=True)),
    }


def build_exits() -> Dict[str, Any]:
    """Return {exit_id: exit_strategy}."""
    return {
        "hl": HLExitStrategy(HLExitConfig(period=21)),
        "boll": BollExitStrategy(BollExitConfig(period=22, k=2.0)),
        "ama": AmaExitStrategy(AmaExitConfig(n=10, fast_period=2, slow_period=30)),
        "atr_trail": AtrTrailExitStrategy(AtrTrailExitConfig(atr_mult=4.5)),
        "term": TermExitStrategy(TermExitConfig(min_bars=2, max_bars=13, min_target_r=1.0)),
        "double_ma": DoubleMaExitStrategy(DoubleMaExitConfig(fast=13, slow=34)),
    }


def make_engine_config(adx_off: bool = False, risk_overrides: dict = None) -> EngineConfig:
    """Load base config from config.yaml, override allow_short=True."""
    cfg = load_config()
    base = build_engine_config(cfg)
    overrides = {"allow_short": True}
    if adx_off:
        overrides["adx_floor"] = 1.0  # trend_score always 1.0 → ADX has no effect
    if risk_overrides:
        overrides.update(risk_overrides)
    return replace(base, **overrides)


def load_bars() -> pd.DataFrame:
    """Load hab_bars.csv and clamp OHLC so engine validation passes."""
    path = ROOT / "data" / "cache" / "normalized" / "hab_bars.csv"
    bars = pd.read_csv(path)
    bars["date"] = pd.to_datetime(bars["date"])
    # Clamp: ensure high >= max(open,close), low <= min(open,close)
    bars["high"] = bars[["high", "open", "close"]].max(axis=1)
    bars["low"] = bars[["low", "open", "close"]].min(axis=1)
    return bars


def filter_group(bars: pd.DataFrame, group: str) -> pd.DataFrame:
    """Filter bars to symbols belonging to a specific group."""
    return bars[bars["group_name"] == group].copy()


def yearly_stats_from_trades(
    trades: pd.DataFrame,
    portfolio_daily: pd.DataFrame,
    initial_capital: float,
) -> List[Dict[str, Any]]:
    """Split trades by year and compute per-year metrics."""
    if trades.empty:
        return []

    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year
    years = sorted(trades["year"].unique())

    # Also split equity curve by year
    pdf = portfolio_daily.copy()
    if not pdf.empty:
        pdf["year"] = pd.to_datetime(pdf["date"]).dt.year

    rows = []
    for yr in years:
        yr_trades = trades[trades["year"] == yr]
        n = len(yr_trades)
        if n == 0:
            continue

        wins = yr_trades[yr_trades["net_pnl"] > 0]
        losses = yr_trades[yr_trades["net_pnl"] <= 0]
        win_rate = len(wins) / n
        pf = (
            wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())
            if len(losses) > 0 and losses["net_pnl"].sum() != 0
            else float("inf") if len(wins) > 0 else 0.0
        )
        avg_r = yr_trades["r_multiple"].mean()

        # Sharpe from equity curve for this year
        sharpe = 0.0
        cagr = 0.0
        max_dd = 0.0
        if not pdf.empty:
            yr_eq = pdf[pdf["year"] == yr]
            if len(yr_eq) > 1:
                daily_ret = yr_eq["equity"].pct_change().dropna()
                if daily_ret.std() > 0:
                    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                eq_start = yr_eq["equity"].iloc[0]
                eq_end = yr_eq["equity"].iloc[-1]
                days = (yr_eq["date"].iloc[-1] - yr_eq["date"].iloc[0]).days
                if eq_start > 0 and days > 0:
                    cagr = (eq_end / eq_start) ** (365.25 / max(days, 1)) - 1.0
                peak = yr_eq["equity"].cummax()
                dd = (yr_eq["equity"] - peak) / peak.where(peak > 0, np.nan)
                max_dd = dd.min() if not dd.isna().all() else 0.0

        # Direction breakdown
        long_trades = len(yr_trades[yr_trades["direction"] == 1])
        short_trades = len(yr_trades[yr_trades["direction"] == -1])

        # Exit reason distribution
        exit_reasons = yr_trades["exit_reason"].value_counts().to_dict()

        rows.append({
            "year": yr,
            "trades": n,
            "sharpe": round(sharpe, 3),
            "cagr": round(cagr, 4),
            "profit_factor": round(pf, 3) if pf != float("inf") else 999.0,
            "win_rate": round(win_rate, 3),
            "avg_r": round(avg_r, 3),
            "max_dd_pct": round(max_dd, 4),
            "long_trades": long_trades,
            "short_trades": short_trades,
            "net_pnl": round(yr_trades["net_pnl"].sum(), 2),
            "exit_reasons": json.dumps(exit_reasons, ensure_ascii=False),
        })
    return rows


def compute_avg_adx_by_year(bars: pd.DataFrame) -> Dict[int, float]:
    """Compute average ADX(20) by year across all symbols in the group."""
    result = {}
    for sym, sdf in bars.groupby("symbol"):
        sdf = sdf.sort_values("date").reset_index(drop=True)
        if len(sdf) < 25:
            continue
        adx_series = compute_adx(sdf["high"], sdf["low"], sdf["close"], period=20)
        sdf = sdf.copy()
        sdf["adx"] = adx_series
        sdf["year"] = sdf["date"].dt.year
        for yr, ydf in sdf.groupby("year"):
            vals = ydf["adx"].dropna()
            if len(vals) > 0:
                result.setdefault(yr, []).append(vals.mean())
    return {yr: round(np.mean(vals), 1) for yr, vals in result.items()}


# =====================================================================
# Layer 1: Strategy Layer
# =====================================================================

def run_layer1(
    bars: pd.DataFrame,
    engine_cfg: EngineConfig,
    entries: Dict[str, Any],
    exits: Dict[str, Any],
) -> pd.DataFrame:
    """Run all combos for all groups. Returns strategy_layer DataFrame."""
    all_rows = []
    combos = [(eid, xid) for eid in entries for xid in exits]
    total = len(ALL_GROUPS) * len(combos)
    done = 0

    for group in ALL_GROUPS:
        group_bars = filter_group(bars, group)
        n_symbols = group_bars["symbol"].nunique()
        if group_bars.empty:
            print(f"  [{group}] SKIP — no data")
            continue

        for entry_id, exit_id in combos:
            combo_id = f"{entry_id}+{exit_id}"
            done += 1
            tag = f"[{done}/{total}] {group} / {combo_id}"

            try:
                slot = StrategySlot(
                    strategy_id=combo_id,
                    entry_strategy=entries[entry_id],
                    exit_strategy=exits[exit_id],
                )
                engine = StrategyEngine(config=engine_cfg, strategies=[slot])
                result = engine.run(group_bars)

                yr_stats = yearly_stats_from_trades(
                    result.trades,
                    result.portfolio_daily,
                    engine_cfg.initial_capital,
                )

                if not yr_stats:
                    print(f"  {tag} — 0 trades")
                    all_rows.append({
                        "group": group,
                        "combo": combo_id,
                        "year": 0,
                        "trades": 0,
                        "sharpe": 0,
                        "cagr": 0,
                        "profit_factor": 0,
                        "win_rate": 0,
                        "avg_r": 0,
                        "max_dd_pct": 0,
                        "long_trades": 0,
                        "short_trades": 0,
                        "net_pnl": 0,
                        "exit_reasons": "{}",
                    })
                else:
                    for row in yr_stats:
                        row["group"] = group
                        row["combo"] = combo_id
                        all_rows.append(row)
                    total_trades = sum(r["trades"] for r in yr_stats)
                    print(f"  {tag} — {total_trades} trades, {len(yr_stats)} years")

            except Exception as e:
                print(f"  {tag} — ERROR: {e}")
                traceback.print_exc()
                all_rows.append({
                    "group": group,
                    "combo": combo_id,
                    "year": 0,
                    "trades": 0,
                    "sharpe": 0,
                    "cagr": 0,
                    "profit_factor": 0,
                    "win_rate": 0,
                    "avg_r": 0,
                    "max_dd_pct": 0,
                    "long_trades": 0,
                    "short_trades": 0,
                    "net_pnl": 0,
                    "exit_reasons": json.dumps({"ERROR": str(e)}),
                })

    df = pd.DataFrame(all_rows)
    # Reorder columns
    cols = ["group", "combo", "year", "trades", "sharpe", "cagr", "profit_factor",
            "win_rate", "avg_r", "max_dd_pct", "long_trades", "short_trades",
            "net_pnl", "exit_reasons"]
    return df[[c for c in cols if c in df.columns]]


# =====================================================================
# Layer 2: Group Layer
# =====================================================================

def pick_best_combo(layer1: pd.DataFrame, group: str,
                    min_primary: int = 30, min_fallback: int = 10) -> str:
    """Pick the best combo using composite ranking: Sharpe + PF + MaxDD.

    Rank each metric independently, sum ranks → lowest sum wins.
    This avoids small-sample bias from any single metric.

    Thresholds: ind_* single-symbol groups should pass (min_primary=15, min_fallback=5)
    because single-symbol 8-year trade counts are naturally lower than multi-symbol groups.
    """
    gdf = layer1[(layer1["group"] == group) & (layer1["year"] > 0)]
    if gdf.empty:
        return ""
    combo_stats = gdf.groupby("combo").agg(
        total_trades=("trades", "sum"),
        total_pnl=("net_pnl", "sum"),
        mean_sharpe=("sharpe", "mean"),
        mean_pf=("profit_factor", "mean"),
        worst_dd=("max_dd_pct", "min"),  # most negative = worst
    ).reset_index()
    # Minimum trade threshold to filter noise
    combo_stats = combo_stats[combo_stats["total_trades"] >= min_primary]
    if combo_stats.empty:
        # Fallback: relaxed threshold
        combo_stats = gdf.groupby("combo").agg(
            total_trades=("trades", "sum"),
            total_pnl=("net_pnl", "sum"),
            mean_sharpe=("sharpe", "mean"),
            mean_pf=("profit_factor", "mean"),
            worst_dd=("max_dd_pct", "min"),
        ).reset_index()
        combo_stats = combo_stats[combo_stats["total_trades"] >= min_fallback]
    if combo_stats.empty:
        return ""
    # Rank each metric (lower rank = better)
    combo_stats["rank_sharpe"] = combo_stats["mean_sharpe"].rank(ascending=False)
    combo_stats["rank_pf"] = combo_stats["mean_pf"].rank(ascending=False)
    combo_stats["rank_dd"] = combo_stats["worst_dd"].rank(ascending=False)  # less negative = better = lower rank
    combo_stats["rank_sum"] = combo_stats["rank_sharpe"] + combo_stats["rank_pf"] + combo_stats["rank_dd"]
    return combo_stats.sort_values("rank_sum").iloc[0]["combo"]


def run_layer2(
    bars: pd.DataFrame,
    engine_cfg: EngineConfig,
    entries: Dict[str, Any],
    exits: Dict[str, Any],
    layer1: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Run best combo per group, return group layer stats + best combo map."""
    all_rows = []
    best_combos = {}

    for group in ALL_GROUPS:
        # Independent single-symbol groups have naturally lower trade counts
        if group.startswith("ind_"):
            best = pick_best_combo(layer1, group, min_primary=15, min_fallback=5)
        else:
            best = pick_best_combo(layer1, group, min_primary=30, min_fallback=10)
        if not best:
            print(f"  [{group}] SKIP — no viable combo")
            continue

        best_combos[group] = best
        entry_id, exit_id = best.split("+")
        group_bars = filter_group(bars, group)
        if group_bars.empty:
            continue

        print(f"  [{group}] best combo: {best}")

        slot = StrategySlot(
            strategy_id=best,
            entry_strategy=entries[entry_id],
            exit_strategy=exits[exit_id],
        )
        engine = StrategyEngine(config=engine_cfg, strategies=[slot])
        result = engine.run(group_bars)
        analyzer = PortfolioAnalyzer(result, engine_cfg)

        # Overall stats
        stats = analyzer.summary_stats()

        # Per-year breakdown
        trades = result.trades.copy()
        if not trades.empty:
            trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year

        pdf = result.portfolio_daily.copy()
        if not pdf.empty:
            pdf["year"] = pd.to_datetime(pdf["date"]).dt.year

        # Signal density
        sig = analyzer.signal_density()
        if not sig.empty:
            sig["year"] = pd.to_datetime(sig["date"]).dt.year

        # Avg ADX by year
        adx_by_year = compute_avg_adx_by_year(group_bars)

        years = set()
        if not trades.empty:
            years.update(trades["year"].unique())
        if not pdf.empty:
            years.update(pdf["year"].unique())
        years = sorted(years)

        for yr in years:
            yr_trades = trades[trades["year"] == yr] if not trades.empty else pd.DataFrame()
            yr_eq = pdf[pdf["year"] == yr] if not pdf.empty else pd.DataFrame()

            # Sharpe/CAGR from equity
            sharpe = 0.0
            cagr = 0.0
            max_dd = 0.0
            if len(yr_eq) > 1:
                dr = yr_eq["equity"].pct_change().dropna()
                if dr.std() > 0:
                    sharpe = dr.mean() / dr.std() * np.sqrt(252)
                eq_s, eq_e = yr_eq["equity"].iloc[0], yr_eq["equity"].iloc[-1]
                days = (yr_eq["date"].iloc[-1] - yr_eq["date"].iloc[0]).days
                if eq_s > 0 and days > 0:
                    cagr = (eq_e / eq_s) ** (365.25 / max(days, 1)) - 1.0
                peak = yr_eq["equity"].cummax()
                dd = (yr_eq["equity"] - peak) / peak.where(peak > 0, np.nan)
                max_dd = dd.min() if not dd.isna().all() else 0.0

            # Trades count
            n_trades = len(yr_trades)

            # Signal density
            yr_sig = sig[sig["year"] == yr] if not sig.empty and "year" in sig.columns else pd.DataFrame()
            signals_fired = int(yr_sig["signals_fired"].sum()) if not yr_sig.empty else 0
            signals_accepted = int(yr_sig["signals_accepted"].sum()) if not yr_sig.empty else 0

            # ADX
            avg_adx = adx_by_year.get(yr, 0.0)

            # Symbol PnL
            sym_pnl = {}
            if not yr_trades.empty:
                sym_pnl = yr_trades.groupby("symbol")["net_pnl"].sum().round(2).to_dict()

            all_rows.append({
                "group": group,
                "best_combo": best,
                "year": yr,
                "sharpe": round(sharpe, 3),
                "cagr": round(cagr, 4),
                "max_dd_pct": round(max_dd, 4),
                "total_trades": n_trades,
                "signals_fired": signals_fired,
                "signals_accepted": signals_accepted,
                "avg_adx": avg_adx,
                "net_pnl": round(yr_trades["net_pnl"].sum(), 2) if not yr_trades.empty else 0,
                "symbol_pnl": json.dumps(sym_pnl, ensure_ascii=False),
            })

    return pd.DataFrame(all_rows), best_combos


# =====================================================================
# Layer 3: Portfolio Layer
# =====================================================================

def run_layer3(
    bars: pd.DataFrame,
    engine_cfg: EngineConfig,
    entries: Dict[str, Any],
    exits: Dict[str, Any],
    best_combos: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Combine all groups with their best combos into one multi-strategy engine."""
    # Build strategy slots — one per group's best combo
    slots = []
    included_groups = set()
    for group, combo in best_combos.items():
        entry_id, exit_id = combo.split("+")
        slot_id = f"{group}_{combo}"
        slots.append(StrategySlot(
            strategy_id=slot_id,
            entry_strategy=entries[entry_id],
            exit_strategy=exits[exit_id],
        ))
        included_groups.add(group)

    if not slots:
        print("  No viable combos for any group — skipping Layer 3")
        return pd.DataFrame(), {}

    # Filter bars to only included groups
    portfolio_bars = bars[bars["group_name"].isin(included_groups)].copy()

    print(f"  Running multi-strategy engine: {len(slots)} slots, "
          f"{portfolio_bars['symbol'].nunique()} symbols")

    engine = StrategyEngine(config=engine_cfg, strategies=slots)
    result = engine.run(portfolio_bars)
    analyzer = PortfolioAnalyzer(result, engine_cfg)

    # Summary stats
    stats = analyzer.summary_stats()

    # Equity curve
    equity = analyzer.equity_curve()

    # Group contribution
    group_contrib = analyzer.group_contribution()

    # Print summary
    print("\n  === Portfolio Summary ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    if not group_contrib.empty:
        print("\n  === Group Contribution ===")
        print(group_contrib.to_string(index=False))

    # Drawdown episodes
    dd_eps = analyzer.drawdown_episodes()
    if not dd_eps.empty:
        worst = dd_eps.sort_values("max_drawdown_pct").iloc[0]
        print(f"\n  Worst drawdown: {worst['max_drawdown_pct']:.2%} "
              f"({worst['start_date']} → {worst['trough_date']})")

    # Risk utilization summary
    risk_util = analyzer.risk_utilization()
    if not risk_util.empty:
        print(f"  Avg risk utilization: {risk_util['risk_usage_pct'].mean():.2%}")
        print(f"  Avg leverage: {risk_util['leverage'].mean():.2f}x")

    return equity, stats


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adx-off", action="store_true", help="Disable ADX filter")
    parser.add_argument("--suffix", type=str, default="", help="Output file suffix")
    parser.add_argument("--risk-per-trade", type=float, default=None)
    parser.add_argument("--portfolio-risk-cap", type=float, default=None)
    parser.add_argument("--group-risk-cap", type=float, default=None, help="Uniform group risk cap")
    args = parser.parse_args()

    adx_off = args.adx_off
    suffix = args.suffix or ("_no_adx" if adx_off else "")
    adx_label = "ADX OFF" if adx_off else "ADX ON"

    # Build risk overrides from CLI
    risk_overrides = {}
    if args.risk_per_trade is not None:
        risk_overrides["risk_per_trade"] = args.risk_per_trade
    if args.portfolio_risk_cap is not None:
        risk_overrides["portfolio_risk_cap"] = args.portfolio_risk_cap
    if args.group_risk_cap is not None:
        cap = args.group_risk_cap
        risk_overrides["group_risk_cap"] = {g: cap for g in GROUPS}

    t0 = time.time()
    print("=" * 60)
    print(f"Three-Layer Backtest")
    print(f"  allow_short=True, {adx_label}")
    print("=" * 60)

    # Setup
    bars = load_bars()
    engine_cfg = make_engine_config(adx_off=adx_off, risk_overrides=risk_overrides or None)
    entries = build_entries()
    exits = build_exits()

    print(f"\nData: {len(bars)} rows, {bars['symbol'].nunique()} symbols")
    print(f"Groups: {GROUPS}")
    print(f"Combos: {len(entries)} entries x {len(exits)} exits = {len(entries)*len(exits)}")
    print(f"Engine: capital={engine_cfg.initial_capital:,.0f}, "
          f"risk={engine_cfg.risk_per_trade}, stop_atr={engine_cfg.stop_atr_mult}, "
          f"adx_floor={engine_cfg.adx_floor}")

    # Layer 1
    print(f"\n{'='*60}")
    print("LAYER 1: Strategy Layer (group x combo x year)")
    print(f"{'='*60}")
    layer1 = run_layer1(bars, engine_cfg, entries, exits)

    out1 = ROOT / "data" / f"backtest_strategy_layer{suffix}.csv"
    layer1.to_csv(out1, index=False)
    print(f"\n  -> Saved: {out1} ({len(layer1)} rows)")

    # Layer 2
    print(f"\n{'='*60}")
    print("LAYER 2: Group Layer (best combo per group x year)")
    print(f"{'='*60}")
    layer2, best_combos = run_layer2(bars, engine_cfg, entries, exits, layer1)

    out2 = ROOT / "data" / f"backtest_group_layer{suffix}.csv"
    if not layer2.empty:
        layer2.to_csv(out2, index=False)
        print(f"\n  -> Saved: {out2} ({len(layer2)} rows)")
    print(f"\n  Best combos: {json.dumps(best_combos, indent=2)}")

    # Layer 3
    print(f"\n{'='*60}")
    print("LAYER 3: Portfolio Layer (combined equity curve)")
    print(f"{'='*60}")
    equity, portfolio_stats = run_layer3(bars, engine_cfg, entries, exits, best_combos)

    out3 = ROOT / "data" / f"backtest_portfolio_layer{suffix}.csv"
    if not equity.empty:
        equity.to_csv(out3, index=False)
        print(f"\n  -> Saved: {out3} ({len(equity)} rows)")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
