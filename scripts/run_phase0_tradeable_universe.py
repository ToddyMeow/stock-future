"""Phase 0 — Tradeable universe filter.

Rule: drop any symbol whose single-contract risk at a typical stop distance
exceeds equity × risk_per_trade (default 3%). These symbols can't fit even
1 contract within the per-trade risk budget, so they'd never trade or
would consume the whole risk allowance on one contract.

single_hand_risk_CNY = stop_atr_mult × ATR(atr_period) × contract_multiplier

Rule check (per bar):
  single_hand_risk > equity × risk_per_trade  →  untradeable bar
  median fraction of untradeable bars > 50%   →  drop symbol

Also reports typical qty at median ATR for each surviving symbol — useful
for checking that it's not just 1-2 contracts per trade.

Output: data/runs/phase0/tradeable_universe.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strats.helpers import wilder_atr

DEFAULT_OUT_DIR = ROOT / "data" / "runs" / "phase0"

DEFAULT_INITIAL_CAPITAL = 1_000_000.0
RISK_PER_TRADE = 0.03
STOP_ATR_MULT = 2.0
ATR_PERIOD = 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL,
                    help="Total equity (default 1_000_000)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"Output dir (default {DEFAULT_OUT_DIR.relative_to(ROOT)})")
    ap.add_argument("--risk-per-trade", type=float, default=RISK_PER_TRADE)
    args = ap.parse_args()

    OUT_DIR = args.output_dir
    initial_capital = float(args.initial_capital)
    risk_per_trade = float(args.risk_per_trade)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = pd.read_csv(ROOT / "data" / "cache" / "normalized" / "hab_bars.csv")
    bars["date"] = pd.to_datetime(bars["date"])
    # OHLC clamp (same as run_three_layer_backtest.load_bars)
    bars["high"] = bars[["high", "open", "close"]].max(axis=1)
    bars["low"] = bars[["low", "open", "close"]].min(axis=1)

    risk_budget = initial_capital * risk_per_trade
    print(f"Phase 0 — Tradeable universe")
    print(f"  initial_capital = {initial_capital:,.0f}")
    print(f"  risk_per_trade  = {risk_per_trade:.2%}  (budget/trade = {risk_budget:,.0f})")
    print(f"  stop = {STOP_ATR_MULT} × ATR({ATR_PERIOD})")
    print(f"  input: {bars['symbol'].nunique()} symbols\n")

    rows = []
    for sym, sdf in bars.groupby("symbol"):
        sdf = sdf.sort_values("date").reset_index(drop=True)
        if len(sdf) < ATR_PERIOD * 3:
            rows.append({
                "symbol": sym, "group": sdf["group_name"].iloc[0],
                "bars": len(sdf), "tradeable": False, "reason": "too-short",
            })
            continue

        atr = wilder_atr(sdf["high"], sdf["low"], sdf["close"], period=ATR_PERIOD)
        mult = float(sdf["contract_multiplier"].iloc[-1])
        # single_hand_risk per bar (CNY)
        shr = STOP_ATR_MULT * atr * mult
        valid = shr.iloc[ATR_PERIOD * 2:].dropna()
        if len(valid) == 0:
            rows.append({
                "symbol": sym, "group": sdf["group_name"].iloc[0],
                "bars": len(sdf), "tradeable": False, "reason": "no-valid-ATR",
            })
            continue

        median_shr = float(valid.median())
        p90_shr = float(valid.quantile(0.90))
        untradeable_bars = (valid > risk_budget).mean()
        # qty at median ATR: floor(budget / shr)
        qty_at_median = risk_budget / median_shr
        qty_at_p90 = risk_budget / p90_shr

        # Tradeable rule: < 50% of bars have single-hand risk over budget
        tradeable = untradeable_bars < 0.50
        reason = "ok" if tradeable else f"{untradeable_bars:.0%}-bars-over-budget"

        rows.append({
            "symbol": sym,
            "group": sdf["group_name"].iloc[0],
            "bars": len(sdf),
            "multiplier": mult,
            "median_price": float(sdf["close"].median()),
            "median_atr": float(atr.iloc[ATR_PERIOD * 2:].dropna().median()),
            "median_single_hand_risk": round(median_shr, 0),
            "p90_single_hand_risk": round(p90_shr, 0),
            f"risk_budget_{int(risk_per_trade*100)}pct": round(risk_budget, 0),
            "qty_at_median_atr": round(qty_at_median, 2),
            "qty_at_p90_atr": round(qty_at_p90, 2),
            "pct_bars_over_budget": round(untradeable_bars, 3),
            "tradeable": tradeable,
            "reason": reason,
        })

    df = pd.DataFrame(rows).sort_values(["tradeable", "group", "symbol"],
                                         ascending=[False, True, True])
    df.to_csv(OUT_DIR / "tradeable_universe.csv", index=False)

    # Summary
    n_in = len(df)
    n_ok = int(df["tradeable"].sum())
    n_drop = n_in - n_ok
    print(f"Result: {n_ok} tradeable, {n_drop} dropped")

    print("\n=== TRADEABLE by group ===")
    tg = df[df["tradeable"]].groupby("group").agg(
        n_syms=("symbol", "count"),
        syms=("symbol", lambda x: ",".join(sorted(x))),
    ).reset_index()
    print(tg.to_string(index=False))

    print("\n=== DROPPED (untradeable) ===")
    drop = df[~df["tradeable"]][["symbol", "group", "median_single_hand_risk",
                                   "pct_bars_over_budget", "reason"]]
    print(drop.to_string(index=False))

    # Save JSON of tradeable symbols for Phase 1 to consume
    tradeable_syms = sorted(df[df["tradeable"]]["symbol"].tolist())
    tradeable_groups = sorted(df[df["tradeable"]]["group"].unique().tolist())
    (OUT_DIR / "tradeable_symbols.json").write_text(json.dumps({
        "initial_capital": initial_capital,
        "risk_per_trade": risk_per_trade,
        "stop_atr_mult": STOP_ATR_MULT,
        "atr_period": ATR_PERIOD,
        "n_tradeable": len(tradeable_syms),
        "n_dropped": n_drop,
        "tradeable_symbols": tradeable_syms,
        "tradeable_groups": tradeable_groups,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT_DIR}/tradeable_universe.csv, tradeable_symbols.json")


if __name__ == "__main__":
    main()
