"""Phase 5 — OOS Validation.

Loads Phase 4 stable combos (6 groups × 1 combo each), runs the Layer 3
multi-strategy engine on the OOS period (2024-01-01 → 2025-12-31) with
full risk controls (same variant as Phase 3).

Output:
  data/runs/phase5/backtest_portfolio_layer_<tag>.csv   — daily equity
  data/runs/phase5/trades_<tag>.csv                      — every trade (19 cols)
  data/runs/phase5/per_group_<tag>.csv                   — per-group PnL
  data/runs/phase5/per_symbol_<tag>.csv                  — per-symbol PnL
  data/runs/phase5/reject_distribution_<tag>.csv         — reject reason counts
  data/runs/phase5/summary_<tag>.json                    — headline metrics

Usage:
  python scripts/run_phase5_oos.py --tag risk3cap6 --group-cap 0.06
  python scripts/run_phase5_oos.py --tag risk3cap8 --group-cap 0.08
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strats.engine import StrategyEngine
from strats.factory import build_engine_config, build_strategy_slots_from_combos
from strats.research_support import load_hab_bars
from scripts.run_phase3_combo_selection import CONFIRMED_SYMS, KEEP_TRADE_COLS

PHASE3_DIR = ROOT / "data" / "runs" / "phase3"
OUT_DIR = ROOT / "data" / "runs" / "phase5"


def load_stable_combos(tag: str) -> pd.DataFrame:
    p = PHASE3_DIR / f"best_combos_stable_{tag}.csv"
    df = pd.read_csv(p)
    # Include both rolling-stable AND explicitly-whitelisted new listings
    return df[df["stability_status"].isin(["stable", "new_listing"])].copy()


def compute_portfolio_metrics(result, engine_cfg) -> dict:
    trades = result.trades
    pdf = result.portfolio_daily
    if trades.empty:
        return {"total_trades": 0}
    n = len(trades)
    net = float(trades["net_pnl"].sum())
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    avg_w = float(wins["net_pnl"].mean()) if len(wins) > 0 else 0.0
    avg_l = float(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
    wl_ratio = (-avg_w / avg_l) if avg_l < 0 else 0.0

    if not pdf.empty:
        eq = pdf["equity"]
        peak = eq.cummax()
        dd = (eq - peak) / peak.where(peak > 0, np.nan)
        max_dd = float(dd.min()) if not dd.isna().all() else 0.0
        ret = eq.pct_change().dropna()
        sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
        initial = float(eq.iloc[0])
        final = float(eq.iloc[-1])
        total_ret = final / initial - 1.0 if initial > 0 else 0.0
        days = (pdf["date"].iloc[-1] - pdf["date"].iloc[0]).days
        cagr = ((final / initial) ** (365.25 / days) - 1.0) if initial > 0 and days > 0 else 0.0
    else:
        max_dd = sharpe = total_ret = cagr = initial = final = 0.0

    return {
        "total_trades": n,
        "total_net_pnl": round(net, 0),
        "expectancy": round(net / n, 1),
        "win_rate": round(len(wins) / n, 3),
        "wl_ratio": round(wl_ratio, 3),
        "avg_winner": round(avg_w, 0),
        "avg_loser": round(avg_l, 0),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr, 4),
        "total_return": round(total_ret, 4),
        "initial_equity": round(initial, 0),
        "final_equity": round(final, 0),
        "long_trades": int((trades["direction"] == 1).sum()),
        "short_trades": int((trades["direction"] == -1).sum()),
    }


def per_group_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "group_name" not in trades.columns:
        return pd.DataFrame()
    rows = []
    for g, sub in trades.groupby("group_name"):
        wins = sub[sub["net_pnl"] > 0]
        losses = sub[sub["net_pnl"] <= 0]
        n = len(sub)
        avg_w = float(wins["net_pnl"].mean()) if len(wins) > 0 else 0.0
        avg_l = float(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
        rows.append({
            "group": g, "trades": n,
            "net_pnl": round(float(sub["net_pnl"].sum()), 0),
            "expectancy": round(float(sub["net_pnl"].sum()) / n, 1),
            "win_rate": round(len(wins) / n, 3),
            "wl_ratio": round(-avg_w / avg_l, 3) if avg_l < 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)


def per_symbol_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for sym, sub in trades.groupby("symbol"):
        wins = sub[sub["net_pnl"] > 0]
        losses = sub[sub["net_pnl"] <= 0]
        n = len(sub)
        avg_w = float(wins["net_pnl"].mean()) if len(wins) > 0 else 0.0
        avg_l = float(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
        group = sub["group_name"].iloc[0] if "group_name" in sub.columns else None
        rows.append({
            "symbol": sym, "group": group, "trades": n,
            "net_pnl": round(float(sub["net_pnl"].sum()), 0),
            "expectancy": round(float(sub["net_pnl"].sum()) / n, 1),
            "win_rate": round(len(wins) / n, 3),
            "wl_ratio": round(-avg_w / avg_l, 3) if avg_l < 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)


def reject_distribution(daily_status: pd.DataFrame) -> pd.DataFrame:
    if daily_status.empty or "risk_reject_reason" not in daily_status.columns:
        return pd.DataFrame()
    # daily_status rows are (symbol, date, ...); risk_reject_reason = code or None
    vc = daily_status["risk_reject_reason"].dropna().value_counts()
    total = vc.sum()
    return pd.DataFrame({
        "reject_reason": vc.index,
        "count": vc.values,
        "pct": (vc.values / total * 100).round(2) if total > 0 else 0,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True,
                    help="Any tag name — reads best_combos_stable_<tag>.csv")
    ap.add_argument("--group-cap", type=float, default=None,
                    help="Override (default: 0.06 if tag=risk3cap6, 0.08 if risk3cap8)")
    ap.add_argument("--risk-per-trade", type=float, default=0.03)
    ap.add_argument("--portfolio-cap", type=float, default=0.20)
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--oos-end", default="2025-12-31")
    ap.add_argument("--initial-capital", type=float, default=None,
                    help="Override default 1_000_000 initial equity")
    ap.add_argument("--universe", type=Path, default=None,
                    help="Optional path to tradeable_symbols.json — restricts "
                         "CONFIRMED_SYMS to its 'tradeable_symbols' intersection")
    ap.add_argument("--output-tag", default=None,
                    help="Override output filename suffix (defaults to --tag). "
                         "Use when running the same stable_combos tag twice with "
                         "different OOS ranges to avoid overwriting.")
    args = ap.parse_args()

    if args.group_cap is None:
        args.group_cap = 0.06 if args.tag == "risk3cap6" else 0.08

    out_tag = args.output_tag if args.output_tag else args.tag

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stable = load_stable_combos(args.tag)
    print(f"=== Phase 5 OOS: tag={args.tag} ===")
    print(f"  risk_per_trade={args.risk_per_trade}, group_cap={args.group_cap}, "
          f"portfolio_cap={args.portfolio_cap}")
    print(f"  OOS: {args.oos_start} → {args.oos_end}")
    print(f"  Stable combos: {len(stable)}")
    for _, row in stable.iterrows():
        print(f"    {row['group']:15s} {row['best_combo']}")

    bars = load_hab_bars()
    # Filter to confirmed symbols AND stable groups
    stable_groups = set(stable["group"].tolist())
    confirmed = set(CONFIRMED_SYMS)
    if args.universe is not None:
        uni = json.loads(args.universe.read_text())
        uni_syms = set(uni["tradeable_symbols"])
        before = len(confirmed)
        confirmed = confirmed & uni_syms
        print(f"  Universe filter: CONFIRMED_SYMS {before} → {len(confirmed)} "
              f"(dropped {sorted(set(CONFIRMED_SYMS) - confirmed)})")
    bars = bars[
        bars["symbol"].isin(confirmed)
        & bars["group_name"].isin(stable_groups)
    ].copy()
    oos_start = pd.to_datetime(args.oos_start)
    oos_end = pd.to_datetime(args.oos_end)
    bars = bars[(bars["date"] >= oos_start) & (bars["date"] <= oos_end)].copy()
    print(f"  Bars filtered: {len(bars)} rows, {bars['symbol'].nunique()} syms, "
          f"{bars['date'].min().date()} → {bars['date'].max().date()}")

    overrides = {
        "risk_per_trade": args.risk_per_trade,
        "portfolio_risk_cap": args.portfolio_cap,
        "group_risk_cap": {g: args.group_cap for g in stable_groups},
        "default_group_risk_cap": args.group_cap,
    }
    if args.initial_capital is not None:
        overrides["initial_capital"] = args.initial_capital
    engine_cfg = build_engine_config(profile="research", overrides=overrides)

    slots = build_strategy_slots_from_combos(
        stable,
        include_adaptive=False,
        allow_short=True,
    )
    sar_slots = sum(1 for s in slots if s.reverse_on_stop is True)
    print(f"  Slots: {len(slots)} ({sar_slots} with SAR on, "
          f"{len(slots) - sar_slots} without)")

    t0 = time.time()
    engine = StrategyEngine(config=engine_cfg, strategies=slots)
    result = engine.run(bars)
    elapsed = time.time() - t0
    print(f"\n  engine run: {elapsed:.0f}s, trades={len(result.trades)}")

    # Portfolio metrics
    metrics = compute_portfolio_metrics(result, engine_cfg)
    print("\n===== OOS Portfolio metrics =====")
    for k, v in metrics.items():
        print(f"  {k:20s} {v}")

    # Per-group breakdown
    pg = per_group_breakdown(result.trades)
    print("\n===== Per-group PnL =====")
    if not pg.empty:
        pd.set_option("display.width", 200)
        print(pg.to_string(index=False))
        pg.to_csv(OUT_DIR / f"per_group_{out_tag}.csv", index=False)

    # Per-symbol
    ps = per_symbol_breakdown(result.trades)
    if not ps.empty:
        ps.to_csv(OUT_DIR / f"per_symbol_{out_tag}.csv", index=False)
        print("\n===== Per-symbol PnL (top/bottom) =====")
        print(pd.concat([ps.head(5), ps.tail(3)]).to_string(index=False))

    # Reject reason distribution
    reject = reject_distribution(result.daily_status)
    if not reject.empty:
        reject.to_csv(OUT_DIR / f"reject_distribution_{out_tag}.csv", index=False)
        print("\n===== Reject reason distribution =====")
        print(reject.to_string(index=False))

    # Save per-trade records
    if not result.trades.empty:
        trade_cols = [c for c in KEEP_TRADE_COLS if c in result.trades.columns]
        tsub = result.trades[trade_cols + ["group_name", "strategy_id"]].copy()
        tsub.to_csv(OUT_DIR / f"trades_{out_tag}.csv", index=False)
        print(f"\n[trades saved] {len(tsub)} rows")

    # Save equity curve
    if not result.portfolio_daily.empty:
        eq = result.portfolio_daily.copy().sort_values("date")
        eq["daily_return"] = eq["equity"].pct_change()
        peak = eq["equity"].cummax()
        eq["drawdown_pct"] = (eq["equity"] - peak) / peak.where(peak > 0, np.nan)
        eq.to_csv(OUT_DIR / f"backtest_portfolio_layer_{out_tag}.csv", index=False)

    # Save summary JSON
    summary = {
        "tag": args.tag,
        "risk_per_trade": args.risk_per_trade,
        "group_cap": args.group_cap,
        "portfolio_cap": args.portfolio_cap,
        "oos_start": args.oos_start, "oos_end": args.oos_end,
        "n_stable_combos": len(stable),
        "stable_groups": sorted(stable_groups),
        "metrics": metrics,
    }
    (OUT_DIR / f"summary_{out_tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {OUT_DIR}/summary_{out_tag}.json + portfolio / per_group / per_symbol / reject / trades")


if __name__ == "__main__":
    main()
