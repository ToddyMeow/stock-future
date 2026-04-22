"""Phase 3 — Combo Selection (IS, full risk controls).

Inputs:
  - 30 Phase 2 confirmed symbols across 13 groups
  - IS period: 2018-01-01 → 2023-12-31 (6 years)
  - 30 combos per group (5 entries × 6 exits)
  - Full risk controls (risk_per_trade, group_cap, portfolio_cap)

Output (per variant):
  data/runs/phase3/combo_grid_<tag>.csv     — group × combo aggregate metrics
  data/runs/phase3/trades_<tag>.csv          — every trade (21 fields)

Usage:
  python scripts/run_phase3_combo_selection.py \\
    --risk-per-trade 0.03 --group-cap 0.06 --portfolio-cap 0.20 \\
    --is-end 2023-12-31 --output-tag risk3cap6

  # Split groups for parallelism:
  python scripts/run_phase3_combo_selection.py --groups building,commodity ...
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
from strats.factory import build_engine_config, build_entries, build_exits
from strats.research_support import load_hab_bars, yearly_stats_from_trades

OUT_DIR = ROOT / "data" / "runs" / "phase3"

# 30 Phase 2 confirmed symbols (≥ 2/3 exit_prob stable)
# NOTE: JD 手动加入（2026-04-19） — Phase 2 per-symbol B_vs_A_p=0.61 不显著,
# 但 25 万本金下 LH 被 Phase 0 剔除（单手 ATR 风险 ¥10094 > ¥7500 阈值），
# livestock 组需要 JD 支撑。JD 历史 2018-2026 完整，rolling 通过概率高。
CONFIRMED_SYMS = [
    # 3/3 stable (17)
    "AO", "AP", "BB", "CF", "CJ", "IC", "IF", "IM", "JM", "LC",
    "LG", "LH", "PS", "RI", "SA", "SI", "TL",
    # 2/3 stable (13)
    "AD", "BZ", "EC", "FB", "FG", "I", "L_F", "OP", "RR", "SH",
    "SP", "T", "V_F",
    # Manual add (1) — 25 万本金 livestock 组兜底
    "JD",
]

ACTIVE_GROUPS = [
    "building", "commodity", "equity_index", "bond", "black_steel",
    "rubber_fiber", "livestock",
    "ind_AP", "ind_BB", "ind_CJ", "ind_EC", "ind_FB", "ind_RI", "ind_RR",
]

KEEP_TRADE_COLS = [
    "symbol", "direction", "signal_date", "entry_date", "exit_date",
    "entry_fill", "exit_fill", "qty", "exit_reason",
    "mfe", "mae", "r_multiple",
    "gross_pnl", "net_pnl",
    "entry_commission_total", "exit_commission_total",
    # v10 SAR attribution (absent for pre-SAR runs, empty for SAR-off runs)
    "entry_type", "reverse_leg_count",
]


def run_combo(
    bars: pd.DataFrame,
    engine_cfg,
    entry_id: str, entry_strategy,
    exit_id: str, exit_strategy,
    group: str,
) -> tuple:
    slot = StrategySlot(
        strategy_id=f"{group}_{entry_id}+{exit_id}",
        entry_strategy=entry_strategy,
        exit_strategy=exit_strategy,
    )
    engine = StrategyEngine(config=engine_cfg, strategies=[slot])
    result = engine.run(bars)

    # Aggregate: total + per-year
    yr_stats = yearly_stats_from_trades(
        result.trades, result.portfolio_daily, engine_cfg.initial_capital
    )
    trades = result.trades

    if trades.empty:
        agg = {
            "group": group, "entry": entry_id, "exit": exit_id,
            "trades": 0, "net_pnl": 0.0, "expectancy": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "max_dd_pct": 0.0, "total_return": 0.0,
            "sharpe": 0.0, "cagr": 0.0,
            "profitable_years": 0, "n_years": 0,
            "long_trades": 0, "short_trades": 0,
        }
        return agg, []

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    n = len(trades)
    net = float(trades["net_pnl"].sum())
    gross_win = float(wins["net_pnl"].sum())
    gross_loss = float(losses["net_pnl"].sum())
    pf = (gross_win / abs(gross_loss)) if gross_loss < 0 else (999.0 if gross_win > 0 else 0.0)
    profitable_years = sum(1 for yr in yr_stats if yr["net_pnl"] > 0)

    # Portfolio stats
    pdf = result.portfolio_daily
    if not pdf.empty:
        eq = pdf["equity"]
        peak = eq.cummax()
        dd = (eq - peak) / peak.where(peak > 0, np.nan)
        max_dd = float(dd.min()) if not dd.isna().all() else 0.0
        ret_series = eq.pct_change().dropna()
        sharpe = float(ret_series.mean() / ret_series.std() * np.sqrt(252)) if ret_series.std() > 0 else 0.0
        initial = float(eq.iloc[0])
        final = float(eq.iloc[-1])
        total_ret = (final / initial - 1.0) if initial > 0 else 0.0
        days = (pdf["date"].iloc[-1] - pdf["date"].iloc[0]).days if "date" in pdf.columns else 0
        # Guard against negative final equity — fractional power of a negative
        # number returns complex; record as -1.0 (total loss) instead.
        if initial > 0 and days > 0 and final > 0:
            cagr = (final / initial) ** (365.25 / days) - 1.0
        elif initial > 0 and final <= 0:
            cagr = -1.0
        else:
            cagr = 0.0
    else:
        max_dd = sharpe = total_ret = cagr = 0.0

    agg = {
        "group": group, "entry": entry_id, "exit": exit_id,
        "trades": n,
        "net_pnl": round(net, 0),
        "expectancy": round(net / n, 1) if n > 0 else 0.0,
        "win_rate": round(len(wins) / n, 3) if n > 0 else 0.0,
        "profit_factor": round(pf, 3),
        "max_dd_pct": round(max_dd, 4),
        "total_return": round(total_ret, 4),
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr, 4),
        "profitable_years": profitable_years,
        "n_years": len(yr_stats),
        "long_trades": int((trades["direction"] == 1).sum()),
        "short_trades": int((trades["direction"] == -1).sum()),
    }

    # Per-trade records (21 cols)
    tsub = trades[[c for c in KEEP_TRADE_COLS if c in trades.columns]].copy()
    tsub["group"] = group
    tsub["entry_id"] = entry_id
    tsub["exit_id"] = exit_id
    per_trade = tsub.to_dict("records")

    return agg, per_trade


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--risk-per-trade", type=float, default=0.03)
    ap.add_argument("--group-cap", type=float, default=0.06)
    ap.add_argument("--portfolio-cap", type=float, default=0.20)
    ap.add_argument("--initial-capital", type=float, default=None,
                    help="Override engine initial_capital (default: config.yaml 1_000_000)")
    ap.add_argument("--is-end", default="2023-12-31")
    ap.add_argument("--is-start", default="2018-01-01")
    ap.add_argument("--groups", default=None,
                    help="Comma-separated groups (default: all 13 ACTIVE_GROUPS)")
    ap.add_argument("--output-tag", default="default",
                    help="Output filename suffix (e.g. 'risk3cap6')")
    ap.add_argument("--confirmed-syms-json", default=None,
                    help="JSON file with keys 'symbols' (list) and 'groups' (list) "
                         "to override CONFIRMED_SYMS / ACTIVE_GROUPS. Typically produced "
                         "from Phase 2 tradeability scoring.")
    # v10: Stop-and-reverse (SAR)
    ap.add_argument("--reverse-on-stop", action="store_true",
                    help="Enable stop-and-reverse: after any stop-out exit, "
                         "synthesize a reverse PendingEntry (opposite direction) "
                         "sized from ATR × --reverse-stop-atr-mult.")
    ap.add_argument("--reverse-stop-atr-mult", type=float, default=3.0,
                    help="ATR multiplier for SAR reverse-entry stop (default 3.0)")
    ap.add_argument("--reverse-chain-max", type=int, default=3,
                    help="Max consecutive SAR reversals per chain (default 3)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.output_tag

    if args.confirmed_syms_json:
        import json as _json
        with open(args.confirmed_syms_json) as _f:
            _override = _json.load(_f)
        confirmed_syms = list(_override["symbols"])
        active_groups = list(_override["groups"])
        print(f"  confirmed_syms override from {args.confirmed_syms_json}: "
              f"{len(confirmed_syms)} syms, {len(active_groups)} groups")
    else:
        confirmed_syms = list(CONFIRMED_SYMS)
        active_groups = list(ACTIVE_GROUPS)

    groups = [g.strip() for g in args.groups.split(",")] if args.groups else list(active_groups)
    print(f"=== Phase 3 combo selection: tag={tag} ===")
    print(f"  risk_per_trade = {args.risk_per_trade}")
    print(f"  group_cap      = {args.group_cap}")
    print(f"  portfolio_cap  = {args.portfolio_cap}")
    print(f"  IS range       = {args.is_start} → {args.is_end}")
    print(f"  groups         = {groups}")

    # Bars: filter to confirmed syms + IS date range
    bars = load_hab_bars()
    bars = bars[bars["symbol"].isin(confirmed_syms)].copy()
    is_end = pd.to_datetime(args.is_end)
    is_start = pd.to_datetime(args.is_start)
    bars = bars[(bars["date"] >= is_start) & (bars["date"] <= is_end)].copy()
    print(f"  bars: {len(bars)} rows, {bars['symbol'].nunique()} syms, "
          f"{bars['date'].min().date()} → {bars['date'].max().date()}")

    risk_overrides = {
        "risk_per_trade": args.risk_per_trade,
        "portfolio_risk_cap": args.portfolio_cap,
        "group_risk_cap": {g: args.group_cap for g in active_groups},
        "default_group_risk_cap": args.group_cap,
    }
    if args.initial_capital is not None:
        risk_overrides["initial_capital"] = args.initial_capital
        print(f"  initial_capital= {args.initial_capital:,.0f} (override)")
    if args.reverse_on_stop:
        risk_overrides["reverse_on_stop"] = True
        risk_overrides["reverse_stop_atr_mult"] = args.reverse_stop_atr_mult
        risk_overrides["reverse_chain_max"] = args.reverse_chain_max
        print(f"  SAR enabled    = ATR×{args.reverse_stop_atr_mult}, chain_max={args.reverse_chain_max}")
    engine_cfg = build_engine_config(profile="research", overrides=risk_overrides)
    print(f"  engine.group_risk_cap (sample) = "
          f"{list(engine_cfg.group_risk_cap.items())[:3]}")

    entries = build_entries(include_adaptive=False)
    exits = build_exits()
    print(f"  combos: {len(entries)} entries × {len(exits)} exits = {len(entries)*len(exits)}")

    all_aggs: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []

    t0 = time.time()
    total = len(groups) * len(entries) * len(exits)
    done = 0

    for group in groups:
        group_bars = bars[bars["group_name"] == group]
        if group_bars.empty:
            print(f"  [{group}] SKIP — no data")
            continue

        for eid, estrategy in entries.items():
            for xid, xstrategy in exits.items():
                done += 1
                tag_log = f"[{done}/{total}] tag={tag} {group} {eid}+{xid}"
                try:
                    agg, trades = run_combo(
                        group_bars, engine_cfg, eid, estrategy, xid, xstrategy, group,
                    )
                    all_aggs.append(agg)
                    all_trades.extend(trades)
                    print(f"  {tag_log} — {agg['trades']} trades, exp={agg['expectancy']:.0f}")
                except Exception as e:
                    print(f"  {tag_log} — ERROR: {e}")
                    traceback.print_exc()

    elapsed = time.time() - t0

    agg_df = pd.DataFrame(all_aggs)
    agg_out = OUT_DIR / f"combo_grid_{tag}.csv"
    agg_df.to_csv(agg_out, index=False)
    print(f"\nSaved {agg_out}  ({len(agg_df)} rows)")

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_out = OUT_DIR / f"trades_{tag}.csv"
        trades_df.to_csv(trades_out, index=False)
        print(f"Saved {trades_out}  ({len(trades_df)} trade rows)")

    print(f"\nElapsed: {elapsed:.0f}s / {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
