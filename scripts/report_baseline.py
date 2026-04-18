"""Generate a markdown report from the baseline three-layer backtest.

Re-runs Layer 3 (needed for full BacktestResult — the three_layer script only
saves aggregated CSVs), then emits a markdown report with overall stats,
per-group contribution, per-year / per-month returns, signal density, risk
utilisation and data-quality snapshot.

Also supports --start / --end for regime slicing (Step 6 of the plan).

Usage:
    python scripts/report_baseline.py --suffix baseline
    python scripts/report_baseline.py --suffix baseline --start 2020-01-01 --end 2020-06-30
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_three_layer_backtest import (
    build_entries, build_exits, make_engine_config, load_bars,
)
from strats.engine import StrategyEngine, StrategySlot
from strats.helpers import PortfolioAnalyzer


def derive_best_combos(group_csv: Path) -> dict[str, str]:
    """Read Layer 2 CSV and extract {group: best_combo}."""
    df = pd.read_csv(group_csv)
    return df.groupby("group")["best_combo"].first().to_dict()


def filter_result_by_date(result, start: pd.Timestamp, end: pd.Timestamp):
    """Return a shallow-copied BacktestResult clipped to [start, end]."""
    pdf = result.portfolio_daily
    if not pdf.empty:
        d = pd.to_datetime(pdf["date"])
        pdf = pdf[(d >= start) & (d <= end)].reset_index(drop=True)
    trades = result.trades
    if not trades.empty and "entry_date" in trades.columns:
        d = pd.to_datetime(trades["entry_date"])
        trades = trades[(d >= start) & (d <= end)].reset_index(drop=True)
    ds = result.daily_status
    if not ds.empty and "date" in ds.columns:
        d = pd.to_datetime(ds["date"])
        ds = ds[(d >= start) & (d <= end)].reset_index(drop=True)
    return dataclasses.replace(result, portfolio_daily=pdf, trades=trades, daily_status=ds)


PCT_FIELDS = {"total_return", "cagr", "max_drawdown_pct", "win_rate"}


def format_stats_table(stats: dict) -> str:
    if not stats:
        return "_No data._\n"
    fields = [
        "total_return", "cagr", "sharpe", "sortino", "max_drawdown_pct",
        "total_trades", "win_rate", "profit_factor", "avg_r_multiple",
        "expectancy", "total_days",
    ]
    lines = ["| metric | value |", "|---|---|"]
    for f in fields:
        v = stats.get(f)
        if v is None:
            continue
        if f in PCT_FIELDS:
            lines.append(f"| {f} | {v*100:+.2f}% |")
        elif isinstance(v, float):
            lines.append(f"| {f} | {v:.4f} |")
        else:
            lines.append(f"| {f} | {v} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="baseline",
                        help="Baseline output suffix (default: baseline)")
    parser.add_argument("--start", default=None, help="Regime start YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Regime end YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=None, help="Output .md path")
    parser.add_argument("--adx-off", action="store_true")
    parser.add_argument("--risk-per-trade", type=float, default=None,
                        help="Override risk_per_trade (must match the original backtest run).")
    parser.add_argument("--portfolio-risk-cap", type=float, default=None,
                        help="Override portfolio_risk_cap (must match the original backtest run).")
    parser.add_argument("--stop-atr-mult", type=float, default=None,
                        help="Override stop_atr_mult (must match the original backtest run).")
    args = parser.parse_args()

    file_suffix = f"_{args.suffix}" if args.suffix else ""
    group_csv = ROOT / "data" / f"backtest_group_layer{file_suffix}.csv"
    if not group_csv.exists():
        print(f"ERROR: {group_csv} not found — run baseline first", file=sys.stderr)
        sys.exit(1)

    best_combos = derive_best_combos(group_csv)
    print(f"Best combos ({len(best_combos)} groups):")
    for g, c in best_combos.items():
        print(f"  {g}: {c}")

    # Re-run Layer 3 to recover full BacktestResult.
    # Any CLI overrides MUST match the original run to get consistent metrics.
    bars = load_bars()
    risk_overrides: dict = {}
    if args.risk_per_trade is not None:
        risk_overrides["risk_per_trade"] = args.risk_per_trade
    if args.portfolio_risk_cap is not None:
        risk_overrides["portfolio_risk_cap"] = args.portfolio_risk_cap
    if args.stop_atr_mult is not None:
        risk_overrides["stop_atr_mult"] = args.stop_atr_mult
    engine_cfg = make_engine_config(adx_off=args.adx_off, risk_overrides=risk_overrides or None)
    entries = build_entries()
    exits = build_exits()
    slots = [
        StrategySlot(
            strategy_id=f"{g}_{c}",
            entry_strategy=entries[c.split("+")[0]],
            exit_strategy=exits[c.split("+")[1]],
        )
        for g, c in best_combos.items()
    ]
    included_groups = set(best_combos.keys())
    portfolio_bars = bars[bars["group_name"].isin(included_groups)].copy()

    print(f"\nRerunning Layer 3 engine ({len(slots)} slots, "
          f"{portfolio_bars['symbol'].nunique()} symbols)...")
    engine = StrategyEngine(config=engine_cfg, strategies=slots)
    result = engine.run(portfolio_bars)

    # Optional regime slicing
    label = args.suffix
    if args.start or args.end:
        start = pd.to_datetime(args.start) if args.start else pd.Timestamp.min
        end = pd.to_datetime(args.end) if args.end else pd.Timestamp.max
        result = filter_result_by_date(result, start, end)
        label = f"{args.suffix}_{args.start or 'begin'}_to_{args.end or 'end'}"

    analyzer = PortfolioAnalyzer(result, engine_cfg)
    stats = analyzer.summary_stats()

    # Build markdown
    now = datetime.now().strftime("%Y-%m-%d")
    out_path = args.out or (ROOT / "data" / "reports" / f"baseline_{label}_{now}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md: list[str] = []
    md.append(f"# 基线回测报告 — {label}\n")
    md.append(f"_生成时间: {now}  |  ADX: {'OFF' if args.adx_off else 'ON'}_\n")
    md.append("")
    md.append("**各组最优 combo:**\n")
    md.append("```json")
    md.append(json.dumps(best_combos, indent=2, ensure_ascii=False))
    md.append("```\n")

    md.append("## 总体指标\n")
    md.append(format_stats_table(stats))

    md.append("## 按组贡献\n")
    gc = analyzer.group_contribution()
    if not gc.empty:
        md.append(gc.to_markdown(index=False, floatfmt=".4f"))
        md.append("")
    else:
        md.append("_No trades._\n")

    md.append("\n## 按年回报\n")
    yearly = analyzer.periodic_returns(freq="Y")
    if not yearly.empty:
        yearly_fmt = yearly.copy()
        yearly_fmt["period"] = pd.to_datetime(yearly_fmt["period"]).dt.year
        yearly_fmt["return_pct"] = (yearly_fmt["return_pct"] * 100).round(2)
        md.append(yearly_fmt.to_markdown(index=False))
        md.append("")
    else:
        md.append("_No data._\n")

    monthly = analyzer.periodic_returns(freq="M")
    if not monthly.empty:
        wins = int((monthly["return_pct"] > 0).sum())
        total = len(monthly)
        best = monthly.loc[monthly["return_pct"].idxmax()]
        worst = monthly.loc[monthly["return_pct"].idxmin()]
        md.append("\n## 按月统计\n")
        md.append(f"- 盈利月占比: **{wins/total:.1%}** ({wins}/{total})")
        md.append(f"- 最佳月: {pd.to_datetime(best['period']).strftime('%Y-%m')} = {best['return_pct']*100:+.2f}%")
        md.append(f"- 最差月: {pd.to_datetime(worst['period']).strftime('%Y-%m')} = {worst['return_pct']*100:+.2f}%")

    risk_util = analyzer.risk_utilization()
    if not risk_util.empty:
        md.append("\n## 风险利用 / 杠杆\n")
        md.append(f"- 平均风险利用率: {risk_util['risk_usage_pct'].mean():.2%}")
        md.append(f"- 中位风险利用率: {risk_util['risk_usage_pct'].median():.2%}")
        md.append(f"- 平均杠杆:       {risk_util['leverage'].mean():.2f}x")
        md.append(f"- 最高杠杆:       {risk_util['leverage'].max():.2f}x")

    sd = analyzer.signal_density()
    if not sd.empty:
        fired = int(sd["signals_fired"].sum())
        accepted = int(sd["signals_accepted"].sum())
        md.append("\n## 信号密度\n")
        md.append(f"- 总信号触发: {fired}")
        md.append(f"- 通过风控接受: {accepted}")
        if fired:
            md.append(f"- 拒绝率: {(1 - accepted/fired):.1%}")

    dq = getattr(result, "data_quality_report", pd.DataFrame())
    if isinstance(dq, pd.DataFrame) and not dq.empty:
        md.append("\n## 数据质量（问题最严重的 10 品种）\n")
        sort_col = "lock_pct" if "lock_pct" in dq.columns else dq.columns[1]
        worst = dq.sort_values(sort_col, ascending=False).head(10)
        md.append(worst.to_markdown(index=False, floatfmt=".4f"))

    dd_eps = analyzer.drawdown_episodes()
    if not dd_eps.empty:
        worst5 = dd_eps.sort_values("max_drawdown_pct").head(5)
        md.append("\n## 最深 5 次回撤\n")
        md.append(worst5.to_markdown(index=False))

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved report: {out_path}")
    print(f"  summary: Sharpe={stats.get('sharpe', 0):.3f}  "
          f"CAGR={stats.get('cagr', 0)*100:+.2f}%  "
          f"MaxDD={stats.get('max_drawdown_pct', 0)*100:+.2f}%  "
          f"Trades={stats.get('total_trades', 0)}")


if __name__ == "__main__":
    main()
