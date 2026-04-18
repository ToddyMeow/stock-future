"""Plot portfolio equity curve + drawdown from a backtest run directory.

Reads `backtest_portfolio_layer.csv` and writes a two-panel PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RUN = Path("data/runs/09_locked_risk_v1")


def plot_equity(csv_path: Path, out_path: Path, title: str | None = None) -> None:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    initial_equity = float(df["equity"].iloc[0])
    final_equity = float(df["equity"].iloc[-1])
    total_return = final_equity / initial_equity - 1.0

    daily_ret = df["daily_return"].dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std()
    else:
        sharpe = float("nan")

    max_dd_pct = float(df["drawdown_pct"].min())  # negative number
    start_date = df["date"].iloc[0].strftime("%Y-%m-%d")
    end_date = df["date"].iloc[-1].strftime("%Y-%m-%d")

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Equity curve
    ax_eq.plot(df["date"], df["equity"], color="#1f77b4", linewidth=1.4)
    ax_eq.axhline(initial_equity, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_eq.set_ylabel("Equity")
    ax_eq.set_title(
        f"Portfolio Equity Curve — {title or csv_path.stem} ({start_date} → {end_date})"
    )
    ax_eq.grid(True, alpha=0.3)
    ax_eq.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )

    stats_text = (
        f"Initial:    {initial_equity:,.0f}\n"
        f"Final:      {final_equity:,.0f}\n"
        f"Total Ret:  {total_return*100:+.2f}%\n"
        f"Sharpe:     {sharpe:.2f}\n"
        f"Max DD:     {max_dd_pct*100:.2f}%"
    )
    ax_eq.text(
        0.015,
        0.97,
        stats_text,
        transform=ax_eq.transAxes,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    # Drawdown
    dd_pct = df["drawdown_pct"] * 100.0
    ax_dd.fill_between(df["date"], 0, dd_pct, color="#d62728", alpha=0.45)
    ax_dd.plot(df["date"], dd_pct, color="#8b0000", linewidth=0.8)
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(True, alpha=0.3)
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    print(f"Saved equity curve: {out_path}")
    print(
        f"  rows={len(df)}  initial={initial_equity:,.0f}  final={final_equity:,.0f}  "
        f"total_ret={total_return*100:+.2f}%  sharpe={sharpe:.2f}  max_dd={max_dd_pct*100:.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Direct path to a portfolio_layer CSV (preferred).",
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help=f"Legacy: run directory containing backtest_portfolio_layer.csv (default: {DEFAULT_RUN}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: next to the CSV).",
    )
    parser.add_argument("--title", type=str, default=None, help="Title override.")
    args = parser.parse_args()

    if args.csv is not None:
        csv_path = args.csv
    else:
        run_dir = args.run if args.run else DEFAULT_RUN
        csv_path = run_dir / "backtest_portfolio_layer.csv"

    out_path: Path = args.out if args.out else csv_path.with_suffix(".png")
    plot_equity(csv_path, out_path, title=args.title)


if __name__ == "__main__":
    main()
