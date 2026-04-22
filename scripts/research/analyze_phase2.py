"""Phase 2 — per-symbol drill-down on alpha-clear groups.

For each symbol in the 13 alpha-clear groups (from Phase 1 labels):
  - per-trade win_rate, wl_ratio (avg_winner / avg_loser)
  - Bootstrap permutation test: best_B vs A (one-sided p<0.05)
  - Hurst exponent on close log-returns (cross-validation of trend structure)
  - Tradeability score combining all signals

Reads:
  data/runs/alpha_benchmark/trades_A_symbol.csv
  data/runs/alpha_benchmark/trades_B_symbol.csv
  data/runs/alpha_benchmark/phase1_labels.csv
  data/cache/normalized/hab_bars.csv

Writes:
  data/runs/alpha_benchmark/phase2_per_symbol.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ABENCH = ROOT / "data" / "runs" / "alpha_benchmark"

MIN_TRADES = 30
N_BOOTSTRAP = 5000
SIG_P = 0.05


def hurst_rs(series: np.ndarray, min_window: int = 16, max_window: int = 512) -> float:
    """Hurst exponent via R/S analysis on log returns.

    Returns H in [0, 1]:
      H ≈ 0.5 → random walk (efficient)
      H > 0.5 → trending / persistent
      H < 0.5 → mean-reverting / anti-persistent
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < max_window * 2:
        max_window = max(min_window * 2, len(series) // 4)
    if len(series) < min_window * 4:
        return np.nan

    # Use log returns
    lr = np.diff(np.log(series))
    n = len(lr)

    windows = []
    w = min_window
    while w <= max_window and w * 4 <= n:
        windows.append(w)
        w = int(w * 1.5)

    rs_values = []
    for w in windows:
        # Split series into non-overlapping windows of size w
        k = n // w
        rs_list = []
        for i in range(k):
            chunk = lr[i * w:(i + 1) * w]
            mean = chunk.mean()
            dev = chunk - mean
            cumdev = np.cumsum(dev)
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1)
            if S > 0 and R > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((w, np.mean(rs_list)))

    if len(rs_values) < 3:
        return np.nan
    log_w = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])
    # Linear fit slope = H
    slope, _ = np.polyfit(log_w, log_rs, 1)
    return float(slope)


def compute_win_rate_wl(trades: np.ndarray) -> tuple:
    """Returns (trades_count, win_rate, wl_ratio)."""
    n = len(trades)
    if n == 0:
        return 0, 0.0, 0.0
    wins = trades[trades > 0]
    losses = trades[trades <= 0]
    win_rate = len(wins) / n
    if len(wins) > 0 and len(losses) > 0 and losses.mean() < 0:
        wl_ratio = wins.mean() / abs(losses.mean())
    else:
        wl_ratio = 0.0
    return n, float(win_rate), float(wl_ratio)


def permutation_test(trades_b: np.ndarray, trades_a: np.ndarray,
                     n_bootstrap: int, rng: np.random.Generator) -> tuple:
    observed_diff = float(trades_b.mean() - trades_a.mean())
    combined = np.concatenate([trades_b, trades_a])
    n_b = len(trades_b)
    count = 0
    for _ in range(n_bootstrap):
        rng.shuffle(combined)
        if combined[:n_b].mean() - combined[n_b:].mean() >= observed_diff:
            count += 1
    return observed_diff, count / n_bootstrap


def main() -> None:
    # Load per-trade data
    tA_path = ABENCH / "trades_A_symbol.csv"
    tB_path = ABENCH / "trades_B_symbol.csv"
    if not (tA_path.exists() and tB_path.exists()):
        print("ERROR: trades_A_symbol.csv and/or trades_B_symbol.csv missing")
        sys.exit(1)
    tA = pd.read_csv(tA_path)
    tB = pd.read_csv(tB_path)
    print(f"Loaded per-trade data:  A={len(tA)}  B={len(tB)}")
    print(f"Syms: A={tA['symbol'].nunique()}  B={tB['symbol'].nunique()}")

    # Load Phase 1 labels for group annotation
    labels = pd.read_csv(ABENCH / "phase1_labels.csv")
    group_label = dict(zip(labels["group"], labels.get("label_v2", labels.get("label"))))

    # Load bars for Hurst
    bars = pd.read_csv(ROOT / "data" / "cache" / "normalized" / "hab_bars.csv",
                       usecols=["date", "symbol", "close"])
    bars["date"] = pd.to_datetime(bars["date"])

    rng = np.random.default_rng(42)
    rows = []

    syms = sorted(set(tA["symbol"]) | set(tB["symbol"]))
    print(f"\nPer-symbol: {len(syms)} symbols")

    for sym in syms:
        a_sub = tA[tA["symbol"] == sym]
        b_sub = tB[tB["symbol"] == sym]
        if a_sub.empty or b_sub.empty:
            continue
        group = a_sub["group"].iloc[0]

        a_pnl = a_sub["net_pnl"].to_numpy(dtype=float)

        # Best B exit for this symbol (by mean, ≥ MIN_TRADES)
        best_exit, best_pnl, best_mean = None, None, -np.inf
        for exit_id, g in b_sub.groupby("exit"):
            pnl = g["net_pnl"].to_numpy(dtype=float)
            if len(pnl) < MIN_TRADES:
                continue
            m = pnl.mean()
            if m > best_mean:
                best_mean = m
                best_exit = exit_id
                best_pnl = pnl

        if best_pnl is None or len(a_pnl) < MIN_TRADES:
            rows.append({
                "group": group, "symbol": sym, "group_label": group_label.get(group),
                "status": "insufficient",
            })
            continue

        # A stats
        a_n, a_wr, a_wl = compute_win_rate_wl(a_pnl)
        # B best stats
        b_n, b_wr, b_wl = compute_win_rate_wl(best_pnl)

        # Bootstrap test
        obs_diff, p_val = permutation_test(best_pnl, a_pnl, N_BOOTSTRAP, rng)

        # Hurst
        sym_close = bars[bars["symbol"] == sym].sort_values("date")["close"].to_numpy()
        h = hurst_rs(sym_close)

        # Tradeability score: 0-100
        # - exit has alpha (p<0.05, diff>0): +40
        # - B wl_ratio > 1.5: +20
        # - B win_rate > 0.35: +10
        # - Hurst > 0.55 (trending): +20, 0.45-0.55: +10, <0.45: 0
        # - obs_diff > 0 & p<0.10 (weak signal): +10 if p >= 0.05 didn't qualify
        score = 0
        sig_exit = (p_val < SIG_P) and (obs_diff > 0)
        if sig_exit:
            score += 40
        elif (p_val < 0.10) and (obs_diff > 0):
            score += 20
        if b_wl > 1.5:
            score += 20
        elif b_wl > 1.2:
            score += 10
        if b_wr > 0.35:
            score += 10
        if h is not None and np.isfinite(h):
            if h > 0.55:
                score += 20
            elif h > 0.45:
                score += 10

        rows.append({
            "group": group, "symbol": sym, "group_label": group_label.get(group),
            "A_trades": a_n, "A_win_rate": round(a_wr, 3), "A_wl_ratio": round(a_wl, 2),
            "A_mean": round(a_pnl.mean(), 1),
            "best_B_exit": best_exit,
            "B_trades": b_n, "B_win_rate": round(b_wr, 3), "B_wl_ratio": round(b_wl, 2),
            "B_mean": round(best_pnl.mean(), 1),
            "B_minus_A_mean": round(obs_diff, 1),
            "B_vs_A_p": round(p_val, 4),
            "exit_sig": sig_exit,
            "hurst": round(h, 3) if h is not None and np.isfinite(h) else None,
            "tradeability_score": score,
        })

    df = pd.DataFrame(rows).sort_values(
        ["tradeability_score", "group", "symbol"], ascending=[False, True, True]
    ).reset_index(drop=True)
    df.to_csv(ABENCH / "phase2_per_symbol.csv", index=False)

    print("\n===== Phase 2 per-symbol tradeability =====")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False))

    print("\n===== Score distribution =====")
    if "tradeability_score" in df.columns:
        print(df["tradeability_score"].describe().to_string())

    print("\n===== Symbols with significant exit alpha (p < 0.05) =====")
    sig = df[df["exit_sig"] == True] if "exit_sig" in df.columns else pd.DataFrame()
    print(f"  {len(sig)} / {len(df)} symbols pass")
    if not sig.empty:
        print(sig[["group", "symbol", "B_vs_A_p", "B_win_rate", "B_wl_ratio", "hurst",
                   "tradeability_score"]].to_string(index=False))

    print(f"\n[saved] {ABENCH}/phase2_per_symbol.csv")


if __name__ == "__main__":
    main()
