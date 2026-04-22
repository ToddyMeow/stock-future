"""Phase 2 Monte Carlo analyzer — per-symbol per-seed distribution.

Extends analyze_phase2.py:
  - For each symbol × experiment, per-seed expectancy / win_rate / wl_ratio
  - Distribution across seeds: mean, std, 5/50/95 percentile
  - Pooled bootstrap permutation test B_best vs A (one-sided 5%)
  - Hurst exponent
  - Stability score: fraction of seeds where B > A on same seed

Reads: trades_{A,B}_symbol.csv
Writes: phase2_per_symbol.csv, phase2_mc_distribution.csv
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
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < max_window * 2:
        max_window = max(min_window * 2, len(series) // 4)
    if len(series) < min_window * 4:
        return np.nan
    lr = np.diff(np.log(series))
    n = len(lr)
    windows = []
    w = min_window
    while w <= max_window and w * 4 <= n:
        windows.append(w); w = int(w * 1.5)
    rs_values = []
    for w in windows:
        k = n // w
        rs_list = []
        for i in range(k):
            chunk = lr[i * w:(i + 1) * w]
            dev = chunk - chunk.mean()
            cumdev = np.cumsum(dev)
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1)
            if S > 0 and R > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((w, np.mean(rs_list)))
    if len(rs_values) < 3:
        return np.nan
    slope, _ = np.polyfit(np.log([x[0] for x in rs_values]),
                          np.log([x[1] for x in rs_values]), 1)
    return float(slope)


def wr_wl(trades: np.ndarray) -> tuple:
    n = len(trades)
    if n == 0:
        return 0, 0.0, 0.0
    wins = trades[trades > 0]
    losses = trades[trades <= 0]
    wr = len(wins) / n
    wl = (wins.mean() / abs(losses.mean())) if len(wins) > 0 and len(losses) > 0 and losses.mean() < 0 else 0.0
    return n, float(wr), float(wl)


def permutation_test(tB: np.ndarray, tA: np.ndarray, n_boot: int,
                     rng: np.random.Generator) -> tuple:
    obs = float(tB.mean() - tA.mean())
    combined = np.concatenate([tB, tA])
    n_b = len(tB)
    c = 0
    for _ in range(n_boot):
        rng.shuffle(combined)
        if combined[:n_b].mean() - combined[n_b:].mean() >= obs:
            c += 1
    return obs, c / n_boot


def main() -> None:
    tA_path = ABENCH / "trades_A_symbol.csv"
    tB_path = ABENCH / "trades_B_symbol.csv"
    if not (tA_path.exists() and tB_path.exists()):
        print("ERROR: missing symbol-scope trade data")
        sys.exit(1)
    tA = pd.read_csv(tA_path)
    tB = pd.read_csv(tB_path)
    n_seeds_A = tA["seed"].nunique()
    n_seeds_B = tB["seed"].nunique()
    print(f"Loaded:  A={len(tA)} ({n_seeds_A} seeds)  B={len(tB)} ({n_seeds_B} seeds)")

    labels = pd.read_csv(ABENCH / "phase1_labels.csv")
    group_label = dict(zip(labels["group"], labels.get("label_v2", labels.get("label"))))

    bars = pd.read_csv(ROOT / "data" / "cache" / "normalized" / "hab_bars.csv",
                       usecols=["date", "symbol", "close"])
    bars["date"] = pd.to_datetime(bars["date"])

    rng = np.random.default_rng(42)
    rows = []
    dist_rows = []

    syms = sorted(set(tA["symbol"]) | set(tB["symbol"]))
    for sym in syms:
        a_sub = tA[tA["symbol"] == sym]
        b_sub = tB[tB["symbol"] == sym]
        if a_sub.empty or b_sub.empty:
            continue
        group = a_sub["group"].iloc[0]

        a_pnl = a_sub["net_pnl"].to_numpy(dtype=float)

        # Best B exit (pool across seeds)
        best_exit, best_pnl, best_mean = None, None, -np.inf
        for exit_id, g in b_sub.groupby("exit"):
            pnl = g["net_pnl"].to_numpy(dtype=float)
            if len(pnl) < MIN_TRADES:
                continue
            m = pnl.mean()
            if m > best_mean:
                best_mean, best_exit, best_pnl = m, exit_id, pnl

        if best_pnl is None or len(a_pnl) < MIN_TRADES:
            rows.append({"group": group, "symbol": sym, "group_label": group_label.get(group),
                         "status": "insufficient"})
            continue

        a_n, a_wr, a_wl = wr_wl(a_pnl)
        b_n, b_wr, b_wl = wr_wl(best_pnl)
        obs_diff, p_val = permutation_test(best_pnl, a_pnl, N_BOOTSTRAP, rng)

        # Per-seed distribution for (A) and (B at best_exit)
        a_per_seed = a_sub.groupby("seed")["net_pnl"].agg(["size", "sum", "mean"]).reset_index()
        a_per_seed.columns = ["seed", "trades", "pnl", "expectancy"]
        b_at_exit = b_sub[b_sub["exit"] == best_exit]
        b_per_seed = b_at_exit.groupby("seed")["net_pnl"].agg(["size", "sum", "mean"]).reset_index()
        b_per_seed.columns = ["seed", "trades", "pnl", "expectancy"]

        a_exp_series = a_per_seed["expectancy"].to_numpy(dtype=float)
        b_exp_series = b_per_seed["expectancy"].to_numpy(dtype=float)

        # Stability: fraction of seeds where B > A (per-seed)
        common = pd.merge(a_per_seed, b_per_seed, on="seed", suffixes=("_a", "_b"))
        stability = (common["expectancy_b"] > common["expectancy_a"]).mean() if not common.empty else None

        # Hurst
        sym_close = bars[bars["symbol"] == sym].sort_values("date")["close"].to_numpy()
        h = hurst_rs(sym_close)

        # Tradeability score (same as before)
        score = 0
        sig_exit = (p_val < SIG_P) and (obs_diff > 0)
        if sig_exit: score += 40
        elif (p_val < 0.10) and (obs_diff > 0): score += 20
        if b_wl > 1.5: score += 20
        elif b_wl > 1.2: score += 10
        if b_wr > 0.35: score += 10
        if h is not None and np.isfinite(h):
            if h > 0.55: score += 20
            elif h > 0.45: score += 10

        rows.append({
            "group": group, "symbol": sym, "group_label": group_label.get(group),
            "A_trades": a_n, "A_wr": round(a_wr, 3), "A_wl": round(a_wl, 2), "A_mean": round(a_pnl.mean(), 1),
            "best_B_exit": best_exit,
            "B_trades": b_n, "B_wr": round(b_wr, 3), "B_wl": round(b_wl, 2), "B_mean": round(best_pnl.mean(), 1),
            "B_minus_A": round(obs_diff, 1),
            "B_vs_A_p": round(p_val, 4),
            "exit_sig": sig_exit,
            "n_seeds_A": len(a_exp_series), "n_seeds_B": len(b_exp_series),
            "A_seed_mean": round(a_exp_series.mean(), 1) if len(a_exp_series) else None,
            "A_seed_std": round(a_exp_series.std(ddof=1), 1) if len(a_exp_series) > 1 else 0,
            "B_seed_mean": round(b_exp_series.mean(), 1) if len(b_exp_series) else None,
            "B_seed_std": round(b_exp_series.std(ddof=1), 1) if len(b_exp_series) > 1 else 0,
            "B_beats_A_fraction": round(stability, 2) if stability is not None else None,
            "hurst": round(h, 3) if h is not None and np.isfinite(h) else None,
            "tradeability_score": score,
        })

        # Save per-seed distribution for chart-friendly analysis
        for _, r in a_per_seed.iterrows():
            dist_rows.append({"group": group, "symbol": sym, "experiment": "A",
                              "seed": int(r["seed"]), "trades": int(r["trades"]),
                              "expectancy": round(float(r["expectancy"]), 1)})
        for _, r in b_per_seed.iterrows():
            dist_rows.append({"group": group, "symbol": sym, "experiment": f"B_{best_exit}",
                              "seed": int(r["seed"]), "trades": int(r["trades"]),
                              "expectancy": round(float(r["expectancy"]), 1)})

    df = pd.DataFrame(rows).sort_values(
        ["tradeability_score", "group", "symbol"], ascending=[False, True, True]
    ).reset_index(drop=True)
    df.to_csv(ABENCH / "phase2_per_symbol.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(ABENCH / "phase2_mc_distribution.csv", index=False)

    print("\n===== Phase 2 MC — significant symbols (p < 0.05) =====")
    sig = df[df.get("exit_sig") == True].copy() if "exit_sig" in df.columns else pd.DataFrame()
    print(f"  {len(sig)} / {len(df)} symbols pass")
    if not sig.empty:
        cols = ["group", "symbol", "group_label", "B_vs_A_p", "B_wr", "B_wl",
                "A_seed_mean", "A_seed_std", "B_seed_mean", "B_seed_std",
                "B_beats_A_fraction", "hurst", "tradeability_score"]
        print(sig[cols].to_string(index=False))

    print("\n===== Score distribution =====")
    print(df["tradeability_score"].describe().to_string())

    print(f"\n[saved] {ABENCH}/phase2_per_symbol.csv, phase2_mc_distribution.csv")


if __name__ == "__main__":
    main()
