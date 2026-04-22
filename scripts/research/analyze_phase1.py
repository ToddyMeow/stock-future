"""Phase 1 analyzer: label each group by alpha source.

Five categories (user's pipeline spec):
  drift_only  : A's per-trade mean is significantly > 0 (bootstrap CI),
                but neither best-B nor best-C beats A significantly
  exit_alpha  : best-B > A significantly (bootstrap permutation p < 0.05)
                and best-C not significantly above A
  entry_alpha : best-C > A significantly, best-B not significantly above A
  both        : both best-B > A and best-C > A significantly
  none        : neither best-B nor best-C beats A (and A not drift-positive)

Selection constraint:
  - A group is only labelled if A_trades >= 30 AND selected "best" B/C/D
    pooled trades >= 30. Otherwise: "insufficient".

Significance:
  - Two-sample: permutation test (user's pseudocode), n_bootstrap=5000,
    one-sided (testing diff > 0), p_value < 0.05.
  - One-sample (drift): bootstrap 2.5% / 97.5% percentile of A's mean
    across resamples with replacement.

Reads:
  data/runs/alpha_benchmark/trades_{A,B,C,D}.csv (per-trade tagged dumps)

Writes:
  data/runs/alpha_benchmark/phase1_labels.csv
"""
from __future__ import annotations

import argparse
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


def load_trades(name: str) -> pd.DataFrame:
    p = ABENCH / f"trades_{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def permutation_test(trades_d: np.ndarray, trades_a: np.ndarray,
                     n_bootstrap: int = N_BOOTSTRAP,
                     rng: np.random.Generator = None) -> tuple:
    """One-sided permutation test: is mean(trades_d) > mean(trades_a)?

    Returns (observed_diff, p_value).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    exp_d = float(trades_d.mean())
    exp_a = float(trades_a.mean())
    observed_diff = exp_d - exp_a

    combined = np.concatenate([trades_d, trades_a])
    n_d = len(trades_d)
    count_exceed = 0
    for _ in range(n_bootstrap):
        rng.shuffle(combined)
        boot_d = combined[:n_d].mean()
        boot_a = combined[n_d:].mean()
        if boot_d - boot_a >= observed_diff:
            count_exceed += 1
    p_value = count_exceed / n_bootstrap
    return observed_diff, p_value


def drift_test(trades_a: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
               rng: np.random.Generator = None) -> tuple:
    """Bootstrap CI for A's mean per-trade P&L. Returns (mean, p2.5, p97.5)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(trades_a)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(trades_a, size=n, replace=True)
        means[i] = sample.mean()
    return float(trades_a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pool_trades(df: pd.DataFrame, group_cols: list) -> dict:
    """Returns dict keyed by tuple of group_cols values → np.ndarray of net_pnl."""
    out = {}
    for key, sub in df.groupby(group_cols):
        out[key] = sub["net_pnl"].to_numpy(dtype=float)
    return out


def best_by_expectancy(pooled: dict, min_trades: int = MIN_TRADES) -> tuple:
    """Given {key: pnl_array}, pick key with max mean (among those meeting min_trades)."""
    best_key, best_mean, best_arr = None, -np.inf, None
    for k, arr in pooled.items():
        if len(arr) < min_trades:
            continue
        m = arr.mean()
        if m > best_mean:
            best_mean, best_key, best_arr = m, k, arr
    return best_key, best_arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--p-threshold", type=float, default=SIG_P)
    ap.add_argument("--min-trades", type=int, default=MIN_TRADES)
    args = ap.parse_args()

    tA = load_trades("A")
    tB = load_trades("B")
    tC = load_trades("C")
    tD = load_trades("D")
    for name, df in [("A", tA), ("B", tB), ("C", tC), ("D", tD)]:
        if df.empty:
            print(f"ERROR: trades_{name}.csv missing or empty")
            sys.exit(1)
    print(f"Loaded per-trade dumps:  A={len(tA)}  B={len(tB)}  C={len(tC)}  D={len(tD)}")
    print(f"Settings: min_trades={args.min_trades}, n_bootstrap={args.n_bootstrap}, "
          f"p_threshold={args.p_threshold}")

    groups = sorted(set(tA["group"]) | set(tB["group"]) | set(tC["group"]) | set(tD["group"]))
    print(f"Groups: {len(groups)}")

    rng = np.random.default_rng(42)
    rows = []

    for g in groups:
        a_trades = tA[tA["group"] == g]["net_pnl"].to_numpy(dtype=float)
        if len(a_trades) < args.min_trades:
            rows.append({"group": g, "label": "insufficient",
                         "A_trades": len(a_trades), "reason": "A<min_trades"})
            continue

        # Pool B per (exit), C per (entry), D per (entry, exit)
        b_pool = pool_trades(tB[tB["group"] == g], ["exit"])
        c_pool = pool_trades(tC[tC["group"] == g], ["entry"])
        d_pool = pool_trades(tD[tD["group"] == g], ["entry", "exit"])

        best_B_key, best_B_arr = best_by_expectancy(b_pool, args.min_trades)
        best_C_key, best_C_arr = best_by_expectancy(c_pool, args.min_trades)
        best_D_key, best_D_arr = best_by_expectancy(d_pool, args.min_trades)

        # Two-sample permutation tests
        b_diff, b_p = (None, None)
        if best_B_arr is not None:
            b_diff, b_p = permutation_test(best_B_arr, a_trades,
                                            n_bootstrap=args.n_bootstrap, rng=rng)
        c_diff, c_p = (None, None)
        if best_C_arr is not None:
            c_diff, c_p = permutation_test(best_C_arr, a_trades,
                                            n_bootstrap=args.n_bootstrap, rng=rng)
        d_diff, d_p = (None, None)
        if best_D_arr is not None:
            d_diff, d_p = permutation_test(best_D_arr, a_trades,
                                            n_bootstrap=args.n_bootstrap, rng=rng)

        # One-sample drift test for A
        a_mean, a_lo, a_hi = drift_test(a_trades, n_bootstrap=args.n_bootstrap, rng=rng)
        drift_positive = a_lo > 0

        has_exit = (b_p is not None) and b_p < args.p_threshold and b_diff > 0
        has_entry = (c_p is not None) and c_p < args.p_threshold and c_diff > 0

        if has_exit and has_entry:
            label = "both"
        elif has_exit:
            label = "exit_alpha"
        elif has_entry:
            label = "entry_alpha"
        elif drift_positive:
            label = "drift_only"
        else:
            label = "none"

        rows.append({
            "group": g,
            "A_trades": len(a_trades),
            "A_mean": round(a_mean, 1),
            "A_CI_lo": round(a_lo, 1),
            "A_CI_hi": round(a_hi, 1),
            "best_B_exit": best_B_key[0] if best_B_key else None,
            "B_trades": len(best_B_arr) if best_B_arr is not None else 0,
            "B_mean": round(best_B_arr.mean(), 1) if best_B_arr is not None else None,
            "B_diff": round(b_diff, 1) if b_diff is not None else None,
            "B_p": round(b_p, 4) if b_p is not None else None,
            "best_C_entry": best_C_key[0] if best_C_key else None,
            "C_trades": len(best_C_arr) if best_C_arr is not None else 0,
            "C_mean": round(best_C_arr.mean(), 1) if best_C_arr is not None else None,
            "C_diff": round(c_diff, 1) if c_diff is not None else None,
            "C_p": round(c_p, 4) if c_p is not None else None,
            "best_D_combo": f"{best_D_key[0]}+{best_D_key[1]}" if best_D_key else None,
            "D_trades": len(best_D_arr) if best_D_arr is not None else 0,
            "D_mean": round(best_D_arr.mean(), 1) if best_D_arr is not None else None,
            "D_diff": round(d_diff, 1) if d_diff is not None else None,
            "D_p": round(d_p, 4) if d_p is not None else None,
            "label": label,
        })

    df = pd.DataFrame(rows).sort_values(["label", "group"]).reset_index(drop=True)
    df.to_csv(ABENCH / "phase1_labels.csv", index=False)

    print("\n===== Phase 1 labels (bootstrap p < {:.2f}, min_trades={}) =====".format(
        args.p_threshold, args.min_trades))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print(df.to_string(index=False))

    print("\n===== Label counts =====")
    print(df["label"].value_counts().to_string())

    print(f"\n[saved] {ABENCH}/phase1_labels.csv")


if __name__ == "__main__":
    main()
