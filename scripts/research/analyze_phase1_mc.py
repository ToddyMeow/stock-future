"""Phase 1 Monte Carlo analyzer — show distribution across 20 seeds.

Extends analyze_phase1.py with per-seed expectancy distribution:
  - For each (group, experiment), compute per-seed expectancy
  - Show mean, std, min, max, 5-95% range across seeds
  - Label stability: fraction of seeds that would yield the same label

Also reruns the pooled bootstrap permutation test for reference.

Reads per-trade dumps (trades_{A,B,C,D}.csv at group scope).

Writes:
  data/runs/alpha_benchmark/phase1_mc_distribution.csv
  data/runs/alpha_benchmark/phase1_labels.csv (overwrite with MC stability)
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


def load(name: str) -> pd.DataFrame:
    p = ABENCH / f"trades_{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def per_seed_expectancy(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Returns one row per (group_cols, seed) with expectancy."""
    if "seed" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby(group_cols + ["seed"], as_index=False).agg(
        trades=("net_pnl", "size"),
        net_pnl=("net_pnl", "sum"),
    )
    agg["expectancy"] = np.where(agg["trades"] > 0, agg["net_pnl"] / agg["trades"], 0.0)
    return agg


def permutation_test(trades_b: np.ndarray, trades_a: np.ndarray,
                     n_boot: int, rng: np.random.Generator) -> tuple:
    obs = float(trades_b.mean() - trades_a.mean())
    combined = np.concatenate([trades_b, trades_a])
    n_b = len(trades_b)
    c = 0
    for _ in range(n_boot):
        rng.shuffle(combined)
        if combined[:n_b].mean() - combined[n_b:].mean() >= obs:
            c += 1
    return obs, c / n_boot


def main() -> None:
    tA = load("A")
    tB = load("B")
    tC = load("C")
    tD = load("D")
    print(f"Loaded:  A={len(tA)} trades ({tA['seed'].nunique() if not tA.empty else 0} seeds)  "
          f"B={len(tB)}  C={len(tC)}  D={len(tD)}")

    if tA.empty or tB.empty or tC.empty:
        print("ERROR: missing experiment data")
        sys.exit(1)

    groups = sorted(tA["group"].unique())
    rng = np.random.default_rng(42)

    # Per-seed expectancy
    print("\n=== Per-seed expectancy distribution ===")
    A_seed = per_seed_expectancy(tA, ["group"])
    B_seed = per_seed_expectancy(tB, ["group", "exit"])  # per (group, exit, seed)
    C_seed = per_seed_expectancy(tC, ["group", "entry"])

    dist_rows = []
    label_rows = []

    for g in groups:
        a_trades_all = tA[tA["group"] == g]["net_pnl"].to_numpy(dtype=float)
        if len(a_trades_all) < MIN_TRADES:
            continue

        # A: per-seed expectancy
        a_ps = A_seed[A_seed["group"] == g]["expectancy"].to_numpy(dtype=float)
        a_stats = {
            "group": g, "experiment": "A",
            "n_seeds": len(a_ps),
            "mean": round(a_ps.mean(), 1),
            "std": round(a_ps.std(ddof=1), 1) if len(a_ps) > 1 else 0,
            "p05": round(np.percentile(a_ps, 5), 1),
            "p50": round(np.percentile(a_ps, 50), 1),
            "p95": round(np.percentile(a_ps, 95), 1),
            "min": round(a_ps.min(), 1),
            "max": round(a_ps.max(), 1),
        }
        dist_rows.append(a_stats)

        # B: pool across seeds to pick best exit, then per-seed dist for that exit
        b_pool = tB[tB["group"] == g].groupby("exit")["net_pnl"].apply(lambda x: x.to_numpy()).to_dict()
        best_B_key, best_B_mean = None, -np.inf
        for exit_id, arr in b_pool.items():
            if len(arr) < MIN_TRADES:
                continue
            if arr.mean() > best_B_mean:
                best_B_mean = arr.mean()
                best_B_key = exit_id
        best_B_arr = b_pool.get(best_B_key, np.array([]))
        if len(best_B_arr) >= MIN_TRADES:
            b_ps = B_seed[(B_seed["group"] == g) & (B_seed["exit"] == best_B_key)]["expectancy"].to_numpy()
            dist_rows.append({
                "group": g, "experiment": f"B_{best_B_key}",
                "n_seeds": len(b_ps),
                "mean": round(b_ps.mean(), 1),
                "std": round(b_ps.std(ddof=1), 1) if len(b_ps) > 1 else 0,
                "p05": round(np.percentile(b_ps, 5), 1),
                "p50": round(np.percentile(b_ps, 50), 1),
                "p95": round(np.percentile(b_ps, 95), 1),
                "min": round(b_ps.min(), 1),
                "max": round(b_ps.max(), 1),
            })

        # C: same treatment
        c_pool = tC[tC["group"] == g].groupby("entry")["net_pnl"].apply(lambda x: x.to_numpy()).to_dict()
        best_C_key, best_C_mean = None, -np.inf
        for entry_id, arr in c_pool.items():
            if len(arr) < MIN_TRADES:
                continue
            if arr.mean() > best_C_mean:
                best_C_mean = arr.mean()
                best_C_key = entry_id
        best_C_arr = c_pool.get(best_C_key, np.array([]))
        if len(best_C_arr) >= MIN_TRADES:
            c_ps = C_seed[(C_seed["group"] == g) & (C_seed["entry"] == best_C_key)]["expectancy"].to_numpy()
            dist_rows.append({
                "group": g, "experiment": f"C_{best_C_key}",
                "n_seeds": len(c_ps),
                "mean": round(c_ps.mean(), 1),
                "std": round(c_ps.std(ddof=1), 1) if len(c_ps) > 1 else 0,
                "p05": round(np.percentile(c_ps, 5), 1),
                "p50": round(np.percentile(c_ps, 50), 1),
                "p95": round(np.percentile(c_ps, 95), 1),
                "min": round(c_ps.min(), 1),
                "max": round(c_ps.max(), 1),
            })

        # Pooled permutation tests
        b_diff = b_p = None
        if len(best_B_arr) >= MIN_TRADES:
            b_diff, b_p = permutation_test(best_B_arr, a_trades_all, N_BOOTSTRAP, rng)
        c_diff = c_p = None
        if len(best_C_arr) >= MIN_TRADES:
            c_diff, c_p = permutation_test(best_C_arr, a_trades_all, N_BOOTSTRAP, rng)

        # D test (single seed, deterministic — use D data as-is)
        d_diff = d_p = best_D_key = None
        if not tD.empty:
            d_pool = tD[tD["group"] == g].groupby(["entry", "exit"])["net_pnl"].apply(lambda x: x.to_numpy()).to_dict()
            best_D_mean = -np.inf
            for (ent, exi), arr in d_pool.items():
                if len(arr) < MIN_TRADES:
                    continue
                if arr.mean() > best_D_mean:
                    best_D_mean = arr.mean()
                    best_D_key = (ent, exi)
            if best_D_key:
                d_arr = d_pool[best_D_key]
                d_diff, d_p = permutation_test(d_arr, a_trades_all, N_BOOTSTRAP, rng)

        # Labels: same as before
        b_pass = (b_p is not None) and b_p < SIG_P and b_diff > 0
        c_pass = (c_p is not None) and c_p < SIG_P and c_diff > 0
        d_pass = (d_p is not None) and d_p < SIG_P and d_diff > 0

        # Seed-level stability: fraction of seeds where B exceeds A's same-seed expectancy
        # (descriptive only)
        stability = None
        if len(a_ps) == len(b_ps) and len(a_ps) > 0:
            # Match by seed (A and B seeds align)
            a_df = A_seed[A_seed["group"] == g].set_index("seed")["expectancy"]
            b_df = B_seed[(B_seed["group"] == g) & (B_seed["exit"] == best_B_key)].set_index("seed")["expectancy"]
            common = a_df.index.intersection(b_df.index)
            diffs = b_df.loc[common] - a_df.loc[common]
            stability = round((diffs > 0).mean(), 2) if len(diffs) > 0 else None

        if b_pass and c_pass:
            label = "both"
        elif c_pass and not b_pass:
            label = "entry_alpha"
        elif b_pass and not c_pass:
            label = "exit_alpha"
        elif d_pass:
            label = "synergy_only"
        else:
            label = "none"

        label_rows.append({
            "group": g,
            "A_trades": len(a_trades_all),
            "A_mean": round(a_trades_all.mean(), 1),
            "best_B": best_B_key,
            "B_trades": len(best_B_arr),
            "B_mean": round(best_B_arr.mean(), 1) if len(best_B_arr) > 0 else None,
            "B_p": round(b_p, 4) if b_p is not None else None,
            "B_seed_mean": round(b_ps.mean(), 1) if len(b_ps) > 0 else None,
            "B_seed_std": round(b_ps.std(ddof=1), 1) if len(b_ps) > 1 else 0,
            "B_beats_A_seeds": stability,
            "best_C": best_C_key,
            "C_trades": len(best_C_arr),
            "C_mean": round(best_C_arr.mean(), 1) if len(best_C_arr) > 0 else None,
            "C_p": round(c_p, 4) if c_p is not None else None,
            "best_D": f"{best_D_key[0]}+{best_D_key[1]}" if best_D_key else None,
            "D_p": round(d_p, 4) if d_p is not None else None,
            "label": label,
        })

    dist_df = pd.DataFrame(dist_rows)
    label_df = pd.DataFrame(label_rows).sort_values(["label", "group"]).reset_index(drop=True)

    dist_df.to_csv(ABENCH / "phase1_mc_distribution.csv", index=False)
    label_df.to_csv(ABENCH / "phase1_labels.csv", index=False)

    print("\n===== Phase 1 MC — Label distribution =====")
    print(label_df["label"].value_counts().to_string())

    print("\n===== Phase 1 MC — Full table =====")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(label_df.to_string(index=False))

    print("\n===== Per-seed distribution (mean ± std, range) =====")
    dist_view = dist_df.copy()
    dist_view["range"] = dist_view.apply(lambda r: f"[{r['p05']}, {r['p95']}]", axis=1)
    dist_view["display"] = dist_view.apply(lambda r: f"{r['mean']} ± {r['std']}", axis=1)
    pvt = dist_view.pivot(index="group", columns="experiment", values="display")
    print(pvt.to_string())

    print(f"\n[saved] {ABENCH}/phase1_labels.csv, phase1_mc_distribution.csv")


if __name__ == "__main__":
    main()
