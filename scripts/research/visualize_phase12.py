"""Visualize Phase 1 + Phase 2 results.

Produces a set of PNGs in data/runs/alpha_benchmark/figures/:
  - phase1_label_heatmap.png        : 23 groups × 3 exit_prob, colored by label
  - phase1_excess_expectancy.png    : per-group excess expectancy (B-A, C-A, D-A)
  - phase1_seed_distribution.png    : per-group 20-seed D-experiment distribution
  - phase2_score_bars.png           : 52 symbols' tradeability score (sorted)
  - phase2_stability_heatmap.png    : 52 symbols × 3 exit_prob significance matrix
  - phase2_winrate_wlratio.png      : win_rate vs wl_ratio scatter, colored by score
  - phase2_score_hurst.png          : tradeability_score vs Hurst scatter
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ABENCH = ROOT / "data" / "runs" / "alpha_benchmark"
OUT = ABENCH / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

LABEL_COLORS = {
    "both": "#2ca02c",            # green
    "entry_alpha": "#1f77b4",     # blue
    "exit_alpha": "#ff7f0e",      # orange
    "synergy_only": "#9467bd",    # purple
    "drift_only": "#8c564b",      # brown
    "none": "#d3d3d3",            # grey
}


def load_phase1_labels() -> pd.DataFrame:
    frames = []
    for prob in ("005", "007", "010"):
        df = pd.read_csv(ABENCH / f"phase1_labels_ep{prob}.csv")
        df["exit_prob"] = f"0.{prob[-2:]}"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_phase2_stability() -> pd.DataFrame:
    frames = []
    for prob in ("005", "007", "010"):
        df = pd.read_csv(ABENCH / f"phase2_per_symbol_ep{prob}.csv")
        df["exit_prob"] = f"0.{prob[-2:]}"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fig_phase1_label_heatmap():
    df = load_phase1_labels()
    # pivot: rows = group, cols = exit_prob, values = label
    pvt = df.pivot(index="group", columns="exit_prob", values="label")

    fig, ax = plt.subplots(figsize=(7, 9))
    y = np.arange(len(pvt.index))
    x = np.arange(len(pvt.columns))
    for i, g in enumerate(pvt.index):
        for j, prob in enumerate(pvt.columns):
            label = pvt.iloc[i, j]
            if pd.isna(label):
                continue
            ax.add_patch(mpatches.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                              facecolor=LABEL_COLORS.get(label, "white"),
                                              edgecolor="black", linewidth=0.5))
            ax.text(j, i, label.replace("_alpha", "").replace("_", " "),
                    ha="center", va="center", fontsize=7,
                    color="white" if label in ("both", "entry_alpha", "exit_alpha", "synergy_only", "drift_only")
                    else "black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ep={c}" for c in pvt.columns])
    ax.set_yticks(y)
    ax.set_yticklabels(pvt.index)
    ax.set_xlim(-0.5, len(pvt.columns) - 0.5)
    ax.set_ylim(-0.5, len(pvt.index) - 0.5)
    ax.invert_yaxis()
    ax.set_title("Phase 1: group alpha label × exit_prob\n"
                 "(rand exit 10% fires less often = bottom row → random holds long)")
    # Legend
    handles = [mpatches.Patch(color=c, label=n) for n, c in LABEL_COLORS.items()]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = OUT / "phase1_label_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def fig_phase1_excess_expectancy():
    df = load_phase1_labels()
    df["B_minus_A"] = df["B_mean"].astype(float) - df["A_mean"].astype(float)
    df["C_minus_A"] = df["C_mean"].astype(float) - df["A_mean"].astype(float)

    # Compute D best-combo excess from trades_D.csv + phase1 A_mean
    trades_d = pd.read_csv(ABENCH / "trades_D.csv")
    d_best_mean = {}
    for g, sub in trades_d.groupby("group"):
        best_mean, best_arr = -np.inf, None
        for (ent, exi), g2 in sub.groupby(["entry", "exit"]):
            arr = g2["net_pnl"].to_numpy(dtype=float)
            if len(arr) < 30:
                continue
            if arr.mean() > best_mean:
                best_mean, best_arr = arr.mean(), arr
        if best_arr is not None:
            d_best_mean[g] = best_mean
    df["D_mean"] = df["group"].map(d_best_mean)
    df["D_minus_A"] = df["D_mean"] - df["A_mean"].astype(float)

    avg = df.groupby("group").agg(
        B_minus_A=("B_minus_A", "mean"),
        C_minus_A=("C_minus_A", "mean"),
        D_minus_A=("D_minus_A", "mean"),
    ).sort_values("D_minus_A", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(avg))
    width = 0.27
    ax.barh(y - width, avg["B_minus_A"], width, label="best B − A (exit alpha)", color="#ff7f0e")
    ax.barh(y, avg["C_minus_A"], width, label="best C − A (entry alpha)", color="#1f77b4")
    ax.barh(y + width, avg["D_minus_A"], width, label="best D − A (combined)", color="#2ca02c")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(avg.index)
    ax.set_xlabel("Excess per-trade expectancy (CNY), averaged over 3 exit_prob")
    ax.set_title("Phase 1: per-group excess expectancy vs rand×rand baseline")
    ax.legend(loc="lower right")
    ax.set_xscale("symlog", linthresh=100)
    plt.tight_layout()
    path = OUT / "phase1_excess_expectancy.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def fig_phase1_seed_distribution():
    """Box plot of per-seed B-exp expectancy across groups (ep=0.07)."""
    dist = pd.read_csv(ABENCH / "phase1_mc_dist_ep007.csv")
    # Keep only experiments starting with 'B_' (random × real exit)
    sub = dist[dist["experiment"].str.startswith("B_")].copy()
    # Sort by mean
    order = sub.groupby("group")["mean"].first().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(11, 7))
    # Box per group: we only have mean/std/p05/p50/p95 summary stats, not raw — synthesize
    data = []
    for g in order:
        row = sub[sub["group"] == g].iloc[0]
        # Approximate box from summary (p05, p50, p95 and ±std around mean)
        low, q1, med, q3, high = row["p05"], row["mean"] - row["std"], row["p50"], row["mean"] + row["std"], row["p95"]
        data.append([low, q1, med, q3, high])
    data = np.array(data, dtype=float)
    x = np.arange(len(order))
    for i, g in enumerate(order):
        low, q1, med, q3, high = data[i]
        ax.plot([i, i], [low, high], color="grey", linewidth=1.0)
        ax.add_patch(mpatches.Rectangle((i - 0.35, q1), 0.7, q3 - q1,
                                          facecolor="steelblue", edgecolor="black", linewidth=0.5))
        ax.plot([i - 0.35, i + 0.35], [med, med], color="black", linewidth=1.2)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("per-seed per-trade expectancy (CNY)  [best B exit @ ep=0.07]")
    ax.set_title("Phase 1: 20-seed distribution of best_B expectancy (synthetic box from μ±σ and p5/p95)")
    ax.set_yscale("symlog", linthresh=500)
    plt.tight_layout()
    path = OUT / "phase1_seed_distribution.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def fig_phase2_score_bars():
    # Use ep=0.07 (middle) + group_label from cross_prob
    df = pd.read_csv(ABENCH / "phase2_per_symbol_ep007.csv")
    df = df[df["tradeability_score"].notna()].copy()
    df = df.sort_values("tradeability_score", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 13))
    colors = [LABEL_COLORS.get(lbl, "#d3d3d3") for lbl in df["group_label"].fillna("none")]
    y = np.arange(len(df))
    ax.barh(y, df["tradeability_score"], color=colors, edgecolor="black", linewidth=0.3)
    for i, (sym, score) in enumerate(zip(df["symbol"], df["tradeability_score"])):
        ax.text(score + 1, i, f"{int(score)}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} ({g})" for s, g in zip(df["symbol"], df["group"])], fontsize=8)
    ax.set_xlabel("Tradeability Score (0–100) @ ep=0.07")
    ax.set_title("Phase 2: per-symbol tradeability score\n"
                 "color = group label from Phase 1 (ep=0.07)")
    handles = [mpatches.Patch(color=c, label=n) for n, c in LABEL_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    plt.tight_layout()
    path = OUT / "phase2_score_bars.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def fig_phase2_stability_heatmap():
    stab = pd.read_csv(ABENCH / "phase2_cross_prob.csv")
    # Extract pass status from each ep column (value like "90(p=0.0003)" or "-")
    def pass_val(s):
        return 0 if s == "-" or pd.isna(s) else 1
    cols = ["ep0.05", "ep0.07", "ep0.10"]
    arr = stab[cols].applymap(pass_val).to_numpy()
    syms = stab["symbol"].tolist()
    groups = stab["group"].tolist()
    # Sort by total passes desc, then by group
    total_pass = arr.sum(axis=1)
    order = np.argsort(-total_pass * 100 + np.arange(len(syms)) * 0.001)

    fig, ax = plt.subplots(figsize=(6.5, 13))
    y = np.arange(len(syms))
    for yi, idx in enumerate(order):
        for xj, col in enumerate(cols):
            passed = arr[idx, xj]
            ax.add_patch(mpatches.Rectangle((xj - 0.45, yi - 0.45), 0.9, 0.9,
                                              facecolor="#2ca02c" if passed else "#d3d3d3",
                                              edgecolor="black", linewidth=0.4))
            if passed:
                ax.text(xj, yi, "✓", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{syms[i]} ({groups[i]})" for i in order], fontsize=8)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(-0.5, len(syms) - 0.5)
    ax.invert_yaxis()
    ax.set_title("Phase 2 stability: symbol × exit_prob significance (bootstrap p<0.05)")
    plt.tight_layout()
    path = OUT / "phase2_stability_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def fig_phase2_winrate_wlratio():
    df = pd.read_csv(ABENCH / "phase2_per_symbol_ep007.csv")
    df = df[df["B_wr"].notna() & df["B_wl"].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = (df["tradeability_score"].fillna(0) + 20) * 2
    sc = ax.scatter(df["B_wr"], df["B_wl"],
                     c=df["tradeability_score"].fillna(0),
                     s=sizes, cmap="viridis", edgecolor="black", linewidth=0.3, alpha=0.85)
    for _, row in df.iterrows():
        if row["tradeability_score"] >= 70:
            ax.annotate(row["symbol"], (row["B_wr"], row["B_wl"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.7)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.7)
    ax.set_xlabel("best-B win_rate (random entry × real exit) @ ep=0.07")
    ax.set_ylabel("best-B avg_winner / |avg_loser| (wl_ratio)")
    ax.set_title("Phase 2: win_rate × wl_ratio per symbol\n"
                 "size & color = tradeability score")
    fig.colorbar(sc, ax=ax, label="tradeability score")
    plt.tight_layout()
    path = OUT / "phase2_winrate_wlratio.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def fig_phase2_score_hurst():
    df = pd.read_csv(ABENCH / "phase2_per_symbol_ep007.csv")
    df = df[df["hurst"].notna() & df["tradeability_score"].notna()].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    sig = df["exit_sig"] == True
    ax.scatter(df.loc[sig, "hurst"], df.loc[sig, "tradeability_score"],
                s=80, color="#2ca02c", edgecolor="black", label="significant (p<0.05)", alpha=0.85)
    ax.scatter(df.loc[~sig, "hurst"], df.loc[~sig, "tradeability_score"],
                s=60, color="#d3d3d3", edgecolor="black", label="not significant", alpha=0.6)
    for _, row in df[sig].iterrows():
        if row["tradeability_score"] >= 70:
            ax.annotate(row["symbol"], (row["hurst"], row["tradeability_score"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, label="H=0.5 (random walk)")
    ax.set_xlabel("Hurst exponent on close log-returns")
    ax.set_ylabel("Tradeability score (0–100)")
    ax.set_title("Phase 2: tradeability score vs Hurst — all tradeable syms are trending (H>0.5)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = OUT / "phase2_score_hurst.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def main():
    fig_phase1_label_heatmap()
    fig_phase1_excess_expectancy()
    fig_phase1_seed_distribution()
    fig_phase2_score_bars()
    fig_phase2_stability_heatmap()
    fig_phase2_winrate_wlratio()
    fig_phase2_score_hurst()
    print(f"\nAll figures in {OUT}")


if __name__ == "__main__":
    main()
