"""Mixed portfolio selection — per-group pick the better of baseline or SAR.

Reads both tags' Phase4 rolling outputs and picks the per-group winner
based on `combo_exp_mean` (mean per-trade expectancy across 6 rolling
windows). A group's winner can be baseline (reverse_on_stop=False) or
SAR (reverse_on_stop=True); unstable variants are disqualified from being
selected even if absolute numbers are higher.

Output:
  data/runs/phase3/best_combos_stable_<mixed_tag>.csv
    Same schema as analyze_phase4_rolling output, plus a `reverse_on_stop`
    column (bool). Used by run_phase5_oos.py to configure per-slot SAR.

Usage:
  python scripts/research/analyze_phase3_mixed.py \\
      --baseline-tag risk3cap8_baseline --sar-tag risk3cap8_sar \\
      --mixed-tag risk3cap8_mixed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PHASE3_DIR = ROOT / "data" / "runs" / "phase3"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-tag", required=True)
    ap.add_argument("--sar-tag", required=True)
    ap.add_argument("--mixed-tag", required=True,
                    help="Output tag for mixed best_combos_stable file")
    args = ap.parse_args()

    b = pd.read_csv(PHASE3_DIR / f"best_combos_stable_{args.baseline_tag}.csv")
    s = pd.read_csv(PHASE3_DIR / f"best_combos_stable_{args.sar_tag}.csv")

    b["reverse_on_stop"] = False
    s["reverse_on_stop"] = True

    # Per group, consider only stable variants and pick the one with the
    # higher combo_exp_mean (Phase4's rolling-mean per-trade expectancy).
    rows = []
    all_groups = sorted(set(b["group"]).union(set(s["group"])))
    for g in all_groups:
        b_row = b[(b["group"] == g) & (b["stability_status"] == "stable")]
        s_row = s[(s["group"] == g) & (s["stability_status"] == "stable")]

        candidates = []
        if not b_row.empty:
            candidates.append(("baseline", b_row.iloc[0]))
        if not s_row.empty:
            candidates.append(("SAR", s_row.iloc[0]))

        if not candidates:
            # Neither stable — carry over the unstable record from whichever
            # tag reported it, for transparency. SAR first if both exist.
            fallback = s[s["group"] == g]
            if fallback.empty:
                fallback = b[b["group"] == g]
            if fallback.empty:
                continue
            row = fallback.iloc[0].to_dict()
            row["winner"] = "NONE_STABLE"
            rows.append(row)
            continue

        best_tag, best_row = max(candidates, key=lambda kv: kv[1]["combo_exp_mean"])
        row = best_row.to_dict()
        row["winner"] = best_tag
        rows.append(row)

    out = pd.DataFrame(rows)
    col_order = [
        "group", "best_combo", "reverse_on_stop", "winner",
        "IS_expectancy", "IS_excess",
        "windows_pass", "windows_total", "windows_absolute_profitable",
        "excess_mean", "excess_min", "excess_max",
        "combo_exp_mean", "combo_exp_min",
        "stability_status",
    ]
    out = out[[c for c in col_order if c in out.columns]]
    out = out.sort_values(["stability_status", "combo_exp_mean"],
                          ascending=[True, False]).reset_index(drop=True)

    out_path = PHASE3_DIR / f"best_combos_stable_{args.mixed_tag}.csv"
    out.to_csv(out_path, index=False)

    print("===== Mixed portfolio (per-group winner) =====")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(out.to_string(index=False))

    n_sar = int((out["reverse_on_stop"] == True).sum())  # noqa: E712
    n_base = int((out["reverse_on_stop"] == False).sum())  # noqa: E712
    total = int((out["stability_status"] == "stable").sum())
    sum_mean = float(out[out["stability_status"] == "stable"]["combo_exp_mean"].sum())
    print()
    print(f"Stable groups          : {total}")
    print(f"  SAR slots            : {n_sar}")
    print(f"  Baseline slots       : {n_base}")
    print(f"Sum combo_exp_mean     : {sum_mean:.0f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
