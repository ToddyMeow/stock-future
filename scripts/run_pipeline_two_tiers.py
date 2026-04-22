#!/usr/bin/env python
"""
Full Alpha Attribution Pipeline driver — two capital tiers (1M + 250k) in parallel.

Runs:
  Phase 0 (two tiers) → Phase 1 MC (shared) → Phase 2 MC (shared)
  → derive confirmed_syms per tier
  → Phase 3 / 4 / 5 (two tiers each)
  → WeChat notification

Concurrency: MAX_PAR subprocess workers. Each subprocess is single-threaded (OMP=1).
Target: 8 concurrent processes on a 16-core M4 Max = ~50% CPU budget.

Logs: /tmp/pipeline_<timestamp>/<phase>.log
Artifacts: data/runs/{phase0_1m,phase0_250k,alpha_benchmark,phase3,phase5}

Usage:
  python scripts/run_pipeline_two_tiers.py
  # Logs tail-able at the path printed on start.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path(f"/tmp/pipeline_{TS}")
LOG_DIR.mkdir(parents=True, exist_ok=True)
MAX_PAR = 8

SEEDS_LIST = [str(s) for s in range(42, 62)]  # 20 seeds
SEEDS_STR = ",".join(SEEDS_LIST)

EXIT_PROBS = [("005", "0.05"), ("007", "0.07"), ("010", "0.10")]

SINGLE_THREAD_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

TIMINGS: List[Dict] = []
PHASE_STATUS: Dict[str, str] = {}


def log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


def run_one(name: str, cmd: List[str]) -> Tuple[int, str, float]:
    t0 = time.time()
    with open(log_path(name), "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              env=SINGLE_THREAD_ENV)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else "FAIL"
    TIMINGS.append({"name": name, "rc": proc.returncode, "elapsed_s": elapsed})
    print(f"  [{status}] {name}  ({elapsed:.0f}s)  → {log_path(name)}",
          flush=True)
    return proc.returncode, name, elapsed


def run_batch(jobs: List[Tuple[str, List[str]]], phase: str) -> bool:
    """Submit up to MAX_PAR in parallel; return True if all succeeded."""
    print(f"\n=== {phase} ({len(jobs)} jobs, max_par={MAX_PAR}) ===", flush=True)
    t0 = time.time()
    any_fail = False
    with ThreadPoolExecutor(max_workers=MAX_PAR) as ex:
        futures = {ex.submit(run_one, name, cmd): name for name, cmd in jobs}
        for fut in as_completed(futures):
            rc, name, _ = fut.result()
            if rc != 0:
                any_fail = True
    elapsed = time.time() - t0
    PHASE_STATUS[phase] = "FAIL" if any_fail else "OK"
    print(f"--- {phase} finished in {elapsed:.0f}s "
          f"({'FAIL' if any_fail else 'OK'}) ---", flush=True)
    return not any_fail


def backup_alpha_benchmark() -> None:
    src = Path("data/runs/alpha_benchmark")
    if not src.exists():
        return
    dst = Path(f"data/runs/_archive/alpha_benchmark_preRiskB_{TS}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"Backed up existing alpha_benchmark/ → {dst}", flush=True)


def send_wechat_final(success: bool, summary_text: str) -> None:
    """Fire-and-forget WeChat notification via Server 酱."""
    try:
        # Use live package; it loads SERVERCHAN_SEND_KEY from live/.env.
        sys.path.insert(0, str(ROOT))
        import asyncio
        from live.alerting import send_wechat  # type: ignore

        title = "Pipeline 完成 ✅" if success else "Pipeline 失败 ❌"
        asyncio.run(send_wechat(title=title, desp=summary_text))
        print(f"WeChat sent: {title}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"WeChat send failed: {e}", flush=True)


def summarize_results() -> str:
    lines = [f"Pipeline run: {TS}", f"Logs: {LOG_DIR}", ""]
    lines.append("## Phase status")
    for phase, status in PHASE_STATUS.items():
        lines.append(f"- {phase}: **{status}**")
    lines.append("")
    lines.append("## Phase 5 metrics")
    for tier, cap in [("1m", "1000000"), ("250k", "250000")]:
        for suffix in ["oos", "fullperiod"]:
            tag = f"v4_{tier}_riskB_{suffix}"
            fp = Path(f"data/runs/phase5/summary_{tag}.json")
            if fp.exists():
                with open(fp) as f:
                    s = json.load(f)
                m = s.get("metrics", {})
                lines.append(
                    f"- **{tag}**: Sharpe {m.get('sharpe', 0):.2f}, "
                    f"CAGR {m.get('cagr', 0)*100:.1f}%, "
                    f"MaxDD {m.get('max_drawdown_pct', 0)*100:.1f}%, "
                    f"Final ¥{m.get('final_equity', 0):,.0f}"
                )
            else:
                lines.append(f"- **{tag}**: (no summary)")
    lines.append("")
    total = sum(t["elapsed_s"] for t in TIMINGS)
    lines.append(f"Total wall-clock: {total/60:.1f} min ({len(TIMINGS)} jobs)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
def phase_0() -> bool:
    jobs = [
        ("p0_1m", ["python", "scripts/run_phase0_tradeable_universe.py",
                   "--initial-capital", "1000000",
                   "--output-dir", "data/runs/phase0_1m"]),
        ("p0_250k", ["python", "scripts/run_phase0_tradeable_universe.py",
                     "--initial-capital", "250000",
                     "--output-dir", "data/runs/phase0_250k"]),
    ]
    return run_batch(jobs, "Phase 0")


def phase_1_mc() -> bool:
    jobs = []
    for ep_tag, ep_val in EXIT_PROBS:
        for exp in ["A", "B", "C"]:
            jobs.append((
                f"p1_{exp}_ep{ep_tag}",
                ["python", "scripts/run_random_benchmark.py",
                 "--experiment", exp,
                 "--seeds", SEEDS_STR,
                 "--exit-prob", ep_val,
                 "--loose-cap",
                 "--output-tag", f"_ep{ep_tag}"]
            ))
        # D is deterministic — 1 seed suffices
        jobs.append((
            f"p1_D_ep{ep_tag}",
            ["python", "scripts/run_random_benchmark.py",
             "--experiment", "D",
             "--seeds", "42",
             "--exit-prob", ep_val,
             "--loose-cap",
             "--output-tag", f"_ep{ep_tag}"]
        ))
    ok = run_batch(jobs, "Phase 1 MC (A/B/C 20 seeds + D 1 seed, × 3 exit_probs)")
    if ok:
        run_one("p1_analyze", ["python", "scripts/research/analyze_phase1_mc.py"])
    return ok


def phase_2_mc() -> bool:
    jobs = []
    for ep_tag, ep_val in EXIT_PROBS:
        for exp in ["A", "B"]:
            jobs.append((
                f"p2_{exp}_ep{ep_tag}",
                ["python", "scripts/run_random_benchmark.py",
                 "--experiment", exp,
                 "--seeds", SEEDS_STR,
                 "--exit-prob", ep_val,
                 "--loose-cap",
                 "--scope", "symbol",
                 "--output-tag", f"_ep{ep_tag}"]
            ))
    ok = run_batch(jobs, "Phase 2 MC (A + B per-symbol × 3 exit_probs)")
    if ok:
        run_one("p2_analyze", ["python", "scripts/research/analyze_phase2_mc.py"])
    return ok


def derive_confirmed_syms() -> bool:
    jobs = [
        ("derive_1m", ["python", "scripts/derive_confirmed_syms.py",
                       "--phase0-json", "data/runs/phase0_1m/tradeable_symbols.json",
                       "--phase2-csv", "data/runs/alpha_benchmark/phase2_per_symbol.csv",
                       "--output-json", "data/runs/phase3/confirmed_syms_1m.json",
                       "--min-score", "50"]),
        ("derive_250k", ["python", "scripts/derive_confirmed_syms.py",
                         "--phase0-json", "data/runs/phase0_250k/tradeable_symbols.json",
                         "--phase2-csv", "data/runs/alpha_benchmark/phase2_per_symbol.csv",
                         "--output-json", "data/runs/phase3/confirmed_syms_250k.json",
                         "--min-score", "50"]),
    ]
    return run_batch(jobs, "Derive confirmed_syms")


def phase_3() -> bool:
    jobs = []
    for tier, cap in [("1m", "1000000"), ("250k", "250000")]:
        jobs.append((
            f"p3_{tier}",
            ["python", "scripts/run_phase3_combo_selection.py",
             "--risk-per-trade", "0.03",
             "--group-cap", "0.08",
             "--portfolio-cap", "0.20",
             "--initial-capital", cap,
             "--confirmed-syms-json", f"data/runs/phase3/confirmed_syms_{tier}.json",
             "--output-tag", f"v4_{tier}_riskB"]
        ))
    ok = run_batch(jobs, "Phase 3 combo selection (1m + 250k)")
    if ok:
        jobs_an = [
            (f"p3_analyze_{tier}",
             ["python", "scripts/research/analyze_phase3.py", "--tag", f"v4_{tier}_riskB"])
            for tier in ["1m", "250k"]
        ]
        run_batch(jobs_an, "Phase 3 analyze (composite ranking)")
    return ok


def phase_4() -> bool:
    jobs = [
        (f"p4_{tier}",
         ["python", "scripts/research/analyze_phase4_rolling.py", "--tag", f"v4_{tier}_riskB"])
        for tier in ["1m", "250k"]
    ]
    return run_batch(jobs, "Phase 4 rolling window")


def phase_5() -> bool:
    jobs = []
    for tier, cap in [("1m", "1000000"), ("250k", "250000")]:
        universe = f"data/runs/phase0_{tier}/tradeable_symbols.json"
        stable_tag = f"v4_{tier}_riskB"
        # OOS 2024-2025
        jobs.append((
            f"p5_{tier}_oos",
            ["python", "scripts/run_phase5_oos.py",
             "--tag", stable_tag,
             "--output-tag", f"{stable_tag}_oos",
             "--group-cap", "0.08",
             "--portfolio-cap", "0.20",
             "--risk-per-trade", "0.03",
             "--initial-capital", cap,
             "--universe", universe]
        ))
        # Full period 2018-2025
        jobs.append((
            f"p5_{tier}_fullperiod",
            ["python", "scripts/run_phase5_oos.py",
             "--tag", stable_tag,
             "--output-tag", f"{stable_tag}_fullperiod",
             "--group-cap", "0.08",
             "--portfolio-cap", "0.20",
             "--risk-per-trade", "0.03",
             "--initial-capital", cap,
             "--universe", universe,
             "--oos-start", "2018-01-01",
             "--oos-end", "2025-12-31"]
        ))
    return run_batch(jobs, "Phase 5 OOS + fullperiod (4 runs)")


def plot_equity_curves() -> None:
    jobs = []
    for tier in ["1m", "250k"]:
        for suffix in ["oos", "fullperiod"]:
            tag = f"v4_{tier}_riskB_{suffix}"
            csv = f"data/runs/phase5/backtest_portfolio_layer_{tag}.csv"
            png = f"data/runs/phase5/equity_{tag}.png"
            jobs.append((
                f"plot_{tier}_{suffix}",
                ["python", "scripts/plot_equity_curve.py",
                 "--csv", csv, "--out", png]
            ))
    run_batch(jobs, "Plot equity curves (best-effort)")


def main() -> None:
    print(f"Pipeline driver started {TS}", flush=True)
    print(f"Log dir: {LOG_DIR}", flush=True)
    print(f"MAX_PAR: {MAX_PAR}", flush=True)

    overall_t0 = time.time()

    backup_alpha_benchmark()

    phases = [
        ("Phase 0", phase_0),
        ("Phase 1 MC", phase_1_mc),
        ("Phase 2 MC", phase_2_mc),
        ("Derive confirmed_syms", derive_confirmed_syms),
        ("Phase 3", phase_3),
        ("Phase 4", phase_4),
        ("Phase 5", phase_5),
    ]
    for phase_name, phase_fn in phases:
        if not phase_fn():
            print(f"\n!!! {phase_name} FAILED — aborting pipeline !!!",
                  flush=True)
            summary = summarize_results()
            print("\n" + summary, flush=True)
            send_wechat_final(success=False, summary_text=summary)
            sys.exit(1)

    plot_equity_curves()

    total = time.time() - overall_t0
    print(f"\nTotal pipeline: {total/60:.1f} min", flush=True)
    summary = summarize_results()
    print("\n" + summary, flush=True)

    (LOG_DIR / "summary.md").write_text(summary)
    send_wechat_final(success=True, summary_text=summary)


if __name__ == "__main__":
    main()
