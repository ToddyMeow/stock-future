"""Parameter sensitivity sweep for Step 5 of the backtest plan.

Holds best_combos fixed from a baseline run and varies one engine knob at a
time. For each grid point reruns only Layer 3 (portfolio layer) and collects
headline metrics into data/sensitivity_<param>.csv.

Special --risk-preset mode bundles (risk_per_trade, group_cap, portfolio_cap)
so a three-tier preset (e.g. A档 2/5/15 vs B档 3/8/20) can be compared as one
atomic change.

Usage:
    python scripts/run_sensitivity_sweep.py                   # all 1-D sweeps
    python scripts/run_sensitivity_sweep.py --params risk_per_trade
    python scripts/run_sensitivity_sweep.py --risk-preset     # A vs B tier comparison
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.research.report_baseline import derive_best_combos
from strats.engine import StrategyEngine, StrategySlot
from strats.factory import build_engine_config, build_entries, build_exits
from strats.helpers import PortfolioAnalyzer
from strats.research_support import GROUPS, load_hab_bars


SWEEPS: dict[str, list] = {
    "risk_per_trade":       [0.005, 0.01, 0.02, 0.03, 0.04],
    "portfolio_risk_cap":   [0.08, 0.12, 0.16, 0.20],
    "stop_atr_mult":        [1.5, 2.0, 2.5, 3.0],
    "adx_floor":            [0.0, 0.2, 0.3, 1.0],  # 1.0 = ADX off via make_engine_config(adx_off=True)
    "enable_dual_stream":   [False, True],
}

# Three-tier risk presets: (risk_per_trade, group_risk_cap_uniform, portfolio_risk_cap).
# A_default keeps the heterogeneous group dict from config.yaml (4%-6% by group).
# Others override all groups to a uniform cap.
RISK_PRESETS: dict[str, dict] = {
    "A_default_2_dict_15":  {"risk_per_trade": 0.02, "portfolio_risk_cap": 0.15},
    "A_uniform_2_5_15":     {"risk_per_trade": 0.02, "portfolio_risk_cap": 0.15, "group_uniform": 0.05},
    "B_moderate_3_6_20":    {"risk_per_trade": 0.03, "portfolio_risk_cap": 0.20, "group_uniform": 0.06},
    "B_aggressive_3_8_20":  {"risk_per_trade": 0.03, "portfolio_risk_cap": 0.20, "group_uniform": 0.08},
}


def run_point(bars, engine_cfg, entries, exits, best_combos) -> dict:
    slots = [
        StrategySlot(
            strategy_id=f"{g}_{c}",
            entry_strategy=entries[c.split("+")[0]],
            exit_strategy=exits[c.split("+")[1]],
        )
        for g, c in best_combos.items()
    ]
    included = set(best_combos.keys())
    portfolio_bars = bars[bars["group_name"].isin(included)].copy()
    engine = StrategyEngine(config=engine_cfg, strategies=slots)
    result = engine.run(portfolio_bars)
    analyzer = PortfolioAnalyzer(result, engine_cfg)
    return analyzer.summary_stats()


def _apply_preset(base_cfg, preset: dict):
    """Apply a RISK_PRESETS entry onto a base EngineConfig."""
    overrides = {k: v for k, v in preset.items() if k != "group_uniform"}
    cfg = replace(base_cfg, **overrides)
    if "group_uniform" in preset:
        uniform = preset["group_uniform"]
        cfg = replace(cfg, group_risk_cap={g: uniform for g in GROUPS})
    return cfg


def sweep_risk_presets(bars, entries, exits, best_combos) -> pd.DataFrame:
    rows: list[dict] = []
    for name, preset in RISK_PRESETS.items():
        t0 = time.time()
        engine_cfg = _apply_preset(build_engine_config(profile="research"), preset)
        label = f"{name}: risk={preset['risk_per_trade']*100:.0f}%  " \
                f"group={preset.get('group_uniform', 'dict')}  " \
                f"port={preset['portfolio_risk_cap']*100:.0f}%"
        try:
            stats = run_point(bars, engine_cfg, entries, exits, best_combos)
            row = {
                "preset": name,
                "risk_per_trade": preset["risk_per_trade"],
                "group_cap": preset.get("group_uniform", "dict(4-6%)"),
                "portfolio_cap": preset["portfolio_risk_cap"],
                "sharpe": stats.get("sharpe", 0.0),
                "cagr": stats.get("cagr", 0.0),
                "max_dd_pct": stats.get("max_drawdown_pct", 0.0),
                "total_return": stats.get("total_return", 0.0),
                "profit_factor": stats.get("profit_factor", 0.0),
                "total_trades": stats.get("total_trades", 0),
                "win_rate": stats.get("win_rate", 0.0),
            }
            print(
                f"  {label:60s}  sharpe={row['sharpe']:+.3f}  "
                f"cagr={row['cagr']*100:+.2f}%  maxDD={row['max_dd_pct']*100:+.2f}%  "
                f"trades={row['total_trades']}  ({time.time()-t0:.0f}s)"
            )
        except Exception as e:
            print(f"  {label}: ERROR — {e}")
            row = {"preset": name, "sharpe": float("nan"), "cagr": float("nan"),
                   "max_dd_pct": float("nan")}
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="baseline")
    parser.add_argument("--params", nargs="+", default=list(SWEEPS.keys()),
                        help="Which params to sweep (space-separated).")
    parser.add_argument("--risk-preset", action="store_true",
                        help="Compare A vs B risk-tier presets (skip 1-D sweeps).")
    args = parser.parse_args()

    run_name = args.suffix or "default"
    group_csv = ROOT / "data" / "runs" / run_name / "backtest_group_layer.csv"
    if not group_csv.exists():
        print(f"ERROR: {group_csv} not found — run baseline first", file=sys.stderr)
        sys.exit(1)

    sens_dir = ROOT / "data" / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)

    best_combos = derive_best_combos(group_csv)
    print(f"Best combos: {json.dumps(best_combos)}")

    bars = load_hab_bars()
    entries = build_entries()
    exits = build_exits()

    if args.risk_preset:
        print(f"\n=== Risk-preset comparison (A vs B tiers) ===")
        df = sweep_risk_presets(bars, entries, exits, best_combos)
        out = sens_dir / "sensitivity_risk_preset.csv"
        df.to_csv(out, index=False)
        print(f"  -> Saved: {out}")
        return

    for param in args.params:
        if param not in SWEEPS:
            print(f"Skipping unknown param: {param}")
            continue
        values = SWEEPS[param]
        print(f"\n=== Sweeping {param} over {values} ===")
        rows: list[dict] = []
        for val in values:
            t0 = time.time()
            if param == "adx_floor" and val == 1.0:
                engine_cfg = build_engine_config(profile="research", adx_off=True)
                label = f"{param}=1.0 (ADX off)"
            else:
                engine_cfg = replace(build_engine_config(profile="research"), **{param: val})
                label = f"{param}={val}"
            try:
                stats = run_point(bars, engine_cfg, entries, exits, best_combos)
                row = {
                    "param": param, "value": val,
                    "sharpe": stats.get("sharpe", 0.0),
                    "cagr": stats.get("cagr", 0.0),
                    "max_dd_pct": stats.get("max_drawdown_pct", 0.0),
                    "total_return": stats.get("total_return", 0.0),
                    "profit_factor": stats.get("profit_factor", 0.0),
                    "total_trades": stats.get("total_trades", 0),
                    "win_rate": stats.get("win_rate", 0.0),
                }
                print(
                    f"  {label:40s}  sharpe={row['sharpe']:+.3f}  "
                    f"cagr={row['cagr']*100:+.2f}%  maxDD={row['max_dd_pct']*100:+.2f}%  "
                    f"trades={row['total_trades']}  ({time.time()-t0:.0f}s)"
                )
            except Exception as e:
                print(f"  {label}: ERROR — {e}")
                row = {"param": param, "value": val, "sharpe": float("nan"),
                       "cagr": float("nan"), "max_dd_pct": float("nan"),
                       "total_return": float("nan"), "profit_factor": float("nan"),
                       "total_trades": 0, "win_rate": float("nan")}
            rows.append(row)

        out = sens_dir / f"sensitivity_{param}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  -> Saved: {out}")


if __name__ == "__main__":
    main()
