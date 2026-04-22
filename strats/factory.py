"""Shared factory for strategy assembly.

Avoids runtime dependencies from production/live paths onto research scripts.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from strats.engine import EngineConfig, StrategySlot
from strats.registry import (
    build_entry_from_config,
    build_entry_preset,
    build_exit_from_config,
    build_exit_preset,
)

DEFAULT_GROUP_RISK_CAPS = {
    "equity_index": 0.04,
    "bond": 0.04,
    "chem_energy": 0.06,
    "rubber_fiber": 0.06,
    "metals": 0.06,
    "black_steel": 0.06,
    "agri": 0.06,
    "building": 0.05,
    "livestock": 0.04,
}


def load_yaml_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        root = Path(__file__).resolve().parent.parent
        path = str(root / "config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_engine_config_from_dict(cfg: Dict[str, Any]) -> EngineConfig:
    e = cfg.get("engine", {})
    raw_group_risk_cap = e.get("group_risk_cap", {})
    if isinstance(raw_group_risk_cap, dict):
        group_risk_cap = {k: float(v) for k, v in raw_group_risk_cap.items()}
    else:
        cap = float(raw_group_risk_cap)
        group_risk_cap = {g: cap for g in DEFAULT_GROUP_RISK_CAPS}

    sar_kwargs: Dict[str, Any] = {}
    if "reverse_on_stop" in e:
        sar_kwargs["reverse_on_stop"] = bool(e["reverse_on_stop"])
    if "reverse_stop_atr_mult" in e:
        sar_kwargs["reverse_stop_atr_mult"] = float(e["reverse_stop_atr_mult"])
    if "reverse_chain_max" in e:
        sar_kwargs["reverse_chain_max"] = int(e["reverse_chain_max"])

    return EngineConfig(
        initial_capital=float(e.get("initial_capital", 1_000_000)),
        atr_period=int(e.get("atr_period", 20)),
        risk_per_trade=float(e.get("risk_per_trade", 0.02)),
        portfolio_risk_cap=float(e.get("portfolio_risk_cap", 0.12)),
        group_risk_cap=group_risk_cap,
        default_group_risk_cap=float(e.get("default_group_risk_cap", 0.02)),
        independent_group_soft_cap=float(e.get("independent_group_soft_cap", 0.08)),
        max_portfolio_leverage=float(e.get("max_portfolio_leverage", 3.0)),
        default_margin_rate=float(e.get("default_margin_rate", 0.10)),
        risk_blowout_cap=float(e.get("risk_blowout_cap", 1.5)),
        risk_blowout_action=e.get("risk_blowout_action", "SHRINK"),
        allow_short=bool(e.get("allow_short", False)),
        stop_atr_mult=float(e.get("stop_atr_mult", 2.0)),
        adx_period=int(e.get("adx_period", 20)),
        adx_scale=float(e.get("adx_scale", 30.0)),
        adx_floor=float(e.get("adx_floor", 0.2)),
        **sar_kwargs,
    )


def build_engine_config(
    profile: str = "research",
    overrides: Optional[Dict[str, Any]] = None,
    *,
    adx_off: bool = False,
    congestion_filter: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> EngineConfig:
    cfg = config if config is not None else load_yaml_config()
    base = build_engine_config_from_dict(cfg)
    merged: Dict[str, Any] = {}

    if profile == "research":
        merged["allow_short"] = True
    elif profile == "yaml":
        pass
    else:
        raise ValueError(f"Unknown engine profile: {profile!r}")

    if adx_off:
        merged["adx_floor"] = 1.0
    if congestion_filter:
        merged["use_congestion_filter"] = True
    if overrides:
        merged.update(overrides)

    return replace(base, **merged)


def build_entries(*, include_adaptive: bool = False, allow_short: bool = True) -> Dict[str, Any]:
    entries = {
        "hl_9": build_entry_preset("hl_9", allow_short=allow_short),
        "hl_21": build_entry_preset("hl_21", allow_short=allow_short),
        "boll": build_entry_preset("boll", allow_short=allow_short),
        "ama": build_entry_preset("ama", allow_short=allow_short),
        "double_ma": build_entry_preset("double_ma", allow_short=allow_short),
    }
    if include_adaptive:
        entries["adaptive_hl"] = build_entry_preset("adaptive_hl", allow_short=allow_short)
        entries["adaptive_boll"] = build_entry_preset("adaptive_boll", allow_short=allow_short)
    return entries


def build_exits() -> Dict[str, Any]:
    return {
        "hl": build_exit_preset("hl"),
        "boll": build_exit_preset("boll"),
        "ama": build_exit_preset("ama"),
        "atr_trail": build_exit_preset("atr_trail"),
        "term": build_exit_preset("term"),
        "double_ma": build_exit_preset("double_ma"),
    }


def build_strategy_slots_from_combos(
    combos: pd.DataFrame,
    *,
    include_adaptive: bool = False,
    allow_short: bool = True,
    strategy_id_col: str = "group",
    combo_col: str = "best_combo",
) -> List[StrategySlot]:
    entries = build_entries(include_adaptive=include_adaptive, allow_short=allow_short)
    exits = build_exits()
    slots: List[StrategySlot] = []
    for _, row in combos.iterrows():
        entry_id, exit_id = str(row[combo_col]).split("+")
        group_name = str(row[strategy_id_col])
        slot_kwargs: Dict[str, Any] = {}
        if "reverse_on_stop" in row.index and pd.notna(row["reverse_on_stop"]):
            slot_kwargs["reverse_on_stop"] = bool(row["reverse_on_stop"])
        if "reverse_stop_atr_mult" in row.index and pd.notna(row.get("reverse_stop_atr_mult")):
            slot_kwargs["reverse_stop_atr_mult"] = float(row["reverse_stop_atr_mult"])
        if "reverse_chain_max" in row.index and pd.notna(row.get("reverse_chain_max")):
            slot_kwargs["reverse_chain_max"] = int(row["reverse_chain_max"])
        slots.append(
            StrategySlot(
                strategy_id=f"{group_name}_{entry_id}+{exit_id}",
                entry_strategy=entries[entry_id],
                exit_strategy=exits[exit_id],
                **slot_kwargs,
            )
        )
    return slots


def build_strategy_slots_from_config(cfg: Dict[str, Any]) -> List[StrategySlot]:
    engine_cfg = build_engine_config_from_dict(cfg)
    allow_short = engine_cfg.allow_short
    strategies_cfg = cfg.get("strategies", [])

    if strategies_cfg:
        slots = []
        for strategy_cfg in strategies_cfg:
            slots.append(
                StrategySlot(
                    strategy_id=strategy_cfg["id"],
                    entry_strategy=build_entry_from_config(strategy_cfg.get("entry", {}), allow_short=allow_short),
                    exit_strategy=build_exit_from_config(strategy_cfg.get("exit", {})),
                )
            )
        return slots

    entry_cfg = cfg.get("entry", {})
    exit_cfg = cfg.get("exit", {})
    legacy_entry = {
        "type": entry_cfg.get("strategy", "hab"),
        **{k: v for k, v in entry_cfg.items() if k != "strategy"},
    }
    legacy_exit = {
        "type": exit_cfg.get("strategy", "hab"),
        **{k: v for k, v in exit_cfg.items() if k != "strategy"},
    }
    return [
        StrategySlot(
            strategy_id="default",
            entry_strategy=build_entry_from_config(legacy_entry, allow_short=allow_short),
            exit_strategy=build_exit_from_config(legacy_exit),
        )
    ]


def build_from_yaml(path: Optional[str] = None) -> tuple[EngineConfig, List[StrategySlot]]:
    cfg = load_yaml_config(path)
    return build_engine_config(profile="yaml", config=cfg), build_strategy_slots_from_config(cfg)
