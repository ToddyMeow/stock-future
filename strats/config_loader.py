"""Backward-compatible wrappers around the shared strategy factory."""

from __future__ import annotations

from typing import Any, Dict, Optional

from strats.engine import StrategyEngine
from strats.factory import (
    build_engine_config as _build_engine_config,
    build_engine_config_from_dict,
    build_from_yaml,
    build_strategy_slots_from_config,
    load_yaml_config,
)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml_config(path)


def build_engine_config(cfg: Dict[str, Any]):
    return build_engine_config_from_dict(cfg)


def build_engine(cfg: Optional[Dict[str, Any]] = None) -> StrategyEngine:
    if cfg is None:
        engine_cfg, strategies = build_from_yaml()
    else:
        engine_cfg = _build_engine_config(profile="yaml", config=cfg)
        strategies = build_strategy_slots_from_config(cfg)
    return StrategyEngine(config=engine_cfg, strategies=strategies)


def get_data_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_config()
    return cfg.get("data", {})
