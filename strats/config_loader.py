"""Load strategy configuration from config.yaml and build StrategyEngine.

Usage:
    from strats.config_loader import load_config, build_engine

    cfg = load_config()                 # reads config.yaml from repo root
    engine = build_engine(cfg)          # returns configured StrategyEngine
    result = engine.run(bars)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from strats.engine import EngineConfig, StrategyEngine
from strats.entries.donchian_entry import DonchianEntryConfig, DonchianEntryStrategy
from strats.entries.hab_entry import HABEntryConfig, HABEntryStrategy
from strats.exits.hab_exit import HABExitConfig, HABExitStrategy


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config.yaml and return as a plain dict."""
    if path is None:
        # Walk up from this file to find repo root
        root = Path(__file__).resolve().parent.parent
        path = str(root / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_engine_config(cfg: Dict[str, Any]) -> EngineConfig:
    """Build EngineConfig from the 'engine' section."""
    e = cfg.get("engine", {})
    return EngineConfig(
        initial_capital=float(e.get("initial_capital", 1_000_000)),
        atr_period=int(e.get("atr_period", 20)),
        risk_per_trade=float(e.get("risk_per_trade", 0.02)),
        portfolio_risk_cap=float(e.get("portfolio_risk_cap", 0.12)),
        group_risk_cap=float(e.get("group_risk_cap", 0.06)),
        max_portfolio_leverage=float(e.get("max_portfolio_leverage", 3.0)),
        default_margin_rate=float(e.get("default_margin_rate", 0.10)),
        risk_blowout_cap=float(e.get("risk_blowout_cap", 1.5)),
        risk_blowout_action=e.get("risk_blowout_action", "SHRINK"),
        allow_short=bool(e.get("allow_short", False)),
    )


def _build_entry_strategy(cfg: Dict[str, Any], allow_short: bool) -> Any:
    """Build the selected entry strategy from the 'entry' section."""
    entry = cfg.get("entry", {})
    strategy_name = entry.get("strategy", "hab").lower()

    if strategy_name == "hab":
        p = entry.get("hab", {})
        return HABEntryStrategy(HABEntryConfig(
            bb_period=int(p.get("bb_period", 20)),
            bb_std=float(p.get("bb_std", 2.0)),
            bb_percentile_lookback=int(p.get("bb_percentile_lookback", 60)),
            bb_percentile_threshold=float(p.get("bb_percentile_threshold", 0.30)),
            box_lookback=int(p.get("box_lookback", 7)),
            box_width_atr_mult=float(p.get("box_width_atr_mult", 1.5)),
            tol_atr_mult=float(p.get("tol_atr_mult", 0.25)),
            breakout_atr_mult=float(p.get("breakout_atr_mult", 0.5)),
            upper_shadow_ratio_max=float(p.get("upper_shadow_ratio_max", 0.25)),
            initial_stop_atr_mult=float(p.get("initial_stop_atr_mult", 0.4)),
            allow_short=allow_short,
        ))

    elif strategy_name == "donchian":
        p = entry.get("donchian", {})
        return DonchianEntryStrategy(DonchianEntryConfig(
            donchian_period=int(p.get("donchian_period", 20)),
            initial_stop_atr_mult=float(p.get("initial_stop_atr_mult", 2.0)),
            allow_short=allow_short,
        ))

    else:
        raise ValueError(f"Unknown entry strategy: {strategy_name!r}. Use 'hab' or 'donchian'.")


def _build_exit_strategy(cfg: Dict[str, Any]) -> Any:
    """Build the selected exit strategy from the 'exit' section."""
    exit_sec = cfg.get("exit", {})
    strategy_name = exit_sec.get("strategy", "hab").lower()

    if strategy_name == "hab":
        p = exit_sec.get("hab", {})
        return HABExitStrategy(HABExitConfig(
            structure_fail_bars=int(p.get("structure_fail_bars", 15)),
            structure_fail_mode=p.get("structure_fail_mode", "CLOSE_BELOW_BOX"),
            structure_fail_atr_buffer=float(p.get("structure_fail_atr_buffer", 0.5)),
            structure_fail_consecutive=int(p.get("structure_fail_consecutive", 2)),
            time_fail_bars=int(p.get("time_fail_bars", 5)),
            time_fail_target_r=float(p.get("time_fail_target_r", 0.5)),
            trail_activate_r=float(p.get("trail_activate_r", 1.0)),
            trail_atr_mult=float(p.get("trail_atr_mult", 2.0)),
        ))

    else:
        raise ValueError(f"Unknown exit strategy: {strategy_name!r}. Use 'hab'.")


def build_engine(cfg: Optional[Dict[str, Any]] = None) -> StrategyEngine:
    """Build a fully configured StrategyEngine from config dict.

    If cfg is None, loads from config.yaml automatically.
    """
    if cfg is None:
        cfg = load_config()

    engine_cfg = build_engine_config(cfg)
    entry_strategy = _build_entry_strategy(cfg, allow_short=engine_cfg.allow_short)
    exit_strategy = _build_exit_strategy(cfg)

    return StrategyEngine(
        config=engine_cfg,
        entry_strategy=entry_strategy,
        exit_strategy=exit_strategy,
    )


def get_data_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract the data section from config."""
    if cfg is None:
        cfg = load_config()
    return cfg.get("data", {})
