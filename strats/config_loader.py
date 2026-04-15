"""Load strategy configuration from config.yaml and build StrategyEngine.

Usage:
    from strats.config_loader import load_config, build_engine

    cfg = load_config()                 # reads config.yaml from repo root
    engine = build_engine(cfg)          # returns configured StrategyEngine
    result = engine.run(bars)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from strats.engine import EngineConfig, StrategyEngine, StrategySlot
from strats.entries.ama_entry import AmaEntryConfig, AmaEntryStrategy
from strats.entries.boll_break_entry import BollBreakEntryConfig, BollBreakEntryStrategy
from strats.entries.donchian_entry import DonchianEntryConfig, DonchianEntryStrategy
from strats.entries.double_ma_entry import DoubleMaEntryConfig, DoubleMaEntryStrategy
from strats.entries.hab_entry import HABEntryConfig, HABEntryStrategy
from strats.entries.hl_entry import HLEntryConfig, HLEntryStrategy
from strats.entries.rand_entry import RandEntryConfig, RandEntryStrategy
from strats.exits.ama_exit import AmaExitConfig, AmaExitStrategy
from strats.exits.atr_trail_exit import AtrTrailExitConfig, AtrTrailExitStrategy
from strats.exits.boll_exit import BollExitConfig, BollExitStrategy
from strats.exits.double_ma_exit import DoubleMaExitConfig, DoubleMaExitStrategy
from strats.exits.hab_exit import HABExitConfig, HABExitStrategy
from strats.exits.hl_exit import HLExitConfig, HLExitStrategy
from strats.exits.rand_exit import RandExitConfig, RandExitStrategy
from strats.exits.term_exit import TermExitConfig, TermExitStrategy


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config.yaml and return as a plain dict."""
    if path is None:
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
        stop_atr_mult=float(e.get("stop_atr_mult", 2.0)),
    )


def _build_entry(entry_cfg: Dict[str, Any], allow_short: bool) -> Any:
    """Build an entry strategy from a config dict."""
    entry_type = entry_cfg.get("type", "hab").lower()

    if entry_type == "hab":
        p = entry_cfg.get("hab", {})
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
            allow_short=allow_short,
        ))
    elif entry_type in ("donchian", "hl"):
        # "donchian" kept for backward compat; both map to HLEntryStrategy
        p = entry_cfg.get(entry_type, {})
        return HLEntryStrategy(HLEntryConfig(
            period=int(p.get("period", p.get("donchian_period", 20))),
            allow_short=allow_short,
        ))
    elif entry_type == "boll":
        p = entry_cfg.get("boll", {})
        return BollBreakEntryStrategy(BollBreakEntryConfig(
            period=int(p.get("period", 22)),
            k=float(p.get("k", 2.0)),
            allow_short=allow_short,
        ))
    elif entry_type == "ama":
        p = entry_cfg.get("ama", {})
        return AmaEntryStrategy(AmaEntryConfig(
            n=int(p.get("n", 10)),
            fast_period=int(p.get("fast_period", 2)),
            slow_period=int(p.get("slow_period", 30)),
            allow_short=allow_short,
        ))
    elif entry_type == "double_ma":
        p = entry_cfg.get("double_ma", {})
        return DoubleMaEntryStrategy(DoubleMaEntryConfig(
            fast=int(p.get("fast", 13)),
            slow=int(p.get("slow", 34)),
            allow_short=allow_short,
        ))
    elif entry_type == "rand":
        p = entry_cfg.get("rand", {})
        return RandEntryStrategy(RandEntryConfig(
            seed=int(p.get("seed", 42)),
            allow_short=allow_short,
        ))
    else:
        raise ValueError(f"Unknown entry type: {entry_type!r}")


def _build_exit(exit_cfg: Dict[str, Any]) -> Any:
    """Build an exit strategy from a config dict."""
    exit_type = exit_cfg.get("type", "hab").lower()

    if exit_type == "hab":
        p = exit_cfg.get("hab", {})
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
    elif exit_type == "hl":
        p = exit_cfg.get("hl", {})
        return HLExitStrategy(HLExitConfig(
            period=int(p.get("period", 21)),
        ))
    elif exit_type == "boll":
        p = exit_cfg.get("boll", {})
        return BollExitStrategy(BollExitConfig(
            period=int(p.get("period", 22)),
            k=float(p.get("k", 2.0)),
        ))
    elif exit_type == "ama":
        p = exit_cfg.get("ama", {})
        return AmaExitStrategy(AmaExitConfig(
            n=int(p.get("n", 10)),
            fast_period=int(p.get("fast_period", 2)),
            slow_period=int(p.get("slow_period", 30)),
        ))
    elif exit_type == "atr_trail":
        p = exit_cfg.get("atr_trail", {})
        return AtrTrailExitStrategy(AtrTrailExitConfig(
            atr_mult=float(p.get("atr_mult", 2.0)),
        ))
    elif exit_type == "term":
        p = exit_cfg.get("term", {})
        return TermExitStrategy(TermExitConfig(
            min_bars=int(p.get("min_bars", 2)),
            max_bars=int(p.get("max_bars", 13)),
            min_target_r=float(p.get("min_target_r", 1.0)),
        ))
    elif exit_type == "double_ma":
        p = exit_cfg.get("double_ma", {})
        return DoubleMaExitStrategy(DoubleMaExitConfig(
            fast=int(p.get("fast", 13)),
            slow=int(p.get("slow", 34)),
        ))
    elif exit_type == "rand":
        p = exit_cfg.get("rand", {})
        return RandExitStrategy(RandExitConfig(
            seed=int(p.get("seed", 42)),
            exit_probability=float(p.get("exit_probability", 0.1)),
            min_bars=int(p.get("min_bars", 1)),
        ))
    else:
        raise ValueError(f"Unknown exit type: {exit_type!r}")


def _build_strategies(cfg: Dict[str, Any], allow_short: bool) -> List[StrategySlot]:
    """Build strategy slots from config."""
    strategies_cfg = cfg.get("strategies", [])

    if strategies_cfg:
        # New multi-strategy format
        slots = []
        for s in strategies_cfg:
            sid = s["id"]
            entry = _build_entry(s.get("entry", {}), allow_short)
            exit_ = _build_exit(s.get("exit", {}))
            slots.append(StrategySlot(sid, entry, exit_))
        return slots

    # Legacy single-strategy format (backward compat)
    entry_cfg = cfg.get("entry", {})
    exit_cfg = cfg.get("exit", {})
    entry = _build_entry(
        {"type": entry_cfg.get("strategy", "hab"), **{k: v for k, v in entry_cfg.items() if k != "strategy"}},
        allow_short,
    )
    exit_ = _build_exit(
        {"type": exit_cfg.get("strategy", "hab"), **{k: v for k, v in exit_cfg.items() if k != "strategy"}},
    )
    return [StrategySlot("default", entry, exit_)]


def build_engine(cfg: Optional[Dict[str, Any]] = None) -> StrategyEngine:
    """Build a fully configured StrategyEngine from config dict."""
    if cfg is None:
        cfg = load_config()

    engine_cfg = build_engine_config(cfg)
    strategies = _build_strategies(cfg, allow_short=engine_cfg.allow_short)

    return StrategyEngine(config=engine_cfg, strategies=strategies)


def get_data_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract the data section from config."""
    if cfg is None:
        cfg = load_config()
    return cfg.get("data", {})
