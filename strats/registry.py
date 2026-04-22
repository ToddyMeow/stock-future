"""Central registry for strategy constructors.

This module is the single source of truth for:
  - research combo presets (`hl_9`, `double_ma`, `atr_trail`, ...)
  - YAML strategy types (`hab`, `rand`, `donchian`, ...)
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from strats.entries.ama_entry import AmaEntryConfig, AmaEntryStrategy
from strats.entries.boll_break_entry import BollBreakEntryConfig, BollBreakEntryStrategy
from strats.entries.double_ma_entry import DoubleMaEntryConfig, DoubleMaEntryStrategy
from strats.entries.hab_entry import HABEntryConfig, HABEntryStrategy
from strats.entries.hl_entry import HLEntryConfig, HLEntryStrategy
from strats.entries.rand_entry import RandEntryConfig, RandEntryStrategy
from strats.entries.adaptive_hl_entry import AdaptiveHLEntryConfig, AdaptiveHLEntryStrategy
from strats.entries.adaptive_boll_entry import (
    AdaptiveBollEntryConfig,
    AdaptiveBollEntryStrategy,
)
from strats.exits.ama_exit import AmaExitConfig, AmaExitStrategy
from strats.exits.atr_trail_exit import AtrTrailExitConfig, AtrTrailExitStrategy
from strats.exits.boll_exit import BollExitConfig, BollExitStrategy
from strats.exits.double_ma_exit import DoubleMaExitConfig, DoubleMaExitStrategy
from strats.exits.hab_exit import HABExitConfig, HABExitStrategy
from strats.exits.hl_exit import HLExitConfig, HLExitStrategy
from strats.exits.rand_exit import RandExitConfig, RandExitStrategy
from strats.exits.term_exit import TermExitConfig, TermExitStrategy

EntryBuilder = Callable[..., Any]
ExitBuilder = Callable[..., Any]


def _build_hl_entry(*, period: int, allow_short: bool) -> Any:
    return HLEntryStrategy(HLEntryConfig(period=period, allow_short=allow_short))


ENTRY_PRESET_BUILDERS: Dict[str, EntryBuilder] = {
    "hl_9": lambda *, allow_short=True: _build_hl_entry(period=9, allow_short=allow_short),
    "hl_21": lambda *, allow_short=True: _build_hl_entry(period=21, allow_short=allow_short),
    "boll": lambda *, allow_short=True: BollBreakEntryStrategy(
        BollBreakEntryConfig(period=22, k=2.0, allow_short=allow_short)
    ),
    "ama": lambda *, allow_short=True: AmaEntryStrategy(
        AmaEntryConfig(n=10, fast_period=2, slow_period=30, allow_short=allow_short)
    ),
    "double_ma": lambda *, allow_short=True: DoubleMaEntryStrategy(
        DoubleMaEntryConfig(fast=13, slow=34, allow_short=allow_short)
    ),
    "adaptive_hl": lambda *, allow_short=True: AdaptiveHLEntryStrategy(
        AdaptiveHLEntryConfig(
            base_period=20,
            adapt_lookback=60,
            min_period_ratio=0.5,
            max_period_ratio=1.5,
            allow_short=allow_short,
        )
    ),
    "adaptive_boll": lambda *, allow_short=True: AdaptiveBollEntryStrategy(
        AdaptiveBollEntryConfig(
            period=22,
            base_k=2.0,
            adapt_lookback=60,
            min_k_ratio=0.5,
            max_k_ratio=1.5,
            allow_short=allow_short,
        )
    ),
}


EXIT_PRESET_BUILDERS: Dict[str, ExitBuilder] = {
    "hl": lambda: HLExitStrategy(HLExitConfig(period=21)),
    "boll": lambda: BollExitStrategy(BollExitConfig(period=22, k=2.0)),
    "ama": lambda: AmaExitStrategy(AmaExitConfig(n=10, fast_period=2, slow_period=30)),
    "atr_trail": lambda: AtrTrailExitStrategy(AtrTrailExitConfig(atr_mult=4.5)),
    "term": lambda: TermExitStrategy(TermExitConfig(min_bars=2, max_bars=13, min_target_r=1.0)),
    "double_ma": lambda: DoubleMaExitStrategy(DoubleMaExitConfig(fast=13, slow=34)),
}


def build_entry_preset(entry_id: str, *, allow_short: bool = True) -> Any:
    try:
        return ENTRY_PRESET_BUILDERS[entry_id](allow_short=allow_short)
    except KeyError as exc:
        raise ValueError(f"Unknown entry preset: {entry_id!r}") from exc


def build_exit_preset(exit_id: str) -> Any:
    try:
        return EXIT_PRESET_BUILDERS[exit_id]()
    except KeyError as exc:
        raise ValueError(f"Unknown exit preset: {exit_id!r}") from exc


def build_entry_from_config(entry_cfg: Dict[str, Any], *, allow_short: bool) -> Any:
    entry_type = str(entry_cfg.get("type", "hab")).lower()

    if entry_type == "hab":
        p = entry_cfg.get("hab", {})
        return HABEntryStrategy(
            HABEntryConfig(
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
            )
        )
    if entry_type in ("donchian", "hl"):
        p = entry_cfg.get(entry_type, {})
        return HLEntryStrategy(
            HLEntryConfig(
                period=int(p.get("period", p.get("donchian_period", 20))),
                allow_short=allow_short,
            )
        )
    if entry_type == "boll":
        p = entry_cfg.get("boll", {})
        return BollBreakEntryStrategy(
            BollBreakEntryConfig(
                period=int(p.get("period", 22)),
                k=float(p.get("k", 2.0)),
                allow_short=allow_short,
            )
        )
    if entry_type == "ama":
        p = entry_cfg.get("ama", {})
        return AmaEntryStrategy(
            AmaEntryConfig(
                n=int(p.get("n", 10)),
                fast_period=int(p.get("fast_period", 2)),
                slow_period=int(p.get("slow_period", 30)),
                allow_short=allow_short,
            )
        )
    if entry_type == "double_ma":
        p = entry_cfg.get("double_ma", {})
        return DoubleMaEntryStrategy(
            DoubleMaEntryConfig(
                fast=int(p.get("fast", 13)),
                slow=int(p.get("slow", 34)),
                allow_short=allow_short,
            )
        )
    if entry_type == "rand":
        p = entry_cfg.get("rand", {})
        return RandEntryStrategy(
            RandEntryConfig(
                seed=int(p.get("seed", 42)),
                allow_short=allow_short,
            )
        )
    if entry_type == "adaptive_hl":
        p = entry_cfg.get("adaptive_hl", {})
        return AdaptiveHLEntryStrategy(
            AdaptiveHLEntryConfig(
                base_period=int(p.get("base_period", 20)),
                adapt_lookback=int(p.get("adapt_lookback", 60)),
                min_period_ratio=float(p.get("min_period_ratio", 0.5)),
                max_period_ratio=float(p.get("max_period_ratio", 1.5)),
                allow_short=allow_short,
            )
        )
    if entry_type == "adaptive_boll":
        p = entry_cfg.get("adaptive_boll", {})
        return AdaptiveBollEntryStrategy(
            AdaptiveBollEntryConfig(
                period=int(p.get("period", 22)),
                base_k=float(p.get("base_k", 2.0)),
                adapt_lookback=int(p.get("adapt_lookback", 60)),
                min_k_ratio=float(p.get("min_k_ratio", 0.5)),
                max_k_ratio=float(p.get("max_k_ratio", 1.5)),
                allow_short=allow_short,
            )
        )

    raise ValueError(f"Unknown entry type: {entry_type!r}")


def build_exit_from_config(exit_cfg: Dict[str, Any]) -> Any:
    exit_type = str(exit_cfg.get("type", "hab")).lower()

    if exit_type == "hab":
        p = exit_cfg.get("hab", {})
        return HABExitStrategy(
            HABExitConfig(
                structure_fail_bars=int(p.get("structure_fail_bars", 15)),
                structure_fail_mode=p.get("structure_fail_mode", "CLOSE_BELOW_BOX"),
                structure_fail_atr_buffer=float(p.get("structure_fail_atr_buffer", 0.5)),
                structure_fail_consecutive=int(p.get("structure_fail_consecutive", 2)),
                time_fail_bars=int(p.get("time_fail_bars", 5)),
                time_fail_target_r=float(p.get("time_fail_target_r", 0.5)),
                trail_activate_r=float(p.get("trail_activate_r", 1.0)),
                trail_atr_mult=float(p.get("trail_atr_mult", 2.0)),
            )
        )
    if exit_type == "hl":
        p = exit_cfg.get("hl", {})
        return HLExitStrategy(HLExitConfig(period=int(p.get("period", 21))))
    if exit_type == "boll":
        p = exit_cfg.get("boll", {})
        return BollExitStrategy(
            BollExitConfig(
                period=int(p.get("period", 22)),
                k=float(p.get("k", 2.0)),
            )
        )
    if exit_type == "ama":
        p = exit_cfg.get("ama", {})
        return AmaExitStrategy(
            AmaExitConfig(
                n=int(p.get("n", 10)),
                fast_period=int(p.get("fast_period", 2)),
                slow_period=int(p.get("slow_period", 30)),
            )
        )
    if exit_type == "atr_trail":
        p = exit_cfg.get("atr_trail", {})
        return AtrTrailExitStrategy(AtrTrailExitConfig(atr_mult=float(p.get("atr_mult", 2.0))))
    if exit_type == "term":
        p = exit_cfg.get("term", {})
        return TermExitStrategy(
            TermExitConfig(
                min_bars=int(p.get("min_bars", 2)),
                max_bars=int(p.get("max_bars", 13)),
                min_target_r=float(p.get("min_target_r", 1.0)),
            )
        )
    if exit_type == "double_ma":
        p = exit_cfg.get("double_ma", {})
        return DoubleMaExitStrategy(
            DoubleMaExitConfig(
                fast=int(p.get("fast", 13)),
                slow=int(p.get("slow", 34)),
            )
        )
    if exit_type == "rand":
        p = exit_cfg.get("rand", {})
        return RandExitStrategy(
            RandExitConfig(
                seed=int(p.get("seed", 42)),
                exit_probability=float(p.get("exit_probability", 0.1)),
                min_bars=int(p.get("min_bars", 1)),
            )
        )

    raise ValueError(f"Unknown exit type: {exit_type!r}")
