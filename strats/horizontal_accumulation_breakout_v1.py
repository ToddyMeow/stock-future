"""HAB strategy facade: thin wrapper around the generic StrategyEngine.

All orchestration logic lives in ``strats.engine.StrategyEngine``.
This module preserves the original ``HorizontalAccumulationBreakoutV1`` API
by mapping ``HABConfig`` fields into ``EngineConfig``, ``HABEntryConfig``,
and ``HABExitConfig``, then delegating to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import pandas as pd

from strats.engine import (
    BacktestResult,
    EngineConfig,
    PendingEntry,
    Position,
    StrategyEngine,
)
from strats.entries.hab_entry import HABEntryConfig, HABEntryStrategy
from strats.exits.hab_exit import HABExitConfig, HABExitStrategy
from strats.helpers import (
    PortfolioAnalyzer,
    detect_hlh_pattern,
    detect_lhl_pattern,
)


@dataclass(frozen=True)
class HABConfig:
    """Configuration for HorizontalAccumulationBreakout_V1."""

    date_col: str = "date"
    symbol_col: str = "symbol"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    open_interest_col: str = "open_interest"
    multiplier_col: str = "contract_multiplier"
    commission_col: str = "commission"
    slippage_col: str = "slippage"
    group_col: str = "group_name"

    initial_capital: float = 1_000_000.0
    atr_period: int = 20
    bb_period: int = 20
    bb_std: float = 2.0
    bb_percentile_lookback: int = 60
    bb_percentile_threshold: float = 0.30

    box_lookback: int = 7
    box_width_atr_mult: float = 1.5
    tol_atr_mult: float = 0.25
    breakout_atr_mult: float = 0.5
    upper_shadow_ratio_max: float = 0.25
    initial_stop_atr_mult: float = 0.4

    risk_per_trade: float = 0.02
    portfolio_risk_cap: float = 0.12
    group_risk_cap: float = 0.06

    structure_fail_bars: int = 15
    time_fail_bars: int = 5
    time_fail_target_r: float = 0.5
    trail_activate_r: float = 1.0
    trail_atr_mult: float = 2.0

    risk_blowout_cap: float = 1.5
    risk_blowout_action: Literal["SHRINK", "CANCEL"] = "SHRINK"

    allow_short: bool = False

    structure_fail_mode: Literal["CLOSE_BELOW_BOX", "CLOSE_BELOW_BOX_MINUS_ATR", "CONSECUTIVE_CLOSE"] = "CLOSE_BELOW_BOX"
    structure_fail_atr_buffer: float = 0.5
    structure_fail_consecutive: int = 2

    margin_rate_col: str = "margin_rate"
    default_margin_rate: float = 0.10
    max_portfolio_leverage: float = 3.0
    eps: float = 1e-12


class HorizontalAccumulationBreakoutV1:
    """Daily-bar long-only breakout strategy with portfolio/group risk control.

    Thin facade that maps HABConfig into the generic StrategyEngine with
    HABEntryStrategy + HABExitStrategy.
    """

    def __init__(self, config: Optional[HABConfig] = None) -> None:
        config = config or HABConfig()
        self.config = config

        engine_cfg = EngineConfig(
            date_col=config.date_col,
            symbol_col=config.symbol_col,
            open_col=config.open_col,
            high_col=config.high_col,
            low_col=config.low_col,
            close_col=config.close_col,
            volume_col=config.volume_col,
            open_interest_col=config.open_interest_col,
            multiplier_col=config.multiplier_col,
            commission_col=config.commission_col,
            slippage_col=config.slippage_col,
            group_col=config.group_col,
            margin_rate_col=config.margin_rate_col,
            initial_capital=config.initial_capital,
            risk_per_trade=config.risk_per_trade,
            portfolio_risk_cap=config.portfolio_risk_cap,
            group_risk_cap=config.group_risk_cap,
            max_portfolio_leverage=config.max_portfolio_leverage,
            default_margin_rate=config.default_margin_rate,
            atr_period=config.atr_period,
            risk_blowout_cap=config.risk_blowout_cap,
            risk_blowout_action=config.risk_blowout_action,
            allow_short=config.allow_short,
            eps=config.eps,
        )

        entry_cfg = HABEntryConfig(
            bb_period=config.bb_period,
            bb_std=config.bb_std,
            bb_percentile_lookback=config.bb_percentile_lookback,
            bb_percentile_threshold=config.bb_percentile_threshold,
            box_lookback=config.box_lookback,
            box_width_atr_mult=config.box_width_atr_mult,
            tol_atr_mult=config.tol_atr_mult,
            breakout_atr_mult=config.breakout_atr_mult,
            upper_shadow_ratio_max=config.upper_shadow_ratio_max,
            initial_stop_atr_mult=config.initial_stop_atr_mult,
            allow_short=config.allow_short,
            eps=config.eps,
        )

        exit_cfg = HABExitConfig(
            structure_fail_bars=config.structure_fail_bars,
            structure_fail_mode=config.structure_fail_mode,
            structure_fail_atr_buffer=config.structure_fail_atr_buffer,
            structure_fail_consecutive=config.structure_fail_consecutive,
            time_fail_bars=config.time_fail_bars,
            time_fail_target_r=config.time_fail_target_r,
            trail_activate_r=config.trail_activate_r,
            trail_atr_mult=config.trail_atr_mult,
        )

        self._engine = StrategyEngine(
            config=engine_cfg,
            entry_strategy=HABEntryStrategy(entry_cfg),
            exit_strategy=HABExitStrategy(exit_cfg),
        )

    def prepare_data(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self._engine.prepare_data(bars)

    def run(self, bars: pd.DataFrame) -> BacktestResult:
        return self._engine.run(bars)

    @property
    def _gap_diagnostics(self) -> Any:
        return self._engine._gap_diagnostics


__all__ = [
    "HABConfig",
    "BacktestResult",
    "HorizontalAccumulationBreakoutV1",
    "PortfolioAnalyzer",
    "detect_hlh_pattern",
    "detect_lhl_pattern",
]
