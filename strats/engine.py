"""Strategy engine: composable entry/exit orchestration.

Usage:
    engine = StrategyEngine(
        config=EngineConfig(...),
        entry_strategy=HABEntryStrategy(HABEntryConfig(...)),
        exit_strategy=HABExitStrategy(HABExitConfig(...)),
    )
    result = engine.run(bars)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from strats.protocols import EntryStrategy, ExitStrategy


@dataclass(frozen=True)
class EngineConfig:
    """Orchestrator-level config: columns, capital, risk, shared params."""

    # Column names
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
    margin_rate_col: str = "margin_rate"

    # Capital and risk
    initial_capital: float = 1_000_000.0
    risk_per_trade: float = 0.02
    portfolio_risk_cap: float = 0.12
    group_risk_cap: float = 0.06
    max_portfolio_leverage: float = 3.0
    default_margin_rate: float = 0.10

    # Shared technical
    atr_period: int = 20

    # Fill mechanics
    risk_blowout_cap: float = 1.5
    risk_blowout_action: Literal["SHRINK", "CANCEL"] = "SHRINK"
    allow_short: bool = False

    eps: float = 1e-12


class StrategyEngine:
    """Composable backtest engine accepting pluggable entry/exit strategies.

    This is the forward-looking API. For the HAB-specific legacy API, use
    ``HorizontalAccumulationBreakoutV1`` which wraps this engine with
    HABEntryStrategy + HABExitStrategy.
    """

    def __init__(
        self,
        config: EngineConfig,
        entry_strategy: Any,
        exit_strategy: Any,
    ) -> None:
        # Import here to avoid circular imports
        from strats.horizontal_accumulation_breakout_v1 import (
            HABConfig,
            HorizontalAccumulationBreakoutV1,
        )

        # Build a HABConfig that maps EngineConfig fields.
        # Entry/exit specific fields use defaults since the strategies carry their own.
        hab_cfg = HABConfig(
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
        self._inner = HorizontalAccumulationBreakoutV1(hab_cfg)
        # Override the entry/exit strategies
        self._inner._entry_strategy = entry_strategy
        self._inner._exit_strategy = exit_strategy
        self.config = config

    def prepare_data(self, bars: Any) -> Any:
        return self._inner.prepare_data(bars)

    def run(self, bars: Any) -> Any:
        return self._inner.run(bars)
