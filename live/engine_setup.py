"""Shared live-mode engine assembly.

Keeps reusable combo loading / slot building / engine profile wiring out of
`signal_service.py` so other live modules do not depend on a CLI service file.
"""

from __future__ import annotations

from typing import Any, List

import pandas as pd

from live.config import COMBOS_CSV_PATH, INITIAL_CAPITAL
from strats.engine import StrategySlot
from strats.factory import build_engine_config, build_strategy_slots_from_combos


LIVE_GROUPS = [
    "building",
    "livestock",
    "rubber_fiber",
    "ind_AP",
    "ind_BB",
    "ind_FB",
    "equity_index",
    "bond",
    "chem_energy",
    "metals",
    "black_steel",
    "agri",
]


def load_final_v3_combos() -> pd.DataFrame:
    """Load the active live combos from the configured CSV path."""
    df = pd.read_csv(COMBOS_CSV_PATH)
    return df[df["stability_status"].isin(["stable", "new_listing"])].copy()


def build_strategies_from_combos(combos: pd.DataFrame) -> List[StrategySlot]:
    """final_v3 combos → StrategySlot list."""
    return build_strategy_slots_from_combos(
        combos,
        include_adaptive=False,
        allow_short=True,
    )


def build_engine_cfg_for_live() -> Any:
    """Live engine parameters.

    Risk 3%, portfolio cap 20%, uniform group cap 6%, allow-short enabled.
    """
    from data.adapters.trading_calendar import TradingCalendar

    try:
        calendar = TradingCalendar.default()
    except Exception:  # noqa: BLE001
        calendar = None

    overrides = {
        "initial_capital": INITIAL_CAPITAL,
        "risk_per_trade": 0.03,
        "portfolio_risk_cap": 0.20,
        "group_risk_cap": {group: 0.06 for group in LIVE_GROUPS},
        "default_group_risk_cap": 0.06,
        "trading_calendar": calendar,
    }
    return build_engine_config(profile="research", overrides=overrides)
