from __future__ import annotations

import pandas as pd

from live.config import INITIAL_CAPITAL
from live.engine_setup import (
    LIVE_GROUPS,
    build_engine_cfg_for_live,
    build_strategies_from_combos,
)


def test_build_engine_cfg_for_live_uses_live_risk_profile():
    cfg = build_engine_cfg_for_live()

    assert cfg.initial_capital == INITIAL_CAPITAL
    assert cfg.allow_short is True
    assert cfg.risk_per_trade == 0.03
    assert cfg.portfolio_risk_cap == 0.20
    assert cfg.default_group_risk_cap == 0.06
    assert {group: cfg.group_risk_cap[group] for group in LIVE_GROUPS} == {
        group: 0.06 for group in LIVE_GROUPS
    }


def test_build_strategies_from_combos_builds_live_slot_ids():
    combos = pd.DataFrame(
        [{"group": "building", "best_combo": "double_ma+boll"}]
    )

    slots = build_strategies_from_combos(combos)

    assert [slot.strategy_id for slot in slots] == ["building_double_ma+boll"]
