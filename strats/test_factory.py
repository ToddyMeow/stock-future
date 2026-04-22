from __future__ import annotations

import pandas as pd

from strats.config_loader import build_engine, load_config
from strats.factory import (
    build_engine_config,
    build_from_yaml,
    build_strategy_slots_from_combos,
    build_strategy_slots_from_config,
)


def test_build_strategy_slots_from_combos_matches_strategy_config_shape():
    combos = pd.DataFrame(
        [{"group": "building", "best_combo": "hl_9+hl"}]
    )
    cfg = {
        "engine": {"allow_short": True},
        "strategies": [
            {
                "id": "building_hl_9+hl",
                "entry": {"type": "hl", "hl": {"period": 9}},
                "exit": {"type": "hl", "hl": {"period": 21}},
            }
        ],
    }

    combo_slots = build_strategy_slots_from_combos(combos, allow_short=True)
    config_slots = build_strategy_slots_from_config(cfg)

    assert [slot.strategy_id for slot in combo_slots] == [slot.strategy_id for slot in config_slots]
    assert type(combo_slots[0].entry_strategy) is type(config_slots[0].entry_strategy)
    assert type(combo_slots[0].exit_strategy) is type(config_slots[0].exit_strategy)
    assert combo_slots[0].entry_strategy.config == config_slots[0].entry_strategy.config
    assert combo_slots[0].exit_strategy.config == config_slots[0].exit_strategy.config


def test_build_from_yaml_matches_config_loader_engine():
    cfg = load_config()
    engine = build_engine(cfg)
    engine_cfg, slots = build_from_yaml()

    assert engine.config == engine_cfg
    assert [slot.strategy_id for slot in engine._strategies] == [slot.strategy_id for slot in slots]


def test_research_profile_forces_short_enabled_without_mutating_yaml_defaults():
    cfg = load_config()
    yaml_cfg = build_engine_config(profile="yaml", config=cfg)
    research_cfg = build_engine_config(profile="research", config=cfg)

    assert yaml_cfg.allow_short is False
    assert research_cfg.allow_short is True
