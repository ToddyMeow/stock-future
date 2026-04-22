"""v4 invariants: SYMBOL_LOCKED gate + use_group_risk_cap toggle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from strats.engine import EngineConfig, StrategyEngine, StrategySlot


# ---------- Shared test doubles (mirror test_margin_cap pattern) ----------


@dataclass(frozen=True)
class _EntryCfg:
    date: pd.Timestamp


class _DummyEntry:
    def __init__(self, cfg: "_EntryCfg | None" = None) -> None:
        self.cfg = cfg

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.cfg is None:
            out["entry_trigger_pass"] = False
            out["entry_direction"] = 0
        else:
            out["entry_trigger_pass"] = out["date"] == self.cfg.date
            out["entry_direction"] = out["entry_trigger_pass"].astype(int)
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {}


class _DummyExit:
    def process_close_phase(self, position, row, next_trade_date) -> None:
        position.completed_bars += 1


def _bars(n: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range(start="2024-01-02", periods=n)
    return pd.DataFrame([
        {
            "date": d, "symbol": "X",
            "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0,
            "volume": 1.0, "open_interest": 100.0,
            "contract_multiplier": 10.0, "commission": 5.0, "slippage": 1.0,
            "group_name": "G",
            "margin_rate": 0.10,
        } for d in dates
    ])


# ---------- SYMBOL_LOCKED gate ----------


def _run_two_slots(lock: bool) -> list[str]:
    """Run two slots (A fires at date[4], B fires at date[5]) on same symbol.

    Returns sorted list of strategy_ids with an open position at run end.
    """
    bars = _bars(n=12)
    cfg = EngineConfig(
        initial_capital=1_000_000.0,
        atr_period=2, adx_period=2,
        risk_per_trade=0.02, stop_atr_mult=2.0,
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 1.0}, default_group_risk_cap=1.0,
        independent_group_soft_cap=1.0, risk_blowout_cap=float("inf"),
        max_portfolio_leverage=1000.0,
        symbol_position_lock=lock,
    )
    strategies = [
        StrategySlot("A", _DummyEntry(_EntryCfg(bars["date"].iloc[4])), _DummyExit()),
        StrategySlot("B", _DummyEntry(_EntryCfg(bars["date"].iloc[5])), _DummyExit()),
    ]
    r = StrategyEngine(config=cfg, strategies=strategies).run(bars)
    if r.open_positions.empty:
        return []
    return sorted(r.open_positions["strategy_id"].tolist())


def test_symbol_lock_blocks_second_strategy_same_symbol() -> None:
    """symbol_position_lock=True: slot A fires on T → opens; slot B fires on T+1
    on SAME symbol → blocked. Only A ends with an open position."""
    assert _run_two_slots(lock=True) == ["A"]


def test_symbol_lock_off_allows_parallel_strategies_same_symbol() -> None:
    """Default symbol_position_lock=False: both slots open parallel positions
    on the same symbol (legacy behaviour preserved)."""
    assert _run_two_slots(lock=False) == ["A", "B"]


# ---------- use_group_risk_cap toggle ----------


def test_group_cap_disabled_lets_otherwise_blocked_signal_through() -> None:
    """With use_group_risk_cap=False: a signal that WOULD be rejected by a tight
    group_cap (0.01%) is allowed through. Enforces correct gate bypass."""
    bars = _bars(n=8)
    sig_date = bars["date"].iloc[4]
    cfg = EngineConfig(
        initial_capital=1_000_000.0,
        atr_period=2, adx_period=2,
        risk_per_trade=0.05, stop_atr_mult=2.0,
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 0.0001},       # deliberately tight — would normally block
        default_group_risk_cap=0.0001,
        independent_group_soft_cap=0.0001,  # also bypassed via flag
        risk_blowout_cap=float("inf"),
        max_portfolio_leverage=1000.0,
        use_group_risk_cap=False,           # ← v4 bypass
    )
    engine = StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(_EntryCfg(sig_date)), _DummyExit())],
    )
    r = engine.run(bars)
    rej = r.daily_status.loc[r.daily_status["date"] == sig_date, "risk_reject_reason"].iloc[0]
    assert rej != "GROUP_RISK_CAP"
    assert rej != "INDEPENDENT_SOFT_CAP"


def test_group_cap_enabled_blocks_as_before() -> None:
    """Default use_group_risk_cap=True still triggers GROUP_RISK_CAP —
    backward compatibility guard."""
    bars = _bars(n=8)
    sig_date = bars["date"].iloc[4]
    cfg = EngineConfig(
        initial_capital=1_000_000.0,
        atr_period=2, adx_period=2,
        risk_per_trade=0.05, stop_atr_mult=2.0,
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 0.0001},
        default_group_risk_cap=0.0001,
        independent_group_soft_cap=0.0001,
        risk_blowout_cap=float("inf"),
        max_portfolio_leverage=1000.0,
        # use_group_risk_cap defaults True
    )
    engine = StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(_EntryCfg(sig_date)), _DummyExit())],
    )
    r = engine.run(bars)
    rej = r.daily_status.loc[r.daily_status["date"] == sig_date, "risk_reject_reason"].iloc[0]
    assert rej == "GROUP_RISK_CAP"
