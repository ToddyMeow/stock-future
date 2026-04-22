"""Margin utilization cap + delivery-month tier (4.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import pytest

from strats.engine import EngineConfig, StrategyEngine, StrategySlot


# ---------- Unit tests on helpers ----------


def test_months_to_delivery_parses_yymm_suffix() -> None:
    # RB2410 = deliver Oct 2024. Today = 2024-06-15 → 4 months out.
    assert StrategyEngine._months_to_delivery("RB2410", pd.Timestamp("2024-06-15")) == 4
    # Same contract in delivery month.
    assert StrategyEngine._months_to_delivery("RB2410", pd.Timestamp("2024-10-01")) == 0
    # 1-char symbol (I for iron ore).
    assert StrategyEngine._months_to_delivery("I2501", pd.Timestamp("2024-12-15")) == 1


def test_months_to_delivery_invalid_inputs() -> None:
    assert StrategyEngine._months_to_delivery(None, pd.Timestamp("2024-01-01")) is None
    assert StrategyEngine._months_to_delivery("RBABCD", pd.Timestamp("2024-01-01")) is None
    # Month 13 invalid.
    assert StrategyEngine._months_to_delivery("RB2413", pd.Timestamp("2024-01-01")) is None


def test_effective_margin_rate_no_schedule_returns_base_plus_broker() -> None:
    cfg = EngineConfig(broker_margin_addon=0.03)  # no tier
    eng = StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(), _DummyExit())],
    )
    assert eng._effective_margin_rate(0.10, "RB2410", pd.Timestamp("2024-01-01")) == pytest.approx(0.13)


def test_effective_margin_rate_applies_tier_by_exact_month() -> None:
    # Schedule keys are exact months-to-delivery.
    cfg = EngineConfig(
        broker_margin_addon=0.02,
        margin_tier_schedule={0: 0.10, 1: 0.05, 2: 0.02},
    )
    eng = StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(), _DummyExit())],
    )
    # RB2410: delivers Oct 2024.
    assert eng._effective_margin_rate(0.10, "RB2410", pd.Timestamp("2024-10-15")) == pytest.approx(0.22)  # 0 months
    assert eng._effective_margin_rate(0.10, "RB2410", pd.Timestamp("2024-09-01")) == pytest.approx(0.17)  # 1 month
    assert eng._effective_margin_rate(0.10, "RB2410", pd.Timestamp("2024-08-01")) == pytest.approx(0.14)  # 2 months
    assert eng._effective_margin_rate(0.10, "RB2410", pd.Timestamp("2024-06-01")) == pytest.approx(0.12)  # 4 months, no tier


# ---------- Integration test: cap rejects a signal ----------


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


def _bars_with_margin(n: int = 10) -> pd.DataFrame:
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


def test_margin_cap_rejects_signal_over_threshold() -> None:
    # Tight cap (1%) forces rejection; qty would demand more margin than cap allows.
    # 1M equity × 1% cap = 10k available margin.
    # At price 100, mult 10, margin 10% (no broker/tier): 1 contract = 100 margin.
    # Normal sizing (risk 10%, ATR ~4, stop_atr_mult 2 → stop 8 pts, per-contract risk 80, budget 100k → qty ~1250 contracts)
    # → candidate_margin ~= 1250 × 100 × 10 × 0.10 = 125k ≫ 10k cap.
    bars = _bars_with_margin(n=8)
    cfg = EngineConfig(
        initial_capital=1_000_000.0,
        atr_period=2, adx_period=2,
        risk_per_trade=0.1, stop_atr_mult=2.0,
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 1.0}, default_group_risk_cap=1.0,
        independent_group_soft_cap=1.0, risk_blowout_cap=float("inf"),
        max_portfolio_leverage=1000.0,   # bypass leverage cap
        max_margin_utilization=0.01,     # 1% — tight
    )
    sig_date = bars["date"].iloc[4]
    engine = StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(_EntryCfg(sig_date)), _DummyExit())],
    )
    r = engine.run(bars)
    rej = r.daily_status.loc[r.daily_status["date"] == sig_date, "risk_reject_reason"]
    assert rej.iloc[0] == "MARGIN_CAP"
    assert len(r.trades) == 0 and len(r.open_positions) == 0


def test_margin_cap_zero_disables_check() -> None:
    # max_margin_utilization=0 → cap disabled, signal not MARGIN_CAP-rejected.
    bars = _bars_with_margin(n=8)
    cfg = EngineConfig(
        initial_capital=1_000_000.0,
        atr_period=2, adx_period=2,
        risk_per_trade=0.1, stop_atr_mult=2.0,
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 1.0}, default_group_risk_cap=1.0,
        independent_group_soft_cap=1.0, risk_blowout_cap=float("inf"),
        max_portfolio_leverage=1000.0,
        max_margin_utilization=0.0,
    )
    sig_date = bars["date"].iloc[4]
    engine = StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(_EntryCfg(sig_date)), _DummyExit())],
    )
    r = engine.run(bars)
    rej = r.daily_status.loc[r.daily_status["date"] == sig_date, "risk_reject_reason"].iloc[0]
    assert rej != "MARGIN_CAP"
