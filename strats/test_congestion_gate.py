"""v5 invariants: CONGESTION_LOCKED gate + choppiness_index helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from strats.engine import EngineConfig, StrategyEngine, StrategySlot
from strats.helpers import choppiness_index


# ---------- choppiness_index unit tests ----------


def test_choppiness_index_low_on_strong_trend() -> None:
    """Pure monotonic up-trend → very low CI (<50)."""
    n = 60
    close = pd.Series(np.linspace(100.0, 150.0, n))
    high = close + 1.0
    low = close - 1.0
    ci = choppiness_index(high, low, close, period=14).dropna()
    assert not ci.empty
    assert ci.median() < 50.0, f"trend CI median {ci.median()} should be <50"


def test_choppiness_index_high_on_oscillating() -> None:
    """Sine-wave oscillation → moderate-to-high CI (>50)."""
    n = 60
    x = np.linspace(0, 10 * np.pi, n)
    close = pd.Series(100.0 + 5.0 * np.sin(x))
    high = close + 2.0
    low = close - 2.0
    ci = choppiness_index(high, low, close, period=14).dropna()
    assert not ci.empty
    assert ci.median() > 50.0, f"oscillation CI median {ci.median()} should be >50"


def test_choppiness_index_nan_warmup() -> None:
    """First `period - 1` values are NaN (warmup)."""
    n = 20
    close = pd.Series(range(100, 100 + n), dtype=float)
    ci = choppiness_index(close + 1, close - 1, close, period=14)
    assert ci.iloc[:13].isna().all()
    assert pd.notna(ci.iloc[13])


# ---------- CONGESTION_LOCKED gate integration ----------


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


def _bars_choppy(n: int = 80) -> pd.DataFrame:
    """Clamped random walk — flat envelope, no net direction.

    Guaranteed to produce a bar with ADX<25 (low directional movement)
    somewhere past warmup → CONGESTION_LOCKED on the default threshold.
    """
    dates = pd.bdate_range(start="2024-01-02", periods=n)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.choice([-2.0, 2.0], size=n))
    close = np.clip(close, 90.0, 110.0)
    return pd.DataFrame([
        {
            "date": dates[i], "symbol": "X",
            "open": close[i] - 0.2, "high": close[i] + 2.0,
            "low": close[i] - 2.0, "close": close[i],
            "volume": 1.0, "open_interest": 100.0,
            "contract_multiplier": 10.0, "commission": 5.0, "slippage": 1.0,
            "group_name": "G", "margin_rate": 0.10,
        } for i in range(n)
    ])


def _bars_trending(n: int = 40) -> pd.DataFrame:
    """Monotonic uptrend → low CPI, potentially higher ADX → tradeable."""
    dates = pd.bdate_range(start="2024-01-02", periods=n)
    close = np.linspace(100.0, 140.0, n)
    return pd.DataFrame([
        {
            "date": dates[i], "symbol": "X",
            "open": close[i] - 0.2, "high": close[i] + 0.5,
            "low": close[i] - 0.5, "close": close[i],
            "volume": 1.0, "open_interest": 100.0,
            "contract_multiplier": 10.0, "commission": 5.0, "slippage": 1.0,
            "group_name": "G", "margin_rate": 0.10,
        } for i in range(n)
    ])


def _permissive_cfg(**overrides) -> EngineConfig:
    base = dict(
        initial_capital=1_000_000.0,
        atr_period=5, adx_period=5, cpi_period=10,
        risk_per_trade=0.02, stop_atr_mult=2.0,
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 1.0}, default_group_risk_cap=1.0,
        independent_group_soft_cap=1.0, risk_blowout_cap=float("inf"),
        max_portfolio_leverage=1000.0,
        min_atr_pct=0.0,  # disable ATR floor for this test
    )
    base.update(overrides)
    return EngineConfig(**base)


def test_congestion_gate_blocks_signal_in_chop() -> None:
    """Sideways market + filter ON → signal rejected as CONGESTION_LOCKED."""
    bars = _bars_choppy(n=80)
    sig_date = bars["date"].iloc[45]  # well past warmup; chop env has low ADX here
    cfg = _permissive_cfg(use_congestion_filter=True)
    r = StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(_EntryCfg(sig_date)), _DummyExit())],
    ).run(bars)
    rej = r.daily_status.loc[r.daily_status["date"] == sig_date, "risk_reject_reason"].iloc[0]
    assert rej == "CONGESTION_LOCKED", f"expected CONGESTION_LOCKED, got {rej!r}"


def test_congestion_gate_off_lets_chop_signal_through() -> None:
    """Default (filter OFF) — same choppy bars pass. Backward-compat guard."""
    bars = _bars_choppy(n=80)
    sig_date = bars["date"].iloc[45]
    cfg = _permissive_cfg()  # use_congestion_filter defaults False
    r = StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(_EntryCfg(sig_date)), _DummyExit())],
    ).run(bars)
    rej = r.daily_status.loc[r.daily_status["date"] == sig_date, "risk_reject_reason"].iloc[0]
    assert rej != "CONGESTION_LOCKED"


def test_congestion_gate_requires_both_conditions() -> None:
    """Relaxed thresholds should let trending-ish bars through the filter."""
    bars = _bars_trending(n=40)
    sig_date = bars["date"].iloc[25]
    cfg = _permissive_cfg(
        use_congestion_filter=True,
        congestion_cpi_threshold=100.0,    # effectively impossible CPI
        congestion_adx_threshold=0.0,      # effectively impossible ADX
    )
    r = StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(_EntryCfg(sig_date)), _DummyExit())],
    ).run(bars)
    rej = r.daily_status.loc[r.daily_status["date"] == sig_date, "risk_reject_reason"].iloc[0]
    assert rej != "CONGESTION_LOCKED"
