"""v6 invariants: profit target partial/full close."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from strats.engine import EngineConfig, StrategyEngine, StrategySlot


@dataclass(frozen=True)
class _EntryCfg:
    date: pd.Timestamp


class _DummyEntry:
    """Fire on a specific date. Always long (direction=1)."""

    def __init__(self, cfg: _EntryCfg) -> None:
        self.cfg = cfg

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["entry_trigger_pass"] = out["date"] == self.cfg.date
        out["entry_direction"] = out["entry_trigger_pass"].astype(int)
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {}


class _PassiveExit:
    """Does nothing — lets profit target or stop drive exits."""

    def process_close_phase(self, position, row, next_trade_date) -> None:
        position.completed_bars += 1


def _bars_with_spike(pre: int, spike_to: float, post: int) -> pd.DataFrame:
    """Flat at 100 for `pre` bars, spike up to `spike_to` for `post` bars."""
    n = pre + post
    dates = pd.bdate_range(start="2024-01-02", periods=n)
    close = np.concatenate([np.full(pre, 100.0), np.full(post, spike_to)])
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame([
        {
            "date": dates[i], "symbol": "X",
            "open": close[i], "high": high[i], "low": low[i], "close": close[i],
            "volume": 1.0, "open_interest": 100.0,
            "contract_multiplier": 10.0, "commission": 5.0, "slippage": 0.0,
            "group_name": "G", "margin_rate": 0.10,
        } for i in range(n)
    ])


def _cfg(**kw) -> EngineConfig:
    base = dict(
        initial_capital=1_000_000.0,
        atr_period=2, adx_period=2, cpi_period=5,
        risk_per_trade=0.02, stop_atr_mult=2.0,
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 1.0}, default_group_risk_cap=1.0,
        independent_group_soft_cap=1.0, risk_blowout_cap=float("inf"),
        max_portfolio_leverage=1000.0, min_atr_pct=0.0,
    )
    base.update(kw)
    return EngineConfig(**base)


def _run_once(bars: pd.DataFrame, sig_date: pd.Timestamp, cfg: EngineConfig):
    return StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(_EntryCfg(sig_date)), _PassiveExit())],
    ).run(bars)


def test_profit_target_full_close_fires_at_5r() -> None:
    """v6b style: close_fraction=1.0 → single FULL trade with PROFIT_TARGET reason."""
    bars = _bars_with_spike(pre=5, spike_to=130.0, post=10)
    sig_date = bars["date"].iloc[3]
    cfg = _cfg(profit_target_atr_r=5.0, profit_target_close_fraction=1.0)
    r = _run_once(bars, sig_date, cfg)
    assert len(r.trades) == 1, f"expected 1 trade, got {len(r.trades)}:\n{r.trades}"
    assert "PROFIT_TARGET" in r.trades["exit_reason"].iloc[0]
    assert r.trades["exit_reason"].iloc[0].endswith("_FULL")
    assert r.open_positions.empty


def test_profit_target_half_close_leaves_remaining() -> None:
    """v6a style: close_fraction=0.5 with qty>=2 → partial close; remaining stays."""
    bars = _bars_with_spike(pre=5, spike_to=130.0, post=10)
    sig_date = bars["date"].iloc[3]
    cfg = _cfg(
        risk_per_trade=0.10,  # bigger budget → qty >= 2 ensured
        profit_target_atr_r=5.0,
        profit_target_close_fraction=0.5,
    )
    r = _run_once(bars, sig_date, cfg)
    partial_trades = r.trades[r.trades["exit_reason"].str.contains("PARTIAL", na=False)]
    assert len(partial_trades) == 1, (
        f"expected 1 partial trade, got {len(partial_trades)}:\n"
        f"{r.trades[['qty','exit_reason']]}"
    )
    partial_qty = int(partial_trades["qty"].iloc[0])
    assert partial_qty >= 1
    open_qty = int(r.open_positions["qty"].sum()) if not r.open_positions.empty else 0
    total_qty = open_qty + int(r.trades["qty"].sum())
    assert abs(partial_qty / total_qty - 0.5) < 0.15, (
        f"partial={partial_qty} total={total_qty} ratio={partial_qty/total_qty:.3f}"
    )


def test_profit_target_off_by_default() -> None:
    """Defaults: profit_target_atr_r=0 → no PROFIT_TARGET-reason trade ever."""
    bars = _bars_with_spike(pre=5, spike_to=200.0, post=20)
    sig_date = bars["date"].iloc[3]
    cfg = _cfg()
    r = _run_once(bars, sig_date, cfg)
    for reason in r.trades["exit_reason"].tolist():
        assert "PROFIT_TARGET" not in (reason or "")


def test_profit_target_triggers_only_once() -> None:
    """Profit target fires at most once per position (full-close path)."""
    bars = _bars_with_spike(pre=5, spike_to=130.0, post=15)
    sig_date = bars["date"].iloc[3]
    cfg = _cfg(profit_target_atr_r=5.0, profit_target_close_fraction=1.0)
    r = _run_once(bars, sig_date, cfg)
    pt_trades = r.trades[r.trades["exit_reason"].str.contains("PROFIT_TARGET", na=False)]
    assert len(pt_trades) == 1
