"""v7 invariants: breakeven stop ratchet (does NOT close, only lifts stop)."""

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
    def process_close_phase(self, position, row, next_trade_date) -> None:
        position.completed_bars += 1


def _bars(close_series: np.ndarray) -> pd.DataFrame:
    n = len(close_series)
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame([
        {
            "date": dates[i], "symbol": "X",
            "open": float(close_series[i]),
            "high": float(close_series[i]) + 1.0,
            "low": float(close_series[i]) - 1.0,
            "close": float(close_series[i]),
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


def _run(bars: pd.DataFrame, sig_date: pd.Timestamp, cfg: EngineConfig):
    return StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", _DummyEntry(_EntryCfg(sig_date)), _PassiveExit())],
    ).run(bars)


def test_breakeven_ratchet_lifts_stop_to_entry() -> None:
    """Spike to 5R then plateau. Stop should ratchet to entry once 5R hit.
    With passive exit (no trailing), position stays open at elevated stop."""
    # Flat 100 → spike to 140 (5R ~= 100 + 5*4 = 120, so 140 > trigger) → stay flat 140
    close = np.concatenate([np.full(5, 100.0), np.full(10, 140.0)])
    bars = _bars(close)
    sig_date = bars["date"].iloc[3]
    cfg = _cfg(breakeven_trigger_atr_r=5.0, breakeven_stop_offset_atr=0.0)
    r = _run(bars, sig_date, cfg)
    # Position should still be open (no exit triggered)
    assert not r.open_positions.empty
    pos = r.open_positions.iloc[0]
    # active_stop should have been ratcheted — at least to entry_fill
    assert pos["active_stop"] >= pos["entry_fill"] - 1e-6, (
        f"expected active_stop ≥ entry_fill, got stop={pos['active_stop']}, entry={pos['entry_fill']}"
    )


def test_breakeven_ratchet_off_by_default() -> None:
    """Default cfg → no breakeven ratchet; initial stop preserved."""
    close = np.concatenate([np.full(5, 100.0), np.full(10, 140.0)])
    bars = _bars(close)
    sig_date = bars["date"].iloc[3]
    cfg = _cfg()
    r = _run(bars, sig_date, cfg)
    assert not r.open_positions.empty
    pos = r.open_positions.iloc[0]
    # active_stop should still be the INITIAL stop (below entry for long)
    assert pos["active_stop"] < pos["entry_fill"]


def test_breakeven_ratchet_with_offset() -> None:
    """offset=1 ATR → stop ratchets to entry + 1×atr_ref."""
    close = np.concatenate([np.full(5, 100.0), np.full(10, 140.0)])
    bars = _bars(close)
    sig_date = bars["date"].iloc[3]
    cfg = _cfg(breakeven_trigger_atr_r=5.0, breakeven_stop_offset_atr=1.0)
    r = _run(bars, sig_date, cfg)
    pos = r.open_positions.iloc[0]
    atr_ref = pos["atr_ref"]
    expected_stop = pos["entry_fill"] + 1.0 * atr_ref
    assert abs(pos["active_stop"] - expected_stop) < 1e-3, (
        f"expected stop={expected_stop:.3f}, got {pos['active_stop']:.3f}"
    )


def test_breakeven_does_not_close_position_on_trigger() -> None:
    """Crucial difference vs profit target: breakeven does NOT produce a trade
    record just because 5R was touched. Position stays open."""
    close = np.concatenate([np.full(5, 100.0), np.full(10, 140.0)])
    bars = _bars(close)
    sig_date = bars["date"].iloc[3]
    cfg = _cfg(breakeven_trigger_atr_r=5.0, breakeven_stop_offset_atr=0.0)
    r = _run(bars, sig_date, cfg)
    # No breakeven-reason trade
    for reason in r.trades["exit_reason"].tolist():
        assert "BREAKEVEN" not in (reason or "")
    # Position still open
    assert not r.open_positions.empty


def test_breakeven_then_retrace_closes_at_breakeven() -> None:
    """After breakeven ratcheted, if price retraces to entry → intraday stop
    triggers at the new (breakeven) stop level."""
    # Flat 100 → spike to 140 → retrace back to 99 (below entry)
    close = np.concatenate([np.full(5, 100.0), np.array([140.0]), np.array([99.0]), np.full(5, 99.0)])
    bars = _bars(close)
    sig_date = bars["date"].iloc[3]
    cfg = _cfg(breakeven_trigger_atr_r=5.0, breakeven_stop_offset_atr=0.0)
    r = _run(bars, sig_date, cfg)
    # Should have closed — trade count >= 1, and position closed near entry
    assert len(r.trades) == 1
    trade = r.trades.iloc[0]
    # Exit fill should be close to entry_fill (breakeven), not the original stop level
    entry_fill = trade["entry_fill"]
    exit_fill = trade["exit_fill"]
    # Exit within small slippage of entry (for breakeven offset=0)
    assert abs(exit_fill - entry_fill) < abs(entry_fill) * 0.05, (
        f"expected exit near entry, got entry={entry_fill:.2f} exit={exit_fill:.2f}"
    )
