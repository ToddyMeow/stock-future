"""Indicator warmup gate (1.7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import pytest

from strats.engine import EngineConfig, StrategyEngine


@dataclass(frozen=True)
class _ScriptedEntryConfig:
    entry_signal_date: pd.Timestamp


class _ScriptedEntry:
    def __init__(self, cfg: _ScriptedEntryConfig) -> None:
        self.cfg = cfg

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["entry_trigger_pass"] = out["date"] == self.cfg.entry_signal_date
        out["entry_direction"] = out["entry_trigger_pass"].astype(int)
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {}


class _NoopExit:
    def process_close_phase(self, position, row, next_trade_date) -> None:
        position.completed_bars += 1


def _bars(n: int, start: str = "2024-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=n)
    rows = []
    for d in dates:
        rows.append({
            "date": d, "symbol": "X",
            "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0,
            "volume": 1.0, "open_interest": 100.0,
            "contract_multiplier": 10.0, "commission": 5.0, "slippage": 1.0,
            "group_name": "G",
        })
    return pd.DataFrame(rows)


def _run(bars, *, signal_date: str, warmup_bars: int):
    cfg = EngineConfig(
        initial_capital=1_000_000.0,
        atr_period=2, adx_period=2,
        risk_per_trade=0.1, stop_atr_mult=2.0,
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 1.0}, default_group_risk_cap=1.0,
        independent_group_soft_cap=1.0, risk_blowout_cap=float("inf"),
        warmup_bars=warmup_bars,
    )
    return StrategyEngine(
        config=cfg,
        entry_strategy=_ScriptedEntry(_ScriptedEntryConfig(pd.Timestamp(signal_date))),
        exit_strategy=_NoopExit(),
    ).run(bars)


def test_warmup_rejects_signal_on_early_bar() -> None:
    # 10 bars; signal on bar 3 (bar_index=3). warmup=5 → rejected.
    bars = _bars(10)
    sig_date = bars["date"].iloc[3]
    r = _run(bars, signal_date=sig_date.strftime("%Y-%m-%d"), warmup_bars=5)
    assert len(r.trades) == 0 and len(r.open_positions) == 0
    rej = r.daily_status.loc[r.daily_status["date"] == sig_date, "risk_reject_reason"]
    assert rej.iloc[0] == "WARMUP_INSUFFICIENT"


def test_warmup_allows_signal_after_warmup_period() -> None:
    # 10 bars; signal on bar 6 (bar_index=6). warmup=5 → passes the gate.
    bars = _bars(10)
    sig_date = bars["date"].iloc[6]
    r = _run(bars, signal_date=sig_date.strftime("%Y-%m-%d"), warmup_bars=5)
    # WARMUP_INSUFFICIENT should NOT be the reject on signal day.
    rej = r.daily_status.loc[r.daily_status["date"] == sig_date, "risk_reject_reason"].iloc[0]
    assert rej != "WARMUP_INSUFFICIENT"
