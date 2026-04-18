"""Tests for 1.4 data quality defenses: ATR floor + DQ report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from strats.engine import EngineConfig, StrategyEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
HAB_BARS = REPO_ROOT / "data" / "cache" / "normalized" / "hab_bars.csv"


# ---------- Minimal scripted strategies (signal day X → fill next day) ----------


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


def _make_bars(rows, symbol="X", group="G") -> pd.DataFrame:
    """rows: list of (date_str, open, high, low, close, volume)."""
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df["group_name"] = group
    df["open_interest"] = 100.0
    df["contract_multiplier"] = 10.0
    df["commission"] = 5.0
    df["slippage"] = 1.0
    return df


def _run_with(bars, *, entry_date: str, min_atr_pct: float):
    cfg = EngineConfig(
        initial_capital=1_000_000.0,
        atr_period=2,
        adx_period=2,
        risk_per_trade=0.1,
        stop_atr_mult=2.0,
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 1.0},
        default_group_risk_cap=1.0,
        independent_group_soft_cap=1.0,
        risk_blowout_cap=float("inf"),
        min_atr_pct=min_atr_pct,
    )
    engine = StrategyEngine(
        config=cfg,
        entry_strategy=_ScriptedEntry(_ScriptedEntryConfig(pd.Timestamp(entry_date))),
        exit_strategy=_NoopExit(),
    )
    return engine.run(bars)


# ---------- ATR floor tests ----------


def test_locked_bars_trigger_atr_below_floor() -> None:
    # All four prices equal on every day ⇒ TR ≡ 0 ⇒ ATR = 0.
    bars = _make_bars([
        ("2024-01-02", 100.0, 100.0, 100.0, 100.0, 1.0),
        ("2024-01-03", 100.0, 100.0, 100.0, 100.0, 1.0),
        ("2024-01-04", 100.0, 100.0, 100.0, 100.0, 1.0),
        ("2024-01-05", 100.0, 100.0, 100.0, 100.0, 1.0),
    ])
    r = _run_with(bars, entry_date="2024-01-04", min_atr_pct=0.0025)
    assert len(r.trades) == 0
    # Signal fired on 2024-01-04 but was rejected.
    rej = r.daily_status.loc[r.daily_status["date"] == pd.Timestamp("2024-01-04"), "risk_reject_reason"]
    assert "ATR_BELOW_FLOOR" in rej.values


def test_floor_zero_disables_check() -> None:
    # Same locked bars but min_atr_pct=0 ⇒ no ATR floor; still rejected by the
    # legacy NON_POSITIVE_RISK backstop (because ATR = 0).
    bars = _make_bars([
        ("2024-01-02", 100.0, 100.0, 100.0, 100.0, 1.0),
        ("2024-01-03", 100.0, 100.0, 100.0, 100.0, 1.0),
        ("2024-01-04", 100.0, 100.0, 100.0, 100.0, 1.0),
        ("2024-01-05", 100.0, 100.0, 100.0, 100.0, 1.0),
    ])
    r = _run_with(bars, entry_date="2024-01-04", min_atr_pct=0.0)
    rej = r.daily_status.loc[r.daily_status["date"] == pd.Timestamp("2024-01-04"), "risk_reject_reason"]
    # Old code path: NON_POSITIVE_RISK wins when ATR == 0.
    assert rej.iloc[0] == "NON_POSITIVE_RISK"


def test_normal_atr_passes_floor() -> None:
    # 2-point daily range on close=100 ⇒ ATR ~2 ⇒ 2% ≫ 0.25% floor.
    bars = _make_bars([
        ("2024-01-02", 100.0, 101.0, 99.0, 100.0, 1.0),
        ("2024-01-03", 100.0, 101.0, 99.0, 100.0, 1.0),
        ("2024-01-04", 100.0, 101.0, 99.0, 100.0, 1.0),
        ("2024-01-05", 100.0, 101.0, 99.0, 100.0, 1.0),
        ("2024-01-08", 100.0, 101.0, 99.0, 100.0, 1.0),
    ])
    r = _run_with(bars, entry_date="2024-01-04", min_atr_pct=0.0025)
    # Position should open and remain open at end of data.
    assert len(r.open_positions) == 1


# ---------- DQ report tests ----------


def test_dq_report_computes_expected_stats() -> None:
    """Exercise the DQ computation directly — it's a pure function on bars."""
    bars = pd.concat([
        _make_bars([
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0, 0.0),  # locked + zero-vol
            ("2024-01-03", 100.0, 101.0, 99.0, 100.0, 1.0),
            ("2024-01-04", 100.0, 101.0, 99.0, 102.0, 1.0),   # close > high anomaly
        ], symbol="A"),
        _make_bars([
            ("2024-01-02", 50.0, 51.0, 49.0, 50.0, 1.0),
            ("2024-01-03", 50.0, 51.0, 49.0, 50.0, 1.0),
        ], symbol="B"),
    ]).reset_index(drop=True)

    # Compute DQ report directly — bypasses validate_input_values which would
    # raise on the anomaly row. This mirrors how a user would inspect DQ
    # before running the full backtest.
    engine = StrategyEngine(
        config=EngineConfig(),
        entry_strategy=_ScriptedEntry(_ScriptedEntryConfig(pd.Timestamp("2099-01-01"))),
        exit_strategy=_NoopExit(),
    )
    rep = engine._compute_data_quality_report(bars)
    assert set(rep.columns) == {
        "symbol", "n_bars", "lock_pct", "near_zero_range_pct",
        "zero_volume_pct", "ohlc_anomaly_count",
    }
    rep_a = rep[rep["symbol"] == "A"].iloc[0]
    assert rep_a["n_bars"] == 3
    assert rep_a["lock_pct"] == pytest.approx(1 / 3)
    assert rep_a["zero_volume_pct"] == pytest.approx(1 / 3)
    assert rep_a["ohlc_anomaly_count"] == 1
    rep_b = rep[rep["symbol"] == "B"].iloc[0]
    assert rep_b["lock_pct"] == 0.0
    assert rep_b["ohlc_anomaly_count"] == 0


def test_dq_report_in_backtest_result() -> None:
    """Full engine.run() threads the DQ report into BacktestResult."""
    bars = _make_bars([
        ("2024-01-02", 100.0, 101.0, 99.0, 100.0, 1.0),
        ("2024-01-03", 100.0, 101.0, 99.0, 100.0, 1.0),
        ("2024-01-04", 100.0, 101.0, 99.0, 100.0, 1.0),
    ])
    r = _run_with(bars, entry_date="2099-01-01", min_atr_pct=0.0025)
    assert not r.data_quality_report.empty
    assert r.data_quality_report.iloc[0]["symbol"] == "X"
    assert r.data_quality_report.iloc[0]["n_bars"] == 3


@pytest.mark.skipif(not HAB_BARS.exists(), reason="hab_bars.csv not available")
def test_dq_report_on_real_bars() -> None:
    """Regression: known low-liquidity symbols should top the lock_pct ranking."""
    bars = pd.read_csv(HAB_BARS, parse_dates=["date"])
    # Single-symbol run to avoid _validate_input_values raise on dirty data.
    rb = bars[bars["symbol"] == "RB"].reset_index(drop=True)
    # Minimal exit so run() completes even with no trades.
    from strats.entries.hl_entry import HLEntryStrategy, HLEntryConfig
    from strats.exits.atr_trail_exit import AtrTrailExitStrategy, AtrTrailExitConfig

    cfg = EngineConfig(min_atr_pct=0.0025)
    engine = StrategyEngine(
        config=cfg,
        entry_strategy=HLEntryStrategy(HLEntryConfig(period=21)),
        exit_strategy=AtrTrailExitStrategy(AtrTrailExitConfig(atr_mult=3.0)),
    )
    r = engine.run(rb)
    rep = r.data_quality_report
    assert len(rep) == 1  # only RB
    rb_stats = rep.iloc[0]
    # RB is a liquid contract — very low lock_pct expected.
    assert rb_stats["lock_pct"] < 0.01
    # And no close/high inversions.
    assert rb_stats["ohlc_anomaly_count"] == 0
