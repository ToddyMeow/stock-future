"""Tests for EngineConfig.enable_dual_stream (1.2 continuous-contract pipeline).

Covers:
  - Backward compatibility when flag is False (default).
  - Validation: missing raw columns raises.
  - Single-symbol trade crossing a known roll: segment P&L + roll_cost + new_pnl
    computed by hand and matched.
  - Trade that does NOT cross a roll under dual_stream ≈ single-stream result
    (when raw == Panama).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import pytest

from strats.engine import EngineConfig, StrategyEngine, StrategySlot


# ---------- Minimal test strategies (always enter long day 1, exit day N) ----------

@dataclass(frozen=True)
class _ScriptedEntryConfig:
    entry_signal_date: pd.Timestamp


class _ScriptedEntryStrategy:
    """Signals one long entry on the configured date."""

    def __init__(self, cfg: _ScriptedEntryConfig) -> None:
        self.cfg = cfg

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["entry_trigger_pass"] = out["date"] == self.cfg.entry_signal_date
        out["entry_direction"] = out["entry_trigger_pass"].astype(int)
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {}


@dataclass(frozen=True)
class _ScriptedExitConfig:
    exit_on_or_after: pd.Timestamp


class _ScriptedExitStrategy:
    """Flags pending_exit once the date is >= exit_on_or_after."""

    def __init__(self, cfg: _ScriptedExitConfig) -> None:
        self.cfg = cfg

    def process_close_phase(self, position, row, next_trade_date) -> None:
        position.completed_bars += 1
        date = pd.Timestamp(row["date"])
        if position.pending_exit_reason is None and date >= self.cfg.exit_on_or_after:
            position.pending_exit_reason = "SCRIPTED_EXIT"
            # Exit fills at next trading day's open
            position.pending_exit_date = pd.Timestamp(next_trade_date) if next_trade_date is not None else date


# ---------- Bar factories ----------


def _make_bars(
    panama: list,           # [(date_str, open, high, low, close), ...]
    raw_overrides: dict = None,  # {date_str: (open_raw, high_raw, low_raw, close_raw)}
    contract_by_date: dict = None,  # {date_str: order_book_id}
    settle_overrides: dict = None,  # {date_str: settle_panama}  (settle_raw defaults to close_raw)
    include_settle: bool = False,
    *,
    symbol: str = "X",
    group: str = "G",
    multiplier: float = 10.0,
    commission: float = 5.0,
    slippage: float = 1.0,
    contract: str = "X2401",
) -> pd.DataFrame:
    raw_overrides = raw_overrides or {}
    contract_by_date = contract_by_date or {}
    settle_overrides = settle_overrides or {}
    rows = []
    for ds, o, h, l, c in panama:
        o_raw, h_raw, l_raw, c_raw = raw_overrides.get(ds, (o, h, l, c))
        row = {
            "date": pd.Timestamp(ds),
            "symbol": symbol,
            "open": o, "high": h, "low": l, "close": c,
            "volume": 1000.0, "open_interest": 100.0,
            "contract_multiplier": multiplier,
            "commission": commission, "slippage": slippage,
            "group_name": group,
            "open_raw": o_raw, "high_raw": h_raw,
            "low_raw": l_raw, "close_raw": c_raw,
            "order_book_id": contract_by_date.get(ds, contract),
        }
        if include_settle or settle_overrides:
            row["settle"] = settle_overrides.get(ds, c)
            row["settle_raw"] = settle_overrides.get(ds, c_raw)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_engine(enable_dual_stream: bool, entry_date: str, exit_date: str) -> StrategyEngine:
    cfg = EngineConfig(
        initial_capital=1_000_000.0,
        atr_period=2,
        adx_period=2,
        risk_per_trade=0.1,
        stop_atr_mult=10.0,       # very wide so the stop never triggers
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 1.0},
        default_group_risk_cap=1.0,
        independent_group_soft_cap=1.0,
        risk_blowout_cap=float("inf"),
        enable_dual_stream=enable_dual_stream,
    )
    entry = _ScriptedEntryStrategy(_ScriptedEntryConfig(pd.Timestamp(entry_date)))
    exit_ = _ScriptedExitStrategy(_ScriptedExitConfig(pd.Timestamp(exit_date)))
    return StrategyEngine(
        config=cfg,
        strategies=[StrategySlot("default", entry, exit_)],
    )


# ---------- Tests ----------


def test_prepare_data_missing_raw_columns_raises() -> None:
    # Minimal Panama-only bars
    bars = pd.DataFrame(
        [
            ("2024-01-02", "X", 100.0, 101.0, 99.0, 100.5, 1000, 100, 10, 5, 1, "G"),
            ("2024-01-03", "X", 100.0, 101.0, 99.0, 100.5, 1000, 100, 10, 5, 1, "G"),
        ],
        columns=[
            "date", "symbol", "open", "high", "low", "close",
            "volume", "open_interest", "contract_multiplier",
            "commission", "slippage", "group_name",
        ],
    )
    bars["date"] = pd.to_datetime(bars["date"])

    engine = _make_engine(enable_dual_stream=True, entry_date="2024-01-02", exit_date="2024-01-03")
    with pytest.raises(ValueError, match="enable_dual_stream requires"):
        engine.run(bars)


def test_no_roll_dual_stream_matches_single_stream_when_raw_equals_panama() -> None:
    # Raw == Panama everywhere, no roll. Entry after ATR warmup.
    # 2024-01-02..2024-01-16 trading days (10 bars).
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
             "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
             "2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17"]
    panama = [
        (d, 100.0 + i * 0.1, 100.5 + i * 0.1, 99.5 + i * 0.1, 100.2 + i * 0.1)
        for i, d in enumerate(dates)
    ]
    bars = _make_bars(panama)

    # Entry on 2024-01-09 (8th bar — ATR period=2 is available), exit 2024-01-15.
    eng_off = _make_engine(enable_dual_stream=False, entry_date="2024-01-09", exit_date="2024-01-15")
    eng_on = _make_engine(enable_dual_stream=True, entry_date="2024-01-09", exit_date="2024-01-15")
    r_off = eng_off.run(bars)
    r_on = eng_on.run(bars)

    assert len(r_off.trades) == 1 and len(r_on.trades) == 1
    t_off = r_off.trades.iloc[0]
    t_on = r_on.trades.iloc[0]

    # Core economics should match: entry/exit fill, gross_pnl, net_pnl.
    assert t_off["entry_fill"] == pytest.approx(t_on["entry_fill"])
    assert t_off["exit_fill"] == pytest.approx(t_on["exit_fill"])
    assert t_off["gross_pnl"] == pytest.approx(t_on["gross_pnl"])
    assert t_off["net_pnl"] == pytest.approx(t_on["net_pnl"])
    # Dual-stream augments: zero rolls on this trade.
    assert int(t_on["rolls_crossed"]) == 0
    assert float(t_on["roll_cost_total"]) == 0.0


def test_single_trade_with_one_roll_applies_segment_pnl_and_roll_cost() -> None:
    # Panama continuous; on 2024-01-11 the contract changes AND raw prices
    # jump down by 20 (new contract trades at a lower absolute level).
    # Entry signal: 2024-01-08 (after ATR warmup), exit: 2024-01-15.
    # Entry fills on 2024-01-09, roll on 2024-01-11, exits on 2024-01-16.
    panama = [
        ("2024-01-02", 100.0, 101.0, 99.5, 100.5),
        ("2024-01-03", 100.5, 102.0, 100.2, 101.8),
        ("2024-01-04", 101.8, 103.0, 101.5, 102.5),
        ("2024-01-05", 102.5, 104.0, 102.2, 103.5),
        ("2024-01-08", 103.5, 104.5, 103.0, 104.0),  # signal day
        ("2024-01-09", 104.0, 105.0, 103.8, 104.7),  # entry-fill day
        ("2024-01-10", 104.7, 105.5, 104.5, 105.2),
        ("2024-01-11", 105.2, 106.0, 105.0, 105.7),  # ROLL day
        ("2024-01-12", 105.7, 106.5, 105.5, 106.2),
        ("2024-01-15", 106.2, 107.0, 106.0, 106.7),  # pending-exit fills today
        ("2024-01-16", 106.7, 107.5, 106.5, 107.2),
    ]
    raw_overrides = {
        "2024-01-02": (100.0, 101.0, 99.5, 100.5),
        "2024-01-03": (100.5, 102.0, 100.2, 101.8),
        "2024-01-04": (101.8, 103.0, 101.5, 102.5),
        "2024-01-05": (102.5, 104.0, 102.2, 103.5),
        "2024-01-08": (103.5, 104.5, 103.0, 104.0),
        "2024-01-09": (104.0, 105.0, 103.8, 104.7),
        "2024-01-10": (104.7, 105.5, 104.5, 105.2),
        # Roll: new contract opens 20 points BELOW the old close of yesterday.
        "2024-01-11": (85.2, 86.0, 85.0, 85.7),
        "2024-01-12": (85.7, 86.5, 85.5, 86.2),
        "2024-01-15": (86.2, 87.0, 86.0, 86.7),
        "2024-01-16": (86.7, 87.5, 86.5, 87.2),
    }
    contract_by_date = {
        **{d: "X2401" for d in ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                                "2024-01-08", "2024-01-09", "2024-01-10"]},
        **{d: "X2405" for d in ["2024-01-11", "2024-01-12", "2024-01-15", "2024-01-16"]},
    }
    bars = _make_bars(panama, raw_overrides=raw_overrides, contract_by_date=contract_by_date)

    eng_on = _make_engine(enable_dual_stream=True, entry_date="2024-01-08", exit_date="2024-01-15")
    r = eng_on.run(bars)
    assert len(r.trades) == 1, f"expected 1 trade, got {len(r.trades)}; daily_status:\n{r.daily_status}"
    t = r.trades.iloc[0]

    # Params: mult=10, comm=5, slip=1.
    # entry_fill is Panama (for backward compat with exit strategies).
    # On 2024-01-09 panama_open + 1 slip = 104.0 + 1 = 105.0.
    assert t["entry_fill"] == pytest.approx(105.0)
    # raw_entry_fill uses raw_open on entry day. Here raw == panama pre-roll → 105.0.
    assert t["raw_entry_fill"] == pytest.approx(105.0)
    qty = int(t["qty"])
    assert qty >= 1

    # Roll: 2024-01-10 (X2401) → 2024-01-11 (X2405). 1 roll.
    assert int(t["rolls_crossed"]) == 1
    assert t["entry_contract"] == "X2401"
    assert t["exit_contract"] == "X2405"

    # realized_segment_1 = (raw_close(2024-01-10) - raw_entry_fill) × mult × qty
    #                    = (105.2 - 105.0) × 10 × qty = 2 × qty
    # new_seg_entry      = raw_open(2024-01-11) = 85.2 (no slippage added here)
    # raw_exit_fill      = raw_open(2024-01-16) - 1 slip = 86.7 - 1 = 85.7
    # final_segment      = (85.7 - 85.2) × 10 × qty = 5 × qty
    # gross_pnl          = 2 × qty + 5 × qty = 7 × qty
    expected_gross = 7.0 * qty
    assert t["gross_pnl"] == pytest.approx(expected_gross)
    assert t["raw_exit_fill"] == pytest.approx(85.7)

    # roll_cost = 2 × (comm × qty + slip × mult × qty) = 2 × (5 + 10) × qty = 30 × qty
    expected_roll_cost = 30.0 * qty
    assert t["roll_cost_total"] == pytest.approx(expected_roll_cost)

    # net_pnl = gross - entry_comm - exit_comm - roll_cost
    expected_net = expected_gross - 5.0 * qty - 5.0 * qty - expected_roll_cost
    assert t["net_pnl"] == pytest.approx(expected_net)


# ---------- Settle-based mark tests (1.3) ----------


def test_daily_equity_uses_settle_when_present() -> None:
    """Close ≠ settle on day 5. Daily equity on day 5 should reflect settle, not close."""
    # Modest daily range so ATR > 0 and entry can size.
    panama = [
        ("2024-01-02", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-03", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-04", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-05", 100.0, 102.0, 98.0, 100.0),
        # Day 5: close stays at 100 but settle jumps to 110.
        ("2024-01-08", 100.0, 110.0, 98.0, 100.0),
        ("2024-01-09", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-10", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-11", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-12", 100.0, 102.0, 98.0, 100.0),
    ]
    settle_overrides = {"2024-01-08": 110.0}
    bars = _make_bars(panama, settle_overrides=settle_overrides)

    eng = _make_engine(enable_dual_stream=False, entry_date="2024-01-04", exit_date="2024-01-12")
    r = eng.run(bars)
    # Position stays open through end-of-data (pending_exit can't fill without next_trade_date).
    assert len(r.open_positions) == 1
    pos = r.open_positions.iloc[0]
    qty = int(pos["qty"])
    entry_fill = float(pos["entry_fill"])

    # Daily equity on 2024-01-08 should mark at settle=110, not close=100.
    pd_df = r.portfolio_daily
    eq_day8 = pd_df.loc[pd_df["date"] == pd.Timestamp("2024-01-08"), "equity"].iloc[0]
    expected_eq_with_settle = 1_000_000.0 - 5.0 * qty + (110.0 - entry_fill) * 10.0 * qty
    expected_eq_with_close = 1_000_000.0 - 5.0 * qty + (100.0 - entry_fill) * 10.0 * qty
    assert eq_day8 == pytest.approx(expected_eq_with_settle)
    assert eq_day8 != pytest.approx(expected_eq_with_close)


def test_daily_equity_falls_back_to_close_when_settle_missing() -> None:
    """Backward compat: bars without `settle` column → engine marks with close."""
    panama = [
        ("2024-01-02", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-03", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-04", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-05", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-08", 100.0, 102.0, 98.0, 100.0),
        ("2024-01-09", 100.0, 102.0, 98.0, 100.0),
    ]
    bars = _make_bars(panama)  # no settle column
    assert "settle" not in bars.columns
    eng = _make_engine(enable_dual_stream=False, entry_date="2024-01-04", exit_date="2024-01-09")
    r = eng.run(bars)
    # Position remains open; verify daily equity uses close=100 (constant).
    assert len(r.open_positions) == 1
    pos = r.open_positions.iloc[0]
    qty = int(pos["qty"])
    entry_fill = float(pos["entry_fill"])
    pd_df = r.portfolio_daily
    eq_end = pd_df.iloc[-1]["equity"]
    expected = 1_000_000.0 - 5.0 * qty + (100.0 - entry_fill) * 10.0 * qty
    assert eq_end == pytest.approx(expected)
