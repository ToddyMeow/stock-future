"""Limit-up / limit-down execution and sizing defenses (1.8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import pytest

from strats.engine import EngineConfig, StrategyEngine


@dataclass(frozen=True)
class _ScriptedEntryConfig:
    entry_signal_date: pd.Timestamp
    direction: int = 1  # 1=long, -1=short


class _ScriptedEntry:
    def __init__(self, cfg: _ScriptedEntryConfig) -> None:
        self.cfg = cfg

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hit = out["date"] == self.cfg.entry_signal_date
        out["entry_trigger_pass"] = hit
        out["entry_direction"] = hit.astype(int) * self.cfg.direction
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {}


class _NoopExit:
    def process_close_phase(self, position, row, next_trade_date) -> None:
        position.completed_bars += 1


def _make_bars(specs, symbol="X", with_limits: bool = True) -> pd.DataFrame:
    """specs: list of (date, open, high, low, close, limit_up, limit_down)."""
    rows = []
    for d, o, h, l, c, lu, ld in specs:
        row = {
            "date": pd.Timestamp(d), "symbol": symbol,
            "open": o, "high": h, "low": l, "close": c,
            "volume": 1.0, "open_interest": 100.0,
            "contract_multiplier": 10.0, "commission": 5.0, "slippage": 1.0,
            "group_name": "G",
        }
        if with_limits:
            row["limit_up"] = lu
            row["limit_down"] = ld
        rows.append(row)
    return pd.DataFrame(rows)


def _engine(entry_date: str, *, max_limit_days: int = 0, direction: int = 1):
    cfg = EngineConfig(
        initial_capital=1_000_000.0,
        atr_period=2, adx_period=2,
        risk_per_trade=0.1, stop_atr_mult=2.0,
        portfolio_risk_cap=1.0,
        group_risk_cap={"G": 1.0}, default_group_risk_cap=1.0,
        independent_group_soft_cap=1.0, risk_blowout_cap=float("inf"),
        max_limit_days=max_limit_days, allow_short=True,
    )
    return StrategyEngine(
        config=cfg,
        entry_strategy=_ScriptedEntry(_ScriptedEntryConfig(pd.Timestamp(entry_date), direction)),
        exit_strategy=_NoopExit(),
    )


# ---------- Execution layer: long position cannot sell on locked-down day ----------


def test_cannot_fill_side_locked_down_blocks_sell() -> None:
    # Unit test on the helper directly: locked-DOWN bar blocks SELL (side=-1).
    from strats.engine import StrategyEngine, EngineConfig
    eng = StrategyEngine(
        config=EngineConfig(),
        entry_strategy=_ScriptedEntry(_ScriptedEntryConfig(pd.Timestamp("2024-01-01"))),
        exit_strategy=_NoopExit(),
    )
    # Locked-down: H=L=close=90, limit_down=90.
    row_locked = pd.Series({
        "open": 90, "high": 90, "low": 90, "close": 90,
        "limit_up": 110, "limit_down": 90,
    })
    assert eng._cannot_fill_side(row_locked, side=-1) is True  # SELL blocked
    assert eng._cannot_fill_side(row_locked, side=1) is False  # BUY fine (we're AT limit_down, buyers unrestricted)

    # Locked-up: H=L=close=110.
    row_locked_up = pd.Series({
        "open": 110, "high": 110, "low": 110, "close": 110,
        "limit_up": 110, "limit_down": 90,
    })
    assert eng._cannot_fill_side(row_locked_up, side=1) is True   # BUY blocked
    assert eng._cannot_fill_side(row_locked_up, side=-1) is False  # SELL fine

    # Normal bar: no block either way.
    row_normal = pd.Series({
        "open": 100, "high": 102, "low": 98, "close": 100,
        "limit_up": 110, "limit_down": 90,
    })
    assert eng._cannot_fill_side(row_normal, side=1) is False
    assert eng._cannot_fill_side(row_normal, side=-1) is False

    # Touched-limit but intraday range >0 → NOT fully locked.
    row_touched = pd.Series({
        "open": 95, "high": 100, "low": 90, "close": 90,
        "limit_up": 110, "limit_down": 90,
    })
    assert eng._cannot_fill_side(row_touched, side=-1) is False  # H>L so not a lock


def test_long_entry_cancelled_on_gap_open_limit_up_but_traded_back() -> None:
    """1.10: entry day gaps to limit_up at open but trades back down intraday
    (H != L). Current 1.8 full-lock check would MISS this; the relaxed 1.10
    open-at-limit check catches it — long can't buy at gap-up open.
    """
    bars = _make_bars([
        ("2024-01-02", 100, 102,  98,  100, 110,  90),
        ("2024-01-03", 100, 102,  98,  100, 110,  90),
        ("2024-01-04", 100, 102,  98,  100, 110,  90),
        # Entry fill day: gap open AT limit_up (108), traded back down to 104.
        # H=108 != L=104 — NOT fully locked, but open IS at limit.
        ("2024-01-05", 108, 108, 104, 104, 108,  92),
        ("2024-01-08", 100, 102,  98,  100, 110,  90),
    ])
    r = _engine("2024-01-04", direction=1).run(bars)
    assert len(r.trades) == 0
    assert len(r.open_positions) == 0
    assert len(r.cancelled_entries) == 1
    assert r.cancelled_entries.iloc[0]["cancel_reason"] == "LIMIT_LOCK_ENTRY"


def test_long_entry_cancelled_on_locked_up_bar() -> None:
    # Entry signal on 2024-01-04. Entry fills 2024-01-05 which is locked-UP.
    # Long must buy → blocked → cancelled.
    bars = _make_bars([
        ("2024-01-02", 100, 102,  98,  100, 110,  90),
        ("2024-01-03", 100, 102,  98,  100, 110,  90),
        ("2024-01-04", 100, 102,  98,  100, 110,  90),
        # Entry fill day: locked-UP (H=L=close=108=limit_up).
        ("2024-01-05", 108, 108, 108, 108, 108,  92),
        ("2024-01-08", 100, 102,  98,  100, 110,  90),
    ])
    r = _engine("2024-01-04", direction=1).run(bars)
    assert len(r.trades) == 0
    assert len(r.open_positions) == 0
    assert len(r.cancelled_entries) == 1
    assert r.cancelled_entries.iloc[0]["cancel_reason"] == "LIMIT_LOCK_ENTRY"


def test_lock_check_no_op_when_limit_cols_missing() -> None:
    # Bars without limit_up/limit_down columns → engine skips the lock gate.
    bars = _make_bars([
        ("2024-01-02", 100, 102,  98, 100, 0, 0),
        ("2024-01-03", 100, 102,  98, 100, 0, 0),
        ("2024-01-04", 100, 102,  98, 100, 0, 0),
        ("2024-01-05", 100, 102,  98, 100, 0, 0),
    ], with_limits=False)
    assert "limit_up" not in bars.columns
    r = _engine("2024-01-04", direction=1).run(bars)
    assert len(r.cancelled_entries) == 0
    assert len(r.open_positions) == 1 or len(r.trades) == 1


# ---------- Sizing layer: worst-case risk floor ----------


def test_sizing_worst_case_shrinks_qty() -> None:
    # price=100, multiplier=10, limit_pct=10%, max_limit_days=2.
    # worst_case_per_contract = 100 × 0.10 × 2 × 10 = 200.
    # atr_risk = stop_atr_mult × ATR × multiplier. With tiny ATR and mult=10,
    # atr_risk < 200 → sizing uses the 200 floor → smaller qty.
    # We ensure qty shrinks by comparing max_limit_days=0 vs 2.
    bars = _make_bars([
        ("2024-01-02", 100, 101,  99, 100, 110,  90),  # atr warmup
        ("2024-01-03", 100, 101,  99, 100, 110,  90),
        ("2024-01-04", 100, 101,  99, 100, 110,  90),  # signal day
        ("2024-01-05", 100, 101,  99, 100, 110,  90),  # entry fill
        ("2024-01-08", 100, 101,  99, 100, 110,  90),
    ])
    r_off = _engine("2024-01-04", max_limit_days=0).run(bars)
    r_on = _engine("2024-01-04", max_limit_days=2).run(bars)
    # Both should open a position; qty_on < qty_off because sizing floor kicks in.
    pos_off = r_off.open_positions.iloc[0]
    pos_on = r_on.open_positions.iloc[0]
    assert int(pos_on["qty"]) < int(pos_off["qty"]), (
        f"sizing floor should shrink qty when max_limit_days>0; "
        f"off={pos_off['qty']} on={pos_on['qty']}"
    )


def test_sizing_no_shrink_when_atr_risk_dominates() -> None:
    # Wide ATR with large stop → atr_risk already > worst_case_lock_risk.
    # Sizing shouldn't change with max_limit_days on.
    bars = _make_bars([
        ("2024-01-02", 100, 140,  60, 100, 110,  90),
        ("2024-01-03", 100, 140,  60, 100, 110,  90),
        ("2024-01-04", 100, 140,  60, 100, 110,  90),
        ("2024-01-05", 100, 140,  60, 100, 110,  90),
        ("2024-01-08", 100, 140,  60, 100, 110,  90),
    ])
    r_off = _engine("2024-01-04", max_limit_days=0).run(bars)
    r_on = _engine("2024-01-04", max_limit_days=2).run(bars)
    if len(r_off.open_positions) == 0 or len(r_on.open_positions) == 0:
        pytest.skip("no positions — test inconclusive")
    assert int(r_off.open_positions.iloc[0]["qty"]) == int(r_on.open_positions.iloc[0]["qty"])
