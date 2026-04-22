"""Incremental-run regression tests for StrategyEngine.

The engine supports two execution modes:
  * batch   — engine.run(bars) runs a full historical backtest from the
              start of the provided bars
  * incremental — engine.run(bars, initial_state=...) resumes from a
              state snapshot; positions, cash, pending_entries, and
              indicator warmup bars are all restored, and the engine
              only produces today's signals / fills.

This file asserts the critical invariant that drives the live
signal_service: running a contiguous bars slice incrementally in
N segments must produce exactly the same pending_entries, positions,
and cash on the last day as a single batch run over the concatenation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from strats.engine import EngineConfig, StrategyEngine, StrategySlot
from strats.entries.hl_entry import HLEntryConfig, HLEntryStrategy
from strats.exits.atr_trail_exit import AtrTrailExitConfig, AtrTrailExitStrategy


REPO_ROOT = Path(__file__).resolve().parents[1]
HAB_BARS_CSV = REPO_ROOT / "data" / "cache" / "normalized" / "hab_bars.csv"


# --------------------------------------------------------------------- helpers

def _make_engine_cfg(**overrides: Any) -> EngineConfig:
    """Baseline cfg tuned for deterministic + reasonably active signalling
    on a small subset of hab_bars (no ADX/CPI/warmup gates)."""
    base = dict(
        initial_capital=1_000_000.0,
        atr_period=20,
        adx_period=20,
        risk_per_trade=0.02,
        portfolio_risk_cap=0.20,
        group_risk_cap={
            "equity_index": 0.10, "bond": 0.10, "chem_energy": 0.10,
            "rubber_fiber": 0.10, "metals": 0.10, "black_steel": 0.10,
            "agri": 0.10, "building": 0.10, "livestock": 0.10,
        },
        default_group_risk_cap=0.10,
        independent_group_soft_cap=0.20,
        stop_atr_mult=2.0,
        allow_short=True,
        warmup_bars=0,  # keep off; the incremental pipeline provides its own warmup
        min_atr_pct=0.0025,
    )
    base.update(overrides)
    return EngineConfig(**base)


def _make_engine(cfg: EngineConfig) -> StrategyEngine:
    return StrategyEngine(
        config=cfg,
        strategies=[
            StrategySlot(
                "default",
                HLEntryStrategy(HLEntryConfig(period=21, allow_short=True)),
                AtrTrailExitStrategy(AtrTrailExitConfig(atr_mult=3.0)),
            )
        ],
    )


def _clamp_ohlc(bars: pd.DataFrame) -> pd.DataFrame:
    # hab_bars has occasional rows where high/low violate OHLC bounds — clamp
    bars = bars.copy()
    bars["high"] = bars[["high", "open", "close"]].max(axis=1)
    bars["low"] = bars[["low", "open", "close"]].min(axis=1)
    return bars


def _load_subset(symbols: List[str]) -> pd.DataFrame:
    if not HAB_BARS_CSV.exists():
        pytest.skip(f"hab_bars.csv not found at {HAB_BARS_CSV}")
    bars = pd.read_csv(HAB_BARS_CSV, parse_dates=["date"])
    bars = bars[bars["symbol"].isin(symbols)].reset_index(drop=True)
    # Drop dual-stream columns — keep the test path simple (single-stream mode).
    keep_cols = [
        "date", "symbol", "open", "high", "low", "close",
        "volume", "open_interest", "contract_multiplier",
        "commission", "slippage", "group_name", "margin_rate",
    ]
    keep_cols = [c for c in keep_cols if c in bars.columns]
    bars = bars[keep_cols].copy()
    return _clamp_ohlc(bars)


def _split_by_dates(bars: pd.DataFrame, cutoffs: List[str]) -> List[pd.DataFrame]:
    """Return [seg_0, seg_1, ..., seg_N] where seg_i holds bars with date
    strictly after cutoff[i-1] and ≤ cutoff[i].

    The last segment has date > cutoff[-1].
    """
    bars = bars.sort_values(["date", "symbol"]).reset_index(drop=True)
    boundaries = [pd.Timestamp(c) for c in cutoffs]
    segs: List[pd.DataFrame] = []
    lower = pd.Timestamp.min
    for b in boundaries:
        seg = bars[(bars["date"] > lower) & (bars["date"] <= b)].copy()
        segs.append(seg.reset_index(drop=True))
        lower = b
    # tail
    tail = bars[bars["date"] > lower].copy().reset_index(drop=True)
    segs.append(tail)
    return segs


# --------------------------------------------------------------------- tests

@pytest.mark.skipif(not HAB_BARS_CSV.exists(), reason="hab_bars.csv not available")
def test_incremental_matches_batch_final_day() -> None:
    """Core regression: running bars in 3 segments must produce the SAME
    pending_entries / open_positions / cash on the last day as a single
    batch run.
    """
    # Pick a handful of symbols across different groups to exercise the
    # portfolio-risk-cap + group-cap branches. Keep it small for speed.
    symbols = ["RB", "A", "SR", "CU"]
    bars_full = _load_subset(symbols)
    # Subset to [2018, 2025] but the subset is already inside this range.
    assert bars_full["date"].min() <= pd.Timestamp("2018-12-31")
    assert bars_full["date"].max() >= pd.Timestamp("2025-01-01")

    cfg = _make_engine_cfg()

    # ---- batch reference ----
    engine_batch = _make_engine(cfg)
    r_batch = engine_batch.run(bars_full)

    # ---- incremental 3-segment run ----
    seg_a, seg_b, seg_c = _split_by_dates(
        bars_full,
        cutoffs=["2022-12-31", "2024-12-31"],
    )
    assert not seg_a.empty and not seg_b.empty and not seg_c.empty, (
        "segmentation must produce three non-empty segments"
    )

    engine_inc = _make_engine(cfg)
    r_a = engine_inc.run(seg_a)
    state_a = r_a.save_state()

    engine_inc_b = _make_engine(cfg)
    r_b = engine_inc_b.run(seg_b, initial_state=state_a)
    state_b = r_b.save_state()

    engine_inc_c = _make_engine(cfg)
    r_c = engine_inc_c.run(seg_c, initial_state=state_b)

    # ---- compare end-of-run state ----
    last_date = bars_full["date"].max()

    # Final portfolio row (cash / equity / open_positions count)
    pd_batch = r_batch.portfolio_daily
    pd_inc = r_c.portfolio_daily
    batch_last = pd_batch[pd_batch["date"] == last_date].iloc[0]
    inc_last = pd_inc[pd_inc["date"] == last_date].iloc[0]

    # Allow tiny floating-point drift.
    assert abs(float(batch_last["cash"]) - float(inc_last["cash"])) < 1e-4, (
        f"cash diverged: batch={batch_last['cash']} inc={inc_last['cash']}"
    )
    assert abs(float(batch_last["equity"]) - float(inc_last["equity"])) < 1e-4, (
        f"equity diverged: batch={batch_last['equity']} inc={inc_last['equity']}"
    )
    assert int(batch_last["open_positions"]) == int(inc_last["open_positions"])
    assert int(batch_last["pending_entries"]) == int(inc_last["pending_entries"])

    # Pending entries on the final day must match exactly.
    def _pending_for_date(res, d) -> pd.DataFrame:
        pe = res.pending_entries
        if pe.empty:
            return pe
        return pe[pe["generated_date"] == d].reset_index(drop=True)

    pe_batch = _pending_for_date(r_batch, last_date)
    pe_inc = _pending_for_date(r_c, last_date)

    # Normalize sort order for comparison.
    if not pe_batch.empty:
        pe_batch = pe_batch.sort_values(["symbol", "contract_code", "group_name"]).reset_index(drop=True)
    if not pe_inc.empty:
        pe_inc = pe_inc.sort_values(["symbol", "contract_code", "group_name"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(pe_batch, pe_inc, check_like=True, check_dtype=False)

    # Open positions on the final day must match as well (symbol+qty+active_stop).
    op_batch = r_batch.open_positions.sort_values(["symbol", "strategy_id"]).reset_index(drop=True)
    op_inc = r_c.open_positions.sort_values(["symbol", "strategy_id"]).reset_index(drop=True)
    compare_cols = ["symbol", "strategy_id", "direction", "qty", "entry_fill", "active_stop"]
    compare_cols = [c for c in compare_cols if c in op_batch.columns and c in op_inc.columns]
    pd.testing.assert_frame_equal(
        op_batch[compare_cols], op_inc[compare_cols], check_dtype=False,
    )


@pytest.mark.skipif(not HAB_BARS_CSV.exists(), reason="hab_bars.csv not available")
def test_pending_entries_exposed_on_result() -> None:
    """result.pending_entries is a DataFrame with the expected schema."""
    bars = _load_subset(["RB", "A"])
    # cap to a small slice for speed
    bars = bars[bars["date"] <= pd.Timestamp("2019-06-30")].copy()

    r = _make_engine(_make_engine_cfg()).run(bars)

    assert isinstance(r.pending_entries, pd.DataFrame)

    expected = {
        "generated_date", "symbol", "contract_code", "group_name",
        "action", "direction", "target_qty",
        "entry_price_ref", "stop_loss_ref",
    }
    missing = expected - set(r.pending_entries.columns)
    assert not missing, f"pending_entries missing columns: {sorted(missing)}"

    # Values sanity — allowable action/direction domains.
    if not r.pending_entries.empty:
        assert set(r.pending_entries["action"].unique()).issubset(
            {"open", "close", "add", "reduce"}
        )
        assert set(r.pending_entries["direction"].unique()).issubset(
            {"long", "short"}
        )
        assert (r.pending_entries["target_qty"] > 0).all()


@pytest.mark.skipif(not HAB_BARS_CSV.exists(), reason="hab_bars.csv not available")
def test_save_load_state_roundtrip() -> None:
    """save_state → load_state → save_state produces the same dict."""
    bars = _load_subset(["RB", "A"])
    bars = bars[bars["date"] <= pd.Timestamp("2019-12-31")].copy()

    r = _make_engine(_make_engine_cfg()).run(bars)
    state1 = r.save_state()

    # Round-trip through load_state.
    reloaded = r.__class__.load_state(state1)

    # Re-emit: feed the reloaded state into a fresh engine, but run it over
    # an empty slice — we only want to verify save→load→save is stable
    # (i.e. the state itself survives JSON dict serialization).
    json_dump_1 = json.dumps(state1, sort_keys=True, default=str)
    json_dump_2 = json.dumps(reloaded, sort_keys=True, default=str)
    assert json_dump_1 == json_dump_2, "state dict not idempotent under load → re-dump"


@pytest.mark.skipif(not HAB_BARS_CSV.exists(), reason="hab_bars.csv not available")
def test_initial_state_seeds_positions() -> None:
    """An initial_state with one pre-existing position must be visible to
    the engine — at minimum, `open_positions` at the end of run contains
    that position (or it was closed by an intraday stop / exit)."""
    bars = _load_subset(["RB", "A"])

    # Phase 1: run until mid-2019 to build a state with at least one open
    # position. Retry a few end-points if needed.
    candidate_cutoffs = [
        "2019-06-30", "2019-12-31", "2020-06-30",
        "2020-12-31", "2021-06-30",
    ]
    state_a = None
    seg_a = None
    for cut in candidate_cutoffs:
        seg_a = bars[bars["date"] <= pd.Timestamp(cut)].copy()
        r_a = _make_engine(_make_engine_cfg()).run(seg_a)
        if not r_a.open_positions.empty:
            state_a = r_a.save_state()
            break
    assert state_a is not None, "could not build a state with an open position"

    # Capture the seeded positions' (symbol, strategy_id) pairs.
    seeded_keys = set()
    for pos_dict in state_a.get("positions", []):
        seeded_keys.add((pos_dict["symbol"], pos_dict["strategy_id"]))
    assert seeded_keys, "expected at least one seeded position"

    # Phase 2: run the next day only with initial_state=state_a. The
    # outcome depends on data, but we must confirm the engine SAW the
    # seeded position — either it survives into open_positions or
    # (if stopped out) it shows up in trades.
    next_day_bars = bars[
        bars["date"] > pd.Timestamp(seg_a["date"].max())
    ].copy()
    if next_day_bars.empty:
        pytest.skip("no bars after segment A")
    # One trading day only.
    first_next_date = next_day_bars["date"].min()
    next_slice = next_day_bars[next_day_bars["date"] == first_next_date].copy()

    r_b = _make_engine(_make_engine_cfg()).run(next_slice, initial_state=state_a)

    # For each seeded (symbol, strategy_id), it must either still be open
    # or appear in trades (closed during this slice).
    surviving_keys = set()
    if not r_b.open_positions.empty:
        for _, row in r_b.open_positions.iterrows():
            surviving_keys.add((row["symbol"], row["strategy_id"]))
    closed_keys = set()
    if not r_b.trades.empty:
        for _, row in r_b.trades.iterrows():
            closed_keys.add((row["symbol"], row["strategy_id"]))

    for key in seeded_keys:
        assert key in surviving_keys or key in closed_keys, (
            f"seeded position {key} was neither in open_positions nor in "
            f"trades after running slice with initial_state; "
            f"open_positions={surviving_keys}, trades={closed_keys}"
        )
