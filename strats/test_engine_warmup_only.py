"""Tests for the `warmup_until` parameter on StrategyEngine.run().

Three tests:

1. test_warmup_only_no_positions
   bars 跨 2018-2022，warmup_until=2022-12-31 → engine 不产任何持仓、trade、
   portfolio_daily 行；但 engine_state 的 last_close_by_symbol / warmup_bars
   都积累齐。

2. test_warmup_then_real_day
   先 warmup 到 2019-06-29，再 real-mode 跑 2019-06-30 一天，跟"从头 batch
   跑到 2019-06-30"在同一日 pending_entries 应一致。

3. test_initial_state_db_positions_override_virtual
   engine_state.positions 里虚构 2 条（WR/SI），initial_state 传入但通过
   engine.run 时这些虚构持仓被保留为 engine 的起手仓（模拟 bug 场景）；
   对比"传入空 positions"的 control 组，engine 不会把虚构仓当成真实仓去产
   close 指令 — 这个 test 的核心是确认 signal_service 改成 DB-first 后，
   传给 engine 的 positions 由 signal_service 控制，**engine 本身的语义是
   信任 initial_state.positions 的**。
"""

from __future__ import annotations

from datetime import date as _date_type
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from strats.engine import EngineConfig, StrategyEngine, StrategySlot
from strats.entries.hl_entry import HLEntryConfig, HLEntryStrategy
from strats.exits.atr_trail_exit import AtrTrailExitConfig, AtrTrailExitStrategy


REPO_ROOT = Path(__file__).resolve().parents[1]
HAB_BARS_CSV = REPO_ROOT / "data" / "cache" / "normalized" / "hab_bars.csv"


def _make_engine_cfg(**overrides: Any) -> EngineConfig:
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
        warmup_bars=0,
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
    bars = bars.copy()
    bars["high"] = bars[["high", "open", "close"]].max(axis=1)
    bars["low"] = bars[["low", "open", "close"]].min(axis=1)
    return bars


def _load_subset(symbols: List[str]) -> pd.DataFrame:
    if not HAB_BARS_CSV.exists():
        pytest.skip(f"hab_bars.csv not found at {HAB_BARS_CSV}")
    bars = pd.read_csv(HAB_BARS_CSV, parse_dates=["date"])
    bars = bars[bars["symbol"].isin(symbols)].reset_index(drop=True)
    keep_cols = [
        "date", "symbol", "open", "high", "low", "close",
        "volume", "open_interest", "contract_multiplier",
        "commission", "slippage", "group_name", "margin_rate",
    ]
    keep_cols = [c for c in keep_cols if c in bars.columns]
    bars = bars[keep_cols].copy()
    return _clamp_ohlc(bars)


# ---------------------------------------------------------------- tests


@pytest.mark.skipif(not HAB_BARS_CSV.exists(), reason="hab_bars.csv not available")
def test_warmup_only_no_positions() -> None:
    """bars 整段都 warmup → positions / trades / portfolio_daily 都为空，
    但 last_close_by_symbol 和 warmup_bars 必须齐全。"""
    bars = _load_subset(["RB", "A", "SR"])
    bars = bars[bars["date"] <= pd.Timestamp("2022-12-31")].copy()

    cfg = _make_engine_cfg(initial_capital=250_000.0)
    engine = _make_engine(cfg)

    warmup_cutoff = pd.Timestamp("2022-12-31")
    result = engine.run(bars, warmup_until=warmup_cutoff)

    # 1. 无持仓 / 无 pending
    assert result.open_positions.empty, \
        f"open_positions should be empty under warmup-only, got {len(result.open_positions)}"
    # pending_entries DataFrame 本身字段存在；没有新增行
    assert result.pending_entries.empty, \
        f"pending_entries rows should be empty, got {len(result.pending_entries)}"
    # trades 空
    assert result.trades.empty, \
        f"trades should be empty under warmup-only, got {len(result.trades)}"
    # portfolio_daily 空（warmup 日期不 append）
    assert result.portfolio_daily.empty, \
        f"portfolio_daily should be empty, got {len(result.portfolio_daily)}"

    # 2. state 里 positions / pending 为空，cash == initial_capital
    state = result.save_state()
    assert state["positions"] == [], f"state.positions must be [], got {state['positions']}"
    assert state["pending_entries"] == [], \
        f"state.pending_entries must be [], got {state['pending_entries']}"
    assert abs(float(state["cash"]) - 250_000.0) < 1e-6, \
        f"state.cash should be initial_capital, got {state['cash']}"

    # 3. 指标信息真的累积上了
    assert len(state["last_close_by_symbol"]) == 3, \
        f"expected 3 symbols in last_close_by_symbol, got {list(state['last_close_by_symbol'].keys())}"
    for sym in ["RB", "A", "SR"]:
        assert sym in state["last_close_by_symbol"]
        assert np.isfinite(state["last_close_by_symbol"][sym])
    # warmup_bars tail 至少拿到几条（不少于 _INCREMENTAL_WARMUP_BARS 与实际日期的 min）
    assert len(state["warmup_bars"]) > 100, \
        f"warmup_bars tail too small: {len(state['warmup_bars'])}"
    # last_date = warmup_cutoff（所有 bar 都在 cutoff 以内 → terminal = cutoff 或 bars.max）
    assert state["last_date"] is not None


@pytest.mark.skipif(not HAB_BARS_CSV.exists(), reason="hab_bars.csv not available")
def test_warmup_then_real_day() -> None:
    """warmup 到 2019-06-29，再跑 real-mode 2019-06-30 一天，
    vs 一次性 batch 跑到 2019-06-30 的 pending_entries 应一致。"""
    bars_full = _load_subset(["RB", "A", "SR"])
    bars_full = bars_full[bars_full["date"] <= pd.Timestamp("2019-06-30")].copy()

    cfg = _make_engine_cfg()

    # 参照：一次 batch
    engine_batch = _make_engine(cfg)
    r_batch = engine_batch.run(bars_full)
    batch_pe_0630 = r_batch.pending_entries[
        r_batch.pending_entries["generated_date"] == pd.Timestamp("2019-06-30")
    ].copy().reset_index(drop=True)

    # 2 段：warmup 到 06-29，再 real 跑 06-30 一天
    warmup_cutoff = pd.Timestamp("2019-06-29")
    engine_w = _make_engine(cfg)
    r_w = engine_w.run(bars_full, warmup_until=warmup_cutoff)
    state_w = r_w.save_state()

    # 第二段：只喂 06-30 当天的 bars + 用 warmup state
    bars_0630 = bars_full[bars_full["date"] == pd.Timestamp("2019-06-30")].copy()
    engine_r = _make_engine(cfg)
    r_real = engine_r.run(bars_0630, initial_state=state_w)

    real_pe = r_real.pending_entries[
        r_real.pending_entries["generated_date"] == pd.Timestamp("2019-06-30")
    ].copy().reset_index(drop=True)

    # 对齐列顺序后比较行数（完整字段严格一致可能被 indicator 初值 drift 影响
    # — 我们核心要验证的是：engine 知道 06-30 是"正常交易日"，而不是静默跳过）
    # 先验行数；再验 symbol+direction 集合一致
    assert len(batch_pe_0630) == len(real_pe), (
        f"pending count 不一致: batch={len(batch_pe_0630)} warmup+real={len(real_pe)}\n"
        f"batch:\n{batch_pe_0630}\nreal:\n{real_pe}"
    )
    if len(real_pe) > 0:
        batch_keys = set(
            (r["symbol"], r["direction"], r["target_qty"])
            for _, r in batch_pe_0630.iterrows()
        )
        real_keys = set(
            (r["symbol"], r["direction"], r["target_qty"])
            for _, r in real_pe.iterrows()
        )
        assert batch_keys == real_keys, \
            f"pending symbols/direction/qty 不一致:\nbatch={batch_keys}\nreal={real_keys}"


@pytest.mark.skipif(not HAB_BARS_CSV.exists(), reason="hab_bars.csv not available")
def test_initial_state_db_positions_override_virtual() -> None:
    """模拟"信号服务重建 initial_state 用 DB positions 覆盖 prev_state
    虚构 positions"的关键语义：

    - 传 prev_state（含虚构 2 条 positions）时，engine 以此作 ground truth
    - 传 clean_state（positions=[]）时，engine 以空为起手
    - 两次 run 的 open_positions / pending_entries 应该不同，
      证明 engine 完全以 initial_state.positions 为准 — 所以 signal_service
      在上层用 DB 真相去覆盖就能让 engine 正确跑
    """
    bars = _load_subset(["RB", "A"])
    # 只跑 06-30 一天
    one_day = bars[bars["date"] == pd.Timestamp("2019-06-28")].copy()
    if one_day.empty:
        pytest.skip("2019-06-28 on those symbols not available")

    cfg = _make_engine_cfg()

    # 先造一个"带虚构持仓 WR/SI 但其实 WR/SI 不在 symbols 里"的 prev_state
    # —— 这是真实 bug 场景的简化复现：prev_state.positions 里有不该有的仓
    virtual_positions = [
        {
            "symbol": "WR", "strategy_id": "default", "group_name": "black_steel",
            "direction": 1,
            "signal_date": "2019-06-25", "entry_date": "2019-06-26",
            "entry_estimate": 100.0, "entry_fill": 100.0,
            "entry_slippage": 0.0, "qty": 3, "contract_multiplier": 10.0,
            "entry_commission_per_contract": 0.0, "atr_ref": 2.0,
            "volume": 0.0, "open_interest": 0.0,
            "initial_stop": 96.0, "active_stop": 96.0,
            "estimated_initial_risk": 4.0, "estimated_order_risk": 120.0,
            "actual_initial_risk": 4.0, "actual_order_risk": 120.0,
            "risk_blowout_vs_estimate": 0.0, "risk_blowout_ratio": None,
            "r_price": 4.0, "r_money": 120.0,
            "highest_high_since_entry": 100.0, "lowest_low_since_entry": 100.0,
            "completed_bars": 2, "pending_exit_reason": None, "pending_exit_date": None,
            "active_stop_series": [], "mfe_price": 100.0, "mae_price": 100.0,
            "consecutive_fail_count": 0, "original_qty": None,
            "qty_shrink_reason": None, "profit_target_triggered": False,
            "breakeven_triggered": False, "metadata": {},
            "current_contract": None, "raw_entry_fill": None,
            "segment_entry_fill": None, "realized_segment_pnl": 0.0,
            "roll_cost_accrued": 0.0, "rolls_crossed": [],
        },
    ]
    prev_state_with_virtual = {
        "last_date": "2019-06-27",
        "cash": 500_000.0,
        "positions": virtual_positions,
        "pending_entries": [],
        "last_close_by_symbol": {},
        "last_raw_close_by_symbol": {},
        "warmup_bars": [],
    }
    # 干净版：signal_service 按方案 B 组装的 initial_state — 去掉虚构仓
    clean_state = dict(prev_state_with_virtual)
    clean_state["positions"] = []

    # ---- with virtual ----
    engine1 = _make_engine(cfg)
    r1 = engine1.run(one_day, initial_state=prev_state_with_virtual)
    virtual_open_syms = set(r1.open_positions["symbol"].tolist()) if not r1.open_positions.empty else set()

    # ---- without virtual ----
    engine2 = _make_engine(cfg)
    r2 = engine2.run(one_day, initial_state=clean_state)
    clean_open_syms = set(r2.open_positions["symbol"].tolist()) if not r2.open_positions.empty else set()

    # with_virtual 里应该有 WR（engine 信了虚构仓）；without 里一定没有
    assert "WR" in virtual_open_syms or any(
        "WR" == t.get("symbol") for t in (
            r1.trades.to_dict(orient="records") if not r1.trades.empty else []
        )
    ), (
        f"engine 应该把 virtual WR 当作真实仓看见；实际 open={virtual_open_syms} "
        f"trades symbols={sorted(set(r1.trades['symbol'])) if not r1.trades.empty else []}"
    )
    assert "WR" not in clean_open_syms, (
        f"clean state 下 engine 不应凭空有 WR；实际 open={clean_open_syms}"
    )
    # 证明：把虚构仓剥掉后，engine 就不会对不存在的 WR 产生平仓 trade
    if not r2.trades.empty:
        assert "WR" not in set(r2.trades["symbol"]), (
            "clean state 下 engine 不该对 WR 产 trade"
        )
