"""Strategy engine: composable entry/exit orchestration.

Usage:
    engine = StrategyEngine(
        config=EngineConfig(...),
        strategies=[
            StrategySlot(
                "default",
                HABEntryStrategy(HABEntryConfig(...)),
                HABExitStrategy(HABExitConfig(...)),
            )
        ],
    )
    result = engine.run(bars)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date as _date_type
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from strats.engine_config import EngineConfig, StrategySlot
from strats.helpers import (
    apply_exit_slippage,
    directional_pnl,
    favorable_excursion,
    adverse_excursion,
    adx as _adx,
    wilder_atr as _wilder_atr,
    choppiness_index as _choppiness_index,
    rolling_last_value_percentile as _rolling_last_value_percentile,
)
from strats.position import PendingEntry, Position
from strats.protocols import EntryStrategy, ExitStrategy
from strats.result import BacktestResult

__all__ = [
    "BacktestResult", "EngineConfig", "PendingEntry", "Position",
    "StrategyEngine", "StrategySlot",
]



class StrategyEngine:
    """Composable backtest engine accepting pluggable entry/exit strategies.

    Two execution modes:
      * batch        — engine.run(bars) runs a full historical backtest.
      * incremental  — engine.run(bars, initial_state=result.save_state())
                       resumes from a prior run; positions, cash, pending
                       entries, and enough raw bars for indicator warmup
                       are restored from the state. Only dates in the new
                       bars slice drive today's trading. Used by
                       live/signal_service to produce daily orders.
    """

    # Direction helpers delegate to module-level functions in helpers.py
    _apply_exit_slippage = staticmethod(apply_exit_slippage)
    _directional_pnl = staticmethod(directional_pnl)
    _favorable_excursion = staticmethod(favorable_excursion)
    _adverse_excursion = staticmethod(adverse_excursion)

    def __init__(
        self,
        config: EngineConfig,
        strategies: List[StrategySlot],
    ) -> None:
        self.config = config
        if not strategies:
            raise ValueError("'strategies' must contain at least one StrategySlot")
        self._strategies = strategies
        self._strategy_map: Dict[str, StrategySlot] = {s.strategy_id: s for s in self._strategies}
        # Backward-compat aliases (first strategy)
        self._entry_strategy = self._strategies[0].entry_strategy
        self._exit_strategy = self._strategies[0].exit_strategy
        self._gap_diagnostics: List[Dict[str, Any]] = []

    def _compute_data_quality_report(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Per-symbol diagnostic stats (1.4) computed on raw input bars.

        Pure diagnostic: does NOT mutate bars, does NOT affect the backtest.
        Exposes the two silent landmines — limit-lock bars that degrade ATR,
        and OHLC anomalies (close outside [low, high]) that will trip
        `_validate_input_values` when the affected symbol isn't excluded.
        """
        cfg = self.config
        required = {cfg.symbol_col, cfg.open_col, cfg.high_col, cfg.low_col,
                    cfg.close_col, cfg.volume_col}
        if not required.issubset(bars.columns) or bars.empty:
            return pd.DataFrame(columns=[
                "symbol", "n_bars", "lock_pct", "near_zero_range_pct",
                "zero_volume_pct", "ohlc_anomaly_count",
            ])

        h = bars[cfg.high_col].astype(float)
        l = bars[cfg.low_col].astype(float)
        o = bars[cfg.open_col].astype(float)
        c = bars[cfg.close_col].astype(float)
        v = bars[cfg.volume_col].astype(float)

        locked = (h == l) & (o == c)
        denom = c.where(c != 0)
        rng_pct = (h - l).div(denom).abs()
        near_zero = rng_pct < 0.001  # range < 0.1% of close
        zero_vol = v == 0
        anomaly = (c > h + 1e-9) | (c < l - 1e-9)

        diag = pd.DataFrame({
            cfg.symbol_col: bars[cfg.symbol_col].values,
            "locked": locked.values,
            "near_zero": near_zero.fillna(False).values,
            "zero_vol": zero_vol.values,
            "anomaly": anomaly.values,
        })
        grouped = diag.groupby(cfg.symbol_col, sort=True).agg(
            n_bars=(cfg.symbol_col, "size"),
            lock_count=("locked", "sum"),
            near_zero_count=("near_zero", "sum"),
            zero_vol_count=("zero_vol", "sum"),
            ohlc_anomaly_count=("anomaly", "sum"),
        )
        grouped["lock_pct"] = grouped["lock_count"] / grouped["n_bars"]
        grouped["near_zero_range_pct"] = grouped["near_zero_count"] / grouped["n_bars"]
        grouped["zero_volume_pct"] = grouped["zero_vol_count"] / grouped["n_bars"]
        report = grouped[[
            "n_bars", "lock_pct", "near_zero_range_pct",
            "zero_volume_pct", "ohlc_anomaly_count",
        ]].reset_index()
        return report.rename(columns={cfg.symbol_col: "symbol"})

    def _normalize_and_validate_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Shared preprocessing used by both prepare_data (public) and
        _prepare_all_strategies (internal run path):
          - validate required columns + dual-stream columns
          - normalize date (tz + .normalize) + sort + duplicate check
          - numeric cast, ffill metadata, validate commission/slippage
          - backfill margin_rate, validate input values
          - record gap diagnostics on self._gap_diagnostics

        Does NOT apply `cfg.exclude_symbols` (kept in prepare_data to preserve
        prior behavior where run() ignored that filter).
        """
        cfg = self.config
        self._validate_input_columns(bars)

        df = bars.copy()
        dt = pd.to_datetime(df[cfg.date_col], errors="raise")
        if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
            dt = dt.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        df[cfg.date_col] = dt.dt.normalize()
        df = df.sort_values([cfg.symbol_col, cfg.date_col]).reset_index(drop=True)

        duplicate_mask = df.duplicated(subset=[cfg.symbol_col, cfg.date_col], keep=False)
        if duplicate_mask.any():
            dupes = df.loc[duplicate_mask, [cfg.symbol_col, cfg.date_col]]
            raise ValueError(f"Duplicate symbol/date rows after date normalization:\n{dupes}")

        numeric_cols = [
            cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col,
            cfg.volume_col, cfg.open_interest_col, cfg.multiplier_col,
            cfg.commission_col, cfg.slippage_col,
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="raise")

        for col in [cfg.multiplier_col, cfg.group_col]:
            df[col] = df.groupby(cfg.symbol_col, sort=False)[col].ffill()
            if df[col].isna().any():
                raise ValueError(
                    f"Leading missing values in '{col}'. "
                    "Cannot backfill from future rows."
                )

        for col in [cfg.commission_col, cfg.slippage_col]:
            if df[col].isna().any():
                raise ValueError(
                    f"Column '{col}' contains missing values. "
                    "Provide an explicit cost schedule."
                )

        if cfg.margin_rate_col not in df.columns:
            df[cfg.margin_rate_col] = cfg.default_margin_rate

        self._validate_input_values(df)

        self._gap_diagnostics = []
        for symbol, sym_df in df.groupby(cfg.symbol_col, sort=False):
            gap_days = sym_df[cfg.date_col].diff().dt.days
            suspicious = gap_days > 10
            if suspicious.any():
                for idx in sym_df.index[suspicious]:
                    self._gap_diagnostics.append({
                        "symbol": symbol,
                        "date": sym_df.loc[idx, cfg.date_col],
                        "gap_days": int(gap_days.loc[idx]),
                    })

        return df

    def prepare_data(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Prepare raw bars for backtest.

        The `date` column is assumed to be trading_date semantic (night-session
        bars already attributed to the next trading day by the data source).
        This is enforced upstream at ingestion by the adapter's TradingCalendar
        validation; the engine does not re-validate on each run for performance.
        See data/adapters/trading_calendar.py.
        """
        cfg = self.config
        df = self._normalize_and_validate_bars(bars)

        if cfg.exclude_symbols:
            df = df[~df[cfg.symbol_col].isin(cfg.exclude_symbols)].copy()

        if df.empty:
            prepared = df.copy()
            for col in self._prepared_extra_columns():
                prepared[col] = pd.Series(dtype="float64")
            return prepared

        prepared_frames: List[pd.DataFrame] = []
        for _, symbol_df in df.groupby(cfg.symbol_col, sort=False):
            prepared_frames.append(self._prepare_symbol_frame(symbol_df.reset_index(drop=True)))

        out = pd.concat(prepared_frames, axis=0, ignore_index=True)
        out = out.sort_values([cfg.date_col, cfg.symbol_col]).reset_index(drop=True)
        return out

    def run(
        self,
        bars: pd.DataFrame,
        initial_state: Optional[Dict[str, Any]] = None,
        warmup_until: Optional[Union[pd.Timestamp, _date_type, str]] = None,
    ) -> BacktestResult:
        """Run the backtest.

        batch mode:
            engine.run(bars) — processes every date in `bars`.
        incremental mode:
            engine.run(new_bars, initial_state=prior_result.save_state())
            Splices a raw-bars tail from `initial_state` onto the front of
            `new_bars` so indicator rolling windows are warmed up, then
            restores positions/cash/pending_entries and runs the main
            loop only over `new_bars`' dates.
        warmup-only mode (2026-04-20 新增):
            engine.run(bars, warmup_until=<date>) — bars 里 date ≤ warmup_until
            的段 **只累积指标 / 更新 last_close_by_symbol**，不填单、不开仓、
            不记 trade、不追加 portfolio_daily 行；bars 里 date > warmup_until
            的段按正常交易逻辑处理。用于实盘"从很早日期重跑出干净 indicator
            snapshot，positions 仍为空 / 完全由 DB 提供"。

        Returns a BacktestResult whose `engine_state` is a JSON-compatible
        snapshot suitable for a future `initial_state`, and whose
        `pending_entries` DataFrame lists every newly-generated order
        across the processed dates.
        """
        cfg = self.config
        data_quality_report = self._compute_data_quality_report(bars)

        # ---- incremental: merge warmup bars from initial_state ----
        resume_state = initial_state or {}
        state_last_date: Optional[pd.Timestamp] = None
        if resume_state.get("last_date"):
            state_last_date = pd.Timestamp(resume_state["last_date"]).normalize()

        # ---- warmup-only cutoff: dates ≤ cutoff skip trading; only indicators update ----
        warmup_until_ts: Optional[pd.Timestamp] = None
        if warmup_until is not None:
            warmup_until_ts = pd.Timestamp(warmup_until).normalize()

        bars_for_prepare = self._merge_warmup_bars(bars, resume_state)

        prepared_by_strategy = self._prepare_all_strategies(bars_for_prepare)

        # Use the first strategy's prepared data for backward compat
        # (daily_status, prepared_data in result, and date extraction)
        first_strategy_id = self._strategies[0].strategy_id
        prepared = prepared_by_strategy[first_strategy_id]

        if prepared.empty:
            return BacktestResult(
                trades=self._empty_trades_frame(),
                daily_status=pd.DataFrame(),
                portfolio_daily=pd.DataFrame(
                    columns=[
                        "date",
                        "cash",
                        "equity",
                        "open_positions",
                        "pending_entries",
                        "open_risk",
                        "portfolio_risk_cap",
                        "accepted_signal_risk_today",
                        "total_notional",
                        "leverage",
                    ]
                ),
                open_positions=self._empty_open_positions_frame(),
                prepared_data=prepared,
                cancelled_entries=self._empty_cancelled_entries_frame(),
                data_quality_report=data_quality_report,
                pending_entries=self._empty_pending_entries_frame(),
                engine_state=self._build_initial_engine_state_empty(),
            )

        all_dates = list(pd.Index(prepared[cfg.date_col]).drop_duplicates().sort_values())
        # In incremental mode, we process dates strictly AFTER the prior
        # state's last_date. Earlier dates were consumed only to warm up
        # indicator rolling windows.
        if state_last_date is not None:
            dates = [d for d in all_dates if d > state_last_date]
        else:
            dates = all_dates

        # Pre-index rows by date for the first strategy (used for phases 1-4)
        rows_by_date: Dict[pd.Timestamp, pd.DataFrame] = {
            date: day_df.sort_values(cfg.symbol_col).reset_index(drop=True)
            for date, day_df in prepared.groupby(cfg.date_col, sort=True)
        }

        # Pre-index rows by date per strategy (for signal generation)
        rows_by_date_by_strategy: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
        for sid, sp in prepared_by_strategy.items():
            rows_by_date_by_strategy[sid] = {
                date: day_df.sort_values(cfg.symbol_col).reset_index(drop=True)
                for date, day_df in sp.groupby(cfg.date_col, sort=True)
            }

        PositionKey = Tuple[str, str]  # (symbol, strategy_id)
        positions: Dict[PositionKey, Position] = {}
        pending_entries: Dict[PositionKey, PendingEntry] = {}
        closed_trades: List[Dict[str, Any]] = []
        cancelled_entries: List[Dict[str, Any]] = []
        portfolio_daily: List[Dict[str, Any]] = []
        risk_reject: Dict[Tuple[pd.Timestamp, str, str], Optional[str]] = {}  # (date, symbol, strategy_id)
        last_close_by_symbol: Dict[str, float] = {}
        # Dual-stream raw-close tracking (populated only when enable_dual_stream).
        # At step 0 (roll check) these hold yesterday's values; step 4 updates
        # them to today's values for tomorrow's iteration.
        last_raw_close_by_symbol: Dict[str, float] = {}
        # Track, per processed date, the signals that were promoted to
        # pending entries. Drives the public `pending_entries` DataFrame.
        per_day_new_pending: List[Dict[str, Any]] = []

        # Decide the marking column once (1.3): prefer `settle` if it's in the
        # prepared frame, else fall back to `close`. The variable name
        # `close_map` is kept for minimal diff — semantically it's a mark map.
        mark_col_effective = cfg.settle_col if cfg.settle_col in prepared.columns else cfg.close_col
        mark_raw_col_effective = (
            cfg.settle_raw_col if cfg.settle_raw_col in prepared.columns else cfg.raw_close_col
        )

        cash = float(cfg.initial_capital)

        # ---- incremental: restore state ----
        if resume_state:
            cash = float(resume_state.get("cash", cash))
            for pos_d in resume_state.get("positions", []):
                pos = self._position_from_state_dict(pos_d)
                positions[(pos.symbol, pos.strategy_id)] = pos
            for pe_d in resume_state.get("pending_entries", []):
                pe = self._pending_entry_from_state_dict(pe_d)
                pending_entries[(pe.symbol, pe.strategy_id)] = pe
            for sym, v in resume_state.get("last_close_by_symbol", {}).items():
                last_close_by_symbol[str(sym)] = float(v)
            for sym, v in resume_state.get("last_raw_close_by_symbol", {}).items():
                last_raw_close_by_symbol[str(sym)] = float(v)

        for date in dates:
            day_df = rows_by_date[date]
            close_map: Dict[str, float] = {
                str(row[cfg.symbol_col]): float(row[mark_col_effective])
                for _, row in day_df.iterrows()
            }
            raw_close_map: Dict[str, float] = (
                {
                    str(row[cfg.symbol_col]): float(row[mark_raw_col_effective])
                    for _, row in day_df.iterrows()
                }
                if cfg.enable_dual_stream
                else {}
            )

            # warmup-only 分支（2026-04-20）：date ≤ warmup_until_ts 时
            # 只更新 last_close_by_symbol（给后续 indicator state 做收尾），
            # 跳过所有 position / pending / trade / portfolio_daily 的变更。
            # 保证最终 engine_state 里 positions/pending=空，cash=初始值；
            # 指标快照（last_close_by_symbol + warmup_bars tail）正常累积。
            if warmup_until_ts is not None and date <= warmup_until_ts:
                for _, row in day_df.iterrows():
                    sym = str(row[cfg.symbol_col])
                    last_close_by_symbol[sym] = float(row[mark_col_effective])
                    if cfg.enable_dual_stream:
                        last_raw_close_by_symbol[sym] = float(row[mark_raw_col_effective])
                continue

            # 0) Dual-stream: if any open position's contract changed today,
            #    close the prior segment and start a new one. Must run BEFORE
            #    step 4 updates last_raw_close_by_symbol — at this point it
            #    still holds yesterday's values, which is what we need.
            if cfg.enable_dual_stream:
                for _, row in day_df.iterrows():
                    symbol = str(row[cfg.symbol_col])
                    keys_for_symbol = [k for k in positions if k[0] == symbol]
                    for key in keys_for_symbol:
                        self._check_and_apply_roll(
                            position=positions[key],
                            row=row,
                            prev_raw_close_by_symbol=last_raw_close_by_symbol,
                        )

            # 1) Existing positions: open-gap stop, pending open exits, intraday stop.
            for _, row in day_df.iterrows():
                symbol = str(row[cfg.symbol_col])
                keys_for_symbol = [k for k in positions if k[0] == symbol]
                for key in keys_for_symbol:
                    position = positions[key]
                    exit_records = self._process_open_and_intraday_for_existing_position(
                        position=position,
                        row=row,
                    )
                    for exit_record in exit_records:
                        cash += float(exit_record.pop("cash_delta"))
                        is_partial = bool(exit_record.pop("is_partial", False))
                        # Preserve exit_reason for SAR hook before closed_trades.append
                        exit_reason_closed = exit_record.get("exit_reason", "")
                        closed_trades.append(exit_record)
                        if not is_partial:
                            del positions[key]
                            # v10: stop-and-reverse synthesis (position var still
                            # refers to the just-closed Position after del).
                            # Per-slot SAR: check slot override before firing.
                            slot_sar_on, _, _ = self._resolve_slot_sar(
                                position.strategy_id
                            )
                            if (
                                slot_sar_on
                                and exit_reason_closed in cfg.reverse_eligible_reasons
                            ):
                                equity_est = (
                                    float(portfolio_daily[-1]["equity"])
                                    if portfolio_daily
                                    else cash
                                )
                                reverse_pending = self._try_synthesize_reverse_entry(
                                    closed_position=position,
                                    row=row,
                                    equity_estimate=equity_est,
                                    positions=positions,
                                    pending_entries=pending_entries,
                                    close_map=close_map,
                                    last_close_by_symbol=last_close_by_symbol,
                                )
                                if reverse_pending is not None:
                                    pending_entries[key] = reverse_pending

            # 2) Pending entries fill at today's open.
            for _, row in day_df.iterrows():
                symbol = str(row[cfg.symbol_col])
                keys_for_symbol = [k for k in pending_entries if k[0] == symbol]
                for key in keys_for_symbol:
                    pending = pending_entries[key]
                    if pending.entry_date != date:
                        continue
                    if key in positions:
                        continue

                    entry_fill = self._estimate_entry_fill_from_row(row, direction=pending.direction)

                    # Limit-lock block (1.8): cancel entry if today's bar is
                    # fully locked against our entry side. Long entry buys
                    # (blocked by locked-UP); short entry sells (blocked by
                    # locked-DOWN). No retry — a fresh signal must fire on a
                    # later day to re-enter.
                    if self._cannot_fill_side(row, pending.direction):
                        cancelled_entries.append(
                            self._build_cancelled_entry(
                                pending=pending,
                                row=row,
                                attempted_entry_fill=entry_fill,
                                cancel_reason="LIMIT_LOCK_ENTRY",
                            )
                        )
                        del pending_entries[key]
                        continue

                    # entry_fill + initial_stop are both in Panama space, so direct compare.
                    stop_invalid = (
                        (pending.direction == 1 and entry_fill <= pending.initial_stop + cfg.eps)
                        or (pending.direction == -1 and entry_fill >= pending.initial_stop - cfg.eps)
                    )
                    if stop_invalid:
                        cancelled_entries.append(
                            self._build_cancelled_entry(
                                pending=pending,
                                row=row,
                                attempted_entry_fill=entry_fill,
                                cancel_reason="OPEN_INVALIDATES_STOP",
                            )
                        )
                        del pending_entries[key]
                        continue

                    # Risk blowout check: enforce hard cap on actual vs estimated risk.
                    contract_multiplier = float(row[cfg.multiplier_col])
                    actual_initial_risk = abs(entry_fill - pending.initial_stop)
                    actual_order_risk = actual_initial_risk * contract_multiplier * pending.qty
                    blowout_ratio = (
                        actual_order_risk / pending.estimated_order_risk
                        if pending.estimated_order_risk > cfg.eps
                        else float("inf")
                    )

                    qty_override: Optional[int] = None
                    original_qty_record: Optional[int] = None
                    shrink_reason: Optional[str] = None

                    if blowout_ratio > cfg.risk_blowout_cap:
                        if cfg.risk_blowout_action == "CANCEL":
                            cancelled_entries.append(
                                self._build_cancelled_entry(
                                    pending=pending,
                                    row=row,
                                    attempted_entry_fill=entry_fill,
                                    cancel_reason="RISK_BLOWOUT_CANCEL",
                                )
                            )
                            del pending_entries[key]
                            continue
                        else:
                            # SHRINK: reduce qty to bring risk within cap.
                            per_contract_actual_risk = actual_initial_risk * contract_multiplier
                            max_allowed_risk = cfg.risk_blowout_cap * pending.estimated_order_risk
                            shrunk_qty = math.floor(max_allowed_risk / per_contract_actual_risk) if per_contract_actual_risk > cfg.eps else 0
                            if shrunk_qty < 1:
                                cancelled_entries.append(
                                    self._build_cancelled_entry(
                                        pending=pending,
                                        row=row,
                                        attempted_entry_fill=entry_fill,
                                        cancel_reason="RISK_BLOWOUT_SHRINK_TO_ZERO",
                                    )
                                )
                                del pending_entries[key]
                                continue
                            original_qty_record = pending.qty
                            qty_override = shrunk_qty
                            shrink_reason = "RISK_BLOWOUT"

                    position, cash_entry_delta = self._fill_pending_entry(
                        pending=pending,
                        row=row,
                        entry_fill=entry_fill,
                        qty_override=qty_override,
                        original_qty=original_qty_record,
                        qty_shrink_reason=shrink_reason,
                    )
                    cash += cash_entry_delta
                    positions[key] = position
                    del pending_entries[key]

                    immediate_records = self._process_open_and_intraday_for_existing_position(
                        position=position,
                        row=row,
                    )
                    for immediate_exit_record in immediate_records:
                        cash += float(immediate_exit_record.pop("cash_delta"))
                        is_partial = bool(immediate_exit_record.pop("is_partial", False))
                        imm_exit_reason = immediate_exit_record.get("exit_reason", "")
                        closed_trades.append(immediate_exit_record)
                        if not is_partial:
                            del positions[key]
                            # v10: stop-and-reverse after same-bar stop-out of
                            # newly filled entry. Per-slot SAR resolution.
                            slot_sar_on, _, _ = self._resolve_slot_sar(
                                position.strategy_id
                            )
                            if (
                                slot_sar_on
                                and imm_exit_reason in cfg.reverse_eligible_reasons
                            ):
                                equity_est = (
                                    float(portfolio_daily[-1]["equity"])
                                    if portfolio_daily
                                    else cash
                                )
                                imm_reverse = self._try_synthesize_reverse_entry(
                                    closed_position=position,
                                    row=row,
                                    equity_estimate=equity_est,
                                    positions=positions,
                                    pending_entries=pending_entries,
                                    close_map=close_map,
                                    last_close_by_symbol=last_close_by_symbol,
                                )
                                if imm_reverse is not None:
                                    pending_entries[key] = imm_reverse

            # 3) Close-phase logic for surviving positions.
            for _, row in day_df.iterrows():
                symbol = str(row[cfg.symbol_col])
                keys_for_symbol = [k for k in positions if k[0] == symbol]
                for key in keys_for_symbol:
                    position = positions[key]
                    slot = self._strategy_map[position.strategy_id]
                    slot.exit_strategy.process_close_phase(
                        position=position, row=row, next_trade_date=row["next_trade_date"],
                    )

            # 4) Update last available mark per symbol (settle when present, else close).
            for _, row in day_df.iterrows():
                sym = str(row[cfg.symbol_col])
                last_close_by_symbol[sym] = float(row[mark_col_effective])
                if cfg.enable_dual_stream:
                    last_raw_close_by_symbol[sym] = float(row[mark_raw_col_effective])

            # 5) Mark portfolio at today's close.
            equity_close = self._compute_equity_close(
                cash=cash,
                positions=positions,
                close_map=close_map,
                last_close_by_symbol=last_close_by_symbol,
                raw_close_map=raw_close_map if cfg.enable_dual_stream else None,
                last_raw_close_by_symbol=last_raw_close_by_symbol if cfg.enable_dual_stream else None,
            )
            open_risk_total, _ = self._compute_open_risk(
                positions=positions,
                close_map=close_map,
                last_close_by_symbol=last_close_by_symbol,
            )

            # 5b) Soft cap — legacy "floating P&L between mark and stop"
            # (definition C). This does NOT reject anything; it logs a
            # warning when the sum across open positions exceeds
            # equity × unrealized_exposure_soft_cap so traders can notice
            # that a winning position chain has stretched stops very far
            # from the current mark. `open_risk_total` above is the hard
            # principal-risk aggregate.
            unrealized_exposure_total, _ = self._compute_unrealized_exposure(
                positions=positions,
                close_map=close_map,
                last_close_by_symbol=last_close_by_symbol,
            )
            soft_cap_ratio = cfg.unrealized_exposure_soft_cap
            soft_cap_triggered = False
            if (
                soft_cap_ratio > 0.0
                and equity_close > cfg.eps
                and unrealized_exposure_total > equity_close * soft_cap_ratio
            ):
                soft_cap_triggered = True
                logger.warning(
                    "unrealized_exposure_soft_cap breach date=%s "
                    "exposure=%.0f equity=%.0f ratio=%.3f threshold=%.3f "
                    "positions=%d",
                    pd.Timestamp(date).date() if hasattr(date, "to_pydatetime") else date,
                    unrealized_exposure_total,
                    equity_close,
                    unrealized_exposure_total / equity_close,
                    soft_cap_ratio,
                    len(positions),
                )

            # 6) Signal generation / next-open pending entries.
            # Initialize risk_reject for all strategies x symbols
            for slot in self._strategies:
                slot_day = rows_by_date_by_strategy[slot.strategy_id].get(date)
                if slot_day is not None:
                    for _, row in slot_day.iterrows():
                        risk_reject[(date, str(row[cfg.symbol_col]), slot.strategy_id)] = None

            accepted_today_risk_total = 0.0
            accepted_notional_today = 0.0
            accepted_margin_today = 0.0
            base_notional = self._compute_total_notional(
                positions, pending_entries, close_map, last_close_by_symbol,
            )
            # Base occupied margin across open positions (4.1). Only compute
            # when the cap is enabled — cheap path stays cheap.
            if cfg.max_margin_utilization > 0.0:
                base_occupied_margin = 0.0
                for pos in positions.values():
                    mark = close_map.get(pos.symbol, last_close_by_symbol.get(pos.symbol, pos.entry_fill))
                    base_rate = cfg.default_margin_rate
                    # Try to pick up today's margin_rate if bars carry it per-row.
                    day_df = rows_by_date.get(date)
                    if day_df is not None:
                        sym_row = day_df[day_df[cfg.symbol_col] == pos.symbol]
                        if not sym_row.empty and cfg.margin_rate_col in sym_row.columns:
                            base_rate = float(sym_row.iloc[0][cfg.margin_rate_col])
                    eff_rate = self._effective_margin_rate(base_rate, pos.current_contract, date)
                    base_occupied_margin += pos.qty * mark * pos.contract_multiplier * eff_rate
            else:
                base_occupied_margin = 0.0

            # v9: optional per-day slot shuffle for order-sensitivity MC.
            # Uses stdlib random seeded deterministically from (seed, date.toordinal())
            # so each day has a reproducible permutation under a fixed global seed.
            if cfg.slot_permutation_seed is None:
                slot_iter = list(self._strategies)
            else:
                import random as _random
                _rng = _random.Random((int(cfg.slot_permutation_seed), int(date.toordinal())))
                slot_iter = list(self._strategies)
                _rng.shuffle(slot_iter)

            for slot in slot_iter:
                slot_day = rows_by_date_by_strategy[slot.strategy_id].get(date)
                if slot_day is None:
                    continue

                candidate_rows = [
                    row
                    for _, row in slot_day.iterrows()
                    if bool(row["entry_trigger_pass"])
                ]
                # 2026-04-22 修 reject 字母序先到先得 bug（见 docs/audit_2026_04_22.md）
                # 原 sort by (next_trade_date, symbol 字母序)，导致字母靠前的
                # symbol 永远先占 portfolio_cap / group_cap，字母靠后的同 bar
                # 一致被 reject（结构性不公平）。
                #
                # 方案演进：
                #   v1 (放弃): per-bar 随机 shuffle — 25 万档 cap binding 频繁时，
                #              每 bar 随机顺序带来成交序列噪声，Sharpe 从 2.03 → 0.32。
                #   v2 (当前): 按 per_contract_risk 升序（proxy: atr_ref × multiplier）。
                #              小仓先占 cap → 多样化保留；大仓后到若撞 cap 就让位给下一个。
                #              deterministic、无字母 bias、无随机噪声。字母序作 tiebreaker。
                def _candidate_sort_key(row):
                    nd = row["next_trade_date"]
                    nd_ts = pd.Timestamp.max if pd.isna(nd) else pd.Timestamp(nd)
                    atr_v = row.get("atr_ref")
                    mult_v = row.get(cfg.multiplier_col)
                    try:
                        risk_proxy = float(atr_v) * float(mult_v)
                    except (TypeError, ValueError):
                        risk_proxy = 0.0
                    return (nd_ts, risk_proxy, str(row[cfg.symbol_col]))
                candidate_rows.sort(key=_candidate_sort_key)

                for row in candidate_rows:
                    symbol = str(row[cfg.symbol_col])
                    key: PositionKey = (symbol, slot.strategy_id)
                    group_name = str(row[cfg.group_col])
                    direction = int(row["entry_direction"])
                    reason: Optional[str] = None

                    # Warmup gate (1.7): skip bars before indicators converge.
                    bar_idx = int(row["_bar_index"]) if "_bar_index" in row.index else 0
                    if cfg.warmup_bars > 0 and bar_idx < cfg.warmup_bars:
                        reason = "WARMUP_INSUFFICIENT"
                    elif pd.isna(row["next_trade_date"]):
                        reason = "NO_NEXT_TRADE_DATE"
                    elif key in positions:
                        reason = "ALREADY_IN_POSITION"
                    elif cfg.symbol_position_lock and any(k[0] == symbol for k in positions):
                        # v4a: another strategy already holds this symbol; first-fire wins.
                        reason = "SYMBOL_LOCKED"
                    elif cfg.symbol_position_lock and any(k[0] == symbol for k in pending_entries):
                        reason = "SYMBOL_LOCKED"
                    elif key in pending_entries:
                        reason = "PENDING_ENTRY_EXISTS"
                    else:
                        entry_estimate = float(row[cfg.close_col])
                        atr_ref_val = float(row["atr_ref"]) if pd.notna(row.get("atr_ref")) else np.nan
                        # ATR floor (1.4): reject when atr_ref is degenerately
                        # small vs close, which would blow up qty on limit-lock
                        # bars. NaN atr_ref also falls here when floor is on.
                        atr_below_floor = (
                            cfg.min_atr_pct > 0.0
                            and abs(entry_estimate) > 0.0
                            and (
                                not np.isfinite(atr_ref_val)
                                or atr_ref_val < cfg.min_atr_pct * abs(entry_estimate)
                            )
                        )
                        # v5: Congestion filter. Hard gate — requires BOTH
                        # Choppiness Index < threshold AND ADX > threshold.
                        # Missing inputs (NaN) are treated as "cannot confirm
                        # trend" → congested (safer default than pass-through).
                        congested = False
                        if cfg.use_congestion_filter:
                            cpi_val = row.get("cpi", np.nan)
                            adx_val = row.get("adx", np.nan)
                            if (
                                not np.isfinite(cpi_val)
                                or not np.isfinite(adx_val)
                                or cpi_val >= cfg.congestion_cpi_threshold
                                or adx_val < cfg.congestion_adx_threshold
                            ):
                                congested = True
                        if congested:
                            reason = "CONGESTION_LOCKED"
                        elif atr_below_floor:
                            reason = "ATR_BELOW_FLOOR"
                        else:
                            contract_multiplier = float(row[cfg.multiplier_col])
                            estimated_initial_risk = cfg.stop_atr_mult * atr_ref_val if np.isfinite(atr_ref_val) else np.nan
                            if direction == 1:
                                initial_stop = entry_estimate - estimated_initial_risk
                            else:
                                initial_stop = entry_estimate + estimated_initial_risk
                            per_contract_risk_est = estimated_initial_risk * contract_multiplier

                            # Limit-lock sizing floor (1.8): if max_limit_days>0 and
                            # bars carry limit_up/limit_down, enforce that the
                            # per-contract risk used for qty >= worst-case N-day lock.
                            if cfg.max_limit_days > 0:
                                lim_up = row.get(cfg.limit_up_col)
                                lim_dn = row.get(cfg.limit_down_col)
                                if pd.notna(lim_up) and pd.notna(lim_dn) and entry_estimate > 0:
                                    up_pct = (float(lim_up) - entry_estimate) / entry_estimate
                                    dn_pct = (entry_estimate - float(lim_dn)) / entry_estimate
                                    limit_pct = max(up_pct, dn_pct, 0.0)
                                    worst_case_per_contract = (
                                        entry_estimate * limit_pct * cfg.max_limit_days
                                        * contract_multiplier
                                    )
                                    per_contract_risk_est = max(per_contract_risk_est, worst_case_per_contract)

                            if not np.isfinite(per_contract_risk_est) or per_contract_risk_est <= 0.0:
                                reason = "NON_POSITIVE_RISK"
                            else:
                                # 2026-04-22 去掉 ADX trend_score 缩放（见 docs/audit_2026_04_22.md）
                                # 原 `effective_risk = risk_per_trade × trend_score(ADX)` 时序反利弗莫尔：
                                # ADX 滞后 → 趋势初期 ADX 低 → 仓轻；趋势末期 ADX 高 → 仓重。
                                # 现在恢复简单：每笔 entry 都用满 risk_per_trade。
                                # 趋势筛选由 use_congestion_filter（CPI + ADX 硬拒）承担。
                                effective_risk = cfg.risk_per_trade
                                risk_budget_single = equity_close * effective_risk
                                qty = math.floor(risk_budget_single / per_contract_risk_est)
                                if qty < 1:
                                    reason = "QTY_LT_1"
                                else:
                                    order_risk = per_contract_risk_est * qty
                                    entry_date = pd.Timestamp(row["next_trade_date"])
                                    effective_open_risk, effective_open_risk_by_group = (
                                        self._compute_effective_open_risk_for_entry_date(
                                            candidate_entry_date=entry_date,
                                            positions=positions,
                                            pending_entries=pending_entries,
                                            close_map=close_map,
                                            last_close_by_symbol=last_close_by_symbol,
                                        )
                                    )
                                    portfolio_cap = equity_close * cfg.portfolio_risk_cap
                                    if group_name.startswith("ind_"):
                                        group_cap = equity_close * cfg.default_group_risk_cap
                                    else:
                                        group_cap = equity_close * cfg.group_risk_cap.get(group_name, cfg.default_group_risk_cap)
                                    portfolio_risk_if_filled = effective_open_risk + order_risk
                                    group_risk_if_filled = effective_open_risk_by_group.get(group_name, 0.0) + order_risk

                                    # Leverage check (base_notional hoisted before loop)
                                    new_notional = entry_estimate * contract_multiplier * qty
                                    total_notional_if_filled = base_notional + accepted_notional_today + new_notional

                                    if portfolio_risk_if_filled > portfolio_cap + cfg.eps:
                                        reason = "PORTFOLIO_RISK_CAP"
                                    elif cfg.use_group_risk_cap and group_risk_if_filled > group_cap + cfg.eps:
                                        reason = "GROUP_RISK_CAP"
                                    elif cfg.use_group_risk_cap and group_name.startswith("ind_"):
                                        ind_risk = sum(r for g, r in effective_open_risk_by_group.items() if g.startswith("ind_"))
                                        if ind_risk + order_risk > equity_close * cfg.independent_group_soft_cap + cfg.eps:
                                            reason = "INDEPENDENT_SOFT_CAP"

                                    if reason is None and equity_close > cfg.eps and total_notional_if_filled / equity_close > cfg.max_portfolio_leverage:
                                        reason = "LEVERAGE_CAP"

                                    # Margin utilization cap (4.1).
                                    candidate_margin = 0.0
                                    if reason is None and cfg.max_margin_utilization > 0.0:
                                        base_rate = float(row[cfg.margin_rate_col]) if cfg.margin_rate_col in row.index else cfg.default_margin_rate
                                        candidate_contract = str(row[cfg.contract_col]) if cfg.contract_col in row.index else None
                                        eff_rate = self._effective_margin_rate(base_rate, candidate_contract, date)
                                        candidate_margin = entry_estimate * contract_multiplier * qty * eff_rate
                                        if base_occupied_margin + accepted_margin_today + candidate_margin > equity_close * cfg.max_margin_utilization + cfg.eps:
                                            reason = "MARGIN_CAP"

                                    if reason is None:
                                        new_pending = PendingEntry(
                                            symbol=symbol,
                                            strategy_id=slot.strategy_id,
                                            group_name=group_name,
                                            direction=direction,
                                            signal_date=date,
                                            entry_date=entry_date,
                                            entry_estimate=entry_estimate,
                                            qty=qty,
                                            atr_ref=float(row["atr_ref"]),
                                            volume=float(row[cfg.volume_col]),
                                            open_interest=float(row[cfg.open_interest_col]),
                                            initial_stop=initial_stop,
                                            estimated_initial_risk=estimated_initial_risk,
                                            estimated_order_risk=order_risk,
                                            contract_multiplier_est=contract_multiplier,
                                            metadata=slot.entry_strategy.build_pending_entry_metadata(row),
                                        )
                                        pending_entries[key] = new_pending
                                        accepted_today_risk_total += order_risk
                                        accepted_notional_today += new_notional
                                        accepted_margin_today += candidate_margin
                                        # Record for result.pending_entries output.
                                        contract_code = (
                                            str(row[cfg.contract_col])
                                            if cfg.contract_col in row.index
                                            and pd.notna(row.get(cfg.contract_col))
                                            else symbol
                                        )
                                        per_day_new_pending.append({
                                            "generated_date": date,
                                            "symbol": symbol,
                                            "contract_code": contract_code,
                                            "group_name": group_name,
                                            "strategy_id": slot.strategy_id,
                                            "action": "open",
                                            "direction": "long" if direction == 1 else "short",
                                            "target_qty": int(qty),
                                            "entry_price_ref": float(entry_estimate),
                                            "stop_loss_ref": float(initial_stop),
                                            "entry_date": entry_date,
                                        })

                    risk_reject[(date, symbol, slot.strategy_id)] = reason

            total_notional = base_notional + accepted_notional_today
            leverage = total_notional / equity_close if equity_close > cfg.eps else 0.0

            portfolio_daily.append(
                {
                    "date": date,
                    "cash": cash,
                    "equity": equity_close,
                    "open_positions": len(positions),
                    "pending_entries": len(pending_entries),
                    "open_risk": open_risk_total,
                    "portfolio_risk_cap": equity_close * cfg.portfolio_risk_cap,
                    "accepted_signal_risk_today": accepted_today_risk_total,
                    "total_notional": total_notional,
                    "leverage": leverage,
                    # Soft-cap diagnostics — logged only, no gate effect.
                    "unrealized_exposure": unrealized_exposure_total,
                    "unrealized_exposure_soft_cap_triggered": soft_cap_triggered,
                }
            )

        trades_df = self._empty_trades_frame()
        if closed_trades:
            trades_df = pd.DataFrame(closed_trades).sort_values(
                ["exit_date", "symbol", "entry_date"]
            ).reset_index(drop=True)

        # Daily status uses the first strategy's prepared data (backward compat).
        # In incremental mode, filter out warmup-only dates so the frame matches
        # the dates we actually processed. warmup_until_ts 同理：这些日期只做
        # indicator 累积，不参与 daily_status 输出。
        daily_status_source = prepared
        if state_last_date is not None:
            daily_status_source = daily_status_source[daily_status_source[cfg.date_col] > state_last_date]
        if warmup_until_ts is not None:
            daily_status_source = daily_status_source[daily_status_source[cfg.date_col] > warmup_until_ts]
        daily_status = daily_status_source.copy()
        risk_reject_df = pd.DataFrame(
            [
                {
                    cfg.date_col: key[0],
                    cfg.symbol_col: key[1],
                    "risk_reject_reason": value,
                }
                for key, value in risk_reject.items()
                if key[2] == first_strategy_id  # only first strategy for daily_status
            ]
        )
        if risk_reject_df.empty:
            daily_status["risk_reject_reason"] = None
        else:
            daily_status = daily_status.merge(
                risk_reject_df,
                how="left",
                on=[cfg.date_col, cfg.symbol_col],
            )

        daily_status = daily_status[self._daily_status_columns(daily_status)].sort_values(
            [cfg.date_col, cfg.symbol_col]
        ).reset_index(drop=True)

        open_positions_df = self._serialize_open_positions(positions)
        # 2026-04-20 修复：增量续跑"0 新 bar"场景（session_date > bars.max_date，没处理任何新日期）
        # portfolio_daily 为空 list，直接 sort_values("date") 会 KeyError。
        # 下游 L874 的 `if not empty` 分支已经兜住空 frame，这里只需避开 sort。
        if portfolio_daily:
            portfolio_daily_df = pd.DataFrame(portfolio_daily).sort_values("date").reset_index(drop=True)
        else:
            portfolio_daily_df = pd.DataFrame()
        cancelled_entries_df = self._empty_cancelled_entries_frame()
        if cancelled_entries:
            cancelled_entries_df = pd.DataFrame(cancelled_entries).sort_values(
                ["cancel_date", "symbol", "entry_date"]
            ).reset_index(drop=True)

        pending_entries_df = self._build_pending_entries_df(per_day_new_pending)

        # Determine the terminal processed date for state.last_date. Prefer
        # the last date in portfolio_daily (every trading-day processed appends
        # one row). 2026-04-20 新增：warmup_until 模式下，warmup 日期不会追加
        # portfolio_daily，但这些日期已经被处理过（indicator / last_close 都
        # 更新了），last_date 应落在 warmup_until_ts 或 bars 里最后一个 warmup
        # 日期（取较小者；避免"将来日期"越界）。否则回退到 state_last_date，
        # 再否则为 None（纯空 slice）。
        if not portfolio_daily_df.empty:
            terminal_date = pd.Timestamp(portfolio_daily_df["date"].iloc[-1])
        elif warmup_until_ts is not None and dates:
            # dates 是 state_last_date 之后全部 prepared 日期；warmup-only
            # 模式下全部 ≤ warmup_until_ts。取 min(max(dates), warmup_until_ts)
            # 作为 terminal：既不会超出 bars 的实际覆盖，又能吃到 warmup 边界。
            max_processed = pd.Timestamp(max(dates))
            terminal_date = min(max_processed, warmup_until_ts)
        elif state_last_date is not None:
            terminal_date = state_last_date
        else:
            terminal_date = None

        engine_state = self._build_engine_state(
            cash=cash,
            positions=positions,
            pending_entries=pending_entries,
            last_close_by_symbol=last_close_by_symbol,
            last_raw_close_by_symbol=last_raw_close_by_symbol,
            terminal_date=terminal_date,
            original_bars=bars,
            merged_bars=bars_for_prepare,
        )

        return BacktestResult(
            trades=trades_df,
            daily_status=daily_status,
            portfolio_daily=portfolio_daily_df,
            open_positions=open_positions_df,
            prepared_data=prepared,
            cancelled_entries=cancelled_entries_df,
            data_quality_report=data_quality_report,
            pending_entries=pending_entries_df,
            engine_state=engine_state,
        )

    def _validate_input_columns(self, bars: pd.DataFrame) -> None:
        cfg = self.config
        required = {
            cfg.date_col,
            cfg.symbol_col,
            cfg.open_col,
            cfg.high_col,
            cfg.low_col,
            cfg.close_col,
            cfg.volume_col,
            cfg.open_interest_col,
            cfg.multiplier_col,
            cfg.commission_col,
            cfg.slippage_col,
            cfg.group_col,
        }
        missing = required - set(bars.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if cfg.enable_dual_stream:
            dual_required = {
                cfg.raw_open_col, cfg.raw_high_col, cfg.raw_low_col,
                cfg.raw_close_col, cfg.contract_col,
            }
            dual_missing = dual_required - set(bars.columns)
            if dual_missing:
                raise ValueError(
                    f"enable_dual_stream requires additional columns; "
                    f"missing: {sorted(dual_missing)}. "
                    f"Run scripts/build_enhanced_bars.py to produce them."
                )

    def _validate_input_values(self, df: pd.DataFrame) -> None:
        cfg = self.config
        numeric_cols = [
            cfg.open_col,
            cfg.high_col,
            cfg.low_col,
            cfg.close_col,
            cfg.volume_col,
            cfg.open_interest_col,
            cfg.multiplier_col,
            cfg.commission_col,
            cfg.slippage_col,
        ]
        numeric_values = df[numeric_cols].to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            raise ValueError("Numeric input contains NaN or infinite values.")

        if cfg.enable_dual_stream:
            for c in [cfg.raw_open_col, cfg.raw_high_col, cfg.raw_low_col, cfg.raw_close_col]:
                raw_values = df[c].astype(float).to_numpy()
                if not np.isfinite(raw_values).all():
                    n = int((~np.isfinite(raw_values)).sum())
                    raise ValueError(f"column '{c}' has {n} NaN/inf values under dual_stream")
            if df[cfg.contract_col].isna().any():
                n = int(df[cfg.contract_col].isna().sum())
                raise ValueError(f"column '{cfg.contract_col}' has {n} NaN values under dual_stream")

        high = df[cfg.high_col].astype(float)
        low = df[cfg.low_col].astype(float)
        open_ = df[cfg.open_col].astype(float)
        close = df[cfg.close_col].astype(float)
        volume = df[cfg.volume_col].astype(float)
        open_interest = df[cfg.open_interest_col].astype(float)
        multiplier = df[cfg.multiplier_col].astype(float)
        commission = df[cfg.commission_col].astype(float)
        slippage = df[cfg.slippage_col].astype(float)

        if (high < low).any():
            raise ValueError("Invalid OHLC: high < low.")
        if (high < pd.concat([open_, close], axis=1).max(axis=1)).any():
            raise ValueError("Invalid OHLC: high < max(open, close).")
        if (low > pd.concat([open_, close], axis=1).min(axis=1)).any():
            raise ValueError("Invalid OHLC: low > min(open, close).")
        if (multiplier <= 0).any():
            raise ValueError("contract_multiplier must be > 0.")
        if (commission < 0).any():
            raise ValueError("commission must be >= 0.")
        if (slippage < 0).any():
            raise ValueError("slippage must be >= 0.")
        if (volume < 0).any():
            raise ValueError("volume must be >= 0.")
        if (open_interest < 0).any():
            raise ValueError("open_interest must be >= 0.")

    def _prepare_symbol_base(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute engine-universal columns (ATR, atr_ref, ADX, next_trade_date)."""
        cfg = self.config
        high = df[cfg.high_col].astype(float)
        low = df[cfg.low_col].astype(float)
        close = df[cfg.close_col].astype(float)

        out = df.copy()
        out["atr"] = _wilder_atr(high=high, low=low, close=close, period=cfg.atr_period)
        out["atr_ref"] = out["atr"].shift(1)
        out["adx"] = _adx(high=high, low=low, close=close, period=cfg.adx_period)
        # v5: Choppiness Index for the congestion filter. Cheap to compute
        # unconditionally; the gate itself is disabled by default.
        out["cpi"] = _choppiness_index(high=high, low=low, close=close, period=cfg.cpi_period)
        out["next_trade_date"] = out[cfg.date_col].shift(-1)
        # 2026-04-22 bugfix：最后一行 shift(-1)=NaN，若 cfg.trading_calendar 提供，
        # 用 calendar.next_trading_day() 填，让 live 模式下最后一 bar 触发的信号
        # 能拿到合法 next_trade_date，不再被 NO_NEXT_TRADE_DATE 错误拒绝。
        if cfg.trading_calendar is not None and len(out) > 0:
            last_idx = out.index[-1]
            if pd.isna(out.at[last_idx, "next_trade_date"]):
                last_d = out.at[last_idx, cfg.date_col]
                if pd.notna(last_d):
                    try:
                        nxt = cfg.trading_calendar.next_trading_day(pd.Timestamp(last_d).date())
                        out.at[last_idx, "next_trade_date"] = pd.Timestamp(nxt)
                    except Exception:  # noqa: BLE001
                        pass  # calendar 查询失败就保留 NaN，fallback 到原逻辑
        # Per-symbol bar index (0-based), used by the warmup gate (1.7).
        out["_bar_index"] = np.arange(len(out), dtype=int)
        return out

    def _prepare_symbol_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backward-compat: base + first strategy's signals."""
        base = self._prepare_symbol_base(df)
        return self._strategies[0].entry_strategy.prepare_signals(base)

    def _prepare_all_strategies(self, bars: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare data for every strategy slot.

        Returns a dict keyed by strategy_id. Each value is a fully-prepared
        DataFrame (base columns + that strategy's signal columns).
        """
        cfg = self.config
        df = self._normalize_and_validate_bars(bars)

        # Build base frames per symbol (ATR, atr_ref, next_trade_date)
        base_frames: List[pd.DataFrame] = []
        for _, symbol_df in df.groupby(cfg.symbol_col, sort=False):
            base_frames.append(self._prepare_symbol_base(symbol_df.reset_index(drop=True)))

        if not base_frames:
            # Empty input
            result: Dict[str, pd.DataFrame] = {}
            for slot in self._strategies:
                empty = df.copy()
                for col in self._prepared_extra_columns():
                    empty[col] = pd.Series(dtype="float64")
                result[slot.strategy_id] = empty
            return result

        base_all = pd.concat(base_frames, axis=0, ignore_index=True)
        base_all = base_all.sort_values([cfg.date_col, cfg.symbol_col]).reset_index(drop=True)

        # For each strategy slot, apply its entry_strategy.prepare_signals
        # on per-symbol copies of the base frame
        result = {}
        for slot in self._strategies:
            slot_frames: List[pd.DataFrame] = []
            for _, symbol_df in base_all.groupby(cfg.symbol_col, sort=False):
                slot_frames.append(
                    slot.entry_strategy.prepare_signals(symbol_df.reset_index(drop=True))
                )
            slot_prepared = pd.concat(slot_frames, axis=0, ignore_index=True)
            slot_prepared = slot_prepared.sort_values([cfg.date_col, cfg.symbol_col]).reset_index(drop=True)
            result[slot.strategy_id] = slot_prepared

        return result

    @staticmethod
    def _months_to_delivery(order_book_id: Optional[str], today: pd.Timestamp) -> Optional[int]:
        """Months between `today` and the delivery month encoded in the last
        four chars of the contract code (YYMM). Returns None if the code is
        missing or doesn't end with 4 digits.

        Examples: RB2410 with today=2024-06-15 → 4; AG2412 with 2024-10-15 → 2.
        """
        if order_book_id is None or not isinstance(order_book_id, str):
            return None
        suffix = order_book_id[-4:]
        if not suffix.isdigit():
            return None
        yy = int(suffix[:2])
        mm = int(suffix[2:])
        if mm < 1 or mm > 12:
            return None
        delivery_year = 2000 + yy
        today_ts = pd.Timestamp(today)
        return (delivery_year - today_ts.year) * 12 + (mm - today_ts.month)

    def _effective_margin_rate(
        self,
        base_rate: float,
        order_book_id: Optional[str],
        today: pd.Timestamp,
    ) -> float:
        """base + broker_addon + tier_addon (4.1).

        `margin_tier_schedule` maps **exact months-to-delivery** → addon.
        E.g. `{0: 0.10, 1: 0.05, 2: 0.02}` means delivery-month gets +10%,
        one-month-out +5%, two-months-out +2%, three+-months-out 0.
        """
        cfg = self.config
        effective = base_rate + cfg.broker_margin_addon
        if cfg.margin_tier_schedule:
            m = self._months_to_delivery(order_book_id, today)
            if m is not None and m >= 0:
                effective += cfg.margin_tier_schedule.get(m, 0.0)
        return effective

    def _panama_to_raw_offset(self, row: pd.Series) -> float:
        """(Panama - raw) for this row/day. Returns 0.0 when dual_stream is off.

        Used to translate Panama-space prices (stops, etc.) to raw-space for
        fills. Computed from `close_col - raw_close_col` since both OHLC Panama
        and OHLC raw differ by the same daily offset.
        """
        cfg = self.config
        if not cfg.enable_dual_stream:
            return 0.0
        return float(row[cfg.close_col]) - float(row[cfg.raw_close_col])

    def _estimate_entry_fill_from_row(self, row: pd.Series, direction: int = 1) -> float:
        """Panama-space entry fill. Unchanged semantics across modes — the
        display/exit-strategy/MFE-MAE surface continues to work against this.
        The raw-space entry (for dual_stream P&L) is tracked separately on
        Position.raw_entry_fill.
        """
        cfg = self.config
        open_price = float(row[cfg.open_col])
        slippage = float(row[cfg.slippage_col])
        if direction == 1:
            return open_price + slippage
        else:
            return open_price - slippage

    def _estimate_raw_entry_fill_from_row(self, row: pd.Series, direction: int = 1) -> float:
        """Raw-space entry fill — only meaningful under dual_stream."""
        cfg = self.config
        open_raw = float(row[cfg.raw_open_col])
        slippage = float(row[cfg.slippage_col])
        return open_raw + slippage if direction == 1 else open_raw - slippage

    def _fill_pending_entry(
        self,
        pending: PendingEntry,
        row: pd.Series,
        entry_fill: Optional[float] = None,
        qty_override: Optional[int] = None,
        original_qty: Optional[int] = None,
        qty_shrink_reason: Optional[str] = None,
    ) -> Tuple[Position, float]:
        cfg = self.config
        effective_qty = qty_override if qty_override is not None else pending.qty
        actual_entry_fill = self._estimate_entry_fill_from_row(row, direction=pending.direction) if entry_fill is None else float(entry_fill)
        entry_slippage = float(row[cfg.slippage_col])
        contract_multiplier = float(row[cfg.multiplier_col])
        entry_commission = float(row[cfg.commission_col])
        cash_delta = -entry_commission * effective_qty

        actual_initial_risk = abs(actual_entry_fill - pending.initial_stop)
        actual_order_risk = actual_initial_risk * contract_multiplier * effective_qty
        risk_blowout_vs_estimate = actual_order_risk - pending.estimated_order_risk
        risk_blowout_ratio = (
            actual_order_risk / pending.estimated_order_risk
            if pending.estimated_order_risk > cfg.eps
            else np.nan
        )

        initial_contract: Optional[str] = None
        raw_entry_fill_init: Optional[float] = None
        segment_entry_init: Optional[float] = None
        if cfg.enable_dual_stream:
            initial_contract = str(row[cfg.contract_col])
            raw_entry_fill_init = self._estimate_raw_entry_fill_from_row(
                row, direction=pending.direction
            )
            segment_entry_init = raw_entry_fill_init

        position = Position(
            symbol=pending.symbol,
            strategy_id=pending.strategy_id,
            group_name=pending.group_name,
            direction=pending.direction,
            signal_date=pending.signal_date,
            entry_date=pending.entry_date,
            entry_estimate=pending.entry_estimate,
            entry_fill=actual_entry_fill,
            entry_slippage=entry_slippage,
            qty=effective_qty,
            contract_multiplier=contract_multiplier,
            entry_commission_per_contract=entry_commission,
            atr_ref=pending.atr_ref,
            volume=pending.volume,
            open_interest=pending.open_interest,
            initial_stop=pending.initial_stop,
            active_stop=pending.initial_stop,
            estimated_initial_risk=pending.estimated_initial_risk,
            estimated_order_risk=pending.estimated_order_risk,
            actual_initial_risk=actual_initial_risk,
            actual_order_risk=actual_order_risk,
            risk_blowout_vs_estimate=risk_blowout_vs_estimate,
            risk_blowout_ratio=risk_blowout_ratio,
            r_price=actual_initial_risk,
            r_money=actual_order_risk,
            highest_high_since_entry=actual_entry_fill,
            lowest_low_since_entry=actual_entry_fill,
            original_qty=original_qty,
            qty_shrink_reason=qty_shrink_reason,
            metadata=pending.metadata.copy(),
            current_contract=initial_contract,
            raw_entry_fill=raw_entry_fill_init,
            segment_entry_fill=segment_entry_init,
            active_stop_series=[
                {
                    "computed_on": pending.signal_date.strftime("%Y-%m-%d"),
                    "effective_from": pending.entry_date.strftime("%Y-%m-%d"),
                    "phase": "signal_init",
                    "active_stop_before": None,
                    "active_stop_after": pending.initial_stop,
                    "trailing_stop_candidate": None,
                    "atr_used": pending.atr_ref,
                    "highest_high_since_entry": None,
                }
            ],
        )
        return position, cash_delta

    def _build_cancelled_entry(
        self,
        pending: PendingEntry,
        row: pd.Series,
        attempted_entry_fill: float,
        cancel_reason: str,
    ) -> Dict[str, Any]:
        cfg = self.config
        contract_multiplier = float(row[cfg.multiplier_col])
        attempted_initial_risk = abs(attempted_entry_fill - pending.initial_stop)
        attempted_order_risk = attempted_initial_risk * contract_multiplier * pending.qty
        risk_blowout_vs_estimate = attempted_order_risk - pending.estimated_order_risk
        risk_blowout_ratio = (
            attempted_order_risk / pending.estimated_order_risk
            if pending.estimated_order_risk > cfg.eps
            else np.nan
        )
        record: Dict[str, Any] = {
            "symbol": pending.symbol,
            "strategy_id": pending.strategy_id,
            "group_name": pending.group_name,
            "signal_date": pending.signal_date,
            "entry_date": pending.entry_date,
            "entry_fill_date": pending.entry_date,
            "cancel_date": pd.Timestamp(row[cfg.date_col]),
            "entry_estimate": pending.entry_estimate,
            "attempted_entry_fill": attempted_entry_fill,
            "entry_slippage": float(row[cfg.slippage_col]),
            "qty": pending.qty,
            "contract_multiplier": contract_multiplier,
            "atr_ref": pending.atr_ref,
            "volume": pending.volume,
            "open_interest": pending.open_interest,
            "initial_stop": pending.initial_stop,
            "estimated_initial_risk": pending.estimated_initial_risk,
            "estimated_order_risk": pending.estimated_order_risk,
            "attempted_actual_initial_risk": attempted_initial_risk,
            "attempted_actual_order_risk": attempted_order_risk,
            "risk_blowout_vs_estimate": risk_blowout_vs_estimate,
            "risk_blowout_ratio": risk_blowout_ratio,
            "cancel_reason": cancel_reason,
        }
        record.update(pending.metadata)
        return record

    def _cannot_fill_side(self, row: pd.Series, side: int) -> bool:
        """Return True when an open-based fill on this bar is blocked by an
        opposing limit (1.8 + 1.10). Criterion is that the OPEN sits at the
        adverse limit — gap-open-to-limit catches both fully-locked days and
        gap-up/down-then-traded-back days, since in both cases a new order at
        the open cannot execute (no counterparty).

        Limits from RQData (`limit_up`/`limit_down`) are in raw contract price
        space. Prefer `raw_open_col` when bars carry it (dual-stream / real
        data); fall back to `open_col` only for synthetic frames that don't
        have raw columns.

        side: +1 for BUY (blocked by open at/above limit_up),
              -1 for SELL (blocked by open at/below limit_down).
        No-op when limit columns are absent or NaN.
        """
        cfg = self.config
        lim_up = row.get(cfg.limit_up_col)
        lim_dn = row.get(cfg.limit_down_col)
        if pd.isna(lim_up) or pd.isna(lim_dn):
            return False
        raw_open = row.get(cfg.raw_open_col)
        if pd.notna(raw_open):
            op = float(raw_open)
        else:
            op = float(row[cfg.open_col])
        if side == 1:
            return op >= float(lim_up) - 1e-6
        return op <= float(lim_dn) + 1e-6

    def _process_open_and_intraday_for_existing_position(
        self,
        position: Position,
        row: pd.Series,
    ) -> List[Dict[str, Any]]:
        """Returns list of trade records. A single bar may produce multiple
        records when a profit target partial-close fires before the intraday
        stop on the remaining qty. Each record has `is_partial: bool` so the
        caller knows whether to delete the position."""
        cfg = self.config
        records: List[Dict[str, Any]] = []

        # Stop CHECKS stay in Panama space — the stop was computed from Panama ATR.
        open_price = float(row[cfg.open_col])
        high_price = float(row[cfg.high_col])
        low_price = float(row[cfg.low_col])
        slippage = float(row[cfg.slippage_col])
        exit_commission = float(row[cfg.commission_col])
        date = pd.Timestamp(row[cfg.date_col])

        d = position.direction

        # Limit-lock block (1.8): if bar is fully locked against the exit side,
        # skip all fills today; the position stays open and retries next bar.
        exit_side = -d
        if self._cannot_fill_side(row, exit_side):
            return records
        raw_offset = self._panama_to_raw_offset(row)
        raw_open = float(row[cfg.raw_open_col]) if cfg.enable_dual_stream else None

        # 1) Gap stop — full close, done.
        gap_stop = (d == 1 and open_price <= position.active_stop) or \
                   (d == -1 and open_price >= position.active_stop)
        if gap_stop:
            rec = self._close_position(
                position=position, exit_date=date,
                exit_fill=self._apply_exit_slippage(open_price, slippage, d),
                raw_exit_fill=(
                    self._apply_exit_slippage(raw_open, slippage, d)
                    if cfg.enable_dual_stream else None
                ),
                exit_reason="STOP_GAP", exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )
            rec["is_partial"] = False
            records.append(rec)
            return records

        # 2) Pending exit — full close, done.
        if position.pending_exit_reason is not None and position.pending_exit_date == date:
            rec = self._close_position(
                position=position, exit_date=date,
                exit_fill=self._apply_exit_slippage(open_price, slippage, d),
                raw_exit_fill=(
                    self._apply_exit_slippage(raw_open, slippage, d)
                    if cfg.enable_dual_stream else None
                ),
                exit_reason=position.pending_exit_reason, exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )
            rec["is_partial"] = False
            records.append(rec)
            return records

        # 3a) v7 Breakeven ratchet — at N×ATR unrealized R, lift the stop
        # to entry ± offset×ATR (but never loosen). Position stays open.
        if (
            cfg.breakeven_trigger_atr_r > 0.0
            and not position.breakeven_triggered
            and position.r_price > 0.0
            and position.qty > 0
        ):
            trigger_level = position.entry_fill + d * cfg.breakeven_trigger_atr_r * position.r_price
            be_hit = (d == 1 and high_price >= trigger_level) or (d == -1 and low_price <= trigger_level)
            if be_hit:
                new_stop = position.entry_fill + d * cfg.breakeven_stop_offset_atr * position.atr_ref
                old_stop = position.active_stop
                tighter = (d == 1 and new_stop > old_stop) or (d == -1 and new_stop < old_stop)
                if tighter:
                    position.active_stop = new_stop
                    position.active_stop_series.append({
                        "computed_on": date.strftime("%Y-%m-%d"),
                        "effective_from": date.strftime("%Y-%m-%d"),
                        "phase": "breakeven_ratchet",
                        "active_stop_before": old_stop,
                        "active_stop_after": new_stop,
                        "trigger_level": float(trigger_level),
                        "atr_used": position.atr_ref,
                        "highest_high_since_entry": position.highest_high_since_entry,
                    })
                position.breakeven_triggered = True

        # 3b) v6 Profit target — at N×ATR unrealized R, close a fraction.
        if (
            cfg.profit_target_atr_r > 0.0
            and not position.profit_target_triggered
            and position.r_price > 0.0
            and position.qty > 0
        ):
            # Target price level
            target_level = position.entry_fill + d * cfg.profit_target_atr_r * position.r_price
            # Long: triggered when high crosses ≥ target. Short: when low crosses ≤ target.
            hit = (d == 1 and high_price >= target_level) or (d == -1 and low_price <= target_level)
            if hit:
                # Fill price: gap beyond target favours us → use open when favourable.
                if d == 1:
                    fill_price = max(target_level, open_price)
                else:
                    fill_price = min(target_level, open_price)
                fill_slipped = self._apply_exit_slippage(fill_price, slippage, d)
                raw_fill: Optional[float] = None
                if cfg.enable_dual_stream:
                    raw_fill = self._apply_exit_slippage(fill_price - raw_offset, slippage, d)

                frac = max(0.0, min(1.0, cfg.profit_target_close_fraction))
                close_qty = int(round(position.qty * frac))
                close_qty = max(1, min(close_qty, position.qty))

                position.profit_target_triggered = True

                tag = f"PROFIT_TARGET_{cfg.profit_target_atr_r:.1f}R"
                if close_qty >= position.qty:
                    # Full close path
                    rec = self._close_position(
                        position=position, exit_date=date,
                        exit_fill=fill_slipped, raw_exit_fill=raw_fill,
                        exit_reason=f"{tag}_FULL",
                        exit_slippage=slippage,
                        exit_commission_per_contract=exit_commission,
                    )
                    rec["is_partial"] = False
                    records.append(rec)
                    return records
                else:
                    # Partial close path — mutate position.qty, continue to stop check.
                    rec = self._partial_close_position(
                        position=position, exit_date=date,
                        exit_fill=fill_slipped, raw_exit_fill=raw_fill,
                        close_qty=close_qty,
                        exit_reason=f"{tag}_PARTIAL",
                        exit_slippage=slippage,
                        exit_commission_per_contract=exit_commission,
                    )
                    rec["is_partial"] = True
                    records.append(rec)
                    # Fall through to check intraday stop on remaining qty.

        # 4) Intraday stop — full close on remaining.
        if position.qty > 0 and low_price <= position.active_stop <= high_price:
            position.mae_price = max(
                position.mae_price,
                self._adverse_excursion(position.active_stop, position.entry_fill, d),
            )
            raw_stop_exit: Optional[float] = None
            if cfg.enable_dual_stream:
                raw_stop_price = position.active_stop - raw_offset
                raw_stop_exit = self._apply_exit_slippage(raw_stop_price, slippage, d)
            rec = self._close_position(
                position=position, exit_date=date,
                exit_fill=self._apply_exit_slippage(position.active_stop, slippage, d),
                raw_exit_fill=raw_stop_exit,
                exit_reason="STOP_INTRADAY", exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )
            rec["is_partial"] = False
            records.append(rec)
            return records

        return records

    def _process_close_phase(self, position: Position, row: pd.Series) -> None:
        self._exit_strategy.process_close_phase(
            position=position, row=row, next_trade_date=row["next_trade_date"],
        )

    def _check_and_apply_roll(
        self,
        position: Position,
        row: pd.Series,
        prev_raw_close_by_symbol: Dict[str, float],
    ) -> None:
        """If the dominant contract changed on `row[date]`, close the prior
        segment and start a new one — applying the roll cost.

        Stops are NOT adjusted: they remain in Panama space, which by
        construction already aligns with the new contract (Panama's
        prev_close_spread makes cross-day Panama returns equal the held
        contract's own returns). Only the raw-space segment accounting
        moves, plus the explicit 2×(commission + slippage×mult×qty) cost.

        No-op when dual_stream is disabled or the contract is unchanged.
        """
        cfg = self.config
        if not cfg.enable_dual_stream:
            return
        today_contract = str(row[cfg.contract_col])
        if position.current_contract == today_contract:
            return

        prev_close_raw = prev_raw_close_by_symbol.get(position.symbol)
        if prev_close_raw is None:
            # Defensive: first time we see this symbol (should not happen
            # because the position was filled at least one day ago).
            position.current_contract = today_contract
            position.segment_entry_fill = float(row[cfg.raw_open_col])
            return

        # Realize prior segment at yesterday's raw close.
        seg_pnl = (
            (prev_close_raw - float(position.segment_entry_fill))
            * position.contract_multiplier
            * position.qty
            * position.direction
        )
        position.realized_segment_pnl += seg_pnl

        # Start new segment at today's raw open — no extra slippage here;
        # the 2×slippage for the round-trip is folded into roll_cost.
        new_seg_entry = float(row[cfg.raw_open_col])
        slip = float(row[cfg.slippage_col])
        comm = float(row[cfg.commission_col])
        roll_cost = 2.0 * (
            comm * position.qty
            + slip * position.contract_multiplier * position.qty
        )
        position.roll_cost_accrued += roll_cost

        position.rolls_crossed.append(
            {
                "date": pd.Timestamp(row[cfg.date_col]).strftime("%Y-%m-%d"),
                "old_contract": position.current_contract,
                "new_contract": today_contract,
                "old_close_raw": prev_close_raw,
                "new_open_raw": new_seg_entry,
                "realized_segment_pnl": seg_pnl,
                "roll_cost": roll_cost,
            }
        )
        position.current_contract = today_contract
        position.segment_entry_fill = new_seg_entry

    def _resolve_slot_sar(self, strategy_id: str) -> Tuple[bool, float, int]:
        """Per-slot SAR resolution: slot-level override wins over EngineConfig.
        Returns (reverse_on_stop, reverse_stop_atr_mult, reverse_chain_max).
        Fallback to EngineConfig values if slot not found or slot override is None.
        """
        cfg = self.config
        slot = self._strategy_map.get(strategy_id)
        if slot is None:
            return (
                cfg.reverse_on_stop,
                cfg.reverse_stop_atr_mult,
                cfg.reverse_chain_max,
            )
        on = cfg.reverse_on_stop if slot.reverse_on_stop is None else slot.reverse_on_stop
        mult = (
            cfg.reverse_stop_atr_mult
            if slot.reverse_stop_atr_mult is None
            else slot.reverse_stop_atr_mult
        )
        chain = (
            cfg.reverse_chain_max
            if slot.reverse_chain_max is None
            else slot.reverse_chain_max
        )
        return bool(on), float(mult), int(chain)

    def _try_synthesize_reverse_entry(
        self,
        closed_position: Position,
        row: pd.Series,
        equity_estimate: float,
        positions: Dict[Tuple[str, str], Position],
        pending_entries: Dict[Tuple[str, str], PendingEntry],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
    ) -> Optional[PendingEntry]:
        """Stop-and-reverse (v10): after a stop-out close, synthesize a
        PendingEntry in the opposite direction sized from ATR × multiplier.

        Returns the PendingEntry if all gates pass (non-NaN ATR, valid
        next_trade_date, chain not maxed, risk caps respected, qty>=1),
        else None. Caller inserts into pending_entries dict.

        Skips trend_score scaling intentionally — SAR is a reactionary
        re-entry, not a fresh trend-confirmed signal. Uses base
        risk_per_trade × current equity for sizing.
        """
        cfg = self.config
        # Per-slot SAR params (slot override > EngineConfig default)
        _slot_on, slot_atr_mult, slot_chain_max = self._resolve_slot_sar(
            closed_position.strategy_id
        )
        prev_leg_count = int(closed_position.metadata.get("reverse_leg_count", 0))
        if prev_leg_count >= slot_chain_max:
            return None

        raw_ntd = row.get("next_trade_date")
        if raw_ntd is None or pd.isna(raw_ntd):
            return None
        next_trade_date = pd.Timestamp(raw_ntd)

        atr_ref_val = (
            float(row["atr_ref"]) if pd.notna(row.get("atr_ref")) else float("nan")
        )
        if not np.isfinite(atr_ref_val) or atr_ref_val <= 0.0:
            return None

        new_direction = -closed_position.direction
        entry_estimate = float(row[cfg.close_col])
        stop_offset = atr_ref_val * slot_atr_mult
        if stop_offset <= cfg.eps:
            return None
        initial_stop = (
            entry_estimate - stop_offset
            if new_direction == 1
            else entry_estimate + stop_offset
        )

        contract_multiplier = float(row[cfg.multiplier_col])
        per_contract_risk_est = stop_offset * contract_multiplier
        if per_contract_risk_est <= cfg.eps:
            return None

        # Use base risk_per_trade — no ADX trend scaling for reversal trades.
        risk_budget_single = equity_estimate * cfg.risk_per_trade
        qty = math.floor(risk_budget_single / per_contract_risk_est)
        if qty < 1:
            return None

        order_risk = per_contract_risk_est * qty
        group_name = closed_position.group_name

        # Block duplicate slot-key (shouldn't happen right after close, but defensive)
        key = (closed_position.symbol, closed_position.strategy_id)
        if key in positions or key in pending_entries:
            return None

        # Risk cap gates (portfolio + group). Mirror phase-6 signal-gen
        # logic but reuse equity_estimate (yesterday's equity_close or
        # initial_capital pre-day-1).
        effective_open_risk, effective_open_risk_by_group = (
            self._compute_effective_open_risk_for_entry_date(
                candidate_entry_date=next_trade_date,
                positions=positions,
                pending_entries=pending_entries,
                close_map=close_map,
                last_close_by_symbol=last_close_by_symbol,
            )
        )
        portfolio_cap = equity_estimate * cfg.portfolio_risk_cap
        if group_name.startswith("ind_"):
            group_cap = equity_estimate * cfg.default_group_risk_cap
        else:
            group_cap = equity_estimate * cfg.group_risk_cap.get(
                group_name, cfg.default_group_risk_cap
            )
        if effective_open_risk + order_risk > portfolio_cap + cfg.eps:
            return None
        if (
            cfg.use_group_risk_cap
            and effective_open_risk_by_group.get(group_name, 0.0) + order_risk
            > group_cap + cfg.eps
        ):
            return None

        metadata: Dict[str, Any] = {
            "entry_type": "SAR_REVERSE",
            "reverse_leg_count": prev_leg_count + 1,
        }
        return PendingEntry(
            symbol=closed_position.symbol,
            strategy_id=closed_position.strategy_id,
            group_name=group_name,
            direction=new_direction,
            signal_date=pd.Timestamp(row[cfg.date_col]),
            entry_date=next_trade_date,
            entry_estimate=entry_estimate,
            qty=qty,
            atr_ref=atr_ref_val,
            volume=float(row[cfg.volume_col]),
            open_interest=float(row[cfg.open_interest_col]),
            initial_stop=initial_stop,
            estimated_initial_risk=stop_offset,
            estimated_order_risk=order_risk,
            contract_multiplier_est=contract_multiplier,
            metadata=metadata,
        )

    def _close_position(
        self,
        position: Position,
        exit_date: pd.Timestamp,
        exit_fill: float,
        exit_reason: str,
        exit_slippage: float,
        exit_commission_per_contract: float,
        raw_exit_fill: Optional[float] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        if cfg.enable_dual_stream and position.segment_entry_fill is not None:
            if raw_exit_fill is None:
                raise ValueError(
                    "dual_stream _close_position missing raw_exit_fill"
                )
            final_segment_pnl = (
                self._directional_pnl(raw_exit_fill, position.segment_entry_fill, position.direction)
                * position.contract_multiplier
                * position.qty
            )
            gross_pnl = position.realized_segment_pnl + final_segment_pnl
        else:
            gross_pnl = (
                self._directional_pnl(exit_fill, position.entry_fill, position.direction)
                * position.contract_multiplier
                * position.qty
            )
        total_entry_commission = position.entry_commission_per_contract * position.qty
        total_exit_commission = exit_commission_per_contract * position.qty
        roll_cost_total = float(position.roll_cost_accrued)
        net_pnl = gross_pnl - total_entry_commission - total_exit_commission - roll_cost_total
        cash_delta = gross_pnl - total_exit_commission - roll_cost_total

        r_money_abs = max(abs(position.r_money), self.config.eps)
        r_multiple = net_pnl / r_money_abs
        mfe_r = position.mfe_price / max(abs(position.r_price), self.config.eps)
        mae_r = position.mae_price / max(abs(position.r_price), self.config.eps)

        record: Dict[str, Any] = {
            "symbol": position.symbol,
            "strategy_id": position.strategy_id,
            "group_name": position.group_name,
            "direction": position.direction,
            "signal_date": position.signal_date,
            "entry_date": position.entry_date,
            "entry_fill_date": position.entry_date,
            "entry_estimate": position.entry_estimate,
            "entry_fill": position.entry_fill,
            "qty": position.qty,
            "contract_multiplier": position.contract_multiplier,
            "atr_ref": position.atr_ref,
            "volume": position.volume,
            "open_interest": position.open_interest,
            "initial_stop": position.initial_stop,
            "active_stop_series": json.dumps(position.active_stop_series, ensure_ascii=False),
            "estimated_initial_risk": position.estimated_initial_risk,
            "estimated_order_risk": position.estimated_order_risk,
            "actual_initial_risk": position.actual_initial_risk,
            "actual_order_risk": position.actual_order_risk,
            "risk_blowout_vs_estimate": position.risk_blowout_vs_estimate,
            "risk_blowout_ratio": position.risk_blowout_ratio,
            "original_qty": position.original_qty,
            "qty_shrink_reason": position.qty_shrink_reason,
            "exit_date": exit_date,
            "exit_fill": exit_fill,
            "exit_reason": exit_reason,
            "r_price": position.r_price,
            "r_money": position.r_money,
            "r_multiple": r_multiple,
            "mfe": position.mfe_price,
            "mae": position.mae_price,
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "entry_slippage": position.entry_slippage,
            "exit_slippage": exit_slippage,
            "entry_commission_total": total_entry_commission,
            "exit_commission_total": total_exit_commission,
            "raw_entry_fill": position.raw_entry_fill,
            "raw_exit_fill": raw_exit_fill,
            "entry_contract": (
                position.rolls_crossed[0]["old_contract"]
                if position.rolls_crossed
                else position.current_contract
            ),
            "exit_contract": position.current_contract,
            "rolls_crossed": len(position.rolls_crossed),
            "roll_cost_total": roll_cost_total,
            "rolls_detail": json.dumps(position.rolls_crossed, ensure_ascii=False),
            "cash_delta": cash_delta,
        }
        record.update(position.metadata)
        return record

    def _partial_close_position(
        self,
        position: Position,
        exit_date: pd.Timestamp,
        exit_fill: float,
        exit_reason: str,
        exit_slippage: float,
        exit_commission_per_contract: float,
        close_qty: int,
        raw_exit_fill: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Close `close_qty` contracts (partial). Mutates position.qty down
        and prorates roll_cost. Commissions are charged only on the closed
        portion — entry commission for the remaining qty is paid at final
        close. Returns a trade record (includes cash_delta)."""
        cfg = self.config
        if close_qty <= 0 or close_qty >= position.qty:
            raise ValueError(
                f"_partial_close_position: close_qty {close_qty} must be in (0, {position.qty})"
            )
        full_qty = position.qty  # snapshot before mutation
        frac = close_qty / full_qty

        if cfg.enable_dual_stream and position.segment_entry_fill is not None:
            if raw_exit_fill is None:
                raise ValueError("dual_stream _partial_close_position missing raw_exit_fill")
            partial_segment_pnl = (
                self._directional_pnl(raw_exit_fill, position.segment_entry_fill, position.direction)
                * position.contract_multiplier
                * close_qty
            )
            realized_portion = position.realized_segment_pnl * frac
            gross_pnl = realized_portion + partial_segment_pnl
            position.realized_segment_pnl -= realized_portion
        else:
            gross_pnl = (
                self._directional_pnl(exit_fill, position.entry_fill, position.direction)
                * position.contract_multiplier
                * close_qty
            )

        entry_comm = position.entry_commission_per_contract * close_qty
        exit_comm = exit_commission_per_contract * close_qty
        roll_cost_portion = float(position.roll_cost_accrued) * frac
        position.roll_cost_accrued -= roll_cost_portion

        net_pnl = gross_pnl - entry_comm - exit_comm - roll_cost_portion
        cash_delta = gross_pnl - exit_comm - roll_cost_portion  # entry_comm paid at entry

        r_money_partial = position.r_money * frac
        r_money_abs = max(abs(r_money_partial), cfg.eps)
        r_multiple = net_pnl / r_money_abs
        mfe_r = position.mfe_price / max(abs(position.r_price), cfg.eps)
        mae_r = position.mae_price / max(abs(position.r_price), cfg.eps)

        record: Dict[str, Any] = {
            "symbol": position.symbol,
            "strategy_id": position.strategy_id,
            "group_name": position.group_name,
            "direction": position.direction,
            "signal_date": position.signal_date,
            "entry_date": position.entry_date,
            "entry_fill_date": position.entry_date,
            "entry_estimate": position.entry_estimate,
            "entry_fill": position.entry_fill,
            "qty": close_qty,
            "contract_multiplier": position.contract_multiplier,
            "atr_ref": position.atr_ref,
            "volume": position.volume,
            "open_interest": position.open_interest,
            "initial_stop": position.initial_stop,
            "active_stop_series": json.dumps(position.active_stop_series, ensure_ascii=False),
            "estimated_initial_risk": position.estimated_initial_risk * frac,
            "estimated_order_risk": position.estimated_order_risk * frac,
            "actual_initial_risk": position.actual_initial_risk * frac,
            "actual_order_risk": position.actual_order_risk * frac,
            "risk_blowout_vs_estimate": position.risk_blowout_vs_estimate,
            "risk_blowout_ratio": position.risk_blowout_ratio,
            "original_qty": position.original_qty,
            "qty_shrink_reason": position.qty_shrink_reason,
            "exit_date": exit_date,
            "exit_fill": exit_fill,
            "exit_reason": exit_reason,
            "r_price": position.r_price,
            "r_money": r_money_partial,
            "r_multiple": r_multiple,
            "mfe": position.mfe_price,
            "mae": position.mae_price,
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "entry_slippage": position.entry_slippage,
            "exit_slippage": exit_slippage,
            "entry_commission_total": entry_comm,
            "exit_commission_total": exit_comm,
            "raw_entry_fill": position.raw_entry_fill,
            "raw_exit_fill": raw_exit_fill,
            "entry_contract": (
                position.rolls_crossed[0]["old_contract"]
                if position.rolls_crossed else position.current_contract
            ),
            "exit_contract": position.current_contract,
            "rolls_crossed": len(position.rolls_crossed),
            "roll_cost_total": roll_cost_portion,
            "rolls_detail": json.dumps(position.rolls_crossed, ensure_ascii=False),
            "cash_delta": cash_delta,
        }
        record.update(position.metadata)

        # Mutate position in place — remaining qty continues to be managed.
        position.qty -= close_qty
        return record

    def _compute_equity_close(
        self,
        cash: float,
        positions: Dict[Tuple[str, str], Position],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
        raw_close_map: Optional[Dict[str, float]] = None,
        last_raw_close_by_symbol: Optional[Dict[str, float]] = None,
    ) -> float:
        cfg = self.config
        unrealized = 0.0
        for position in positions.values():
            symbol = position.symbol
            if cfg.enable_dual_stream and position.segment_entry_fill is not None:
                rmap = raw_close_map or {}
                last_rmap = last_raw_close_by_symbol or {}
                mark_price = rmap.get(symbol, last_rmap.get(symbol, position.segment_entry_fill))
                segment_unreal = (
                    self._directional_pnl(mark_price, position.segment_entry_fill, position.direction)
                    * position.contract_multiplier
                    * position.qty
                )
                unrealized += position.realized_segment_pnl + segment_unreal - position.roll_cost_accrued
            else:
                mark_price = close_map.get(symbol, last_close_by_symbol.get(symbol, position.entry_fill))
                unrealized += self._directional_pnl(mark_price, position.entry_fill, position.direction) * position.contract_multiplier * position.qty
        return cash + unrealized

    def _compute_open_risk(
        self,
        positions: Dict[Tuple[str, str], Position],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """Principal-risk aggregate (definition B, 2026-04-19).

        For each open position, computes the loss relative to entry if the
        stop were hit now:

            long  : max(entry_fill − active_stop, 0) × mult × qty
            short : max(active_stop − entry_fill, 0) × mult × qty

        Behaviour vs the legacy definition C:
          - Fresh position (stop below entry for long): risk ≈ initial risk.
          - Stop ratcheted past entry (trail into profit): risk = 0 →
            releases portfolio/group cap so winners don't keep hogging it.
          - Degenerate slippage case where entry ≤ stop (long): clamped to 0.

        `close_map` / `last_close_by_symbol` are kept in the signature for
        backward compatibility with call sites; principal risk does not
        actually need the mark price.
        """
        del close_map  # principal risk is mark-price independent
        del last_close_by_symbol
        open_risk_total = 0.0
        by_group: Dict[str, float] = {}
        for position in positions.values():
            principal_risk = max(
                self._directional_pnl(
                    position.entry_fill, position.active_stop, position.direction,
                ),
                0.0,
            ) * position.contract_multiplier * position.qty
            open_risk_total += principal_risk
            by_group[position.group_name] = (
                by_group.get(position.group_name, 0.0) + principal_risk
            )
        return open_risk_total, by_group

    def _compute_effective_open_risk_for_entry_date(
        self,
        candidate_entry_date: pd.Timestamp,
        positions: Dict[Tuple[str, str], Position],
        pending_entries: Dict[Tuple[str, str], PendingEntry],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """Principal-risk projection at a candidate entry date (definition B).

        Mirrors `_compute_open_risk` for live positions and adds any pending
        entries whose entry_date ≤ candidate_entry_date (`estimated_order_risk`
        is already the principal risk at fill).
        """
        del close_map
        del last_close_by_symbol
        total = 0.0
        by_group: Dict[str, float] = {}

        for position in positions.values():
            if position.pending_exit_date is not None and position.pending_exit_date <= candidate_entry_date:
                continue
            principal_risk = max(
                self._directional_pnl(
                    position.entry_fill, position.active_stop, position.direction,
                ),
                0.0,
            ) * position.contract_multiplier * position.qty
            total += principal_risk
            by_group[position.group_name] = (
                by_group.get(position.group_name, 0.0) + principal_risk
            )

        for pending in pending_entries.values():
            if pending.entry_date <= candidate_entry_date:
                total += pending.estimated_order_risk
                by_group[pending.group_name] = (
                    by_group.get(pending.group_name, 0.0) + pending.estimated_order_risk
                )

        return total, by_group

    def _compute_unrealized_exposure(
        self,
        positions: Dict[Tuple[str, str], Position],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """Floating-P&L exposure between current mark and active stop (legacy C).

        This is the old "risk" definition that the engine used before the
        2026-04-19 refactor. It is NOT used as a hard gate anymore; it only
        drives the `unrealized_exposure_soft_cap` warning.

            long  : max(current − active_stop, 0) × mult × qty
            short : max(active_stop − current, 0) × mult × qty

        Intuition: "how much floating profit is the stop currently
        protecting". It grows when the stop trails into profit, so a
        trending winner can look large here even though its principal risk
        is zero.
        """
        total = 0.0
        by_group: Dict[str, float] = {}
        for position in positions.values():
            symbol = position.symbol
            current_price = close_map.get(
                symbol, last_close_by_symbol.get(symbol, position.entry_fill),
            )
            exposure = max(
                self._directional_pnl(
                    current_price, position.active_stop, position.direction,
                ),
                0.0,
            ) * position.contract_multiplier * position.qty
            total += exposure
            by_group[position.group_name] = (
                by_group.get(position.group_name, 0.0) + exposure
            )
        return total, by_group

    def _compute_total_notional(
        self,
        positions: Dict[Tuple[str, str], Position],
        pending_entries: Dict[Tuple[str, str], PendingEntry],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
    ) -> float:
        total = 0.0
        for position in positions.values():
            symbol = position.symbol
            mark_price = close_map.get(symbol, last_close_by_symbol.get(symbol, position.entry_fill))
            total += mark_price * position.contract_multiplier * position.qty
        for pending in pending_entries.values():
            total += pending.entry_estimate * pending.contract_multiplier_est * pending.qty
        return total

    def _serialize_open_positions(self, positions: Dict[Tuple[str, str], Position]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for position in positions.values():
            row_dict: Dict[str, Any] = {
                "symbol": position.symbol,
                "strategy_id": position.strategy_id,
                "group_name": position.group_name,
                "direction": position.direction,
                "signal_date": position.signal_date,
                "entry_date": position.entry_date,
                "entry_fill_date": position.entry_date,
                "entry_estimate": position.entry_estimate,
                "entry_fill": position.entry_fill,
                "qty": position.qty,
                "contract_multiplier": position.contract_multiplier,
                "atr_ref": position.atr_ref,
                "volume": position.volume,
                "open_interest": position.open_interest,
                "initial_stop": position.initial_stop,
                "active_stop": position.active_stop,
                "active_stop_series": json.dumps(position.active_stop_series, ensure_ascii=False),
                "completed_bars": position.completed_bars,
                "pending_exit_reason": position.pending_exit_reason,
                "pending_exit_date": position.pending_exit_date,
                "estimated_initial_risk": position.estimated_initial_risk,
                "estimated_order_risk": position.estimated_order_risk,
                "actual_initial_risk": position.actual_initial_risk,
                "actual_order_risk": position.actual_order_risk,
                "risk_blowout_vs_estimate": position.risk_blowout_vs_estimate,
                "risk_blowout_ratio": position.risk_blowout_ratio,
                "r_price": position.r_price,
                "r_money": position.r_money,
                "mfe": position.mfe_price,
                "mae": position.mae_price,
                "original_qty": position.original_qty,
                "qty_shrink_reason": position.qty_shrink_reason,
                "entry_slippage": position.entry_slippage,
                "entry_commission_per_contract": position.entry_commission_per_contract,
            }
            row_dict.update(position.metadata)
            rows.append(row_dict)
        if not rows:
            return self._empty_open_positions_frame()
        return pd.DataFrame(rows).sort_values(["symbol", "entry_date"]).reset_index(drop=True)

    # Engine-universal columns always present in prepared data
    _ENGINE_BASE_COLUMNS = ["atr", "atr_ref", "next_trade_date"]
    _ENGINE_SIGNAL_COLUMNS = ["entry_trigger_pass", "entry_direction"]

    def _prepared_extra_columns(self) -> List[str]:
        """Minimum extra columns guaranteed by the engine (for empty frame)."""
        return self._ENGINE_BASE_COLUMNS + self._ENGINE_SIGNAL_COLUMNS

    def _daily_status_columns(self, prepared: pd.DataFrame) -> List[str]:
        """Build daily_status column list dynamically from actual prepared data."""
        cfg = self.config
        # Universal engine columns first
        base = [
            cfg.date_col, cfg.symbol_col, cfg.group_col,
            cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col,
            cfg.volume_col, cfg.open_interest_col,
            "atr", "atr_ref",
        ]
        # Entry-strategy-specific columns (everything the strategy added beyond engine base)
        skip = set(base) | {"next_trade_date", "risk_reject_reason"}
        skip |= {cfg.multiplier_col, cfg.commission_col, cfg.slippage_col, cfg.margin_rate_col}
        extra = [c for c in prepared.columns if c not in skip]
        # Always end with required signal columns
        tail = ["entry_trigger_pass", "entry_direction", "risk_reject_reason"]
        # Deduplicate while preserving order
        seen = set()
        result = []
        for c in base + extra + tail:
            if c not in seen and c in prepared.columns:
                seen.add(c)
                result.append(c)
        return result

    def _empty_trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "symbol",
                "strategy_id",
                "group_name",
                "direction",
                "signal_date",
                "entry_date",
                "entry_fill_date",
                "entry_estimate",
                "entry_fill",
                "qty",
                "contract_multiplier",
                "atr_ref",
                "volume",
                "open_interest",
                "initial_stop",
                "active_stop_series",
                "estimated_initial_risk",
                "estimated_order_risk",
                "actual_initial_risk",
                "actual_order_risk",
                "risk_blowout_vs_estimate",
                "risk_blowout_ratio",
                "original_qty",
                "qty_shrink_reason",
                "exit_date",
                "exit_fill",
                "exit_reason",
                "r_price",
                "r_money",
                "r_multiple",
                "mfe",
                "mae",
                "mfe_r",
                "mae_r",
                "gross_pnl",
                "net_pnl",
                "entry_slippage",
                "exit_slippage",
                "entry_commission_total",
                "exit_commission_total",
            ]
        )

    def _empty_open_positions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "symbol",
                "strategy_id",
                "group_name",
                "direction",
                "signal_date",
                "entry_date",
                "entry_fill_date",
                "entry_estimate",
                "entry_fill",
                "qty",
                "contract_multiplier",
                "atr_ref",
                "volume",
                "open_interest",
                "initial_stop",
                "active_stop",
                "active_stop_series",
                "completed_bars",
                "pending_exit_reason",
                "pending_exit_date",
                "estimated_initial_risk",
                "estimated_order_risk",
                "actual_initial_risk",
                "actual_order_risk",
                "risk_blowout_vs_estimate",
                "risk_blowout_ratio",
                "r_price",
                "r_money",
                "mfe",
                "mae",
                "original_qty",
                "qty_shrink_reason",
                "entry_slippage",
                "entry_commission_per_contract",
            ]
        )

    def _empty_cancelled_entries_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "symbol",
                "strategy_id",
                "group_name",
                "signal_date",
                "entry_date",
                "entry_fill_date",
                "cancel_date",
                "entry_estimate",
                "attempted_entry_fill",
                "entry_slippage",
                "qty",
                "contract_multiplier",
                "atr_ref",
                "volume",
                "open_interest",
                "initial_stop",
                "estimated_initial_risk",
                "estimated_order_risk",
                "attempted_actual_initial_risk",
                "attempted_actual_order_risk",
                "risk_blowout_vs_estimate",
                "risk_blowout_ratio",
                "cancel_reason",
            ]
        )

    # ── Incremental run support (P1a) ────────────────────────────────

    # Minimum number of history bars per symbol carried over as warmup
    # between incremental runs. Must exceed the largest rolling window
    # used by any entry/exit strategy (HL 21, double_ma 34, boll 22,
    # HAB box 7 + bb 20 + percentile lookback 60, AMA slow 30, ATR/ADX
    # up to 10× their 20 period for EMA convergence). 500 is safe and
    # cheap — stored raw bars are a small dict, not a DataFrame.
    _INCREMENTAL_WARMUP_BARS = 500

    def _empty_pending_entries_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "generated_date", "symbol", "contract_code", "group_name",
            "strategy_id", "action", "direction", "target_qty",
            "entry_price_ref", "stop_loss_ref", "entry_date",
        ])

    def _build_pending_entries_df(
        self, rows: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        if not rows:
            return self._empty_pending_entries_frame()
        df = pd.DataFrame(rows)
        cols = [
            "generated_date", "symbol", "contract_code", "group_name",
            "strategy_id", "action", "direction", "target_qty",
            "entry_price_ref", "stop_loss_ref", "entry_date",
        ]
        return df[cols].sort_values(
            ["generated_date", "symbol", "strategy_id"]
        ).reset_index(drop=True)

    # ---- warmup bars handling ----

    def _merge_warmup_bars(
        self, bars: pd.DataFrame, resume_state: Dict[str, Any]
    ) -> pd.DataFrame:
        """In incremental mode, prepend the warmup-bar tail from
        `resume_state` to `bars` so rolling indicators stay valid.

        Warmup rows are de-duplicated against `bars` on (date, symbol):
        callers can overlap safely.

        In batch mode (no warmup_bars in state), returns `bars` unchanged.
        """
        cfg = self.config
        warmup_records = resume_state.get("warmup_bars")
        if not warmup_records:
            return bars
        warmup_df = pd.DataFrame(warmup_records)
        # Normalize date column to datetime for consistent concat/dedup.
        if cfg.date_col in warmup_df.columns:
            warmup_df[cfg.date_col] = pd.to_datetime(warmup_df[cfg.date_col])
        if bars.empty:
            merged = warmup_df
        else:
            # Ensure all columns in bars are preserved; fill absent warmup
            # columns with NaN via a union.
            merged = pd.concat([warmup_df, bars], axis=0, ignore_index=True, sort=False)
        # Drop duplicate (date, symbol) keeping the LAST occurrence so
        # anything in the freshly-provided `bars` wins.
        merged = merged.drop_duplicates(
            subset=[cfg.date_col, cfg.symbol_col], keep="last"
        ).reset_index(drop=True)
        return merged

    def _extract_warmup_bars(self, merged_bars: pd.DataFrame) -> List[Dict[str, Any]]:
        """Tail of `merged_bars` kept as raw state for the next incremental
        run. Takes the last N trading dates system-wide and returns all
        rows on those dates.
        """
        cfg = self.config
        if merged_bars is None or merged_bars.empty:
            return []
        dates = (
            pd.Index(merged_bars[cfg.date_col])
            .drop_duplicates()
            .sort_values()
        )
        if len(dates) == 0:
            return []
        n = min(self._INCREMENTAL_WARMUP_BARS, len(dates))
        tail_dates = set(dates[-n:])
        tail = merged_bars[merged_bars[cfg.date_col].isin(tail_dates)].copy()
        # Coerce timestamps to ISO strings for JSON compatibility.
        tail[cfg.date_col] = pd.to_datetime(tail[cfg.date_col]).dt.strftime("%Y-%m-%d")
        return tail.to_dict(orient="records")

    # ---- state <-> dict converters ----

    def _position_to_state_dict(self, pos: Position) -> Dict[str, Any]:
        def _ts(x):
            return pd.Timestamp(x).strftime("%Y-%m-%d") if pd.notna(x) else None
        return {
            "symbol": pos.symbol,
            "strategy_id": pos.strategy_id,
            "group_name": pos.group_name,
            "direction": int(pos.direction),
            "signal_date": _ts(pos.signal_date),
            "entry_date": _ts(pos.entry_date),
            "entry_estimate": float(pos.entry_estimate),
            "entry_fill": float(pos.entry_fill),
            "entry_slippage": float(pos.entry_slippage),
            "qty": int(pos.qty),
            "contract_multiplier": float(pos.contract_multiplier),
            "entry_commission_per_contract": float(pos.entry_commission_per_contract),
            "atr_ref": float(pos.atr_ref),
            "volume": float(pos.volume),
            "open_interest": float(pos.open_interest),
            "initial_stop": float(pos.initial_stop),
            "active_stop": float(pos.active_stop),
            "estimated_initial_risk": float(pos.estimated_initial_risk),
            "estimated_order_risk": float(pos.estimated_order_risk),
            "actual_initial_risk": float(pos.actual_initial_risk),
            "actual_order_risk": float(pos.actual_order_risk),
            "risk_blowout_vs_estimate": float(pos.risk_blowout_vs_estimate),
            "risk_blowout_ratio": (
                float(pos.risk_blowout_ratio)
                if pos.risk_blowout_ratio is not None
                and np.isfinite(pos.risk_blowout_ratio)
                else None
            ),
            "r_price": float(pos.r_price),
            "r_money": float(pos.r_money),
            "highest_high_since_entry": float(pos.highest_high_since_entry),
            "lowest_low_since_entry": float(pos.lowest_low_since_entry),
            "completed_bars": int(pos.completed_bars),
            "pending_exit_reason": pos.pending_exit_reason,
            "pending_exit_date": _ts(pos.pending_exit_date),
            "active_stop_series": list(pos.active_stop_series),
            "mfe_price": float(pos.mfe_price),
            "mae_price": float(pos.mae_price),
            "consecutive_fail_count": int(pos.consecutive_fail_count),
            "original_qty": (
                int(pos.original_qty) if pos.original_qty is not None else None
            ),
            "qty_shrink_reason": pos.qty_shrink_reason,
            "profit_target_triggered": bool(pos.profit_target_triggered),
            "breakeven_triggered": bool(pos.breakeven_triggered),
            "metadata": _jsonable(pos.metadata),
            "current_contract": pos.current_contract,
            "raw_entry_fill": (
                float(pos.raw_entry_fill) if pos.raw_entry_fill is not None else None
            ),
            "segment_entry_fill": (
                float(pos.segment_entry_fill) if pos.segment_entry_fill is not None else None
            ),
            "realized_segment_pnl": float(pos.realized_segment_pnl),
            "roll_cost_accrued": float(pos.roll_cost_accrued),
            "rolls_crossed": list(pos.rolls_crossed),
        }

    def _position_from_state_dict(self, d: Dict[str, Any]) -> Position:
        def _ts(x):
            return pd.Timestamp(x) if x is not None else None
        rbr = d.get("risk_blowout_ratio")
        return Position(
            symbol=d["symbol"],
            strategy_id=d["strategy_id"],
            group_name=d["group_name"],
            direction=int(d["direction"]),
            signal_date=_ts(d["signal_date"]),
            entry_date=_ts(d["entry_date"]),
            entry_estimate=float(d["entry_estimate"]),
            entry_fill=float(d["entry_fill"]),
            entry_slippage=float(d["entry_slippage"]),
            qty=int(d["qty"]),
            contract_multiplier=float(d["contract_multiplier"]),
            entry_commission_per_contract=float(d["entry_commission_per_contract"]),
            atr_ref=float(d["atr_ref"]),
            volume=float(d["volume"]),
            open_interest=float(d["open_interest"]),
            initial_stop=float(d["initial_stop"]),
            active_stop=float(d["active_stop"]),
            estimated_initial_risk=float(d["estimated_initial_risk"]),
            estimated_order_risk=float(d["estimated_order_risk"]),
            actual_initial_risk=float(d["actual_initial_risk"]),
            actual_order_risk=float(d["actual_order_risk"]),
            risk_blowout_vs_estimate=float(d["risk_blowout_vs_estimate"]),
            risk_blowout_ratio=float(rbr) if rbr is not None else float("nan"),
            r_price=float(d["r_price"]),
            r_money=float(d["r_money"]),
            highest_high_since_entry=float(d["highest_high_since_entry"]),
            lowest_low_since_entry=float(d["lowest_low_since_entry"]),
            completed_bars=int(d.get("completed_bars", 0)),
            pending_exit_reason=d.get("pending_exit_reason"),
            pending_exit_date=_ts(d.get("pending_exit_date")),
            active_stop_series=list(d.get("active_stop_series", [])),
            mfe_price=float(d.get("mfe_price", 0.0)),
            mae_price=float(d.get("mae_price", 0.0)),
            consecutive_fail_count=int(d.get("consecutive_fail_count", 0)),
            original_qty=(
                int(d["original_qty"]) if d.get("original_qty") is not None else None
            ),
            qty_shrink_reason=d.get("qty_shrink_reason"),
            profit_target_triggered=bool(d.get("profit_target_triggered", False)),
            breakeven_triggered=bool(d.get("breakeven_triggered", False)),
            metadata=dict(d.get("metadata", {})),
            current_contract=d.get("current_contract"),
            raw_entry_fill=(
                float(d["raw_entry_fill"]) if d.get("raw_entry_fill") is not None else None
            ),
            segment_entry_fill=(
                float(d["segment_entry_fill"]) if d.get("segment_entry_fill") is not None else None
            ),
            realized_segment_pnl=float(d.get("realized_segment_pnl", 0.0)),
            roll_cost_accrued=float(d.get("roll_cost_accrued", 0.0)),
            rolls_crossed=list(d.get("rolls_crossed", [])),
        )

    def _pending_entry_to_state_dict(self, pe: PendingEntry) -> Dict[str, Any]:
        def _ts(x):
            return pd.Timestamp(x).strftime("%Y-%m-%d") if pd.notna(x) else None
        return {
            "symbol": pe.symbol,
            "strategy_id": pe.strategy_id,
            "group_name": pe.group_name,
            "direction": int(pe.direction),
            "signal_date": _ts(pe.signal_date),
            "entry_date": _ts(pe.entry_date),
            "entry_estimate": float(pe.entry_estimate),
            "qty": int(pe.qty),
            "atr_ref": float(pe.atr_ref),
            "volume": float(pe.volume),
            "open_interest": float(pe.open_interest),
            "initial_stop": float(pe.initial_stop),
            "estimated_initial_risk": float(pe.estimated_initial_risk),
            "estimated_order_risk": float(pe.estimated_order_risk),
            "contract_multiplier_est": float(pe.contract_multiplier_est),
            "metadata": _jsonable(pe.metadata),
        }

    def _pending_entry_from_state_dict(self, d: Dict[str, Any]) -> PendingEntry:
        def _ts(x):
            return pd.Timestamp(x) if x is not None else None
        return PendingEntry(
            symbol=d["symbol"],
            strategy_id=d["strategy_id"],
            group_name=d["group_name"],
            direction=int(d["direction"]),
            signal_date=_ts(d["signal_date"]),
            entry_date=_ts(d["entry_date"]),
            entry_estimate=float(d["entry_estimate"]),
            qty=int(d["qty"]),
            atr_ref=float(d["atr_ref"]),
            volume=float(d["volume"]),
            open_interest=float(d["open_interest"]),
            initial_stop=float(d["initial_stop"]),
            estimated_initial_risk=float(d["estimated_initial_risk"]),
            estimated_order_risk=float(d["estimated_order_risk"]),
            contract_multiplier_est=float(d["contract_multiplier_est"]),
            metadata=dict(d.get("metadata", {})),
        )

    def _build_initial_engine_state_empty(self) -> Dict[str, Any]:
        return {
            "last_date": None,
            "cash": float(self.config.initial_capital),
            "positions": [],
            "pending_entries": [],
            "last_close_by_symbol": {},
            "last_raw_close_by_symbol": {},
            "warmup_bars": [],
        }

    def _build_engine_state(
        self,
        cash: float,
        positions: Dict[Tuple[str, str], Position],
        pending_entries: Dict[Tuple[str, str], PendingEntry],
        last_close_by_symbol: Dict[str, float],
        last_raw_close_by_symbol: Dict[str, float],
        terminal_date: Optional[pd.Timestamp],
        original_bars: pd.DataFrame,
        merged_bars: pd.DataFrame,
    ) -> Dict[str, Any]:
        return {
            "last_date": (
                terminal_date.strftime("%Y-%m-%d")
                if terminal_date is not None else None
            ),
            "cash": float(cash),
            "positions": [self._position_to_state_dict(p) for p in positions.values()],
            "pending_entries": [
                self._pending_entry_to_state_dict(pe) for pe in pending_entries.values()
            ],
            "last_close_by_symbol": {
                str(k): float(v) for k, v in last_close_by_symbol.items()
            },
            "last_raw_close_by_symbol": {
                str(k): float(v) for k, v in last_raw_close_by_symbol.items()
            },
            "warmup_bars": self._extract_warmup_bars(merged_bars),
        }


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-compatible primitives.

    Keeps nested dicts/lists structural; maps pandas/numpy scalars to
    native Python types; falls back to str() for unknown objects so we
    never crash on an unexpected strategy metadata value.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, np.generic):
        return obj.item()
    # Unknown — degrade gracefully.
    try:
        return str(obj)
    except Exception:
        return None


from strats.engine_prepare import (  # noqa: E402
    _compute_data_quality_report as _prepare_compute_data_quality_report,
    _normalize_and_validate_bars as _prepare_normalize_and_validate_bars,
    _prepare_all_strategies as _prepare_all_strategies_impl,
    _prepare_symbol_base as _prepare_symbol_base_impl,
    _prepare_symbol_frame as _prepare_symbol_frame_impl,
    _validate_input_columns as _prepare_validate_input_columns,
    _validate_input_values as _prepare_validate_input_values,
)
from strats.engine_result import (  # noqa: E402
    _build_pending_entries_df as _result_build_pending_entries_df,
    _daily_status_columns as _result_daily_status_columns,
    _empty_cancelled_entries_frame as _result_empty_cancelled_entries_frame,
    _empty_open_positions_frame as _result_empty_open_positions_frame,
    _empty_pending_entries_frame as _result_empty_pending_entries_frame,
    _empty_trades_frame as _result_empty_trades_frame,
    _prepared_extra_columns as _result_prepared_extra_columns,
    _serialize_open_positions as _result_serialize_open_positions,
)
from strats.engine_risk import (  # noqa: E402
    _compute_effective_open_risk_for_entry_date as _risk_compute_effective_open_risk_for_entry_date,
    _compute_open_risk as _risk_compute_open_risk,
    _compute_total_notional as _risk_compute_total_notional,
    _compute_unrealized_exposure as _risk_compute_unrealized_exposure,
    _effective_margin_rate as _risk_effective_margin_rate,
    _months_to_delivery as _risk_months_to_delivery,
)
from strats.engine_runtime import run as _runtime_run  # noqa: E402
from strats.engine_state import (  # noqa: E402
    _INCREMENTAL_WARMUP_BARS as _STATE_INCREMENTAL_WARMUP_BARS,
    _build_engine_state as _state_build_engine_state,
    _build_initial_engine_state_empty as _state_build_initial_engine_state_empty,
    _extract_warmup_bars as _state_extract_warmup_bars,
    _merge_warmup_bars as _state_merge_warmup_bars,
    _pending_entry_from_state_dict as _state_pending_entry_from_state_dict,
    _pending_entry_to_state_dict as _state_pending_entry_to_state_dict,
    _position_from_state_dict as _state_position_from_state_dict,
    _position_to_state_dict as _state_position_to_state_dict,
)

StrategyEngine.run = _runtime_run

StrategyEngine._compute_data_quality_report = _prepare_compute_data_quality_report
StrategyEngine._normalize_and_validate_bars = _prepare_normalize_and_validate_bars
StrategyEngine._validate_input_columns = _prepare_validate_input_columns
StrategyEngine._validate_input_values = _prepare_validate_input_values
StrategyEngine._prepare_symbol_base = _prepare_symbol_base_impl
StrategyEngine._prepare_symbol_frame = _prepare_symbol_frame_impl
StrategyEngine._prepare_all_strategies = _prepare_all_strategies_impl

StrategyEngine._months_to_delivery = staticmethod(_risk_months_to_delivery)
StrategyEngine._effective_margin_rate = _risk_effective_margin_rate
StrategyEngine._compute_open_risk = _risk_compute_open_risk
StrategyEngine._compute_effective_open_risk_for_entry_date = (
    _risk_compute_effective_open_risk_for_entry_date
)
StrategyEngine._compute_unrealized_exposure = _risk_compute_unrealized_exposure
StrategyEngine._compute_total_notional = _risk_compute_total_notional

StrategyEngine._serialize_open_positions = _result_serialize_open_positions
StrategyEngine._prepared_extra_columns = _result_prepared_extra_columns
StrategyEngine._daily_status_columns = _result_daily_status_columns
StrategyEngine._empty_trades_frame = _result_empty_trades_frame
StrategyEngine._empty_open_positions_frame = _result_empty_open_positions_frame
StrategyEngine._empty_cancelled_entries_frame = _result_empty_cancelled_entries_frame
StrategyEngine._empty_pending_entries_frame = _result_empty_pending_entries_frame
StrategyEngine._build_pending_entries_df = _result_build_pending_entries_df

StrategyEngine._INCREMENTAL_WARMUP_BARS = _STATE_INCREMENTAL_WARMUP_BARS
StrategyEngine._merge_warmup_bars = _state_merge_warmup_bars
StrategyEngine._extract_warmup_bars = _state_extract_warmup_bars
StrategyEngine._position_to_state_dict = _state_position_to_state_dict
StrategyEngine._position_from_state_dict = _state_position_from_state_dict
StrategyEngine._pending_entry_to_state_dict = _state_pending_entry_to_state_dict
StrategyEngine._pending_entry_from_state_dict = _state_pending_entry_from_state_dict
StrategyEngine._build_initial_engine_state_empty = _state_build_initial_engine_state_empty
StrategyEngine._build_engine_state = _state_build_engine_state
