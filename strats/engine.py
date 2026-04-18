"""Strategy engine: composable entry/exit orchestration.

Usage:
    engine = StrategyEngine(
        config=EngineConfig(...),
        entry_strategy=HABEntryStrategy(HABEntryConfig(...)),
        exit_strategy=HABExitStrategy(HABExitConfig(...)),
    )
    result = engine.run(bars)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from strats.engine_config import EngineConfig, StrategySlot
from strats.helpers import (
    apply_exit_slippage,
    directional_pnl,
    favorable_excursion,
    adverse_excursion,
    adx as _adx,
    wilder_atr as _wilder_atr,
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

    This is the forward-looking API. For the HAB-specific legacy API, use
    ``HorizontalAccumulationBreakoutV1`` which wraps this engine with
    HABEntryStrategy + HABExitStrategy.
    """

    # Direction helpers delegate to module-level functions in helpers.py
    _apply_exit_slippage = staticmethod(apply_exit_slippage)
    _directional_pnl = staticmethod(directional_pnl)
    _favorable_excursion = staticmethod(favorable_excursion)
    _adverse_excursion = staticmethod(adverse_excursion)

    def __init__(
        self,
        config: EngineConfig,
        strategies: Optional[List[StrategySlot]] = None,
        # Backward compat: single entry/exit
        entry_strategy: Any = None,
        exit_strategy: Any = None,
    ) -> None:
        self.config = config
        # If old-style single strategy, wrap in StrategySlot
        if strategies is not None:
            self._strategies = strategies
        elif entry_strategy is not None and exit_strategy is not None:
            self._strategies = [StrategySlot("default", entry_strategy, exit_strategy)]
        else:
            raise ValueError("Provide either 'strategies' list or 'entry_strategy'+'exit_strategy'")
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

    def run(self, bars: pd.DataFrame) -> BacktestResult:
        cfg = self.config
        data_quality_report = self._compute_data_quality_report(bars)
        prepared_by_strategy = self._prepare_all_strategies(bars)

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
            )

        dates = list(pd.Index(prepared[cfg.date_col]).drop_duplicates().sort_values())

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

        # Decide the marking column once (1.3): prefer `settle` if it's in the
        # prepared frame, else fall back to `close`. The variable name
        # `close_map` is kept for minimal diff — semantically it's a mark map.
        mark_col_effective = cfg.settle_col if cfg.settle_col in prepared.columns else cfg.close_col
        mark_raw_col_effective = (
            cfg.settle_raw_col if cfg.settle_raw_col in prepared.columns else cfg.raw_close_col
        )

        cash = float(cfg.initial_capital)

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
                    exit_record = self._process_open_and_intraday_for_existing_position(
                        position=position,
                        row=row,
                    )
                    if exit_record is not None:
                        cash += float(exit_record.pop("cash_delta"))
                        closed_trades.append(exit_record)
                        del positions[key]

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

                    immediate_exit_record = self._process_open_and_intraday_for_existing_position(
                        position=position,
                        row=row,
                    )
                    if immediate_exit_record is not None:
                        cash += float(immediate_exit_record.pop("cash_delta"))
                        closed_trades.append(immediate_exit_record)
                        del positions[key]

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

            for slot in self._strategies:
                slot_day = rows_by_date_by_strategy[slot.strategy_id].get(date)
                if slot_day is None:
                    continue

                candidate_rows = [
                    row
                    for _, row in slot_day.iterrows()
                    if bool(row["entry_trigger_pass"])
                ]
                candidate_rows.sort(
                    key=lambda row: (
                        pd.Timestamp.max
                        if pd.isna(row["next_trade_date"])
                        else pd.Timestamp(row["next_trade_date"]),
                        str(row[cfg.symbol_col]),
                    )
                )

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
                        if atr_below_floor:
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
                                # ADX trend filter: scale risk by group trend strength
                                adx_val = float(row["adx"]) if pd.notna(row.get("adx")) else cfg.adx_scale
                                trend_score = max(min(adx_val / cfg.adx_scale, 1.0), cfg.adx_floor)
                                effective_risk = cfg.risk_per_trade * trend_score
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
                                    elif group_risk_if_filled > group_cap + cfg.eps:
                                        reason = "GROUP_RISK_CAP"
                                    elif group_name.startswith("ind_"):
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
                                        pending_entries[key] = PendingEntry(
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
                                        accepted_today_risk_total += order_risk
                                        accepted_notional_today += new_notional
                                        accepted_margin_today += candidate_margin

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
                }
            )

        trades_df = self._empty_trades_frame()
        if closed_trades:
            trades_df = pd.DataFrame(closed_trades).sort_values(
                ["exit_date", "symbol", "entry_date"]
            ).reset_index(drop=True)

        # Daily status uses the first strategy's prepared data (backward compat)
        daily_status = prepared.copy()
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
        portfolio_daily_df = pd.DataFrame(portfolio_daily).sort_values("date").reset_index(drop=True)
        cancelled_entries_df = self._empty_cancelled_entries_frame()
        if cancelled_entries:
            cancelled_entries_df = pd.DataFrame(cancelled_entries).sort_values(
                ["cancel_date", "symbol", "entry_date"]
            ).reset_index(drop=True)

        return BacktestResult(
            trades=trades_df,
            daily_status=daily_status,
            portfolio_daily=portfolio_daily_df,
            open_positions=open_positions_df,
            prepared_data=prepared,
            cancelled_entries=cancelled_entries_df,
            data_quality_report=data_quality_report,
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
        out["next_trade_date"] = out[cfg.date_col].shift(-1)
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
    ) -> Optional[Dict[str, Any]]:
        cfg = self.config
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
        # Long → SELL on exit (blocked by locked-DOWN); Short → BUY (by locked-UP).
        exit_side = -d
        if self._cannot_fill_side(row, exit_side):
            return None
        # Under dual_stream, compute raw fills in parallel. Panama fills remain
        # the display value in trade records; raw fills drive P&L via _close_position.
        raw_offset = self._panama_to_raw_offset(row)
        raw_open = float(row[cfg.raw_open_col]) if cfg.enable_dual_stream else None

        # 1) Gap stop
        gap_stop = (d == 1 and open_price <= position.active_stop) or \
                   (d == -1 and open_price >= position.active_stop)
        if gap_stop:
            return self._close_position(
                position=position, exit_date=date,
                exit_fill=self._apply_exit_slippage(open_price, slippage, d),
                raw_exit_fill=(
                    self._apply_exit_slippage(raw_open, slippage, d)
                    if cfg.enable_dual_stream else None
                ),
                exit_reason="STOP_GAP", exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )

        # 2) Pending exit
        if position.pending_exit_reason is not None and position.pending_exit_date == date:
            return self._close_position(
                position=position, exit_date=date,
                exit_fill=self._apply_exit_slippage(open_price, slippage, d),
                raw_exit_fill=(
                    self._apply_exit_slippage(raw_open, slippage, d)
                    if cfg.enable_dual_stream else None
                ),
                exit_reason=position.pending_exit_reason, exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )

        # 3) Intraday stop
        if low_price <= position.active_stop <= high_price:
            position.mae_price = max(
                position.mae_price,
                self._adverse_excursion(position.active_stop, position.entry_fill, d),
            )
            raw_stop_exit: Optional[float] = None
            if cfg.enable_dual_stream:
                raw_stop_price = position.active_stop - raw_offset
                raw_stop_exit = self._apply_exit_slippage(raw_stop_price, slippage, d)
            return self._close_position(
                position=position, exit_date=date,
                exit_fill=self._apply_exit_slippage(position.active_stop, slippage, d),
                raw_exit_fill=raw_stop_exit,
                exit_reason="STOP_INTRADAY", exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )

        return None

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
        open_risk_total = 0.0
        by_group: Dict[str, float] = {}
        for position in positions.values():
            symbol = position.symbol
            current_price = close_map.get(symbol, last_close_by_symbol.get(symbol, position.entry_fill))
            current_risk = max(self._directional_pnl(current_price, position.active_stop, position.direction), 0.0) * position.contract_multiplier * position.qty
            open_risk_total += current_risk
            by_group[position.group_name] = by_group.get(position.group_name, 0.0) + current_risk
        return open_risk_total, by_group

    def _compute_effective_open_risk_for_entry_date(
        self,
        candidate_entry_date: pd.Timestamp,
        positions: Dict[Tuple[str, str], Position],
        pending_entries: Dict[Tuple[str, str], PendingEntry],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        by_group: Dict[str, float] = {}

        for position in positions.values():
            if position.pending_exit_date is not None and position.pending_exit_date <= candidate_entry_date:
                continue
            symbol = position.symbol
            current_price = close_map.get(symbol, last_close_by_symbol.get(symbol, position.entry_fill))
            current_risk = max(self._directional_pnl(current_price, position.active_stop, position.direction), 0.0) * position.contract_multiplier * position.qty
            total += current_risk
            by_group[position.group_name] = by_group.get(position.group_name, 0.0) + current_risk

        for pending in pending_entries.values():
            if pending.entry_date <= candidate_entry_date:
                total += pending.estimated_order_risk
                by_group[pending.group_name] = (
                    by_group.get(pending.group_name, 0.0) + pending.estimated_order_risk
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
