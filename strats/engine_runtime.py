"""Runtime entrypoint extracted from StrategyEngine."""

from __future__ import annotations

from datetime import date as _date_type
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from strats.result import BacktestResult


def run(
    self,
    bars: pd.DataFrame,
    initial_state: Optional[Dict[str, Any]] = None,
    warmup_until: Optional[Union[pd.Timestamp, _date_type, str]] = None,
) -> BacktestResult:
    cfg = self.config
    data_quality_report = self._compute_data_quality_report(bars)

    resume_state = initial_state or {}
    state_last_date: Optional[pd.Timestamp] = None
    if resume_state.get("last_date"):
        state_last_date = pd.Timestamp(resume_state["last_date"]).normalize()

    warmup_until_ts: Optional[pd.Timestamp] = None
    if warmup_until is not None:
        warmup_until_ts = pd.Timestamp(warmup_until).normalize()

    bars_for_prepare = self._merge_warmup_bars(bars, resume_state)
    prepared_by_strategy = self._prepare_all_strategies(bars_for_prepare)

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
    if state_last_date is not None:
        dates = [d for d in all_dates if d > state_last_date]
    else:
        dates = all_dates

    rows_by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        date: day_df.sort_values(cfg.symbol_col).reset_index(drop=True)
        for date, day_df in prepared.groupby(cfg.date_col, sort=True)
    }

    rows_by_date_by_strategy: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
    for sid, sp in prepared_by_strategy.items():
        rows_by_date_by_strategy[sid] = {
            date: day_df.sort_values(cfg.symbol_col).reset_index(drop=True)
            for date, day_df in sp.groupby(cfg.date_col, sort=True)
        }

    PositionKey = tuple[str, str]
    positions = {}
    pending_entries = {}
    closed_trades = []
    cancelled_entries = []
    portfolio_daily = []
    risk_reject: Dict[tuple[pd.Timestamp, str, str], Optional[str]] = {}
    last_close_by_symbol: Dict[str, float] = {}
    last_raw_close_by_symbol: Dict[str, float] = {}
    per_day_new_pending = []

    mark_col_effective = cfg.settle_col if cfg.settle_col in prepared.columns else cfg.close_col
    mark_raw_col_effective = (
        cfg.settle_raw_col if cfg.settle_raw_col in prepared.columns else cfg.raw_close_col
    )

    cash = float(cfg.initial_capital)

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
        close_map = {
            str(row[cfg.symbol_col]): float(row[mark_col_effective])
            for _, row in day_df.iterrows()
        }
        raw_close_map = (
            {
                str(row[cfg.symbol_col]): float(row[mark_raw_col_effective])
                for _, row in day_df.iterrows()
            }
            if cfg.enable_dual_stream
            else {}
        )

        if warmup_until_ts is not None and date <= warmup_until_ts:
            for _, row in day_df.iterrows():
                sym = str(row[cfg.symbol_col])
                last_close_by_symbol[sym] = float(row[mark_col_effective])
                if cfg.enable_dual_stream:
                    last_raw_close_by_symbol[sym] = float(row[mark_raw_col_effective])
            continue

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
                    exit_reason_closed = exit_record.get("exit_reason", "")
                    closed_trades.append(exit_record)
                    if not is_partial:
                        del positions[key]
                        slot_sar_on, _, _ = self._resolve_slot_sar(position.strategy_id)
                        if slot_sar_on and exit_reason_closed in cfg.reverse_eligible_reasons:
                            equity_est = float(portfolio_daily[-1]["equity"]) if portfolio_daily else cash
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

        for _, row in day_df.iterrows():
            symbol = str(row[cfg.symbol_col])
            keys_for_symbol = [k for k in pending_entries if k[0] == symbol]
            for key in keys_for_symbol:
                pending = pending_entries[key]
                if pending.entry_date != date or key in positions:
                    continue

                entry_fill = self._estimate_entry_fill_from_row(row, direction=pending.direction)
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

                contract_multiplier = float(row[cfg.multiplier_col])
                actual_initial_risk = abs(entry_fill - pending.initial_stop)
                actual_order_risk = actual_initial_risk * contract_multiplier * pending.qty
                blowout_ratio = (
                    actual_order_risk / pending.estimated_order_risk
                    if pending.estimated_order_risk > cfg.eps
                    else float("inf")
                )

                qty_override = None
                original_qty_record = None
                shrink_reason = None

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
                    per_contract_actual_risk = actual_initial_risk * contract_multiplier
                    max_allowed_risk = cfg.risk_blowout_cap * pending.estimated_order_risk
                    shrunk_qty = (
                        int(np.floor(max_allowed_risk / per_contract_actual_risk))
                        if per_contract_actual_risk > cfg.eps
                        else 0
                    )
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
                        slot_sar_on, _, _ = self._resolve_slot_sar(position.strategy_id)
                        if slot_sar_on and imm_exit_reason in cfg.reverse_eligible_reasons:
                            equity_est = float(portfolio_daily[-1]["equity"]) if portfolio_daily else cash
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

        for _, row in day_df.iterrows():
            symbol = str(row[cfg.symbol_col])
            keys_for_symbol = [k for k in positions if k[0] == symbol]
            for key in keys_for_symbol:
                position = positions[key]
                slot = self._strategy_map[position.strategy_id]
                slot.exit_strategy.process_close_phase(
                    position=position,
                    row=row,
                    next_trade_date=row["next_trade_date"],
                )

        for _, row in day_df.iterrows():
            sym = str(row[cfg.symbol_col])
            last_close_by_symbol[sym] = float(row[mark_col_effective])
            if cfg.enable_dual_stream:
                last_raw_close_by_symbol[sym] = float(row[mark_raw_col_effective])

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

        unrealized_exposure_total, _ = self._compute_unrealized_exposure(
            positions=positions,
            close_map=close_map,
            last_close_by_symbol=last_close_by_symbol,
        )
        soft_cap_ratio = cfg.unrealized_exposure_soft_cap
        soft_cap_triggered = bool(
            soft_cap_ratio > 0.0
            and equity_close > cfg.eps
            and unrealized_exposure_total > equity_close * soft_cap_ratio
        )

        for slot in self._strategies:
            slot_day = rows_by_date_by_strategy[slot.strategy_id].get(date)
            if slot_day is not None:
                for _, row in slot_day.iterrows():
                    risk_reject[(date, str(row[cfg.symbol_col]), slot.strategy_id)] = None

        accepted_today_risk_total = 0.0
        accepted_notional_today = 0.0
        accepted_margin_today = 0.0
        base_notional = self._compute_total_notional(
            positions,
            pending_entries,
            close_map,
            last_close_by_symbol,
        )

        if cfg.max_margin_utilization > 0.0:
            base_occupied_margin = 0.0
            for pos in positions.values():
                mark = close_map.get(pos.symbol, last_close_by_symbol.get(pos.symbol, pos.entry_fill))
                base_rate = cfg.default_margin_rate
                sym_row = day_df[day_df[cfg.symbol_col] == pos.symbol]
                if not sym_row.empty and cfg.margin_rate_col in sym_row.columns:
                    base_rate = float(sym_row.iloc[0][cfg.margin_rate_col])
                eff_rate = self._effective_margin_rate(base_rate, pos.current_contract, date)
                base_occupied_margin += pos.qty * mark * pos.contract_multiplier * eff_rate
        else:
            base_occupied_margin = 0.0

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

            candidate_rows = [row for _, row in slot_day.iterrows() if bool(row["entry_trigger_pass"])]

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

                bar_idx = int(row["_bar_index"]) if "_bar_index" in row.index else 0
                if cfg.warmup_bars > 0 and bar_idx < cfg.warmup_bars:
                    reason = "WARMUP_INSUFFICIENT"
                elif pd.isna(row["next_trade_date"]):
                    reason = "NO_NEXT_TRADE_DATE"
                elif key in positions:
                    reason = "ALREADY_IN_POSITION"
                elif cfg.symbol_position_lock and any(k[0] == symbol for k in positions):
                    reason = "SYMBOL_LOCKED"
                elif cfg.symbol_position_lock and any(k[0] == symbol for k in pending_entries):
                    reason = "SYMBOL_LOCKED"
                elif key in pending_entries:
                    reason = "PENDING_ENTRY_EXISTS"
                else:
                    entry_estimate = float(row[cfg.close_col])
                    atr_ref_val = float(row["atr_ref"]) if pd.notna(row.get("atr_ref")) else np.nan
                    atr_below_floor = (
                        cfg.min_atr_pct > 0.0
                        and abs(entry_estimate) > 0.0
                        and (
                            not np.isfinite(atr_ref_val)
                            or atr_ref_val < cfg.min_atr_pct * abs(entry_estimate)
                        )
                    )
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
                        initial_stop = (
                            entry_estimate - estimated_initial_risk
                            if direction == 1
                            else entry_estimate + estimated_initial_risk
                        )
                        per_contract_risk_est = estimated_initial_risk * contract_multiplier
                        if cfg.max_limit_days > 0:
                            lim_up = row.get(cfg.limit_up_col)
                            lim_dn = row.get(cfg.limit_down_col)
                            if pd.notna(lim_up) and pd.notna(lim_dn) and entry_estimate > 0:
                                up_pct = (float(lim_up) - entry_estimate) / entry_estimate
                                dn_pct = (entry_estimate - float(lim_dn)) / entry_estimate
                                limit_pct = max(up_pct, dn_pct, 0.0)
                                worst_case_per_contract = (
                                    entry_estimate
                                    * limit_pct
                                    * cfg.max_limit_days
                                    * contract_multiplier
                                )
                                per_contract_risk_est = max(
                                    per_contract_risk_est,
                                    worst_case_per_contract,
                                )
                        if not np.isfinite(per_contract_risk_est) or per_contract_risk_est <= 0.0:
                            reason = "NON_POSITIVE_RISK"
                        else:
                            effective_risk = cfg.risk_per_trade
                            risk_budget_single = equity_close * effective_risk
                            qty = int(np.floor(risk_budget_single / per_contract_risk_est))
                            if qty < 1:
                                reason = "QTY_LT_1"
                            else:
                                order_risk = per_contract_risk_est * qty
                                entry_date = pd.Timestamp(row["next_trade_date"])
                                effective_open_risk, effective_open_risk_by_group = self._compute_effective_open_risk_for_entry_date(
                                    candidate_entry_date=entry_date,
                                    positions=positions,
                                    pending_entries=pending_entries,
                                    close_map=close_map,
                                    last_close_by_symbol=last_close_by_symbol,
                                )
                                portfolio_cap = equity_close * cfg.portfolio_risk_cap
                                group_cap = (
                                    equity_close * cfg.default_group_risk_cap
                                    if group_name.startswith("ind_")
                                    else equity_close
                                    * cfg.group_risk_cap.get(
                                        group_name, cfg.default_group_risk_cap
                                    )
                                )
                                portfolio_risk_if_filled = effective_open_risk + order_risk
                                group_risk_if_filled = effective_open_risk_by_group.get(group_name, 0.0) + order_risk
                                new_notional = entry_estimate * contract_multiplier * qty
                                total_notional_if_filled = (
                                    base_notional + accepted_notional_today + new_notional
                                )
                                if portfolio_risk_if_filled > portfolio_cap + cfg.eps:
                                    reason = "PORTFOLIO_RISK_CAP"
                                elif cfg.use_group_risk_cap and group_risk_if_filled > group_cap + cfg.eps:
                                    reason = "GROUP_RISK_CAP"
                                elif cfg.use_group_risk_cap and group_name.startswith("ind_"):
                                    ind_risk = sum(
                                        r for g, r in effective_open_risk_by_group.items() if g.startswith("ind_")
                                    )
                                    if ind_risk + order_risk > equity_close * cfg.independent_group_soft_cap + cfg.eps:
                                        reason = "INDEPENDENT_SOFT_CAP"
                                if (
                                    reason is None
                                    and equity_close > cfg.eps
                                    and total_notional_if_filled / equity_close > cfg.max_portfolio_leverage
                                ):
                                    reason = "LEVERAGE_CAP"
                                candidate_margin = 0.0
                                if reason is None and cfg.max_margin_utilization > 0.0:
                                    base_rate = (
                                        float(row[cfg.margin_rate_col])
                                        if cfg.margin_rate_col in row.index
                                        else cfg.default_margin_rate
                                    )
                                    candidate_contract = (
                                        str(row[cfg.contract_col])
                                        if cfg.contract_col in row.index
                                        else None
                                    )
                                    eff_rate = self._effective_margin_rate(base_rate, candidate_contract, date)
                                    candidate_margin = entry_estimate * contract_multiplier * qty * eff_rate
                                    if (
                                        base_occupied_margin + accepted_margin_today + candidate_margin
                                        > equity_close * cfg.max_margin_utilization + cfg.eps
                                    ):
                                        reason = "MARGIN_CAP"
                                if reason is None:
                                    new_pending = self._pending_entry_from_state_dict(
                                        {
                                            "symbol": symbol,
                                            "strategy_id": slot.strategy_id,
                                            "group_name": group_name,
                                            "direction": direction,
                                            "signal_date": date,
                                            "entry_date": entry_date,
                                            "entry_estimate": entry_estimate,
                                            "qty": qty,
                                            "atr_ref": float(row["atr_ref"]),
                                            "volume": float(row[cfg.volume_col]),
                                            "open_interest": float(row[cfg.open_interest_col]),
                                            "initial_stop": initial_stop,
                                            "estimated_initial_risk": estimated_initial_risk,
                                            "estimated_order_risk": order_risk,
                                            "contract_multiplier_est": contract_multiplier,
                                            "metadata": slot.entry_strategy.build_pending_entry_metadata(row),
                                        }
                                    )
                                    pending_entries[key] = new_pending
                                    accepted_today_risk_total += order_risk
                                    accepted_notional_today += new_notional
                                    accepted_margin_today += candidate_margin
                                    contract_code = (
                                        str(row[cfg.contract_col])
                                        if cfg.contract_col in row.index and pd.notna(row.get(cfg.contract_col))
                                        else symbol
                                    )
                                    per_day_new_pending.append(
                                        {
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
                                        }
                                    )

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
                "unrealized_exposure": unrealized_exposure_total,
                "unrealized_exposure_soft_cap_triggered": soft_cap_triggered,
            }
        )

    trades_df = self._empty_trades_frame()
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades).sort_values(
            ["exit_date", "symbol", "entry_date"]
        ).reset_index(drop=True)

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
            if key[2] == first_strategy_id
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

    if not portfolio_daily_df.empty:
        terminal_date = pd.Timestamp(portfolio_daily_df["date"].iloc[-1])
    elif warmup_until_ts is not None and dates:
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
