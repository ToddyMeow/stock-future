
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from strats.helpers import (
    apply_exit_slippage,
    directional_pnl,
    favorable_excursion,
    adverse_excursion,
    wilder_atr as _wilder_atr,
    rolling_last_value_percentile as _rolling_last_value_percentile,
    detect_hlh_pattern,
    detect_lhl_pattern,
    PortfolioAnalyzer,
)
from strats.entries.hab_entry import HABEntryStrategy, HABEntryConfig
from strats.exits.hab_exit import HABExitStrategy, HABExitConfig


@dataclass(frozen=True)
class HABConfig:
    """Configuration for HorizontalAccumulationBreakout_V1."""

    date_col: str = "date"
    symbol_col: str = "symbol"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    open_interest_col: str = "open_interest"
    multiplier_col: str = "contract_multiplier"
    commission_col: str = "commission"
    slippage_col: str = "slippage"
    group_col: str = "group_name"

    initial_capital: float = 1_000_000.0
    atr_period: int = 20
    bb_period: int = 20
    bb_std: float = 2.0
    bb_percentile_lookback: int = 60
    bb_percentile_threshold: float = 0.30

    box_lookback: int = 7
    box_width_atr_mult: float = 1.5
    tol_atr_mult: float = 0.25
    breakout_atr_mult: float = 0.5
    upper_shadow_ratio_max: float = 0.25
    initial_stop_atr_mult: float = 0.4

    risk_per_trade: float = 0.02
    portfolio_risk_cap: float = 0.12
    group_risk_cap: float = 0.06

    structure_fail_bars: int = 15
    time_fail_bars: int = 5
    time_fail_target_r: float = 0.5
    trail_activate_r: float = 1.0
    trail_atr_mult: float = 2.0

    risk_blowout_cap: float = 1.5
    risk_blowout_action: Literal["SHRINK", "CANCEL"] = "SHRINK"

    allow_short: bool = False

    structure_fail_mode: Literal["CLOSE_BELOW_BOX", "CLOSE_BELOW_BOX_MINUS_ATR", "CONSECUTIVE_CLOSE"] = "CLOSE_BELOW_BOX"
    structure_fail_atr_buffer: float = 0.5
    structure_fail_consecutive: int = 2

    margin_rate_col: str = "margin_rate"
    default_margin_rate: float = 0.10
    max_portfolio_leverage: float = 3.0

    eps: float = 1e-12


@dataclass
class PendingEntry:
    symbol: str
    group_name: str
    direction: int  # 1=long, -1=short
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_estimate: float
    qty: int
    box_high: float
    box_low: float
    atr_ref: float
    bb_percentile: float
    bandwidth: float
    shadow_ratio: float
    tol: float
    volume: float
    open_interest: float
    initial_stop: float
    estimated_initial_risk: float
    estimated_order_risk: float
    contract_multiplier_est: float


@dataclass
class Position:
    symbol: str
    group_name: str
    direction: int  # 1=long, -1=short
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_estimate: float
    entry_fill: float
    entry_slippage: float
    qty: int
    contract_multiplier: float
    entry_commission_per_contract: float
    box_high: float
    box_low: float
    atr_ref: float
    bb_percentile: float
    bandwidth: float
    shadow_ratio: float
    tol: float
    volume: float
    open_interest: float
    initial_stop: float
    active_stop: float
    estimated_initial_risk: float
    estimated_order_risk: float
    actual_initial_risk: float
    actual_order_risk: float
    risk_blowout_vs_estimate: float
    risk_blowout_ratio: float
    r_price: float
    r_money: float
    highest_high_since_entry: float
    lowest_low_since_entry: float
    completed_bars: int = 0
    pending_exit_reason: Optional[str] = None
    pending_exit_date: Optional[pd.Timestamp] = None
    active_stop_series: List[Dict[str, Any]] = field(default_factory=list)
    mfe_price: float = 0.0
    mae_price: float = 0.0
    consecutive_fail_count: int = 0
    original_qty: Optional[int] = None
    qty_shrink_reason: Optional[str] = None


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily_status: pd.DataFrame
    portfolio_daily: pd.DataFrame
    open_positions: pd.DataFrame
    prepared_data: pd.DataFrame
    cancelled_entries: pd.DataFrame


class HorizontalAccumulationBreakoutV1:
    """Daily-bar long-only breakout strategy with portfolio/group risk control.

    Design notes:
    - Signal generation and execution are separated so the research version can run
      on continuous back-adjusted series now, while a later contract-level fill
      layer can replace only the execution side.
    - Signal inputs rely only on completed bars.
    - Position size is estimated on signal-day close and never re-sized from the
      next open, even when a gap worsens the true initial risk.
    """

    def __init__(self, config: Optional[HABConfig] = None) -> None:
        self.config = config or HABConfig()
        cfg = self.config
        self._entry_strategy = HABEntryStrategy(HABEntryConfig(
            bb_period=cfg.bb_period,
            bb_std=cfg.bb_std,
            bb_percentile_lookback=cfg.bb_percentile_lookback,
            bb_percentile_threshold=cfg.bb_percentile_threshold,
            box_lookback=cfg.box_lookback,
            box_width_atr_mult=cfg.box_width_atr_mult,
            tol_atr_mult=cfg.tol_atr_mult,
            breakout_atr_mult=cfg.breakout_atr_mult,
            upper_shadow_ratio_max=cfg.upper_shadow_ratio_max,
            initial_stop_atr_mult=cfg.initial_stop_atr_mult,
            allow_short=cfg.allow_short,
            eps=cfg.eps,
        ))
        self._exit_strategy = HABExitStrategy(HABExitConfig(
            structure_fail_bars=cfg.structure_fail_bars,
            structure_fail_mode=cfg.structure_fail_mode,
            structure_fail_atr_buffer=cfg.structure_fail_atr_buffer,
            structure_fail_consecutive=cfg.structure_fail_consecutive,
            time_fail_bars=cfg.time_fail_bars,
            time_fail_target_r=cfg.time_fail_target_r,
            trail_activate_r=cfg.trail_activate_r,
            trail_atr_mult=cfg.trail_atr_mult,
        ))

    # Direction helpers delegate to module-level functions in helpers.py
    _apply_exit_slippage = staticmethod(apply_exit_slippage)
    _directional_pnl = staticmethod(directional_pnl)
    _favorable_excursion = staticmethod(favorable_excursion)
    _adverse_excursion = staticmethod(adverse_excursion)

    def prepare_data(self, bars: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        self._validate_input_columns(bars)

        df = bars.copy()

        # Fix 1: Normalize dates to trading-day granularity.
        dt = pd.to_datetime(df[cfg.date_col], errors="raise")
        if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
            dt = dt.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        df[cfg.date_col] = dt.dt.normalize()

        df = df.sort_values([cfg.symbol_col, cfg.date_col]).reset_index(drop=True)

        # Duplicate check on normalized dates (catches same-day different timestamps).
        duplicate_mask = df.duplicated(subset=[cfg.symbol_col, cfg.date_col], keep=False)
        if duplicate_mask.any():
            dupes = df.loc[duplicate_mask, [cfg.symbol_col, cfg.date_col]]
            raise ValueError(f"Duplicate symbol/date rows after date normalization:\n{dupes}")

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
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="raise")

        # Fix 2: No bfill — only ffill for metadata, reject leading NaNs.
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

        # Backward compatibility: add margin_rate if missing.
        if cfg.margin_rate_col not in df.columns:
            df[cfg.margin_rate_col] = cfg.default_margin_rate

        self._validate_input_values(df)

        # Fix 3: Gap audit — warn about suspicious data gaps per symbol.
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
        prepared = self.prepare_data(bars)

        if prepared.empty:
            return BacktestResult(
                trades=self._empty_trades_frame(),
                daily_status=pd.DataFrame(columns=self._daily_status_columns()),
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
            )

        dates = list(pd.Index(prepared[cfg.date_col]).drop_duplicates().sort_values())
        rows_by_date: Dict[pd.Timestamp, pd.DataFrame] = {
            date: day_df.sort_values(cfg.symbol_col).reset_index(drop=True)
            for date, day_df in prepared.groupby(cfg.date_col, sort=True)
        }

        positions: Dict[str, Position] = {}
        pending_entries: Dict[str, PendingEntry] = {}
        closed_trades: List[Dict[str, Any]] = []
        cancelled_entries: List[Dict[str, Any]] = []
        portfolio_daily: List[Dict[str, Any]] = []
        risk_reject: Dict[Tuple[pd.Timestamp, str], Optional[str]] = {}
        last_close_by_symbol: Dict[str, float] = {}

        cash = float(cfg.initial_capital)

        for date in dates:
            day_df = rows_by_date[date]
            close_map: Dict[str, float] = {
                str(row[cfg.symbol_col]): float(row[cfg.close_col])
                for _, row in day_df.iterrows()
            }

            # 1) Existing positions: open-gap stop, pending open exits, intraday stop.
            # 1）现有持仓处理：开仓跳空止损、挂单止损、日内止损
            for _, row in day_df.iterrows():
                symbol = str(row[cfg.symbol_col])
                if symbol not in positions:
                    continue
                position = positions[symbol]
                exit_record = self._process_open_and_intraday_for_existing_position(
                    position=position,
                    row=row,
                )
                if exit_record is not None:
                    cash += float(exit_record.pop("cash_delta"))
                    closed_trades.append(exit_record)
                    del positions[symbol]

            # 2) Pending entries fill at today's open.
            # 2）待开仓处理
            for _, row in day_df.iterrows():
                symbol = str(row[cfg.symbol_col])
                pending = pending_entries.get(symbol)
                if pending is None or pending.entry_date != date:
                    continue
                if symbol in positions:
                    continue

                entry_fill = self._estimate_entry_fill_from_row(row, direction=pending.direction)
                # Direction-aware stop invalidation check
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
                    del pending_entries[symbol]
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
                        del pending_entries[symbol]
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
                            del pending_entries[symbol]
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
                positions[symbol] = position
                del pending_entries[symbol]

                immediate_exit_record = self._process_open_and_intraday_for_existing_position(
                    position=position,
                    row=row,
                )
                if immediate_exit_record is not None:
                    cash += float(immediate_exit_record.pop("cash_delta"))
                    closed_trades.append(immediate_exit_record)
                    del positions[symbol]

            # 3) Close-phase logic for surviving positions.

            for _, row in day_df.iterrows():
                symbol = str(row[cfg.symbol_col])
                if symbol not in positions:
                    continue
                self._process_close_phase(position=positions[symbol], row=row)

            # 4) Update last available close per symbol.
            for _, row in day_df.iterrows():
                last_close_by_symbol[str(row[cfg.symbol_col])] = float(row[cfg.close_col])

            # 5) Mark portfolio at today's close.
            # 5）计算今日收盘时的权益
            equity_close = self._compute_equity_close(
                cash=cash,
                positions=positions,
                close_map=close_map,
                last_close_by_symbol=last_close_by_symbol,
            )
            open_risk_total, _ = self._compute_open_risk(
                positions=positions,
                close_map=close_map,
                last_close_by_symbol=last_close_by_symbol,
            )

            # 6) Signal generation / next-open pending entries.
            # 6）信号生成 / 下一个交易日待开仓处理
            for _, row in day_df.iterrows():
                risk_reject[(date, str(row[cfg.symbol_col]))] = None

            accepted_today_risk_total = 0.0
            accepted_notional_today = 0.0
            base_notional = self._compute_total_notional(
                positions, pending_entries, close_map, last_close_by_symbol,
            )
            candidate_rows = [
                row
                for _, row in day_df.iterrows()
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
                group_name = str(row[cfg.group_col])
                direction = int(row["entry_direction"])
                reason: Optional[str] = None

                if pd.isna(row["next_trade_date"]):
                    reason = "NO_NEXT_TRADE_DATE"
                elif symbol in positions:
                    reason = "ALREADY_IN_POSITION"
                elif symbol in pending_entries:
                    reason = "PENDING_ENTRY_EXISTS"
                else:
                    entry_estimate = float(row[cfg.close_col])
                    initial_stop = float(row["initial_stop"])
                    contract_multiplier = float(row[cfg.multiplier_col])
                    estimated_initial_risk = abs(entry_estimate - initial_stop)
                    per_contract_risk_est = estimated_initial_risk * contract_multiplier

                    if not np.isfinite(per_contract_risk_est) or per_contract_risk_est <= 0.0:
                        reason = "NON_POSITIVE_RISK"
                    else:
                        risk_budget_single = equity_close * cfg.risk_per_trade
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
                            group_cap = equity_close * cfg.group_risk_cap
                            portfolio_risk_if_filled = effective_open_risk + order_risk
                            group_risk_if_filled = effective_open_risk_by_group.get(group_name, 0.0) + order_risk

                            # Leverage check (base_notional hoisted before loop)
                            new_notional = entry_estimate * contract_multiplier * qty
                            total_notional_if_filled = base_notional + accepted_notional_today + new_notional

                            if portfolio_risk_if_filled > portfolio_cap + cfg.eps:
                                reason = "PORTFOLIO_RISK_CAP"
                            elif group_risk_if_filled > group_cap + cfg.eps:
                                reason = "GROUP_RISK_CAP"
                            elif equity_close > cfg.eps and total_notional_if_filled / equity_close > cfg.max_portfolio_leverage:
                                reason = "LEVERAGE_CAP"
                            else:
                                pending_entries[symbol] = PendingEntry(
                                    symbol=symbol,
                                    group_name=group_name,
                                    direction=direction,
                                    signal_date=date,
                                    entry_date=entry_date,
                                    entry_estimate=entry_estimate,
                                    qty=qty,
                                    box_high=float(row["box_high"]),
                                    box_low=float(row["box_low"]),
                                    atr_ref=float(row["atr_ref"]),
                                    bb_percentile=float(row["bb_percentile"]),
                                    bandwidth=float(row["bandwidth"]),
                                    shadow_ratio=float(row["shadow_ratio"]),
                                    tol=float(row["tol"]),
                                    volume=float(row[cfg.volume_col]),
                                    open_interest=float(row[cfg.open_interest_col]),
                                    initial_stop=initial_stop,
                                    estimated_initial_risk=estimated_initial_risk,
                                    estimated_order_risk=order_risk,
                                    contract_multiplier_est=contract_multiplier,
                                )
                                accepted_today_risk_total += order_risk
                                accepted_notional_today += new_notional

                risk_reject[(date, symbol)] = reason

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

        daily_status = prepared.copy()
        risk_reject_df = pd.DataFrame(
            [
                {
                    cfg.date_col: key[0],
                    cfg.symbol_col: key[1],
                    "risk_reject_reason": value,
                }
                for key, value in risk_reject.items()
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

        daily_status = daily_status[self._daily_status_columns()].sort_values(
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

    def _prepare_symbol_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        high = df[cfg.high_col].astype(float)
        low = df[cfg.low_col].astype(float)
        close = df[cfg.close_col].astype(float)

        # Engine base: ATR + atr_ref + next_trade_date (universal)
        out = df.copy()
        out["atr"] = _wilder_atr(high=high, low=low, close=close, period=cfg.atr_period)
        out["atr_ref"] = out["atr"].shift(1)
        out["next_trade_date"] = out[cfg.date_col].shift(-1)

        # Delegate signal computation to entry strategy
        out = self._entry_strategy.prepare_signals(out)
        return out

    def _estimate_entry_fill_from_row(self, row: pd.Series, direction: int = 1) -> float:
        #估算开仓价格
        cfg = self.config
        open_price = float(row[cfg.open_col])
        slippage = float(row[cfg.slippage_col])
        if direction == 1:
            return open_price + slippage
        else:
            return open_price - slippage

    def _fill_pending_entry(
        self,
        pending: PendingEntry,
        row: pd.Series,
        entry_fill: Optional[float] = None,
        qty_override: Optional[int] = None,
        original_qty: Optional[int] = None,
        qty_shrink_reason: Optional[str] = None,
    ) -> Tuple[Position, float]:

    # 开仓处理


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

        position = Position(
            symbol=pending.symbol,
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
            box_high=pending.box_high,
            box_low=pending.box_low,
            atr_ref=pending.atr_ref,
            bb_percentile=pending.bb_percentile,
            bandwidth=pending.bandwidth,
            shadow_ratio=pending.shadow_ratio,
            tol=pending.tol,
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
        # 取消开仓处理
        cfg = self.config
        contract_multiplier = float(row[cfg.multiplier_col])
        # 计算初始风险
        attempted_initial_risk = abs(attempted_entry_fill - pending.initial_stop)
        # 计算订单风险
        attempted_order_risk = attempted_initial_risk * contract_multiplier * pending.qty
        # 计算风险blowout vs 估计
        risk_blowout_vs_estimate = attempted_order_risk - pending.estimated_order_risk
        # 计算风险blowout ratio
        risk_blowout_ratio = (
            attempted_order_risk / pending.estimated_order_risk
            if pending.estimated_order_risk > cfg.eps
            else np.nan
        )
        # 返回取消开仓处理结果
        return {
            "symbol": pending.symbol,
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
            "box_high": pending.box_high,
            "box_low": pending.box_low,
            "atr_ref": pending.atr_ref,
            "bb_percentile": pending.bb_percentile,
            "bandwidth": pending.bandwidth,
            "shadow_ratio": pending.shadow_ratio,
            "tol": pending.tol,
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

    def _process_open_and_intraday_for_existing_position(
        self,
        position: Position,
        row: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        # 现有持仓处理：开仓跳空止损、挂单止损、日内止损
        cfg = self.config
        open_price = float(row[cfg.open_col])
        high_price = float(row[cfg.high_col])
        low_price = float(row[cfg.low_col])
        slippage = float(row[cfg.slippage_col])
        exit_commission = float(row[cfg.commission_col])
        date = pd.Timestamp(row[cfg.date_col])

        d = position.direction

        # 1）开仓跳空止损
        gap_stop = (d == 1 and open_price <= position.active_stop) or \
                   (d == -1 and open_price >= position.active_stop)
        if gap_stop:
            return self._close_position(
                position=position, exit_date=date,
                exit_fill=self._apply_exit_slippage(open_price, slippage, d),
                exit_reason="STOP_GAP", exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )

        # 2）挂单止损
        if position.pending_exit_reason is not None and position.pending_exit_date == date:
            return self._close_position(
                position=position, exit_date=date,
                exit_fill=self._apply_exit_slippage(open_price, slippage, d),
                exit_reason=position.pending_exit_reason, exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )

        # 3）日内止损 (condition is direction-independent: stop within bar range)
        if low_price <= position.active_stop <= high_price:
            position.mae_price = max(
                position.mae_price,
                self._adverse_excursion(position.active_stop, position.entry_fill, d),
            )
            return self._close_position(
                position=position, exit_date=date,
                exit_fill=self._apply_exit_slippage(position.active_stop, slippage, d),
                exit_reason="STOP_INTRADAY", exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )

        return None

    def _process_close_phase(self, position: Position, row: pd.Series) -> None:
        self._exit_strategy.process_close_phase(
            position=position, row=row, next_trade_date=row["next_trade_date"],
        )

    def _close_position(
        self,
        position: Position,
        exit_date: pd.Timestamp,
        exit_fill: float,
        exit_reason: str,
        exit_slippage: float,
        exit_commission_per_contract: float,
    ) -> Dict[str, Any]:
        gross_pnl = self._directional_pnl(exit_fill, position.entry_fill, position.direction) * position.contract_multiplier * position.qty
        total_entry_commission = position.entry_commission_per_contract * position.qty
        total_exit_commission = exit_commission_per_contract * position.qty
        net_pnl = gross_pnl - total_entry_commission - total_exit_commission
        cash_delta = gross_pnl - total_exit_commission

        r_money_abs = max(abs(position.r_money), self.config.eps)
        r_multiple = net_pnl / r_money_abs
        mfe_r = position.mfe_price / max(abs(position.r_price), self.config.eps)
        mae_r = position.mae_price / max(abs(position.r_price), self.config.eps)

        return {
            "symbol": position.symbol,
            "group_name": position.group_name,
            "direction": position.direction,
            "signal_date": position.signal_date,
            "entry_date": position.entry_date,
            "entry_fill_date": position.entry_date,
            "entry_estimate": position.entry_estimate,
            "entry_fill": position.entry_fill,
            "qty": position.qty,
            "contract_multiplier": position.contract_multiplier,
            "box_high": position.box_high,
            "box_low": position.box_low,
            "atr_ref": position.atr_ref,
            "bb_percentile": position.bb_percentile,
            "bandwidth": position.bandwidth,
            "shadow_ratio": position.shadow_ratio,
            "tol": position.tol,
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
            "cash_delta": cash_delta,
        }

    # 计算今日收盘时的权益
    def _compute_equity_close(
        self,
        cash: float,
        positions: Dict[str, Position],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
    ) -> float:
    

        unrealized = 0.0
        for symbol, position in positions.items():
            mark_price = close_map.get(symbol, last_close_by_symbol.get(symbol, position.entry_fill))
            unrealized += self._directional_pnl(mark_price, position.entry_fill, position.direction) * position.contract_multiplier * position.qty
        return cash + unrealized



    # 计算开仓风险
    def _compute_open_risk(
        self,
        positions: Dict[str, Position],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:

        open_risk_total = 0.0
        by_group: Dict[str, float] = {}
        # 计算开仓风险
        for symbol, position in positions.items():
            current_price = close_map.get(symbol, last_close_by_symbol.get(symbol, position.entry_fill))
            current_risk = max(self._directional_pnl(current_price, position.active_stop, position.direction), 0.0) * position.contract_multiplier * position.qty
            open_risk_total += current_risk
            # 计算分组风险
            by_group[position.group_name] = by_group.get(position.group_name, 0.0) + current_risk
        return open_risk_total, by_group

    # 计算有效开仓风险
    def _compute_effective_open_risk_for_entry_date(
        self,
        candidate_entry_date: pd.Timestamp,
        positions: Dict[str, Position],
        pending_entries: Dict[str, PendingEntry],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:


        total = 0.0
        by_group: Dict[str, float] = {}

        for symbol, position in positions.items():
            if position.pending_exit_date is not None and position.pending_exit_date <= candidate_entry_date:
                continue
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
        positions: Dict[str, Position],
        pending_entries: Dict[str, PendingEntry],
        close_map: Dict[str, float],
        last_close_by_symbol: Dict[str, float],
    ) -> float:
        total = 0.0
        for symbol, position in positions.items():
            mark_price = close_map.get(symbol, last_close_by_symbol.get(symbol, position.entry_fill))
            total += mark_price * position.contract_multiplier * position.qty
        for pending in pending_entries.values():
            total += pending.entry_estimate * pending.contract_multiplier_est * pending.qty
        return total

    def _serialize_open_positions(self, positions: Dict[str, Position]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        # 序列化现有持仓
        for position in positions.values():
            rows.append(
                {
                    "symbol": position.symbol,
                    "group_name": position.group_name,
                    "direction": position.direction,
                    "signal_date": position.signal_date,
                    "entry_date": position.entry_date,
                    "entry_fill_date": position.entry_date,
                    "entry_estimate": position.entry_estimate,
                    "entry_fill": position.entry_fill,
                    "qty": position.qty,
                    "contract_multiplier": position.contract_multiplier,
                    "box_high": position.box_high,
                    "box_low": position.box_low,
                    "atr_ref": position.atr_ref,
                    "bb_percentile": position.bb_percentile,
                    "bandwidth": position.bandwidth,
                    "shadow_ratio": position.shadow_ratio,
                    "tol": position.tol,
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
            )
        if not rows:
            return self._empty_open_positions_frame()
        return pd.DataFrame(rows).sort_values(["symbol", "entry_date"]).reset_index(drop=True)

    def _prepared_extra_columns(self) -> List[str]:
        return [
            "atr",
            "bb_mid",
            "bb_upper",
            "bb_lower",
            "bandwidth_denom",
            "bandwidth",
            "bb_percentile",
            "atr_ref",
            "box_high",
            "box_low",
            "box_width",
            "tol",
            "box_width_pass",
            "is_box",
            "has_upper_test_1",
            "has_lower_confirm",
            "has_upper_test_2",
            "has_lower_test_1",
            "has_upper_confirm",
            "has_lower_test_2",
            "shadow_ratio",
            "bb_filter_pass",
            "entry_trigger_pass",
            "entry_direction",
            "initial_stop",
            "next_trade_date",
        ]

    def _daily_status_columns(self) -> List[str]:
        cfg = self.config
        return [
            cfg.date_col,
            cfg.symbol_col,
            cfg.group_col,
            cfg.open_col,
            cfg.high_col,
            cfg.low_col,
            cfg.close_col,
            cfg.volume_col,
            cfg.open_interest_col,
            "atr",
            "atr_ref",
            "bandwidth",
            "bb_percentile",
            "box_high",
            "box_low",
            "box_width",
            "tol",
            "shadow_ratio",
            "is_box",
            "has_upper_test_1",
            "has_lower_confirm",
            "has_upper_test_2",
            "has_lower_test_1",
            "has_upper_confirm",
            "has_lower_test_2",
            "bb_filter_pass",
            "entry_trigger_pass",
            "entry_direction",
            "risk_reject_reason",
        ]

    def _empty_trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "symbol",
                "group_name",
                "direction",
                "signal_date",
                "entry_date",
                "entry_fill_date",
                "entry_estimate",
                "entry_fill",
                "qty",
                "contract_multiplier",
                "box_high",
                "box_low",
                "atr_ref",
                "bb_percentile",
                "bandwidth",
                "shadow_ratio",
                "tol",
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
                "group_name",
                "direction",
                "signal_date",
                "entry_date",
                "entry_fill_date",
                "entry_estimate",
                "entry_fill",
                "qty",
                "contract_multiplier",
                "box_high",
                "box_low",
                "atr_ref",
                "bb_percentile",
                "bandwidth",
                "shadow_ratio",
                "tol",
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
                "box_high",
                "box_low",
                "atr_ref",
                "bb_percentile",
                "bandwidth",
                "shadow_ratio",
                "tol",
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


__all__ = [
    "HABConfig",
    "BacktestResult",
    "HorizontalAccumulationBreakoutV1",
    "PortfolioAnalyzer",
    "detect_hlh_pattern",
    "detect_lhl_pattern",
]
