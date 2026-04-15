
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
    risk_blowout_action: str = "SHRINK"

    allow_short: bool = False

    structure_fail_mode: str = "CLOSE_BELOW_BOX"
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

    def prepare_data(self, bars: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        self._validate_input_columns(bars)

        df = bars.copy()
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="raise")
        df = df.sort_values([cfg.symbol_col, cfg.date_col]).reset_index(drop=True)

        duplicate_mask = df.duplicated(subset=[cfg.symbol_col, cfg.date_col])
        if duplicate_mask.any():
            dupes = df.loc[duplicate_mask, [cfg.symbol_col, cfg.date_col]]
            raise ValueError(f"Duplicate symbol/date rows found:\n{dupes}")

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

        for col in [cfg.multiplier_col, cfg.commission_col, cfg.slippage_col, cfg.group_col]:
            df[col] = df.groupby(cfg.symbol_col, sort=False)[col].ffill().bfill()
            if df[col].isna().any():
                raise ValueError(f"Column '{col}' contains missing values after fill.")

        # Backward compatibility: add margin_rate if missing.
        if cfg.margin_rate_col not in df.columns:
            df[cfg.margin_rate_col] = cfg.default_margin_rate

        self._validate_input_values(df)

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

                            # Leverage check
                            new_notional = entry_estimate * contract_multiplier * qty
                            total_notional = self._compute_total_notional(
                                positions, pending_entries, close_map, last_close_by_symbol,
                            ) + new_notional

                            if portfolio_risk_if_filled > portfolio_cap + cfg.eps:
                                reason = "PORTFOLIO_RISK_CAP"
                            elif group_risk_if_filled > group_cap + cfg.eps:
                                reason = "GROUP_RISK_CAP"
                            elif equity_close > cfg.eps and total_notional / equity_close > cfg.max_portfolio_leverage:
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

                risk_reject[(date, symbol)] = reason

            total_notional = self._compute_total_notional(
                positions, pending_entries, close_map, last_close_by_symbol,
            )
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

        # 1）计算ATR
        atr = _wilder_atr(high=high, low=low, close=close, period=cfg.atr_period)
        # 2）计算BB
        bb_mid = close.rolling(cfg.bb_period, min_periods=cfg.bb_period).mean()
        bb_std = close.rolling(cfg.bb_period, min_periods=cfg.bb_period).std(ddof=0)
        bb_upper = bb_mid + cfg.bb_std * bb_std
        bb_lower = bb_mid - cfg.bb_std * bb_std


        # Positive normalization is deliberate so extreme near-zero or negative-price
        # regimes cannot flip bandwidth sign and contaminate compression percentiles.
        # 3）计算Bandwidth
        bandwidth_denom = bb_mid.abs().where(bb_mid.abs() > cfg.eps, np.nan)
        bandwidth = (bb_upper - bb_lower) / bandwidth_denom
        # 4）计算BB百分位
        bb_percentile = _rolling_last_value_percentile(
            bandwidth.shift(1),
            cfg.bb_percentile_lookback,
        )

        # 5）计算ATR参考值
        atr_ref = atr.shift(1)
        # 6）计算Box高
        box_high = high.shift(1).rolling(cfg.box_lookback, min_periods=cfg.box_lookback).max()
        box_low = low.shift(1).rolling(cfg.box_lookback, min_periods=cfg.box_lookback).min()
        box_width = box_high - box_low
        tol = cfg.tol_atr_mult * atr_ref

        # 7）计算上边缘测试1
        has_upper_test_1 = np.zeros(len(df), dtype=bool)
        # 8）计算下边缘确认
        has_lower_confirm = np.zeros(len(df), dtype=bool)
        # 9）计算上边缘测试2
        has_upper_test_2 = np.zeros(len(df), dtype=bool)
        h_l_h_valid = np.zeros(len(df), dtype=bool)

        # Short-side pattern arrays
        has_lower_test_1 = np.zeros(len(df), dtype=bool)
        has_upper_confirm = np.zeros(len(df), dtype=bool)
        has_lower_test_2 = np.zeros(len(df), dtype=bool)
        l_h_l_valid = np.zeros(len(df), dtype=bool)

        for i in range(len(df)):
            if i < cfg.box_lookback:
                continue
            if pd.isna(box_high.iloc[i]) or pd.isna(box_low.iloc[i]) or pd.isna(tol.iloc[i]):
                continue

            win_h = high.iloc[i - cfg.box_lookback : i].tolist()
            win_l = low.iloc[i - cfg.box_lookback : i].tolist()
            bh = float(box_high.iloc[i])
            bl = float(box_low.iloc[i])
            t = float(tol.iloc[i])

            valid, flag_u1, flag_l1, flag_u2 = detect_hlh_pattern(
                high_window=win_h, low_window=win_l,
                box_high=bh, box_low=bl, tol=t,
            )
            h_l_h_valid[i] = valid
            has_upper_test_1[i] = flag_u1
            has_lower_confirm[i] = flag_l1
            has_upper_test_2[i] = flag_u2

            if cfg.allow_short:
                s_valid, s_lt1, s_uc, s_lt2 = detect_lhl_pattern(
                    high_window=win_h, low_window=win_l,
                    box_high=bh, box_low=bl, tol=t,
                )
                l_h_l_valid[i] = s_valid
                has_lower_test_1[i] = s_lt1
                has_upper_confirm[i] = s_uc
                has_lower_test_2[i] = s_lt2


        box_width_pass = box_width <= cfg.box_width_atr_mult * atr_ref
        # 11）是否形成箱体（long side）
        is_box_long = box_width_pass & h_l_h_valid
        # Short side box
        is_box_short = box_width_pass & l_h_l_valid

        spread = (high - low).abs()
        upper_shadow_ratio = (high - close) / np.maximum(spread, cfg.eps)
        lower_shadow_ratio = (close - low) / np.maximum(spread, cfg.eps)
        bb_filter_pass = bb_percentile <= cfg.bb_percentile_threshold

        # 12）计算入场触发通过（long）
        long_trigger = (
            is_box_long
            & bb_filter_pass
            & (close > box_high + cfg.breakout_atr_mult * atr_ref)
            & (upper_shadow_ratio <= cfg.upper_shadow_ratio_max)
        )

        # Short trigger
        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = (
                is_box_short
                & bb_filter_pass
                & (close < box_low - cfg.breakout_atr_mult * atr_ref)
                & (lower_shadow_ratio <= cfg.upper_shadow_ratio_max)
            )

        # Combined: long takes priority if both fire on same bar
        entry_trigger_pass = long_trigger | short_trigger
        # Direction: 1=long, -1=short, 0=no signal
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger] = 1
        entry_direction[short_trigger & ~long_trigger] = -1

        # 13）计算初始止损（direction-dependent, stored per-direction）
        initial_stop_long = box_low - cfg.initial_stop_atr_mult * atr_ref
        initial_stop_short = box_high + cfg.initial_stop_atr_mult * atr_ref
        # Pick the right stop based on direction
        initial_stop = initial_stop_long.copy()
        short_mask = entry_direction == -1
        initial_stop[short_mask] = initial_stop_short[short_mask]

        out = df.copy()
        out["atr"] = atr
        out["bb_mid"] = bb_mid
        out["bb_upper"] = bb_upper
        out["bb_lower"] = bb_lower
        out["bandwidth_denom"] = bandwidth_denom
        out["bandwidth"] = bandwidth
        out["bb_percentile"] = bb_percentile
        out["atr_ref"] = atr_ref
        out["box_high"] = box_high
        out["box_low"] = box_low
        out["box_width"] = box_width
        out["tol"] = tol
        out["box_width_pass"] = box_width_pass.fillna(False)
        out["is_box"] = (is_box_long | is_box_short).fillna(False)
        out["has_upper_test_1"] = has_upper_test_1
        out["has_lower_confirm"] = has_lower_confirm
        out["has_upper_test_2"] = has_upper_test_2
        out["has_lower_test_1"] = has_lower_test_1
        out["has_upper_confirm"] = has_upper_confirm
        out["has_lower_test_2"] = has_lower_test_2
        out["shadow_ratio"] = upper_shadow_ratio
        out["bb_filter_pass"] = bb_filter_pass.fillna(False)
        out["entry_trigger_pass"] = entry_trigger_pass.fillna(False)
        out["entry_direction"] = entry_direction
        out["initial_stop"] = initial_stop
        out["next_trade_date"] = out[cfg.date_col].shift(-1)
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
            exit_fill = open_price - slippage if d == 1 else open_price + slippage
            return self._close_position(
                position=position,
                exit_date=date,
                exit_fill=exit_fill,
                exit_reason="STOP_GAP",
                exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )

        # 2）挂单止损
        if position.pending_exit_reason is not None and position.pending_exit_date == date:
            exit_fill = open_price - slippage if d == 1 else open_price + slippage
            return self._close_position(
                position=position,
                exit_date=date,
                exit_fill=exit_fill,
                exit_reason=position.pending_exit_reason,
                exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )

        # 3）日内止损
        intraday_stop = (d == 1 and low_price <= position.active_stop <= high_price) or \
                        (d == -1 and low_price <= position.active_stop <= high_price)
        if intraday_stop:
            exit_fill = position.active_stop - slippage if d == 1 else position.active_stop + slippage
            if d == 1:
                position.mae_price = max(
                    position.mae_price,
                    max(position.entry_fill - position.active_stop, 0.0),
                )
            else:
                position.mae_price = max(
                    position.mae_price,
                    max(position.active_stop - position.entry_fill, 0.0),
                )
            return self._close_position(
                position=position,
                exit_date=date,
                exit_fill=exit_fill,
                exit_reason="STOP_INTRADAY",
                exit_slippage=slippage,
                exit_commission_per_contract=exit_commission,
            )

        return None

    def _process_close_phase(self, position: Position, row: pd.Series) -> None:
        # 收盘处理
        cfg = self.config
        d = position.direction
        high_price = float(row[cfg.high_col])
        low_price = float(row[cfg.low_col])
        close_price = float(row[cfg.close_col])
        date = pd.Timestamp(row[cfg.date_col])
        next_trade_date = row["next_trade_date"]

        # 1）更新最高高、最低低
        position.highest_high_since_entry = max(position.highest_high_since_entry, high_price)
        position.lowest_low_since_entry = min(position.lowest_low_since_entry, low_price)
        position.completed_bars += 1
        # 2）计算MFE、MAE (direction-aware)
        if d == 1:
            position.mfe_price = max(
                position.mfe_price,
                max(position.highest_high_since_entry - position.entry_fill, 0.0),
            )
            position.mae_price = max(
                position.mae_price,
                max(position.entry_fill - position.lowest_low_since_entry, 0.0),
            )
        else:
            position.mfe_price = max(
                position.mfe_price,
                max(position.entry_fill - position.lowest_low_since_entry, 0.0),
            )
            position.mae_price = max(
                position.mae_price,
                max(position.highest_high_since_entry - position.entry_fill, 0.0),
            )

        if position.pending_exit_reason is None and pd.notna(next_trade_date):
            # 3）计算结构失败 (direction-aware, mode-aware)
            atr_now = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
            if position.completed_bars <= cfg.structure_fail_bars:
                if d == 1:
                    if cfg.structure_fail_mode == "CLOSE_BELOW_BOX":
                        bar_fail = close_price <= position.box_high
                    elif cfg.structure_fail_mode == "CLOSE_BELOW_BOX_MINUS_ATR":
                        bar_fail = close_price <= position.box_high - cfg.structure_fail_atr_buffer * atr_now
                    else:
                        bar_fail = close_price <= position.box_high
                else:
                    if cfg.structure_fail_mode == "CLOSE_BELOW_BOX":
                        bar_fail = close_price >= position.box_low
                    elif cfg.structure_fail_mode == "CLOSE_BELOW_BOX_MINUS_ATR":
                        bar_fail = close_price >= position.box_low + cfg.structure_fail_atr_buffer * atr_now
                    else:
                        bar_fail = close_price >= position.box_low

                if cfg.structure_fail_mode == "CONSECUTIVE_CLOSE":
                    if bar_fail:
                        position.consecutive_fail_count += 1
                    else:
                        position.consecutive_fail_count = 0
                    struct_trigger = position.consecutive_fail_count >= cfg.structure_fail_consecutive
                else:
                    struct_trigger = bar_fail

                if struct_trigger:
                    position.pending_exit_reason = "STRUCT_FAIL"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)

            # 4）计算时间失败 (direction-aware) — checked independently from struct fail
            if position.pending_exit_reason is None and position.completed_bars == cfg.time_fail_bars:
                if d == 1:
                    time_fail_cond = position.highest_high_since_entry < position.entry_fill + cfg.time_fail_target_r * position.r_price
                else:
                    time_fail_cond = position.entry_fill - position.lowest_low_since_entry < cfg.time_fail_target_r * position.r_price
                if time_fail_cond:
                    position.pending_exit_reason = "TIME_FAIL"
                    position.pending_exit_date = pd.Timestamp(next_trade_date)

        # 5）计算ATR
        atr_today = float(row["atr"]) if pd.notna(row["atr"]) else np.nan
        trailing_stop_candidate: Optional[float] = None
        active_stop_before = position.active_stop

        # 6）计算活动止损 (direction-aware)
        if np.isfinite(atr_today):
            if d == 1:
                trail_activated = position.highest_high_since_entry >= position.entry_fill + cfg.trail_activate_r * position.r_price
                if trail_activated:
                    trailing_stop_candidate = position.highest_high_since_entry - cfg.trail_atr_mult * atr_today
                    position.active_stop = max(position.active_stop, trailing_stop_candidate)
            else:
                trail_activated = position.entry_fill - position.lowest_low_since_entry >= cfg.trail_activate_r * position.r_price
                if trail_activated:
                    trailing_stop_candidate = position.lowest_low_since_entry + cfg.trail_atr_mult * atr_today
                    position.active_stop = min(position.active_stop, trailing_stop_candidate)

        position.active_stop_series.append(
            {
                "computed_on": date.strftime("%Y-%m-%d"),
                "effective_from": (
                    pd.Timestamp(next_trade_date).strftime("%Y-%m-%d")
                    if pd.notna(next_trade_date)
                    else None
                ),
                "phase": "close_update",
                "active_stop_before": active_stop_before,
                "active_stop_after": position.active_stop,
                "trailing_stop_candidate": trailing_stop_candidate,
                "atr_used": atr_today if np.isfinite(atr_today) else None,
                "highest_high_since_entry": position.highest_high_since_entry,
            }
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
        if position.direction == 1:
            gross_pnl = (exit_fill - position.entry_fill) * position.contract_multiplier * position.qty
        else:
            gross_pnl = (position.entry_fill - exit_fill) * position.contract_multiplier * position.qty
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
            if position.direction == 1:
                unrealized += (mark_price - position.entry_fill) * position.contract_multiplier * position.qty
            else:
                unrealized += (position.entry_fill - mark_price) * position.contract_multiplier * position.qty
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
            # 计算当前价格
            current_price = close_map.get(symbol, last_close_by_symbol.get(symbol, position.entry_fill))
            # 计算当前风险 (direction-aware)
            if position.direction == 1:
                current_risk = max(current_price - position.active_stop, 0.0) * position.contract_multiplier * position.qty
            else:
                current_risk = max(position.active_stop - current_price, 0.0) * position.contract_multiplier * position.qty
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
            if position.direction == 1:
                current_risk = max(current_price - position.active_stop, 0.0) * position.contract_multiplier * position.qty
            else:
                current_risk = max(position.active_stop - current_price, 0.0) * position.contract_multiplier * position.qty
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


class PortfolioAnalyzer:
    """Post-hoc analysis of BacktestResult for portfolio-level metrics."""

    def __init__(self, result: BacktestResult, config: HABConfig) -> None:
        self.result = result
        self.config = config

    def equity_curve(self) -> pd.DataFrame:
        pdf = self.result.portfolio_daily.copy()
        if pdf.empty:
            return pd.DataFrame(columns=["date", "equity", "peak", "drawdown", "drawdown_pct", "daily_return"])
        out = pdf[["date", "equity"]].copy()
        out["peak"] = out["equity"].cummax()
        out["drawdown"] = out["equity"] - out["peak"]
        out["drawdown_pct"] = out["drawdown"] / out["peak"].where(out["peak"] > 0, np.nan)
        out["daily_return"] = out["equity"].pct_change()
        return out

    def drawdown_episodes(self) -> pd.DataFrame:
        ec = self.equity_curve()
        if ec.empty:
            return pd.DataFrame(columns=["start_date", "trough_date", "recovery_date", "max_drawdown_pct", "duration_days"])
        episodes = []
        in_dd = False
        start = None
        trough = None
        trough_pct = 0.0
        for _, row in ec.iterrows():
            if row["drawdown"] < 0:
                if not in_dd:
                    in_dd = True
                    start = row["date"]
                    trough = row["date"]
                    trough_pct = row["drawdown_pct"]
                if row["drawdown_pct"] < trough_pct:
                    trough = row["date"]
                    trough_pct = row["drawdown_pct"]
            else:
                if in_dd:
                    episodes.append({
                        "start_date": start,
                        "trough_date": trough,
                        "recovery_date": row["date"],
                        "max_drawdown_pct": trough_pct,
                        "duration_days": (row["date"] - start).days,
                    })
                    in_dd = False
        if in_dd:
            episodes.append({
                "start_date": start,
                "trough_date": trough,
                "recovery_date": pd.NaT,
                "max_drawdown_pct": trough_pct,
                "duration_days": (ec["date"].iloc[-1] - start).days,
            })
        return pd.DataFrame(episodes)

    def group_contribution(self) -> pd.DataFrame:
        trades = self.result.trades
        if trades.empty:
            return pd.DataFrame(columns=["group_name", "net_pnl_sum", "trade_count", "win_count", "loss_count", "win_rate", "avg_r_multiple"])
        grouped = trades.groupby("group_name").agg(
            net_pnl_sum=("net_pnl", "sum"),
            trade_count=("net_pnl", "count"),
            win_count=("net_pnl", lambda x: (x > 0).sum()),
            loss_count=("net_pnl", lambda x: (x <= 0).sum()),
            avg_r_multiple=("r_multiple", "mean"),
        ).reset_index()
        grouped["win_rate"] = grouped["win_count"] / grouped["trade_count"].where(grouped["trade_count"] > 0, 1)
        return grouped

    def risk_utilization(self) -> pd.DataFrame:
        pdf = self.result.portfolio_daily.copy()
        if pdf.empty:
            return pd.DataFrame(columns=["date", "risk_usage_pct", "leverage"])
        out = pdf[["date"]].copy()
        out["risk_usage_pct"] = pdf["open_risk"] / pdf["portfolio_risk_cap"].where(pdf["portfolio_risk_cap"] > 0, np.nan)
        out["leverage"] = pdf.get("leverage", 0.0)
        return out

    def signal_density(self) -> pd.DataFrame:
        ds = self.result.daily_status
        if ds.empty:
            return pd.DataFrame(columns=["date", "signals_fired", "signals_accepted", "signals_rejected"])
        triggered = ds[ds["entry_trigger_pass"] == True]
        if triggered.empty:
            return pd.DataFrame(columns=["date", "signals_fired", "signals_accepted", "signals_rejected"])
        grouped = triggered.groupby("date").agg(
            signals_fired=("entry_trigger_pass", "count"),
            signals_rejected=("risk_reject_reason", lambda x: x.notna().sum()),
        ).reset_index()
        grouped["signals_accepted"] = grouped["signals_fired"] - grouped["signals_rejected"]
        return grouped

    def periodic_returns(self, freq: str = "M") -> pd.DataFrame:
        ec = self.equity_curve()
        if ec.empty:
            return pd.DataFrame(columns=["period", "return_pct"])
        ts = ec.set_index("date")["equity"]
        resampled = ts.resample(freq).last().dropna()
        returns = resampled.pct_change().dropna()
        out = returns.reset_index()
        out.columns = ["period", "return_pct"]
        return out

    def summary_stats(self) -> Dict[str, Any]:
        ec = self.equity_curve()
        trades = self.result.trades
        if ec.empty or len(ec) < 2:
            return {}

        total_days = (ec["date"].iloc[-1] - ec["date"].iloc[0]).days
        years = max(total_days / 365.25, 1e-6)
        final_equity = ec["equity"].iloc[-1]
        initial_equity = ec["equity"].iloc[0]
        total_return = final_equity / initial_equity - 1.0 if initial_equity > 0 else 0.0
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0

        daily_returns = ec["daily_return"].dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0.0
        downside = daily_returns[daily_returns < 0]
        sortino = (daily_returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0.0
        max_dd_pct = ec["drawdown_pct"].min() if not ec["drawdown_pct"].isna().all() else 0.0

        stats: Dict[str, Any] = {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown_pct": max_dd_pct,
            "total_days": total_days,
        }

        if not trades.empty:
            wins = trades[trades["net_pnl"] > 0]
            losses = trades[trades["net_pnl"] <= 0]
            stats["total_trades"] = len(trades)
            stats["win_rate"] = len(wins) / len(trades) if len(trades) > 0 else 0.0
            stats["profit_factor"] = (
                wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())
                if len(losses) > 0 and losses["net_pnl"].sum() != 0
                else float("inf") if len(wins) > 0 else 0.0
            )
            stats["avg_r_multiple"] = trades["r_multiple"].mean()
            stats["expectancy"] = trades["net_pnl"].mean()

        return stats


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    # 计算TR：最高价-最低价、最高价-前一日收盘价、最低价-前一日收盘价中的最大值
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out = pd.Series(np.nan, index=tr.index, dtype=float)
    if len(tr) < period:
        return out

    tr_values = tr.to_numpy(dtype=float)
    atr_values = np.full(len(tr_values), np.nan, dtype=float)
    atr_values[period - 1] = np.nanmean(tr_values[:period])
    for i in range(period, len(tr_values)):
        atr_values[i] = ((atr_values[i - 1] * (period - 1)) + tr_values[i]) / period
    return pd.Series(atr_values, index=tr.index)


def _rolling_last_value_percentile(values: pd.Series, window: int) -> pd.Series:
    # 计算最后一个值的百分位
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        sample = arr[i - window + 1 : i + 1]
        if np.isnan(sample).any():
            continue
        out[i] = float(np.sum(sample <= sample[-1])) / float(window)
    return pd.Series(out, index=values.index)


def detect_hlh_pattern(
    high_window: List[float],
    low_window: List[float],
    box_high: float,
    box_low: float,
    tol: float,
) -> Tuple[bool, bool, bool, bool]:
    """Detect the mandatory High-Low-High sequence within the box window."""

    state = 0
    has_upper_test_1 = False
    has_lower_confirm = False
    has_upper_test_2 = False

    for h, l in zip(high_window, low_window):
        upper_hit = h >= box_high - tol
        lower_hit = l <= box_low + tol

        if state == 0 and upper_hit:
            has_upper_test_1 = True
            state = 1
            continue

        if state == 1 and lower_hit:
            has_lower_confirm = True
            state = 2
            continue

        if state == 2 and upper_hit:
            has_upper_test_2 = True
            state = 3
            break

    valid = has_upper_test_1 and has_lower_confirm and has_upper_test_2
    return valid, has_upper_test_1, has_lower_confirm, has_upper_test_2


def detect_lhl_pattern(
    high_window: List[float],
    low_window: List[float],
    box_high: float,
    box_low: float,
    tol: float,
) -> Tuple[bool, bool, bool, bool]:
    """Detect the mandatory Low-High-Low sequence within the box window (short side)."""

    state = 0
    has_lower_test_1 = False
    has_upper_confirm = False
    has_lower_test_2 = False

    for h, l in zip(high_window, low_window):
        upper_hit = h >= box_high - tol
        lower_hit = l <= box_low + tol

        if state == 0 and lower_hit:
            has_lower_test_1 = True
            state = 1
            continue

        if state == 1 and upper_hit:
            has_upper_confirm = True
            state = 2
            continue

        if state == 2 and lower_hit:
            has_lower_test_2 = True
            state = 3
            break

    valid = has_lower_test_1 and has_upper_confirm and has_lower_test_2
    return valid, has_lower_test_1, has_upper_confirm, has_lower_test_2


__all__ = [
    "HABConfig",
    "BacktestResult",
    "HorizontalAccumulationBreakoutV1",
    "PortfolioAnalyzer",
    "detect_hlh_pattern",
    "detect_lhl_pattern",
]
