"""State helpers extracted from StrategyEngine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strats.position import PendingEntry, Position

_INCREMENTAL_WARMUP_BARS = 500


def _merge_warmup_bars(self, bars: pd.DataFrame, resume_state: Dict[str, Any]) -> pd.DataFrame:
    cfg = self.config
    warmup_records = resume_state.get("warmup_bars")
    if not warmup_records:
        return bars
    warmup_df = pd.DataFrame(warmup_records)
    if cfg.date_col in warmup_df.columns:
        warmup_df[cfg.date_col] = pd.to_datetime(warmup_df[cfg.date_col])
    if bars.empty:
        merged = warmup_df
    else:
        merged = pd.concat([warmup_df, bars], axis=0, ignore_index=True, sort=False)
    merged = merged.drop_duplicates(
        subset=[cfg.date_col, cfg.symbol_col],
        keep="last",
    ).reset_index(drop=True)
    return merged


def _extract_warmup_bars(self, merged_bars: pd.DataFrame) -> List[Dict[str, Any]]:
    cfg = self.config
    if merged_bars is None or merged_bars.empty:
        return []
    dates = pd.Index(merged_bars[cfg.date_col]).drop_duplicates().sort_values()
    if len(dates) == 0:
        return []
    n = min(_INCREMENTAL_WARMUP_BARS, len(dates))
    tail_dates = set(dates[-n:])
    tail = merged_bars[merged_bars[cfg.date_col].isin(tail_dates)].copy()
    tail[cfg.date_col] = pd.to_datetime(tail[cfg.date_col]).dt.strftime("%Y-%m-%d")
    return tail.to_dict(orient="records")


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
            if pos.risk_blowout_ratio is not None and np.isfinite(pos.risk_blowout_ratio)
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
        "original_qty": int(pos.original_qty) if pos.original_qty is not None else None,
        "qty_shrink_reason": pos.qty_shrink_reason,
        "profit_target_triggered": bool(pos.profit_target_triggered),
        "breakeven_triggered": bool(pos.breakeven_triggered),
        "metadata": _jsonable(pos.metadata),
        "current_contract": pos.current_contract,
        "raw_entry_fill": float(pos.raw_entry_fill) if pos.raw_entry_fill is not None else None,
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
        original_qty=int(d["original_qty"]) if d.get("original_qty") is not None else None,
        qty_shrink_reason=d.get("qty_shrink_reason"),
        profit_target_triggered=bool(d.get("profit_target_triggered", False)),
        breakeven_triggered=bool(d.get("breakeven_triggered", False)),
        metadata=dict(d.get("metadata", {})),
        current_contract=d.get("current_contract"),
        raw_entry_fill=float(d["raw_entry_fill"]) if d.get("raw_entry_fill") is not None else None,
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
    positions,
    pending_entries,
    last_close_by_symbol: Dict[str, float],
    last_raw_close_by_symbol: Dict[str, float],
    terminal_date: Optional[pd.Timestamp],
    original_bars: pd.DataFrame,
    merged_bars: pd.DataFrame,
) -> Dict[str, Any]:
    del original_bars
    return {
        "last_date": terminal_date.strftime("%Y-%m-%d") if terminal_date is not None else None,
        "cash": float(cash),
        "positions": [self._position_to_state_dict(p) for p in positions.values()],
        "pending_entries": [
            self._pending_entry_to_state_dict(pe) for pe in pending_entries.values()
        ],
        "last_close_by_symbol": {str(k): float(v) for k, v in last_close_by_symbol.items()},
        "last_raw_close_by_symbol": {
            str(k): float(v) for k, v in last_raw_close_by_symbol.items()
        },
        "warmup_bars": self._extract_warmup_bars(merged_bars),
    }


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, np.generic):
        return obj.item()
    try:
        return str(obj)
    except Exception:
        return None
