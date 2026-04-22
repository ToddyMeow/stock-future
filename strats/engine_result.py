"""Result helpers extracted from StrategyEngine."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd


def _serialize_open_positions(self, positions) -> pd.DataFrame:
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


def _prepared_extra_columns(self) -> List[str]:
    return self._ENGINE_BASE_COLUMNS + self._ENGINE_SIGNAL_COLUMNS


def _daily_status_columns(self, prepared: pd.DataFrame) -> List[str]:
    cfg = self.config
    base = [
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
    ]
    skip = set(base) | {"next_trade_date", "risk_reject_reason"}
    skip |= {cfg.multiplier_col, cfg.commission_col, cfg.slippage_col, cfg.margin_rate_col}
    extra = [c for c in prepared.columns if c not in skip]
    tail = ["entry_trigger_pass", "entry_direction", "risk_reject_reason"]
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


def _empty_pending_entries_frame(self) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "generated_date",
            "symbol",
            "contract_code",
            "group_name",
            "strategy_id",
            "action",
            "direction",
            "target_qty",
            "entry_price_ref",
            "stop_loss_ref",
            "entry_date",
        ]
    )


def _build_pending_entries_df(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return self._empty_pending_entries_frame()
    df = pd.DataFrame(rows)
    cols = [
        "generated_date",
        "symbol",
        "contract_code",
        "group_name",
        "strategy_id",
        "action",
        "direction",
        "target_qty",
        "entry_price_ref",
        "stop_loss_ref",
        "entry_date",
    ]
    return df[cols].sort_values(["generated_date", "symbol", "strategy_id"]).reset_index(
        drop=True
    )
