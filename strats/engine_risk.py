"""Risk helpers extracted from StrategyEngine."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd


def _months_to_delivery(order_book_id: Optional[str], today: pd.Timestamp) -> Optional[int]:
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
    cfg = self.config
    effective = base_rate + cfg.broker_margin_addon
    if cfg.margin_tier_schedule:
        m = self._months_to_delivery(order_book_id, today)
        if m is not None and m >= 0:
            effective += cfg.margin_tier_schedule.get(m, 0.0)
    return effective


def _compute_open_risk(
    self,
    positions,
    close_map,
    last_close_by_symbol,
) -> Tuple[float, Dict[str, float]]:
    del close_map
    del last_close_by_symbol
    open_risk_total = 0.0
    by_group: Dict[str, float] = {}
    for position in positions.values():
        principal_risk = max(
            self._directional_pnl(position.entry_fill, position.active_stop, position.direction),
            0.0,
        ) * position.contract_multiplier * position.qty
        open_risk_total += principal_risk
        by_group[position.group_name] = by_group.get(position.group_name, 0.0) + principal_risk
    return open_risk_total, by_group


def _compute_effective_open_risk_for_entry_date(
    self,
    candidate_entry_date: pd.Timestamp,
    positions,
    pending_entries,
    close_map,
    last_close_by_symbol,
) -> Tuple[float, Dict[str, float]]:
    del close_map
    del last_close_by_symbol
    total = 0.0
    by_group: Dict[str, float] = {}

    for position in positions.values():
        if position.pending_exit_date is not None and position.pending_exit_date <= candidate_entry_date:
            continue
        principal_risk = max(
            self._directional_pnl(position.entry_fill, position.active_stop, position.direction),
            0.0,
        ) * position.contract_multiplier * position.qty
        total += principal_risk
        by_group[position.group_name] = by_group.get(position.group_name, 0.0) + principal_risk

    for pending in pending_entries.values():
        if pending.entry_date <= candidate_entry_date:
            total += pending.estimated_order_risk
            by_group[pending.group_name] = (
                by_group.get(pending.group_name, 0.0) + pending.estimated_order_risk
            )

    return total, by_group


def _compute_unrealized_exposure(
    self,
    positions,
    close_map,
    last_close_by_symbol,
) -> Tuple[float, Dict[str, float]]:
    total = 0.0
    by_group: Dict[str, float] = {}
    for position in positions.values():
        symbol = position.symbol
        current_price = close_map.get(
            symbol,
            last_close_by_symbol.get(symbol, position.entry_fill),
        )
        exposure = max(
            self._directional_pnl(current_price, position.active_stop, position.direction),
            0.0,
        ) * position.contract_multiplier * position.qty
        total += exposure
        by_group[position.group_name] = by_group.get(position.group_name, 0.0) + exposure
    return total, by_group


def _compute_total_notional(
    self,
    positions,
    pending_entries,
    close_map,
    last_close_by_symbol,
) -> float:
    total = 0.0
    for position in positions.values():
        symbol = position.symbol
        mark_price = close_map.get(
            symbol,
            last_close_by_symbol.get(symbol, position.entry_fill),
        )
        total += mark_price * position.contract_multiplier * position.qty
    for pending in pending_entries.values():
        total += pending.entry_estimate * pending.contract_multiplier_est * pending.qty
    return total
