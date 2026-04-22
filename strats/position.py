"""PendingEntry + Position dataclasses — pulled out of engine.py.

Re-exported from strats.engine for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class PendingEntry:
    symbol: str
    strategy_id: str
    group_name: str
    direction: int  # 1=long, -1=short
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_estimate: float
    qty: int
    atr_ref: float
    volume: float
    open_interest: float
    initial_stop: float
    estimated_initial_risk: float
    estimated_order_risk: float
    contract_multiplier_est: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    strategy_id: str
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
    atr_ref: float
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
    # v6: Profit-target state. True once 5R target has been hit and a
    # (partial or full) close was executed. Prevents re-triggering.
    profit_target_triggered: bool = False
    # v7: Breakeven-ratchet state. True once the breakeven trigger has fired
    # and the stop has been ratcheted upward.
    breakeven_triggered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Dual-stream segment accounting (populated only when EngineConfig.enable_dual_stream).
    # `entry_fill` stays in Panama space for backward compat with exit strategies'
    # MFE/MAE computation. `raw_entry_fill` records the raw-space entry fill used
    # for actual P&L. `segment_entry_fill` starts == raw_entry_fill and mutates at
    # each roll; realized_segment_pnl accumulates closed segments.
    current_contract: Optional[str] = None
    raw_entry_fill: Optional[float] = None
    segment_entry_fill: Optional[float] = None
    realized_segment_pnl: float = 0.0
    roll_cost_accrued: float = 0.0
    rolls_crossed: List[Dict[str, Any]] = field(default_factory=list)
