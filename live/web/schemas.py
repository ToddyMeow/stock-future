from __future__ import annotations

from datetime import date as _date
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from live.db.models import Alert, Position


class VetoBody(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


class RollConfirmRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=10)
    old_contract: str = Field(min_length=1, max_length=20)
    new_contract: str = Field(min_length=1, max_length=20)
    old_close_price: Decimal = Field(gt=0)
    new_open_price: Decimal = Field(gt=0)
    closed_at: datetime | None = None
    opened_at: datetime | None = None
    note: str | None = Field(default=None, max_length=500)


class RollConfirmResponse(BaseModel):
    symbol: str
    old_contract: str
    new_contract: str
    qty: int
    old_close_price: Decimal
    new_open_price: Decimal
    closed_instruction_id: UUID
    opened_instruction_id: UUID
    new_position: Position


class UniverseSymbol(BaseModel):
    symbol: str
    group_name: str
    combo: str
    contract_code: str | None = None
    last_settle: Decimal | None = None
    last_settle_date: _date | None = None
    contract_multiplier: Decimal | None = None
    single_contract_risk: Decimal | None = None
    tradeable_250k: bool
    in_position: bool = False
    position_qty: int | None = None
    last_session_date: _date | None = None
    last_session: str | None = None
    last_entry_trigger: bool | None = None
    last_entry_direction: int | None = None
    last_reject_reason: str | None = None
    last_miss_reason: str | None = None


class EngineStateSummary(BaseModel):
    session_date: _date
    session: str
    state_last_date: _date | None = None
    state_positions_count: int
    state_bytes: int
    created_at: datetime


class InstructionsDayCount(BaseModel):
    date: _date
    total: int
    by_status: dict[str, int]


class CapitalSnapshot(BaseModel):
    date: _date
    equity: Decimal
    cash: Decimal
    open_positions_mv: Decimal
    drawdown_from_peak: Decimal | None = None
    peak_equity_to_date: Decimal | None = None


class LaunchdSlot(BaseModel):
    label: str
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    description: str
    next_fire: datetime


class EngineStatus(BaseModel):
    latest_state: Optional[EngineStateSummary] = None
    instructions_by_date: list[InstructionsDayCount]
    alerts_24h_count: dict[str, int]
    recent_alerts: list[Alert]
    db_health: str
    launchd_schedule: list[LaunchdSlot]
    server_time: datetime
    server_timezone: str
    capital_snapshot: Optional[CapitalSnapshot] = None


class FormulasContext(BaseModel):
    initial_capital: Decimal
    risk_per_trade: float
    portfolio_risk_cap: float
    group_risk_cap_default: float
    max_portfolio_leverage: float
    soft_stop_pct: float
    soft_stop_enabled: bool
    unrealized_exposure_soft_cap: float
    stop_atr_mult: float
    atr_period: int
    equity: Decimal | None
    cash: Decimal | None
    peak_equity: Decimal | None
    drawdown_from_peak: Decimal | None
    snapshot_date: _date | None = None
    risk_budget_per_trade: Decimal
    portfolio_cap_amount: Decimal
    group_cap_default_amount: Decimal
    leverage_cap_amount: Decimal
    positions_count: int
    total_principal_risk: Decimal
    total_unrealized_exposure: Decimal
    total_notional: Decimal
    current_leverage: Decimal
