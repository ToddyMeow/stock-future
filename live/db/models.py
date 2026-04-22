"""
live/db/models.py — Pydantic v2 数据模型

Schema 定义：/Users/mm/Trading/stock-future/live/db/schema.sql（6 表）
用途：
  - 读模型（Position / Instruction / Fill / DailyPnl / Bar / Alert）供 API 响应序列化
  - 写模型（*Create）供 API request body 校验
  - 枚举给前端 / 后端共享常量

约定：
  - 所有时间字段用 UTC `datetime`（带 tzinfo）；序列化到 ISO-8601
  - 金额 / 价格用 `Decimal` 保持精度
  - `model_config = ConfigDict(from_attributes=True)` — v2 语法，替代 v1 的 `orm_mode = True`

依赖：pydantic>=2.0（需由 P1c agent 的 requirements.txt 引入）
"""
from __future__ import annotations

from datetime import date as _date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# Enums — 状态机 / 固定词表（与 schema.sql CHECK 约束同步）
# =====================================================================


class InstructionStatus(str, Enum):
    """instructions.status 状态机。"""

    pending = "pending"
    fully_filled = "fully_filled"
    partially_filled = "partially_filled"
    vetoed = "vetoed"
    skipped = "skipped"
    expired = "expired"


class InstructionAction(str, Enum):
    """开 / 平 / 加 / 减仓。"""

    open = "open"
    close = "close"
    add = "add"
    reduce = "reduce"


class InstructionDirection(str, Enum):
    long = "long"
    short = "short"


class InstructionSession(str, Enum):
    day = "day"
    night = "night"


class AlertSeverity(str, Enum):
    info = "info"
    warn = "warn"
    critical = "critical"


class TriggerSource(str, Enum):
    """fills.trigger_source — 成交归因。"""

    user_manual = "user_manual"
    stop_loss = "stop_loss"
    take_profit = "take_profit"
    roll_contract = "roll_contract"


# =====================================================================
# 读模型（ORM → API 响应）
# =====================================================================


class Position(BaseModel):
    """当前持仓快照。

    基础字段来自 positions 表；带 `last_price` / `contract_multiplier` /
    `unrealized_pnl` / `notional_mv` / `last_settle_date` 的"enriched"字段
    由 account_state.get_positions_enriched() 通过 JOIN bars 表派生填充，
    在普通 get_positions() 返回时为 None（全 optional 保兼容）。

    命名约定：
      - last_price：最近一日 bars.settle（"最近一日结算价"），对齐前端 types.ts
        已有的 last_price 字段名，方便后续接行情源时平滑切换
      - last_settle_date：那一日日期，作数据新鲜度溯源
      - contract_multiplier：bars.contract_multiplier（AO=20 / JD=5 / 等）
      - unrealized_pnl：(last_price - avg_entry_price) * qty * multiplier
      - notional_mv：|qty| * last_price * multiplier（名义市值）
    """

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    contract_code: str
    qty: int = Field(description="正=多头手数，负=空头手数；0 手在 DB 层删除")
    avg_entry_price: Decimal
    stop_loss_price: Decimal | None = None
    group_name: str
    opened_at: datetime
    last_updated_at: datetime
    notes: str | None = None

    # ---- enriched 字段（bars JOIN 派生，纯展示用，DB 层不持久化）----
    last_price: Decimal | None = None
    last_settle_date: _date | None = None
    contract_multiplier: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    notional_mv: Decimal | None = None


class Instruction(BaseModel):
    """调仓指令（信号服务产出）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    generated_at: datetime
    session_date: _date
    session: InstructionSession
    symbol: str
    contract_code: str
    action: InstructionAction
    direction: InstructionDirection
    target_qty: int
    entry_price_ref: Decimal | None = None
    stop_loss_ref: Decimal | None = None
    group_name: str
    status: InstructionStatus = InstructionStatus.pending
    veto_reason: str | None = None
    broker_stop_order_id: str | None = Field(
        default=None, max_length=40, description="客户端止损单号，用户手填"
    )
    created_at: datetime
    updated_at: datetime


class Fill(BaseModel):
    """成交明细（一条 instruction 可多条 fill）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    instruction_id: UUID
    filled_qty: int = Field(gt=0)
    filled_price: Decimal = Field(gt=0)
    filled_at: datetime
    trigger_source: TriggerSource = TriggerSource.user_manual
    note: str | None = None
    created_at: datetime


class DailyPnl(BaseModel):
    """每日结算快照（权益曲线 + soft stop 源）。"""

    model_config = ConfigDict(from_attributes=True)

    date: _date
    equity: Decimal
    cash: Decimal
    open_positions_mv: Decimal
    realized_pnl_today: Decimal = Decimal("0")
    unrealized_pnl_today: Decimal = Decimal("0")
    soft_stop_triggered: bool = False
    drawdown_from_peak: Decimal | None = None
    peak_equity_to_date: Decimal | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


class Bar(BaseModel):
    """K 线日线数据（历史 + 实盘单一来源，22 列对齐 hab_bars.csv）。"""

    model_config = ConfigDict(from_attributes=True)

    date: _date
    symbol: str
    order_book_id: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    settle: Decimal
    volume: float
    open_interest: float
    contract_multiplier: Decimal
    commission: Decimal
    slippage: Decimal
    group_name: str
    margin_rate: Decimal
    open_raw: Decimal | None = None
    high_raw: Decimal | None = None
    low_raw: Decimal | None = None
    close_raw: Decimal | None = None
    settle_raw: Decimal | None = None
    limit_up: Decimal | None = None
    limit_down: Decimal | None = None
    created_at: datetime


class Alert(BaseModel):
    """审计日志 / 告警（Server 酱 + SMS 下游订阅）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    event_at: datetime
    severity: AlertSeverity
    event_type: str
    message: str
    payload: dict[str, Any] | None = None


class RollCandidate(BaseModel):
    """主力合约切换候选（Q6）。

    持仓的 contract_code ≠ bars 最新主力 order_book_id 时产生一条候选，
    需要用户手动"平旧开新"（MVP 不自动产调仓指令）。

    价格字段序列化为 Decimal 保精度；前端 JSON 收到后 Number() 即可。
    Optional 价格字段兼容"bars 表缺该 contract_code 历史数据"的极端场景。
    """

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    current_contract: str
    new_dominant_contract: str
    last_observed_date: _date
    current_last_price: Decimal | None = None
    new_last_price: Decimal | None = None
    avg_entry_price: Decimal
    qty: int
    group_name: str


# =====================================================================
# 写模型（API request body）
# =====================================================================


class InstructionCreate(BaseModel):
    """POST /api/instructions — signal_service 内部用。"""

    session_date: _date
    session: InstructionSession
    symbol: str
    contract_code: str
    action: InstructionAction
    direction: InstructionDirection
    target_qty: int = Field(gt=0)
    entry_price_ref: Decimal | None = None
    stop_loss_ref: Decimal | None = None
    group_name: str


class FillCreate(BaseModel):
    """POST /api/fills — 前端回填。"""

    instruction_id: UUID
    filled_qty: int = Field(gt=0, description="本次成交量；累计 ≤ target_qty")
    filled_price: Decimal = Field(gt=0)
    filled_at: datetime
    trigger_source: TriggerSource = TriggerSource.user_manual
    note: str | None = None


__all__ = [
    # Enums
    "InstructionStatus",
    "InstructionAction",
    "InstructionDirection",
    "InstructionSession",
    "AlertSeverity",
    "TriggerSource",
    # Read models
    "Position",
    "Instruction",
    "Fill",
    "DailyPnl",
    "Bar",
    "Alert",
    "RollCandidate",
    # Write models
    "InstructionCreate",
    "FillCreate",
]
