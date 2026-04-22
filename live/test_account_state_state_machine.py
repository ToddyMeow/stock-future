"""apply_fill 状态机测试（sqlite+aiosqlite in-memory）

覆盖场景：
  A  target=3 · 一次 fill qty=3 → fully_filled
  B  target=3 · fill 2 → partially_filled；再 fill 1 → fully_filled
  C  target=3 · fill 4（超额）→ fully_filled，positions.qty 按实际 +4
  D  action=open + direction=long → positions.qty 正；action=close → 减持
  E  veto_instruction 必填 reason（空字符串 ValueError）

注：模型层字段校验用 Pydantic；本测试走 apply_fill 语义 + SQL 写入。
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# 保证仓库根在 sys.path（以便 import live.*）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 必须在 import live.* 之前设置一个占位的 DATABASE_URL，
# 否则 live.config 会在 module 层 raise KeyError
os.environ.setdefault("DATABASE_URL", "postgresql://placeholder:placeholder@localhost/placeholder")
os.environ.setdefault("DATABASE_URL_ASYNC", "postgresql+asyncpg://placeholder:placeholder@localhost/placeholder")

from live import account_state  # noqa: E402
from live.db.models import FillCreate, TriggerSource  # noqa: E402


# =====================================================================
# SQLite schema（最简，覆盖测试要用到的约束）
# =====================================================================

_SQLITE_SCHEMA = """
CREATE TABLE positions (
  symbol TEXT NOT NULL,
  contract_code TEXT NOT NULL,
  qty INTEGER NOT NULL,
  avg_entry_price REAL NOT NULL,
  stop_loss_price REAL,
  group_name TEXT NOT NULL,
  opened_at TEXT NOT NULL,
  last_updated_at TEXT NOT NULL,
  notes TEXT,
  PRIMARY KEY (symbol, contract_code)
);

CREATE TABLE instructions (
  id TEXT PRIMARY KEY,
  generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  session_date TEXT NOT NULL,
  session TEXT NOT NULL,
  symbol TEXT NOT NULL,
  contract_code TEXT NOT NULL,
  action TEXT NOT NULL,
  direction TEXT NOT NULL,
  target_qty INTEGER NOT NULL,
  entry_price_ref REAL,
  stop_loss_ref REAL,
  group_name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  veto_reason TEXT,
  broker_stop_order_id TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE fills (
  id TEXT PRIMARY KEY,
  instruction_id TEXT NOT NULL,
  filled_qty INTEGER NOT NULL,
  filled_price REAL NOT NULL,
  filled_at TEXT NOT NULL,
  trigger_source TEXT NOT NULL DEFAULT 'user_manual',
  note TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE daily_pnl (
  date TEXT PRIMARY KEY,
  equity REAL NOT NULL,
  cash REAL NOT NULL,
  open_positions_mv REAL NOT NULL,
  realized_pnl_today REAL NOT NULL DEFAULT 0,
  unrealized_pnl_today REAL NOT NULL DEFAULT 0,
  soft_stop_triggered INTEGER NOT NULL DEFAULT 0,
  drawdown_from_peak REAL,
  peak_equity_to_date REAL,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  severity TEXT NOT NULL,
  event_type TEXT NOT NULL,
  message TEXT NOT NULL,
  payload TEXT
);
"""


async def _init_sqlite_engine() -> AsyncEngine:
    # file: memory shared URI 对同一 engine 的多 connection 可见
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        for stmt in _SQLITE_SCHEMA.strip().split(";\n\n"):
            s = stmt.strip()
            if s:
                await conn.execute(text(s))
    return engine


async def _insert_instruction(
    engine: AsyncEngine,
    *,
    action: str = "open",
    direction: str = "long",
    target: int = 3,
    symbol: str = "AG",
    contract: str = "AG2501",
) -> str:
    """插入一条 pending instruction，返回 id。"""
    iid = str(uuid4())
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO instructions
                  (id, session_date, session, symbol, contract_code,
                   action, direction, target_qty, group_name, status,
                   generated_at, created_at, updated_at)
                VALUES
                  (:id, :sd, 'day', :sym, :con, :act, :dir, :tq, 'metals', 'pending',
                   :now, :now, :now)
                """
            ),
            {
                "id": iid,
                "sd": "2025-12-31",
                "sym": symbol,
                "con": contract,
                "act": action,
                "dir": direction,
                "tq": target,
                "now": datetime.now(timezone.utc).isoformat(),
            },
        )
    return iid


# =====================================================================
# Fixtures
# =====================================================================


@pytest_asyncio.fixture
async def engine():
    """每个测试独立的 in-memory sqlite engine。"""
    eng = await _init_sqlite_engine()
    account_state.set_engine(eng)
    yield eng
    await eng.dispose()


# =====================================================================
# Case A — target=3 · 一次 fill qty=3 → fully_filled
# =====================================================================


@pytest.mark.asyncio
async def test_case_a_single_fill_fully(engine):
    iid = await _insert_instruction(engine, target=3)
    fill = FillCreate(
        instruction_id=iid,
        filled_qty=3,
        filled_price=Decimal("100.0"),
        filled_at=datetime.now(timezone.utc),
        trigger_source=TriggerSource.user_manual,
    )
    await account_state.apply_fill(fill)

    # 指令状态
    inst = await account_state.get_instruction(iid)
    assert inst is not None
    assert inst.status.value == "fully_filled"

    # 持仓
    positions = await account_state.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AG"
    assert positions[0].qty == 3
    assert positions[0].avg_entry_price == Decimal("100.0")


# =====================================================================
# Case B — target=3 · fill 2 → partial，再 fill 1 → fully
# =====================================================================


@pytest.mark.asyncio
async def test_case_b_partial_then_full(engine):
    iid = await _insert_instruction(engine, target=3)

    fill1 = FillCreate(
        instruction_id=iid,
        filled_qty=2,
        filled_price=Decimal("100.0"),
        filled_at=datetime.now(timezone.utc),
    )
    await account_state.apply_fill(fill1)

    inst = await account_state.get_instruction(iid)
    assert inst.status.value == "partially_filled"

    pos = await account_state.get_positions()
    assert pos[0].qty == 2

    fill2 = FillCreate(
        instruction_id=iid,
        filled_qty=1,
        filled_price=Decimal("101.0"),
        filled_at=datetime.now(timezone.utc),
    )
    await account_state.apply_fill(fill2)

    inst2 = await account_state.get_instruction(iid)
    assert inst2.status.value == "fully_filled"

    pos2 = await account_state.get_positions()
    assert pos2[0].qty == 3
    # open 场景下第 2 笔成交走的是 "防御 combine" 分支（open + 已有 position）
    # 会按 weight 加权：(2 * 100 + 1 * 101) / 3 = 100.333...
    avg = float(pos2[0].avg_entry_price)
    assert abs(avg - 100.3333333) < 1e-3, f"avg_entry_price={avg}"


# =====================================================================
# Case C — target=3 · fill 4 超额 → fully_filled，positions.qty +4
# =====================================================================


@pytest.mark.asyncio
async def test_case_c_overfill(engine):
    iid = await _insert_instruction(engine, target=3)

    fill = FillCreate(
        instruction_id=iid,
        filled_qty=4,
        filled_price=Decimal("100.0"),
        filled_at=datetime.now(timezone.utc),
    )
    await account_state.apply_fill(fill)

    inst = await account_state.get_instruction(iid)
    # 超额也视为 fully
    assert inst.status.value == "fully_filled"

    pos = await account_state.get_positions()
    # 实际成交 4 手，持仓按 4 手计
    assert pos[0].qty == 4


# =====================================================================
# Case D — open long 正向，close 减持
# =====================================================================


@pytest.mark.asyncio
async def test_case_d_open_long_then_close(engine):
    # 开仓 long 3 手
    iid_open = await _insert_instruction(
        engine, action="open", direction="long", target=3
    )
    await account_state.apply_fill(
        FillCreate(
            instruction_id=iid_open,
            filled_qty=3,
            filled_price=Decimal("100.0"),
            filled_at=datetime.now(timezone.utc),
        )
    )
    pos = await account_state.get_positions()
    assert pos[0].qty == 3  # 多头

    # 平仓 close long 2 手
    iid_close = await _insert_instruction(
        engine, action="close", direction="long", target=2
    )
    await account_state.apply_fill(
        FillCreate(
            instruction_id=iid_close,
            filled_qty=2,
            filled_price=Decimal("105.0"),
            filled_at=datetime.now(timezone.utc),
        )
    )
    pos_after = await account_state.get_positions()
    assert len(pos_after) == 1
    assert pos_after[0].qty == 1  # 剩 1 手

    # 再全部平掉 → 行被删
    iid_close2 = await _insert_instruction(
        engine, action="close", direction="long", target=1
    )
    await account_state.apply_fill(
        FillCreate(
            instruction_id=iid_close2,
            filled_qty=1,
            filled_price=Decimal("106.0"),
            filled_at=datetime.now(timezone.utc),
        )
    )
    pos_final = await account_state.get_positions()
    assert pos_final == []


# =====================================================================
# Case E — veto 必填 reason
# =====================================================================


@pytest.mark.asyncio
async def test_case_e_veto_requires_reason(engine):
    iid = await _insert_instruction(engine, target=3)

    # 空字符串 → ValueError
    with pytest.raises(ValueError):
        await account_state.veto_instruction(iid, "")

    with pytest.raises(ValueError):
        await account_state.veto_instruction(iid, "   ")

    # 正确 reason 生效
    inst = await account_state.veto_instruction(iid, "主力切换等手动处理")
    assert inst.status.value == "vetoed"
    assert inst.veto_reason == "主力切换等手动处理"
