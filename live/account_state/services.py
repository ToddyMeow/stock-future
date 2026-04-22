"""Business/state-transition services for account_state."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from live.account_state.db import dec_str, get_engine, get_sessionmaker
from live.account_state.queries import get_instruction, get_position
from live.account_state.repositories import upsert_position
from live.db.models import Fill, FillCreate, TriggerSource

logger = logging.getLogger(__name__)


async def veto_instruction(id_: UUID, reason: str):
    if not reason or not reason.strip():
        raise ValueError("veto_instruction 必须填 reason")

    async with get_sessionmaker()() as s:
        async with s.begin():
            await s.execute(
                text(
                    """
                    UPDATE instructions
                       SET status = 'vetoed',
                           veto_reason = :r,
                           updated_at = :now
                     WHERE id = :id
                    """
                ),
                {"id": str(id_), "r": reason.strip(), "now": datetime.now(timezone.utc)},
            )
    inst = await get_instruction(id_)
    if inst is None:
        raise ValueError(f"instruction {id_} 不存在")
    return inst


async def skip_instruction(id_: UUID):
    async with get_sessionmaker()() as s:
        async with s.begin():
            await s.execute(
                text(
                    """
                    UPDATE instructions
                       SET status = 'skipped', updated_at = :now
                     WHERE id = :id
                    """
                ),
                {"id": str(id_), "now": datetime.now(timezone.utc)},
            )
    inst = await get_instruction(id_)
    if inst is None:
        raise ValueError(f"instruction {id_} 不存在")
    return inst


async def expire_pending_before(cutoff: datetime) -> int:
    async with get_sessionmaker()() as s:
        async with s.begin():
            rs = await s.execute(
                text(
                    """
                    UPDATE instructions
                       SET status = 'expired', updated_at = :now
                     WHERE status = 'pending' AND created_at < :cutoff
                    """
                ),
                {"cutoff": cutoff, "now": datetime.now(timezone.utc)},
            )
    return rs.rowcount or 0


async def apply_fill(fill: FillCreate) -> Fill:
    async with get_sessionmaker()() as s:
        async with s.begin():
            inst_row = (
                await s.execute(
                    text(
                        """
                        SELECT id, symbol, contract_code, action, direction,
                               target_qty, entry_price_ref, group_name, status
                          FROM instructions
                         WHERE id = :id
                        """
                    ),
                    {"id": str(fill.instruction_id)},
                )
            ).first()
            if inst_row is None:
                raise ValueError(f"instruction {fill.instruction_id} 不存在")
            m = inst_row._mapping

            from uuid import uuid4 as _uuid4

            await s.execute(
                text(
                    """
                    INSERT INTO fills
                      (id, instruction_id, filled_qty, filled_price, filled_at,
                       trigger_source, note)
                    VALUES
                      (:id, :iid, :qty, :price, :at, :src, :note)
                    """
                ),
                {
                    "id": str(_uuid4()),
                    "iid": str(fill.instruction_id),
                    "qty": int(fill.filled_qty),
                    "price": dec_str(fill.filled_price),
                    "at": fill.filled_at,
                    "src": fill.trigger_source.value
                    if isinstance(fill.trigger_source, TriggerSource)
                    else str(fill.trigger_source),
                    "note": fill.note,
                },
            )

            total_row = (
                await s.execute(
                    text(
                        "SELECT COALESCE(SUM(filled_qty), 0) FROM fills "
                        "WHERE instruction_id = :id"
                    ),
                    {"id": str(fill.instruction_id)},
                )
            ).first()
            total_filled = int(total_row[0]) if total_row else 0
            new_status = "fully_filled" if total_filled >= int(m["target_qty"]) else "partially_filled"
            await s.execute(
                text(
                    """
                    UPDATE instructions
                       SET status = :st, updated_at = :now
                     WHERE id = :id
                    """
                ),
                {
                    "st": new_status,
                    "now": datetime.now(timezone.utc),
                    "id": str(fill.instruction_id),
                },
            )

            await _apply_position_change(
                s,
                symbol=str(m["symbol"]),
                contract_code=str(m["contract_code"]),
                action=str(m["action"]),
                direction=str(m["direction"]),
                filled_qty=int(fill.filled_qty),
                filled_price=Decimal(str(fill.filled_price)),
                group=str(m["group_name"]),
                entry_price_ref=m["entry_price_ref"],
            )

            rs = await s.execute(
                text(
                    """
                    SELECT id, instruction_id, filled_qty, filled_price, filled_at,
                           trigger_source, note, created_at
                      FROM fills
                     WHERE instruction_id = :id
                     ORDER BY created_at DESC
                     LIMIT 1
                    """
                ),
                {"id": str(fill.instruction_id)},
            )
            new_fill_row = rs.first()

    return Fill.model_validate(dict(new_fill_row._mapping))


async def _apply_position_change(
    session: AsyncSession,
    *,
    symbol: str,
    contract_code: str,
    action: str,
    direction: str,
    filled_qty: int,
    filled_price: Decimal,
    group: str,
    entry_price_ref: Any,
) -> None:
    del entry_price_ref
    sign = 1 if direction == "long" else -1

    row = (
        await session.execute(
            text(
                """
                SELECT qty, avg_entry_price
                  FROM positions
                 WHERE symbol = :symbol AND contract_code = :contract
                """
            ),
            {"symbol": symbol, "contract": contract_code},
        )
    ).first()

    if action == "open":
        new_qty = sign * filled_qty
        if row is not None:
            old_qty = int(row._mapping["qty"])
            old_avg = Decimal(str(row._mapping["avg_entry_price"]))
            combined_qty = old_qty + new_qty
            if combined_qty == 0:
                await session.execute(
                    text(
                        "DELETE FROM positions WHERE symbol=:symbol AND "
                        "contract_code=:contract"
                    ),
                    {"symbol": symbol, "contract": contract_code},
                )
                return
            new_avg = (abs(old_qty) * old_avg + filled_qty * filled_price) / (
                abs(old_qty) + filled_qty
            )
            await upsert_position(
                symbol=symbol,
                contract_code=contract_code,
                qty=combined_qty,
                avg_price=new_avg,
                stop=None,
                group=group,
                session=session,
            )
        else:
            await upsert_position(
                symbol=symbol,
                contract_code=contract_code,
                qty=new_qty,
                avg_price=filled_price,
                stop=None,
                group=group,
                session=session,
            )
        return

    if action == "add":
        if row is None:
            await upsert_position(
                symbol=symbol,
                contract_code=contract_code,
                qty=sign * filled_qty,
                avg_price=filled_price,
                stop=None,
                group=group,
                session=session,
            )
            return
        old_qty = int(row._mapping["qty"])
        old_avg = Decimal(str(row._mapping["avg_entry_price"]))
        new_qty = old_qty + sign * filled_qty
        if new_qty == 0:
            await session.execute(
                text(
                    "DELETE FROM positions WHERE symbol=:symbol AND "
                    "contract_code=:contract"
                ),
                {"symbol": symbol, "contract": contract_code},
            )
            return
        new_avg = (old_avg * abs(old_qty) + filled_price * filled_qty) / (abs(old_qty) + filled_qty)
        await upsert_position(
            symbol=symbol,
            contract_code=contract_code,
            qty=new_qty,
            avg_price=new_avg,
            stop=None,
            group=group,
            session=session,
        )
        return

    if action in ("close", "reduce"):
        if row is None:
            logger.warning(
                "[apply_fill] 无持仓但收到 %s 操作：symbol=%s contract=%s qty=%s",
                action,
                symbol,
                contract_code,
                filled_qty,
            )
            return
        old_qty = int(row._mapping["qty"])
        new_qty = old_qty - sign * filled_qty
        if (old_qty > 0 and new_qty <= 0) or (old_qty < 0 and new_qty >= 0):
            await session.execute(
                text(
                    "DELETE FROM positions WHERE symbol=:symbol AND "
                    "contract_code=:contract"
                ),
                {"symbol": symbol, "contract": contract_code},
            )
            return
        old_avg = Decimal(str(row._mapping["avg_entry_price"]))
        await upsert_position(
            symbol=symbol,
            contract_code=contract_code,
            qty=new_qty,
            avg_price=old_avg,
            stop=None,
            group=group,
            session=session,
        )
        return

    raise ValueError(f"未知 action：{action}")


async def apply_roll(
    *,
    symbol: str,
    old_contract: str,
    new_contract: str,
    old_close_price: Decimal,
    new_open_price: Decimal,
    closed_at: datetime | None = None,
    opened_at: datetime | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    if old_close_price <= 0 or new_open_price <= 0:
        raise ValueError("old_close_price / new_open_price 必须为正")
    if old_contract == new_contract:
        raise ValueError("old_contract 与 new_contract 不能相同")

    now = datetime.now(timezone.utc)
    closed_at_final = closed_at if closed_at is not None else now
    opened_at_final = opened_at if opened_at is not None else now

    hour_utc = now.hour + now.minute / 60.0
    session_tag = "night" if 12.0 <= hour_utc <= 18.5 else "day"
    session_date = now.date()

    from uuid import uuid4 as _uuid4

    close_instr_id = str(_uuid4())
    open_instr_id = str(_uuid4())
    close_fill_id = str(_uuid4())
    open_fill_id = str(_uuid4())

    async with get_sessionmaker()() as s:
        async with s.begin():
            row = (
                await s.execute(
                    text(
                        """
                        SELECT qty, avg_entry_price, group_name
                          FROM positions
                         WHERE symbol = :symbol AND contract_code = :contract
                         FOR UPDATE
                        """
                    ),
                    {"symbol": symbol, "contract": old_contract},
                )
            ).first()
            if row is None:
                raise ValueError(
                    f"positions 中无 symbol={symbol} contract_code={old_contract} 的行"
                )
            m = row._mapping
            qty_signed = int(m["qty"])
            group_name = str(m["group_name"])
            target_qty = abs(qty_signed)
            direction = "long" if qty_signed > 0 else "short"

            instr_sql = text(
                """
                INSERT INTO instructions
                  (id, generated_at, session_date, session, symbol, contract_code,
                   action, direction, target_qty, entry_price_ref, stop_loss_ref,
                   group_name, status, created_at, updated_at)
                VALUES
                  (:id, :gen, :sd, :ss, :sym, :ct, :action, :dir, :tq,
                   :epr, NULL, :grp, 'fully_filled', :now, :now)
                """
            )
            await s.execute(
                instr_sql,
                {
                    "id": close_instr_id,
                    "gen": now,
                    "sd": session_date,
                    "ss": session_tag,
                    "sym": symbol,
                    "ct": old_contract,
                    "action": "close",
                    "dir": direction,
                    "tq": target_qty,
                    "epr": dec_str(old_close_price),
                    "grp": group_name,
                    "now": now,
                },
            )
            await s.execute(
                instr_sql,
                {
                    "id": open_instr_id,
                    "gen": now,
                    "sd": session_date,
                    "ss": session_tag,
                    "sym": symbol,
                    "ct": new_contract,
                    "action": "open",
                    "dir": direction,
                    "tq": target_qty,
                    "epr": dec_str(new_open_price),
                    "grp": group_name,
                    "now": now,
                },
            )

            fill_sql = text(
                """
                INSERT INTO fills
                  (id, instruction_id, filled_qty, filled_price, filled_at,
                   trigger_source, note)
                VALUES
                  (:id, :iid, :qty, :price, :at, 'roll_contract', :note)
                """
            )
            await s.execute(
                fill_sql,
                {
                    "id": close_fill_id,
                    "iid": close_instr_id,
                    "qty": target_qty,
                    "price": dec_str(old_close_price),
                    "at": closed_at_final,
                    "note": note,
                },
            )
            await s.execute(
                fill_sql,
                {
                    "id": open_fill_id,
                    "iid": open_instr_id,
                    "qty": target_qty,
                    "price": dec_str(new_open_price),
                    "at": opened_at_final,
                    "note": note,
                },
            )

            await s.execute(
                text(
                    "DELETE FROM positions "
                    "WHERE symbol = :symbol AND contract_code = :contract"
                ),
                {"symbol": symbol, "contract": old_contract},
            )
            await s.execute(
                text(
                    """
                    INSERT INTO positions
                      (symbol, contract_code, qty, avg_entry_price,
                       stop_loss_price, group_name, opened_at, last_updated_at)
                    VALUES
                      (:symbol, :contract, :qty, :avg, NULL, :grp,
                       :opened_at, :now)
                    """
                ),
                {
                    "symbol": symbol,
                    "contract": new_contract,
                    "qty": qty_signed,
                    "avg": dec_str(new_open_price),
                    "grp": group_name,
                    "opened_at": opened_at_final,
                    "now": now,
                },
            )

            from json import dumps as _dumps

            payload = {
                "symbol": symbol,
                "old_contract": old_contract,
                "new_contract": new_contract,
                "qty": qty_signed,
                "old_close_price": str(old_close_price),
                "new_open_price": str(new_open_price),
                "close_instruction_id": close_instr_id,
                "open_instruction_id": open_instr_id,
                "close_fill_id": close_fill_id,
                "open_fill_id": open_fill_id,
                "group_name": group_name,
                "closed_at": closed_at_final.isoformat(),
                "opened_at": opened_at_final.isoformat(),
                "note": note,
            }
            if get_engine().dialect.name == "postgresql":
                alert_sql = text(
                    """
                    INSERT INTO alerts (severity, event_type, message, payload)
                    VALUES (:sev, :et, :msg, CAST(:payload AS JSONB))
                    """
                )
            else:
                alert_sql = text(
                    """
                    INSERT INTO alerts (severity, event_type, message, payload, event_at)
                    VALUES (:sev, :et, :msg, :payload, :now)
                    """
                )
            await s.execute(
                alert_sql,
                {
                    "sev": "info",
                    "et": "contract_rolled",
                    "msg": f"{symbol} {old_contract}→{new_contract} 换约完成 qty={qty_signed}",
                    "payload": _dumps(payload),
                    "now": now,
                },
            )

        new_position = await get_position(symbol, new_contract)

    return {
        "symbol": symbol,
        "old_contract": old_contract,
        "new_contract": new_contract,
        "qty": qty_signed,
        "old_close_price": old_close_price,
        "new_open_price": new_open_price,
        "closed_instruction_id": close_instr_id,
        "opened_instruction_id": open_instr_id,
        "new_position": new_position,
    }
