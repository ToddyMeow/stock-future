"""Read-side queries for account_state."""

from __future__ import annotations

from datetime import date as _date
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import text

from live.account_state.db import get_engine, get_sessionmaker
from live.db.models import Alert, DailyPnl, Instruction, Position, RollCandidate


async def get_positions() -> list[Position]:
    sql = text(
        """
        SELECT symbol, contract_code, qty, avg_entry_price, stop_loss_price,
               group_name, opened_at, last_updated_at, notes
          FROM positions
         ORDER BY group_name, symbol
        """
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql)
        return [Position.model_validate(dict(r._mapping)) for r in rs]


async def get_positions_enriched() -> list[Position]:
    sql = text(
        """
        SELECT p.symbol,
               p.contract_code,
               p.qty,
               p.avg_entry_price,
               p.stop_loss_price,
               p.group_name,
               p.opened_at,
               p.last_updated_at,
               p.notes,
               (SELECT b.settle FROM bars b
                 WHERE b.order_book_id = p.contract_code
                 ORDER BY b.date DESC LIMIT 1)               AS last_settle,
               (SELECT b.date FROM bars b
                 WHERE b.order_book_id = p.contract_code
                 ORDER BY b.date DESC LIMIT 1)               AS last_settle_date,
               (SELECT b.contract_multiplier FROM bars b
                 WHERE b.order_book_id = p.contract_code
                 ORDER BY b.date DESC LIMIT 1)               AS contract_multiplier
          FROM positions p
         ORDER BY p.group_name, p.symbol
        """
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql)
        rows = [dict(r._mapping) for r in rs]

    result: list[Position] = []
    for row in rows:
        avg_entry = Decimal(str(row["avg_entry_price"]))
        qty = int(row["qty"])
        raw_settle = row.get("last_settle")
        raw_mult = row.get("contract_multiplier")
        raw_date = row.get("last_settle_date")

        if raw_settle is None:
            last_price: Decimal = avg_entry
            multiplier: Optional[Decimal] = None
            unrealized: Decimal = Decimal("0")
            notional: Optional[Decimal] = None
            last_settle_date: Optional[_date] = None
        else:
            last_price = Decimal(str(raw_settle))
            multiplier = Decimal(str(raw_mult)) if raw_mult is not None else None
            last_settle_date = _date.fromisoformat(raw_date) if isinstance(raw_date, str) else raw_date
            if multiplier is not None:
                unrealized = (last_price - avg_entry) * Decimal(qty) * multiplier
                notional = Decimal(abs(qty)) * last_price * multiplier
            else:
                unrealized = Decimal("0")
                notional = None

        result.append(
            Position(
                symbol=row["symbol"],
                contract_code=row["contract_code"],
                qty=qty,
                avg_entry_price=avg_entry,
                stop_loss_price=(
                    Decimal(str(row["stop_loss_price"]))
                    if row.get("stop_loss_price") is not None
                    else None
                ),
                group_name=row["group_name"],
                opened_at=row["opened_at"],
                last_updated_at=row["last_updated_at"],
                notes=row.get("notes"),
                last_price=last_price,
                last_settle_date=last_settle_date,
                contract_multiplier=multiplier,
                unrealized_pnl=unrealized,
                notional_mv=notional,
            )
        )
    return result


async def get_position(symbol: str, contract_code: str) -> Optional[Position]:
    sql = text(
        """
        SELECT symbol, contract_code, qty, avg_entry_price, stop_loss_price,
               group_name, opened_at, last_updated_at, notes
          FROM positions
         WHERE symbol = :symbol AND contract_code = :contract
        """
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql, {"symbol": symbol, "contract": contract_code})
        row = rs.first()
    if row is None:
        return None
    return Position.model_validate(dict(row._mapping))


async def get_instructions(
    session_date: _date, session: str | None = None
) -> list[Instruction]:
    clauses = "WHERE session_date = :sd"
    params: dict[str, Any] = {"sd": session_date}
    if session is not None:
        clauses += " AND session = :ss"
        params["ss"] = session

    sql = text(
        f"""
        SELECT id, generated_at, session_date, session, symbol, contract_code,
               action, direction, target_qty, entry_price_ref, stop_loss_ref,
               group_name, status, veto_reason, broker_stop_order_id,
               created_at, updated_at
          FROM instructions
          {clauses}
         ORDER BY session_date, session, group_name, symbol
        """
    )
    async with get_sessionmaker()() as s:
        rs = await s.execute(sql, params)
        return [Instruction.model_validate(dict(r._mapping)) for r in rs]


async def get_instructions_with_fills(
    session_date: _date, session: str | None = None
) -> list[dict[str, Any]]:
    clauses = "WHERE session_date = :sd"
    params: dict[str, Any] = {"sd": session_date}
    if session is not None:
        clauses += " AND session = :ss"
        params["ss"] = session

    if get_engine().dialect.name == "postgresql":
        sql = text(
            f"""
            SELECT *
              FROM v_instructions_with_fills
              {clauses}
             ORDER BY session_date, session, group_name, symbol
            """
        )
    else:
        sql = text(
            f"""
            SELECT i.*,
                   COALESCE(SUM(f.filled_qty), 0) AS filled_qty_total,
                   COUNT(f.id) AS fill_count,
                   CASE WHEN COUNT(f.id) = 0 THEN NULL
                        ELSE SUM(f.filled_qty * f.filled_price)
                             / NULLIF(SUM(f.filled_qty), 0) END AS avg_filled_price
              FROM instructions i
              LEFT JOIN fills f ON f.instruction_id = i.id
              {clauses.replace('session_date', 'i.session_date').replace('session ', 'i.session ')}
             GROUP BY i.id
             ORDER BY i.session_date, i.session, i.group_name, i.symbol
            """
        )

    async with get_sessionmaker()() as s:
        rs = await s.execute(sql, params)
        return [dict(r._mapping) for r in rs]


async def get_instruction(id_: UUID) -> Optional[Instruction]:
    sql = text(
        """
        SELECT id, generated_at, session_date, session, symbol, contract_code,
               action, direction, target_qty, entry_price_ref, stop_loss_ref,
               group_name, status, veto_reason, broker_stop_order_id,
               created_at, updated_at
          FROM instructions
         WHERE id = :id
        """
    )
    async with get_sessionmaker()() as s:
        rs = await s.execute(sql, {"id": str(id_)})
        row = rs.first()
    if row is None:
        return None
    return Instruction.model_validate(dict(row._mapping))


async def read_daily_pnl_single(date_: _date) -> DailyPnl:
    sql = text(
        """
        SELECT date, equity, cash, open_positions_mv, realized_pnl_today,
               unrealized_pnl_today, soft_stop_triggered, drawdown_from_peak,
               peak_equity_to_date, notes, created_at, updated_at
          FROM daily_pnl
         WHERE date = :date
        """
    )
    async with get_sessionmaker()() as s:
        rs = await s.execute(sql, {"date": date_})
        row = rs.first()
    if row is None:
        raise ValueError(f"daily_pnl 无 {date_} 行")
    return DailyPnl.model_validate(dict(row._mapping))


async def get_daily_pnl_range(from_: _date, to: _date) -> list[DailyPnl]:
    sql = text(
        """
        SELECT date, equity, cash, open_positions_mv, realized_pnl_today,
               unrealized_pnl_today, soft_stop_triggered, drawdown_from_peak,
               peak_equity_to_date, notes, created_at, updated_at
          FROM daily_pnl
         WHERE date BETWEEN :f AND :t
         ORDER BY date
        """
    )
    async with get_sessionmaker()() as s:
        rs = await s.execute(sql, {"f": from_, "t": to})
        return [DailyPnl.model_validate(dict(r._mapping)) for r in rs]


async def get_latest_bar_date() -> Optional[_date]:
    sql = text("SELECT MAX(date) FROM bars")
    async with get_sessionmaker()() as s:
        rs = await s.execute(sql)
        row = rs.first()
    if row is None or row[0] is None:
        return None
    return _date.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]


async def get_roll_candidates() -> list[RollCandidate]:
    import asyncio as _asyncio

    loop = _asyncio.get_event_loop()
    rows = await loop.run_in_executor(None, _detect_rolls_sync)
    return [RollCandidate.model_validate(r) for r in rows]


def _detect_rolls_sync() -> list[dict[str, Any]]:
    from live.roll_detector import detect_rolls

    return detect_rolls()


async def get_recent_alerts(n: int = 50, severity: str | None = None) -> list[Alert]:
    if severity is None:
        sql = text(
            """
            SELECT id, event_at, severity, event_type, message, payload
              FROM alerts
             ORDER BY event_at DESC
             LIMIT :n
            """
        )
        params: dict[str, Any] = {"n": n}
    else:
        sql = text(
            """
            SELECT id, event_at, severity, event_type, message, payload
              FROM alerts
             WHERE severity = :sev
             ORDER BY event_at DESC
             LIMIT :n
            """
        )
        params = {"n": n, "sev": severity}

    async with get_sessionmaker()() as s:
        rs = await s.execute(sql, params)
        rows = [dict(r._mapping) for r in rs]

    if get_engine().dialect.name != "postgresql":
        import json as _json

        for d in rows:
            if d.get("payload") and isinstance(d["payload"], str):
                d["payload"] = _json.loads(d["payload"])

    return [Alert.model_validate(d) for d in rows]
