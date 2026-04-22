"""Write-side repository functions for account_state."""

from __future__ import annotations

from datetime import date as _date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from live.account_state.db import dec_str, get_engine, get_sessionmaker
from live.account_state.queries import read_daily_pnl_single
from live.db.models import Alert, DailyPnl


async def upsert_position(
    symbol: str,
    contract_code: str,
    qty: int,
    avg_price: Decimal,
    stop: Decimal | None,
    group: str,
    direction: str | None = None,
    *,
    session: AsyncSession | None = None,
) -> None:
    del direction
    if qty == 0:
        raise ValueError("qty=0 请用 delete_position")

    now = datetime.now(timezone.utc)
    if get_engine().dialect.name == "postgresql":
        sql = text(
            """
            INSERT INTO positions
              (symbol, contract_code, qty, avg_entry_price, stop_loss_price,
               group_name, opened_at, last_updated_at)
            VALUES
              (:symbol, :contract, :qty, :avg, :stop, :group, :now, :now)
            ON CONFLICT (symbol, contract_code) DO UPDATE SET
              qty = EXCLUDED.qty,
              avg_entry_price = EXCLUDED.avg_entry_price,
              stop_loss_price = EXCLUDED.stop_loss_price,
              group_name = EXCLUDED.group_name,
              last_updated_at = EXCLUDED.last_updated_at
            """
        )
    else:
        sql = text(
            """
            INSERT OR REPLACE INTO positions
              (symbol, contract_code, qty, avg_entry_price, stop_loss_price,
               group_name, opened_at, last_updated_at)
            VALUES
              (:symbol, :contract, :qty, :avg, :stop, :group,
               COALESCE((SELECT opened_at FROM positions
                          WHERE symbol=:symbol AND contract_code=:contract), :now),
               :now)
            """
        )

    params = {
        "symbol": symbol,
        "contract": contract_code,
        "qty": qty,
        "avg": dec_str(avg_price),
        "stop": dec_str(stop) if stop is not None else None,
        "group": group,
        "now": now,
    }

    if session is not None:
        await session.execute(sql, params)
    else:
        async with get_sessionmaker()() as s:
            await s.execute(sql, params)
            await s.commit()


async def delete_position(
    symbol: str,
    contract_code: str,
    *,
    session: AsyncSession | None = None,
) -> None:
    sql = text("DELETE FROM positions WHERE symbol = :symbol AND contract_code = :contract")
    params = {"symbol": symbol, "contract": contract_code}
    if session is not None:
        await session.execute(sql, params)
    else:
        async with get_sessionmaker()() as s:
            await s.execute(sql, params)
            await s.commit()


async def record_daily_pnl(
    date_: _date,
    equity: Decimal,
    cash: Decimal,
    open_mv: Decimal,
    realized: Decimal,
    unrealized: Decimal,
    drawdown_from_peak: Decimal | None = None,
    peak_equity: Decimal | None = None,
    notes: str | None = None,
) -> DailyPnl:
    if get_engine().dialect.name == "postgresql":
        sql = text(
            """
            INSERT INTO daily_pnl
              (date, equity, cash, open_positions_mv, realized_pnl_today,
               unrealized_pnl_today, drawdown_from_peak, peak_equity_to_date, notes)
            VALUES
              (:date, :eq, :cash, :mv, :rp, :up, :dd, :peak, :notes)
            ON CONFLICT (date) DO UPDATE SET
              equity = EXCLUDED.equity,
              cash = EXCLUDED.cash,
              open_positions_mv = EXCLUDED.open_positions_mv,
              realized_pnl_today = EXCLUDED.realized_pnl_today,
              unrealized_pnl_today = EXCLUDED.unrealized_pnl_today,
              drawdown_from_peak = EXCLUDED.drawdown_from_peak,
              peak_equity_to_date = EXCLUDED.peak_equity_to_date,
              notes = EXCLUDED.notes
            """
        )
    else:
        sql = text(
            """
            INSERT OR REPLACE INTO daily_pnl
              (date, equity, cash, open_positions_mv, realized_pnl_today,
               unrealized_pnl_today, drawdown_from_peak, peak_equity_to_date,
               soft_stop_triggered, notes, created_at, updated_at)
            VALUES
              (:date, :eq, :cash, :mv, :rp, :up, :dd, :peak,
               COALESCE((SELECT soft_stop_triggered FROM daily_pnl WHERE date=:date), 0),
               :notes, :now, :now)
            """
        )
    params = {
        "date": date_,
        "eq": dec_str(equity),
        "cash": dec_str(cash),
        "mv": dec_str(open_mv),
        "rp": dec_str(realized),
        "up": dec_str(unrealized),
        "dd": dec_str(drawdown_from_peak) if drawdown_from_peak is not None else None,
        "peak": dec_str(peak_equity) if peak_equity is not None else None,
        "notes": notes,
        "now": datetime.now(timezone.utc),
    }
    async with get_sessionmaker()() as s:
        async with s.begin():
            await s.execute(sql, params)
    return await read_daily_pnl_single(date_)


async def mark_soft_stop_triggered(date_: _date) -> None:
    sql = text(
        """
        UPDATE daily_pnl SET soft_stop_triggered = TRUE, updated_at = :now
         WHERE date = :date
        """
    )
    if get_engine().dialect.name != "postgresql":
        sql = text(
            """
            UPDATE daily_pnl SET soft_stop_triggered = 1, updated_at = :now
             WHERE date = :date
            """
        )
    async with get_sessionmaker()() as s:
        async with s.begin():
            await s.execute(sql, {"date": date_, "now": datetime.now(timezone.utc)})


async def insert_alert(
    severity: str,
    event_type: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> Alert:
    from json import dumps

    if get_engine().dialect.name == "postgresql":
        sql = text(
            """
            INSERT INTO alerts (severity, event_type, message, payload)
            VALUES (:sev, :et, :msg, CAST(:payload AS JSONB))
            RETURNING id, event_at, severity, event_type, message, payload
            """
        )
    else:
        sql = text(
            """
            INSERT INTO alerts (severity, event_type, message, payload, event_at)
            VALUES (:sev, :et, :msg, :payload, :now)
            """
        )

    params = {
        "sev": severity,
        "et": event_type,
        "msg": message,
        "payload": dumps(payload) if payload is not None else None,
        "now": datetime.now(timezone.utc),
    }

    async with get_sessionmaker()() as s:
        async with s.begin():
            if get_engine().dialect.name == "postgresql":
                rs = await s.execute(sql, params)
                row = rs.first()
                return Alert.model_validate(dict(row._mapping))

            await s.execute(sql, params)
            rs = await s.execute(
                text(
                    "SELECT id, event_at, severity, event_type, message, payload "
                    "FROM alerts ORDER BY id DESC LIMIT 1"
                )
            )
            row = rs.first()
            d = dict(row._mapping)
            if d.get("payload") and isinstance(d["payload"], str):
                import json as _json

                d["payload"] = _json.loads(d["payload"])
            return Alert.model_validate(d)
