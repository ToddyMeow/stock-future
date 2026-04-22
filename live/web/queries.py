from __future__ import annotations

from datetime import date as _date
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text

from live.account_state import get_sessionmaker
from live.config import APP_TIMEZONE
from live.web.schemas import CapitalSnapshot, EngineStateSummary, InstructionsDayCount


async def load_group_combo_map() -> dict[str, str]:
    sql = text("SELECT group_name, combo FROM universe_group_combos")
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql)
        return {row.group_name: row.combo for row in rs}


async def load_universe_base() -> list[dict[str, Any]]:
    sql = text(
        "SELECT symbol, group_name FROM universe_symbols "
        "WHERE enabled = TRUE ORDER BY symbol"
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql)
        return [{"symbol": row.symbol, "group_name": row.group_name} for row in rs]


async def fetch_latest_bar_by_symbol(symbols: list[str]) -> dict[str, dict[str, Any]]:
    if not symbols:
        return {}
    sql = text(
        """
        SELECT DISTINCT ON (symbol)
               symbol, order_book_id, settle, date, contract_multiplier
          FROM bars
         WHERE symbol = ANY(:symbols)
         ORDER BY symbol, date DESC
        """
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql, {"symbols": symbols})
        rows = [dict(r._mapping) for r in rs]

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_date = row.get("date")
        if isinstance(raw_date, str):
            try:
                raw_date = _date.fromisoformat(raw_date)
            except ValueError:
                raw_date = None
        out[str(row["symbol"])] = {
            "contract_code": row.get("order_book_id"),
            "last_settle": row.get("settle"),
            "last_settle_date": raw_date,
            "contract_multiplier": row.get("contract_multiplier"),
        }
    return out


async def fetch_atr20_by_symbol(symbols: list[str]) -> dict[str, Decimal]:
    if not symbols:
        return {}
    sql = text(
        """
        WITH lagged AS (
            SELECT symbol, date, high, low, close,
                   LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close
              FROM bars
             WHERE symbol = ANY(:symbols)
        ),
        ranked AS (
            SELECT symbol, high, low, close, prev_close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
              FROM lagged
             WHERE prev_close IS NOT NULL
        )
        SELECT symbol,
               AVG(GREATEST(high - low, ABS(high - prev_close), ABS(low - prev_close))) AS atr20
          FROM ranked
         WHERE rn <= 20
         GROUP BY symbol
        """
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql, {"symbols": symbols})
        return {
            str(row._mapping["symbol"]): Decimal(str(row._mapping["atr20"]))
            for row in rs
            if row._mapping["atr20"] is not None
        }


async def fetch_positions_by_symbol(symbols: list[str]) -> dict[str, int]:
    if not symbols:
        return {}
    sql = text(
        """
        SELECT symbol, SUM(qty) AS total_qty
          FROM positions
         WHERE symbol = ANY(:symbols)
         GROUP BY symbol
        """
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql, {"symbols": symbols})
        rows = [dict(r._mapping) for r in rs]

    out: dict[str, int] = {}
    for row in rows:
        total = row.get("total_qty")
        if total is not None and int(total) != 0:
            out[str(row["symbol"])] = int(total)
    return out


async def fetch_latest_diagnostics_by_symbol(
    symbols: list[str],
) -> dict[str, dict[str, Any]]:
    if not symbols:
        return {}
    sql = text(
        """
        SELECT DISTINCT ON (symbol)
               symbol, session_date, session,
               entry_trigger, entry_direction, reject_reason, miss_reason
          FROM signal_diagnostics
         WHERE symbol = ANY(:symbols)
         ORDER BY symbol, session_date DESC, session DESC
        """
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql, {"symbols": symbols})
        return {row._mapping["symbol"]: dict(row._mapping) for row in rs}


async def fetch_latest_engine_state() -> EngineStateSummary | None:
    sql = text(
        """
        SELECT session_date, session, state, created_at
          FROM engine_states
         ORDER BY created_at DESC
         LIMIT 1
        """
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql)
        row = rs.first()
    if row is None:
        return None

    mapping = row._mapping
    state = mapping["state"]
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:  # noqa: BLE001
            state = {}

    last_date_raw = state.get("last_date") if isinstance(state, dict) else None
    last_date: _date | None = None
    if isinstance(last_date_raw, str):
        try:
            last_date = _date.fromisoformat(last_date_raw)
        except ValueError:
            last_date = None

    positions = state.get("positions") if isinstance(state, dict) else None
    positions_count = len(positions) if isinstance(positions, (list, dict)) else 0
    try:
        state_bytes = len(json.dumps(state, default=str))
    except Exception:  # noqa: BLE001
        state_bytes = 0

    return EngineStateSummary(
        session_date=mapping["session_date"],
        session=str(mapping["session"]),
        state_last_date=last_date,
        state_positions_count=positions_count,
        state_bytes=state_bytes,
        created_at=mapping["created_at"],
    )


async def fetch_instructions_by_date(days: int = 14) -> list[InstructionsDayCount]:
    tz = ZoneInfo(APP_TIMEZONE)
    today_local = datetime.now(tz).date()
    from_date = today_local - timedelta(days=days - 1)

    sql = text(
        """
        SELECT session_date, status, COUNT(*) AS cnt
          FROM instructions
         WHERE session_date >= :f AND session_date <= :t
         GROUP BY session_date, status
        """
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql, {"f": from_date, "t": today_local})
        rows = [dict(r._mapping) for r in rs]

    buckets: dict[_date, dict[str, int]] = {
        from_date + timedelta(days=i): {} for i in range(days)
    }
    for row in rows:
        raw_date = row["session_date"]
        session_date = raw_date if isinstance(raw_date, _date) else _date.fromisoformat(str(raw_date))
        status = str(row["status"])
        buckets.setdefault(session_date, {})
        buckets[session_date][status] = buckets[session_date].get(status, 0) + int(row["cnt"])

    return [
        InstructionsDayCount(date=day, total=sum(by_status.values()), by_status=by_status)
        for day, by_status in sorted(buckets.items())
    ]


async def fetch_alerts_24h_count() -> dict[str, int]:
    sql = text(
        """
        SELECT severity, COUNT(*) AS cnt
          FROM alerts
         WHERE event_at >= :since
         GROUP BY severity
        """
    )
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql, {"since": since})
        rows = [dict(r._mapping) for r in rs]

    out = {"info": 0, "warn": 0, "critical": 0}
    for row in rows:
        severity = str(row["severity"])
        if severity in out:
            out[severity] = int(row["cnt"])
    return out


async def fetch_capital_snapshot() -> CapitalSnapshot | None:
    sql = text(
        """
        SELECT date, equity, cash, open_positions_mv,
               drawdown_from_peak, peak_equity_to_date
          FROM daily_pnl
         ORDER BY date DESC
         LIMIT 1
        """
    )
    async with get_sessionmaker()() as session:
        rs = await session.execute(sql)
        row = rs.first()
    if row is None:
        return None

    mapping = row._mapping
    return CapitalSnapshot(
        date=mapping["date"],
        equity=Decimal(str(mapping["equity"])),
        cash=Decimal(str(mapping["cash"])),
        open_positions_mv=Decimal(str(mapping["open_positions_mv"])),
        drawdown_from_peak=(
            Decimal(str(mapping["drawdown_from_peak"]))
            if mapping["drawdown_from_peak"] is not None
            else None
        ),
        peak_equity_to_date=(
            Decimal(str(mapping["peak_equity_to_date"]))
            if mapping["peak_equity_to_date"] is not None
            else None
        ),
    )
