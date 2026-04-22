"""Shared DB/session state for account_state."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from live.config import DATABASE_URL_ASYNC

_engine: AsyncEngine = create_async_engine(DATABASE_URL_ASYNC, pool_pre_ping=True, future=True)
_sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
    _engine, expire_on_commit=False
)


def set_engine(engine: AsyncEngine) -> None:
    global _engine, _sessionmaker
    _engine = engine
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)


def get_engine() -> AsyncEngine:
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return _sessionmaker


async def ping_db() -> bool:
    async with _engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True


def dec_str(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, datetime):
        return v
    return v
