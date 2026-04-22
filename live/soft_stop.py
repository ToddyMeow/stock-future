"""live/soft_stop.py — 日内权益回撤软熔断

规则：
  - 读最近 30 天 daily_pnl，计算 peak equity
  - today drawdown = (peak - equity) / peak
  - >= SOFT_STOP_PCT 就 mark soft_stop_triggered
  - signal_service 跑前查 is_soft_stop_active；True 则过滤 open/add 指令
"""
from __future__ import annotations

import logging
from datetime import date as _date, timedelta
from decimal import Decimal

from sqlalchemy import text

from live.account_state import (
    get_sessionmaker,
    mark_soft_stop_triggered,
)
from live.config import SOFT_STOP_ENABLED, SOFT_STOP_PCT

logger = logging.getLogger(__name__)


async def check_and_update(today: _date) -> bool:
    """读最近 30 天 daily_pnl，计算 drawdown 并在越线时写 soft_stop_triggered。

    返回：是否触发
    """
    if not SOFT_STOP_ENABLED:
        logger.info("[soft_stop] SOFT_STOP_ENABLED=false，跳过 %s 的检查", today)
        return False
    start = today - timedelta(days=30)
    sm = get_sessionmaker()
    async with sm() as s:
        rs = await s.execute(
            text(
                """
                SELECT date, equity
                  FROM daily_pnl
                 WHERE date BETWEEN :f AND :t
                 ORDER BY date
                """
            ),
            {"f": start, "t": today},
        )
        rows = [(r._mapping["date"], Decimal(str(r._mapping["equity"]))) for r in rs]

    if not rows:
        logger.info("[soft_stop] %s 无 daily_pnl 历史数据，视为未触发", today)
        return False

    # today 当日 equity（若尚未入库则视为未触发）
    today_eq: Decimal | None = None
    peak: Decimal = Decimal("0")
    for d, eq in rows:
        if eq > peak:
            peak = eq
        if str(d) == str(today):
            today_eq = eq

    if today_eq is None:
        logger.info("[soft_stop] %s 当日无 equity 记录，未触发", today)
        return False

    if peak <= 0:
        logger.warning("[soft_stop] peak=0，跳过判定")
        return False

    dd = (peak - today_eq) / peak
    threshold = Decimal(str(SOFT_STOP_PCT))
    triggered = dd >= threshold
    logger.info(
        "[soft_stop] %s peak=%s today=%s dd=%.4f threshold=%.4f triggered=%s",
        today,
        peak,
        today_eq,
        float(dd),
        float(threshold),
        triggered,
    )

    if triggered:
        await mark_soft_stop_triggered(today)
    return triggered


async def is_soft_stop_active(session_date: _date) -> bool:
    """signal_service 跑前调用。查当天或昨天 soft_stop_triggered。"""
    if not SOFT_STOP_ENABLED:
        return False
    yesterday = session_date - timedelta(days=1)
    sm = get_sessionmaker()
    async with sm() as s:
        rs = await s.execute(
            text(
                """
                SELECT soft_stop_triggered
                  FROM daily_pnl
                 WHERE date IN (:today, :yday)
                   AND soft_stop_triggered = :true_val
                """
            ),
            {
                "today": session_date,
                "yday": yesterday,
                # sqlite 用 1，PG 用 True；用 1 两边都能匹配（PG boolean 自动 cast）
                "true_val": True,
            },
        )
        row = rs.first()
    return row is not None


__all__ = ["check_and_update", "is_soft_stop_active"]
