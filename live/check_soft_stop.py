"""live/check_soft_stop.py — soft_stop 检查小脚本（launchd 定点跑）

职责：
  1. 读今日 daily_pnl equity → 计算相对 peak 的 drawdown
  2. 超过 SOFT_STOP_PCT（默认 5%）时在 daily_pnl 表 mark soft_stop_triggered
  3. 打印一行摘要（launchd 的 StandardOutPath 会收）

CLI：
  python -m live.check_soft_stop                   # 默认 today（本地日期）
  python -m live.check_soft_stop --date 2025-12-31 # 指定日
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date as _date

from live.soft_stop import check_and_update


def _parse_date(s: str) -> _date:
    return _date.fromisoformat(s)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--date",
        type=_parse_date,
        default=_date.today(),
        help="检查日期 (YYYY-MM-DD)，默认今天",
    )
    args = ap.parse_args()

    # Q5：非交易日 skip
    from live.trading_calendar import require_trading_day_or_exit
    require_trading_day_or_exit(args.date, "soft_stop_check")

    triggered = asyncio.run(check_and_update(args.date))
    status = "TRIGGERED" if triggered else "ok"
    print(f"[soft_stop_check] date={args.date.isoformat()} status={status}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _cli()
