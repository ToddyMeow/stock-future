"""live/trading_calendar.py — 中国期货交易日历薄包装（Q5 + Q25）

复用 data/adapters/trading_calendar.TradingCalendar，暴露 live 模块用的极简 API：
  - is_trading_day(d) -> bool
  - require_trading_day_or_exit(d, module_name) — 非交易日 print + sys.exit(0)

数据源：data/cache/calendar/cn_futures_trading_days.csv（至 2027-12-31）

用途：
  - data_pipeline / signal_service / daily_pnl_settlement / daily_report / check_soft_stop
    入口统一调 require_trading_day_or_exit，非交易日跳过整个 job。
  - launchd 每日照常 trigger，脚本内部判断 → 省去 plist 配 Weekday 的维护。

注：sys.exit(0) 是 "正常退出"，launchd 不会把它当成失败，不会重试。
"""
from __future__ import annotations

import os
import sys
from datetime import date as _date
from typing import Optional

from data.adapters.trading_calendar import TradingCalendar

_calendar: Optional[TradingCalendar] = None


def _get_calendar() -> TradingCalendar:
    """懒加载，避免 import 时就读 CSV。"""
    global _calendar
    if _calendar is None:
        _calendar = TradingCalendar.default()
    return _calendar


def is_trading_day(d: _date) -> bool:
    """是否为中国期货交易日（周末 + 国家法定假日 False）。"""
    return _get_calendar().is_trading_day(d)


def require_trading_day_or_exit(d: _date, module_name: str) -> None:
    """非交易日 → print 一条日志 + sys.exit(0) 正常退出。

    launchd 每日 trigger 的脚本在 main 入口调这个；非交易日立即安静退出，
    不产指令、不生成报告、不写 alerts。

    手动想在非交易日补跑：设环境变量 `LIVE_SKIP_CALENDAR_CHECK=1` 绕过。
      LIVE_SKIP_CALENDAR_CHECK=1 python -m live.daily_pnl_settlement --date 2025-12-27
    """
    if os.environ.get("LIVE_SKIP_CALENDAR_CHECK") == "1":
        return
    if not is_trading_day(d):
        print(f"[{module_name}] {d} 不是交易日（周末 / 节假日），跳过 "
              f"(设 LIVE_SKIP_CALENDAR_CHECK=1 可强制运行)")
        sys.exit(0)


__all__ = ["is_trading_day", "require_trading_day_or_exit"]
