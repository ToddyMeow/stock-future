"""live/report_service.py — 日报渲染 + Server 酱推送

流程：
  1. 从 DB 取当日 instructions（含 fills）+ daily_pnl 30 天窗
  2. Jinja2 渲染 live/templates/daily_report.html → HTML 字符串
  3. 写到 live/reports/YYYY-MM-DD.html
  4. 调 alerting.send_wechat 推送摘要（权益 / 今日 PnL / 指令数 / 链接占位）

CLI：python -m live.report_service --date 2025-12-31
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import date as _date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from live.account_state import (
    get_daily_pnl_range,
    get_instructions_with_fills,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Jinja2 环境
# =====================================================================

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_REPORTS_DIR = Path(__file__).parent / "reports"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


# =====================================================================
# 汇总 / 聚合
# =====================================================================


def _summarize_status(instructions: list[dict[str, Any]]) -> dict[str, int]:
    counters = {
        "fully_filled": 0,
        "partially_filled": 0,
        "vetoed": 0,
        "skipped": 0,
        "pending": 0,
        "expired": 0,
    }
    for inst in instructions:
        st = str(inst.get("status", "pending"))
        counters[st] = counters.get(st, 0) + 1
    return counters


# =====================================================================
# 主渲染函数
# =====================================================================


async def render_daily_report(report_date: _date) -> str:
    """拉数据、渲染 HTML、返回字符串（不落盘、不推送）。"""
    # 1. 当日指令
    try:
        instructions = await get_instructions_with_fills(report_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[report] get_instructions_with_fills 异常: %s；降级空列表", exc)
        instructions = []

    # 2. 最近 30 日 daily_pnl
    from datetime import timedelta
    start = report_date - timedelta(days=30)
    try:
        pnl_rows = await get_daily_pnl_range(start, report_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[report] get_daily_pnl_range 异常: %s；降级空序列", exc)
        pnl_rows = []

    # KPI
    today_pnl_row = None
    for r in pnl_rows:
        if r.date == report_date:
            today_pnl_row = r
            break

    equity = Decimal("0")
    pnl_today = Decimal("0")
    drawdown = Decimal("0")
    if today_pnl_row is not None:
        equity = today_pnl_row.equity
        pnl_today = (
            today_pnl_row.realized_pnl_today + today_pnl_row.unrealized_pnl_today
        )
        if today_pnl_row.drawdown_from_peak is not None:
            drawdown = today_pnl_row.drawdown_from_peak

    pnl_30d = sum(
        (r.realized_pnl_today + r.unrealized_pnl_today for r in pnl_rows),
        Decimal("0"),
    )

    # 权益曲线 JS 数据
    equity_series = [
        {"t": str(r.date), "e": float(r.equity)} for r in pnl_rows
    ]

    # 否决明细
    vetoed = [i for i in instructions if str(i.get("status")) == "vetoed"]

    summary = _summarize_status(instructions)

    tpl = _env.get_template("daily_report.html")
    html = tpl.render(
        report_date=str(report_date),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        equity=float(equity),
        pnl_today=float(pnl_today),
        pnl_30d=float(pnl_30d),
        drawdown=float(drawdown),
        instructions=instructions,
        vetoed=vetoed,
        summary=summary,
        equity_series_js=json.dumps(equity_series),
    )
    return html


async def write_and_push_daily_report(report_date: _date) -> Path:
    """渲染 + 写文件 + 推送微信摘要。返回 HTML 路径。"""
    html = await render_daily_report(report_date)
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"{report_date.isoformat()}.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("[report] 写入 %s (%d bytes)", out_path, out_path.stat().st_size)

    # 简短摘要（从 KPI 再拉一次避免重复计算成本）
    try:
        from datetime import timedelta
        from live.account_state import get_daily_pnl_range
        from live.alerting import send_wechat

        rows = await get_daily_pnl_range(report_date - timedelta(days=1), report_date)
        today_row = next((r for r in rows if r.date == report_date), None)
        equity = float(today_row.equity) if today_row else 0.0
        pnl_today = (
            float(today_row.realized_pnl_today + today_row.unrealized_pnl_today)
            if today_row else 0.0
        )
        desp = (
            f"# 日报就绪 · {report_date}\n\n"
            f"- 权益：{equity:.0f}\n"
            f"- 今日 PnL：{pnl_today:+.0f}\n"
            f"- 报告路径：`{out_path}`\n\n"
            f"> 打开 http://localhost:3000/reports/{report_date} 查看详情"
        )
        await send_wechat(title=f"日报 · {report_date}", desp_markdown=desp)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[report] 推送微信摘要失败（HTML 已写）: %s", exc)

    return out_path


# =====================================================================
# CLI
# =====================================================================


def _parse_date(s: str) -> _date:
    return _date.fromisoformat(s)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, type=_parse_date, help="YYYY-MM-DD")
    ap.add_argument(
        "--no-push",
        action="store_true",
        help="只写 HTML，不推送 Server 酱",
    )
    args = ap.parse_args()

    # Q25：非交易日跳过（不生成日报、不推送微信）
    from live.trading_calendar import require_trading_day_or_exit
    require_trading_day_or_exit(args.date, "report_service")

    async def _main() -> None:
        if args.no_push:
            html = await render_daily_report(args.date)
            _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            p = _REPORTS_DIR / f"{args.date.isoformat()}.html"
            p.write_text(html, encoding="utf-8")
            print(f"[report] wrote {p} ({p.stat().st_size} bytes)")
        else:
            p = await write_and_push_daily_report(args.date)
            print(f"[report] wrote {p} ({p.stat().st_size} bytes)")

    asyncio.run(_main())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _cli()


__all__ = ["render_daily_report", "write_and_push_daily_report"]
