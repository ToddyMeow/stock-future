"""live/roll_detector.py — 主力合约切换检测 + 微信告警（Q6）

职责：
  期货品种的主力合约会按月/季切换（dominant contract rolling）。当持仓
  的 contract_code 不等于 bars 表中该 symbol 最新一日的 order_book_id 时，
  说明用户正持有"旧主力"，需要人工"平旧开新"。

  MVP 不自动生成调仓指令；只做检测 + 前端红色 badge + 微信提醒。

检测逻辑：
  对每个 positions 行（符号，合约代码）：
    1) 查 bars 表里相同 symbol 最新一日的 order_book_id（= 当前主力）
    2) 若 positions.contract_code != 当前主力 → 候选换约
  返回每条候选的 symbol / 当前合约 / 新主力合约 / 价格信息。

CLI：
  python -m live.roll_detector                    # 打印 JSON
  python -m live.roll_detector --no-alert         # 仅打印，不发告警
  python -m live.roll_detector --dry-run          # 不写 alerts / 不发微信

节奏：
  launchd com.stockfuture.roll_detector 每交易日 14:30 跑一次（日盘收盘前
  30 分钟，给用户时间手工换约）。非交易日 TradingCalendar.is_trading_day
  直接 short-circuit 退出。

去重：
  同一个 (symbol) 在 alerts 表 24h 内已有同 event_type 的告警 → 跳过，避免
  微信刷屏（主力切换过渡期通常数日内都会报）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import date as _date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import psycopg

# 仓库根目录进 sys.path（方便在任意子路径下 python -m 调用）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.config import DATABASE_URL  # noqa: E402


# =====================================================================
# 纯 SQL（sync psycopg，和 daily_pnl_settlement 风格一致）
# =====================================================================

# 单条查询：对每个持仓联 bars 最新主力；只返回真正需要换约的行
# latest_dominant CTE：按 symbol 分组取 bars.date 最新的 order_book_id
# 外层 JOIN positions，过滤 new_dominant != contract_code
_DETECT_SQL = """
WITH latest_dominant AS (
  SELECT DISTINCT ON (symbol)
         symbol,
         order_book_id AS new_dominant,
         date         AS last_date,
         settle       AS new_settle
    FROM bars
   ORDER BY symbol, date DESC
)
SELECT p.symbol,
       p.contract_code AS current_contract,
       ld.new_dominant,
       ld.last_date,
       ld.new_settle,
       p.avg_entry_price,
       p.qty,
       p.group_name,
       (SELECT settle FROM bars b
          WHERE b.order_book_id = p.contract_code
          ORDER BY date DESC LIMIT 1)           AS current_last_price,
       (SELECT date FROM bars b
          WHERE b.order_book_id = p.contract_code
          ORDER BY date DESC LIMIT 1)           AS current_last_date
  FROM positions p
  JOIN latest_dominant ld ON ld.symbol = p.symbol
 WHERE ld.new_dominant != p.contract_code
 ORDER BY p.symbol, p.contract_code
"""


# =====================================================================
# 核心检测函数（sync）
# =====================================================================


def detect_rolls(dsn: Optional[str] = None) -> list[dict[str, Any]]:
    """查 DB，返回换约候选列表（dict 形式）。

    返回字段（字符串化 Decimal / date，方便 JSON 序列化）：
      symbol / current_contract / new_dominant_contract /
      last_observed_date / current_last_price / new_last_price /
      avg_entry_price / qty / group_name
    """
    dsn = dsn or DATABASE_URL
    result: list[dict[str, Any]] = []

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_DETECT_SQL)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()

    for r in rows:
        d = dict(zip(cols, r))
        # 统一字段名对齐 Pydantic RollCandidate
        result.append(
            {
                "symbol": str(d["symbol"]),
                "current_contract": str(d["current_contract"]),
                "new_dominant_contract": str(d["new_dominant"]),
                "last_observed_date": (
                    d["last_date"].isoformat()
                    if isinstance(d["last_date"], _date) else str(d["last_date"])
                ),
                "current_last_price": (
                    str(d["current_last_price"])
                    if d["current_last_price"] is not None else None
                ),
                "new_last_price": (
                    str(d["new_settle"])
                    if d["new_settle"] is not None else None
                ),
                "avg_entry_price": str(d["avg_entry_price"]),
                "qty": int(d["qty"]),
                "group_name": str(d["group_name"]),
            }
        )
    return result


# =====================================================================
# 去重：alerts 表最近 24h 是否已推过同 symbol 告警
# =====================================================================


def _recent_alert_exists(conn: psycopg.Connection, symbol: str) -> bool:
    """24h 内已存在同 symbol 的 contract_roll_required 告警 → 认为已推。

    用 payload JSONB 字段 'symbol' 做过滤；无此字段的历史告警不会误判。
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM alerts
             WHERE event_type = 'contract_roll_required'
               AND payload->>'symbol' = %s
               AND event_at > NOW() - INTERVAL '24 hours'
             LIMIT 1
            """,
            (symbol,),
        )
        return cur.fetchone() is not None


# =====================================================================
# 告警推送（调 alert_escalate 以走统一路径）
# =====================================================================


async def _escalate_async(
    severity: str, event_type: str, message: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """async wrapper；从 sync CLI 里 asyncio.run 调用。"""
    from live.alerting import alert_escalate

    return await alert_escalate(severity, event_type, message, payload)


def _push_alert_for_candidate(
    candidate: dict[str, Any], dry_run: bool = False
) -> dict[str, Any]:
    """对单条候选推 warn 级告警；返回 {alert_id, wechat, sms} dict。"""
    msg = (
        f"主力合约切换：{candidate['symbol']} "
        f"当前持仓 {candidate['current_contract']} → "
        f"新主力 {candidate['new_dominant_contract']}\n"
        f"数量 {candidate['qty']} 手 · 均价 {candidate['avg_entry_price']}\n"
        f"旧合约最新结算 {candidate.get('current_last_price') or '—'} · "
        f"新合约最新结算 {candidate.get('new_last_price') or '—'}\n"
        f"需要人工操作：平旧开新。"
    )
    if dry_run:
        print(f"[roll_detector] --dry-run 不发告警：{msg}")
        return {"alert_id": None, "wechat": False, "sms": False, "dry_run": True}

    return asyncio.run(
        _escalate_async("warn", "contract_roll_required", msg, candidate)
    )


# =====================================================================
# 主流程：检测 + 去重 + 告警
# =====================================================================


def run(
    *,
    dsn: Optional[str] = None,
    send_alerts: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """检测换约候选；按 24h 去重后对新候选逐条发告警。

    返回摘要：
      candidates_total / alerted / skipped_duplicates / dry_run / elapsed_s
    """
    dsn = dsn or DATABASE_URL
    t0 = time.time()

    candidates = detect_rolls(dsn=dsn)
    alerted: list[dict[str, Any]] = []
    skipped: list[str] = []

    if send_alerts and candidates:
        with psycopg.connect(dsn) as conn:
            for c in candidates:
                if _recent_alert_exists(conn, c["symbol"]):
                    skipped.append(c["symbol"])
                    print(
                        f"[roll_detector] 跳过 {c['symbol']}（24h 内已告警过）"
                    )
                    continue
                res = _push_alert_for_candidate(c, dry_run=dry_run)
                alerted.append(
                    {
                        "symbol": c["symbol"],
                        "current_contract": c["current_contract"],
                        "new_dominant_contract": c["new_dominant_contract"],
                        "alert_result": res,
                    }
                )

    summary = {
        "candidates_total": len(candidates),
        "candidates": candidates,
        "alerted": alerted,
        "skipped_duplicates": skipped,
        "dry_run": dry_run,
        "elapsed_s": round(time.time() - t0, 3),
    }
    return summary


# =====================================================================
# CLI
# =====================================================================


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-alert",
        action="store_true",
        help="只打印候选，不发告警（不写 alerts / 不发微信）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="走告警路径但不真发（打日志；alerts 表仍会写，由 alert_escalate 决定）",
    )
    ap.add_argument(
        "--skip-non-trading",
        action="store_true",
        help="非交易日直接退出（launchd 默认行为）",
    )
    args = ap.parse_args()

    # 非交易日短路（launchd plist 会传 --skip-non-trading）
    if args.skip_non_trading:
        try:
            from data.adapters.trading_calendar import TradingCalendar

            cal = TradingCalendar.default()
            today = datetime.now(timezone.utc).date()
            if not cal.is_trading_day(today):
                print(f"[roll_detector] {today} 非交易日，退出")
                return
        except Exception as exc:  # noqa: BLE001
            # 日历不可达时不阻断（降级继续跑，宁可多报）
            print(f"[roll_detector] 日历读取失败（降级继续）：{exc}")

    summary = run(send_alerts=not args.no_alert, dry_run=args.dry_run)
    print("[roll_detector] 汇总：")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    _cli()


__all__ = [
    "detect_rolls",
    "run",
]
