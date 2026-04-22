"""live/daily_pnl_settlement.py — 每日 mark-to-market 结算 job（R1 / Q18）

职责：
  盘后按 --date 结算当日 equity / 持仓市值 / 回撤，UPSERT 到 daily_pnl 表。
  soft_stop.check_and_update 以此为数据源判回撤。

业务逻辑（期货保证金交易）：
  - initial_capital = env INITIAL_CAPITAL 或 1_000_000
  - 每个 position 的 unrealized_pnl_total = (settle - avg_entry) * qty * multiplier
    - qty > 0 (long) 时 settle > avg 为正
    - qty < 0 (short) 时 settle < avg 为正（qty 负号自然反转）
  - equity = initial_capital + Σ unrealized_pnl_total + realized_pnl_cumulative
    一期简化：realized_pnl_cumulative 暂设 0（下个迭代从 fills 算）
  - open_positions_mv = Σ |qty| * settle * multiplier（名义市值）
  - cash = equity - open_positions_mv（近似，不考虑 margin lock）
  - drawdown_from_peak = (peak - equity) / peak，peak = max(历史 ∪ today)

数据来源：
  - positions 表：当前持仓
  - bars 表：(contract_code, date) → settle, contract_multiplier
  - 合约映射：positions.contract_code 直接对应 bars.order_book_id
  - Fallback 顺序：
    1) bars 表 (contract_code, date = target_date) → 精确匹配
    2) bars 表 (contract_code, date < target_date) → 最近一个可用日期
    3) 两者都没 → 用 avg_entry_price（unrealized_pnl = 0）+ 警告

CLI：
  python -m live.daily_pnl_settlement --date 2025-12-24
  python -m live.daily_pnl_settlement --date 2025-12-24 --dry-run
  python -m live.daily_pnl_settlement               # 默认 today
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date as _date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psycopg

# 仓库根目录进 sys.path（方便在任意子路径下 python -m 调用）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.config import DATABASE_URL  # noqa: E402


# =====================================================================
# 常量
# =====================================================================
# 默认初始资金 100 万；可通过 env INITIAL_CAPITAL 覆盖
DEFAULT_INITIAL_CAPITAL = Decimal("1000000")


def get_initial_capital() -> Decimal:
    """读 env INITIAL_CAPITAL；不存在或非法时回默认 100 万。"""
    raw = os.environ.get("INITIAL_CAPITAL", "").strip()
    if not raw:
        return DEFAULT_INITIAL_CAPITAL
    try:
        return Decimal(raw)
    except Exception:
        print(f"[daily_pnl_settlement] 警告：INITIAL_CAPITAL={raw!r} 非法，使用默认 1_000_000")
        return DEFAULT_INITIAL_CAPITAL


# =====================================================================
# 纯函数：equity 计算（可单测，不碰 DB）
# =====================================================================


def calc_equity(
    positions: Iterable[Dict[str, Any]],
    settle_map: Dict[str, Dict[str, Decimal]],
    initial_capital: Decimal,
    realized_pnl_cumulative: Decimal = Decimal("0"),
) -> Dict[str, Decimal]:
    """根据 positions + settle_map 计算 equity / open_mv / unrealized。

    参数：
      positions：iterable of dict，至少含
        - contract_code: str
        - qty: int（正=多 负=空）
        - avg_entry_price: Decimal
      settle_map：{contract_code: {"settle": Decimal, "multiplier": Decimal, "source": str}}
        source ∈ {"exact", "fallback_prev", "avg_entry_fallback"}（仅用于日志/note）
      initial_capital：初始资金（Decimal）
      realized_pnl_cumulative：历史累计已实现 pnl（一期默认 0）

    返回 dict：
      equity / cash / open_positions_mv / unrealized_pnl_total / positions_count
    """
    unrealized_total = Decimal("0")
    open_mv = Decimal("0")
    count = 0

    for pos in positions:
        count += 1
        code = pos["contract_code"]
        qty = int(pos["qty"])
        avg = Decimal(str(pos["avg_entry_price"]))

        entry = settle_map.get(code)
        if entry is None:
            # 理论上 settle_map 会兜底到 avg_entry，但调用方若漏填这里也退回避免炸
            settle = avg
            multiplier = Decimal("1")
        else:
            settle = Decimal(str(entry["settle"]))
            multiplier = Decimal(str(entry["multiplier"]))

        # 期货 mark-to-market：(settle - avg) * qty * multiplier
        # qty 正负自动带方向
        unreal = (settle - avg) * Decimal(qty) * multiplier
        unrealized_total += unreal

        # 名义市值：|qty| * settle * multiplier（和方向无关）
        open_mv += Decimal(abs(qty)) * settle * multiplier

    equity = initial_capital + unrealized_total + realized_pnl_cumulative
    cash = equity - open_mv

    return {
        "equity": equity,
        "cash": cash,
        "open_positions_mv": open_mv,
        "unrealized_pnl_total": unrealized_total,
        "positions_count": Decimal(count),
    }


def calc_drawdown(
    today_equity: Decimal, history_peak: Optional[Decimal]
) -> Dict[str, Decimal]:
    """根据 today_equity + 历史 peak 算 drawdown。

    peak = max(history_peak or 0, today_equity)
    drawdown = (peak - equity) / peak，peak=0 时 drawdown=0
    """
    prev_peak = history_peak if history_peak is not None else Decimal("0")
    peak = max(prev_peak, today_equity)
    if peak <= 0:
        return {"peak": Decimal("0"), "drawdown_from_peak": Decimal("0")}
    drawdown = (peak - today_equity) / peak
    return {"peak": peak, "drawdown_from_peak": drawdown}


# =====================================================================
# DB 读：positions / bars settle / 历史 peak / 昨日 equity
# =====================================================================


def fetch_positions(conn: psycopg.Connection) -> List[Dict[str, Any]]:
    """读当前 positions 全量。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT symbol, contract_code, qty, avg_entry_price, group_name
              FROM positions
             ORDER BY group_name, symbol, contract_code
            """
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def fetch_settle_for_positions(
    conn: psycopg.Connection,
    positions: List[Dict[str, Any]],
    target_date: _date,
) -> Dict[str, Dict[str, Any]]:
    """为每个 contract_code 拉 settle + multiplier。

    优先精确匹配 date = target_date；不命中则取 date < target_date 最近一条。
    两者都无 → 用 avg_entry_price + multiplier=1 兜底（caller 会警告）。
    返回 {contract_code: {"settle", "multiplier", "source", "bar_date"}}
    """
    result: Dict[str, Dict[str, Any]] = {}
    if not positions:
        return result

    with conn.cursor() as cur:
        for pos in positions:
            code = pos["contract_code"]
            # 1) 精确匹配 target_date
            cur.execute(
                """
                SELECT date, settle, contract_multiplier
                  FROM bars
                 WHERE order_book_id = %s AND date = %s
                 LIMIT 1
                """,
                (code, target_date),
            )
            row = cur.fetchone()
            if row is not None:
                result[code] = {
                    "settle": row[1],
                    "multiplier": row[2],
                    "source": "exact",
                    "bar_date": row[0],
                }
                continue

            # 2) 回退到最近 date < target_date
            cur.execute(
                """
                SELECT date, settle, contract_multiplier
                  FROM bars
                 WHERE order_book_id = %s AND date < %s
                 ORDER BY date DESC
                 LIMIT 1
                """,
                (code, target_date),
            )
            row = cur.fetchone()
            if row is not None:
                result[code] = {
                    "settle": row[1],
                    "multiplier": row[2],
                    "source": "fallback_prev",
                    "bar_date": row[0],
                }
                print(
                    f"[daily_pnl_settlement] 警告：{code} 在 {target_date} 无 bars 数据，"
                    f"回退到 {row[0]} (settle={row[1]})"
                )
                continue

            # 3) 完全无数据 → avg_entry_price 兜底（unrealized=0）+ 写 alerts 表留痕
            result[code] = {
                "settle": pos["avg_entry_price"],
                "multiplier": Decimal("1"),
                "source": "avg_entry_fallback",
                "bar_date": None,
            }
            msg = (
                f"{code} 在 {target_date} 及之前均无 bars 数据，"
                f"使用 avg_entry_price={pos['avg_entry_price']} 兜底（unrealized=0）"
            )
            print(f"[daily_pnl_settlement] 警告：{msg}")
            # Q32：alerts 表留痕（不推微信避免主力切换过渡期刷屏；运维面板可查）
            try:
                with conn.cursor() as alert_cur:
                    alert_cur.execute(
                        """
                        INSERT INTO alerts (severity, event_type, message, payload)
                        VALUES (%s, %s, %s, %s::jsonb)
                        """,
                        (
                            "warn",
                            "settlement_bars_missing",
                            msg,
                            json.dumps(
                                {
                                    "date": str(target_date),
                                    "contract_code": code,
                                    "symbol": pos.get("symbol"),
                                    "avg_entry_price": str(pos["avg_entry_price"]),
                                    "source": "avg_entry_fallback",
                                }
                            ),
                        ),
                    )
            except Exception as e:
                # alerts 写失败不阻断 settlement（日志有就够）
                print(f"[daily_pnl_settlement] 警告：写 alerts 失败 {e}")
    return result


def fetch_history_peak_equity(
    conn: psycopg.Connection, target_date: _date
) -> Optional[Decimal]:
    """取 daily_pnl 里 date < target_date 的最大 equity（历史 peak）。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MAX(equity) FROM daily_pnl WHERE date < %s
            """,
            (target_date,),
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return Decimal(str(row[0]))


def fetch_yesterday_equity(
    conn: psycopg.Connection, target_date: _date
) -> Optional[Decimal]:
    """取最近一个 date < target_date 的 equity（用于算 unrealized_pnl_today 差额）。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT equity FROM daily_pnl
             WHERE date < %s
             ORDER BY date DESC
             LIMIT 1
            """,
            (target_date,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return Decimal(str(row[0]))


# =====================================================================
# DB 写：UPSERT daily_pnl（不覆盖 soft_stop_triggered）
# =====================================================================


def upsert_daily_pnl(
    conn: psycopg.Connection,
    target_date: _date,
    equity: Decimal,
    cash: Decimal,
    open_mv: Decimal,
    realized_pnl_today: Decimal,
    unrealized_pnl_today: Decimal,
    drawdown_from_peak: Decimal,
    peak_equity: Decimal,
    notes: str,
) -> None:
    """UPSERT daily_pnl 一行。

    ON CONFLICT (date) DO UPDATE — 不更新 soft_stop_triggered 字段
    （该字段由 soft_stop.check_and_update 负责维护）。
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO daily_pnl
              (date, equity, cash, open_positions_mv,
               realized_pnl_today, unrealized_pnl_today,
               drawdown_from_peak, peak_equity_to_date,
               soft_stop_triggered, notes)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s)
            ON CONFLICT (date) DO UPDATE SET
              equity = EXCLUDED.equity,
              cash = EXCLUDED.cash,
              open_positions_mv = EXCLUDED.open_positions_mv,
              realized_pnl_today = EXCLUDED.realized_pnl_today,
              unrealized_pnl_today = EXCLUDED.unrealized_pnl_today,
              drawdown_from_peak = EXCLUDED.drawdown_from_peak,
              peak_equity_to_date = EXCLUDED.peak_equity_to_date,
              notes = EXCLUDED.notes
              -- 注意：soft_stop_triggered 不在这里更新，由 soft_stop 模块维护
            """,
            (
                target_date,
                equity,
                cash,
                open_mv,
                realized_pnl_today,
                unrealized_pnl_today,
                drawdown_from_peak,
                peak_equity,
                notes,
            ),
        )


# =====================================================================
# 主流程
# =====================================================================


def settle(target_date: _date, dry_run: bool = False, dsn: Optional[str] = None) -> Dict[str, Any]:
    """结算 target_date 的 daily_pnl，返回摘要 dict。

    返回字段：
      equity / cash / open_positions_mv / unrealized_pnl_today /
      drawdown_from_peak / peak_equity_to_date / positions_count /
      fallback_count / status
    """
    dsn = dsn or DATABASE_URL
    t0 = time.time()
    initial_capital = get_initial_capital()

    summary: Dict[str, Any] = {
        "date": str(target_date),
        "dry_run": dry_run,
        "initial_capital": str(initial_capital),
    }

    with psycopg.connect(dsn) as conn:
        # 1) 读持仓
        positions = fetch_positions(conn)
        summary["positions_count"] = len(positions)

        # 2) 拉 settle（含 fallback）
        settle_map = fetch_settle_for_positions(conn, positions, target_date)
        fallback_count = sum(
            1 for v in settle_map.values() if v["source"] != "exact"
        )
        summary["fallback_count"] = fallback_count
        # 归类统计
        summary["settle_sources"] = {
            "exact": sum(1 for v in settle_map.values() if v["source"] == "exact"),
            "fallback_prev": sum(
                1 for v in settle_map.values() if v["source"] == "fallback_prev"
            ),
            "avg_entry_fallback": sum(
                1 for v in settle_map.values() if v["source"] == "avg_entry_fallback"
            ),
        }

        # 3) 计算 equity 等
        eq = calc_equity(
            positions=positions,
            settle_map={
                k: {"settle": v["settle"], "multiplier": v["multiplier"]}
                for k, v in settle_map.items()
            },
            initial_capital=initial_capital,
            realized_pnl_cumulative=Decimal("0"),  # 一期简化
        )
        equity = eq["equity"]
        cash = eq["cash"]
        open_mv = eq["open_positions_mv"]

        # 4) drawdown
        history_peak = fetch_history_peak_equity(conn, target_date)
        dd = calc_drawdown(equity, history_peak)
        peak = dd["peak"]
        drawdown = dd["drawdown_from_peak"]

        # 5) unrealized_pnl_today = today_equity - yesterday_equity
        yday_eq = fetch_yesterday_equity(conn, target_date)
        if yday_eq is None:
            # 无昨日记录：用 0；上游可通过 notes 看到
            unrealized_today = Decimal("0")
        else:
            unrealized_today = equity - yday_eq

        realized_today = Decimal("0")  # 一期简化

        # 6) notes
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        fb_note = (
            f" fallback={fallback_count}" if fallback_count > 0 else ""
        )
        notes = f"auto_settled_at={now}{fb_note}"

        # 填 summary
        summary["equity"] = str(equity)
        summary["cash"] = str(cash)
        summary["open_positions_mv"] = str(open_mv)
        summary["unrealized_pnl_today"] = str(unrealized_today)
        summary["realized_pnl_today"] = str(realized_today)
        summary["drawdown_from_peak"] = str(drawdown)
        summary["peak_equity_to_date"] = str(peak)
        summary["history_peak_before_today"] = (
            str(history_peak) if history_peak is not None else None
        )
        summary["yesterday_equity"] = str(yday_eq) if yday_eq is not None else None

        print(
            f"[daily_pnl_settlement] date={target_date} positions={len(positions)} "
            f"equity={equity} open_mv={open_mv} dd={drawdown} peak={peak}"
        )

        if dry_run:
            summary["status"] = "dry_run"
            summary["elapsed_s"] = round(time.time() - t0, 2)
            print("[daily_pnl_settlement] --dry-run：不写 DB")
            return summary

        # 7) UPSERT
        upsert_daily_pnl(
            conn,
            target_date=target_date,
            equity=equity,
            cash=cash,
            open_mv=open_mv,
            realized_pnl_today=realized_today,
            unrealized_pnl_today=unrealized_today,
            drawdown_from_peak=drawdown,
            peak_equity=peak,
            notes=notes,
        )
        conn.commit()
        summary["status"] = "ok"
        print(f"[daily_pnl_settlement] UPSERT daily_pnl 完毕：date={target_date}")

    summary["elapsed_s"] = round(time.time() - t0, 2)
    return summary


# =====================================================================
# CLI
# =====================================================================


def _parse_date(s: str) -> _date:
    return _date.fromisoformat(s)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--date",
        type=_parse_date,
        default=_date.today(),
        help="结算日期 YYYY-MM-DD（默认今天）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只计算、不写 DB")
    args = ap.parse_args()

    # Q5：非交易日 skip
    from live.trading_calendar import require_trading_day_or_exit
    require_trading_day_or_exit(args.date, "daily_pnl_settlement")

    summary = settle(target_date=args.date, dry_run=args.dry_run)
    print("\n[daily_pnl_settlement] 汇总：")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()


__all__ = [
    "calc_equity",
    "calc_drawdown",
    "settle",
    "get_initial_capital",
    "DEFAULT_INITIAL_CAPITAL",
]
