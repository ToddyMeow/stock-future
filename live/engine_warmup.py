"""live/engine_warmup.py — 纯 indicator warmup（方案 B 辅助脚本）

目的：从很早的日期一口气喂一整段 bars 给 engine，让 engine 把 ATR / ADX /
rolling 窗口 / last_close_by_symbol / warmup_bars tail 全部积累满，但
**不产生任何虚拟持仓 / 现金变动 / trade**。产出的 engine_state 可当作
signal_service 冷启动时的"干净指标底座"。

CLI:
  python -m live.engine_warmup --until 2026-04-17 [--start 2023-01-01]
                                [--dry-run]

写入行为：
  默认：session_date=<--until>, session='day' 一行 UPSERT 到 engine_states。
  --dry-run：只打印，不入库。

和 signal_service 的关系：
  - signal_service 每次用 engine_warmup 产出的 state 当 indicator 种子
  - positions 永远现取 DB（方案 B）；engine_warmup 不关心持仓
  - 每跑一次 engine_warmup 会 **覆盖** 该 session_date × session 的 state
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strats.engine import StrategyEngine  # noqa: E402

from live.config import DATABASE_URL  # noqa: E402
from live.data_pipeline import get_bars_for_engine  # noqa: E402
from live.engine_setup import (  # noqa: E402
    load_final_v3_combos,
    build_strategies_from_combos,
    build_engine_cfg_for_live,
)
from live.signal_service import (  # noqa: E402
    upsert_engine_state,
)


def run_engine_warmup(
    until_date: _date,
    start_date: _date,
    session: str = "day",
    dry_run: bool = False,
    dsn: Optional[str] = None,
) -> Dict[str, Any]:
    """跑一次纯 indicator warmup。

    参数：
      until_date: warmup 截止（含）。engine 会把 bars 里 date ≤ until_date
                  的段全部拿来累积指标，但不生成任何 position/pending/trade。
      start_date: bars 起始（含）。默认 2023-01-01 足够覆盖 _INCREMENTAL_WARMUP_BARS=500
                  天 + 各策略最长 rolling 窗口。
      session:    写入 engine_states 时的 session 标签（day/night）。
      dry_run:    True 时只跑不写库。
      dsn:        覆盖 DATABASE_URL。

    返回摘要 dict。
    """
    dsn = dsn or DATABASE_URL
    t0 = time.time()
    summary: Dict[str, Any] = {
        "until_date": str(until_date),
        "start_date": str(start_date),
        "session": session,
        "dry_run": dry_run,
    }

    # 1. 加载 combos + 待拉 symbols
    combos = load_final_v3_combos()
    groups = sorted(combos["group"].unique().tolist())
    print(f"[engine_warmup] combos: {len(combos)} slot(s) · groups={groups}")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT symbol FROM bars WHERE group_name = ANY(%s) ORDER BY symbol",
                (groups,),
            )
            symbols = [r[0] for r in cur.fetchall()]
        print(f"[engine_warmup] symbols ({len(symbols)}): {symbols}")

        # 2. 拉 bars（warmup_days 用 start_date 反推）
        warmup_days = (until_date - start_date).days
        if warmup_days < 0:
            raise ValueError(
                f"start_date={start_date} > until_date={until_date}"
            )
        bars = get_bars_for_engine(
            symbols=symbols,
            end_date=until_date,
            warmup_days=warmup_days,
            dsn=dsn,
        )
        if bars.empty:
            raise RuntimeError(
                f"No bars for symbols {symbols[:3]}... in "
                f"[{start_date}, {until_date}]"
            )
        bars["high"] = bars[["high", "open", "close"]].max(axis=1)
        bars["low"] = bars[["low", "open", "close"]].min(axis=1)
        print(
            f"[engine_warmup] bars: {len(bars)} rows  "
            f"{bars['date'].min().date()} → {bars['date'].max().date()}  "
            f"{bars['symbol'].nunique()} symbols"
        )
        summary["bars_rows"] = len(bars)

        # 3. 跑 engine（warmup_until=until_date → 全部 date ≤ until 的段都
        #    只累积指标；没 trade / 没仓 / 没 cash 变化）
        slots = build_strategies_from_combos(combos)
        engine_cfg = build_engine_cfg_for_live()
        engine = StrategyEngine(config=engine_cfg, strategies=slots)

        t_engine = time.time()
        result = engine.run(bars, warmup_until=until_date)
        print(f"[engine_warmup] engine.run: {time.time() - t_engine:.1f}s")

        # 4. 组装只保留指标的 state
        full_state = result.save_state()
        indicator_state: Dict[str, Any] = {
            "last_date": full_state.get("last_date"),
            "last_close_by_symbol": full_state.get("last_close_by_symbol", {}),
            "last_raw_close_by_symbol": full_state.get("last_raw_close_by_symbol", {}),
            "warmup_bars": full_state.get("warmup_bars", []),
            # 方案 B 明确：不存 positions/cash/pending_entries
            "positions": [],
            "cash": None,
            "pending_entries": [],
        }
        state_bytes = len(json.dumps(indicator_state, default=str))
        summary["state_bytes"] = state_bytes
        summary["state_last_date"] = indicator_state["last_date"]
        summary["last_close_count"] = len(indicator_state["last_close_by_symbol"])
        summary["warmup_bars_count"] = len(indicator_state["warmup_bars"])
        print(
            f"[engine_warmup] indicator state: {state_bytes} bytes, "
            f"last_close={summary['last_close_count']} symbols, "
            f"warmup_bars tail={summary['warmup_bars_count']} rows"
        )
        # 产出合理性检查：positions 必须为空（warmup-only 模式下）
        if full_state.get("positions"):
            print(
                f"[engine_warmup] WARN: full_state.positions 非空（"
                f"{len(full_state['positions'])} 条）— 此 warn 不该出现，"
                f"说明 engine warmup_until 分支失效，请检查"
            )
        if dry_run:
            print("[engine_warmup] --dry-run：不写 DB")
            summary["status"] = "dry_run"
            summary["elapsed_s"] = round(time.time() - t0, 2)
            return summary

        # 5. UPSERT 到 engine_states（session_date = until_date）
        upsert_engine_state(conn, until_date, session, indicator_state)
        conn.commit()
        print(
            f"[engine_warmup] UPSERT engine_states ({until_date}, {session}) 完毕"
        )
        summary["status"] = "ok"

    summary["elapsed_s"] = round(time.time() - t0, 2)
    return summary


# =====================================================================
# CLI
# =====================================================================


def _parse_date(s: str) -> _date:
    return _date.fromisoformat(s)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--until", required=True, type=_parse_date,
                    help="warmup 截止日期（含）YYYY-MM-DD")
    ap.add_argument("--start", type=_parse_date, default=_date(2023, 1, 1),
                    help="warmup 起始日期（含），默认 2023-01-01")
    ap.add_argument("--session", choices=["day", "night"], default="day",
                    help="写入 engine_states 的 session 标签，默认 day")
    ap.add_argument("--dry-run", action="store_true", help="不写 DB")
    args = ap.parse_args()

    summary = run_engine_warmup(
        until_date=args.until,
        start_date=args.start,
        session=args.session,
        dry_run=args.dry_run,
    )
    print("\n[engine_warmup] 汇总：")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
