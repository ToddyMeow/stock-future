"""live/signal_service.py — 信号服务（每日 2 次；cron 调用）

核心流程：
  1. 读 final_v3 combos 配置
  2. 幂等：检查 instructions 里是否已跑过该 session_date × session（>0 → short-circuit）
  3. 读当前 positions / engine_states 上次快照
  4. 从 bars 表拉 [end_date - warmup_days, end_date] 的相关 symbols
  5. 构造 EngineConfig + strategies，调 StrategyEngine.run(bars, initial_state=prev)
  6. 过滤 result.pending_entries 到 generated_date == session_date
  7. UPSERT 到 instructions 表 + engine_states 表
  8. 打印汇总

CLI:
  python -m live.signal_service --date 2025-12-31 --session day [--dry-run] [--force]

关于幂等：
  默认看到同一 (session_date, session) 已有 instructions 时直接 short-circuit。
  --force：无脑覆盖（先 DELETE instructions + DELETE engine_states 再跑）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date as _date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

# 确保从仓库根能 import strats / scripts
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strats.engine import StrategyEngine  # noqa: E402
from strats.factory import (  # noqa: E402
    build_engine_config,
)

from live.config import DATABASE_URL, INITIAL_CAPITAL  # noqa: E402
from live.data_pipeline import get_bars_for_engine  # noqa: E402
from live.engine_setup import (  # noqa: E402
    build_engine_cfg_for_live,
    build_strategies_from_combos,
    load_final_v3_combos,
)


# =====================================================================
# 常量
# =====================================================================
# 保守给 1200 天（~3.3 年日历日）；final_v3 含 ind_* 组用的指标链路较长
# （Wilder ATR/ADX × 长周期 + AMA 10/30 + Bollinger 22 + Boll 百分位 60 日）。
# 短 warmup 会让指标尾段收敛不稳，同一 session_date 跑两次可能输出不同。
WARMUP_DAYS = 1200


# =====================================================================
# 幂等 / 状态读写
# =====================================================================


def check_existing_instructions(
    conn: psycopg.Connection, session_date: _date, session: str
) -> int:
    """检查 instructions 里是否已有该 (session_date, session) 的行。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM instructions WHERE session_date=%s AND session=%s",
            (session_date, session),
        )
        (n,) = cur.fetchone()
    return int(n)


def load_prev_engine_state(
    conn: psycopg.Connection, session_date: _date, session: str
) -> Optional[Dict[str, Any]]:
    """读 engine_states 里 session_date 之前（<）最近一条的 state。

    注：严格小于，不取本日自己（本日的状态如果是 force 重跑会覆盖）。
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT state
              FROM engine_states
             WHERE (session_date, session) < (%s, %s)
             ORDER BY session_date DESC, session DESC
             LIMIT 1
            """,
            (session_date, session),
        )
        row = cur.fetchone()
    if row is None:
        return None
    state = row[0]
    # psycopg 会把 JSONB 解析成 dict；极端情况下收到 str 时兜底
    if isinstance(state, str):
        state = json.loads(state)
    return state


# =====================================================================
# DB positions → engine state（方案 B 核心：实盘持仓以 DB 为唯一真相）
# =====================================================================


def _fetch_db_positions(conn: psycopg.Connection) -> List[Dict[str, Any]]:
    """同步版从 positions 表拉当前真实持仓。

    返回 list[dict]，字段对齐 positions 表（symbol/contract_code/qty/
    avg_entry_price/stop_loss_price/group_name/opened_at）。
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT symbol, contract_code, qty, avg_entry_price, stop_loss_price,
                   group_name, opened_at
              FROM positions
             ORDER BY group_name, symbol
            """
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows


def _fetch_current_cash(
    conn: psycopg.Connection, session_date: _date, fallback: float
) -> float:
    """从 daily_pnl 取最新（< session_date）的 cash；缺失回退 fallback。

    daily_pnl 是 T 日收盘后入库的当日权益快照；signal_service 本 session 用
    的"起手 cash"语义上是"session 开始时账户现金"，回测定义下等于上一
    结算日收盘后的 cash。
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT cash FROM daily_pnl
             WHERE date < %s
             ORDER BY date DESC LIMIT 1
            """,
            (session_date,),
        )
        row = cur.fetchone()
    if row is None:
        return float(fallback)
    return float(row[0])


def _build_strategy_id_lookup(combos: pd.DataFrame) -> Dict[str, str]:
    """group_name → strategy_id（假设 final_v3 每组只 1 个 combo；若多于 1
    组合取第一条并 warn）。"""
    lookup: Dict[str, str] = {}
    for _, row in combos.iterrows():
        group = str(row["group"])
        sid = f"{group}_{row['best_combo']}"
        if group in lookup and lookup[group] != sid:
            print(
                f"[signal_service] WARN: group={group} 有多个 combo "
                f"({lookup[group]} vs {sid})，用第一个"
            )
            continue
        lookup[group] = sid
    return lookup


def _db_position_to_engine_state_dict(
    db_row: Dict[str, Any],
    strategy_id: str,
    atr_ref: float,
    contract_multiplier: float,
) -> Dict[str, Any]:
    """把一行 DB positions 行，映射成 engine state 可吃的 position dict。

    缺省字段策略（DB 不存的 engine 内部记忆字段）：
      - entry_estimate / entry_fill = avg_entry_price
      - entry_slippage / entry_commission_per_contract = 0（已经 paid 了）
      - initial_stop / active_stop = stop_loss_price；缺失时用 entry - 2×ATR
      - r_price / r_money = |entry_fill - initial_stop| × [1, qty×multiplier]
      - MFE/MAE/highest_high_since_entry/lowest_low_since_entry = entry_fill
      - completed_bars = 0（粗略；会让 exit 里依赖 completed_bars 的判定
        稍微偏保守，但不影响止损判断）
      - signal_date = entry_date（近似；从 DB 读的 entry_date 就是交易日）
      - metadata = {}；active_stop_series = 单一 seed 行
      - risk_blowout_* = 0（历史；真实 blowout 只在 fill 那一刻有意义）

    权衡说明：MFE/MAE/completed_bars 影响 breakeven_ratchet / profit_target 等
    诊断分支；对纯 ATR trail 止损 / 止损单不会改判。这些字段的历史值在 engine
    从 DB 重建时无法恢复（DB 没存），接受"从今天开始追踪"的语义。
    """
    import pandas as _pd

    qty_signed = int(db_row["qty"])
    direction = 1 if qty_signed > 0 else -1
    qty_abs = abs(qty_signed)
    entry_fill = float(db_row["avg_entry_price"])

    stop = db_row.get("stop_loss_price")
    if stop is None:
        # 兜底：entry ± 2×ATR（与 engine 默认 stop_atr_mult=2 一致）
        initial_stop = entry_fill - direction * 2.0 * float(atr_ref)
    else:
        initial_stop = float(stop)

    r_price = abs(entry_fill - initial_stop)
    r_money = r_price * contract_multiplier * qty_abs

    # opened_at 可能是 TIMESTAMPTZ（datetime）；engine 只要日期
    opened_at = db_row.get("opened_at")
    if opened_at is None:
        entry_date_str = None
    elif isinstance(opened_at, str):
        entry_date_str = opened_at[:10]
    else:
        entry_date_str = _pd.Timestamp(opened_at).strftime("%Y-%m-%d")

    return {
        "symbol": str(db_row["symbol"]),
        "strategy_id": strategy_id,
        "group_name": str(db_row["group_name"]),
        "direction": int(direction),
        "signal_date": entry_date_str,
        "entry_date": entry_date_str,
        "entry_estimate": entry_fill,
        "entry_fill": entry_fill,
        "entry_slippage": 0.0,
        "qty": int(qty_abs),
        "contract_multiplier": float(contract_multiplier),
        "entry_commission_per_contract": 0.0,
        "atr_ref": float(atr_ref),
        "volume": 0.0,
        "open_interest": 0.0,
        "initial_stop": float(initial_stop),
        "active_stop": float(initial_stop),
        "estimated_initial_risk": r_price,
        "estimated_order_risk": r_money,
        "actual_initial_risk": r_price,
        "actual_order_risk": r_money,
        "risk_blowout_vs_estimate": 0.0,
        "risk_blowout_ratio": None,
        "r_price": r_price,
        "r_money": r_money,
        "highest_high_since_entry": entry_fill,
        "lowest_low_since_entry": entry_fill,
        "completed_bars": 0,
        "pending_exit_reason": None,
        "pending_exit_date": None,
        "active_stop_series": [],
        "mfe_price": entry_fill,
        "mae_price": entry_fill,
        "consecutive_fail_count": 0,
        "original_qty": None,
        "qty_shrink_reason": None,
        "profit_target_triggered": False,
        "breakeven_triggered": False,
        "metadata": {},
        "current_contract": str(db_row.get("contract_code") or ""),
        "raw_entry_fill": None,
        "segment_entry_fill": None,
        "realized_segment_pnl": 0.0,
        "roll_cost_accrued": 0.0,
        "rolls_crossed": [],
    }


def _build_initial_state_from_db(
    conn: psycopg.Connection,
    session_date: _date,
    prev_state: Optional[Dict[str, Any]],
    combos: pd.DataFrame,
    bars: pd.DataFrame,
) -> Dict[str, Any]:
    """组装方案 B 的 initial_state：
       - 指标 / rolling window：沿用 prev_state 的 warmup_bars / last_close_by_symbol
       - positions：完全用 DB 真实持仓（override prev_state.positions）
       - cash：取 daily_pnl 最新快照，fallback INITIAL_CAPITAL
       - pending_entries：[]（没成交的历史 pending 过期；避免 stale 残留）
       - last_date：沿用 prev_state（让 engine 只处理 session_date 当日）

    参数 bars：当前已拉的 bars DataFrame，用来为 DB positions 补 atr_ref /
    contract_multiplier 字段。
    """
    # 起手状态骨架：如果 prev_state 存在，保留其指标快照；否则空白。
    # 2026-04-22 改：last_date 往前推 3 天（足够跨周末/节假日），让 engine
    # 重新评估最近几根 bar，result.daily_status 才能包含最新 bar 的诊断
    # （不然 state_last_date=bars_end 时 engine 0 bar 跑，diag 空）。
    # 历史 pending 会被 engine 重产，但 run_signal_service 里有 filter
    # generated_date == session_date 兜底，不会重复写 instructions。
    if prev_state:
        last_d = prev_state.get("last_date")
        rewound = None
        if last_d:
            rewound = (pd.Timestamp(last_d) - pd.Timedelta(days=3)).date()
        state: Dict[str, Any] = {
            "last_date": rewound,
            "last_close_by_symbol": dict(prev_state.get("last_close_by_symbol", {})),
            "last_raw_close_by_symbol": dict(
                prev_state.get("last_raw_close_by_symbol", {})
            ),
            "warmup_bars": list(prev_state.get("warmup_bars", [])),
        }
    else:
        state = {
            "last_date": None,
            "last_close_by_symbol": {},
            "last_raw_close_by_symbol": {},
            "warmup_bars": [],
        }

    # cash：daily_pnl 最新 < session_date；缺失用 INITIAL_CAPITAL
    cash = _fetch_current_cash(conn, session_date, fallback=INITIAL_CAPITAL)
    state["cash"] = float(cash)

    # positions：完全从 DB 重建，忽略 prev_state.positions
    strat_lookup = _build_strategy_id_lookup(combos)
    db_rows = _fetch_db_positions(conn)

    # 为每个持仓拿到 atr_ref + contract_multiplier（从 bars 最新一行查）
    latest_atr: Dict[str, float] = {}
    latest_mult: Dict[str, float] = {}
    if not bars.empty:
        bars_sorted = bars.sort_values(["symbol", "date"])
        last_rows = bars_sorted.groupby("symbol", sort=False).tail(1)
        for _, br in last_rows.iterrows():
            sym = str(br["symbol"])
            # bars 表可能没算 ATR；signal_service 拉的是 raw bar。
            # 粗略 ATR 估算：取最近 20 天 high-low 均值（如果 bars 只 1 天
            # 就用当日 range）。完整 ATR 由 engine 的 prepare 重算，这里
            # 只是给 position 一个合理初始 atr_ref（用于 r_price 计算）。
            sym_bars = bars_sorted[bars_sorted["symbol"] == sym].tail(20)
            if len(sym_bars) > 0:
                tr = (sym_bars["high"] - sym_bars["low"]).abs()
                atr_est = float(tr.mean()) if len(tr) > 0 else float(br["close"]) * 0.02
            else:
                atr_est = float(br["close"]) * 0.02
            latest_atr[sym] = atr_est
            latest_mult[sym] = float(br.get("contract_multiplier") or 1.0)

    engine_positions: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for db_row in db_rows:
        group = str(db_row["group_name"])
        sid = strat_lookup.get(group)
        if sid is None:
            skipped.append(f"{db_row['symbol']}({group})")
            continue
        sym = str(db_row["symbol"])
        engine_positions.append(
            _db_position_to_engine_state_dict(
                db_row=db_row,
                strategy_id=sid,
                atr_ref=latest_atr.get(sym, float(db_row["avg_entry_price"]) * 0.02),
                contract_multiplier=latest_mult.get(sym, 1.0),
            )
        )
    if skipped:
        print(
            f"[signal_service] WARN: {len(skipped)} DB position(s) group 不在 combos: "
            f"{skipped}（无法映射 strategy_id，跳过）"
        )

    state["positions"] = engine_positions
    state["pending_entries"] = []  # 方案 B 明确：历史 pending 不继承，每 session 从零

    return state


def purge_existing(
    conn: psycopg.Connection, session_date: _date, session: str
) -> None:
    """--force 路径：删掉本 session 既有的 instructions + engine_states。"""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM instructions WHERE session_date=%s AND session=%s",
            (session_date, session),
        )
        cur.execute(
            "DELETE FROM engine_states WHERE session_date=%s AND session=%s",
            (session_date, session),
        )


def _check_soft_stop_sync(
    conn: psycopg.Connection, session_date: _date
) -> bool:
    """同步查 daily_pnl：当天或昨天 soft_stop_triggered=TRUE → True。

    注：account_state.is_soft_stop_active 是 async 版本；signal_service 用 psycopg
    同步连接，所以这里直接 SQL 查，避免引入 asyncio.run。

    Q3：若 SOFT_STOP_ENABLED=false 直接 return False（一期先关软熔断）。
    """
    from datetime import timedelta
    from live.config import SOFT_STOP_ENABLED
    if not SOFT_STOP_ENABLED:
        return False
    yesterday = session_date - timedelta(days=1)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM daily_pnl
             WHERE date IN (%s, %s) AND soft_stop_triggered = TRUE
             LIMIT 1
            """,
            (session_date, yesterday),
        )
        row = cur.fetchone()
    return row is not None


# =====================================================================
# 写入
# =====================================================================


def insert_instructions(
    conn: psycopg.Connection,
    pending_df: pd.DataFrame,
    session_date: _date,
    session: str,
) -> int:
    """批量写 pending_entries → instructions 表（唯一约束命中则跳过）。"""
    if pending_df.empty:
        return 0

    sql = """
        INSERT INTO instructions
          (session_date, session, symbol, contract_code,
           action, direction, target_qty,
           entry_price_ref, stop_loss_ref, group_name)
        VALUES
          (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_date, session, symbol, contract_code, action) DO NOTHING
    """
    inserted = 0
    with conn.cursor() as cur:
        for _, row in pending_df.iterrows():
            cur.execute(
                sql,
                (
                    session_date,
                    session,
                    str(row["symbol"]),
                    str(row["contract_code"]),
                    str(row["action"]),
                    str(row["direction"]),
                    int(row["target_qty"]),
                    float(row["entry_price_ref"]) if pd.notna(row.get("entry_price_ref")) else None,
                    float(row["stop_loss_ref"]) if pd.notna(row.get("stop_loss_ref")) else None,
                    str(row["group_name"]),
                ),
            )
            inserted += cur.rowcount
    return inserted


def upsert_engine_state(
    conn: psycopg.Connection,
    session_date: _date,
    session: str,
    state: Dict[str, Any],
) -> None:
    """UPSERT 到 engine_states。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO engine_states (session_date, session, state)
            VALUES (%s, %s, %s)
            ON CONFLICT (session_date, session) DO UPDATE
                SET state = EXCLUDED.state, created_at = NOW()
            """,
            (session_date, session, Jsonb(state)),
        )


def _entry_name_from_combo(strategy_id: str, group_name: str) -> str:
    """strategy_id 形如 'building_double_ma+boll' 或 'ind_AP_double_ma+boll'
    → 把已知 group_name 前缀剥掉得 combo → "+"前半段是 entry。

    注意：group_name 本身可能含下划线（ind_AP / ind_BB），所以不能简单 split。
    """
    prefix = f"{group_name}_"
    combo = strategy_id[len(prefix):] if strategy_id.startswith(prefix) else strategy_id
    return combo.split("+", 1)[0] if "+" in combo else combo


def _compute_miss_reason(
    entry_name: str,
    symbol_bars: pd.DataFrame,
    slot_entry_strategy,
) -> str:
    """对单个 symbol 的 bars 调 entry strategy 的 prepare，生成"为什么没 trigger"的中文解释。

    各 entry 策略产生的 indicator 字段：
      hl_9 / hl_21: channel_high / channel_low
      boll:         boll_upper / boll_lower / boll_ma
      double_ma:    ma_fast / ma_slow
      ama:          ama / er
    """
    if symbol_bars is None or symbol_bars.empty:
        return ""
    try:
        prepared = slot_entry_strategy.prepare_signals(symbol_bars)
    except Exception as e:  # noqa: BLE001
        return f"prepare 失败: {type(e).__name__}"
    if prepared.empty:
        return ""
    row = prepared.iloc[-1]
    close = float(row.get("close") or 0)
    # bars 里没 atr 列（engine 内部才算）— 这里自己算 Wilder ATR-20 作为"σ 距离"单位
    from strats.helpers import wilder_atr as _wilder_atr
    try:
        atr_series = _wilder_atr(
            symbol_bars["high"].astype(float),
            symbol_bars["low"].astype(float),
            symbol_bars["close"].astype(float),
            20,
        )
        atr_val = float(atr_series.iloc[-1])
        atr = atr_val if pd.notna(atr_val) and atr_val > 0 else 0.0
    except Exception:  # noqa: BLE001
        atr = 0.0

    if entry_name in ("hl_9", "hl_21"):
        ch = row.get("channel_high")
        cl = row.get("channel_low")
        try:
            ch_v = float(ch) if pd.notna(ch) else None
            cl_v = float(cl) if pd.notna(cl) else None
        except (TypeError, ValueError):
            return f"{entry_name}: 缺通道数据"
        if ch_v is None or cl_v is None or atr <= 0:
            return f"{entry_name}: 通道未就绪"
        up_gap = (ch_v - close) / atr
        dn_gap = (close - cl_v) / atr
        period = entry_name.split("_")[-1]
        return (f"{entry_name}: close {close:.2f}，前{period}日高 {ch_v:.2f}"
                f"（上破差 {up_gap:+.1f}σ），前{period}日低 {cl_v:.2f}"
                f"（下破差 {dn_gap:+.1f}σ）")

    if entry_name == "boll":
        up = row.get("boll_upper")
        lo = row.get("boll_lower")
        try:
            up_v = float(up) if pd.notna(up) else None
            lo_v = float(lo) if pd.notna(lo) else None
        except (TypeError, ValueError):
            return "boll: 缺带数据"
        if up_v is None or lo_v is None or atr <= 0:
            return "boll: 布林带未就绪"
        up_gap = (up_v - close) / atr
        dn_gap = (close - lo_v) / atr
        return (f"boll: close {close:.2f} 在带内 [{lo_v:.2f}, {up_v:.2f}]，"
                f"破上差 {up_gap:+.1f}σ / 破下差 {dn_gap:+.1f}σ")

    if entry_name == "double_ma":
        fast = row.get("ma_fast")
        slow = row.get("ma_slow")
        try:
            fast_v = float(fast) if pd.notna(fast) else None
            slow_v = float(slow) if pd.notna(slow) else None
        except (TypeError, ValueError):
            return "double_ma: 缺均线数据"
        if fast_v is None or slow_v is None or slow_v == 0:
            return "double_ma: 均线未就绪"
        pct = (fast_v - slow_v) / slow_v * 100
        arrow = "›" if fast_v > slow_v else "‹"
        trend = "多" if fast_v > slow_v else "空"
        return (f"double_ma: 快线 {fast_v:.2f} {arrow} 慢线 {slow_v:.2f} "
                f"差 {pct:+.2f}%（当前{trend}头势，未触发上穿/下穿）")

    if entry_name == "ama":
        ama_v = row.get("ama")
        er_v = row.get("er")
        try:
            ama_f = float(ama_v) if pd.notna(ama_v) else None
            er_f = float(er_v) if pd.notna(er_v) else None
        except (TypeError, ValueError):
            return "ama: 缺 AMA 数据"
        if ama_f is None or er_f is None:
            return "ama: 未就绪"
        diff_pct = (close - ama_f) / ama_f * 100 if ama_f else 0
        return (f"ama: close {close:.2f} vs AMA {ama_f:.2f} 差 {diff_pct:+.2f}%，"
                f"效率比 er={er_f:.3f}")

    return f"未知 entry 类型 '{entry_name}'"


def upsert_signal_diagnostics(
    conn: psycopg.Connection,
    daily_status: pd.DataFrame,
    strat_lookup: Dict[str, str],
    session_date: _date,
    session: str,
    bars: pd.DataFrame = None,
    slots: list = None,
) -> int:
    """把 engine result.daily_status 的最后一 bar 每 symbol 写进 signal_diagnostics。

    目的：让盯盘品种页看到"今天这个品种 engine 是怎么判的" —— 有没有 trigger，
    trigger 了有没有被风控拒，拒了是什么原因。

    miss_reason（新，2026-04-22）：当 entry_trigger=False，独立调用该 slot 的
    entry_strategy.prepare_signals(sym_bars) 算出 indicator 值，生成中文描述。
    因为 daily_status 只含 first strategy 的字段，其他 slot 要独立 prepare。
    """
    if daily_status is None or daily_status.empty:
        return 0
    # 最新一日（理论上 = session_date 的 T-1 close bar）
    last_date = daily_status["date"].max()
    last_bars = daily_status[daily_status["date"] == last_date]

    # slot_by_group：用 strat_lookup 的反向映射（stratety_id → group）稳定反查，
    # 避免 strategy_id 里 group 含下划线（ind_AP / ind_BB）时 split 切错。
    strat_to_group = {sid: grp for grp, sid in strat_lookup.items()}
    slot_by_group = {}
    if slots:
        for s in slots:
            grp = strat_to_group.get(s.strategy_id)
            if grp:
                slot_by_group[grp] = s

    sql = """
        INSERT INTO signal_diagnostics
          (session_date, session, symbol, strategy_id, group_name,
           bar_date, close_price, atr, entry_trigger, entry_direction,
           reject_reason, miss_reason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_date, session, symbol, strategy_id)
        DO UPDATE SET
            group_name      = EXCLUDED.group_name,
            bar_date        = EXCLUDED.bar_date,
            close_price     = EXCLUDED.close_price,
            atr             = EXCLUDED.atr,
            entry_trigger   = EXCLUDED.entry_trigger,
            entry_direction = EXCLUDED.entry_direction,
            reject_reason   = EXCLUDED.reject_reason,
            miss_reason     = EXCLUDED.miss_reason,
            created_at      = NOW()
    """
    n = 0
    with conn.cursor() as cur:
        for _, r in last_bars.iterrows():
            grp = str(r.get("group_name") or "")
            sid = strat_lookup.get(grp)
            if not sid:
                continue
            sym = str(r["symbol"])
            bar_dt = r.get("date")
            close_v = r.get("close")
            atr_v = r.get("atr")
            trig = bool(r.get("entry_trigger_pass", False))
            direc = r.get("entry_direction")
            rej = r.get("risk_reject_reason")

            # miss_reason: 只在未 trigger 时算
            miss = None
            if not trig and bars is not None and slot_by_group:
                slot = slot_by_group.get(grp)
                if slot is not None:
                    sym_bars = bars[bars["symbol"] == sym].sort_values("date")
                    if not sym_bars.empty:
                        entry_name = _entry_name_from_combo(sid, grp)
                        miss = _compute_miss_reason(entry_name, sym_bars, slot.entry_strategy)

            cur.execute(
                sql,
                (
                    session_date, session, sym, sid, grp,
                    pd.to_datetime(bar_dt).date() if pd.notna(bar_dt) else None,
                    float(close_v) if pd.notna(close_v) else None,
                    float(atr_v) if pd.notna(atr_v) else None,
                    trig,
                    int(direc) if pd.notna(direc) else None,
                    str(rej) if pd.notna(rej) and str(rej).strip() else None,
                    miss,
                ),
            )
            n += 1
    return n


# =====================================================================
# 主流程
# =====================================================================


def run_signal_service(
    session_date: _date,
    session: str,
    dry_run: bool = False,
    force: bool = False,
    dsn: Optional[str] = None,
) -> Dict[str, Any]:
    """跑一次信号服务。返回摘要 dict 供调用方打印 / 日志。"""
    dsn = dsn or DATABASE_URL
    assert session in ("day", "night"), f"session 必须是 'day'/'night'，收到 {session!r}"

    t0 = time.time()
    summary: Dict[str, Any] = {
        "session_date": str(session_date),
        "session": session,
        "dry_run": dry_run,
        "force": force,
    }

    # --------- 1. 加载 combos ---------
    combos = load_final_v3_combos()
    groups = sorted(combos["group"].unique().tolist())
    print(f"[signal_service] combos: {len(combos)} slot(s) · groups={groups}")
    summary["combos"] = len(combos)
    summary["groups"] = groups

    # --------- 2. 连云 PG，做幂等检查 + 读上次状态 ---------
    with psycopg.connect(dsn) as conn:
        existing = check_existing_instructions(conn, session_date, session)
        if existing > 0 and not force:
            print(
                f"[signal_service] already run: {existing} instructions exist for "
                f"({session_date}, {session}) — short-circuit (use --force to override)"
            )
            summary["status"] = "short_circuit"
            summary["existing_count"] = existing
            summary["elapsed_s"] = round(time.time() - t0, 2)
            return summary

        if force and existing > 0:
            print(f"[signal_service] --force: purging {existing} existing instructions")
            purge_existing(conn, session_date, session)
            conn.commit()

        prev_state = load_prev_engine_state(conn, session_date, session)
        if prev_state is None:
            print("[signal_service] 无先前 engine_state，冷启动（无指标快照）")
        else:
            last_d = prev_state.get("last_date")
            print(f"[signal_service] 加载先前 engine_state 的指标快照（last_date={last_d}）")
        summary["prev_state_last_date"] = prev_state.get("last_date") if prev_state else None

        # --------- 3. 从 bars 表拉数据（groups 对应的全部 symbols） ---------
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT symbol FROM bars WHERE group_name = ANY(%s) ORDER BY symbol",
                (groups,),
            )
            symbols = [r[0] for r in cur.fetchall()]

        print(f"[signal_service] symbols ({len(symbols)}): {symbols}")
        summary["symbols"] = symbols

        bars = get_bars_for_engine(
            symbols=symbols,
            end_date=session_date,
            warmup_days=WARMUP_DAYS,
            dsn=dsn,
        )
        if bars.empty:
            raise RuntimeError(
                f"No bars for symbols {symbols[:3]}... in "
                f"[{session_date} - {WARMUP_DAYS}d, {session_date}]"
            )
        # OHLC clamp —— hab_bars 偶尔有越界；engine 的 _validate_input_values 严格
        bars["high"] = bars[["high", "open", "close"]].max(axis=1)
        bars["low"] = bars[["low", "open", "close"]].min(axis=1)
        print(
            f"[signal_service] bars: {len(bars)} rows  "
            f"{bars['date'].min().date()} → {bars['date'].max().date()}  "
            f"{bars['symbol'].nunique()} symbols"
        )
        summary["bars_rows"] = len(bars)
        summary["bars_start"] = str(bars["date"].min().date())
        summary["bars_end"] = str(bars["date"].max().date())

        # --------- 3b. 方案 B 核心：initial_state 用 DB 真实持仓 override ---------
        initial_state = _build_initial_state_from_db(
            conn=conn,
            session_date=session_date,
            prev_state=prev_state,
            combos=combos,
            bars=bars,
        )
        summary["db_positions_count"] = len(initial_state.get("positions", []))
        summary["initial_cash"] = initial_state.get("cash")
        print(
            f"[signal_service] initial_state 组装：positions(来自 DB)="
            f"{len(initial_state.get('positions', []))} "
            f"cash={initial_state.get('cash'):.2f} "
            f"indicator last_date={initial_state.get('last_date')}"
        )

        # --------- 4. 跑 engine ---------
        slots = build_strategies_from_combos(combos)
        engine_cfg = build_engine_cfg_for_live()
        engine = StrategyEngine(config=engine_cfg, strategies=slots)

        t_engine = time.time()
        result = engine.run(bars, initial_state=initial_state)
        print(f"[signal_service] engine.run: {time.time() - t_engine:.1f}s")

        # --------- 5. 过滤本 session 产出的新 pending_entries ---------
        pe = result.pending_entries
        if pe.empty:
            session_pe = pe
        else:
            pe = pe.copy()
            pe["generated_date"] = pd.to_datetime(pe["generated_date"]).dt.date
            session_pe = pe[pe["generated_date"] == session_date].reset_index(drop=True)

        print(f"[signal_service] 新指令数（generated_date={session_date}）：{len(session_pe)}")
        summary["pending_count"] = len(session_pe)
        if not session_pe.empty:
            summary["pending_symbols"] = sorted(session_pe["symbol"].unique().tolist())

        # --------- 5b. soft_stop 检查：触发则过滤 open/add ---------
        # 查当天或昨天是否有 soft_stop_triggered=TRUE
        soft_stop_active = _check_soft_stop_sync(conn, session_date)
        summary["soft_stop_active"] = soft_stop_active
        if soft_stop_active and not session_pe.empty:
            before = len(session_pe)
            session_pe = session_pe[~session_pe["action"].isin(["open", "add"])].reset_index(drop=True)
            filtered = before - len(session_pe)
            print(
                f"[signal_service] soft_stop_active_filtered_{filtered}_entries "
                f"(保留 close/reduce；剩余 {len(session_pe)} 条)"
            )
            summary["soft_stop_filtered"] = filtered
            summary["pending_count"] = len(session_pe)

        # engine_state 快照。方案 B：每 session 只保存"指标快照"，不存
        # positions/cash/pending_entries —— 下一次 session 时这些会被 DB
        # 真相重建。这样 engine_state 和 DB 不会对不上；持仓永远以 DB 为准。
        full_state = result.save_state()
        indicator_state = {
            "last_date": full_state.get("last_date"),
            "last_close_by_symbol": full_state.get("last_close_by_symbol", {}),
            "last_raw_close_by_symbol": full_state.get("last_raw_close_by_symbol", {}),
            "warmup_bars": full_state.get("warmup_bars", []),
            # positions / cash / pending_entries 故意不存
            "positions": [],
            "cash": None,
            "pending_entries": [],
        }
        state = indicator_state
        state_bytes = len(json.dumps(state, default=str))
        summary["state_bytes"] = state_bytes
        print(f"[signal_service] engine_state 指标快照：{state_bytes} bytes（不含 positions/cash）")

        if dry_run:
            print("[signal_service] --dry-run：不写 DB")
            if not session_pe.empty:
                print(session_pe.to_string(index=False))
            summary["status"] = "dry_run"
            summary["elapsed_s"] = round(time.time() - t0, 2)
            return summary

        # --------- 6. 写 DB ---------
        inserted = insert_instructions(conn, session_pe, session_date, session)
        upsert_engine_state(conn, session_date, session, state)
        strat_lookup_local = _build_strategy_id_lookup(combos)
        diag_n = upsert_signal_diagnostics(
            conn, result.daily_status, strat_lookup_local, session_date, session,
            bars=bars, slots=slots,
        )
        conn.commit()

        print(
            f"[signal_service] 写 DB 完毕：instructions +{inserted} / "
            f"engine_states upsert ({session_date}, {session}) / "
            f"signal_diagnostics upsert {diag_n} 行"
        )
        summary["status"] = "ok"
        summary["inserted"] = inserted
        summary["diagnostics_rows"] = diag_n

    summary["elapsed_s"] = round(time.time() - t0, 2)
    return summary


# =====================================================================
# CLI
# =====================================================================


def _parse_date(s: str) -> _date:
    return _date.fromisoformat(s)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, type=_parse_date, help="session_date (YYYY-MM-DD)")
    ap.add_argument("--session", required=True, choices=["day", "night"])
    ap.add_argument("--dry-run", action="store_true", help="不写 DB，仅打印")
    ap.add_argument("--force", action="store_true", help="覆盖已有 instructions + engine_states")
    args = ap.parse_args()

    # Q5：非交易日 skip（launchd 每日照触发，脚本内部判断）
    from live.trading_calendar import require_trading_day_or_exit
    require_trading_day_or_exit(args.date, "signal_service")

    summary = run_signal_service(
        session_date=args.date,
        session=args.session,
        dry_run=args.dry_run,
        force=args.force,
    )
    print("\n[signal_service] 汇总：")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
