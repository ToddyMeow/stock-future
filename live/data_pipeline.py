"""live/data_pipeline.py — bars 表的增量写入 + 查询辅助

两种写入模式：
  - append_latest_bar_from_rqdata(target_date, symbols, dsn, dry_run):
      实盘模式。launchd 每日 15:15（日盘收盘后）+ 02:45（夜盘收盘后）触发，
      去 RQData 实时拉当日主力合约 OHLC/settle/volume/oi/limit 并 UPSERT
      到云 PG `bars` 表（ON CONFLICT (order_book_id, date) DO NOTHING）。
  - append_latest_bars_from_csv(csv_path, n_days, dsn):
      开发 / fallback 模式，从 hab_bars.csv 直接读 N 天。
      保留用于回填测试和脱机环境（RQData token 失效时应急）。

读取：
  - get_bars_for_engine(symbols, end_date, warmup_days, dsn):
      从 bars 表把 [end_date - warmup_days, end_date] 的所有指定 symbol 行
      拉成 DataFrame，供 signal_service 喂给 engine.run()。

依赖：psycopg3（同步连接）；rqdatac（实盘必须装）。
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import psycopg

from live.config import DATABASE_URL, RQDATAC_USER, RQDATAC_PASSWORD

# bars 表的 22 列，严格对齐 schema.sql 的列顺序
# （hab_bars.csv 第 20 列是 order_book_id，schema 里 order_book_id 排第 3；
#  这里用一个显式列表控制 INSERT 顺序，两边都靠名字匹配）
BARS_COLUMNS: List[str] = [
    "date",
    "symbol",
    "order_book_id",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "volume",
    "open_interest",
    "contract_multiplier",
    "commission",
    "slippage",
    "group_name",
    "margin_rate",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "settle_raw",
    "limit_up",
    "limit_down",
]

# tradeable_symbols.json 的位置（phase0 产出）
_ROOT = Path(__file__).resolve().parents[1]
TRADEABLE_SYMBOLS_JSON = _ROOT / "data" / "runs" / "phase0" / "tradeable_symbols.json"


# =====================================================================
# 写入：从 RQData 增量上数据（实盘主路径）
# =====================================================================


_COMMISSION_SPECS_JSON = _ROOT / "data" / "cache" / "commission_specs.json"


def _load_tradeable_universe() -> List[str]:
    """返回全 91 symbol universe（来自 commission_specs.json）。

    2026-04-22 改：不再读 data/runs/phase0/tradeable_symbols.json。
    原因：该文件受研究流程 Phase 0 重跑影响会变动（曾被减到 83，
    导致 IC/IF/IM/PD/PT/L_F/PP_F/V_F 在 04-20+ 无数据入库），
    而 live data_pipeline 的职责是"把所有候选 symbol 当日数据都拉进来"，
    不应该被策略研究流程的参数变化耦合。

    commission_specs.json 是静态的商品手续费规格，symbols 集合稳定，
    每个品种进表都要算佣金 → 与 data_pipeline 职责正交。
    """
    import json
    if not _COMMISSION_SPECS_JSON.exists():
        # 严格 fallback：文件缺了就抛错而不是 silently 返回空集
        raise RuntimeError(
            f"commission_specs.json 缺失：{_COMMISSION_SPECS_JSON}。"
            "运行 scripts/fetch_commission_specs.py 重新生成。"
        )
    with _COMMISSION_SPECS_JSON.open("r", encoding="utf-8") as fh:
        specs = json.load(fh)
    return sorted(specs.keys())


def _query_position_symbols(dsn: str) -> List[str]:
    """从 positions 表查当前持仓的 symbol 唯一集合。"""
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM positions")
            rows = cur.fetchall()
    return sorted({r[0] for r in rows if r[0]})


def _resolve_symbols(
    dsn: str,
    user_symbols: Optional[Iterable[str]],
) -> List[str]:
    """合并「持仓 symbol ∪ tradeable universe ∪ 用户参数」去重返回。

    覆盖原则：
      - 持仓必须拉（否则当日止损 / 平仓单拿不到价格）
      - tradeable universe 全拉（方便当日新信号立即开单）
      - user_symbols 作扩展（回填 / 临时 debug 用）
    """
    symbols = set()
    try:
        symbols.update(_query_position_symbols(dsn))
    except Exception as e:
        # 非致命 — 首次启动 positions 表可能为空
        print(f"[data_pipeline] WARN: 读 positions 失败：{e}")
    symbols.update(_load_tradeable_universe())
    if user_symbols:
        symbols.update(s.strip().upper() for s in user_symbols if s.strip())
    return sorted(symbols)


def _init_rqdata() -> None:
    """初始化 rqdatac；无凭证则抛出具体错误。"""
    if not RQDATAC_USER or not RQDATAC_PASSWORD:
        raise RuntimeError(
            "RQDATAC_USER / RQDATAC_PASSWORD 未设置。"
            "请检查 live/.env 或仓库根 .env（两处都找不到时会是空字符串）。"
        )
    try:
        import rqdatac
    except ImportError as e:
        raise RuntimeError(f"rqdatac 未安装：{e}") from e
    rqdatac.init(RQDATAC_USER, RQDATAC_PASSWORD)


def _fetch_dominant_contracts(
    symbols: List[str],
    target_date: _date,
) -> dict:
    """批量查 symbol → 当日主力合约 order_book_id 的映射。

    调用 rqfutures.get_dominant(sym, start=target, end=target) 得 Series。
    返回 dict: {symbol: order_book_id}；查不到的 symbol 省略。
    """
    from rqdatac import futures as rqfutures

    mapping: dict = {}
    date_str = target_date.strftime("%Y-%m-%d")
    for sym in symbols:
        try:
            s = rqfutures.get_dominant(sym, start_date=date_str, end_date=date_str)
        except Exception as e:
            print(f"[data_pipeline] WARN: get_dominant({sym}) 失败：{e}")
            continue
        if s is None or len(s) == 0:
            # 当日该 symbol 无主力（品种停牌 / 未上市 / 非交易日）
            continue
        mapping[sym] = str(s.iloc[-1])
    return mapping


def _fetch_bars_price(
    order_book_ids: List[str],
    target_date: _date,
) -> pd.DataFrame:
    """批量查 order_book_id 的当日 OHLC/settle/volume/oi/limit。

    返回平坦 DataFrame，列含 order_book_id / date / open / high / low /
    close / settlement / volume / open_interest / limit_up / limit_down。
    """
    import rqdatac

    date_str = target_date.strftime("%Y-%m-%d")
    df = rqdatac.get_price(
        order_book_ids=order_book_ids,
        start_date=date_str,
        end_date=date_str,
        frequency="1d",
        fields=[
            "open", "high", "low", "close", "settlement",
            "volume", "open_interest", "limit_up", "limit_down",
        ],
        adjust_type="none",
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    # 单 symbol 时 RQData 可能返回 index 只有 date（不带 order_book_id 层）
    out = df.reset_index()
    if "order_book_id" not in out.columns:
        # 单合约场景，补上
        out["order_book_id"] = order_book_ids[0] if len(order_book_ids) == 1 else None
    # 规范化 date 列
    if "date" not in out.columns and "datetime" in out.columns:
        out = out.rename(columns={"datetime": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def _enrich_and_build_rows(
    price_df: pd.DataFrame,
    sym_to_ob: dict,
) -> List[tuple]:
    """把 RQData 原始行扩成 BARS_COLUMNS 对齐的 22 列 row tuple。

    补齐字段：
      - symbol：反查 sym_to_ob
      - contract_multiplier：rqdatac.instruments(ob).contract_multiplier
      - group_name / commission / slippage / margin_rate：
          来自 data/adapters/futures_static_meta.get_meta(symbol)
          （commission 走 by_volume / by_money 分支，跟 adapter 保持一致）
      - raw 列：实盘单日拉的就是未 roll-adjust 的原始价，直接复用 OHLC/settle
      - limit_up / limit_down：直接用 RQData 返回
    """
    import rqdatac
    from data.adapters.futures_static_meta import get_meta

    # 反查 ob → symbol
    ob_to_sym = {ob: sym for sym, ob in sym_to_ob.items()}

    # 缓存 instruments 信息（一次合约多天回填时能复用；本函数单日不重复查）
    instrument_cache: dict = {}

    rows: List[tuple] = []
    for _, r in price_df.iterrows():
        ob = str(r["order_book_id"])
        symbol = ob_to_sym.get(ob)
        if symbol is None:
            # 极偶然：price_df 里有个 ob 不在 sym_to_ob（不应该发生）
            continue

        # 合约乘数
        if ob not in instrument_cache:
            try:
                ins = rqdatac.instruments(ob)
                instrument_cache[ob] = {
                    "multiplier": float(getattr(ins, "contract_multiplier", 0) or 0),
                    "exchange": str(getattr(ins, "exchange", "") or ""),
                }
            except Exception as e:
                print(f"[data_pipeline] WARN: instruments({ob}) 失败：{e}")
                instrument_cache[ob] = {"multiplier": 0.0, "exchange": ""}
        info = instrument_cache[ob]
        multiplier = info["multiplier"]
        if multiplier <= 0:
            print(f"[data_pipeline] WARN: {ob} contract_multiplier 为 0，跳过此行")
            continue

        # 本地 meta（commission / slippage / group_name / margin_rate）
        meta = get_meta(symbol, exchange=info["exchange"] or None)

        # commission：by_money 需要 rate × close × multiplier
        close_val = float(r.get("close") if pd.notna(r.get("close")) else 0.0)
        if meta.commission_type == "by_money" and meta.commission_rate is not None:
            commission = float(meta.commission_rate) * abs(close_val) * multiplier
        else:
            commission = float(
                meta.commission_rate if meta.commission_rate is not None else meta.commission
            )

        # settle 兜底：RQData 里叫 settlement；偶尔缺失就用 close
        settle_val = r.get("settlement")
        if pd.isna(settle_val):
            settle_val = r.get("close")

        row = (
            r["date"],                              # date
            symbol,                                 # symbol
            ob,                                     # order_book_id
            float(r["open"]),                       # open
            float(r["high"]),                       # high
            float(r["low"]),                        # low
            close_val,                              # close
            float(settle_val) if pd.notna(settle_val) else None,  # settle
            float(r["volume"]) if pd.notna(r["volume"]) else 0.0,  # volume
            float(r["open_interest"]) if pd.notna(r["open_interest"]) else 0.0,  # open_interest
            float(multiplier),                      # contract_multiplier
            float(commission),                      # commission
            float(meta.slippage),                   # slippage
            meta.group_name,                        # group_name
            float(meta.margin_rate),                # margin_rate
            # raw 列：实盘单日取的就是原始价，与 OHLC 一致
            float(r["open"]),                       # open_raw
            float(r["high"]),                       # high_raw
            float(r["low"]),                        # low_raw
            close_val,                              # close_raw
            float(settle_val) if pd.notna(settle_val) else None,  # settle_raw
            float(r["limit_up"]) if pd.notna(r.get("limit_up")) else None,  # limit_up
            float(r["limit_down"]) if pd.notna(r.get("limit_down")) else None,  # limit_down
        )
        rows.append(row)
    return rows


def _upsert_bars(rows: List[tuple], dsn: str) -> int:
    """INSERT ON CONFLICT (order_book_id, date) DO NOTHING；返回真实新增行数。"""
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(BARS_COLUMNS))
    col_list = ", ".join(BARS_COLUMNS)
    sql = (
        f"INSERT INTO bars ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT (order_book_id, date) DO NOTHING"
    )
    inserted = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(sql, row)
                inserted += cur.rowcount
        conn.commit()
    return inserted


def append_latest_bar_from_rqdata(
    target_date: _date,
    symbols: Optional[List[str]] = None,
    dsn: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """从 RQData 拉 target_date 当日主力合约 bar，UPSERT 进 bars 表。

    完整 6-step 流程：
      1. rqdatac.init(user, password) — 从 live.config 读凭证
      2. require_trading_day_or_exit(target_date) — 非交易日直接正常退出
      3. 合并「持仓 ∪ tradeable universe ∪ symbols 参数」得待拉 symbol 集
      4. rqfutures.get_dominant(sym, target_date) → symbol → order_book_id 映射
      5. rqdatac.get_price(obs, target_date) 批量拉 OHLC/settle/volume/oi/limit
      6. get_meta(symbol) + instruments(ob) 补齐 contract_multiplier /
         group_name / commission / slippage / margin_rate → INSERT ON CONFLICT

    参数：
      target_date: 要拉的交易日（通常是当天或昨天）
      symbols:     可选，扩展待拉 symbol 列表（与持仓/universe 合并）
      dsn:         覆盖默认 DATABASE_URL
      dry_run:     True 时只打印前 5 行 + 行数，不写 DB

    返回：
      实际插入行数（dry_run=True 时返回"将插入"的行数）
    """
    dsn = dsn or DATABASE_URL

    # Step 1: RQData 初始化
    t0 = time.time()
    _init_rqdata()

    # Step 2: 交易日检查（非交易日正常退出）
    from live.trading_calendar import require_trading_day_or_exit
    require_trading_day_or_exit(target_date, "data_pipeline.rqdata")

    # Step 3: 汇总待拉 symbol 集合
    target_symbols = _resolve_symbols(dsn, symbols)
    if not target_symbols:
        print(f"[data_pipeline] WARN: {target_date} 无待拉 symbol（持仓空 + universe 文件缺失）")
        return 0
    print(f"[data_pipeline] 目标 symbol 数 = {len(target_symbols)}  日期 = {target_date}")

    # Step 4: 查当日主力合约
    t_dom = time.time()
    sym_to_ob = _fetch_dominant_contracts(target_symbols, target_date)
    print(f"[data_pipeline] get_dominant 命中 {len(sym_to_ob)}/{len(target_symbols)}  "
          f"耗时 {time.time() - t_dom:.2f}s")
    if not sym_to_ob:
        print(f"[data_pipeline] WARN: {target_date} 无任一 symbol 有主力合约，跳过")
        return 0

    # Step 5: 批量拉当日 bar
    t_price = time.time()
    price_df = _fetch_bars_price(list(sym_to_ob.values()), target_date)
    print(f"[data_pipeline] get_price 拉到 {len(price_df)} 行  "
          f"耗时 {time.time() - t_price:.2f}s")
    if price_df.empty:
        print(f"[data_pipeline] WARN: RQData 返回空，{target_date} 可能非交易日或数据未结算")
        return 0

    # Step 6: 扩成 22 列 + UPSERT
    rows = _enrich_and_build_rows(price_df, sym_to_ob)
    print(f"[data_pipeline] 组装完毕，准备写入 {len(rows)} 行  总耗时 {time.time() - t0:.2f}s")

    if dry_run:
        # 打印前 5 行预览（每列简化）
        print("[data_pipeline] --dry-run：不写 DB；前 5 行预览：")
        for row in rows[:5]:
            # 只展示 (date, symbol, order_book_id, open, high, low, close, settle, volume, oi)
            print("  ", row[:10])
        return len(rows)

    inserted = _upsert_bars(rows, dsn)
    print(f"[data_pipeline] 实际新增行数 = {inserted}  （重复 ON CONFLICT 命中不计）")
    return inserted


# =====================================================================
# 写入：从 CSV 增量上数据（开发 / fallback 模式）
# =====================================================================


def append_latest_bars_from_csv(
    csv_path: str,
    n_days: int = 1,
    dsn: Optional[str] = None,
) -> int:
    """从 hab_bars.csv 读最后 n_days 个交易日的所有行，INSERT ON CONFLICT DO NOTHING 进 bars 表。

    参数：
      csv_path: hab_bars.csv 的绝对路径
      n_days:   读最后多少天（按唯一 date 倒序排序后取前 n 个）；
                传 0 → 仍然连通 PG 跑一次空插入（sanity check）
      dsn:      覆盖默认 DATABASE_URL（可选）

    返回：实际新增行数（ON CONFLICT 命中的不计）
    """
    dsn = dsn or DATABASE_URL
    src = Path(csv_path)
    if not src.exists():
        raise FileNotFoundError(f"CSV 不存在：{src}")

    # 读取并切最后 n_days
    df = pd.read_csv(src, parse_dates=["date"])

    if n_days > 0:
        # 取最后 n_days 个不重复交易日的所有行
        latest_dates = sorted(df["date"].unique(), reverse=True)[:n_days]
        df = df[df["date"].isin(latest_dates)].copy()
    elif n_days == 0:
        # 空插入 — 用来 sanity check 连通性（相当于 head(0) 然后走 CONFLICT）
        df = df.iloc[:0].copy()
    else:
        raise ValueError(f"n_days 必须 >= 0，收到 {n_days}")

    # 对齐列顺序：丢掉 CSV 里有但 schema 没有的列（若存在）；缺失列用 None
    for c in BARS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[BARS_COLUMNS].copy()

    # date 转字符串（psycopg 能直接吃 date，但统一 ISO 比较稳）
    df["date"] = pd.to_datetime(df["date"]).dt.date

    if df.empty:
        # 仍然开连接跑 noop，验证 DSN 可达
        with psycopg.connect(dsn) as _conn:
            pass
        return 0

    # executemany + ON CONFLICT DO NOTHING（bars 主键 (order_book_id, date)）
    placeholders = ", ".join(["%s"] * len(BARS_COLUMNS))
    col_list = ", ".join(BARS_COLUMNS)
    sql = (
        f"INSERT INTO bars ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT (order_book_id, date) DO NOTHING"
    )

    # 将 NaN 转 None（psycopg 不认 pandas.NaN）
    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]

    inserted = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # 用 rowcount 累加（executemany 报的是最后一次，不可靠）；
            # 改用单条 execute 累加 cur.rowcount
            for r in rows:
                cur.execute(sql, r)
                inserted += cur.rowcount
        conn.commit()
    return inserted


# =====================================================================
# 读取：为 engine.run() 准备 bars DataFrame
# =====================================================================


def get_bars_for_engine(
    symbols: Iterable[str],
    end_date: _date,
    warmup_days: int = 60,
    dsn: Optional[str] = None,
) -> pd.DataFrame:
    """从 bars 表拉 [end_date - warmup_days, end_date] 所有指定 symbol 的行。

    参数：
      symbols:      品种代码列表，如 ["RB", "CU"]
      end_date:     查询末端（含）
      warmup_days:  往前回溯多少天（含 end_date 自身）
      dsn:          覆盖默认 DATABASE_URL

    返回：
      pd.DataFrame，22 列对齐 BARS_COLUMNS，date 列为 pd.Timestamp。
      按 (symbol, date) 升序。
    """
    dsn = dsn or DATABASE_URL
    symbols = list(symbols)
    if not symbols:
        return pd.DataFrame(columns=BARS_COLUMNS + ["created_at"])

    start_date = end_date - timedelta(days=warmup_days)
    # 组装 query — symbol IN (...) 用参数数组
    sql = f"""
        SELECT {", ".join(BARS_COLUMNS)}
        FROM bars
        WHERE date BETWEEN %s AND %s
          AND symbol = ANY(%s)
        ORDER BY symbol, date
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (start_date, end_date, symbols))
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    # date 转 Timestamp（engine.prepare_data 期望 datetime）
    df["date"] = pd.to_datetime(df["date"])
    # numeric 列转 float（psycopg 默认 Decimal，engine 会转 float 但明确更干净）
    numeric_cols = [
        "open", "high", "low", "close", "settle", "volume", "open_interest",
        "contract_multiplier", "commission", "slippage", "margin_rate",
        "open_raw", "high_raw", "low_raw", "close_raw", "settle_raw",
        "limit_up", "limit_down",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =====================================================================
# CLI
#   新模式：python -m live.data_pipeline --mode rqdata [--date YYYY-MM-DD]
#                                       [--symbols RB,CU] [--dry-run]
#   旧模式：python -m live.data_pipeline --mode csv --csv ... --n-days 1
# =====================================================================


def _parse_date(s: str) -> _date:
    return _date.fromisoformat(s)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["rqdata", "csv"],
        default="rqdata",
        help="拉 bar 数据源：rqdata = 线上 RQData（实盘默认）；csv = 本地 hab_bars.csv（fallback / 测试）",
    )
    # rqdata 模式参数
    ap.add_argument(
        "--date",
        type=_parse_date,
        default=None,
        help="rqdata 模式专用；目标交易日 YYYY-MM-DD。默认 = 今天（date.today()）",
    )
    ap.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="rqdata 模式专用；逗号分隔的 symbol 扩展列表（与持仓 ∪ universe 取并集）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="rqdata 模式专用；仅打印前 5 行 + 总行数，不写 DB",
    )
    # csv 模式参数
    ap.add_argument(
        "--csv",
        default="data/cache/normalized/hab_bars.csv",
        help="csv 模式专用；hab_bars.csv 路径（相对仓库根或绝对路径）",
    )
    ap.add_argument(
        "--n-days",
        type=int,
        default=0,
        help="csv 模式专用；上传最后 N 天（0 = 仅连通测试）",
    )
    args = ap.parse_args()

    if args.mode == "rqdata":
        target = args.date or _date.today()
        symbols = None
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

        print(f"[data_pipeline] mode=rqdata  date={target}  "
              f"symbols={symbols or '(持仓∪universe)'}  dry_run={args.dry_run}")
        t0 = time.time()
        try:
            inserted = append_latest_bar_from_rqdata(
                target_date=target,
                symbols=symbols,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"[data_pipeline] ERROR: {type(e).__name__}: {e}")
            # TODO：接入 alerting 后，这里写 alerts (severity=critical, event_type=data_fetch_failed)
            raise
        elapsed = time.time() - t0
        mode_tag = "dry-run 预计" if args.dry_run else "实际"
        print(f"[data_pipeline] {mode_tag}新增 = {inserted}  总耗时 {elapsed:.2f}s")
        # TODO：接入 alerting，把 elapsed 写 alerts (severity=info,
        #   event_type=data_pipeline_latency)；>30s 升 warn，>60s 升 critical
        return

    # ----- csv mode -----
    # Q5：非交易日 skip（--n-days 0 的连通测试也一并跳过）
    from live.trading_calendar import require_trading_day_or_exit
    require_trading_day_or_exit(_date.today(), "data_pipeline.csv")

    csv_path = args.csv
    if not Path(csv_path).is_absolute():
        # 默认按仓库根解析
        repo_root = Path(__file__).resolve().parents[1]
        csv_path = str(repo_root / csv_path)

    print(f"[data_pipeline] mode=csv  CSV={csv_path}  n_days={args.n_days}")
    t0 = time.time()
    inserted = append_latest_bars_from_csv(csv_path, n_days=args.n_days)
    elapsed = time.time() - t0
    print(f"[data_pipeline] 实际新增行数 = {inserted}  耗时 {elapsed:.2f}s")


if __name__ == "__main__":
    _cli()
