#!/usr/bin/env python
"""
Publish universe config from strategy-research CSV → RDS.

Reads:
  --csv-symbols  data/runs/phase0_250k/final_v3_comparison.csv  (symbol, group)
  --csv-combos   data/runs/phase3/best_combos_stable_final_v3.csv  (group, combo)

Writes (single transaction, TRUNCATE + INSERT):
  universe_group_combos
  universe_symbols

Usage:
  python scripts/publish_universe_to_db.py                      # uses final_v3 defaults
  python scripts/publish_universe_to_db.py \
    --csv-symbols data/runs/phase0_1m/v4_compare.csv \
    --csv-combos  data/runs/phase3/best_combos_stable_v4_1m_riskB.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.config import DATABASE_URL  # noqa: E402
import psycopg  # noqa: E402


DEFAULT_CSV_SYMBOLS = (
    ROOT / "data" / "runs" / "phase0_250k" / "final_v3_comparison.csv"
)
DEFAULT_CSV_COMBOS = (
    ROOT / "data" / "runs" / "phase3" / "best_combos_stable_final_v3.csv"
)


def read_symbols(path: Path) -> list[tuple[str, str]]:
    """Return [(symbol, group_name), ...]."""
    rows: list[tuple[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sym = (r.get("symbol") or "").strip()
            grp = (r.get("group") or r.get("group_name") or "").strip()
            if sym and grp:
                rows.append((sym, grp))
    return rows


def read_combos(path: Path) -> list[tuple[str, str]]:
    """Return [(group_name, combo), ...]."""
    rows: list[tuple[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            grp = (r.get("group") or r.get("group_name") or "").strip()
            combo = (r.get("combo") or r.get("best_combo") or "").strip()
            if grp and combo:
                rows.append((grp, combo))
    return rows


def to_sync_url(async_url: str) -> str:
    """asyncpg URL → psycopg sync URL."""
    return (
        async_url.replace("postgresql+asyncpg://", "postgresql://")
                 .replace("postgres+asyncpg://", "postgresql://")
    )


def publish(symbols: Iterable[tuple[str, str]],
            combos: Iterable[tuple[str, str]],
            dsn: str) -> tuple[int, int]:
    symbols = list(symbols)
    combos = list(combos)
    combo_groups = {g for g, _ in combos}
    symbol_groups = {g for _, g in symbols}
    missing = symbol_groups - combo_groups
    if missing:
        raise RuntimeError(
            f"每个 symbol 的 group 必须在 combos 文件里有条目；缺失: {sorted(missing)}"
        )

    with psycopg.connect(dsn) as conn, conn.transaction():
        with conn.cursor() as cur:
            cur.execute("DELETE FROM universe_symbols")
            cur.execute("DELETE FROM universe_group_combos")
            cur.executemany(
                "INSERT INTO universe_group_combos (group_name, combo) VALUES (%s, %s)",
                combos,
            )
            cur.executemany(
                "INSERT INTO universe_symbols (symbol, group_name, enabled) VALUES (%s, %s, TRUE)",
                symbols,
            )
    return len(combos), len(symbols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-symbols", type=Path, default=DEFAULT_CSV_SYMBOLS)
    ap.add_argument("--csv-combos", type=Path, default=DEFAULT_CSV_COMBOS)
    args = ap.parse_args()

    print(f"symbols CSV: {args.csv_symbols}")
    print(f"combos  CSV: {args.csv_combos}")

    symbols = read_symbols(args.csv_symbols)
    combos = read_combos(args.csv_combos)

    print(f"parsed {len(symbols)} symbols, {len(combos)} group-combo entries")
    print(f"sample symbol: {symbols[0] if symbols else '<empty>'}")
    print(f"sample combo:  {combos[0] if combos else '<empty>'}")

    dsn = to_sync_url(DATABASE_URL)
    n_combos, n_syms = publish(symbols, combos, dsn)
    print(f"\n[OK] published: {n_combos} group-combos, {n_syms} symbols to RDS")


if __name__ == "__main__":
    main()
