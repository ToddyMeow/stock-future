"""daily_pnl_settlement 单元测试

覆盖纯函数 calc_equity + calc_drawdown 的核心计算逻辑，
不依赖数据库。integration 测试（settle + RDS）标 @pytest.mark.integration 默认跳过。
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal
from pathlib import Path

import pytest

# 仓库根目录进 sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 必须在 import live.* 之前占位 DATABASE_URL，否则 live.config 会 KeyError
os.environ.setdefault(
    "DATABASE_URL", "postgresql://placeholder:placeholder@localhost/placeholder"
)
os.environ.setdefault(
    "DATABASE_URL_ASYNC",
    "postgresql+asyncpg://placeholder:placeholder@localhost/placeholder",
)

from live.daily_pnl_settlement import calc_drawdown, calc_equity  # noqa: E402


# =====================================================================
# calc_equity —— 无持仓
# =====================================================================


def test_empty_positions_equity_equals_initial():
    """空仓：equity=initial_capital，open_mv=0，unrealized=0。"""
    result = calc_equity(
        positions=[],
        settle_map={},
        initial_capital=Decimal("1000000"),
    )
    assert result["equity"] == Decimal("1000000")
    assert result["open_positions_mv"] == Decimal("0")
    assert result["unrealized_pnl_total"] == Decimal("0")
    assert result["cash"] == Decimal("1000000")
    assert result["positions_count"] == Decimal("0")


# =====================================================================
# calc_equity —— 多头盈利
# =====================================================================


def test_long_position_mark_to_market():
    """1 手 AO long @ 2850，settle 2900，multiplier 20
       → unrealized = (2900-2850)*1*20 = 1000；equity = 1_000_000 + 1000 = 1_001_000。
    """
    positions = [
        {
            "contract_code": "AO2505",
            "qty": 1,
            "avg_entry_price": Decimal("2850"),
        }
    ]
    settle_map = {
        "AO2505": {"settle": Decimal("2900"), "multiplier": Decimal("20")}
    }
    result = calc_equity(
        positions=positions,
        settle_map=settle_map,
        initial_capital=Decimal("1000000"),
    )
    # unrealized = (2900-2850) * 1 * 20 = 1000
    assert result["unrealized_pnl_total"] == Decimal("1000")
    assert result["equity"] == Decimal("1001000")
    # open_mv = |1| * 2900 * 20 = 58_000
    assert result["open_positions_mv"] == Decimal("58000")
    # cash = equity - open_mv
    assert result["cash"] == Decimal("1001000") - Decimal("58000")
    assert result["positions_count"] == Decimal("1")


# =====================================================================
# calc_equity —— 空头盈利（qty 负号）
# =====================================================================


def test_short_position_mark_to_market():
    """1 手 JD short @ 3200，settle 3150，multiplier 5
       → unrealized = (3150-3200) * (-1) * 5 = 250（负负得正）
       → equity = 1_000_000 + 250 = 1_000_250。
    """
    positions = [
        {
            "contract_code": "JD2505",
            "qty": -1,
            "avg_entry_price": Decimal("3200"),
        }
    ]
    settle_map = {
        "JD2505": {"settle": Decimal("3150"), "multiplier": Decimal("5")}
    }
    result = calc_equity(
        positions=positions,
        settle_map=settle_map,
        initial_capital=Decimal("1000000"),
    )
    # unrealized = (3150-3200) * (-1) * 5 = 250
    assert result["unrealized_pnl_total"] == Decimal("250")
    assert result["equity"] == Decimal("1000250")
    # open_mv = |−1| * 3150 * 5 = 15_750
    assert result["open_positions_mv"] == Decimal("15750")


# =====================================================================
# calc_equity —— 多品种混合
# =====================================================================


def test_mixed_positions_aggregate():
    """AO long 赚 + JD short 赚 应叠加到 equity。"""
    positions = [
        {
            "contract_code": "AO2505",
            "qty": 1,
            "avg_entry_price": Decimal("2850"),
        },
        {
            "contract_code": "JD2505",
            "qty": -1,
            "avg_entry_price": Decimal("3200"),
        },
    ]
    settle_map = {
        "AO2505": {"settle": Decimal("2900"), "multiplier": Decimal("20")},
        "JD2505": {"settle": Decimal("3150"), "multiplier": Decimal("5")},
    }
    result = calc_equity(
        positions=positions,
        settle_map=settle_map,
        initial_capital=Decimal("1000000"),
    )
    # 1000 + 250 = 1250
    assert result["unrealized_pnl_total"] == Decimal("1250")
    assert result["equity"] == Decimal("1001250")
    # 58000 + 15750 = 73750
    assert result["open_positions_mv"] == Decimal("73750")
    assert result["positions_count"] == Decimal("2")


# =====================================================================
# calc_drawdown
# =====================================================================


def test_drawdown_calc():
    """历史 peak=1_100_000，今日 equity=1_034_000 → drawdown = 0.06。"""
    result = calc_drawdown(
        today_equity=Decimal("1034000"),
        history_peak=Decimal("1100000"),
    )
    assert result["peak"] == Decimal("1100000")
    # (1_100_000 - 1_034_000) / 1_100_000 = 0.06
    assert result["drawdown_from_peak"] == Decimal("0.06")


def test_drawdown_new_peak_today():
    """今日 equity 超过历史 peak → peak 更新为今日，drawdown=0。"""
    result = calc_drawdown(
        today_equity=Decimal("1200000"),
        history_peak=Decimal("1100000"),
    )
    assert result["peak"] == Decimal("1200000")
    assert result["drawdown_from_peak"] == Decimal("0")


def test_drawdown_no_history_peak():
    """首次结算，没有历史 peak → peak = today_equity，drawdown=0。"""
    result = calc_drawdown(
        today_equity=Decimal("1000000"),
        history_peak=None,
    )
    assert result["peak"] == Decimal("1000000")
    assert result["drawdown_from_peak"] == Decimal("0")


def test_drawdown_zero_peak_safe():
    """peak=0 时返回 drawdown=0 而不是除零炸裂。"""
    result = calc_drawdown(
        today_equity=Decimal("0"),
        history_peak=Decimal("0"),
    )
    assert result["peak"] == Decimal("0")
    assert result["drawdown_from_peak"] == Decimal("0")


# =====================================================================
# calc_equity —— avg_entry fallback（settle_map 里缺某合约）
# =====================================================================


def test_missing_settle_uses_avg_entry_fallback():
    """如 settle_map 缺某 contract_code，会用 avg_entry + multiplier=1 兜底
       → unrealized = 0（即 (avg - avg) * qty * 1）。
    """
    positions = [
        {
            "contract_code": "MISS001",
            "qty": 2,
            "avg_entry_price": Decimal("500"),
        }
    ]
    # settle_map 不含 MISS001
    result = calc_equity(
        positions=positions,
        settle_map={},
        initial_capital=Decimal("1000000"),
    )
    assert result["unrealized_pnl_total"] == Decimal("0")
    # equity 不变
    assert result["equity"] == Decimal("1000000")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
