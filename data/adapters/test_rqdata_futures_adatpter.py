from __future__ import annotations

import pandas as pd
import pytest

from datetime import date

from data.adapters.futures_static_meta import get_meta
from data.adapters.rqdata_futures_adatpter import (
    RQDataFuturesResearchAdapter,
    RQSymbolSpec,
)
from data.adapters.trading_calendar import TradingCalendar


def test_get_meta_falls_back_for_unknown_underlying() -> None:
    meta = get_meta("ZZ", exchange="CFFEX", product="Index")

    assert meta.commission == 5.0
    assert meta.slippage == 1.0
    assert meta.group_name == "index"


def test_normalize_one_uses_dataframe_contract_multiplier() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [10],
            "open_interest": [100],
            "contract_multiplier": [20],
        }
    )
    adapter = RQDataFuturesResearchAdapter()

    normalized = adapter.normalize_one(
        rq_df=df,
        symbol_spec=RQSymbolSpec(
            rq_symbol="ZZ_DOM",
            underlying_symbol="ZZ",
            strategy_symbol="ZZ",
            exchange="CFFEX",
            product="Index",
        ),
    )

    assert normalized.loc[0, "contract_multiplier"] == 20.0
    assert normalized.loc[0, "group_name"] == "index"


def test_normalize_one_falls_back_to_symbol_spec_multiplier() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "open": [3.0],
            "high": [3.1],
            "low": [2.9],
            "close": [3.05],
            "volume": [30],
            "open_interest": [300],
        }
    )
    adapter = RQDataFuturesResearchAdapter(default_slippage_override=2.5)

    normalized = adapter.normalize_one(
        rq_df=df,
        symbol_spec=RQSymbolSpec(
            rq_symbol="CU_DOM",
            underlying_symbol="CU",
            strategy_symbol="CU",
            contract_multiplier=5,
        ),
    )

    assert normalized.loc[0, "contract_multiplier"] == 5.0
    # Commission is now computed per-row from RQData's by_money rate for CU
    # (~5.5e-5 × close × multiplier). The exact value depends on the cached
    # commission spec; we only assert it's positive and finite here.
    assert normalized.loc[0, "commission"] > 0
    assert pd.notna(normalized.loc[0, "commission"])
    assert normalized.loc[0, "slippage"] == 2.5


def test_normalize_one_raises_when_date_not_a_trading_day() -> None:
    # 2024-02-13 falls inside Spring Festival holiday (non-trading).
    cal = TradingCalendar([date(2024, 2, 8), date(2024, 2, 19)])
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-13"]),
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [10],
            "open_interest": [100],
            "contract_multiplier": [5],
        }
    )
    adapter = RQDataFuturesResearchAdapter(calendar=cal)
    with pytest.raises(ValueError) as exc:
        adapter.normalize_one(
            rq_df=df,
            symbol_spec=RQSymbolSpec(
                rq_symbol="CU_DOM",
                underlying_symbol="CU",
                strategy_symbol="CU",
            ),
        )
    msg = str(exc.value)
    assert "2024-02-13" in msg
    assert "symbol=CU" in msg


def test_normalize_one_passes_when_calendar_accepts_all_dates() -> None:
    cal = TradingCalendar([date(2024, 2, 7), date(2024, 2, 8), date(2024, 2, 19)])
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-08"]),
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [10],
            "open_interest": [100],
            "contract_multiplier": [5],
        }
    )
    adapter = RQDataFuturesResearchAdapter(calendar=cal)
    normalized = adapter.normalize_one(
        rq_df=df,
        symbol_spec=RQSymbolSpec(
            rq_symbol="CU_DOM",
            underlying_symbol="CU",
            strategy_symbol="CU",
        ),
    )
    assert len(normalized) == 1


def test_normalize_one_requires_multiplier_source() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "open": [3.0],
            "high": [3.1],
            "low": [2.9],
            "close": [3.05],
            "volume": [30],
            "open_interest": [300],
        }
    )
    adapter = RQDataFuturesResearchAdapter()

    with pytest.raises(ValueError, match="Missing contract_multiplier"):
        adapter.normalize_one(
            rq_df=df,
            symbol_spec=RQSymbolSpec(
                rq_symbol="ZZ_DOM",
                underlying_symbol="ZZ",
                strategy_symbol="ZZ",
            ),
        )
