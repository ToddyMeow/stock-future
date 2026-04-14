from __future__ import annotations

import pandas as pd
import pytest

from data.adapters.futures_static_meta import get_meta
from data.adapters.rqdata_futures_adatpter import (
    RQDataFuturesResearchAdapter,
    RQSymbolSpec,
)


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
    assert normalized.loc[0, "commission"] == 6.0
    assert normalized.loc[0, "slippage"] == 2.5


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
