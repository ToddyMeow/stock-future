from __future__ import annotations

import pandas as pd
import pytest

from data.adapters.ohlc_repair import repair_ohlc_envelope


def test_close_above_high_expands_high() -> None:
    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [102.0],
    })
    changes = repair_ohlc_envelope(
        df, high_col="high", low_col="low", enveloped_cols=["open", "close"],
    )
    assert changes == 1
    assert df.loc[0, "high"] == 102.0
    assert df.loc[0, "low"] == 99.0


def test_close_below_low_expands_low() -> None:
    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [98.0],
    })
    changes = repair_ohlc_envelope(
        df, high_col="high", low_col="low", enveloped_cols=["open", "close"],
    )
    assert changes == 1
    assert df.loc[0, "high"] == 101.0
    assert df.loc[0, "low"] == 98.0


def test_normal_bars_unchanged() -> None:
    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5],
    })
    before = df.copy()
    changes = repair_ohlc_envelope(
        df, high_col="high", low_col="low", enveloped_cols=["open", "close"],
    )
    assert changes == 0
    pd.testing.assert_frame_equal(df, before)


def test_includes_settle_when_present() -> None:
    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [100.5], "settle": [103.0],  # settle > high
    })
    changes = repair_ohlc_envelope(
        df, high_col="high", low_col="low",
        enveloped_cols=["open", "close", "settle"],
    )
    assert changes == 1
    assert df.loc[0, "high"] == 103.0


def test_missing_enveloped_col_ignored() -> None:
    # Passing a column name that doesn't exist is a no-op for that column.
    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5],
    })
    changes = repair_ohlc_envelope(
        df, high_col="high", low_col="low",
        enveloped_cols=["open", "close", "settle"],  # no settle column
    )
    assert changes == 0


def test_missing_high_or_low_returns_zero() -> None:
    df = pd.DataFrame({"open": [100.0], "close": [101.0]})
    assert repair_ohlc_envelope(
        df, high_col="high", low_col="low", enveloped_cols=["open", "close"],
    ) == 0


def test_multiple_rows_counts_correctly() -> None:
    df = pd.DataFrame({
        "open":  [100.0, 100.0, 100.0],
        "high":  [101.0, 101.0, 101.0],
        "low":   [99.0,  99.0,  99.0],
        "close": [100.5, 102.0, 98.0],  # row 0 ok, row 1 above, row 2 below
    })
    changes = repair_ohlc_envelope(
        df, high_col="high", low_col="low", enveloped_cols=["open", "close"],
    )
    assert changes == 2
    assert df.loc[0, "high"] == 101.0 and df.loc[0, "low"] == 99.0
    assert df.loc[1, "high"] == 102.0
    assert df.loc[2, "low"] == 98.0


def test_empty_df_returns_zero() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert repair_ohlc_envelope(
        df, high_col="high", low_col="low", enveloped_cols=["open", "close"],
    ) == 0


def test_does_not_repair_h_lt_l() -> None:
    # h < l is real corruption — method B does NOT swap them, it just
    # envelopes around the enveloped cols. Caller should fail loud elsewhere.
    df = pd.DataFrame({
        "open": [100.0], "high": [98.0], "low": [102.0], "close": [101.0],
    })
    # After envelope: h = max(98, 100, 101) = 101; l = min(102, 100, 101) = 100
    # So still l > h — callers still need to detect real corruption.
    repair_ohlc_envelope(
        df, high_col="high", low_col="low", enveloped_cols=["open", "close"],
    )
    assert df.loc[0, "high"] == 101.0
    assert df.loc[0, "low"] == 100.0
    # The bar is still invalid (h < l after repair), but that's the caller's problem.
