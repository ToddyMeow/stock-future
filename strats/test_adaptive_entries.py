"""v5 Adaptive entries: AdaptiveHL and AdaptiveBoll signal invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strats.entries.adaptive_hl_entry import (
    AdaptiveHLEntryConfig, AdaptiveHLEntryStrategy,
)
from strats.entries.adaptive_boll_entry import (
    AdaptiveBollEntryConfig, AdaptiveBollEntryStrategy,
)


def _synth_df(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame({
        "date": dates, "symbol": "X",
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close.astype(float),
        "volume": 100.0, "open_interest": 100.0,
    })
    df["atr"] = (df["high"] - df["low"]).rolling(10, min_periods=1).mean()
    return df


# ---------- AdaptiveHL ----------


def test_adaptive_hl_returns_required_columns() -> None:
    close = np.linspace(100, 130, 100)
    df = _synth_df(close)
    out = AdaptiveHLEntryStrategy(
        AdaptiveHLEntryConfig(base_period=20, adapt_lookback=40, allow_short=True)
    ).prepare_signals(df)
    for c in ("entry_trigger_pass", "entry_direction", "channel_high",
              "channel_low", "adaptive_eff_period", "atr_rank"):
        assert c in out.columns


def test_adaptive_hl_effective_period_in_bounds() -> None:
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.normal(0, 1, 200))
    df = _synth_df(close)
    out = AdaptiveHLEntryStrategy(
        AdaptiveHLEntryConfig(base_period=20, min_period_ratio=0.5,
                              max_period_ratio=1.5, adapt_lookback=60)
    ).prepare_signals(df)
    eff = out["adaptive_eff_period"].dropna()
    assert eff.min() >= 10  # 20 × 0.5
    assert eff.max() <= 30  # 20 × 1.5


def test_adaptive_hl_fires_on_breakout() -> None:
    # Flat baseline then abrupt jump — channel_high stays at 101, close jumps to 120.
    close = np.concatenate([np.full(40, 100.0), np.full(40, 120.0)])
    df = _synth_df(close)
    out = AdaptiveHLEntryStrategy(
        AdaptiveHLEntryConfig(base_period=20, allow_short=False)
    ).prepare_signals(df)
    assert out["entry_trigger_pass"].any()
    assert out.loc[out["entry_trigger_pass"], "entry_direction"].eq(1).all()


# ---------- AdaptiveBoll ----------


def test_adaptive_boll_returns_required_columns() -> None:
    close = np.linspace(100, 130, 100)
    df = _synth_df(close)
    out = AdaptiveBollEntryStrategy(
        AdaptiveBollEntryConfig(period=22, base_k=2.0, adapt_lookback=40, allow_short=True)
    ).prepare_signals(df)
    for c in ("entry_trigger_pass", "entry_direction", "boll_upper",
              "boll_lower", "boll_ma", "adaptive_k", "sigma_rank"):
        assert c in out.columns


def test_adaptive_boll_effective_k_in_bounds() -> None:
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.normal(0, 1, 200))
    df = _synth_df(close)
    out = AdaptiveBollEntryStrategy(
        AdaptiveBollEntryConfig(period=22, base_k=2.0,
                                min_k_ratio=0.5, max_k_ratio=1.5,
                                adapt_lookback=60)
    ).prepare_signals(df)
    k = out["adaptive_k"].dropna()
    assert k.min() >= 1.0 - 1e-9  # 2.0 × 0.5
    assert k.max() <= 3.0 + 1e-9  # 2.0 × 1.5


def test_adaptive_boll_fires_on_breakout() -> None:
    close = np.concatenate([np.full(50, 100.0), np.linspace(100, 140, 50)])
    df = _synth_df(close)
    out = AdaptiveBollEntryStrategy(
        AdaptiveBollEntryConfig(period=22, base_k=2.0, allow_short=False)
    ).prepare_signals(df)
    assert out["entry_trigger_pass"].any()
    assert out.loc[out["entry_trigger_pass"], "entry_direction"].eq(1).all()
