"""DSR + PBO overfitting diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strats.helpers import deflated_sharpe, pbo_cscv


# ── DSR ─────────────────────────────────────────────────────────────────


def test_dsr_positive_strategy_high_probability() -> None:
    # Consistent positive-mean returns → strong evidence even after 1-trial deflation.
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(loc=0.001, scale=0.01, size=252 * 2))
    dsr = deflated_sharpe(r, n_trials=1)
    assert 0.9 <= dsr <= 1.0


def test_dsr_many_trials_deflates_pure_luck() -> None:
    # Zero-mean returns with 1000 trials → DSR should be well below 0.5.
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(loc=0.0, scale=0.01, size=252 * 2))
    dsr_1 = deflated_sharpe(r, n_trials=1)
    dsr_1000 = deflated_sharpe(r, n_trials=1000)
    # More trials → more deflation → lower DSR.
    assert dsr_1000 < dsr_1


def test_dsr_nan_on_degenerate_input() -> None:
    assert np.isnan(deflated_sharpe(pd.Series([0.01] * 5), n_trials=10))  # too few
    assert np.isnan(deflated_sharpe(pd.Series([0.0] * 100), n_trials=10))  # zero std


# ── PBO ─────────────────────────────────────────────────────────────────


def test_pbo_near_half_when_all_strategies_identical_noise() -> None:
    # All columns are independent noise with identical distribution.
    # IS best ≠ OOS best purely by chance → PBO near 0.5 (pure luck).
    rng = np.random.default_rng(7)
    T, N = 256, 20
    rm = pd.DataFrame(rng.normal(0, 0.01, size=(T, N)))
    pbo = pbo_cscv(rm, n_splits=16)
    assert 0.35 <= pbo <= 0.65, f"expected ~0.5, got {pbo}"


def test_pbo_low_when_one_strategy_truly_dominates() -> None:
    # One strategy has clear positive drift; others are noise. The best IS
    # strategy is consistently also the best OOS → PBO should be near 0.
    rng = np.random.default_rng(11)
    T, N = 256, 20
    rm = pd.DataFrame(rng.normal(0, 0.01, size=(T, N)))
    rm.iloc[:, 0] = rng.normal(0.002, 0.01, size=T)  # edge strategy
    pbo = pbo_cscv(rm, n_splits=16)
    assert pbo < 0.25, f"expected PBO < 0.25 for dominant strategy, got {pbo}"


def test_pbo_requires_even_splits_and_enough_data() -> None:
    with pytest.raises(ValueError, match="n_splits must be even"):
        pbo_cscv(pd.DataFrame(np.zeros((100, 3))), n_splits=7)
    # Too few rows → NaN
    tiny = pd.DataFrame(np.random.normal(size=(10, 3)))
    assert np.isnan(pbo_cscv(tiny, n_splits=16))
