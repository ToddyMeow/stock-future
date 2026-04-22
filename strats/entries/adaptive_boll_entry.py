"""v5 — Adaptive Bollinger Band breakout entry.

The band multiplier `k` scales with the current volatility regime:
  effective_k(t) = base_k × [min_ratio + (max_ratio - min_ratio) × sigma_rank(t)]

where `sigma_rank(t)` is the rolling percentile of σ/μ (stdev over mean of
`period`-bar close) over the last `adapt_lookback` days. The rationale:

- Compressed regime (sigma_rank low) → small k → tight bands → catch
  compression breakout early
- High-volatility regime (sigma_rank high) → large k → wide bands → avoid
  false breakouts on noise

Direction semantics mirror BollBreakEntryStrategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdaptiveBollEntryConfig:
    period: int = 22
    base_k: float = 2.0
    adapt_lookback: int = 60
    min_k_ratio: float = 0.5    # compressed regime → 0.5 × base (tight)
    max_k_ratio: float = 1.5    # high-vol regime → 1.5 × base (wide)
    allow_short: bool = False


class AdaptiveBollEntryStrategy:
    """Volatility-adaptive Bollinger Band breakout."""

    def __init__(self, config: Optional[AdaptiveBollEntryConfig] = None) -> None:
        self.config = config or AdaptiveBollEntryConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if df.empty:
            out = df.copy()
            out["entry_trigger_pass"] = False
            out["entry_direction"] = 0
            return out

        close = df["close"].astype(float)
        close_shifted = close.shift(1)
        ma = close_shifted.rolling(cfg.period, min_periods=cfg.period).mean()
        std = close_shifted.rolling(cfg.period, min_periods=cfg.period).std(ddof=0)

        # Volatility rank: percentile of σ/μ over adapt_lookback days
        sigma_pct = std / ma.where(ma != 0, np.nan).abs()
        sigma_rank = sigma_pct.rolling(cfg.adapt_lookback, min_periods=10).rank(pct=True)
        sigma_rank = sigma_rank.fillna(0.5)

        min_k = cfg.base_k * cfg.min_k_ratio
        max_k = cfg.base_k * cfg.max_k_ratio
        effective_k = min_k + (max_k - min_k) * sigma_rank

        upper = ma + effective_k * std
        lower = ma - effective_k * std

        long_trigger = close > upper
        short_trigger = pd.Series(False, index=df.index)
        if cfg.allow_short:
            short_trigger = close < lower

        entry_trigger_pass = (long_trigger | short_trigger).fillna(False)
        entry_direction = pd.Series(0, index=df.index, dtype=int)
        entry_direction[long_trigger.fillna(False)] = 1
        entry_direction[short_trigger.fillna(False) & ~long_trigger.fillna(False)] = -1

        out = df.copy()
        out["boll_ma"] = ma
        out["boll_upper"] = upper
        out["boll_lower"] = lower
        out["adaptive_k"] = effective_k
        out["sigma_rank"] = sigma_rank
        out["entry_trigger_pass"] = entry_trigger_pass
        out["entry_direction"] = entry_direction
        return out

    def build_pending_entry_metadata(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "boll_ma": float(row["boll_ma"]) if pd.notna(row.get("boll_ma")) else np.nan,
            "boll_upper": float(row["boll_upper"]) if pd.notna(row.get("boll_upper")) else np.nan,
            "boll_lower": float(row["boll_lower"]) if pd.notna(row.get("boll_lower")) else np.nan,
            "adaptive_k": float(row["adaptive_k"]) if pd.notna(row.get("adaptive_k")) else np.nan,
            "sigma_rank": float(row["sigma_rank"]) if pd.notna(row.get("sigma_rank")) else np.nan,
        }
