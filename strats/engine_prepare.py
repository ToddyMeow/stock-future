"""Preparation helpers extracted from StrategyEngine."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from strats.helpers import (
    adx as _adx,
    choppiness_index as _choppiness_index,
    wilder_atr as _wilder_atr,
)


def _compute_data_quality_report(self, bars: pd.DataFrame) -> pd.DataFrame:
    cfg = self.config
    required = {
        cfg.symbol_col,
        cfg.open_col,
        cfg.high_col,
        cfg.low_col,
        cfg.close_col,
        cfg.volume_col,
    }
    if not required.issubset(bars.columns) or bars.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "n_bars",
                "lock_pct",
                "near_zero_range_pct",
                "zero_volume_pct",
                "ohlc_anomaly_count",
            ]
        )

    h = bars[cfg.high_col].astype(float)
    l = bars[cfg.low_col].astype(float)
    o = bars[cfg.open_col].astype(float)
    c = bars[cfg.close_col].astype(float)
    v = bars[cfg.volume_col].astype(float)

    locked = (h == l) & (o == c)
    denom = c.where(c != 0)
    rng_pct = (h - l).div(denom).abs()
    near_zero = rng_pct < 0.001
    zero_vol = v == 0
    anomaly = (c > h + 1e-9) | (c < l - 1e-9)

    diag = pd.DataFrame(
        {
            cfg.symbol_col: bars[cfg.symbol_col].values,
            "locked": locked.values,
            "near_zero": near_zero.fillna(False).values,
            "zero_vol": zero_vol.values,
            "anomaly": anomaly.values,
        }
    )
    grouped = diag.groupby(cfg.symbol_col, sort=True).agg(
        n_bars=(cfg.symbol_col, "size"),
        lock_count=("locked", "sum"),
        near_zero_count=("near_zero", "sum"),
        zero_vol_count=("zero_vol", "sum"),
        ohlc_anomaly_count=("anomaly", "sum"),
    )
    grouped["lock_pct"] = grouped["lock_count"] / grouped["n_bars"]
    grouped["near_zero_range_pct"] = grouped["near_zero_count"] / grouped["n_bars"]
    grouped["zero_volume_pct"] = grouped["zero_vol_count"] / grouped["n_bars"]
    report = grouped[
        [
            "n_bars",
            "lock_pct",
            "near_zero_range_pct",
            "zero_volume_pct",
            "ohlc_anomaly_count",
        ]
    ].reset_index()
    return report.rename(columns={cfg.symbol_col: "symbol"})


def _normalize_and_validate_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
    cfg = self.config
    self._validate_input_columns(bars)

    df = bars.copy()
    dt = pd.to_datetime(df[cfg.date_col], errors="raise")
    if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
        dt = dt.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    df[cfg.date_col] = dt.dt.normalize()
    df = df.sort_values([cfg.symbol_col, cfg.date_col]).reset_index(drop=True)

    duplicate_mask = df.duplicated(subset=[cfg.symbol_col, cfg.date_col], keep=False)
    if duplicate_mask.any():
        dupes = df.loc[duplicate_mask, [cfg.symbol_col, cfg.date_col]]
        raise ValueError(f"Duplicate symbol/date rows after date normalization:\n{dupes}")

    numeric_cols = [
        cfg.open_col,
        cfg.high_col,
        cfg.low_col,
        cfg.close_col,
        cfg.volume_col,
        cfg.open_interest_col,
        cfg.multiplier_col,
        cfg.commission_col,
        cfg.slippage_col,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    for col in [cfg.multiplier_col, cfg.group_col]:
        df[col] = df.groupby(cfg.symbol_col, sort=False)[col].ffill()
        if df[col].isna().any():
            raise ValueError(
                f"Leading missing values in '{col}'. Cannot backfill from future rows."
            )

    for col in [cfg.commission_col, cfg.slippage_col]:
        if df[col].isna().any():
            raise ValueError(
                f"Column '{col}' contains missing values. Provide an explicit cost schedule."
            )

    if cfg.margin_rate_col not in df.columns:
        df[cfg.margin_rate_col] = cfg.default_margin_rate

    self._validate_input_values(df)

    self._gap_diagnostics = []
    for symbol, sym_df in df.groupby(cfg.symbol_col, sort=False):
        gap_days = sym_df[cfg.date_col].diff().dt.days
        suspicious = gap_days > 10
        if suspicious.any():
            for idx in sym_df.index[suspicious]:
                self._gap_diagnostics.append(
                    {
                        "symbol": symbol,
                        "date": sym_df.loc[idx, cfg.date_col],
                        "gap_days": int(gap_days.loc[idx]),
                    }
                )

    return df


def _validate_input_columns(self, bars: pd.DataFrame) -> None:
    cfg = self.config
    required = {
        cfg.date_col,
        cfg.symbol_col,
        cfg.open_col,
        cfg.high_col,
        cfg.low_col,
        cfg.close_col,
        cfg.volume_col,
        cfg.open_interest_col,
        cfg.multiplier_col,
        cfg.commission_col,
        cfg.slippage_col,
        cfg.group_col,
    }
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if cfg.enable_dual_stream:
        dual_required = {
            cfg.raw_open_col,
            cfg.raw_high_col,
            cfg.raw_low_col,
            cfg.raw_close_col,
            cfg.contract_col,
        }
        dual_missing = dual_required - set(bars.columns)
        if dual_missing:
            raise ValueError(
                "enable_dual_stream requires additional columns; "
                f"missing: {sorted(dual_missing)}. "
                "Run scripts/build_enhanced_bars.py to produce them."
            )


def _validate_input_values(self, df: pd.DataFrame) -> None:
    cfg = self.config
    numeric_cols = [
        cfg.open_col,
        cfg.high_col,
        cfg.low_col,
        cfg.close_col,
        cfg.volume_col,
        cfg.open_interest_col,
        cfg.multiplier_col,
        cfg.commission_col,
        cfg.slippage_col,
    ]
    numeric_values = df[numeric_cols].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Numeric input contains NaN or infinite values.")

    if cfg.enable_dual_stream:
        for c in [cfg.raw_open_col, cfg.raw_high_col, cfg.raw_low_col, cfg.raw_close_col]:
            raw_values = df[c].astype(float).to_numpy()
            if not np.isfinite(raw_values).all():
                n = int((~np.isfinite(raw_values)).sum())
                raise ValueError(f"column '{c}' has {n} NaN/inf values under dual_stream")
        if df[cfg.contract_col].isna().any():
            n = int(df[cfg.contract_col].isna().sum())
            raise ValueError(f"column '{cfg.contract_col}' has {n} NaN values under dual_stream")

    high = df[cfg.high_col].astype(float)
    low = df[cfg.low_col].astype(float)
    open_ = df[cfg.open_col].astype(float)
    close = df[cfg.close_col].astype(float)
    volume = df[cfg.volume_col].astype(float)
    open_interest = df[cfg.open_interest_col].astype(float)
    multiplier = df[cfg.multiplier_col].astype(float)
    commission = df[cfg.commission_col].astype(float)
    slippage = df[cfg.slippage_col].astype(float)

    if (high < low).any():
        raise ValueError("Invalid OHLC: high < low.")
    if (high < pd.concat([open_, close], axis=1).max(axis=1)).any():
        raise ValueError("Invalid OHLC: high < max(open, close).")
    if (low > pd.concat([open_, close], axis=1).min(axis=1)).any():
        raise ValueError("Invalid OHLC: low > min(open, close).")
    if (multiplier <= 0).any():
        raise ValueError("contract_multiplier must be > 0.")
    if (commission < 0).any():
        raise ValueError("commission must be >= 0.")
    if (slippage < 0).any():
        raise ValueError("slippage must be >= 0.")
    if (volume < 0).any():
        raise ValueError("volume must be >= 0.")
    if (open_interest < 0).any():
        raise ValueError("open_interest must be >= 0.")


def _prepare_symbol_base(self, df: pd.DataFrame) -> pd.DataFrame:
    cfg = self.config
    high = df[cfg.high_col].astype(float)
    low = df[cfg.low_col].astype(float)
    close = df[cfg.close_col].astype(float)

    out = df.copy()
    out["atr"] = _wilder_atr(high=high, low=low, close=close, period=cfg.atr_period)
    out["atr_ref"] = out["atr"].shift(1)
    out["adx"] = _adx(high=high, low=low, close=close, period=cfg.adx_period)
    out["cpi"] = _choppiness_index(high=high, low=low, close=close, period=cfg.cpi_period)
    out["next_trade_date"] = out[cfg.date_col].shift(-1)
    if cfg.trading_calendar is not None and len(out) > 0:
        last_idx = out.index[-1]
        if pd.isna(out.at[last_idx, "next_trade_date"]):
            last_d = out.at[last_idx, cfg.date_col]
            if pd.notna(last_d):
                try:
                    nxt = cfg.trading_calendar.next_trading_day(pd.Timestamp(last_d).date())
                    out.at[last_idx, "next_trade_date"] = pd.Timestamp(nxt)
                except Exception:
                    pass
    out["_bar_index"] = np.arange(len(out), dtype=int)
    return out


def _prepare_symbol_frame(self, df: pd.DataFrame) -> pd.DataFrame:
    base = self._prepare_symbol_base(df)
    return self._strategies[0].entry_strategy.prepare_signals(base)


def _prepare_all_strategies(self, bars: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cfg = self.config
    df = self._normalize_and_validate_bars(bars)

    base_frames: List[pd.DataFrame] = []
    for _, symbol_df in df.groupby(cfg.symbol_col, sort=False):
        base_frames.append(self._prepare_symbol_base(symbol_df.reset_index(drop=True)))

    if not base_frames:
        result: Dict[str, pd.DataFrame] = {}
        for slot in self._strategies:
            empty = df.copy()
            for col in self._prepared_extra_columns():
                empty[col] = pd.Series(dtype="float64")
            result[slot.strategy_id] = empty
        return result

    base_all = pd.concat(base_frames, axis=0, ignore_index=True)
    base_all = base_all.sort_values([cfg.date_col, cfg.symbol_col]).reset_index(drop=True)

    result = {}
    for slot in self._strategies:
        slot_frames: List[pd.DataFrame] = []
        for _, symbol_df in base_all.groupby(cfg.symbol_col, sort=False):
            slot_frames.append(
                slot.entry_strategy.prepare_signals(symbol_df.reset_index(drop=True))
            )
        slot_prepared = pd.concat(slot_frames, axis=0, ignore_index=True)
        slot_prepared = slot_prepared.sort_values([cfg.date_col, cfg.symbol_col]).reset_index(
            drop=True
        )
        result[slot.strategy_id] = slot_prepared

    return result
