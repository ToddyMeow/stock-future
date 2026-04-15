"""Shared helpers for strategy modules.

Direction-aware helpers, technical indicators, pattern detection, and
portfolio analytics. All functions are stateless module-level utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ── Direction-aware helpers ────────────────────────────────────────────────


def apply_exit_slippage(base_price: float, slippage: float, direction: int) -> float:
    return base_price - slippage if direction == 1 else base_price + slippage


def directional_pnl(exit_price: float, entry_price: float, direction: int) -> float:
    """Signed P&L per unit: positive = favorable for the position."""
    return (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)


def favorable_excursion(extreme_price: float, entry_price: float, direction: int) -> float:
    if direction == 1:
        return max(extreme_price - entry_price, 0.0)
    return max(entry_price - extreme_price, 0.0)


def adverse_excursion(extreme_price: float, entry_price: float, direction: int) -> float:
    if direction == 1:
        return max(entry_price - extreme_price, 0.0)
    return max(extreme_price - entry_price, 0.0)


# ── Technical indicators ──────────────────────────────────────────────────


def wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's recursive smoothing (used by ATR and ADX). NaN-safe."""
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < period:
        return out
    out[period - 1] = np.nanmean(values[:period])
    for i in range(period, len(values)):
        prev = out[i - 1]
        cur = values[i]
        if np.isnan(cur):
            out[i] = prev
        elif np.isnan(prev):
            out[i] = cur
        else:
            out[i] = ((prev * (period - 1)) + cur) / period
    return out


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Average Directional Index (ADX). Returns values 0-100."""
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(h)

    # True Range
    prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
    tr = np.maximum(np.maximum(np.abs(h - l), np.abs(h - prev_c)), np.abs(l - prev_c))

    # Directional Movement
    up_move = np.zeros(n); down_move = np.zeros(n)
    up_move[1:] = h[1:] - h[:-1]
    down_move[1:] = l[:-1] - l[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder smooth all three
    atr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)

    # +DI, -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_smooth / np.where(atr_smooth > 0, atr_smooth, np.nan)
        minus_di = 100.0 * minus_dm_smooth / np.where(atr_smooth > 0, atr_smooth, np.nan)
        dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, np.nan)

    # ADX = Wilder smooth of DX
    adx_values = wilder_smooth(dx, period)
    return pd.Series(adx_values, index=high.index)


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return pd.Series(wilder_smooth(tr.to_numpy(dtype=float), period), index=tr.index)


def rolling_last_value_percentile(values: pd.Series, window: int) -> pd.Series:
    """Midpoint percentile: (count_less + 0.5 * count_equal) / window.

    When all values are equal the percentile is 0.5 (not 1.0), so a flat
    narrow-bandwidth window is correctly treated as moderate compression.
    """
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        sample = arr[i - window + 1 : i + 1]
        if np.isnan(sample).any():
            continue
        last = sample[-1]
        less = float(np.sum(sample < last))
        equal = float(np.sum(sample == last))
        out[i] = (less + 0.5 * equal) / float(window)
    return pd.Series(out, index=values.index)


# ── Pattern detection ─────────────────────────────────────────────────────


def detect_hlh_pattern(
    high_window: List[float],
    low_window: List[float],
    box_high: float,
    box_low: float,
    tol: float,
) -> Tuple[bool, bool, bool, bool]:
    """Detect the mandatory High-Low-High sequence within the box window."""

    state = 0
    has_upper_test_1 = False
    has_lower_confirm = False
    has_upper_test_2 = False

    for h, l in zip(high_window, low_window):
        upper_hit = h >= box_high - tol
        lower_hit = l <= box_low + tol

        if state == 0 and upper_hit:
            has_upper_test_1 = True
            state = 1
            continue

        if state == 1 and lower_hit:
            has_lower_confirm = True
            state = 2
            continue

        if state == 2 and upper_hit:
            has_upper_test_2 = True
            state = 3
            break

    valid = has_upper_test_1 and has_lower_confirm and has_upper_test_2
    return valid, has_upper_test_1, has_lower_confirm, has_upper_test_2


def detect_lhl_pattern(
    high_window: List[float],
    low_window: List[float],
    box_high: float,
    box_low: float,
    tol: float,
) -> Tuple[bool, bool, bool, bool]:
    """Detect the mandatory Low-High-Low sequence within the box window (short side)."""

    state = 0
    has_lower_test_1 = False
    has_upper_confirm = False
    has_lower_test_2 = False

    for h, l in zip(high_window, low_window):
        upper_hit = h >= box_high - tol
        lower_hit = l <= box_low + tol

        if state == 0 and lower_hit:
            has_lower_test_1 = True
            state = 1
            continue

        if state == 1 and upper_hit:
            has_upper_confirm = True
            state = 2
            continue

        if state == 2 and lower_hit:
            has_lower_test_2 = True
            state = 3
            break

    valid = has_lower_test_1 and has_upper_confirm and has_lower_test_2
    return valid, has_lower_test_1, has_upper_confirm, has_lower_test_2


# ── Portfolio analytics ───────────────────────────────────────────────────


class PortfolioAnalyzer:
    """Post-hoc analysis of BacktestResult for portfolio-level metrics."""

    def __init__(self, result: Any, config: Any) -> None:
        self.result = result
        self.config = config

    def equity_curve(self) -> pd.DataFrame:
        pdf = self.result.portfolio_daily.copy()
        if pdf.empty:
            return pd.DataFrame(columns=["date", "equity", "peak", "drawdown", "drawdown_pct", "daily_return"])
        out = pdf[["date", "equity"]].copy()
        out["peak"] = out["equity"].cummax()
        out["drawdown"] = out["equity"] - out["peak"]
        out["drawdown_pct"] = out["drawdown"] / out["peak"].where(out["peak"] > 0, np.nan)
        out["daily_return"] = out["equity"].pct_change()
        return out

    def drawdown_episodes(self) -> pd.DataFrame:
        ec = self.equity_curve()
        if ec.empty:
            return pd.DataFrame(columns=["start_date", "trough_date", "recovery_date", "max_drawdown_pct", "duration_days"])
        episodes: List[Dict[str, Any]] = []
        in_dd = False
        start = None
        trough = None
        trough_pct = 0.0
        for _, row in ec.iterrows():
            if row["drawdown"] < 0:
                if not in_dd:
                    in_dd = True
                    start = row["date"]
                    trough = row["date"]
                    trough_pct = row["drawdown_pct"]
                if row["drawdown_pct"] < trough_pct:
                    trough = row["date"]
                    trough_pct = row["drawdown_pct"]
            else:
                if in_dd:
                    episodes.append({
                        "start_date": start,
                        "trough_date": trough,
                        "recovery_date": row["date"],
                        "max_drawdown_pct": trough_pct,
                        "duration_days": (row["date"] - start).days,
                    })
                    in_dd = False
        if in_dd:
            episodes.append({
                "start_date": start,
                "trough_date": trough,
                "recovery_date": pd.NaT,
                "max_drawdown_pct": trough_pct,
                "duration_days": (ec["date"].iloc[-1] - start).days,
            })
        return pd.DataFrame(episodes)

    def group_contribution(self) -> pd.DataFrame:
        trades = self.result.trades
        if trades.empty:
            return pd.DataFrame(columns=["group_name", "net_pnl_sum", "trade_count", "win_count", "loss_count", "win_rate", "avg_r_multiple"])
        grouped = trades.groupby("group_name").agg(
            net_pnl_sum=("net_pnl", "sum"),
            trade_count=("net_pnl", "count"),
            win_count=("net_pnl", lambda x: (x > 0).sum()),
            loss_count=("net_pnl", lambda x: (x <= 0).sum()),
            avg_r_multiple=("r_multiple", "mean"),
        ).reset_index()
        grouped["win_rate"] = grouped["win_count"] / grouped["trade_count"].where(grouped["trade_count"] > 0, 1)
        return grouped

    def risk_utilization(self) -> pd.DataFrame:
        pdf = self.result.portfolio_daily.copy()
        if pdf.empty:
            return pd.DataFrame(columns=["date", "risk_usage_pct", "leverage"])
        out = pdf[["date"]].copy()
        out["risk_usage_pct"] = pdf["open_risk"] / pdf["portfolio_risk_cap"].where(pdf["portfolio_risk_cap"] > 0, np.nan)
        out["leverage"] = pdf.get("leverage", 0.0)
        return out

    def signal_density(self) -> pd.DataFrame:
        ds = self.result.daily_status
        if ds.empty:
            return pd.DataFrame(columns=["date", "signals_fired", "signals_accepted", "signals_rejected"])
        triggered = ds[ds["entry_trigger_pass"] == True]
        if triggered.empty:
            return pd.DataFrame(columns=["date", "signals_fired", "signals_accepted", "signals_rejected"])
        grouped = triggered.groupby("date").agg(
            signals_fired=("entry_trigger_pass", "count"),
            signals_rejected=("risk_reject_reason", lambda x: x.notna().sum()),
        ).reset_index()
        grouped["signals_accepted"] = grouped["signals_fired"] - grouped["signals_rejected"]
        return grouped

    def periodic_returns(self, freq: str = "M") -> pd.DataFrame:
        ec = self.equity_curve()
        if ec.empty:
            return pd.DataFrame(columns=["period", "return_pct"])
        ts = ec.set_index("date")["equity"]
        resampled = ts.resample(freq).last().dropna()
        returns = resampled.pct_change().dropna()
        out = returns.reset_index()
        out.columns = ["period", "return_pct"]
        return out

    def summary_stats(self) -> Dict[str, Any]:
        ec = self.equity_curve()
        trades = self.result.trades
        if ec.empty or len(ec) < 2:
            return {}

        total_days = (ec["date"].iloc[-1] - ec["date"].iloc[0]).days
        years = max(total_days / 365.25, 1e-6)
        final_equity = ec["equity"].iloc[-1]
        initial_equity = ec["equity"].iloc[0]
        total_return = final_equity / initial_equity - 1.0 if initial_equity > 0 else 0.0
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0

        daily_returns = ec["daily_return"].dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0.0
        downside = daily_returns[daily_returns < 0]
        sortino = (daily_returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0.0
        max_dd_pct = ec["drawdown_pct"].min() if not ec["drawdown_pct"].isna().all() else 0.0

        stats: Dict[str, Any] = {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown_pct": max_dd_pct,
            "total_days": total_days,
        }

        if not trades.empty:
            wins = trades[trades["net_pnl"] > 0]
            losses = trades[trades["net_pnl"] <= 0]
            stats["total_trades"] = len(trades)
            stats["win_rate"] = len(wins) / len(trades) if len(trades) > 0 else 0.0
            stats["profit_factor"] = (
                wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())
                if len(losses) > 0 and losses["net_pnl"].sum() != 0
                else float("inf") if len(wins) > 0 else 0.0
            )
            stats["avg_r_multiple"] = trades["r_multiple"].mean()
            stats["expectancy"] = trades["net_pnl"].mean()

        return stats
