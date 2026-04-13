from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from data.adapters.futures_static_meta import get_meta


@dataclass(frozen=True)
class RQSymbolSpec:
    """
    rq_symbol:
        RQData 查询标识
        研究阶段可以是你自己的研究对象标识，例如:
        - RB_DOM
        - I_DOM
        - M_DOM
        或者真实合约：
        - RB2505
        - I2509

    underlying_symbol:
        用于映射静态元数据的底层品种符号，例如:
        - RB
        - I
        - M
        - IF

    strategy_symbol:
        喂给 HAB v1 的稳定 symbol。
        研究阶段建议直接用 underlying_symbol
    """
    rq_symbol: str
    underlying_symbol: str
    strategy_symbol: str


class RQDataFuturesResearchAdapter:
    REQUIRED_COLUMNS = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    }

    OUTPUT_COLUMNS = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "contract_multiplier",
        "commission",
        "slippage",
        "group_name",
    ]

    def __init__(
        self,
        default_slippage_override: Optional[float] = None,
        drop_zero_volume_rows: bool = False,
    ) -> None:
        self.default_slippage_override = default_slippage_override
        self.drop_zero_volume_rows = drop_zero_volume_rows

    def normalize_one(
        self,
        rq_df: pd.DataFrame,
        symbol_spec: RQSymbolSpec,
    ) -> pd.DataFrame:
        """
        输入可以是：
        1. index 为 DatetimeIndex，columns 包含 open/high/low/close/volume/open_interest
        2. 普通 DataFrame，但必须能解析出 date 列
        """
        meta = get_meta(symbol_spec.underlying_symbol)
        df = rq_df.copy()

        df = self._ensure_date_column(df)
        self._validate_columns(df)

        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
                "symbol": symbol_spec.strategy_symbol,
                "open": pd.to_numeric(df["open"], errors="coerce"),
                "high": pd.to_numeric(df["high"], errors="coerce"),
                "low": pd.to_numeric(df["low"], errors="coerce"),
                "close": pd.to_numeric(df["close"], errors="coerce"),
                "volume": pd.to_numeric(df["volume"], errors="coerce"),
                "open_interest": pd.to_numeric(df["open_interest"], errors="coerce"),
                "contract_multiplier": float(meta.contract_multiplier),
                "commission": float(meta.commission),
                "slippage": float(
                    self.default_slippage_override
                    if self.default_slippage_override is not None
                    else meta.slippage
                ),
                "group_name": meta.group_name,
            }
        )

        out = self._basic_clean(out)
        out = out[self.OUTPUT_COLUMNS]
        return out

    def normalize_many(
        self,
        rq_dfs: List[pd.DataFrame],
        symbol_specs: List[RQSymbolSpec],
    ) -> pd.DataFrame:
        if len(rq_dfs) != len(symbol_specs):
            raise ValueError("rq_dfs and symbol_specs must have the same length.")

        frames = []
        for rq_df, spec in zip(rq_dfs, symbol_specs):
            frames.append(self.normalize_one(rq_df, spec))

        out = pd.concat(frames, axis=0, ignore_index=True)
        out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
        return out

    def save_csv(self, df: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def save_parquet(self, df: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "date" in out.columns:
            return out

        if "datetime" in out.columns:
            out = out.rename(columns={"datetime": "date"})
            return out

        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            first_col = out.columns[0]
            out = out.rename(columns={first_col: "date"})
            return out

        raise ValueError(
            "RQData dataframe must contain 'date' or 'datetime', "
            "or use DatetimeIndex."
        )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"RQData dataframe missing required columns: {sorted(missing)}"
            )

    def _basic_clean(self, out: pd.DataFrame) -> pd.DataFrame:
        out = out.loc[out["date"].notna()].copy()

        essential_cols = ["open", "high", "low", "close", "volume", "open_interest"]
        out = out.dropna(subset=essential_cols).copy()

        out["volume"] = out["volume"].clip(lower=0.0)
        out["open_interest"] = out["open_interest"].clip(lower=0.0)

        if self.drop_zero_volume_rows:
            out = out.loc[out["volume"] > 0].copy()

        out = (
            out.sort_values(["date", "symbol"])
            .drop_duplicates(subset=["date", "symbol"], keep="last")
            .reset_index(drop=True)
        )

        return out