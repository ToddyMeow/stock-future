from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class FuturesMeta:
    contract_multiplier: float
    commission: float
    slippage: float
    group_name: str


FUTURES_META: Dict[str, FuturesMeta] = {
    # 黑色
    "RB": FuturesMeta(contract_multiplier=10, commission=3.0, slippage=1.0, group_name="black"),
    "I": FuturesMeta(contract_multiplier=100, commission=6.0, slippage=1.0, group_name="black"),
    "J": FuturesMeta(contract_multiplier=100, commission=12.0, slippage=1.0, group_name="black"),
    "JM": FuturesMeta(contract_multiplier=60, commission=6.0, slippage=1.0, group_name="black"),

    # 能化
    "SC": FuturesMeta(contract_multiplier=1000, commission=20.0, slippage=0.5, group_name="energy"),
    "MA": FuturesMeta(contract_multiplier=10, commission=3.0, slippage=1.0, group_name="chem"),

    # 农产品
    "M": FuturesMeta(contract_multiplier=10, commission=3.0, slippage=1.0, group_name="agri"),
    "Y": FuturesMeta(contract_multiplier=10, commission=3.0, slippage=2.0, group_name="agri"),

    # 有色
    "CU": FuturesMeta(contract_multiplier=5, commission=6.0, slippage=10.0, group_name="base_metal"),

    # 贵金属
    "AU": FuturesMeta(contract_multiplier=1000, commission=10.0, slippage=0.05, group_name="precious"),

    # 金融
    "IF": FuturesMeta(contract_multiplier=300, commission=25.0, slippage=0.4, group_name="index"),
    "IC": FuturesMeta(contract_multiplier=200, commission=25.0, slippage=0.4, group_name="index"),
    "IH": FuturesMeta(contract_multiplier=300, commission=25.0, slippage=0.4, group_name="index"),
    "IM": FuturesMeta(contract_multiplier=200, commission=25.0, slippage=0.4, group_name="index"),
}


def get_meta(underlying_symbol: str) -> FuturesMeta:
    """
    underlying_symbol 示例:
    - RB
    - I
    - M
    - IF
    """
    key = underlying_symbol.upper()
    if key not in FUTURES_META:
        raise KeyError(
            f"Metadata not configured for underlying_symbol={underlying_symbol!r}"
        )
    return FUTURES_META[key]