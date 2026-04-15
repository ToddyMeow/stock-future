from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class FuturesMeta:
    commission: float
    slippage: float
    group_name: str
    margin_rate: float = 0.10


DEFAULT_FUTURES_META = FuturesMeta(
    commission=5.0,
    slippage=1.0,
    group_name="unknown",
    margin_rate=0.10,
)

EXCHANGE_GROUP_MAP = {
    "CFFEX": "index",
    "INE": "energy",
    "SHFE": "shfe",
    "DCE": "dce",
    "CZCE": "czce",
    "GFEX": "gfex",
}

PRODUCT_GROUP_MAP = {
    "commodity": "commodity",
    "index": "index",
    "government bond": "rate",
}

FUTURES_META_OVERRIDES: Dict[str, FuturesMeta] = {
    # 黑色
    "RB": FuturesMeta(commission=3.0, slippage=1.0, group_name="black"),
    "I": FuturesMeta(commission=6.0, slippage=1.0, group_name="black"),
    "J": FuturesMeta(commission=12.0, slippage=1.0, group_name="black"),
    "JM": FuturesMeta(commission=6.0, slippage=1.0, group_name="black"),

    # 能化
    "SC": FuturesMeta(commission=20.0, slippage=0.5, group_name="energy"),
    "MA": FuturesMeta(commission=3.0, slippage=1.0, group_name="chem"),

    # 农产品
    "M": FuturesMeta(commission=3.0, slippage=1.0, group_name="agri"),
    "Y": FuturesMeta(commission=3.0, slippage=2.0, group_name="agri"),

    # 有色
    "CU": FuturesMeta(commission=6.0, slippage=10.0, group_name="base_metal"),

    # 贵金属
    "AU": FuturesMeta(commission=10.0, slippage=0.05, group_name="precious"),

    # 金融
    "IF": FuturesMeta(commission=25.0, slippage=0.4, group_name="index"),
    "IC": FuturesMeta(commission=25.0, slippage=0.4, group_name="index"),
    "IH": FuturesMeta(commission=25.0, slippage=0.4, group_name="index"),
    "IM": FuturesMeta(commission=25.0, slippage=0.4, group_name="index"),
}


def infer_group_name(
    underlying_symbol: str,
    *,
    exchange: Optional[str] = None,
    product: Optional[str] = None,
) -> str:
    key = underlying_symbol.upper()
    override = FUTURES_META_OVERRIDES.get(key)
    if override is not None:
        return override.group_name

    if product:
        product_key = product.strip().lower()
        if product_key in PRODUCT_GROUP_MAP:
            return PRODUCT_GROUP_MAP[product_key]

    if exchange:
        exchange_key = exchange.strip().upper()
        if exchange_key in EXCHANGE_GROUP_MAP:
            return EXCHANGE_GROUP_MAP[exchange_key]

    return DEFAULT_FUTURES_META.group_name


def get_meta(
    underlying_symbol: str,
    *,
    exchange: Optional[str] = None,
    product: Optional[str] = None,
) -> FuturesMeta:
    """
    underlying_symbol 示例:
    - RB
    - I
    - M
    - IF

    This is a local override layer for backtest assumptions. Contract discovery
    and contract_multiplier should come from RQData, not from this file.
    """
    key = underlying_symbol.upper()
    override = FUTURES_META_OVERRIDES.get(key)
    if override is not None:
        return override

    return FuturesMeta(
        commission=DEFAULT_FUTURES_META.commission,
        slippage=DEFAULT_FUTURES_META.slippage,
        group_name=infer_group_name(
            key,
            exchange=exchange,
            product=product,
        ),
    )
