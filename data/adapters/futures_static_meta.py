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

FUTURES_GROUP_MAP = {
    "BU":"chem_energy","EB":"chem_energy","EG":"chem_energy","FU":"chem_energy",
    "L":"chem_energy","MA":"chem_energy","PF":"chem_energy","PG":"chem_energy",
    "PP":"chem_energy","SC":"chem_energy","TA":"chem_energy","V":"chem_energy",
    "CF":"rubber_fiber","CY":"rubber_fiber","RU":"rubber_fiber","SR":"rubber_fiber",
    "I":"black_steel","J":"black_steel","JM":"black_steel","RB":"black_steel","ZC":"black_steel","SF":"black_steel",
    "FG":"building","SA":"building","SP":"building","AO":"building","LC":"building",
    "SI":"building","SH":"building","SM":"building","UR":"building","WR":"building",
    "IC":"equity_index","IF":"equity_index","IM":"equity_index",
    "AG":"metals","AL":"metals","AU":"metals","CU":"metals","NI":"metals",
    "PB":"metals","SN":"metals","SS":"metals","ZN":"metals",
    "T":"bond","TL":"bond","TS":"bond",
    "A":"agri","B":"agri","C":"agri","CS":"agri","M":"agri","OI":"agri",
    "P":"agri","PK":"agri","RM":"agri","RS":"agri","Y":"agri",
    "JD":"livestock","LH":"livestock",
    "AP":"ind_AP","BB":"ind_BB","CJ":"ind_CJ","EC":"ind_EC","FB":"ind_FB",
    "JR":"ind_JR","LR":"ind_LR","LU":"ind_LU","PM":"ind_PM","RI":"ind_RI",
    "RR":"ind_RR","WH":"ind_WH",
}

EXCLUDED_SYMBOLS = {"BC","PR","PX","NR","HC","IH","TF","BR","LG","AD","BZ","OP","PD","PT","PP_F","V_F","L_F"}

FUTURES_META_OVERRIDES: Dict[str, FuturesMeta] = {
    # 黑色
    "RB": FuturesMeta(commission=3.0, slippage=1.0, group_name="black_steel"),
    "I": FuturesMeta(commission=6.0, slippage=1.0, group_name="black_steel"),
    "J": FuturesMeta(commission=12.0, slippage=1.0, group_name="black_steel"),
    "JM": FuturesMeta(commission=6.0, slippage=1.0, group_name="black_steel"),

    # 能化
    "SC": FuturesMeta(commission=20.0, slippage=0.5, group_name="chem_energy"),
    "MA": FuturesMeta(commission=3.0, slippage=1.0, group_name="chem_energy"),

    # 农产品
    "M": FuturesMeta(commission=3.0, slippage=1.0, group_name="agri"),
    "Y": FuturesMeta(commission=3.0, slippage=2.0, group_name="agri"),

    # 有色
    "CU": FuturesMeta(commission=6.0, slippage=10.0, group_name="metals"),

    # 贵金属
    "AU": FuturesMeta(commission=10.0, slippage=0.05, group_name="metals"),

    # 金融
    "IF": FuturesMeta(commission=25.0, slippage=0.4, group_name="equity_index"),
    "IC": FuturesMeta(commission=25.0, slippage=0.4, group_name="equity_index"),
    "IH": FuturesMeta(commission=25.0, slippage=0.4, group_name="equity_index"),
    "IM": FuturesMeta(commission=25.0, slippage=0.4, group_name="equity_index"),
}


def infer_group_name(
    underlying_symbol: str,
    *,
    exchange: Optional[str] = None,
    product: Optional[str] = None,
) -> str:
    key = underlying_symbol.upper()

    # 1. Explicit group map (most specific)
    group = FUTURES_GROUP_MAP.get(key)
    if group is not None:
        return group

    # 2. Meta overrides (commission/slippage layer)
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
