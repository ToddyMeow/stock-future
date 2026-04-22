from __future__ import annotations

import asyncio
from datetime import date as _date
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

from live import account_state
from live.config import APP_TIMEZONE, INITIAL_CAPITAL, SOFT_STOP_ENABLED, SOFT_STOP_PCT
from live.engine_setup import build_engine_cfg_for_live
from live.web.queries import (
    fetch_alerts_24h_count,
    fetch_atr20_by_symbol,
    fetch_capital_snapshot,
    fetch_instructions_by_date,
    fetch_latest_bar_by_symbol,
    fetch_latest_diagnostics_by_symbol,
    fetch_latest_engine_state,
    fetch_positions_by_symbol,
    load_group_combo_map,
    load_universe_base,
)
from live.web.schemas import EngineStatus, FormulasContext, LaunchdSlot, UniverseSymbol


logger = logging.getLogger(__name__)

_PLIST_DESCRIPTIONS: dict[str, str] = {
    "com.stockfuture.signal_day": "日盘信号生成",
    "com.stockfuture.signal_night": "夜盘信号生成",
    "com.stockfuture.data_after_dayclose": "日盘收盘后数据拉取",
    "com.stockfuture.data_after_nightclose": "夜盘收盘后数据拉取",
    "com.stockfuture.daily_pnl_settlement": "每日 PnL 结算",
    "com.stockfuture.daily_report": "每日报告生成",
    "com.stockfuture.roll_detector": "主力合约切换检测",
    "com.stockfuture.soft_stop_check": "软熔断检查",
}
_PLIST_DIR = Path(__file__).resolve().parents[1] / "scheduler"
_RE_LABEL = re.compile(
    r"<key>\s*Label\s*</key>\s*<string>\s*([^<]+?)\s*</string>",
    re.DOTALL,
)
_RE_STARTCAL_BLOCK = re.compile(
    r"<key>\s*StartCalendarInterval\s*</key>\s*(<array>.*?</array>|<dict>.*?</dict>)",
    re.DOTALL,
)
_RE_HOUR = re.compile(
    r"<key>\s*Hour\s*</key>\s*<integer>\s*(\d+)\s*</integer>",
    re.DOTALL,
)
_RE_MINUTE = re.compile(
    r"<key>\s*Minute\s*</key>\s*<integer>\s*(\d+)\s*</integer>",
    re.DOTALL,
)
_RE_SINGLE_DICT = re.compile(r"<dict>.*?</dict>", re.DOTALL)


async def build_report_payload(report_date: _date) -> dict[str, Any]:
    instructions, pnl_rows = await asyncio.gather(
        account_state.get_instructions_with_fills(report_date),
        account_state.get_daily_pnl_range(report_date - timedelta(days=30), report_date),
    )

    today = next((row for row in pnl_rows if row.date == report_date), None)
    equity = float(today.equity) if today else 0.0
    pnl_today = float(today.realized_pnl_today + today.unrealized_pnl_today) if today else 0.0
    drawdown = float(today.drawdown_from_peak) if today and today.drawdown_from_peak else 0.0

    summary = {
        "fully_filled": 0,
        "partially_filled": 0,
        "vetoed": 0,
        "skipped": 0,
        "pending": 0,
        "expired": 0,
    }
    for instruction in instructions:
        status = str(instruction.get("status", "pending"))
        summary[status] = summary.get(status, 0) + 1

    vetoed_summary = [
        _jsonize_mapping(instruction)
        for instruction in instructions
        if str(instruction.get("status")) == "vetoed"
    ]

    return {
        "report_date": report_date.isoformat(),
        "equity": equity,
        "pnl_today": pnl_today,
        "drawdown": drawdown,
        "instructions_summary": summary,
        "vetoed_summary": vetoed_summary,
        "equity_series": [
            {"date": row.date.isoformat(), "equity": float(row.equity)} for row in pnl_rows
        ],
    }


async def build_universe_payload() -> list[dict[str, Any]]:
    base_rows = await load_universe_base()
    group_combo = await load_group_combo_map()
    symbols = [row["symbol"] for row in base_rows if row["symbol"]]

    bars_map, pos_map, atr20_map, diag_map = await asyncio.gather(
        fetch_latest_bar_by_symbol(symbols),
        fetch_positions_by_symbol(symbols),
        fetch_atr20_by_symbol(symbols),
        fetch_latest_diagnostics_by_symbol(symbols),
    )

    out: list[dict[str, Any]] = []
    for row in base_rows:
        symbol = row["symbol"]
        group = row["group_name"]
        bars = bars_map.get(symbol, {})
        position_qty = pos_map.get(symbol)
        atr20 = atr20_map.get(symbol)
        mult_raw = bars.get("contract_multiplier")
        multiplier = Decimal(str(mult_raw)) if mult_raw is not None else None
        if atr20 is not None and multiplier is not None:
            single_contract_risk: Decimal | None = (
                Decimal("2") * atr20 * multiplier
            ).quantize(Decimal("1"))
            tradeable = single_contract_risk < Decimal("7500")
        else:
            single_contract_risk = None
            tradeable = False

        diag = diag_map.get(symbol, {})
        payload = UniverseSymbol(
            symbol=symbol,
            group_name=group,
            combo=group_combo.get(group, ""),
            contract_code=bars.get("contract_code"),
            last_settle=(
                Decimal(str(bars["last_settle"]))
                if bars.get("last_settle") is not None
                else None
            ),
            last_settle_date=bars.get("last_settle_date"),
            contract_multiplier=multiplier,
            single_contract_risk=single_contract_risk,
            tradeable_250k=tradeable,
            in_position=position_qty is not None,
            position_qty=position_qty,
            last_session_date=diag.get("session_date"),
            last_session=diag.get("session"),
            last_entry_trigger=diag.get("entry_trigger"),
            last_entry_direction=diag.get("entry_direction"),
            last_reject_reason=diag.get("reject_reason"),
            last_miss_reason=diag.get("miss_reason"),
        )
        out.append(payload.model_dump(mode="json"))
    return out


async def build_engine_status_payload() -> dict[str, Any]:
    tz = ZoneInfo(APP_TIMEZONE)
    server_time = datetime.now(tz)

    try:
        await account_state.ping_db()
        db_health = "ok"
    except Exception as exc:  # noqa: BLE001
        db_health = f"error: {type(exc).__name__}: {exc}"

    launchd_schedule = _build_launchd_schedule()

    latest_state, instructions_by_date, alerts_24h, recent_alerts_rows, capital = await asyncio.gather(
        _safe(fetch_latest_engine_state(), "latest_state", None),
        _safe(fetch_instructions_by_date(14), "instructions_by_date", []),
        _safe(fetch_alerts_24h_count(), "alerts_24h", {"info": 0, "warn": 0, "critical": 0}),
        _safe(account_state.get_recent_alerts(10, None), "recent_alerts", []),
        _safe(fetch_capital_snapshot(), "capital", None),
    )

    return EngineStatus(
        latest_state=latest_state,
        instructions_by_date=instructions_by_date,
        alerts_24h_count=alerts_24h,
        recent_alerts=recent_alerts_rows,
        db_health=db_health,
        launchd_schedule=launchd_schedule,
        server_time=server_time,
        server_timezone=APP_TIMEZONE,
        capital_snapshot=capital,
    ).model_dump(mode="json")


async def build_formulas_context_payload() -> dict[str, Any]:
    cfg_section = _build_formulas_config_section()
    initial_capital: Decimal = cfg_section["initial_capital"]
    risk_per_trade: float = cfg_section["risk_per_trade"]
    portfolio_risk_cap: float = cfg_section["portfolio_risk_cap"]
    group_risk_cap_default: float = cfg_section["group_risk_cap_default"]
    max_portfolio_leverage: float = cfg_section["max_portfolio_leverage"]

    positions, capital = await asyncio.gather(
        _safe(account_state.get_positions_enriched(), "positions", []),
        _safe(fetch_capital_snapshot(), "capital", None),
    )

    if capital is not None:
        equity: Decimal | None = capital.equity
        cash: Decimal | None = capital.cash
        peak_equity: Decimal | None = capital.peak_equity_to_date
        drawdown: Decimal | None = capital.drawdown_from_peak
        snapshot_date: _date | None = capital.date
    else:
        equity = initial_capital
        cash = initial_capital
        peak_equity = None
        drawdown = None
        snapshot_date = None

    equity_for_calc = equity if equity is not None else initial_capital
    risk_budget_per_trade = (equity_for_calc * Decimal(str(risk_per_trade))).quantize(Decimal("0.01"))
    portfolio_cap_amount = (equity_for_calc * Decimal(str(portfolio_risk_cap))).quantize(Decimal("0.01"))
    group_cap_default_amount = (
        equity_for_calc * Decimal(str(group_risk_cap_default))
    ).quantize(Decimal("0.01"))
    leverage_cap_amount = (
        equity_for_calc * Decimal(str(max_portfolio_leverage))
    ).quantize(Decimal("0.01"))

    position_agg = _aggregate_positions_for_formulas(positions)
    total_notional = position_agg["total_notional"]
    current_leverage = total_notional / equity_for_calc if equity_for_calc > 0 else Decimal("0")

    return FormulasContext(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        portfolio_risk_cap=portfolio_risk_cap,
        group_risk_cap_default=group_risk_cap_default,
        max_portfolio_leverage=max_portfolio_leverage,
        soft_stop_pct=cfg_section["soft_stop_pct"],
        soft_stop_enabled=cfg_section["soft_stop_enabled"],
        unrealized_exposure_soft_cap=cfg_section["unrealized_exposure_soft_cap"],
        stop_atr_mult=cfg_section["stop_atr_mult"],
        atr_period=cfg_section["atr_period"],
        equity=equity,
        cash=cash,
        peak_equity=peak_equity,
        drawdown_from_peak=drawdown,
        snapshot_date=snapshot_date,
        risk_budget_per_trade=risk_budget_per_trade,
        portfolio_cap_amount=portfolio_cap_amount,
        group_cap_default_amount=group_cap_default_amount,
        leverage_cap_amount=leverage_cap_amount,
        positions_count=len(positions),
        total_principal_risk=position_agg["total_principal_risk"],
        total_unrealized_exposure=position_agg["total_unrealized_exposure"],
        total_notional=total_notional,
        current_leverage=current_leverage.quantize(Decimal("0.0001")),
    ).model_dump(mode="json")


async def _safe(coro, name: str, default):
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] %s failed: %s", __name__, name, exc)
        return default


def _build_formulas_config_section() -> dict[str, Any]:
    engine_cfg = build_engine_cfg_for_live()
    return {
        "initial_capital": Decimal(str(INITIAL_CAPITAL)),
        "risk_per_trade": float(engine_cfg.risk_per_trade),
        "portfolio_risk_cap": float(engine_cfg.portfolio_risk_cap),
        "group_risk_cap_default": float(engine_cfg.default_group_risk_cap),
        "max_portfolio_leverage": float(engine_cfg.max_portfolio_leverage),
        "soft_stop_pct": SOFT_STOP_PCT,
        "soft_stop_enabled": SOFT_STOP_ENABLED,
        "unrealized_exposure_soft_cap": float(engine_cfg.unrealized_exposure_soft_cap),
        "stop_atr_mult": float(engine_cfg.stop_atr_mult),
        "atr_period": int(engine_cfg.atr_period),
    }


def _aggregate_positions_for_formulas(positions: list[Any]) -> dict[str, Decimal]:
    total_principal_risk = Decimal("0")
    total_unrealized_exposure = Decimal("0")
    total_notional = Decimal("0")

    for position in positions:
        qty = int(getattr(position, "qty", 0) or 0)
        if qty == 0:
            continue
        abs_qty = Decimal(abs(qty))
        direction_sign = 1 if qty > 0 else -1
        avg_entry = Decimal(str(getattr(position, "avg_entry_price", 0) or 0))
        stop = getattr(position, "stop_loss_price", None)
        last = getattr(position, "last_price", None)
        multiplier_raw = getattr(position, "contract_multiplier", None)
        multiplier = Decimal(str(multiplier_raw)) if multiplier_raw is not None else Decimal("0")

        if stop is not None and multiplier > 0:
            stop_value = Decimal(str(stop))
            diff = (avg_entry - stop_value) * direction_sign
            if diff > 0:
                total_principal_risk += diff * multiplier * abs_qty

        if stop is not None and last is not None and multiplier > 0:
            last_value = Decimal(str(last))
            stop_value = Decimal(str(stop))
            diff = (last_value - stop_value) * direction_sign
            if diff > 0:
                total_unrealized_exposure += diff * multiplier * abs_qty

        if last is not None and multiplier > 0:
            total_notional += abs_qty * Decimal(str(last)) * multiplier

    return {
        "total_principal_risk": total_principal_risk,
        "total_unrealized_exposure": total_unrealized_exposure,
        "total_notional": total_notional,
    }


def _parse_plist_slots(plist_path: Path) -> list[tuple[str, int, int]]:
    try:
        raw = plist_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[engine-status] failed reading %s: %s", plist_path.name, exc)
        return []

    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    label_match = _RE_LABEL.search(raw)
    block_match = _RE_STARTCAL_BLOCK.search(raw)
    if not label_match or not block_match:
        return []

    label = label_match.group(1).strip()
    block = block_match.group(1)
    slots: list[tuple[str, int, int]] = []
    if block.startswith("<array>"):
        for dict_xml in [m.group(0) for m in _RE_SINGLE_DICT.finditer(block)]:
            pair = _extract_hour_minute(dict_xml)
            if pair is not None:
                slots.append((label, pair[0], pair[1]))
    else:
        pair = _extract_hour_minute(block)
        if pair is not None:
            slots.append((label, pair[0], pair[1]))
    return slots


def _extract_hour_minute(dict_xml: str) -> tuple[int, int] | None:
    hour_match = _RE_HOUR.search(dict_xml)
    minute_match = _RE_MINUTE.search(dict_xml)
    if not hour_match or not minute_match:
        return None
    return int(hour_match.group(1)), int(minute_match.group(1))


def _next_fire(now_local: datetime, hour: int, minute: int) -> datetime:
    candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now_local:
        candidate = candidate + timedelta(days=1)
    return candidate


def _build_launchd_schedule() -> list[LaunchdSlot]:
    now_local = datetime.now(ZoneInfo(APP_TIMEZONE))
    out: list[LaunchdSlot] = []
    for plist in sorted(_PLIST_DIR.glob("*.plist")):
        for label, hour, minute in _parse_plist_slots(plist):
            out.append(
                LaunchdSlot(
                    label=label,
                    hour=hour,
                    minute=minute,
                    description=_PLIST_DESCRIPTIONS.get(label, label),
                    next_fire=_next_fire(now_local, hour, minute),
                )
            )
    out.sort(key=lambda slot: slot.next_fire)
    return out


def _jsonize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    from datetime import date as _d, datetime as _dt
    from decimal import Decimal as _Decimal
    from uuid import UUID

    out: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, UUID):
            out[key] = str(value)
        elif isinstance(value, _Decimal):
            out[key] = float(value)
        elif isinstance(value, (_dt, _d)):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out
