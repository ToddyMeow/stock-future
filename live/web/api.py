"""Thin FastAPI routing layer for the live trading API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date as _date
import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from live import account_state
from live.db.models import FillCreate
from live.web.schemas import RollConfirmRequest, RollConfirmResponse, VetoBody
from live.web.services import (
    build_engine_status_payload,
    build_formulas_context_payload,
    build_report_payload,
    build_universe_payload,
)


logger = logging.getLogger(__name__)
__version__ = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await account_state.ping_db()
        logger.info("[lifespan] DB reachable")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[lifespan] DB unreachable (continuing): %s", exc)
    yield
    logger.info("[lifespan] shutting down")


app = FastAPI(
    title="velvet-anchor-api",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    try:
        await account_state.ping_db()
        db_status = "reachable"
    except Exception as exc:  # noqa: BLE001
        db_status = f"unreachable: {type(exc).__name__}"
    return {"status": "ok", "db": db_status, "version": __version__}


@app.get("/api/instructions")
async def list_instructions(
    date: _date = Query(..., description="session_date"),
    session: Optional[str] = Query(None, pattern="^(day|night)$"),
) -> list[dict[str, Any]]:
    try:
        rows = await account_state.get_instructions_with_fills(date, session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[/api/instructions] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [_jsonize_mapping(row) for row in rows]


@app.post("/api/instructions/{id_}/veto")
async def veto_instruction(id_: UUID, body: VetoBody) -> dict[str, Any]:
    try:
        inst = await account_state.veto_instruction(id_, body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("[veto] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return inst.model_dump(mode="json")


@app.post("/api/instructions/{id_}/skip")
async def skip_instruction(id_: UUID) -> dict[str, Any]:
    try:
        inst = await account_state.skip_instruction(id_)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("[skip] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return inst.model_dump(mode="json")


@app.post("/api/fills")
async def post_fill(fill: FillCreate) -> dict[str, Any]:
    try:
        new_fill = await account_state.apply_fill(fill)
        inst = await account_state.get_instruction(fill.instruction_id)
        return {
            "fill": new_fill.model_dump(mode="json"),
            "instruction": inst.model_dump(mode="json") if inst else None,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("[post_fill] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/positions")
async def list_positions() -> list[dict[str, Any]]:
    try:
        positions = await account_state.get_positions_enriched()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[positions] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [position.model_dump(mode="json") for position in positions]


@app.get("/api/rolls")
async def list_roll_candidates() -> list[dict[str, Any]]:
    try:
        rolls = await account_state.get_roll_candidates()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[rolls] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [roll.model_dump(mode="json") for roll in rolls]


@app.post("/api/rolls/confirm")
async def confirm_roll(body: RollConfirmRequest) -> dict[str, Any]:
    try:
        result = await account_state.apply_roll(
            symbol=body.symbol,
            old_contract=body.old_contract,
            new_contract=body.new_contract,
            old_close_price=body.old_close_price,
            new_open_price=body.new_open_price,
            closed_at=body.closed_at,
            opened_at=body.opened_at,
            note=body.note,
        )
    except ValueError as exc:
        message = str(exc)
        if "positions 中无" in message:
            raise HTTPException(status_code=404, detail=message) from exc
        raise HTTPException(status_code=400, detail=message) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("[rolls/confirm] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RollConfirmResponse(
        symbol=result["symbol"],
        old_contract=result["old_contract"],
        new_contract=result["new_contract"],
        qty=result["qty"],
        old_close_price=result["old_close_price"],
        new_open_price=result["new_open_price"],
        closed_instruction_id=result["closed_instruction_id"],
        opened_instruction_id=result["opened_instruction_id"],
        new_position=result["new_position"],
    ).model_dump(mode="json")


@app.get("/api/daily_pnl")
async def list_daily_pnl(
    from_: _date = Query(..., alias="from"),
    to: _date = Query(...),
) -> list[dict[str, Any]]:
    try:
        rows = await account_state.get_daily_pnl_range(from_, to)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[daily_pnl] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [row.model_dump(mode="json") for row in rows]


@app.get("/api/history")
async def history_by_date(date: _date = Query(...)) -> dict[str, Any]:
    try:
        instructions = await account_state.get_instructions_with_fills(date)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[history] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "date": date.isoformat(),
        "instructions": [_jsonize_mapping(row) for row in instructions],
    }


@app.get("/api/reports/{report_date}")
async def report_json(report_date: _date) -> dict[str, Any]:
    try:
        return await build_report_payload(report_date)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[report] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/alerts/recent")
async def recent_alerts(
    n: int = Query(50, ge=1, le=500),
    severity: Optional[str] = Query(None, pattern="^(info|warn|critical)$"),
) -> list[dict[str, Any]]:
    try:
        rows = await account_state.get_recent_alerts(n, severity)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[alerts] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [row.model_dump(mode="json") for row in rows]


@app.get("/api/analytics/breakdown")
async def analytics_breakdown() -> dict[str, list]:
    return {"by_symbol": [], "by_group": []}


@app.get("/api/universe")
async def list_universe() -> list[dict[str, Any]]:
    try:
        return await build_universe_payload()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[universe] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/engine-status")
async def engine_status() -> dict[str, Any]:
    try:
        return await build_engine_status_payload()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[engine-status] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/formulas-context")
async def formulas_context() -> dict[str, Any]:
    try:
        return await build_formulas_context_payload()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[formulas-context] failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _jsonize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    from datetime import date as _d, datetime as _dt
    from decimal import Decimal as _Decimal

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
