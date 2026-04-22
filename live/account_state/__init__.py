"""Account-state facade.

Public imports stay stable (`from live import account_state`) while the
implementation is split by responsibility.
"""

from live.account_state.db import (
    get_engine,
    get_sessionmaker,
    ping_db,
    set_engine,
)
from live.account_state.queries import (
    get_daily_pnl_range,
    get_instruction,
    get_instructions,
    get_instructions_with_fills,
    get_latest_bar_date,
    get_position,
    get_positions,
    get_positions_enriched,
    get_recent_alerts,
    get_roll_candidates,
)
from live.account_state.repositories import (
    delete_position,
    insert_alert,
    mark_soft_stop_triggered,
    record_daily_pnl,
    upsert_position,
)
from live.account_state.services import (
    apply_fill,
    apply_roll,
    expire_pending_before,
    skip_instruction,
    veto_instruction,
)

__all__ = [
    "get_engine",
    "set_engine",
    "get_sessionmaker",
    "ping_db",
    "get_positions",
    "get_positions_enriched",
    "get_position",
    "upsert_position",
    "delete_position",
    "get_instructions",
    "get_instructions_with_fills",
    "get_instruction",
    "veto_instruction",
    "skip_instruction",
    "expire_pending_before",
    "apply_fill",
    "record_daily_pnl",
    "get_daily_pnl_range",
    "mark_soft_stop_triggered",
    "get_latest_bar_date",
    "get_roll_candidates",
    "apply_roll",
    "insert_alert",
    "get_recent_alerts",
]
