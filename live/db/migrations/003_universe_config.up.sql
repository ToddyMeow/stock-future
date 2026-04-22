-- Migration 003: universe config (symbols + group→combo mapping)
--
-- Replaces the CSV-based pipeline artefacts used by /api/universe:
--   data/runs/phase0_250k/final_v3_comparison.csv    (13 symbols, group_name)
--   data/runs/phase3/best_combos_stable_final_v3.csv (per-group combo binding)
--
-- Publish workflow:
--   1. Strategy research produces CSVs (unchanged).
--   2. Run `python scripts/publish_universe_to_db.py --csv-symbols ... --csv-combos ...`
--      → TRUNCATE + INSERT into these tables inside a single transaction.
--   3. /api/universe reads from these tables on every request.
--
-- The signal_service (launchd on Mac) still reads the CSV directly for now;
-- it will switch to DB in a later migration when the scheduler moves to ECS.

BEGIN;

CREATE TABLE IF NOT EXISTS universe_group_combos (
    group_name  TEXT        PRIMARY KEY,
    combo       TEXT        NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS universe_symbols (
    symbol      TEXT        PRIMARY KEY,
    group_name  TEXT        NOT NULL REFERENCES universe_group_combos(group_name) ON UPDATE CASCADE,
    enabled     BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_universe_symbols_group ON universe_symbols(group_name);
CREATE INDEX IF NOT EXISTS idx_universe_symbols_enabled ON universe_symbols(enabled) WHERE enabled;

COMMIT;
