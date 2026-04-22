BEGIN;
ALTER TABLE signal_diagnostics DROP COLUMN IF EXISTS miss_reason;
COMMIT;
