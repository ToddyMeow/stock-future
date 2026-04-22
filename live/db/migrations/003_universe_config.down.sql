-- Down migration 003: drop universe config tables.
BEGIN;
DROP TABLE IF EXISTS universe_symbols;
DROP TABLE IF EXISTS universe_group_combos;
COMMIT;
