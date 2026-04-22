-- Migration 002: engine_states 表（回滚）
-- Usage: psql -U ... -d ... -f 002_engine_states.down.sql

DROP TABLE IF EXISTS engine_states CASCADE;
