-- Migration 002: engine_states 表（up）
-- 目的：存储 StrategyEngine.save_state() 输出的 JSON 快照，用于 signal_service 增量续跑
-- 主键：(session_date, session) — 每个交易日的每 session 一行，幂等 UPSERT
-- 依赖：001_initial (仅函数依赖；本表无 FK)
-- Usage: psql -U ... -d ... -f 002_engine_states.up.sql

CREATE TABLE engine_states (
  session_date    DATE          NOT NULL,
  session         VARCHAR(5)    NOT NULL,              -- 'day' | 'night'
  state           JSONB         NOT NULL,              -- engine.save_state() 输出
  created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  PRIMARY KEY (session_date, session),
  CONSTRAINT chk_engine_states_session CHECK (session IN ('day', 'night'))
);

COMMENT ON TABLE  engine_states IS 'Engine 增量续跑所需的 state 快照（每 session 一行）';
COMMENT ON COLUMN engine_states.session_date IS '业务日（state 所属交易日）';
COMMENT ON COLUMN engine_states.session IS '日盘 day / 夜盘 night';
COMMENT ON COLUMN engine_states.state IS 'StrategyEngine.save_state() 输出的 JSON 快照（含 positions / cash / pending_entries / last_date 等）';

CREATE INDEX idx_engine_states_created ON engine_states (created_at DESC);
-- rationale: 按创建时间倒序查最近快照（调度巡检）
