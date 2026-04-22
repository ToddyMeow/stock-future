-- Migration 001: initial schema (rollback)
-- Usage: psql -U postgres -d stock_future -f 001_initial.down.sql
-- 顺序：先 views，再表（FK 依赖反向），最后 function

DROP VIEW  IF EXISTS v_group_exposure              CASCADE;
DROP VIEW  IF EXISTS v_instructions_with_fills     CASCADE;

DROP TABLE IF EXISTS alerts                        CASCADE;
DROP TABLE IF EXISTS bars                          CASCADE;
DROP TABLE IF EXISTS daily_pnl                     CASCADE;
DROP TABLE IF EXISTS fills                         CASCADE;  -- 依赖 instructions
DROP TABLE IF EXISTS instructions                  CASCADE;
DROP TABLE IF EXISTS positions                     CASCADE;

DROP FUNCTION IF EXISTS set_updated_at()           CASCADE;
