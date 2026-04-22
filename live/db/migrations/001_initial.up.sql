-- Migration 001: initial schema (up)
-- 创建 6 张表 + 触发器 + 视图（positions / instructions / fills / daily_pnl / bars / alerts）
-- Usage: psql -U postgres -d stock_future -f 001_initial.up.sql
\i ../schema.sql
