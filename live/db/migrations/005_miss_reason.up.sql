-- Migration 005: signal_diagnostics 加 miss_reason 字段
-- 当 entry_trigger=False 时，描述"因为什么具体条件没满足"。
-- 示例值：
--   "double_ma: 快线 2806.9 < 慢线 2963.2 差 -5.3%（空头势，未上穿）"
--   "hl_21: close 7855 未破 21日高 9172(差 -5.8σ) / 低 7823(差 +0.1σ)"
--   "boll: close 147.8 在带内 [145.2, 153.1]，破上差 4.7σ"
BEGIN;
ALTER TABLE signal_diagnostics ADD COLUMN IF NOT EXISTS miss_reason TEXT;
COMMIT;
