-- Migration 004: signal_diagnostics
--
-- 每次 signal_service 跑完 engine 后，把 engine 的 daily_status 最后一个 bar
-- (T-1 close = 当次 session 对应的信号 bar) 的**每 symbol × strategy_id**
-- 诊断信息 upsert 到这张表。目的：让盯盘品种页看见"今天为什么没信号"
-- —— 是没 trigger（市场没机会），还是被 group_cap / portfolio_cap / other 拒了。
--
-- 字段语义：
--   entry_trigger  : engine 在这个 bar 上是否判定了 entry（价格突破 / 均线交叉 等策略条件）
--   entry_direction: 1 = long, -1 = short, NULL = 没 trigger
--   reject_reason  : entry_trigger=true 后被风控拒的原因（NULL 表示通过，对应生成 pending_entry）
--   reject_reason  内容枚举：
--     PORTFOLIO_RISK_CAP / GROUP_RISK_CAP / INDEPENDENT_SOFT_CAP / LEVERAGE_CAP / MARGIN_CAP
--     / QTY_LT_1 / ATR_BELOW_FLOOR / CONGESTION_LOCKED / ALREADY_IN_POSITION
--     / PENDING_ENTRY_EXISTS / SYMBOL_LOCKED / WARMUP_INSUFFICIENT / NO_NEXT_TRADE_DATE
--     / NON_POSITIVE_RISK

BEGIN;

CREATE TABLE IF NOT EXISTS signal_diagnostics (
    session_date    DATE         NOT NULL,
    session         VARCHAR(10)  NOT NULL,
    symbol          VARCHAR(16)  NOT NULL,
    strategy_id     VARCHAR(128) NOT NULL,
    group_name      VARCHAR(32)  NOT NULL,
    bar_date        DATE,
    close_price     NUMERIC,
    atr             NUMERIC,
    entry_trigger   BOOLEAN      NOT NULL DEFAULT FALSE,
    entry_direction INT,
    reject_reason   VARCHAR(48),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_date, session, symbol, strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_signal_diag_session_date
    ON signal_diagnostics(session_date DESC, session);
CREATE INDEX IF NOT EXISTS idx_signal_diag_symbol
    ON signal_diagnostics(symbol, session_date DESC);

COMMIT;
