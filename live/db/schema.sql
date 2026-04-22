-- =====================================================================
-- 期货半自动实盘交易系统 — PostgreSQL Schema
-- =====================================================================
-- Target:   PostgreSQL 16+
-- Purpose:  半自动模式账户状态 —
--           指令 / 成交回填 / 持仓 / 每日权益 / K线 / 审计告警
-- Design:   3NF · snake_case · 表名复数 · TIMESTAMPTZ 统一 UTC
-- Keys:     PG14+ 原生 gen_random_uuid()；BIGSERIAL 用于事件 / bars
-- Scope:    6 张表（positions, instructions, fills, daily_pnl, bars, alerts）
-- =====================================================================

-- ---------------------------------------------------------------------
-- Extensions (PG 14+ 原生 gen_random_uuid，无需 pgcrypto)
-- ---------------------------------------------------------------------
-- 无

-- ---------------------------------------------------------------------
-- Utility: updated_at 自动更新触发器
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- =====================================================================
-- Table 1: positions  — 当前持仓（每 symbol × 合约代码一行，空仓删除）
-- =====================================================================
CREATE TABLE positions (
  symbol            VARCHAR(10)     NOT NULL,
  contract_code     VARCHAR(20)     NOT NULL,               -- AO2505 等真实合约
  qty               INTEGER         NOT NULL,                -- 正=多 负=空
  avg_entry_price   NUMERIC(14, 4)  NOT NULL,                -- 成本均价（含 roll 调整）
  stop_loss_price   NUMERIC(14, 4),                          -- 当前动态止损；NULL 表无止损
  group_name        VARCHAR(32)     NOT NULL,                -- agri / metals / ind_AP ...
  opened_at         TIMESTAMPTZ     NOT NULL,
  last_updated_at   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
  notes             TEXT,

  CONSTRAINT pk_positions            PRIMARY KEY (symbol, contract_code),
  CONSTRAINT chk_positions_qty       CHECK (qty <> 0),       -- 0 手=删除行
  CONSTRAINT chk_positions_price_pos CHECK (avg_entry_price > 0)
);

COMMENT ON TABLE  positions IS '当前未平仓头寸；空仓删除，不保留 qty=0 行';
COMMENT ON COLUMN positions.qty IS '正=多头手数，负=空头手数';
COMMENT ON COLUMN positions.stop_loss_price IS '动态止损价；NULL 表示策略本阶段不挂止损';
COMMENT ON COLUMN positions.group_name IS '组名，用于组级风险暴露约束';

CREATE INDEX idx_positions_group ON positions (group_name);
-- rationale: 按组聚合查仓位暴露度（group_cap 检查）

-- trigger
CREATE TRIGGER trg_positions_updated_at
  BEFORE UPDATE ON positions
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- =====================================================================
-- Table 2: instructions  — 每日调仓指令（信号服务产出）
-- =====================================================================
CREATE TABLE instructions (
  id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
  generated_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
  session_date           DATE            NOT NULL,           -- 业务日（信号所属交易日）
  session                VARCHAR(5)      NOT NULL,           -- day / night
  symbol                 VARCHAR(10)     NOT NULL,
  contract_code          VARCHAR(20)     NOT NULL,
  action                 VARCHAR(10)     NOT NULL,           -- open / close / add / reduce
  direction              VARCHAR(5)      NOT NULL,           -- long / short
  target_qty             INTEGER         NOT NULL,           -- 本笔目标调整量（正数；方向看 action+direction）
  entry_price_ref        NUMERIC(14, 4),                     -- 参考入场价（信号生成时快照）
  stop_loss_ref          NUMERIC(14, 4),                     -- 参考止损
  group_name             VARCHAR(32)     NOT NULL,
  status                 VARCHAR(20)     NOT NULL DEFAULT 'pending',
  veto_reason            TEXT,
  broker_stop_order_id   VARCHAR(40),                        -- 客户端止损单号（可选，用户手填便于对账）
  created_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
  updated_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

  CONSTRAINT chk_instructions_session    CHECK (session IN ('day', 'night')),
  CONSTRAINT chk_instructions_action     CHECK (action IN ('open', 'close', 'add', 'reduce')),
  CONSTRAINT chk_instructions_direction  CHECK (direction IN ('long', 'short')),
  CONSTRAINT chk_instructions_target_qty CHECK (target_qty > 0),
  CONSTRAINT chk_instructions_status     CHECK (
    status IN ('pending', 'fully_filled', 'partially_filled', 'vetoed', 'skipped', 'expired')
  ),
  CONSTRAINT chk_instructions_veto       CHECK (
    (status = 'vetoed' AND veto_reason IS NOT NULL AND length(trim(veto_reason)) > 0)
    OR status <> 'vetoed'
  ),
  CONSTRAINT uq_instructions_session_sym UNIQUE (session_date, session, symbol, contract_code, action)
  -- rationale: 同日同 session 同 symbol 同 action 只允许一条（防重复插入）
);

COMMENT ON TABLE  instructions IS '每日信号服务产出的调仓指令，状态机驱动';
COMMENT ON COLUMN instructions.session_date IS '信号所属交易日（非创建日）';
COMMENT ON COLUMN instructions.status IS
  '状态机：pending → fully_filled / partially_filled / vetoed / skipped / expired';
COMMENT ON COLUMN instructions.broker_stop_order_id IS
  '客户端挂的止损单号，用户手填；可选，便于盘后对账';

CREATE INDEX idx_instructions_date_session
  ON instructions (session_date, session);
-- rationale: 前端按日期+盘次查今日指令（主要查询路径）

CREATE INDEX idx_instructions_status_pending
  ON instructions (status)
  WHERE status = 'pending';
-- rationale: 局部索引，调度器轮询未处理指令

CREATE INDEX idx_instructions_symbol
  ON instructions (symbol, session_date DESC);
-- rationale: 按品种回溯历史指令

CREATE TRIGGER trg_instructions_updated_at
  BEFORE UPDATE ON instructions
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- =====================================================================
-- Table 3: fills  — 成交回填（一条 instruction 可多次 fill 支持部分成交）
-- =====================================================================
CREATE TABLE fills (
  id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
  instruction_id    UUID            NOT NULL,
  filled_qty        INTEGER         NOT NULL,                -- 本次成交量（正数）
  filled_price      NUMERIC(14, 4)  NOT NULL,
  filled_at         TIMESTAMPTZ     NOT NULL,                -- 实际成交时间（用户手填）
  trigger_source    VARCHAR(20)     NOT NULL DEFAULT 'user_manual',
  note              TEXT,
  created_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

  CONSTRAINT fk_fills_instruction   FOREIGN KEY (instruction_id) REFERENCES instructions(id) ON DELETE RESTRICT,
  CONSTRAINT chk_fills_qty_positive CHECK (filled_qty > 0),
  CONSTRAINT chk_fills_price_pos    CHECK (filled_price > 0),
  CONSTRAINT chk_fills_trigger      CHECK (
    trigger_source IN ('user_manual', 'stop_loss', 'take_profit', 'roll_contract')
  )
);

COMMENT ON TABLE  fills IS '成交明细；每条 instruction 可多次 fill（部分成交）';
COMMENT ON COLUMN fills.filled_qty IS '本次成交量，正数；总成交 = sum(filled_qty)';
COMMENT ON COLUMN fills.trigger_source IS
  '成交触发来源 — user_manual（用户手动）/ stop_loss（盘中止损触发）/ take_profit / roll_contract（主力切换）';

CREATE INDEX idx_fills_instruction ON fills (instruction_id);
-- rationale: 查一条 instruction 的全部 fills（状态推断 + UI 展示）

CREATE INDEX idx_fills_filled_at ON fills (filled_at DESC);
-- rationale: 按时间查历史成交（账单视图）

CREATE INDEX idx_fills_trigger_source ON fills (trigger_source, filled_at DESC);
-- rationale: 止损 / 止盈归因统计


-- =====================================================================
-- Table 4: daily_pnl  — 每日账户快照（权益曲线 + soft stop 判定源）
-- =====================================================================
CREATE TABLE daily_pnl (
  date                    DATE            PRIMARY KEY,
  equity                  NUMERIC(16, 2)  NOT NULL,           -- 当日结算权益
  cash                    NUMERIC(16, 2)  NOT NULL,
  open_positions_mv       NUMERIC(16, 2)  NOT NULL,           -- 持仓市值
  realized_pnl_today      NUMERIC(16, 2)  NOT NULL DEFAULT 0,
  unrealized_pnl_today    NUMERIC(16, 2)  NOT NULL DEFAULT 0,
  soft_stop_triggered     BOOLEAN         NOT NULL DEFAULT FALSE,
  drawdown_from_peak      NUMERIC(6, 4),                      -- 0.0530 = 5.30%
  peak_equity_to_date     NUMERIC(16, 2),
  notes                   TEXT,
  created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
  updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

  CONSTRAINT chk_daily_pnl_equity_pos CHECK (equity >= 0)
);

COMMENT ON TABLE  daily_pnl IS '每日结算快照，用于权益曲线 + soft stop 判定';
COMMENT ON COLUMN daily_pnl.drawdown_from_peak IS '当前回撤比例，小数（0.053 = 5.3%）';
COMMENT ON COLUMN daily_pnl.soft_stop_triggered IS '当日是否触发 soft stop（降级只产 close）';

CREATE INDEX idx_daily_pnl_soft_stop
  ON daily_pnl (date DESC)
  WHERE soft_stop_triggered = TRUE;
-- rationale: 查最近 soft stop 触发日（告警/报表）

CREATE TRIGGER trg_daily_pnl_updated_at
  BEFORE UPDATE ON daily_pnl
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- =====================================================================
-- Table 5: bars  — K 线数据（历史 + 实盘 单一来源；22 列同源 hab_bars.csv）
-- =====================================================================
-- PK: (order_book_id, date) — 同一真实合约 × 日期唯一
--     symbol 级主力序列可能跨合约滚动，故用 order_book_id 而非 symbol 做主键
-- =====================================================================
CREATE TABLE bars (
  date                 DATE              NOT NULL,
  symbol               VARCHAR(10)       NOT NULL,           -- 品种代码（主力序列，如 AG / AO）
  order_book_id        VARCHAR(20)       NOT NULL,           -- 真实合约代码（如 AG1806 / AO2505）
  open                 NUMERIC(14, 4)    NOT NULL,
  high                 NUMERIC(14, 4)    NOT NULL,
  low                  NUMERIC(14, 4)    NOT NULL,
  close                NUMERIC(14, 4)    NOT NULL,
  settle               NUMERIC(14, 4)    NOT NULL,           -- 结算价
  volume               DOUBLE PRECISION  NOT NULL,
  open_interest        DOUBLE PRECISION  NOT NULL,
  contract_multiplier  NUMERIC(10, 4)    NOT NULL,           -- 合约乘数（每手对应的标的数量）
  commission           NUMERIC(10, 4)    NOT NULL,           -- 手续费（每手元）
  slippage             NUMERIC(10, 4)    NOT NULL,           -- 滑点（tick 数）
  group_name           VARCHAR(32)       NOT NULL,           -- 分组（agri / metals ...）
  margin_rate          NUMERIC(6, 4)     NOT NULL,           -- 保证金率（0.1 = 10%）
  open_raw             NUMERIC(14, 4),                       -- 未经 roll 调整的原始价
  high_raw             NUMERIC(14, 4),
  low_raw              NUMERIC(14, 4),
  close_raw            NUMERIC(14, 4),
  settle_raw           NUMERIC(14, 4),
  limit_up             NUMERIC(14, 4),                       -- 涨停价
  limit_down           NUMERIC(14, 4),                       -- 跌停价
  created_at           TIMESTAMPTZ       NOT NULL DEFAULT NOW(),

  CONSTRAINT pk_bars               PRIMARY KEY (order_book_id, date),
  CONSTRAINT chk_bars_hl           CHECK (high >= low),
  CONSTRAINT chk_bars_volume_nn    CHECK (volume >= 0),
  CONSTRAINT chk_bars_oi_nn        CHECK (open_interest >= 0),
  CONSTRAINT chk_bars_margin_range CHECK (margin_rate > 0 AND margin_rate < 1),
  CONSTRAINT chk_bars_multiplier   CHECK (contract_multiplier > 0)
);

COMMENT ON TABLE  bars IS
  'K 线日线数据（历史 2018- 至今 + 实盘每日追加，单一来源）；22 列与 hab_bars.csv 同源';
COMMENT ON COLUMN bars.symbol IS '品种主力序列代码，如 AG；主力切换时同一 symbol 会换 order_book_id';
COMMENT ON COLUMN bars.order_book_id IS '真实期货合约代码，如 AG1806 / AO2505';
COMMENT ON COLUMN bars.settle IS '结算价 — 每日强平/保证金判定基准';
COMMENT ON COLUMN bars.close_raw IS '未经主力连续调整的原始收盘价（调试 / 审计用）';
COMMENT ON COLUMN bars.limit_up IS '当日涨停价；超过此价无法开多仓（实盘预警用）';
COMMENT ON COLUMN bars.contract_multiplier IS '每手对应的标的数量（如 AG 是 15 kg/手）';
COMMENT ON COLUMN bars.margin_rate IS '保证金率小数（0.1 = 10%）';

CREATE INDEX idx_bars_symbol_date
  ON bars (symbol, date DESC);
-- rationale: signal_service 按品种查最近 N 日（主力连续）做 warmup

CREATE INDEX idx_bars_date
  ON bars (date DESC);
-- rationale: 按日期跨品种查（portfolio 聚合 / 当日快照）


-- =====================================================================
-- Table 6: alerts  — 审计日志 / 告警（之前叫 system_events，plan 统一叫 alerts）
-- =====================================================================
CREATE TABLE alerts (
  id            BIGSERIAL       PRIMARY KEY,
  event_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
  severity      VARCHAR(10)     NOT NULL,
  event_type    VARCHAR(40)     NOT NULL,                    -- data_fetch_failed / soft_stop_triggered / alert_sent / ...
  message       TEXT            NOT NULL,
  payload       JSONB,                                        -- 结构化上下文

  CONSTRAINT chk_alerts_severity CHECK (severity IN ('info', 'warn', 'critical'))
);

COMMENT ON TABLE  alerts IS '系统事件审计日志 / 告警链路源；Server 酱 + SMS 下游订阅者';
COMMENT ON COLUMN alerts.payload IS '结构化上下文 JSONB — 灵活承载 symbol / 错误码 / 阈值等';

CREATE INDEX idx_alerts_at
  ON alerts (event_at DESC);
-- rationale: 时间倒序查最近事件（运维面板主要查询）

CREATE INDEX idx_alerts_critical
  ON alerts (event_at DESC)
  WHERE severity = 'critical';
-- rationale: 局部索引加速 critical 告警查询

CREATE INDEX idx_alerts_type
  ON alerts (event_type, event_at DESC);
-- rationale: 按事件类型统计（告警去重 / 失败频率）


-- =====================================================================
-- Views (便捷视图)
-- =====================================================================

-- 指令带聚合成交量 —— 便于 UI 和 status 推断
CREATE OR REPLACE VIEW v_instructions_with_fills AS
SELECT
  i.*,
  COALESCE(SUM(f.filled_qty), 0)::INTEGER                       AS filled_qty_total,
  COUNT(f.id)                                                    AS fill_count,
  CASE
    WHEN COUNT(f.id) = 0                                 THEN NULL
    ELSE SUM(f.filled_qty * f.filled_price) / NULLIF(SUM(f.filled_qty), 0)
  END                                                            AS avg_filled_price
FROM instructions i
LEFT JOIN fills f ON f.instruction_id = i.id
GROUP BY i.id;

COMMENT ON VIEW v_instructions_with_fills IS '指令 + 聚合成交明细（UI 列表 / status 推断源）';


-- 当前组内风险暴露（portfolio_cap / group_cap 决策源）
CREATE OR REPLACE VIEW v_group_exposure AS
SELECT
  group_name,
  COUNT(*)                                                       AS position_count,
  SUM(qty * avg_entry_price)                                     AS total_notional,
  SUM(CASE WHEN qty > 0 THEN qty * avg_entry_price ELSE 0 END)   AS long_notional,
  SUM(CASE WHEN qty < 0 THEN -qty * avg_entry_price ELSE 0 END)  AS short_notional
FROM positions
GROUP BY group_name;

COMMENT ON VIEW v_group_exposure IS '按 group_name 聚合的当前持仓名义暴露（多 / 空分列）';


-- =====================================================================
-- END
-- =====================================================================
