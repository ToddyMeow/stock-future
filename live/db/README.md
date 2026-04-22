# `live/db/` — 半自动实盘交易数据库架构

**目标数据库**：PostgreSQL 16+
**核心文件**：
- [`schema.sql`](schema.sql) — 完整建表 + 索引 + 约束 + 视图（6 表）
- [`models.py`](models.py) — Pydantic v2 模型（读/写/枚举）
- [`migrations/001_initial.up.sql`](migrations/001_initial.up.sql) — 初始化
- [`migrations/001_initial.down.sql`](migrations/001_initial.down.sql) — rollback

---

## 一、ER 图（ASCII）

```
        ┌───────────────────────────────┐
        │          instructions         │
        ├───────────────────────────────┤
        │ id                 UUID  (PK) │
        │ generated_at       TZ         │
        │ session_date       DATE       │
        │ session            day|night  │
        │ symbol             VARCHAR    │
        │ contract_code      VARCHAR    │───┐
        │ action             open/...   │   │ 1
        │ direction          long/short │   │
        │ target_qty         INT        │   │
        │ entry_price_ref    NUMERIC    │   │
        │ stop_loss_ref      NUMERIC    │   │
        │ group_name         VARCHAR    │   │
        │ status             (enum6)    │   │
        │ veto_reason        TEXT       │   │
        │ broker_stop_order_id VARCHAR  │   │
        └───────────────────────────────┘   │
                                            │ N
                                            ▼
  ┌──────────────────────────┐     ┌───────────────────────────┐
  │        positions         │     │            fills          │
  ├──────────────────────────┤     ├───────────────────────────┤
  │ symbol         (PK)      │     │ id              UUID (PK) │
  │ contract_code  (PK)      │     │ instruction_id  UUID (FK) │
  │ qty            INT       │     │ filled_qty      INT       │
  │ avg_entry_price NUMERIC  │     │ filled_price    NUMERIC   │
  │ stop_loss_price NUMERIC  │     │ filled_at       TZ        │
  │ group_name     VARCHAR   │     │ trigger_source  (enum4)   │
  │ opened_at      TZ        │     │ note            TEXT      │
  │ last_updated_at TZ       │     └───────────────────────────┘
  └──────────────────────────┘

  ┌──────────────────────────┐     ┌───────────────────────────┐
  │       daily_pnl          │     │          alerts           │
  ├──────────────────────────┤     ├───────────────────────────┤
  │ date           (PK)      │     │ id           BIGSERIAL PK │
  │ equity                   │     │ event_at     TZ           │
  │ cash                     │     │ severity     info/warn/c  │
  │ open_positions_mv        │     │ event_type   VARCHAR      │
  │ realized_pnl_today       │     │ message      TEXT         │
  │ unrealized_pnl_today     │     │ payload      JSONB        │
  │ soft_stop_triggered      │     └───────────────────────────┘
  │ drawdown_from_peak       │
  │ peak_equity_to_date      │
  └──────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │                           bars                             │
  ├────────────────────────────────────────────────────────────┤
  │ order_book_id   VARCHAR  (PK)                              │
  │ date            DATE     (PK)                              │
  │ symbol          VARCHAR                                    │
  │ open/high/low/close/settle   NUMERIC                       │
  │ volume / open_interest       DOUBLE PRECISION              │
  │ contract_multiplier / commission / slippage   NUMERIC      │
  │ group_name / margin_rate                                   │
  │ open_raw / high_raw / low_raw / close_raw / settle_raw     │
  │ limit_up / limit_down        NUMERIC                       │
  │                                                            │
  │ 历史 2018-至今 + 实盘每日追加，单一来源（22 列同 hab_bars.csv）│
  └────────────────────────────────────────────────────────────┘

  (positions / daily_pnl / bars / alerts 之间无外键，独立维度)
```

---

## 二、表职责

| 表 | 职责 | 写入方 | 读取方 |
|---|---|---|---|
| `positions` | 当前活跃持仓快照（空仓删除） | `account_state.apply_fill()` / `signal_service` 平仓 | 前端持仓页、`signal_service` 算目标仓位 |
| `instructions` | 每日调仓指令（含状态机 + 止损单号） | `signal_service.generate()` | 前端指令页、`account_state.apply_fill` |
| `fills` | 成交回填明细（支持部分成交 + 归因） | 前端 POST `/api/fills` | 前端历史页、status 推断、止损归因 |
| `daily_pnl` | 每日结算快照 | 日末结算 job | 权益曲线、`soft_stop.check_and_update` |
| `bars` | K 线数据（历史 + 实盘单一来源） | 用户一次性上传 + `data_pipeline` 每日 append | `signal_service` warmup、前端分析 |
| `alerts` | 审计日志 / 告警 | 所有模块（告警 / 失败 / 触发） | 运维面板、Server 酱 / SMS 下游 |

---

## 三、状态机（instructions.status）

```
            ┌──────────┐
            │ pending  │  （signal_service 新建时默认）
            └────┬─────┘
                 │
     ┌───────────┼─────────────────┬───────────┬───────────┐
     ▼           ▼                 ▼           ▼           ▼
┌──────────┐ ┌────────────────┐ ┌────────┐ ┌────────┐ ┌─────────┐
│fully_    │ │partially_      │ │vetoed  │ │skipped │ │expired  │
│filled    │ │filled          │ │        │ │        │ │         │
└──────────┘ └────────────────┘ └────────┘ └────────┘ └─────────┘
   │              │                  │         │           │
   │              │                  └ veto_   │           └ 次日 08:30 仍
   │              │                    reason  │             pending 则 job
   │              │                    NOT NULL│             标 expired
   │              │                    （CHECK）│
   │              │                            │
   │              └── sum(fills) < target      └── 用户主动跳过
   │              └── 可再追加 fill → fully_filled
   │
   └── sum(fills) >= target
```

**状态转移由 app 层维护**（`live/account_state.py`），推荐规则：
- POST `/api/fills`：插入 `fills` 行；根据 `sum(filled_qty)` 与 `target_qty` 更新 `status`
  - `sum(filled_qty) >= target_qty` → `fully_filled`
  - `0 < sum(filled_qty) < target_qty` → `partially_filled`
- POST `/api/instructions/{id}/veto`：`status=vetoed` + `veto_reason` 必填（DB 层 CHECK 兜底）
- POST `/api/instructions/{id}/skip`：`status=skipped`
- 定时 job：次日开盘前把超时 `pending` / `partially_filled` 标 `expired`（partial 不特意补）

---

## 四、关键设计决策（Rationale）

### D1. `positions` 的 PK 是 (symbol, contract_code) 不是 UUID
- **为什么**：合约切换（AO2505 → AO2509）时新合约 = 新行 = 天然隔离；查询按 (symbol, contract_code) 直接走 PK
- **权衡**：不支持"同合约多段开仓"（多次加仓合并为一行）。一期策略不需要，避免过设计

### D2. 空仓删除 `positions` 行（`chk_positions_qty CHECK (qty <> 0)`）
- **为什么**：活跃持仓表保持精简；历史持仓可通过 `fills` 重放
- **好处**：`SELECT * FROM positions` 即是当前风险暴露，无需 `WHERE qty != 0` 过滤

### D3. `instructions` 主键 UUID 而非 BIGSERIAL
- **为什么**：未来 signal_service 若分布式（多账户并行）可离线生成 id
- **成本**：UUID 索引比 BIGSERIAL 大 ~2.7x；一期数据量 ≤10k/年，可忽略
- **备选**：等数据量 >1M 时迁移到 `UUID v7`（需 `pg_uuidv7` 扩展，可排序）

### D4. `fills` 独立表支持部分成交
- **为什么**：半自动模式下用户可能分笔下单（10 手分 3-4-3 成交）；单 instruction 的 `filled_qty_total` 通过 view 聚合
- **权衡**：每次 UI 刷新要走 view；数据量小性能无虞

### D5. `instructions` 唯一约束 `(session_date, session, symbol, contract_code, action)`
- **为什么**：防 signal_service 重复跑导致重复指令；幂等安全
- **边界**：允许同日同 symbol 同时有 `open` + `close`？一般不会（一旦 pending close 就不该再 open），但约束上未禁

### D6. `soft_stop_triggered` 在 `daily_pnl` 而非独立表
- **为什么**：触发是当日事件，`daily_pnl` 是当日一行，天然贴合；状态转移 = 改这行即可
- **读路径**：`signal_service` 跑前查 `daily_pnl` 当日或昨日 `soft_stop_triggered`

### D7. `alerts.payload` 用 JSONB
- **为什么**：事件类型多样（数据失败的 symbol / API 错误码 / soft stop 的触发阈值），灵活 schema
- **成本**：JSONB 比结构化列略大，但事件体量小（~10 条/日），可接受

### D8. 全部时间用 `TIMESTAMPTZ`（带时区）
- **为什么**：中国期货跨日夜盘 + 未来可能多时区分析；统一 UTC 存储，展示时转 Asia/Shanghai
- **例外**：`session_date` / `bars.date` / `daily_pnl.date` 用 `DATE`（业务日是纯日期概念，无时区）

### D9. 局部索引（Partial Index）
- `idx_instructions_status_pending WHERE status='pending'` — 调度轮询只关心未处理
- `idx_daily_pnl_soft_stop WHERE soft_stop_triggered=TRUE` — 只索引触发日
- `idx_alerts_critical WHERE severity='critical'` — 告警检索
- **为什么**：节省空间 + 查询路径更短；PG 查询规划器会自动用

### D10. `gen_random_uuid()` 不用 `pgcrypto`
- PG 14+ 原生支持，无需扩展

### D11. `bars` 是单表（历史 + 实盘合一），不是分表
- **为什么**：plan 第 83 行用户已决策 ——「把 hab_bars.csv 2018-至今全部上传到云 PG `bars` 表，后续实盘每日新 bar 追加进同一张表」
- **好处**：
  - `signal_service` 只需一条 `WHERE symbol = ? AND date >= ?` 即拿到主力连续序列（warmup + 今日），无需 UNION historical / live 两表
  - 数据源单一，避免历史 / 实盘 schema 漂移（22 列必须锁死同源）
  - 主力合约切换时同 `symbol` 自然会换 `order_book_id`，PK `(order_book_id, date)` 天然隔离
- **权衡**：单表行数 = 品种数 × 年交易日 × 年数 ≈ 40 × 240 × 10 ≈ 10w 行，PG 轻松；若未来扩到分钟级再考虑按 `date` 范围分区
- **一致性校验**：`signal_service` 启动时打印 `MIN(date)` / `MAX(date)` / `COUNT(*)` / `COUNT(DISTINCT symbol)` 供用户核对（plan 第 167 行风险）

### D12. `fills.trigger_source` 给成交归因
- **为什么**：盘中止损单触发成交 vs 用户手动下单，两者对策略评估意义不同
- **四值**：`user_manual` / `stop_loss` / `take_profit` / `roll_contract`
- **用途**：报告能拆出「多少 PnL 来自止损 vs 正常平仓」，辅助策略迭代

### D13. `instructions.broker_stop_order_id` 可选字段
- **为什么**：用户在客户端挂止损单后可回填单号，便于盘后与券商对账 / 追查问题单
- **可选**：`VARCHAR(40) NULL`，不影响主流程

---

## 五、典型查询

### Q1. 前端"今日指令"页（指令 + 聚合成交）
```sql
SELECT *
FROM v_instructions_with_fills
WHERE session_date = CURRENT_DATE AND session = 'day'
ORDER BY group_name, symbol;
```

### Q2. 当前持仓页
```sql
SELECT * FROM positions ORDER BY group_name, symbol;
```

### Q3. 权益曲线（最近 90 天）
```sql
SELECT date, equity, drawdown_from_peak, soft_stop_triggered
FROM daily_pnl
WHERE date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY date;
```

### Q4. Soft stop 判定（`signal_service` 开工前）
```sql
SELECT soft_stop_triggered, drawdown_from_peak
FROM daily_pnl
WHERE date = CURRENT_DATE - INTERVAL '1 day';
```

### Q5. 组级风险暴露（portfolio_cap 检查）
```sql
SELECT * FROM v_group_exposure;
```

### Q6. 今日人工否决汇总（日报）
```sql
SELECT symbol, contract_code, target_qty, veto_reason
FROM instructions
WHERE session_date = CURRENT_DATE AND status = 'vetoed';
```

### Q7. 最近 critical 告警
```sql
SELECT event_at, event_type, message, payload
FROM alerts
WHERE severity = 'critical'
ORDER BY event_at DESC
LIMIT 50;
```

### Q8. 签 T 日收盘后查最新 bar（data_pipeline 验证 append 成功）
```sql
SELECT *
FROM bars
WHERE symbol = 'AG' AND date = CURRENT_DATE
ORDER BY order_book_id;
```

### Q9. `signal_service` 查最近 N 日 bars 做 warmup
```sql
-- 单品种主力连续（含多合约 — 取 volume 最大的作主力）
SELECT *
FROM bars
WHERE symbol = 'AG'
  AND date >= CURRENT_DATE - INTERVAL '365 days'
ORDER BY date, order_book_id;
```

### Q10. 止损触发 PnL 归因（日报子模块）
```sql
SELECT
  i.symbol,
  i.group_name,
  SUM(f.filled_qty * f.filled_price) AS stop_loss_notional,
  COUNT(*)                            AS stop_loss_fill_count
FROM fills f
JOIN instructions i ON i.id = f.instruction_id
WHERE f.trigger_source = 'stop_loss'
  AND f.filled_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY i.symbol, i.group_name
ORDER BY stop_loss_notional DESC;
```

---

## 六、优化与扩展建议

### 当前（≤10k 行/年 调仓指令 · ≈10w 行 bars 历史）
- 现有索引充分，无需额外操作
- 周级 `VACUUM ANALYZE instructions, bars` 足够

### 中期（10k-100k/年 或 多年积累）
- `alerts` 按月 TTL 清理（`DELETE WHERE event_at < NOW() - INTERVAL '180 days'`）
- `bars` 若扩到分钟级 → 按 `date` RANGE 月分区

### 长期（>100k/年 或 多账户）
- `instructions` 按 `session_date` RANGE 分区（月粒度）
- 切换 `UUID v4 → v7`（`pg_uuidv7` 扩展），恢复排序能力
- 读写分离：权益曲线读跑只读副本

### 安全
- 应用账号用 least privilege：只给 `app_user` 授 `INSERT/SELECT/UPDATE` on 6 表 + view
- `alerts` 只 INSERT + SELECT，禁 UPDATE/DELETE（审计不可篡改）
- `bars` 可给 `app_user` 只读 + `bars_loader` 账号写（数据上传专用）
- 凭证放 `live/.env`（已在根 `.gitignore`）

---

## 七、初始化（本地开发）

```bash
# 1. 启动 docker-compose pg (TODO: live/docker-compose.yml，由后续 milestone 做)
docker compose -f live/docker-compose.yml up -d

# 2. 创建 database
psql -h localhost -U postgres -c "CREATE DATABASE stock_future;"

# 3. 应用 schema
psql -h localhost -U postgres -d stock_future -f live/db/schema.sql

# 4. 验证
psql -h localhost -U postgres -d stock_future -c "\dt"
# 应看到 6 张表：positions, instructions, fills, daily_pnl, bars, alerts

# 5. 上传历史 bars（用户负责）
# 参考 plan 第 83 行：把 hab_bars.csv 2018-至今 22 列同源上传
psql -c "\COPY bars(date,symbol,open,high,low,close,settle,volume,open_interest,\
  contract_multiplier,commission,slippage,group_name,margin_rate,\
  open_raw,high_raw,low_raw,close_raw,settle_raw,order_book_id,limit_up,limit_down) \
  FROM 'data/cache/normalized/hab_bars.csv' WITH (FORMAT CSV, HEADER TRUE);"
```

### Rollback
```bash
psql -h localhost -U postgres -d stock_future -f live/db/migrations/001_initial.down.sql
```

---

## 八、Pydantic v2 模型（`models.py`）

| 类别 | 模型 | 用途 |
|---|---|---|
| 读 | `Position` / `Instruction` / `Fill` / `DailyPnl` / `Bar` / `Alert` | API 响应序列化（`from_attributes=True`） |
| 写 | `InstructionCreate` / `FillCreate` | API request body 校验 |
| 枚举 | `InstructionStatus` / `InstructionAction` / `InstructionDirection` / `InstructionSession` / `AlertSeverity` / `TriggerSource` | 状态机 / 固定词表，前后端共享 |

**约定**：
- 时间 → `datetime`（UTC，带 tzinfo），业务日 → `date`
- 金额 / 价格 → `Decimal` 保精度
- `model_config = ConfigDict(from_attributes=True)` — v2 语法（v1 的 `orm_mode` 已废弃）

**依赖状态**：项目根尚无 `requirements.txt`，需 P1c agent 建立并加入 `pydantic>=2.0`。

---

## 九、未纳入本 schema 的（一期外）

- **Broker / 真实订单**：半自动模式不需要，指令 + fills 即闭环
- **多账户**：单账户一期足够；多账户扩展加 `account_id` 字段 + `accounts` 表
- **手续费 / 滑点明细**：已在 `bars` 表静态列 + `fills.note`；后续独立 `fill_costs` 表更规范
- **引擎内部状态**（ATR、ADX、信号打分）：属于计算中间态，不入 DB；需要时写 `strats/` 的 parquet 缓存
- **盘中分钟数据**：日线策略不需要

---

## 十、版本历史

| 版本 | 日期 | 说明 |
|---|---|---|
| 001 | 2026-04-19 | 初始 schema：6 表 + 2 view + 触发器 + Pydantic v2 models |
| 002 | 2026-04-19 | 架构师重审：5→6 表（加 `bars` 历史+实盘合一 / `system_events` 改名 `alerts`）；`instructions.status` 改 `fully_filled` / `partially_filled`；加 `fills.trigger_source` 归因字段 + `instructions.broker_stop_order_id`；加 D11-D13 决策 |
