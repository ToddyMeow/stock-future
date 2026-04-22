# live/ — 半自动实盘 MVP 运维手册

本目录是 stock-future 的实盘交易模块（MVP 阶段，半自动：系统产信号，用户在期货客户端挂单 / 回填成交）。

- 后端：FastAPI（`live/web/api.py`）+ PostgreSQL（阿里云 RDS）
- 前端：`/Users/mm/Trading/stock-future-ui/`（Next.js 14，独立仓）
- 调度：launchd（`live/scheduler/*.plist`，6 个时点）
- 通知：Server 酱（微信）+ 阿里云 SMS（critical 告警兜底）

---

## 一、概述（架构一图）

```
┌────────────────────────────────────────────────────────────────────────┐
│   T 日时间线（日盘 09:00-15:00 / 夜盘 21:00-次日 02:30）              │
├────────────────────────────────────────────────────────────────────────┤
│  15:00  日盘收盘                                                       │
│  15:05  [launchd] data_pipeline → 拉日盘 bar 进 bars 表                │
│  15:10  [launchd] signal_service night → 产夜盘指令写 instructions 表  │
│  15:15  [signal_service 内调] alerting → Server 酱推送"夜盘 X 条指令" │
│  15:30  [launchd] check_soft_stop → 看权益回撤是否越 5% 阈值           │
│  20:00- [用户] 打开前端 /instructions 看指令 → 客户端挂夜盘 stop/新单  │
│  21:00  夜盘开盘，挂单按价格触发                                       │
│  22:00- [用户] 前端回填夜盘实际成交（filled_qty/veto/skip）           │
│  02:30  夜盘收盘                                                       │
│  02:35  [launchd] data_pipeline → 拉夜盘 bar                           │
│  02:40  [launchd] signal_service day → 产日盘指令                      │
│  02:45  [launchd] check_soft_stop                                      │
│  08:00- [用户] 前端看日盘指令 → 客户端挂单                             │
│  09:00  日盘开盘                                                       │
│  09:30- [用户] 前端回填日盘成交                                        │
│  18:00  [launchd] report_service → 写日报 HTML + Server 酱摘要         │
└────────────────────────────────────────────────────────────────────────┘
```

每日用户操作时长目标：2 次 × ~5 分钟 = 10 分钟/天。

---

## 二、模块清单

| 文件 | 职责 |
| --- | --- |
| `config.py` | 集中读 `.env`；暴露 `DATABASE_URL` / `SERVERCHAN_SEND_KEY` / `SOFT_STOP_PCT` 等 |
| `data_pipeline.py` | 从 `hab_bars.csv` 读最后 N 天 → `INSERT ON CONFLICT DO NOTHING` 进 `bars` 表（mock；实盘接 RQData 同 schema）；另有 `get_bars_for_engine(end_date, warmup_days)` 供 signal 调用 |
| `signal_service.py` | 读 final_v3 combos + 当前 positions + engine_states 上次快照 → 拼 bars 喂 `StrategyEngine.run()` → 写 `instructions` + 新 `engine_states`；**幂等**（同 session_date+session 已跑过直接 short-circuit，`--force` 覆盖） |
| `account_state.py` | 6 表 CRUD（positions / instructions / fills / daily_pnl / bars / alerts） + partial fill 状态机（sum(fills) 对齐 target_qty 推 status） + `record_daily_pnl()` |
| `web/api.py` | FastAPI 10 路由（instructions / fills / positions / daily_pnl / history / reports / health）；CORS 开 `localhost:3000` |
| `report_service.py` | Jinja2 渲染 `templates/daily_report.html` → 写 `reports/YYYY-MM-DD.html` → 调 alerting 推 Server 酱摘要 |
| `alerting.py` | `send_wechat()` Server 酱（SendKey 空时 stub）；`send_sms()` 阿里云 SMS（stub，先记 `logs/sms_would_send.jsonl`）；`alert_escalate(severity, ...)` 按级路由 + 写 alerts 表 |
| `soft_stop.py` | `check_and_update(today)` 计算今日相对 30 日 peak 的 drawdown；`is_soft_stop_active(date)` signal_service 跑前调用 |
| `check_soft_stop.py` | soft_stop 的 CLI 小脚本，给 launchd 在 15:30 + 02:45 定时跑 |
| `db/schema.sql` | 6 表 + 2 视图（`v_instructions_with_fills` / `v_group_exposure`）+ 状态机触发器 |
| `db/migrations/` | 001_initial + 002_engine_states；`up.sql` / `down.sql` 对 |
| `scheduler/*.plist` | launchd 计划任务（6 份，见下文）；**需要用户手动 `launchctl load`** |
| `templates/daily_report.html` | 日报 Jinja 模板（权益 / 今日 PnL / 指令执行汇总 / 否决归因） |

---

## 三、环境变量（`.env`）

首次部署：`cp live/.env.example live/.env` 然后按下表填。

| 变量 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- |
| `DATABASE_URL` | 是 | — | 阿里云 RDS 同步 DSN；格式 `postgresql://user:pwd@host:5432/db`；特殊字符 URL 编码 |
| `DATABASE_URL_ASYNC` | 否 | 自动推导 | asyncpg 用；不填会把 `postgresql://` 替成 `postgresql+asyncpg://` |
| `SOFT_STOP_PCT` | 否 | `0.05` | 日内权益回撤软熔断阈值（5%）；触发后下个 session 只产 close |
| `APP_TIMEZONE` | 否 | `Asia/Shanghai` | 展示层时区（计算层用 DB 的 `TIMESTAMPTZ`） |
| `SERVERCHAN_SEND_KEY` | 否 | 空（stub） | ft07.com 注册后的 `SCT...` Key；空则日报 / 告警只打日志不发微信 |
| `ALIYUN_SMS_ACCESS_KEY_ID` | 否 | 空（stub） | 阿里云 SMS AccessKey ID |
| `ALIYUN_SMS_ACCESS_KEY_SECRET` | 否 | 空（stub） | 阿里云 SMS AccessKey Secret |
| `ALIYUN_SMS_SIGN_NAME` | 否 | 空 | 短信签名名称 |
| `ALIYUN_SMS_TEMPLATE_CODE` | 否 | 空 | 短信模板 CODE |
| `PHONE_NUMBER` | 否 | 空 | 接收 critical SMS 的手机号 |

**安全**：`.env` 在 `.gitignore` 明确列入，绝不 commit。

---

## 四、首次部署步骤

前提：Mac 有 Python 3.10（miniconda / venv 均可），Node.js 18+（前端），已开通阿里云 RDS（PostgreSQL 14+），本机能 ping 通 RDS host。

### 1. 安装依赖

```bash
cd /Users/mm/Trading/stock-future
pip install -r live/requirements.txt          # FastAPI / SQLAlchemy / psycopg / jinja2 / ...
cd /Users/mm/Trading/stock-future-ui
npm install                                    # 前端
```

### 2. 建表 + 导入历史 bars

```bash
# 建 6 表 + 2 视图
psql "$DATABASE_URL" -f /Users/mm/Trading/stock-future/live/db/schema.sql

# 应用 engine_states migration
psql "$DATABASE_URL" -f /Users/mm/Trading/stock-future/live/db/migrations/002_engine_states.up.sql

# 上传 hab_bars.csv（135k 行，约 5-10 分钟）
cd /Users/mm/Trading/stock-future
python scripts/upload_hab_bars_to_pg.py
```

### 3. 配 `.env`

```bash
cd /Users/mm/Trading/stock-future/live
cp .env.example .env
# 编辑填 DATABASE_URL / SERVERCHAN_SEND_KEY（其它先留空给 stub 用）
```

### 4. 启后端 API（常驻）

```bash
cd /Users/mm/Trading/stock-future
uvicorn live.web.api:app --host 0.0.0.0 --port 8000 --reload
# 或用 launchd 常驻化（后续加 com.stockfuture.api.plist；MVP 阶段手开即可）
```

### 5. 启前端

```bash
cd /Users/mm/Trading/stock-future-ui
cp .env.local.example .env.local     # 确认 NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                          # http://localhost:3000
```

### 6. 激活 launchd 调度（6 个 plist）

```bash
mkdir -p ~/Library/LaunchAgents
cp /Users/mm/Trading/stock-future/live/scheduler/*.plist ~/Library/LaunchAgents/

# 逐个 load（一条条跑方便出错时定位）
launchctl load ~/Library/LaunchAgents/com.stockfuture.data_after_dayclose.plist
launchctl load ~/Library/LaunchAgents/com.stockfuture.signal_night.plist
launchctl load ~/Library/LaunchAgents/com.stockfuture.data_after_nightclose.plist
launchctl load ~/Library/LaunchAgents/com.stockfuture.signal_day.plist
launchctl load ~/Library/LaunchAgents/com.stockfuture.daily_report.plist
launchctl load ~/Library/LaunchAgents/com.stockfuture.soft_stop_check.plist

# 确认
launchctl list | grep com.stockfuture
# 期望 6 行，Status 列为 0（上次退出码）或 -（从未跑过）
```

### 7. 冒烟测试（立即手动触发一次）

```bash
launchctl start com.stockfuture.signal_night
tail -f /Users/mm/Trading/stock-future/live/logs/launchd/signal_night.{out,err}.log
# 看到 "[signal_service] 汇总: ..." 代表跑通
```

---

## 五、日常操作

### 手动跑一次 signal（调试 / 补跑）

```bash
cd /Users/mm/Trading/stock-future
python -m live.signal_service --date 2026-04-19 --session day
# --dry-run  不写 DB
# --force    覆盖已有 instructions + engine_states
```

### 手动跑一次日报

```bash
cd /Users/mm/Trading/stock-future
python -m live.report_service --date 2026-04-19
# --no-push  只写 HTML 不发 Server 酱
```

打开 `live/reports/2026-04-19.html` 浏览器看（或走前端 `/reports/2026-04-19`）。

### 查看 / 改 soft stop 阈值

```bash
# 看当前
grep SOFT_STOP_PCT live/.env                   # 0.05 = 5%

# 改
sed -i '' 's/SOFT_STOP_PCT=.*/SOFT_STOP_PCT=0.07/' live/.env
# 重启 uvicorn 和 launchd（环境变量 fork 时读）
launchctl unload ~/Library/LaunchAgents/com.stockfuture.soft_stop_check.plist
launchctl load ~/Library/LaunchAgents/com.stockfuture.soft_stop_check.plist
```

### 回填 fills

打开前端 `http://localhost:3000/instructions` → 今日未回填指令 → 每行填 `filled_qty` + `filled_price` → 点 "提交"。

- 全部成交：`fully_filled`
- 只部分成交（`filled_qty < target_qty`）：`partially_filled`
- 否决：必填 `veto_reason`
- 跳过：`skipped`

### 查看历史报告

前端 `http://localhost:3000/history` → 选日期 → 看指令 + fills 折叠明细。

### 看 Server 酱 / SMS 日志

```bash
tail -f live/logs/sms_would_send.jsonl             # SMS stub（凭证未配时）
# 微信推送：直接看手机 Server 酱 APP；或 alerts 表：
psql "$DATABASE_URL" -c "SELECT created_at, severity, title FROM alerts ORDER BY created_at DESC LIMIT 10;"
```

---

## 六、故障排查

### Q1. launchd 任务没跑 / `launchctl list | grep com.stockfuture` 看不到

检查：
1. plist 路径对不对：`ls ~/Library/LaunchAgents/com.stockfuture.*`
2. `launchctl load` 时是否报错（XML 语法 / 权限）
3. 日志文件是否有 err：`cat live/logs/launchd/signal_day.err.log`
4. launchd 守护进程：`ps aux | grep launchd`（正常应该有 `/sbin/launchd`）

重新激活：
```bash
launchctl unload ~/Library/LaunchAgents/com.stockfuture.signal_day.plist
launchctl load ~/Library/LaunchAgents/com.stockfuture.signal_day.plist
```

### Q2. signal_service 说"已跑过" short-circuit

正常幂等保护。要强行重跑加 `--force`：
```bash
python -m live.signal_service --date 2026-04-19 --session day --force
```

### Q3. engine.run() 报错 / bars 数据不够

`signal_service` 需要 `WARMUP_DAYS=1200`（~3.3 年）历史 bars 做指标预热。检查：
```sql
SELECT MIN(date), MAX(date), COUNT(*) FROM bars;
-- 期望 date 范围覆盖 [session_date - 1200 日, session_date]
```

不够就补跑 `scripts/upload_hab_bars_to_pg.py` 或等 `data_pipeline` 自动追加。

### Q4. Server 酱推不出去

1. 检查 `SERVERCHAN_SEND_KEY` 是否填：`grep SERVERCHAN live/.env`
2. 手测：
```bash
python -c "import asyncio; from live.alerting import send_wechat; asyncio.run(send_wechat('测试', '手动触发'))"
```
3. 看 `alerts` 表是否有记录（stub 时也会写）

### Q5. 前端显示"后端未连接 / 用 mock"

1. 后端 API 起了吗：`curl http://localhost:8000/api/health` 应返 `{"ok":true}`
2. 前端 `.env.local` 的 `NEXT_PUBLIC_API_URL` 对不对
3. CORS：浏览器控制台看有没有跨域报错；`live/web/api.py` 里 `allow_origins` 应包含 `http://localhost:3000`

### Q6. RDS 连不上

1. 白名单：RDS 控制台把本机公网 IP 加进去
2. 密码 URL 编码：`@` 要写 `%40`，`/` 写 `%2F`
3. 手测：`psql "$DATABASE_URL" -c "SELECT 1"`

### Q7. launchd 环境变量缺失（`DATABASE_URL` not defined）

launchd 跑出来的环境**不继承 shell 的 `.zshrc`**。`.env` 靠 `live/config.py` 的 `load_dotenv()` 读取，所以只要 `live/.env` 填对就 OK。如果你把变量放 shell export 而不是 `.env`，launchd 看不到。

---

## 七、架构图（数据流）

```
            ┌─────────────┐      mock/RQData       ┌──────────────────┐
            │  hab_bars   │───────────────────────>│  bars 表（RDS）  │
            │  CSV 种子   │   data_pipeline.py     └────────┬─────────┘
            └─────────────┘                                 │
                                                            │ 拉最近
                                                            │ 1200 日
                                                            ▼
┌─────────────┐  上次快照   ┌─────────────────┐      ┌──────────────────┐
│ engine_     │◀───────────│ signal_service  │<─────│ StrategyEngine   │
│ states(RDS) │─存快照────>│   (幂等 CLI)    │      │ final_v3 combos  │
└─────────────┘            └────────┬────────┘      └──────────────────┘
                                    │ pending_entries
                                    ▼
                           ┌─────────────────┐  Server 酱
                           │ instructions 表 │ ─────────▶ 用户微信
                           │    (RDS)        │              │
                           └────────┬────────┘              │ 看内容
                                    │                        ▼
                                    │             ┌──────────────────┐
                                    │◀────── GET  │  前端 /instr-    │
                                    │             │  uctions         │
                                    │─── UPDATE──▶│  (回填 / veto)   │
                                    │             └──────────────────┘
                                    │                        │
                                    │                        │ POST fill
                                    ▼                        ▼
                           ┌─────────────────┐      ┌──────────────────┐
                           │   fills 表      │─────>│ partial fill     │
                           │   (RDS)         │      │ 状态机推 status  │
                           └────────┬────────┘      └──────────────────┘
                                    │
                                    │ sum(filled_qty) signed
                                    ▼
                           ┌─────────────────┐
                           │  positions 表   │───┐
                           │    (RDS)        │   │
                           └─────────────────┘   │
                                                 │ mark-to-market
                                                 ▼
                                        ┌──────────────────┐
                                        │  daily_pnl 表    │
                                        │    (RDS)         │
                                        └────────┬─────────┘
                                                 │
                                     18:00 cron  │ peak DD
                                                 ▼
                                    ┌───────────────────────┐
                                    │  soft_stop.check      │
                                    │   & update            │
                                    └───────────────────────┘
                                                 │
                                                 ▼
                                    ┌───────────────────────┐
                                    │  report_service        │
                                    │   + Server 酱摘要     │
                                    └───────────────────────┘
```

---

## 八、launchd 时点总表

| Label | Cron-like | 动作 |
| --- | --- | --- |
| `com.stockfuture.data_after_dayclose` | 每天 15:05 | `data_pipeline --n-days 1`（拉日盘 bar） |
| `com.stockfuture.signal_night` | 每天 15:10 | `signal_service --session night` |
| `com.stockfuture.soft_stop_check`（第 1 次） | 每天 15:30 | `check_soft_stop`（日盘收盘 equity） |
| `com.stockfuture.daily_report` | 每天 18:00 | `report_service`（日报 + Server 酱推送） |
| `com.stockfuture.data_after_nightclose` | 每天 02:35 | `data_pipeline --n-days 1`（拉夜盘 bar） |
| `com.stockfuture.signal_day` | 每天 02:40 | `signal_service --session day` |
| `com.stockfuture.soft_stop_check`（第 2 次） | 每天 02:45 | `check_soft_stop`（夜盘收盘 equity） |

注意：launchd 不判交易日（春节 / 周末也触发）。非交易日 signal_service 内部读 `data/cache/trading_calendar.csv` 直接 return，data_pipeline 的 ON CONFLICT 也幂等。

---

## 九、不做的事（MVP 边界）

- ❌ CTP / 券商 API 自动下单（半自动永不做）
- ❌ 盘中分钟级监控（用户挂单兜底）
- ❌ Paper trade wall clock 验证（下阶段）
- ❌ 主力合约自动切换（只告警，用户手切；下阶段做）

---

## 十、参考

- [handoff-zesty-meteor.md](../../.claude/plans/handoff-zesty-meteor.md) —— MVP 启动 plan
- [fluffy-dancing-fox.md](../../.claude/plans/fluffy-dancing-fox.md) —— 续作 plan
- [open-questions-awaiting-user.md](../../.claude/plans/open-questions-awaiting-user.md) —— 等用户决策的问题
- [db/README.md](db/README.md) —— DB schema + 状态机 ER 图
