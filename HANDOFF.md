# HANDOFF — Minimal Context

**项目**：中国期货量化趋势跟踪回测（日线）。91 个品种，2018-01 至 2025-12。

**仓库**：`git@github.com:ToddyMeow/stock-future.git` / 分支 `main`

---

## 先读这两个文件

- 📐 [ARCHITECTURE.md](ARCHITECTURE.md) — 静态分层图 + 所有设计决策索引
- [strats/engine.py](strats/engine.py) `StrategyEngine` — 主编排器（1660 LOC）

## 快速验证

```bash
python -m pytest -q           # 期望 144 passing
cat data/cache/normalized/hab_bars.csv | head -1    # 22 列正常
```

---

## 数据管道（按顺序）

一次性全流程（假设 `.env` 里 `RQDATAC_USER` + `RQDATAC_PASSWORD` 已配）：

```bash
python scripts/build_trading_calendar.py              # 1.1 — 交易日历
python scripts/download_rqdata_futures.py \
  --config scripts/download_rqdata_futures.cfg \
  --overwrite                                          # OHLC + settle
python scripts/download_dominant_contracts.py         # 1.2 — 合约代码
python scripts/download_limit_prices.py               # 1.8 — 涨跌停板价
python scripts/fetch_commission_specs.py              # 5.2 — 真实费率
python scripts/build_enhanced_bars.py                 # 合并 + OHLC 修复
python scripts/apply_commissions.py                   # 5.2 补丁（可选，future download 已自动）
```

**产物**：`data/cache/normalized/hab_bars.csv`（22 列，~135k 行）

---

## 关键 invariants（不要破坏）

1. `date` 必须是交易日 — adapter 层通过 `TradingCalendar.validate_trading_days` 强制
2. 信号在 T 收盘、执行在 T+1 开盘（= 21:00 夜盘开）。永远不能同 bar 开平。不变量测试：[strats/test_execution_policy.py](strats/test_execution_policy.py)
3. `settle` 驱动逐日盯市 / 保证金占用；`close` 驱动指标计算。entries/exits 不得引用 settle（静态扫描测试保护）
4. `entries/`、`exits/` 都通过 `Protocol` 约束，不用继承
5. `hab_bars.csv` 的 `date` 列是真实交易日（回归测试检验）；h < l 等硬错永远 raise，不要 auto-repair

---

## Gotchas（踩过的坑）

- **Panama close 在 I/P/LU/EC 等深偏移品种会变负**。用 `|close|` 或 `close_raw` 做 commission / limit-lock 比较
- **`margin_rate` 在 4.1 之前完全不被引擎使用**。只有 `max_margin_utilization > 0` 时才有 cap gate 激活
- **默认关闭的配置**：`warmup_bars=0` / `max_limit_days=0` / `max_margin_utilization=0` / `min_atr_pct=0.0025`。生产推荐值见 `EngineConfig` docstring
- **dual-stream（1.2）需要 `enable_dual_stream=True`** — 才启用 per-contract 段账户 + 真实 roll 成本。否则退回 Panama 单流
- **89 个品种默认 commission 占位符 5 yuan 在 5.2 已替换为 RQData 真实费率**（by_money / by_volume）。如果你看到 commission 列全 5.0，说明 `apply_commissions.py` 没跑过
- **RQData 的 `future_commission_margin` 只给当前快照**，没有历史 per-day 费率（这是 5.2 选 symbol-level static + tier schedule 而不是 per-day 的原因）
- **OHLC 有 166 行 close/settle 超出 [low, high]**（低流动性品种锁死 + settle 修正），`ohlc_repair.py` method B 在 adapter 自动扩包 high/low

---

## 已完成的结构性审计（详情见 ARCHITECTURE.md）

- **1.1** 夜盘归属 → `TradingCalendar` + adapter 校验
- **1.2** 连续合约 → Panama 单流默认 + 可选 dual-stream + `_check_and_apply_roll`
- **1.3** settle vs close → 逐日 mark 用 settle，指标用 close
- **1.4** OHLC 坏数据 → 方式 B 扩包修复 + `ATR_BELOW_FLOOR` + DQ 报告
- **1.5/1.6** 信号时点 + look-ahead → 零代码改动 + 不变量测试
- **1.7** 指标预热 → `warmup_bars` 门禁（默认 0）
- **1.8/1.10** 涨跌停 + gap-open → `_cannot_fill_side` + sizing 底线
- **4.1** 保证金 cap → `max_margin_utilization` + tier schedule
- **5.2** 真实费率 → `commission_specs.json` + by_money/by_volume 分流
- **5.3** 盯市公式 → 已被 1.3 等价覆盖，零代码

## 故意 deferred（不要重开）

| 项 | 原因 |
|---|---|
| 5.1 平今/平昨 lot tracking | 2.06% 日线 trades 同日开平，~250 LOC 不值 |
| 4.2 交割月硬检查 | 活跃品种 dominant auto-roll 已保护；低流动性品种被 ATR floor + limit-lock 间接过滤 |
| ratio-adjusted 连续合约 | Panama 对点值指标（ATR/Donchian/BB）等价；深偏移会生成负价 |
| forced_close 追保模拟 | 入场期 cap 已足；追保模型需要 engine 新状态机，无明确 ROI |
| StrategyEngine.run() 拆 phase 方法 | 400 行内部有清晰 `# 0. 1. 2...` 注释；外部不可见 |

---

## 品种分组（`data/adapters/futures_static_meta.py::FUTURES_GROUP_MAP`）

| 组 | 品种 | group_risk_cap |
|----|------|---------------|
| equity_index | IC, IF, IM | 4% |
| bond | T, TL, TS | 4% |
| metals | AG, AL, AU, CU, NI, PB, SN, SS, ZN | 6% |
| black_steel | I, J, JM, RB, SF, ZC | 6% |
| chem_energy | BU, EB, EG, FU, L, MA, PF, PG, PP, SC, TA, V | 6% |
| rubber_fiber | CF, CY, RU, SR | 6% |
| agri | A, B, C, CS, M, OI, P, PK, RM, RS, Y | 6% |
| building | AO, FG, LC, SA, SH, SI, SM, SP, UR, WR | 5% |
| livestock | JD, LH | 4% |
| independent (12 个低相关) | AP, BB, CJ, EC, FB, JR, LR, LU, PM, RI, RR, WH | 每品种 2%，合计 8% |

**排除（`EXCLUDED_SYMBOLS`）**：BC, PR, PX, NR, HC, IH, TF, BR, LG, AD, BZ, OP, PD, PT, PP_F, V_F, L_F

---

## 开发守则

- **Karpathy #2 Simplicity**：新加 feature 前先 push back — 有没有数据证明问题真实？我们的默认是否已覆盖？
- **Karpathy #3 Surgical**：不重构没坏的；每行改动追得到用户诉求
- **测试先行**：改引擎行为 → 先加一条 invariant / behavior 测试；改策略 → 加回归测试
- **commit 信息**：follow 已有风格（`feat(engine): XXX` / `fix(helpers): YYY` / `doc: ZZZ`）
- **don't skip pre-commit hooks**；如 hook fail 则 fix 根因后重新 commit（不是 --amend）

---

## 上手建议（新 agent 前 3 件事）

1. 跑 `python -m pytest -q`，确认 144 passing
2. 读 [ARCHITECTURE.md](ARCHITECTURE.md) 5 分钟，建立分层心智模型
3. 用 [scripts/run_three_layer_backtest.py](scripts/run_three_layer_backtest.py) 跑一次端到端感受产出
