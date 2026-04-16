# Agent Handoff — 项目状态快照

## 项目概述

中国期货量化趋势跟踪系统。可组合的入场/出场策略矩阵，多策略共用资金池，基于相关性的品种分组风控。

**仓库**: `git@github.com:ToddyMeow/stock-future.git`
**数据**: `data/cache/normalized/hab_bars.csv` (11MB, 91 symbols, 2018-2025 日线)

---

## 架构

```
config.yaml                    ← 所有超参数的唯一来源
strats/
  engine.py                    ← StrategyEngine 编排器 + EngineConfig + PendingEntry/Position/BacktestResult
  protocols.py                 ← EntryStrategy / ExitStrategy Protocol
  helpers.py                   ← wilder_smooth, wilder_atr, adx, direction helpers, PortfolioAnalyzer
  config_loader.py             ← load_config() + build_engine() 从 YAML 构建引擎
  entries/                     ← 7 个入场策略
    hl_entry.py                  HLEntryStrategy (N日通道突破)
    boll_break_entry.py          BollBreakEntryStrategy (布林带突破)
    ama_entry.py                 AmaEntryStrategy (Kaufman自适应均线)
    double_ma_entry.py           DoubleMaEntryStrategy (双均线交叉)
    hab_entry.py                 HABEntryStrategy (横盘蓄势箱体突破)
    rand_entry.py                RandEntryStrategy (随机入场基准)
    donchian_entry.py            DonchianEntryStrategy (alias → HLEntry)
  exits/                       ← 8 个出场策略
    hl_exit.py                   HLExitStrategy (通道反向)
    boll_exit.py                 BollExitStrategy (布林带反向)
    ama_exit.py                  AmaExitStrategy (AMA翻转)
    atr_trail_exit.py            AtrTrailExitStrategy (ATR移动止损)
    term_exit.py                 TermExitStrategy (定期平仓)
    double_ma_exit.py            DoubleMaExitStrategy (双均线交叉)
    hab_exit.py                  HABExitStrategy (结构/时间失败 + ATR跟踪)
    rand_exit.py                 RandExitStrategy (随机出场基准)
data/adapters/
  futures_static_meta.py       ← FUTURES_GROUP_MAP (品种→组), EXCLUDED_SYMBOLS, FuturesMeta
  rqdata_futures_adatpter.py   ← RQData 数据适配器
```

### 核心 API

```python
# 方式 1: 从 config.yaml 构建
from strats.config_loader import load_config, build_engine
engine = build_engine(load_config())
result = engine.run(bars)

# 方式 2: 手动构建多策略
from strats.engine import StrategyEngine, EngineConfig, StrategySlot
engine = StrategyEngine(
    config=EngineConfig(...),
    strategies=[
        StrategySlot("hl_9", HLEntryStrategy(...), HLExitStrategy(...)),
        StrategySlot("ama", AmaEntryStrategy(...), HABExitStrategy(...)),
    ],
)
```

### R 的定义

R = `stop_atr_mult × ATR(atr_period)`，由引擎统一计算，入场策略不管止损。
`initial_stop = close ∓ R`（多头减、空头加）。

### ADX 趋势滤波器

```
trend_score = clip(ADX(20) / adx_scale, adx_floor, 1.0)
effective_risk = risk_per_trade × trend_score
```
震荡时自动缩仓（ADX<10 → 仅 20% 仓位），不影响信号生成。

---

## 品种分组（基于 Pearson 相关性聚类）

| 组 | 品种 | 组内相关 | group_risk_cap |
|----|------|---------|---------------|
| equity_index | IC, IF, IM | 0.79 | 4% |
| bond | T, TL, TS | 0.82 | 4% |
| chem_energy | BU, EB, EG, FU, L, MA, PF, PG, PP, SC, TA, V | 0.55 | 6% |
| rubber_fiber | CF, CY, RU, SR | 0.42 | 6% |
| metals | AG, AL, AU, CU, NI, PB, SN, SS, ZN | 0.38 | 6% |
| black_steel | I, J, JM, RB, SF, ZC | 0.32 | 6% |
| agri | A, B, C, CS, M, OI, P, PK, RM, RS, Y | 0.29 | 6% |
| building | AO, FG, LC, SA, SH, SI, SM, SP, UR, WR | 0.23 | 5% |
| livestock | JD, LH | 0.17 | 4% |
| independent (12个) | AP,BB,CJ,EC,FB,JR,LR,LU,PM,RI,RR,WH | 0.02 | 每品种2%, 合计8% |

排除品种: BC,PR,PX,NR,HC,IH,TF,BR,LG,AD,BZ,OP,PD,PT,PP_F,V_F,L_F

---

## 已完成的回测结果

### Long+Short 全组 OOS 验证（70/30 split, 无 ADX）

| 组 | 通过 OOS | 最佳组合 | 测试 Sharpe | 测试 CAGR |
|----|---------|---------|-----------|---------|
| **building** | 10/10 全过 | hl_21+atr_trail | 1.51 | +45.5% |
| **metals** | 1 pass | ama+ama | 0.36 | +5.9% |
| **livestock** | 9/10 pass | hl_9+boll | 0.24 | +2.3% |
| **rubber_fiber** | 7/10 pass | boll+term_2_13 | 0.89 | +12.2% |
| **black_steel** | 2 pass (弱) | hl_9+boll | 0.03 | -3.6% |
| agri | 全崩 | — | — | — |
| chem_energy | 全崩 | — | — | — |
| bond | 无 survivors | — | — | — |
| equity_index | 无交易 | — | — | — |

---

## 当前中断的任务

### 1. 后台回测正在运行 (Task ID: bc26b67nr)

**任务**: 三层回测（策略层 × 品种组层 × 组合层），ADX ON/OFF 对比，allow_short=True，全 9 组 × 35 组合。

**关键修复**: 回测脚本中必须用 `FUTURES_GROUP_MAP` 覆盖 CSV 里的旧 `group_name`：
```python
bars['group_name'] = bars['symbol'].map(FUTURES_GROUP_MAP).fillna(bars['group_name'])
```
CSV 里的 `group_name` 是旧值（commodity/black/index），和新分组（chem_energy/black_steel/agri 等）不匹配。

**输出文件**:
- `data/backtest_strategy_layer.csv` — 每行 = group × combo × adx × year
- `data/backtest_group_layer.csv` — 每行 = group × year (含 avg_adx)

**需要的后续分析** (等回测完):
- 策略层: Top combos per group, 逐年 Sharpe/PF, 退出原因分布, ADX ON vs OFF 对比
- 品种组层: 组内品种贡献分布, 信号密度逐年, 平均 ADX 逐年
- 组合层: 合并资金曲线, 组间逐年归因, 最长连续亏损, 风控利用率

### 2. CSV group_name 未更新

`data/cache/normalized/hab_bars.csv` 中的 `group_name` 列仍然是旧值。需要重新运行数据管道 (`scripts/download_rqdata_futures.py`) 或手动更新 CSV。**临时方案**: 在回测脚本中用 `FUTURES_GROUP_MAP` 覆盖（已在最新回测脚本中实现）。

### 3. config.yaml 中的 allow_short 还是 false

用户希望做多做空并行。config.yaml 里 `allow_short: false`，但回测脚本中手动设了 True。后续需要决定 config.yaml 的默认值。

---

## 测试状态

40 个测试全过 (33 HAB engine + 7 Donchian entry)。
```bash
cd strats && python -m pytest test_horizontal_accumulation_breakout_v1.py entries/test_donchian_entry.py -q
```

**旧门面已删除**: `horizontal_accumulation_breakout_v1.py` 不存在了。测试直接用 `EngineConfig` + `HABEntryConfig` + `HABExitConfig`，通过 `make_test_engine()` helper 构建。

---

## 关键技术决策记录

1. **R 统一由引擎定义** — 入场不管止损, R = stop_atr_mult × ATR, config.yaml 一处调
2. **ADX 趋势滤波器** — 震荡时缩仓而非停止交易, `trend_score = clip(ADX/30, 0.2, 1.0)`
3. **ATR NaN 修复** — `wilder_smooth()` 遇 NaN 保持前值, 不再永久传播
4. **PendingEntry/Position 用 metadata dict** — 策略特有字段不硬编码
5. **group_risk_cap 从 float → Dict** — 每组独立上限, 基于组内相关性
6. **independent 品种** — 12 个低相关品种不设组限, 合计 soft cap 8%
7. **品种排除** — 高相关冗余对 (BC-CU 0.99) 和数据不足品种

---

## Group Avg ADX by Year (参考)

```
Group             2018   2019   2020   2021   2022   2023   2024   2025
agri              23.7   24.4   24.8   19.2   22.4   20.7   20.8   16.7
black_steel       19.7   20.3   24.0   26.4   19.2   23.7   22.3   22.8
bond              27.8   22.9   23.9   19.5   18.1   22.5   22.1   22.2
building          43.6   25.3   22.5   24.4   18.9   24.1   22.7   22.7
chem_energy       25.6   23.1   23.3   20.6   18.4   19.4   18.0   20.7
equity_index      22.9   19.2   20.4   17.5   25.2   20.4   26.0   23.1
livestock         28.2   22.3   32.1   23.5   20.0   21.2   18.3   23.6
metals            20.7   21.7   24.0   20.7   20.7   19.5   23.9   18.9
rubber_fiber      27.3   20.2   23.2   18.6   21.9   21.4   20.3   19.5
```
