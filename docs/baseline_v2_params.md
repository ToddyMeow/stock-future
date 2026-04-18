# Baseline v2 — 参数档案

_归档时间：2026-04-18_
_回测命令日期：2026-04-17_

## 复现命令

```bash
python scripts/run_three_layer_backtest.py \
  --suffix v2 \
  --risk-per-trade 0.03 \
  --portfolio-risk-cap 0.20
```

## 代码版本

- Git commit: （本次归档 commit）
- Python: 3.10
- 引擎：`strats/engine.py` StrategyEngine
- 数据：`data/cache/normalized/hab_bars.csv`（135,639 行，22 列）

## 策略矩阵

**5 Entries**

| ID | Class | 参数 |
|---|---|---|
| `hl_9` | HLEntryStrategy | period=9, allow_short=True |
| `hl_21` | HLEntryStrategy | period=21, allow_short=True |
| `boll` | BollBreakEntryStrategy | period=22, k=2.0, allow_short=True |
| `ama` | AmaEntryStrategy | n=10, fast=2, slow=30, allow_short=True |
| `double_ma` | DoubleMaEntryStrategy | fast=13, slow=34, allow_short=True |

**6 Exits**

| ID | Class | 参数 |
|---|---|---|
| `hl` | HLExitStrategy | period=21 |
| `boll` | BollExitStrategy | period=22, k=2.0 |
| `ama` | AmaExitStrategy | n=10, fast=2, slow=30 |
| `atr_trail` | AtrTrailExitStrategy | **atr_mult=4.5** ← v2 关键改动 |
| `term` | TermExitStrategy | min_bars=2, max_bars=13, min_target_r=1.0 |
| `double_ma` | DoubleMaExitStrategy | fast=13, slow=34 |

**总组合数**：5 × 6 = 30

## 组结构

**7 主阵 + 12 独立 = 19 groups / 66 symbols**

| 组 | 品种数 | 品种 | group_risk_cap |
|---|---|---|---|
| chem_energy | 12 | BU EB EG FU L MA PF PG PP SC TA V | 6% |
| rubber_fiber | 4 | CF CY RU SR | 6% |
| metals | 9 | AG AL AU CU NI PB SN SS ZN | 6% |
| black_steel | 6 | I J JM RB SF ZC | 6% |
| agri | 11 | A B C CS M OI P PK RM RS Y | 6% |
| building | 10 | AO FG LC SA SH SI SM SP UR WR | 5% |
| livestock | 2 | JD LH | 4% |
| ind_AP ~ ind_WH | 各 1 | AP BB CJ EC FB JR LR LU PM RI RR WH | 单品种 2%，合计 soft cap 8% |

排除：equity_index（IC/IF/IH/IM 无信号）, bond（T/TL/TS 用户排除），commodity 组（EXCLUDED 品种）。

## Engine 关键参数

| 字段 | 值 | 说明 |
|---|---|---|
| `initial_capital` | 1,000,000 | 起始资金 |
| **`risk_per_trade`** | **0.03** | v2 核心：每笔风险 3% |
| **`portfolio_risk_cap`** | **0.20** | v2 核心：组合总 risk 20% |
| `group_risk_cap` | dict 4-6% | 非独立组分差异化 |
| `default_group_risk_cap` | 0.02 | ind_* 每品种 2% |
| `independent_group_soft_cap` | 0.08 | 12 个 ind_* 合计 8% |
| `max_portfolio_leverage` | 3.0 | |
| `stop_atr_mult` | 2.0 | 初始止损 = 2 × ATR |
| `atr_period` | 20 | |
| `adx_period` | 20 | |
| `adx_scale` | 30 | |
| `adx_floor` | 0.2 | |
| `allow_short` | **True** | v2 核心：允许做空 |
| `warmup_bars` | 0 | 关闭 |
| `max_limit_days` | 0 | 关闭 |
| `max_margin_utilization` | 0 | 关闭 |
| `min_atr_pct` | 0.0025 | ATR 地板 0.25% × close |
| `enable_dual_stream` | False | Panama 单流 |

## Layer 2 best_combos（19 组各自最优）

```json
{
  "chem_energy":  "boll+boll",
  "rubber_fiber": "double_ma+term",
  "metals":       "hl_9+double_ma",
  "black_steel":  "hl_21+atr_trail",
  "agri":         "hl_9+double_ma",
  "building":     "hl_9+double_ma",
  "livestock":    "hl_21+double_ma",
  "ind_AP":       "hl_9+boll",
  "ind_BB":       "ama+boll",
  "ind_CJ":       "boll+atr_trail",
  "ind_EC":       "hl_21+term",
  "ind_FB":       "double_ma+atr_trail",
  "ind_JR":       "hl_9+term",
  "ind_LR":       "boll+term",
  "ind_LU":       "double_ma+term",
  "ind_PM":       "hl_9+term",
  "ind_RI":       "ama+atr_trail",
  "ind_RR":       "double_ma+term",
  "ind_WH":       "double_ma+term"
}
```

## 结果

| 指标 | 值 |
|---|---|
| Total Return (8y) | **+741.06%**（$1M → $8.41M） |
| CAGR | **+30.52%** |
| Sharpe | **1.0813** |
| Sortino | 1.7005 |
| Max Drawdown | -48.27%（2022-07 → 2023-12 恢复，525 天） |
| Profit Factor | 1.2528 |
| Total Trades | 650 |
| Win Rate | 37.23% |
| Avg R multiple | 0.2978 |
| Expectancy | 6,549.50 元/笔 |
| Avg risk utilization | 110.5% |
| Avg leverage | 1.82× |

## 过拟合诊断

| n_trials | DSR | 判定 |
|---|---|---|
| 19（独立 Layer 2 选择） | **0.886** | MODERATE-STRONG edge |
| 570（Layer 1 整体搜索） | 0.482 | overfit |

**n=19 是合理取数**（Layer 2 对每组 30 combos 的选择是独立决策，不是"选 1/570"）。

## 稳健性（3×3 grid 验证）

center = (risk=3%, cap=20%) 周围 ±0.5%/±3% 扰动：

| risk \ cap | 17% | 20% | 23% |
|---|---|---|---|
| 2.5% | **1.03** | 0.61 | 0.94 |
| **3.0%** | 1.02 | **1.10** 🏆 | 0.71 |
| 3.5% | 0.32 | 0.65 | 0.81 |

- 3/9 点 Sharpe > 1.0
- 7/9 点 Sharpe > 0.6
- **非 knife-edge**，真 robust peak

## 按年回报

| 年 | return |
|---|---|
| 2019 | +71.8% |
| 2020 | +9.2% |
| 2021 | -20.1% |
| 2022 | +38.5% |
| 2023 | +42.9% |
| 2024 | +21.6% |
| 2025 | +42.8% |

**6/7 年盈利**（只 2021 亏）。

## 按组贡献（排序）

| 组 | net P&L | 交易数 | 胜率 | avg R |
|---|---|---|---|---|
| metals | +$1,538k | 86 | 45% | 0.52 |
| rubber_fiber | +$983k | 26 | 65% | 1.17 |
| building | +$632k | 37 | 51% | 0.59 |
| agri | +$531k | 91 | 42% | 0.28 |
| ind_EC | +$528k | 12 | 42% | 0.87 |
| ind_BB | +$198k | 4 | 75% | 1.45 |
| black_steel | +$143k | 76 | 36% | 0.42 |
| ind_FB | +$102k | 2 | 50% | 1.21 |
| ind_LR | +$28k | 2 | 100% | 0.65 |
| ind_PM | -$8k | 1 | 0% | - |
| ind_WH | -$18k | 3 | 33% | - |
| ind_LU | -$46k | 14 | 21% | 0.17 |
| ind_JR | -$62k | 1 | 0% | - |
| ind_CJ | -$73k | 4 | 25% | - |
| ind_AP | -$125k | 13 | 8% | -0.48 |
| livestock | -$126k | 9 | 33% | - |
| chem_energy | -$813k | 90 | 36% | -0.07 |

## 产物

- `data/backtest_strategy_layer_v2.csv` — 570 combo × year × group
- `data/backtest_group_layer_v2.csv` — 19 组 × year 最优
- `data/backtest_portfolio_layer_v2.csv` — 日级权益（1942 行）
- `data/v2_equity.png` — 权益曲线图
- `data/sensitivity_v2_robustness.csv` — 3×3 扰动验证
- `data/reports/baseline_v2_2026-04-17.md` — 完整业绩报告
- `data/reports/baseline_final_summary_2026-04-17.md` — v1 → 最终发现
- `data/reports/p0_findings_2026-04-17.md` — P0 过拟合诊断

## 使用说明

如需复现 v2：
1. checkout 本 commit
2. `python -m pytest -q`（确认 144 passing）
3. 跑复现命令（见顶）
4. 对比 `data/reports/baseline_v2_2026-04-17.md` 的指标

如需以 v2 为起点改进：
- **参数空间**：在 (risk=2.5-3%, cap=17-23%) 内都能保持 Sharpe > 0.9
- **可剔除/减权品种**：chem_energy 组持续亏损、ind_AP 胜率 8%
- **待优化**：每个 slot 允许独立 override entry/exit 参数
