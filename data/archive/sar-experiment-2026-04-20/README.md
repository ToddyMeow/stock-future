# SAR 实验归档 — 2026-04-20

## 结论（TL;DR）

**丢弃。Stop-and-reverse 全局开关在 2024-2025 OOS 下负 alpha，per-slot mixed 组合仅与 baseline 打平。Main 分支不采用。**

## 实验背景

在现有 Phase0-5 pipeline 里测试一个新想法："当任何持仓因 stop-loss 平仓时，立刻反向开仓"（classic stop-and-reverse / SAR pattern）。目标：**评估 SAR 能不能作为实际交易策略加入 live portfolio**。

配置：
- 风控：`risk_per_trade=0.03, group_cap=0.08, portfolio_cap=0.20`（即 3/8/20）
- SAR 参数：反向仓 stop = 当前 ATR × 3.0，最多级联 3 次
- 对比口径：**同一配置**下 baseline (SAR off) vs pure SAR vs per-slot mixed

## OOS 结果（2024-01 → 2025-12，~15 个月）

| Portfolio | Slots | Trades | Net PnL | Sharpe | CAGR | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline** (all SAR-off) | 7 | 78 | **+664,276** | **2.08** | 41.7% | −19.4% |
| **Pure SAR** (all SAR-on) | 6 | 85 | **−71,797** | 0.15 | 1.1% | −28.5% |
| **Mixed** (4 SAR + 3 baseline) | 7 | 99 | **+788,591** | 1.77 | 38.6% | **−18.2%** |

**核心发现：**
1. Pure SAR 在 OOS **完全崩盘**——IS 看起来有 alpha，OOS 亏 7 万。教科书级 regime overfitting。
2. IS mixed 预测 sum_exp_mean = 217,072（baseline 的 +21%），但 OOS 仅赢 baseline 18%（一半水分被 IS→OOS degradation 吞掉）。
3. IS 选出的 4 个 SAR 组合里，**只有 ind_AP 在 OOS 确认 SAR 边际**（+38k vs baseline +23k）；其他 3 个 SAR 组合在 OOS 大幅劣化甚至亏钱（rubber_fiber SAR −10k vs baseline +117k = -127k swing）。

## 目录结构

```
sar-experiment-2026-04-20/
├── README.md                              本文件
├── phase3/                                14 个 CSV
│   ├── combo_grid_risk3cap8_{baseline,sar}.csv
│   ├── trades_risk3cap8_{baseline,sar}.csv
│   ├── combo_ranked_risk3cap8_{baseline,sar}.csv
│   ├── best_combos_risk3cap8_{baseline,sar}.csv
│   ├── phase4_rolling_risk3cap8_{baseline,sar}.csv
│   ├── best_combos_stable_risk3cap8_{baseline,sar,mixed}.csv
│   └── sar_diff_risk3cap8_sar.csv         baseline vs SAR per-combo delta
└── phase5/                                18 个文件
    ├── backtest_portfolio_layer_risk3cap8_{baseline,sar,mixed}.csv  每日权益曲线
    ├── per_group_risk3cap8_{baseline,sar,mixed}.csv                 per-group OOS PnL
    ├── per_symbol_risk3cap8_{baseline,sar,mixed}.csv                per-symbol OOS PnL
    ├── reject_distribution_risk3cap8_{baseline,sar,mixed}.csv       风控拒单分布
    ├── trades_risk3cap8_{baseline,sar,mixed}.csv                    完整 OOS 交易
    └── summary_risk3cap8_{baseline,sar,mixed}.json                  Portfolio 指标
```

## 代码位置

**所有改动在 `sar-experiment` 分支，不合并 main。**

### 修改的文件
- `strats/engine_config.py` — EngineConfig 新增 4 个 SAR 字段 + StrategySlot 新增 3 个 per-slot override 字段
- `strats/engine.py` — `_resolve_slot_sar()`, `_try_synthesize_reverse_entry()`, phase 1/2 hooks (~100 行)
- `strats/config_loader.py` — yaml → EngineConfig 的 SAR 透传
- `config.yaml` — `reverse_on_stop` / `reverse_stop_atr_mult` / `reverse_chain_max` 默认 off
- `scripts/run_phase3_combo_selection.py` — CLI flags `--reverse-on-stop` 等 + cagr bug fix (complex → −1.0) + trades CSV 加 `entry_type` / `reverse_leg_count` 列
- `scripts/run_phase5_oos.py` — 从 best_combos_stable 读 `reverse_on_stop` 做 per-slot 配置

### 新增脚本
- `scripts/research/analyze_phase3_sar_diff.py` — baseline vs SAR 的 per-combo delta 报告
- `scripts/research/analyze_phase3_mixed.py` — per-group pick winner 生成 mixed best_combos

### Plan 文件
- `~/.claude/plans/portfolio-random-entry-random-exit-majestic-spark.md` — 原始计划与决策

## 复现命令

从 `sar-experiment` 分支 checkout 后：

```bash
# Step 1: Phase3 双跑（baseline + SAR at 3/8/20）
python scripts/run_phase3_combo_selection.py \
    --risk-per-trade 0.03 --group-cap 0.08 --portfolio-cap 0.20 \
    --output-tag risk3cap8_baseline

python scripts/run_phase3_combo_selection.py \
    --risk-per-trade 0.03 --group-cap 0.08 --portfolio-cap 0.20 \
    --reverse-on-stop --reverse-stop-atr-mult 3.0 \
    --output-tag risk3cap8_sar

# Step 2: 分析与 diff 报告
python scripts/research/analyze_phase3.py --tag risk3cap8_baseline
python scripts/research/analyze_phase3.py --tag risk3cap8_sar
python scripts/research/analyze_phase3_sar_diff.py \
    --baseline-tag risk3cap8_baseline --sar-tag risk3cap8_sar

# Step 3: Phase4 rolling stability
python scripts/research/analyze_phase4_rolling.py --tag risk3cap8_baseline
python scripts/research/analyze_phase4_rolling.py --tag risk3cap8_sar

# Step 4: Mixed portfolio 选择（per-group pick winner）
python scripts/research/analyze_phase3_mixed.py \
    --baseline-tag risk3cap8_baseline --sar-tag risk3cap8_sar \
    --mixed-tag risk3cap8_mixed

# Step 5: Phase5 OOS 3 个组合
python scripts/run_phase5_oos.py --tag risk3cap8_baseline --group-cap 0.08
python scripts/run_phase5_oos.py --tag risk3cap8_sar --group-cap 0.08
python scripts/run_phase5_oos.py --tag risk3cap8_mixed --group-cap 0.08
```

## 后续行动建议

1. **Live portfolio 继续用 baseline 的 `risk3cap6` 版本**（6 个 stable combos），本次 3/8/20 + SAR 的探索结论为负
2. 如需保留 ind_AP 的 SAR 信号优势，**需要更长 OOS（≥ 2 年）再验证**，样本太小不足以单独落地
3. `sar-experiment` 分支保留**作为负面结论的 archive**，未来若需复活可 `git checkout sar-experiment`
4. 本次顺带修了一个既有 bug：`run_phase3_combo_selection.py` 的 cagr 计算在 equity 跌至负数时返回 complex number 导致 silent drop 一整个 combo。这个修复值得 cherry-pick 到 main（独立于 SAR 代码）

## 实验耗时

- 设计与问答：~45 min
- 代码改动：~30 min（engine 改动 + Phase3/5 脚本 + 2 个新 scripts）
- 计算：~30 min（Phase3 双跑 8 min × 2，Phase5 OOS < 2 min × 3，分析即时）
- 总：~2 小时
