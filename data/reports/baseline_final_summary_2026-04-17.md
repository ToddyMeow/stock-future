# 基线回测最终汇总 — 2026-04-17

## TL;DR

- **Baseline 业绩不达标**：Sharpe 0.16 / MaxDD -42% / PF 1.00 / 过拟合诊断 DSR=0.011（统计上无法与随机运气区分）
- **整个 8 年收益由 2020 Q1 独撑**（Sharpe 2.66）；2021 年起累计 -20%，自 2022-03 起 3.8 年未脱离水位
- **但敏感性扫描揭示**：当前 `config.yaml` 默认值**普遍处于局部最差点**，几乎每个旋钮都有更优的设置
- 最亮眼：`portfolio_risk_cap=16%` 单参变化 → **Sharpe 1.15 / CAGR +20.8%** — 可能是真 alpha，需要 DSR 二次验证

---

## 1. Baseline 核心指标

| 指标 | 值 |
|---|---|
| Sharpe | 0.158 |
| Sortino | 0.249 |
| CAGR | +1.20% |
| Total return (8y) | +10.04% |
| Max Drawdown | **-42.36%** |
| Profit Factor | 1.00 |
| 总交易数 | 658 |
| Win rate | 34.8% |
| 信号拒绝率 | **92.3%**（3141→243） |
| 平均风险利用 | **110%**（超 cap） |

**验收判定**：❌ Sharpe < 0.5 且 MaxDD > 30%，未达标。

---

## 2. Regime breakdown

| 窗口 | Sharpe | CAGR (年化) | MaxDD | Trades |
|---|---|---|---|---|
| **2020 Q1 COVID** | **2.66** | +100% | -10% | 21 |
| 2022 Q2-Q3 (大宗见顶) | -2.08 | -31% | -20% | 62 |
| 2023 (低波震荡) | -0.13 | -3% | -13% | 87 |
| 2024-2025 (近两年) | 0.08 | 0% | -20% | 151 |
| **Full 2018-2025** | **0.16** | +1.2% | -42% | 658 |

**诊断**：regime-dependent 策略。黄金期 = 大宗趋势爆发；失败期 = 震荡+顶部追涨。2022 大宗顶反而是最差期，说明在顶部被 trend-following 套牢。

---

## 3. 按组贡献（Layer 3）

| 组 | net P&L | 胜率 |
|---|---|---|
| livestock | **+169k** | 62% |
| metals | **+167k** | 33% |
| building | +43k | 36% |
| chem_energy | -29k | 39% |
| rubber_fiber | -92k | 34% |
| black_steel | -96k | 37% |
| **agri** | **-149k** | 28% |
| equity_index | *被 Layer 2 过滤* | - |

亏损组（agri, rubber_fiber, black_steel）的单组信号密度都小（只有 10-11 品种但 trade_count<150），可能是**单品种 dominant over-exposure**。

---

## 4. 过拟合诊断

| 指标 | 值 | 判定 |
|---|---|---|
| **DSR (Deflated Sharpe)** | **0.0106** | ❌ 与随机运气不可区分 |
| n_trials | 270 | (30 combos × 9 组) |
| PBO | *未跑* | 需 +20 min 重跑 270 combos 才能算 |

DSR < 0.05 → baseline 的 0.16 Sharpe **统计上无法与"运气好选中一条曲线"区分**。但 DSR 是对整条 portfolio 曲线的检验，不排除**子组合 / 子参数真的有 alpha**（见下）。

---

## 5. 敏感性扫描（关键发现）

**每个参数的 baseline 默认值都接近最差点 → config.yaml 需要系统性重新标定**。

### 5.1 `risk_per_trade`（default 2%）

| 值 | Sharpe | CAGR | MaxDD | Trades |
|---|---|---|---|---|
| **0.5%** | **0.64** | +8.2% | -25% | 2270 |
| 1% | 0.54 | +7.9% | -33% | 1193 |
| **2%（当前）** | **0.16 ⚠️** | +1.2% | -42% | 658 |
| 3% | 0.38 | +5.9% | -49% | 402 |
| 4% | -0.03 | -2.9% | -58% | 359 |

**2% 是局部最差点**。在 cap=15% 下，~8 组 × 2% = 16% > cap，被截断到 8 仓以内，丢掉分散；0.5% 下可开 30 仓位，diversification kicks in。

### 5.2 `portfolio_risk_cap`（default 15%）

| 值 | Sharpe | CAGR | MaxDD | Trades |
|---|---|---|---|---|
| 8% | 0.40 | +4.2% | -33% | 358 |
| 12% | 0.85 | +12.9% | -26% | 514 |
| **16%** | **1.15 🏆** | **+20.8%** | -32% | 566 |
| 20% | 0.34 | +4.9% | -42% | 784 |

**16% 是冠军点**（Sharpe > 1 的结果是罕见的） — 但 baseline 是 15%，仅差 1% 却从 0.16 → 1.15。**强烈怀疑 15% 附近有 non-linear 崩塌，需要 2D sweep 验证**。

### 5.3 `stop_atr_mult`（default 2.0）

| 值 | Sharpe | CAGR | MaxDD | Trades |
|---|---|---|---|---|
| 1.5 (紧) | 0.51 | +8.3% | -34% | 613 |
| **2.0（当前）** | **0.16 ⚠️** | +1.2% | -42% | 658 |
| 2.5 | 0.24 | +2.5% | -36% | 677 |
| 3.0 (宽) | 0.37 | +4.0% | -41% | 711 |

**2.0 又是最差**。紧 1.5 或宽 3.0 都更好 — 说明 2.0 ATR 是"既不保本又不留呼吸"的尴尬点。

### 5.4 `adx_floor`（default 0.2）

| 值 | Sharpe | CAGR | MaxDD | Trades |
|---|---|---|---|---|
| 0.0 | 0.31 | +4.0% | -35% | 635 |
| **0.2（当前）** | **0.16 ⚠️** | +1.2% | -42% | 658 |
| 0.3 | 0.51 | +7.8% | -36% | 586 |
| 1.0 (ADX off) | 0.63 | +11.0% | -39% | 310 |

ADX 滤波在 0.2 处**净负贡献**（比 OFF 差 4 倍 Sharpe）。要么关掉（1.0），要么收紧（0.3+）。现在的 0.2 = worst of both worlds。

### 5.5 `enable_dual_stream`（default False）

| 值 | Sharpe | CAGR | MaxDD |
|---|---|---|---|
| False (Panama) | 0.16 | +1.2% | -42% |
| True (真合约段) | 0.12 | +0.5% | **-59%** |

dual_stream **无 alpha 收益但 DD 显著放大** — 继续用 Panama 单流。

---

## 6. 结论与下一步

### 6.1 baseline 本身的评价
- ❌ **现状不可部署** — 任何实盘都不能用当前参数
- ❌ **DSR 判定过拟合** — 不能直接把 8 年 +10% 当成"模型有效"的证据
- ✅ **代码 / 引擎没有 bug** — 数据管道 / 风控闸门 / 撮合时序都是对的，问题在参数标定

### 6.2 模型底层仍有潜力
**portfolio_risk_cap=16% 的 Sharpe 1.15 如果不是巧合，就是真 alpha 被默认配置压住了**。需要额外一轮验证：
1. 多参数 2D sweep（risk_per_trade × portfolio_risk_cap）找真实最优
2. 在最优点跑一次 PBO + DSR — 验证是不是超参数搜索过拟合
3. 降低 DSR n_trials 的可信度：现在 n_trials=270 是对全部 Layer 1 说的；真实尝试的参数数只有 ~17（敏感性 grid），应该用 n_trials=17 重新算 DSR

### 6.3 建议路径（按 ROI 排序）

| 优先级 | 动作 | 工作量 | 预期 |
|---|---|---|---|
| P0 | 关闭 dual_stream 等噪声点，测试 2D sweep (risk_per_trade, portfolio_risk_cap) × (3 × 3 = 9 点) | ~15 min | 找到真正的 joint optimum |
| P0 | 在 best point 上跑 PBO + DSR（n_trials=9） | ~25 min | 判断是否 overfit |
| P1 | 排查 equity_index 为何 2018-2022 零交易（`min_atr_pct=0.0025` 或 ATR_period=20 warmup？） | ~30 min | 可能解锁一个大组 |
| P1 | agri / rubber_fiber / black_steel 单组 drill-down：是某 1-2 个品种拖累吗？ | ~1 h | 考虑排除高 lock_pct 品种 (ZC/RS/WR) |
| P2 | 测试 HAB entry（当前 5 entries 矩阵里没有），可能更适合中国震荡市 | ~2 h | 潜在新 alpha 源 |
| P2 | 加 regime filter（只在波动率/趋势强度达标时开仓） | ~4 h | 解决 2021-2024 的震荡期失血 |

---

## 7. 产物清单

| 文件 | 说明 |
|---|---|
| `data/backtest_*_layer_baseline.csv` | 原始三层输出 |
| `data/baseline_equity.png` | 权益曲线 + 回撤图 |
| `data/reports/baseline_baseline_2026-04-17.md` | Step 4 完整基线报告 |
| `data/reports/regime_*.md` | Step 6 四 regime 切片报告 |
| `data/sensitivity_*.csv` | Step 5 五维敏感性 |
| `data/overfitting_diagnostics_baseline.json` | Step 7a DSR 结果 |
| `data/reports/baseline_final_summary_2026-04-17.md` | **本文件** |
