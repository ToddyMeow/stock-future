# Scripts

本目录现在只保留两类东西：

- `supported commands`
  - 稳定 CLI 入口，负责调库，不再承担共享业务装配职责。
  - 例如：`run_three_layer_backtest.py`、`run_phase3_combo_selection.py`、`run_phase5_oos.py`、`run_random_benchmark.py`
- `research-only commands`
  - 一次性分析、诊断、画图、报告生成。
  - 已迁到 `scripts/research/`，这些脚本可以依赖 `strats.factory` / `strats.research_support`，但不应该成为别的运行路径的依赖。

## Rules

- `live/` 不允许导入 `scripts/`
- `strats/` 不允许导入 `scripts/`
- 共享装配走 `strats.registry` 与 `strats.factory`
- 共用研究辅助函数走 `strats.research_support`
- `scripts/run_three_layer_backtest.py` 现在是 CLI wrapper + backward-compatible shim，不再是唯一真相层

## Current Shared Entry Points

- `strats.factory.build_engine_config()`
- `strats.factory.build_strategy_slots_from_combos()`
- `strats.factory.build_from_yaml()`
- `strats.research_support.load_hab_bars()`
- `strats.research_support.yearly_stats_from_trades()`
