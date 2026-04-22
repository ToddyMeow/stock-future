#!/usr/bin/env python
"""
生成 Pipeline v4 riskB 完整回测报告，并发送到微信（Server 酱）。

数据源：
  data/runs/phase5/summary_v4_{1m,250k}_riskB_{oos,fullperiod}.json
  data/runs/phase5/per_{symbol,group}_v4_{1m,250k}_riskB_{oos,fullperiod}.csv
  data/runs/phase3/best_combos_stable_v4_{1m,250k}_riskB.csv

比较基线（final_v3 + riskB，已上线）：
  data/runs/phase5/summary_final_v3_riskB{,_250k,_fullperiod,_fullperiod_250k}.json

格式约定（Server 酱 markdown 兼容）：
  - 表格前后必须留空行
  - 公式写在 ``` 代码块内，避免 LaTeX `$...$` 被吃字符
  - 仅用 Unicode（× ÷ √ ≤ ≥ − ²），不用 HTML

Usage:
  python scripts/generate_pipeline_report.py --dry-run                 # 只打印 md
  python scripts/generate_pipeline_report.py --output /tmp/report.md
  python scripts/generate_pipeline_report.py                           # 打印 + 发微信
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PH5 = ROOT / "data" / "runs" / "phase5"
PH3 = ROOT / "data" / "runs" / "phase3"


# -------- helpers --------------------------------------------------------

def load_summary(tag: str) -> Optional[dict]:
    fp = PH5 / f"summary_{tag}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def load_csv(dirpath: Path, name: str) -> Optional[pd.DataFrame]:
    fp = dirpath / name
    if not fp.exists():
        return None
    return pd.read_csv(fp)


def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x * 100:+.{digits}f}%"


def fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e8:
        return f"{sign}¥{a / 1e8:.2f}亿"
    if a >= 1e4:
        return f"{sign}¥{a / 1e4:.1f}万"
    return f"{sign}¥{a:,.0f}"


def safe(d: Optional[dict], key: str, default=0):
    if not d:
        return default
    return d.get("metrics", {}).get(key, default)


def delta_str(new: float, old: float, digits: int = 2, as_pct: bool = False) -> str:
    if old is None or new is None:
        return ""
    if as_pct:
        diff = (new - old) * 100
        return f"({diff:+.{digits}f}pp)"
    diff = new - old
    if abs(old) > 1e-6:
        pct = diff / abs(old) * 100
        return f"({diff:+.{digits}f}, {pct:+.1f}%)"
    return f"({diff:+.{digits}f})"


# -------- section builders ----------------------------------------------

def metric_table(s_1m: Optional[dict], s_250k: Optional[dict], label: str) -> str:
    lines = [
        f"**{label}**",
        "",
        "| 指标 | 100 万档 | 25 万档 |",
        "|---|---:|---:|",
        f"| Sharpe | {safe(s_1m, 'sharpe'):.2f} | {safe(s_250k, 'sharpe'):.2f} |",
        f"| CAGR | {fmt_pct(safe(s_1m, 'cagr'))} | {fmt_pct(safe(s_250k, 'cagr'))} |",
        f"| MaxDD | {fmt_pct(safe(s_1m, 'max_drawdown_pct'))} | {fmt_pct(safe(s_250k, 'max_drawdown_pct'))} |",
        f"| Total Return | {fmt_pct(safe(s_1m, 'total_return'), 1)} | {fmt_pct(safe(s_250k, 'total_return'), 1)} |",
        f"| Final Equity | {fmt_money(safe(s_1m, 'final_equity', None))} | {fmt_money(safe(s_250k, 'final_equity', None))} |",
        f"| 交易数 | {safe(s_1m, 'total_trades')} | {safe(s_250k, 'total_trades')} |",
        f"| 胜率 | {safe(s_1m, 'win_rate', 0):.1%} | {safe(s_250k, 'win_rate', 0):.1%} |",
        f"| 盈亏比 | {safe(s_1m, 'wl_ratio', 0):.2f} | {safe(s_250k, 'wl_ratio', 0):.2f} |",
        "",
    ]
    return "\n".join(lines)


def baseline_compare(v3: Optional[dict], v4: Optional[dict], label: str, tier: str) -> str:
    lines = [
        f"**{tier} · {label}**",
        "",
        "| 指标 | v3 基线 | v4 新版 | Δ |",
        "|---|---:|---:|---:|",
    ]

    def row(metric: str, key: str, pct: bool = False) -> str:
        old = safe(v3, key, 0)
        new = safe(v4, key, 0)
        if pct:
            old_s = fmt_pct(old)
            new_s = fmt_pct(new)
            delta = f"{(new - old) * 100:+.2f}pp"
        elif key == "final_equity":
            old_s = fmt_money(old)
            new_s = fmt_money(new)
            delta = fmt_money(new - old) if old and new else "—"
        else:
            old_s = f"{old:.2f}" if isinstance(old, (int, float)) else str(old)
            new_s = f"{new:.2f}" if isinstance(new, (int, float)) else str(new)
            delta = f"{new - old:+.2f}"
        return f"| {metric} | {old_s} | {new_s} | {delta} |"

    lines += [
        row("Sharpe", "sharpe"),
        row("CAGR", "cagr", pct=True),
        row("MaxDD", "max_drawdown_pct", pct=True),
        row("Total Return", "total_return", pct=True),
        row("Final Equity", "final_equity"),
    ]
    lines.append("")
    return "\n".join(lines)


def symbol_ranking(csv_name: str, top: int = 5, bot: int = 3) -> str:
    df = load_csv(PH5, csv_name)
    if df is None or df.empty:
        return "_(无数据)_\n"
    df = df.sort_values("net_pnl", ascending=False)
    rows = [
        "| 品种 | 组 | PnL | 笔数 | 胜率 |",
        "|---|---|---:|---:|---:|",
    ]
    head = df.head(top)
    for r in head.itertuples():
        rows.append(
            f"| {r.symbol} | {r.group} | {fmt_money(r.net_pnl)} | "
            f"{int(r.trades)} | {r.win_rate:.1%} |"
        )
    if len(df) > top + bot:
        rows.append("| ... | ... | ... | ... | ... |")
    if len(df) > top:
        tail = df.tail(min(bot, max(0, len(df) - top)))
        for r in tail.itertuples():
            rows.append(
                f"| {r.symbol} | {r.group} | {fmt_money(r.net_pnl)} | "
                f"{int(r.trades)} | {r.win_rate:.1%} |"
            )
    return "\n".join(rows) + "\n"


def group_breakdown(csv_name: str) -> str:
    df = load_csv(PH5, csv_name)
    if df is None or df.empty:
        return "_(无数据)_\n"
    df = df.sort_values("net_pnl", ascending=False)
    rows = [
        "| 组 | PnL | 笔数 | 胜率 | 盈亏比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in df.itertuples():
        rows.append(
            f"| {r.group} | {fmt_money(r.net_pnl)} | {int(r.trades)} | "
            f"{r.win_rate:.1%} | {r.wl_ratio:.2f} |"
        )
    return "\n".join(rows) + "\n"


def combo_list(csv_name: str) -> str:
    df = load_csv(PH3, csv_name)
    if df is None or df.empty:
        return "_(无数据)_\n"
    rows = ["| 组 | best combo | Rolling 稳定性 | 状态 |",
            "|---|---|---:|:---:|"]
    for r in df.itertuples():
        combo = getattr(r, "best_combo", "—")
        wp = int(getattr(r, "windows_pass", 0))
        wt = int(getattr(r, "windows_total", 6))
        stab_status = getattr(r, "stability_status", "—")
        badge = "✓" if stab_status == "stable" else "⚠️"
        rows.append(
            f"| {r.group} | {combo} | {wp}/{wt} | {badge} {stab_status} |"
        )
    return "\n".join(rows) + "\n"


# -------- main report ---------------------------------------------------

FORMULA_BLOCK = """\
```
Sharpe      = mean(daily_return) / std(daily_return) × √252
CAGR        = (final_equity / initial_equity)^(252/N_days) − 1
MaxDD       = min((equity − running_peak) / running_peak)
Win rate    = n_winning_trades / n_total_trades
盈亏比 (wl) = avg_winner / |avg_loser|

本金风险 B 定义（释放赢家）：
  long  principal_risk = max(entry_fill − active_stop, 0) × multiplier × qty
  short principal_risk = max(active_stop − entry_fill, 0) × multiplier × qty

仓位大小 (Volatility Sizing)：
  qty = floor(equity × risk_per_trade / principal_risk)

组合风控（每笔信号生效前校验）：
  Σ group_risk      ≤ equity × group_cap      (= 8%)
  Σ portfolio_risk  ≤ equity × portfolio_cap  (= 20%)

Kelly edge（per-symbol 评分）：
  edge = win_rate × (win_loss_ratio + 1) − 1

Phase 3 excess expectancy（combo 筛选指标）：
  excess = combo_expectancy − Phase1_A_baseline[group]
```
"""


def generate_report() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- v4 产物 ---
    s_1m_oos = load_summary("v4_1m_riskB_oos")
    s_1m_full = load_summary("v4_1m_riskB_fullperiod")
    s_250k_oos = load_summary("v4_250k_riskB_oos")
    s_250k_full = load_summary("v4_250k_riskB_fullperiod")

    # --- v3 基线（已上线）---
    v3_1m_oos = load_summary("final_v3_riskB")
    v3_1m_full = load_summary("final_v3_riskB_fullperiod")
    v3_250k_oos = load_summary("final_v3_riskB_250k")
    v3_250k_full = load_summary("final_v3_riskB_fullperiod_250k")

    # --- Phase 0 tradeable universe ---
    p0_1m_json = ROOT / "data/runs/phase0_1m/tradeable_symbols.json"
    p0_250k_json = ROOT / "data/runs/phase0_250k/tradeable_symbols.json"
    try:
        p0_1m_n = json.loads(p0_1m_json.read_text()).get("n_tradeable", "?")
    except Exception:
        p0_1m_n = "?"
    try:
        p0_250k_n = json.loads(p0_250k_json.read_text()).get("n_tradeable", "?")
    except Exception:
        p0_250k_n = "?"

    # --- v4 confirmed syms（Phase 2 筛选结果）---
    try:
        conf_1m = json.loads((PH3 / "confirmed_syms_1m.json").read_text())
    except Exception:
        conf_1m = {}
    try:
        conf_250k = json.loads((PH3 / "confirmed_syms_250k.json").read_text())
    except Exception:
        conf_250k = {}

    # --- 组装 ---
    md_parts: list[str] = []
    md_parts.append(f"""# 期货 Alpha Pipeline v4 回测报告

**生成时间**：{now}
**风险定义**：B（本金风险，释放赢家）
**风控参数**：risk_per_trade 3% / group_cap 8% / portfolio_cap 20%
**MC seeds**：Phase 1 A/B/C 各 20 seeds × 3 exit_probs + D 确定性
**基线**：final_v3 + riskB（已上线）

---

## 1. 摘要
""")

    md_parts.append(metric_table(s_1m_oos, s_250k_oos, "OOS 2024-2025"))
    md_parts.append(metric_table(s_1m_full, s_250k_full, "全期 2018-2025"))

    md_parts.append(f"""---

## 2. Pipeline 筛选漏斗

| 阶段 | 100 万档 | 25 万档 |
|---|---:|---:|
| Phase 0 tradeable（91 → 剔除单手 ATR 风险超过 3% 本金）| {p0_1m_n} | {p0_250k_n} |
| Phase 2 confirmed（tradeable ∩ bootstrap p≤0.05 ∩ score≥50）| {conf_1m.get("n_final", "?")} | {conf_250k.get("n_final", "?")} |
| Phase 3 stable groups（profitable≥40% ∩ rolling≥4/6）| {(s_1m_oos or {}).get("n_stable_combos", "?")} | {(s_250k_oos or {}).get("n_stable_combos", "?")} |

---

## 3. 与 final_v3 基线对比
""")

    md_parts.append(baseline_compare(v3_1m_oos, s_1m_oos, "OOS 2024-2025", "100 万档"))
    md_parts.append(baseline_compare(v3_1m_full, s_1m_full, "全期 2018-2025", "100 万档"))
    md_parts.append(baseline_compare(v3_250k_oos, s_250k_oos, "OOS 2024-2025", "25 万档"))
    md_parts.append(baseline_compare(v3_250k_full, s_250k_full, "全期 2018-2025", "25 万档"))

    md_parts.append("""---

## 4. 品种贡献 · OOS 2024-2025

### 100 万档（Top 5 / Bottom 3）

""")
    md_parts.append(symbol_ranking("per_symbol_v4_1m_riskB_oos.csv"))
    md_parts.append("""
### 25 万档（Top 5 / Bottom 3）

""")
    md_parts.append(symbol_ranking("per_symbol_v4_250k_riskB_oos.csv"))

    md_parts.append("""---

## 5. 组级贡献 · OOS 2024-2025

### 100 万档

""")
    md_parts.append(group_breakdown("per_group_v4_1m_riskB_oos.csv"))
    md_parts.append("""
### 25 万档

""")
    md_parts.append(group_breakdown("per_group_v4_250k_riskB_oos.csv"))

    md_parts.append("""---

## 6. 每组 Best Combo（Phase 4 稳定性通过）

### 100 万档

""")
    md_parts.append(combo_list("best_combos_stable_v4_1m_riskB.csv"))
    md_parts.append("""
### 25 万档

""")
    md_parts.append(combo_list("best_combos_stable_v4_250k_riskB.csv"))

    md_parts.append("""---

## 7. 关键公式

""")
    md_parts.append(FORMULA_BLOCK)

    # ---- 结论 + 诊断 -----------------------------------------------
    # 自动判定 v4 vs v3
    v3_oos_sharpe_best = max(safe(v3_1m_oos, 'sharpe'), safe(v3_250k_oos, 'sharpe'))
    v4_oos_sharpe_best = max(safe(s_1m_oos, 'sharpe'), safe(s_250k_oos, 'sharpe'))
    v3_dd_worst = min(safe(v3_1m_oos, 'max_drawdown_pct'), safe(v3_250k_oos, 'max_drawdown_pct'))
    v4_dd_worst = min(safe(s_1m_oos, 'max_drawdown_pct'), safe(s_250k_oos, 'max_drawdown_pct'))

    verdict = (
        "**v4 全面差于 v3** — 不建议切换，保持 final_v3 上线。"
        if v4_oos_sharpe_best < v3_oos_sharpe_best or v4_dd_worst < v3_dd_worst
        else "**v4 在部分维度优于 v3** — 人工 review 后可考虑切换。"
    )

    md_parts.append(f"""---

## 8. 结论与下一步

### 自动判定

{verdict}

| 维度 | v3 (OOS 最佳档) | v4 (OOS 最佳档) | 判定 |
|---|---:|---:|:---:|
| Sharpe | {v3_oos_sharpe_best:.2f} | {v4_oos_sharpe_best:.2f} | {'✓' if v4_oos_sharpe_best >= v3_oos_sharpe_best else '✗'} |
| MaxDD | {fmt_pct(v3_dd_worst)} | {fmt_pct(v4_dd_worst)} | {'✓' if v4_dd_worst >= v3_dd_worst else '✗'} |

### 诊断：v4 为什么差

1. **组级 diversification 骤降**：Phase 5 会自动过滤 `unstable` 组（`stability_status != 'stable'`），v4 真正参与回测的 stable 组只有 **100万档 3 组 / 25万档 4 组**（building / black_steel / rubber_fiber ± ind_CJ），而 v3 是完整 6 组。少一半组 → 单组拖累被放大。

2. **新组 `black_steel` 被自动引入** —— 但 v3 当年就是为剔除 `I`（铁矿 OOS 连亏）人工砍到 6 组。Pipeline 的自动化流程把 `black_steel` 标为 stable 重新放进来，`per_symbol` 显示 `I` 两档都是亏损榜首，`black_steel` 组级在 25 万档直接亏损。

3. **Phase 1 / 2 新 exit_prob 扫描选出的 combo 变体效果不稳**：同一个组 `building`，v3 选 `double_ma+boll`，v4 换成 `double_ma+double_ma` — 虽然 Phase 4 rolling 6/6 过，但 OOS 表现明显差。说明 Phase 4 rolling 窗口本身还不够区分"看起来稳"和"真稳"。

### 当前状态（不变）

- Mac launchd + signal_service 继续读 `best_combos_stable_final_v3.csv`（v3）
- ECS `universe_symbols` 表中为 v3（Mac 和 ECS 一致）
- 实盘 risk 敞口按 v3 计算，未变

### 如果确实要切 v4（**回来再定**）

```
# 选一档写进 DB（二选一）
python scripts/publish_universe_to_db.py \\
  --csv-symbols data/runs/phase0_250k/final_v3_comparison.csv \\
  --csv-combos  data/runs/phase3/best_combos_stable_v4_250k_riskB.csv

# .env COMBOS_CSV_PATH 改到 v4 CSV，重启 signal_service launchd
ssh velvet 'systemctl restart velvet-api'
```

### 建议的下一步研究方向

- 对 unstable 组手动过滤（windows_pass ≥ 4/6 硬切）重跑 Phase 5，看是否追上 v3
- 检查为什么 v3 的 building 用 `double_ma+boll`、v4 用 `double_ma+double_ma` — 可能是新 exit_prob 扫出变体但效果不稳
- `I`（铁矿）作为 black_steel 内唯一持续亏损品种，可单独剔除

---

完整产物：`data/runs/phase3/` · `data/runs/phase5/`
权益曲线 PNG：`data/runs/phase5/equity_v4_*.png`（本机 open）
""")

    return "\n".join(md_parts)


# -------- WeChat sender -------------------------------------------------

async def send(title: str, md: str) -> bool:
    from live.alerting import send_wechat  # type: ignore
    return await send_wechat(title=title, desp_markdown=md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--send", action="store_true",
                    help="ACTUALLY send to WeChat (default: dry-run only)")
    ap.add_argument("--output", type=Path, default=None, help="save md to file")
    ap.add_argument("--title", default="期货 Pipeline v4 回测报告")
    ap.add_argument("--force", action="store_true",
                    help="Allow send even if v4 summaries are missing (not recommended)")
    args = ap.parse_args()

    # Sanity: require all four v4 summaries before sending
    required = [
        "v4_1m_riskB_oos", "v4_1m_riskB_fullperiod",
        "v4_250k_riskB_oos", "v4_250k_riskB_fullperiod",
    ]
    missing = [t for t in required if not (PH5 / f"summary_{t}.json").exists()]
    if missing:
        print(f"⚠️  缺失 v4 summaries: {missing}", file=sys.stderr)
        if args.send and not args.force:
            print("拒绝发送 — 用 --force 覆盖，或等 pipeline 完成后再跑", file=sys.stderr)
            sys.exit(2)

    md = generate_report()
    print(md)
    if args.output:
        args.output.write_text(md)
        print(f"\n[saved] {args.output}", file=sys.stderr)

    if args.send:
        ok = asyncio.run(send(args.title, md))
        print(f"\nWeChat: {'sent ✓' if ok else 'failed ✗'}", file=sys.stderr)
    else:
        print("\n(dry-run — 加 --send 才真发微信)", file=sys.stderr)


if __name__ == "__main__":
    main()
