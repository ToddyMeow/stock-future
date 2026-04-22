"""live/config.py — 集中配置加载模块

职责：
  - 从 live/.env 读环境变量
  - 派生 async DSN（若未显式提供，由同步 DSN 推导）
  - 暴露常量给其它 live/ 模块使用

约定：
  - DATABASE_URL 是 URL-encoded 的同步 DSN（psycopg3 / psql 都直接可用）
  - DATABASE_URL_ASYNC 是 asyncpg / sqlalchemy+asyncpg 用的变体
  - 其它凭证未填时给空字符串，让上游代码按需降级（SMS stub 等）
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# =====================================================================
# 加载 .env
#   - 先加 live/.env（db / 告警凭证）
#   - 再加仓库根 .env（RQData 凭证放那里，研究和实盘共用）
#   - override=False：shell 环境 > live/.env > 根 .env（保证部署时可覆盖）
# =====================================================================
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH, override=False)

ROOT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if ROOT_ENV_PATH.exists():
    load_dotenv(ROOT_ENV_PATH, override=False)

# =====================================================================
# 数据库
# =====================================================================
# 同步 DSN — psycopg3 / psql 直用（URL 编码特殊字符）
DATABASE_URL: str = os.environ["DATABASE_URL"]

# 异步 DSN — 若 .env 没写，尝试把 postgresql:// 替成 postgresql+asyncpg://
_default_async = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
DATABASE_URL_ASYNC: str = os.environ.get("DATABASE_URL_ASYNC", _default_async)

# =====================================================================
# 业务参数
# =====================================================================
# 软熔断开关（Q3：一期先关，等实盘稳定再启用）
SOFT_STOP_ENABLED: bool = os.environ.get("SOFT_STOP_ENABLED", "false").lower() in ("true", "1", "yes")
# 软熔断阈值（日内回撤）：0.07 = 7%（Q3 用户决策）
SOFT_STOP_PCT: float = float(os.environ.get("SOFT_STOP_PCT", "0.07"))

# 初始资金 — engine / signal_service / daily_pnl_settlement 全部共用
# 2026-04-20 发现 engine 默认 100 万跑 warmup，和账户实际 25 万严重错配，
# 持仓 / cash / pending 数字全部按 100 万算。统一走 config 暴露。
INITIAL_CAPITAL: float = float(os.environ.get("INITIAL_CAPITAL", "1000000"))

# 时区（展示层用，非计算）
APP_TIMEZONE: str = os.environ.get("APP_TIMEZONE", "Asia/Shanghai")

# =====================================================================
# 告警通道凭证（P1c 用；未填空字符串 → 上游降级为 stub）
# =====================================================================
SERVERCHAN_SEND_KEY: str = os.environ.get("SERVERCHAN_SEND_KEY", "")

ALIYUN_SMS_ACCESS_KEY_ID: str = os.environ.get("ALIYUN_SMS_ACCESS_KEY_ID", "")
ALIYUN_SMS_ACCESS_KEY_SECRET: str = os.environ.get("ALIYUN_SMS_ACCESS_KEY_SECRET", "")
ALIYUN_SMS_SIGN_NAME: str = os.environ.get("ALIYUN_SMS_SIGN_NAME", "")
ALIYUN_SMS_TEMPLATE_CODE: str = os.environ.get("ALIYUN_SMS_TEMPLATE_CODE", "")

PHONE_NUMBER: str = os.environ.get("PHONE_NUMBER", "")

# =====================================================================
# RQData 凭证（data_pipeline 每日拉真实 bar 用；root .env 里带前导空格，需 strip）
# =====================================================================
RQDATAC_USER: str = os.environ.get("RQDATAC_USER", "").strip()
RQDATAC_PASSWORD: str = os.environ.get("RQDATAC_PASSWORD", "").strip()

# =====================================================================
# Combos CSV 路径（signal_service 读的策略组合文件）
#   默认 final_v3；可通过 .env 的 COMBOS_CSV_PATH 覆盖（相对/绝对皆可）
#   相对路径基于仓库根 /Users/mm/Trading/stock-future/
# =====================================================================
_DEFAULT_COMBOS_CSV = "data/runs/phase3/best_combos_stable_final_v3.csv"
_raw_combos = os.environ.get("COMBOS_CSV_PATH", _DEFAULT_COMBOS_CSV).strip()
_REPO_ROOT = Path(__file__).resolve().parents[1]
_p = Path(_raw_combos)
COMBOS_CSV_PATH: Path = _p if _p.is_absolute() else _REPO_ROOT / _p


__all__ = [
    "DATABASE_URL",
    "DATABASE_URL_ASYNC",
    "SOFT_STOP_ENABLED",
    "SOFT_STOP_PCT",
    "INITIAL_CAPITAL",
    "APP_TIMEZONE",
    "SERVERCHAN_SEND_KEY",
    "ALIYUN_SMS_ACCESS_KEY_ID",
    "ALIYUN_SMS_ACCESS_KEY_SECRET",
    "ALIYUN_SMS_SIGN_NAME",
    "ALIYUN_SMS_TEMPLATE_CODE",
    "PHONE_NUMBER",
    "RQDATAC_USER",
    "RQDATAC_PASSWORD",
    "COMBOS_CSV_PATH",
]
