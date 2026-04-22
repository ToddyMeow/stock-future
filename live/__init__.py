"""live — 半自动实盘交易系统后端模块

子模块一览：
  config          读 .env 的统一配置
  data_pipeline   CSV → 云 PG bars 表（mock） + bars 查询
  signal_service  信号服务：读持仓 → engine.run → 写 instructions + engine_states
  account_state   (P1c) SQLAlchemy async 6 表 CRUD + partial fill 状态机
  web.api         (P1c) FastAPI REST 接口
  alerting        (P1c) Server 酱 / 阿里云 SMS / alerts 表路由
  soft_stop       (P1c) 日内权益回撤熔断
  report_service  (P1c) Jinja2 日报 HTML + Server 酱摘要推送
"""
