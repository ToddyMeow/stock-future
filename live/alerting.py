"""live/alerting.py — 告警通道（微信 + 短信 + DB 审计）

分级：
  info / warn  → Server 酱（微信）
  critical     → Server 酱 + 阿里云 SMS（+ 写 alerts 表）

降级：
  - SERVERCHAN_SEND_KEY 为空 → send_wechat 打日志 stub + 写 alerts 表，但返回 False
  - ALIYUN_SMS_ACCESS_KEY_ID 为空 → send_sms 打日志 + 追加一行到
    live/logs/sms_would_send.jsonl，返回 False

注意：所有 HTTP 调用用 httpx.AsyncClient（支持 MockTransport 注入，便于测试）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import httpx

from live.config import (
    ALIYUN_SMS_ACCESS_KEY_ID,
    PHONE_NUMBER,
    SERVERCHAN_SEND_KEY,
)

logger = logging.getLogger(__name__)


# =====================================================================
# HTTP transport（测试可注入 MockTransport）
# =====================================================================

_httpx_transport: Optional[httpx.AsyncBaseTransport] = None


def set_httpx_transport(transport: Optional[httpx.AsyncBaseTransport]) -> None:
    """测试注入 MockTransport 用；None 恢复默认。"""
    global _httpx_transport
    _httpx_transport = transport


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=_httpx_transport, timeout=10.0)


# =====================================================================
# 日志 stub 目录
# =====================================================================

_LOG_DIR = Path(__file__).parent / "logs"


def _sms_stub_path() -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR / "sms_would_send.jsonl"


# =====================================================================
# 微信 Server 酱推送
# =====================================================================


async def send_wechat(title: str, desp_markdown: str) -> bool:
    """Server 酱推送。SENDKEY 空时打日志 stub 返回 False。

    参考：https://sct.ftqq.com/
    endpoint：https://sctapi.ftqq.com/<SENDKEY>.send
    body：title + desp（支持 markdown）
    """
    # 运行时读 config 模块，方便测试 monkeypatch
    from live import config as _cfg

    key = _cfg.SERVERCHAN_SEND_KEY
    if not key:
        logger.info("[WECHAT STUB] title=%s desp=%s", title, desp_markdown[:200])
        return False

    url = f"https://sctapi.ftqq.com/{key}.send"
    try:
        async with _client() as client:
            resp = await client.post(
                url, data={"title": title, "desp": desp_markdown}
            )
            if resp.status_code >= 400:
                logger.warning(
                    "[WECHAT] send failed status=%s body=%s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("[WECHAT] send exception: %s", exc)
        return False


# =====================================================================
# 阿里云 SMS 推送
# =====================================================================


async def send_sms(phone: str, template_code: str, params: dict[str, Any]) -> bool:
    """阿里云短信发送。AccessKeyID 空 → stub（写 jsonl + 日志）返回 False。"""
    from live import config as _cfg

    ak_id = _cfg.ALIYUN_SMS_ACCESS_KEY_ID
    if not ak_id:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "phone": phone,
            "template_code": template_code,
            "params": params,
            "note": "stub_no_credentials",
        }
        path = _sms_stub_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "[SMS STUB] phone=%s tpl=%s params=%s (wrote %s)",
            phone,
            template_code,
            params,
            path,
        )
        return False

    # 真发：阿里云 python SDK（aliyun-python-sdk-core + dysmsapi）
    # 同步 SDK，用 run_in_executor 包装避免阻塞事件循环
    try:
        def _send_sync() -> bool:
            from aliyunsdkcore.client import AcsClient  # type: ignore
            from aliyunsdkcore.request import CommonRequest  # type: ignore

            client = AcsClient(
                _cfg.ALIYUN_SMS_ACCESS_KEY_ID,
                _cfg.ALIYUN_SMS_ACCESS_KEY_SECRET,
                "cn-hangzhou",
            )
            req = CommonRequest()
            req.set_accept_format("json")
            req.set_domain("dysmsapi.aliyuncs.com")
            req.set_method("POST")
            req.set_version("2017-05-25")
            req.set_action_name("SendSms")
            req.add_query_param("PhoneNumbers", phone)
            req.add_query_param("SignName", _cfg.ALIYUN_SMS_SIGN_NAME)
            req.add_query_param("TemplateCode", template_code)
            req.add_query_param("TemplateParam", json.dumps(params, ensure_ascii=False))
            resp = client.do_action_with_exception(req)
            # 返回 bytes/json；Code=OK 才算成功
            try:
                data = json.loads(resp)
                return data.get("Code") == "OK"
            except Exception:  # noqa: BLE001
                return True  # 没有异常即默认成功

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _send_sync)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[SMS] send exception: %s", exc)
        return False


# =====================================================================
# 分级路由
# =====================================================================


async def alert_escalate(
    severity: Literal["info", "warn", "critical"],
    event_type: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """分级路由 + 写 alerts 表。

    返回：
      {"alert_id": ..., "wechat": bool, "sms": bool, "severity": ...}
    """
    # 1) 写 alerts（若 DB 不可达则只打日志，不 block 告警发送）
    alert_id: Optional[int] = None
    try:
        from live.account_state import insert_alert

        alert = await insert_alert(severity, event_type, message, payload)
        alert_id = alert.id
    except Exception as exc:  # noqa: BLE001
        logger.warning("[alert] insert_alert 失败（降级继续）: %s", exc)

    # 2) 微信（info / warn / critical 都发）
    wechat_ok = await send_wechat(
        title=f"[{severity.upper()}] {event_type}",
        desp_markdown=message,
    )

    # 3) critical 额外发 SMS
    sms_ok = False
    if severity == "critical":
        phone = PHONE_NUMBER or os.environ.get("PHONE_NUMBER", "")
        tpl = os.environ.get("ALIYUN_SMS_TEMPLATE_CODE", "")
        sms_ok = await send_sms(
            phone=phone,
            template_code=tpl,
            params={"msg": message[:50]},
        )

    return {
        "alert_id": alert_id,
        "wechat": wechat_ok,
        "sms": sms_ok,
        "severity": severity,
    }


__all__ = [
    "send_wechat",
    "send_sms",
    "alert_escalate",
    "set_httpx_transport",
]
