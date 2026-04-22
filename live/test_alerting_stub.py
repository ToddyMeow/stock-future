"""alerting 降级 / stub 测试

覆盖：
  - SERVERCHAN_SEND_KEY 为空 → send_wechat 返回 False，且没 HTTP 请求发出
  - ALIYUN_SMS_ACCESS_KEY_ID 为空 → send_sms 返回 False，写一行 jsonl stub
  - alert_escalate('critical', ...) → send_wechat + send_sms 都被调到（key 填则 HTTP mock 命中）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("DATABASE_URL", "postgresql://placeholder:placeholder@localhost/placeholder")
os.environ.setdefault("DATABASE_URL_ASYNC", "postgresql+asyncpg://placeholder:placeholder@localhost/placeholder")


from live import alerting  # noqa: E402


# =====================================================================
# HTTP 请求追踪
# =====================================================================


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return httpx.Response(200, json={"code": 0, "message": "success"})


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def recorder():
    r = _Recorder()
    transport = httpx.MockTransport(r.handler)
    alerting.set_httpx_transport(transport)
    yield r
    alerting.set_httpx_transport(None)


@pytest.fixture
def tmp_logs(tmp_path, monkeypatch):
    """把 alerting._LOG_DIR 重定向到 tmp_path，避免污染项目目录。"""
    monkeypatch.setattr(alerting, "_LOG_DIR", tmp_path)
    return tmp_path


# =====================================================================
# 1. SENDKEY 空 → stub，无 HTTP 请求
# =====================================================================


@pytest.mark.asyncio
async def test_send_wechat_stubs_when_key_empty(recorder, monkeypatch):
    # 把 config.SERVERCHAN_SEND_KEY 置空
    from live import config as _cfg
    monkeypatch.setattr(_cfg, "SERVERCHAN_SEND_KEY", "")

    ok = await alerting.send_wechat("title", "body")
    assert ok is False
    assert recorder.calls == []  # 确认未发 HTTP


# =====================================================================
# 2. SENDKEY 有值 → 真发 HTTP（通过 MockTransport 命中）
# =====================================================================


@pytest.mark.asyncio
async def test_send_wechat_posts_when_key_set(recorder, monkeypatch):
    from live import config as _cfg
    monkeypatch.setattr(_cfg, "SERVERCHAN_SEND_KEY", "SCT_TEST_KEY")

    ok = await alerting.send_wechat("T", "desp")
    assert ok is True
    assert len(recorder.calls) == 1
    assert "sctapi.ftqq.com/SCT_TEST_KEY.send" in str(recorder.calls[0].url)


# =====================================================================
# 3. ALIYUN AK 空 → send_sms stub 写 jsonl
# =====================================================================


@pytest.mark.asyncio
async def test_send_sms_stub_writes_jsonl(tmp_logs, monkeypatch):
    from live import config as _cfg
    monkeypatch.setattr(_cfg, "ALIYUN_SMS_ACCESS_KEY_ID", "")

    ok = await alerting.send_sms(
        phone="13800138000", template_code="SMS_TEST", params={"msg": "hello"}
    )
    assert ok is False

    jsonl = tmp_logs / "sms_would_send.jsonl"
    assert jsonl.exists()
    content = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    row = json.loads(content[0])
    assert row["phone"] == "13800138000"
    assert row["template_code"] == "SMS_TEST"
    assert row["params"]["msg"] == "hello"
    assert row["note"] == "stub_no_credentials"


# =====================================================================
# 4. alert_escalate critical → 调 send_wechat AND send_sms
# =====================================================================


@pytest.mark.asyncio
async def test_alert_escalate_critical_fires_both(monkeypatch, recorder, tmp_logs):
    """critical 场景：wechat 和 sms 都要被调到。
    - wechat 走 MockTransport 回 200
    - sms AK 空 → 走 stub 路径（也算被"调到"，返回 False 合理）
    - insert_alert DB 失败要被吞掉不 block 告警
    """
    from live import alerting as _alert
    from live import config as _cfg

    monkeypatch.setattr(_cfg, "SERVERCHAN_SEND_KEY", "SCT_KEY")
    monkeypatch.setattr(_cfg, "ALIYUN_SMS_ACCESS_KEY_ID", "")  # stub 路径

    # 替换 insert_alert，让它不依赖 DB（返回 mock Alert）
    async def _fake_insert_alert(severity, event_type, message, payload=None):
        from live.db.models import Alert
        from datetime import datetime, timezone
        return Alert(
            id=1,
            event_at=datetime.now(timezone.utc),
            severity=severity,
            event_type=event_type,
            message=message,
            payload=payload,
        )

    monkeypatch.setattr(
        "live.account_state.insert_alert", _fake_insert_alert
    )

    # 跟踪 send_wechat / send_sms 的真正调用
    wechat_called = {"n": 0}
    sms_called = {"n": 0}
    orig_wechat = _alert.send_wechat
    orig_sms = _alert.send_sms

    async def _spy_wechat(title, desp_markdown):
        wechat_called["n"] += 1
        return await orig_wechat(title, desp_markdown)

    async def _spy_sms(phone, template_code, params):
        sms_called["n"] += 1
        return await orig_sms(phone, template_code, params)

    monkeypatch.setattr(_alert, "send_wechat", _spy_wechat)
    monkeypatch.setattr(_alert, "send_sms", _spy_sms)

    result = await _alert.alert_escalate(
        severity="critical",
        event_type="test_event",
        message="test message",
        payload={"k": "v"},
    )

    # wechat 和 sms 都被调用了（wechat HTTP 命中 MockTransport 返回 True；sms stub 返回 False）
    assert wechat_called["n"] == 1
    assert sms_called["n"] == 1
    assert result["severity"] == "critical"
    assert result["wechat"] is True
    assert result["sms"] is False
    # wechat HTTP 落在 recorder
    assert len(recorder.calls) == 1


# =====================================================================
# 5. alert_escalate info → 只调 wechat，不调 sms
# =====================================================================


@pytest.mark.asyncio
async def test_alert_escalate_info_no_sms(monkeypatch, recorder):
    from live import alerting as _alert
    from live import config as _cfg

    monkeypatch.setattr(_cfg, "SERVERCHAN_SEND_KEY", "SCT_KEY")

    async def _fake_insert_alert(severity, event_type, message, payload=None):
        from live.db.models import Alert
        from datetime import datetime, timezone
        return Alert(
            id=2,
            event_at=datetime.now(timezone.utc),
            severity=severity,
            event_type=event_type,
            message=message,
            payload=payload,
        )

    monkeypatch.setattr("live.account_state.insert_alert", _fake_insert_alert)

    sms_called = {"n": 0}
    orig_sms = _alert.send_sms

    async def _spy_sms_info(phone, template_code, params):
        sms_called["n"] += 1
        return await orig_sms(phone, template_code, params)

    monkeypatch.setattr(_alert, "send_sms", _spy_sms_info)

    result = await _alert.alert_escalate(
        severity="info", event_type="ok", message="just a note"
    )

    assert sms_called["n"] == 0
    assert result["sms"] is False
    # wechat 仍发了一次
    assert len(recorder.calls) == 1
