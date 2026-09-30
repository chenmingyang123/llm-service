"""结构化输出的测试。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import completion
from app.main import app
from app.schemas import AnswerOut


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def fake_result(content: str, **kw):
    """冒充 call_chat 的返回值。"""
    base = {
        "content": content,
        "reasoning_content": "",
        "usage": {"in": 12, "out": 34, "cached": 0},
        "latency_s": 0.42,
        "provider": "deepseek",
        "model": "deepseek-flash",
    }
    base.update(kw)
    return base


VALID = '{"answer": "RAG 是检索增强生成", "confidence": 0.9, "tags": ["RAG"]}'


def test_strip_fence():
    assert completion.strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert completion.strip_fence('{"a": 1}') == '{"a": 1}'


def test_parse_success():
    a = completion.parse_structured(VALID)
    assert isinstance(a, AnswerOut)
    assert a.tags == ["RAG"]


def test_parse_salvages_surrounding_text():
    """模型经常先客套一句再给 JSON，应该救回来而不是直接判失败。"""
    a = completion.parse_structured('好的，这是结果：\n' + VALID + '\n希望有帮助。')
    assert a.confidence == 0.9


def test_parse_rejects_bad_confidence():
    """confidence 超出 0-1 必须被契约挡住 —— 这正是 Pydantic 存在的意义。"""
    bad = '{"answer": "x", "confidence": 9.9, "tags": []}'
    with pytest.raises(ValidationError):
        completion.parse_structured(bad)


def test_retry_then_succeed(monkeypatch):
    """第一次给垃圾输出，第二次给对的 —— 应该重试一次后成功。"""
    calls = []

    def fake(*args, **kwargs):
        calls.append(kwargs.get("messages") or args[1])
        if len(calls) == 1:
            return fake_result("抱歉，我无法回答。")
        return fake_result(VALID)

    monkeypatch.setattr(completion, "call_chat", fake)

    out = completion.structured_chat(_settings(), message="什么是 RAG？", retries=1)

    assert out["ok"] is True
    assert out["retries_used"] == 1
    assert len(calls) == 2
    assert any("修" in str(m) or "error" in str(m).lower() for m in calls[1])


def test_exhaust_retries_reports_failure(monkeypatch):
    """重试用尽后要如实说失败，把原始输出和错误原因都留下来。"""
    monkeypatch.setattr(completion, "call_chat", lambda *a, **k: fake_result("不是 JSON"))
    out = completion.structured_chat(_settings(), message="hi", retries=1)

    assert out["ok"] is False
    assert out["retries_used"] == 1
    assert out["validation_error"]
    assert out["raw"] == "不是 JSON"


def test_chat_endpoint(client, monkeypatch):
    monkeypatch.setattr(completion, "call_chat", lambda *a, **k: fake_result(VALID))
    r = client.post("/chat", json={"message": "什么是 RAG？"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["data"]["answer"]
    assert r.headers.get("x-request-id")


def test_chat_rejects_unknown_provider(client):
    """未知 provider 要在调用层就被挡住，而不是发出去碰运气。"""
    r = client.post("/chat", json={"message": "hi", "provider": "不存在的厂"})
    assert r.status_code == 502
    assert "未知 provider" in r.json()["detail"]


def _settings():
    from app.config import Settings
    return Settings()
