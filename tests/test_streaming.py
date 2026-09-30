"""流式 SSE 的测试。全 mock。"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import streaming
from app.config import Settings
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class FakeResp:
    """冒充 urllib 的响应对象：能进 with，能逐行迭代。"""

    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._lines)


def sse(*objs) -> list[bytes]:
    out = [f"data: {json.dumps(o, ensure_ascii=False)}".encode() for o in objs]
    out.append(b"data: [DONE]")
    return out


def install(monkeypatch, lines):
    monkeypatch.setattr(streaming.urllib.request, "urlopen",
                        lambda req, timeout=None: FakeResp(lines))
    monkeypatch.setattr(streaming, "append_entry", lambda s, e: None)
    monkeypatch.setattr(streaming, "estimate_cny", lambda *a, **k: 0.001)


def test_sse_format():
    text = streaming.sse_format({"type": "delta", "text": "好"})
    assert text.startswith("event: delta\n")
    assert '"text": "好"' in text
    assert text.endswith("\n\n")


def test_stream_accumulates_text(monkeypatch):
    install(monkeypatch, sse(
        {"choices": [{"delta": {"content": "你"}}]},
        {"choices": [{"delta": {"content": "好"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ))
    evs = list(streaming.iter_chat_events(Settings(), [{"role": "user", "content": "hi"}],
                                          record_cost=False))
    deltas = [e["text"] for e in evs if e["type"] == "delta"]
    assert deltas == ["你", "好"]
    done = [e for e in evs if e["type"] == "done"][0]
    assert done["content"] == "你好"


def test_usage_comes_from_empty_choices_chunk(monkeypatch):
    """OpenAI 的挂法：单独一个 choices 为空的块带 usage。"""
    install(monkeypatch, sse(
        {"choices": [{"delta": {"content": "A"}}]},
        {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                                  "prompt_tokens_details": {"cached_tokens": 3}}},
    ))
    evs = list(streaming.iter_chat_events(Settings(), [{"role": "user", "content": "hi"}]))
    usage = [e for e in evs if e["type"] == "usage"][0]
    assert usage["usage"] == {"in": 11, "out": 7, "cached": 3}
    assert usage["cost_cny"] == 0.001
    assert usage["first_token_s"] is not None


def test_usage_attached_to_finish_reason_chunk(monkeypatch):
    """DeepSeek 的挂法（实测）：usage 挂在带 finish_reason 的那一块上，"""
    install(monkeypatch, sse(
        {"choices": [{"delta": {"content": "A"}}], "usage": None},
        {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 9,
                   "prompt_tokens_details": {"cached_tokens": 2}}},
    ))
    evs = list(streaming.iter_chat_events(Settings(), [{"role": "user", "content": "hi"}]))
    usage = [e for e in evs if e["type"] == "usage"][0]
    assert usage["usage"] == {"in": 5, "out": 9, "cached": 2}


def test_tool_arguments_are_reassembled(monkeypatch):
    """坑 2：arguments 被切成好几块。"""
    install(monkeypatch, sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "get_curr", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "ent_weather", "arguments": '{"city"'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ': "杭州"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ))
    evs = list(streaming.iter_chat_events(Settings(), [{"role": "user", "content": "hi"}],
                                          record_cost=False))
    done = [e for e in evs if e["type"] == "done"][0]
    assert len(done["tool_calls"]) == 1
    assert done["tool_calls"][0]["function"]["name"] == "get_current_weather"
    assert json.loads(done["tool_calls"][0]["function"]["arguments"]) == {"city": "杭州"}
    assert done["finish_reason"] == "tool_calls"


def test_stream_error_is_reported(monkeypatch):
    import io
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(b""))

    monkeypatch.setattr(streaming.urllib.request, "urlopen", boom)
    evs = list(streaming.iter_chat_events(Settings(), [{"role": "user", "content": "hi"}],
                                          record_cost=False))
    err = [e for e in evs if e["type"] == "error"][0]
    assert err["retryable"] is True
    assert "429" in err["message"]


def test_stream_endpoint(client, monkeypatch):
    def fake(s, messages, **kw):
        yield {"type": "delta", "text": "杭"}
        yield {"type": "delta", "text": "州"}
        yield {"type": "usage", "usage": {"in": 1, "out": 2, "cached": 0},
               "cost_cny": 0.001, "latency_s": 0.5}
        yield {"type": "done", "content": "杭州", "tool_calls": []}

    monkeypatch.setattr("app.main.iter_chat_events", fake)
    r = client.post("/chat/stream", json={"message": "说点什么"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "event: delta" in r.text
    assert "event: done" in r.text
