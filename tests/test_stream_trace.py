"""W4 块 E｜/agent/stream 的 trace 接入测试。"""
from __future__ import annotations

import json

import pytest

from app.observe import read_trace, KIND_ROOT, KIND_TOOL, KIND_CONFIRM


def _fake_tool_loop(events: list[dict], result: dict):
    """假的 run_tool_loop：按脚本发事件然后返回。"""
    def _run(settings, question, **kw):
        emit = kw.get("on_event")
        for ev in events:
            if emit is not None:
                emit(ev)
        return result
    return _run


def _sse_events(text: str) -> list[dict]:
    """把 SSE 响应体拆成事件字典。"""
    out = []
    for line in text.splitlines():
        if line.startswith("data:"):
            out.append(json.loads(line[5:].strip()))
    return out


def _result(**over) -> dict:
    base = {"ok": True, "answer": "答案", "turns": 1, "tool_calls_made": 1,
            "trace": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "cached_tokens": 0},
            "cost_cny": 0.0002, "latency_s": 0.3}
    base.update(over)
    return base


def _prep(monkeypatch, tmp_path) -> tuple:
    """装好假工具循环与临时 trace 目录，返回 (TestClient, observe 模块)。"""
    import app.main as m
    import app.observe as observe
    from fastapi.testclient import TestClient

    monkeypatch.setattr(observe, "TRACE_DIR", str(tmp_path))
    return TestClient(m.app), observe, m


def test_stream_trace_off_leaves_no_trace(tmp_path, monkeypatch):
    """默认关：不留文件、不带响应头 —— 不是所有请求都值得落一条 trace。"""
    client, observe, m = _prep(monkeypatch, tmp_path)
    monkeypatch.setattr(
        m, "run_tool_loop",
        _fake_tool_loop([{"type": "final", "answer": "ok"}], _result()))

    r = client.post("/agent/stream", json={"question": "q"})
    assert r.status_code == 200
    assert "X-Trace-Id" not in r.headers
    assert list(tmp_path.glob("*.jsonl")) == []


def test_stream_trace_on_exposes_id_on_both_channels(tmp_path, monkeypatch):
    """trace_id 必须头里和流里都有。"""
    client, observe, m = _prep(monkeypatch, tmp_path)
    events = [
        {"type": "tool_start", "turn": 1, "tools": ["search_docs"]},
        {"type": "tool_done", "tool": "search_docs", "result": "命中 3 条"},
        {"type": "final", "answer": "答案"},
    ]
    monkeypatch.setattr(m, "run_tool_loop", _fake_tool_loop(events, _result()))

    r = client.post("/agent/stream", json={"question": "q", "trace": True})
    assert r.status_code == 200

    tid = r.headers.get("X-Trace-Id")
    assert tid, "响应头必须带 trace_id：流开始前就能拿到，客户端才好预先准备"

    evs = _sse_events(r.text)
    assert evs[0]["type"] == "trace_started"
    assert evs[0]["trace_id"] == tid

    summary = [e for e in evs if e["type"] == "summary"][0]
    assert summary["trace_id"] == tid

    recs = read_trace(str(tmp_path / f"{tid}.jsonl"))
    assert any(x["kind"] == KIND_TOOL for x in recs), "工具调用要进 span 树"
    root = [x for x in recs if x["kind"] == KIND_ROOT][0]
    assert root["meta"]["cost_cny"] == 0.0002


def test_stream_confirm_becomes_its_own_span(tmp_path, monkeypatch):
    """不单独计时，你会把「人没点」误判成「系统慢」。"""
    client, observe, m = _prep(monkeypatch, tmp_path)
    events = [
        {"type": "tool_start", "turn": 1, "tools": ["create_ticket"]},
        {"type": "need_confirm", "tool": "create_ticket", "confirm_id": "c1",
         "args": {"title": "t"}, "request_id": "r1", "reason": "写操作"},
        {"type": "confirm_result", "confirm_id": "c1", "approved": True},
        {"type": "tool_done", "tool": "create_ticket", "result": "已创建"},
        {"type": "final", "answer": "ok"},
    ]
    monkeypatch.setattr(m, "run_tool_loop", _fake_tool_loop(events, _result()))

    r = client.post("/agent/stream", json={"question": "q", "trace": True})
    tid = r.headers["X-Trace-Id"]
    recs = read_trace(str(tmp_path / f"{tid}.jsonl"))

    conf = [x for x in recs if x["kind"] == KIND_CONFIRM]
    assert len(conf) == 1
    assert conf[0]["meta"]["approved"] is True
    assert not conf[0].get("error")


def test_stream_confirm_rejected_is_not_an_error(tmp_path, monkeypatch):
    """被拒绝要留痕，但不能记成 error —— 否则错误率会被人为因素污染。"""
    client, observe, m = _prep(monkeypatch, tmp_path)
    events = [
        {"type": "need_confirm", "tool": "create_ticket", "confirm_id": "c1",
         "args": {}, "request_id": "r1", "reason": "写操作"},
        {"type": "confirm_result", "confirm_id": "c1", "approved": False},
        {"type": "final", "answer": "已取消"},
    ]
    monkeypatch.setattr(m, "run_tool_loop", _fake_tool_loop(events, _result()))

    r = client.post("/agent/stream", json={"question": "q", "trace": True})
    recs = read_trace(str(tmp_path / f"{r.headers['X-Trace-Id']}.jsonl"))

    conf = [x for x in recs if x["kind"] == KIND_CONFIRM][0]
    assert conf["meta"]["approved"] is False
    assert not conf.get("error"), "人为拒绝不能进 error 字段"


def test_stream_exception_still_closes_root_span(tmp_path, monkeypatch):
    """异常退出没有 final 事件，根 span 会一直悬着 —— 必须兜底关掉。"""
    client, observe, m = _prep(monkeypatch, tmp_path)

    def boom(settings, question, **kw):
        raise RuntimeError("后端炸了")

    monkeypatch.setattr(m, "run_tool_loop", boom)

    r = client.post("/agent/stream", json={"question": "q", "trace": True})
    tid = r.headers["X-Trace-Id"]

    evs = _sse_events(r.text)
    assert any(e["type"] == "error" for e in evs)

    recs = read_trace(str(tmp_path / f"{tid}.jsonl"))
    root = [x for x in recs if x["kind"] == KIND_ROOT][0]
    assert root["dur_ms"] is not None, "根 span 必须是闭合的，不能悬空"
    assert "后端炸了" in (root.get("error") or "")


def test_stream_trace_does_not_change_agent_args(tmp_path, monkeypatch):
    """块 E 的立身之本：接 trace 不动 Agent 一行代码。"""
    client, observe, m = _prep(monkeypatch, tmp_path)
    seen = {}

    def spy(settings, question, **kw):
        seen.update(kw)
        return _result()

    monkeypatch.setattr(m, "run_tool_loop", spy)

    client.post("/agent/stream", json={"question": "q", "trace": True})
    assert callable(seen.get("on_event")), "on_event 必须是真钩子，不能是 None"
    assert seen.get("allowed_tools")
    assert isinstance(seen.get("confirmed"), set)
    assert seen.get("confirm_registry") is not None
