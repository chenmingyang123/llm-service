"""W4 块 E｜可观测层测试。"""
from __future__ import annotations

import json
import time

import pytest

from app.observe import (
    Tracer, MemorySink, JSONLSink, LangfuseExporter, langfuse_exporter,
    make_trace_hook, read_trace, merge_patches, trace_path,
    KIND_ROOT, KIND_STEP, KIND_TOOL, KIND_CONFIRM,
)


def make_tracer(**kw) -> Tracer:
    return Tracer(name="test.run", sink=MemorySink(), **kw)


def feed(hook, events: list[dict]) -> None:
    for e in events:
        hook(e)


def test_root_span_created_on_init():
    tr = make_tracer()
    assert tr.root.kind == KIND_ROOT
    assert tr.root.parent is None
    assert tr.spans == [tr.root]


def test_child_parent_and_duration():
    tr = make_tracer()
    child = tr.start_span("tool.x", kind=KIND_TOOL)
    assert child.parent == tr.root_id

    time.sleep(0.01)
    tr.end_span(child, output={"n": 1})

    assert child.ended
    assert child.dur_ms is not None and child.dur_ms >= 10
    assert child.output == {"n": 1}
    assert len(tr.sink.lines) == 1


def test_end_span_is_idempotent():
    """幂等是刻意的：让「with 里显式 end」和「退出时兜底 end」共存。"""
    tr = make_tracer()
    sp = tr.start_span("x")
    tr.end_span(sp, output="first")
    first_dur = sp.dur_ms
    tr.end_span(sp, output="second")
    assert sp.output == "first"
    assert sp.dur_ms == first_dur
    assert len(tr.sink.lines) == 1


def test_context_manager_closes_normally():
    tr = make_tracer()
    with tr.span("step", kind=KIND_STEP) as sp:
        pass
    assert sp.ended and sp.status == "ok" and sp.dur_ms is not None


def test_context_manager_records_error_and_reraises():
    """异常要记成 error，但**必须继续往外抛** —— 观测层绝不吞业务异常。"""
    tr = make_tracer()
    with pytest.raises(RuntimeError, match="boom"):
        with tr.span("tool.write", kind=KIND_TOOL) as sp:
            raise RuntimeError("boom")
    assert sp.status == "error"
    assert "boom" in (sp.error or "")
    assert len(tr.sink.lines) == 1


def test_finish_root_closes_leftovers():
    tr = make_tracer()
    orphan = tr.start_span("orphan", kind=KIND_TOOL)
    tr.finish_root(output={"answer": "x"})
    assert orphan.ended
    assert orphan.status == "abandoned"
    assert tr.root.ended and tr.root.output == {"answer": "x"}


def test_react_vocabulary_builds_step_tool_tree():
    tr = make_tracer()
    hook = make_trace_hook(tr, task="查天气")
    feed(hook, [
        {"type": "step_start", "step": 1},
        {"type": "thought", "step": 1, "thought": "先查天气"},
        {"type": "action", "step": 1, "tool": "get_current_weather",
         "args": '{"city":"上海"}'},
        {"type": "observation", "step": 1, "tool": "get_current_weather",
         "result": {"temp": 26}, "error": ""},
        {"type": "final", "step": 2, "answer": "上海 26 度"},
    ])

    steps = [s for s in tr.spans if s.kind == KIND_STEP]
    tools = [s for s in tr.spans if s.kind == KIND_TOOL]
    assert len(steps) == 1 and len(tools) == 1
    assert tools[0].parent == steps[0].span_id
    assert steps[0].parent == tr.root_id
    assert steps[0].meta["thought"] == "先查天气"
    assert tr.root.output == {"answer": "上海 26 度"}
    assert tr.root.status == "ok"


def test_tool_loop_vocabulary_multiple_tools_per_turn():
    tr = make_tracer()
    hook = make_trace_hook(tr)
    feed(hook, [
        {"type": "tool_start", "turn": 1, "tools": ["get_current_weather",
                                                    "query_orders_stats"]},
        {"type": "tool_done", "turn": 1, "tool": "get_current_weather",
         "result": {"temp": 26}, "error": ""},
        {"type": "tool_done", "turn": 1, "tool": "query_orders_stats",
         "result": {"n": 128}, "error": ""},
        {"type": "final", "turn": 2, "answer": "好了"},
    ])
    tools = [s for s in tr.spans if s.kind == KIND_TOOL]
    assert [s.name for s in tools] == ["tool.get_current_weather",
                                       "tool.query_orders_stats"]
    assert all(s.parent == tr.root_id for s in tools)
    assert all(s.ended for s in tools)


def test_same_tool_twice_keeps_two_spans():
    """同名工具连调两次不能被 dict 覆盖掉 —— FIFO 队列是为此存在的。"""
    tr = make_tracer()
    hook = make_trace_hook(tr)
    feed(hook, [
        {"type": "tool_start", "turn": 1, "tools": ["get_current_weather"]},
        {"type": "tool_start", "turn": 1, "tools": ["get_current_weather"]},
        {"type": "tool_done", "turn": 1, "tool": "get_current_weather",
         "result": 1, "error": ""},
        {"type": "tool_done", "turn": 1, "tool": "get_current_weather",
         "result": 2, "error": ""},
        {"type": "final", "turn": 2, "answer": "x"},
    ])
    tools = [s for s in tr.spans if s.kind == KIND_TOOL]
    assert len(tools) == 2
    assert [s.output for s in tools] == [1, 2]


def test_error_is_separate_field_not_baked_into_output():
    tr = make_tracer()
    hook = make_trace_hook(tr)
    feed(hook, [
        {"type": "action", "step": 1, "tool": "get_order_detail", "args": "{}"},
        {"type": "observation", "step": 1, "tool": "get_order_detail",
         "result": None, "error": "Field required（order_id）"},
        {"type": "final", "step": 2, "answer": "x"},
    ])
    rec = tr.sink.lines[0]
    assert rec["error"] == "Field required（order_id）"
    assert rec["status"] == "error"
    assert rec["output"] is None


def test_stop_marks_root_as_error():
    tr = make_tracer()
    hook = make_trace_hook(tr)
    feed(hook, [{"type": "stop", "step": 6, "reason": "max_steps_reached"}])
    assert tr.root.status == "error"
    assert "max_steps_reached" in (tr.root.error or "")
    assert tr.root.meta["stop_reason"] == "max_steps_reached"


def test_step_closed_when_next_step_starts():
    """ReAct 事件流里没有 step_end，漏了这条推断会让每一步的耗时都撑到全程结束。"""
    tr = make_tracer()
    hook = make_trace_hook(tr)
    feed(hook, [
        {"type": "step_start", "step": 1},
        {"type": "step_start", "step": 2},
        {"type": "final", "step": 2, "answer": "x"},
    ])
    steps = [s for s in tr.spans if s.kind == KIND_STEP]
    assert len(steps) == 2
    assert all(s.ended for s in steps)
    assert all(s.status == "ok" for s in steps)
    assert steps[0].dur_ms is not None and steps[0].dur_ms < (tr.root.dur_ms or 1e9)


def test_confirm_span_records_human_wait():
    tr = make_tracer()
    hook = make_trace_hook(tr)
    feed(hook, [
        {"type": "tool_start", "turn": 1, "tools": ["create_ticket"]},
        {"type": "need_confirm", "turn": 1, "confirm_id": "c1",
         "tool": "create_ticket", "args": "{}", "reason": "写操作"},
        {"type": "confirm_result", "turn": 1, "confirm_id": "c1",
         "approved": False, "tool": "create_ticket"},
        {"type": "tool_done", "turn": 1, "tool": "create_ticket",
         "result": None, "error": "未获人工确认"},
        {"type": "final", "turn": 2, "answer": "x"},
    ])
    conf = [s for s in tr.spans if s.kind == KIND_CONFIRM]
    assert len(conf) == 1 and conf[0].ended
    assert conf[0].status == "ok"
    assert conf[0].meta["approved"] is False


def test_jsonl_roundtrip(tmp_path):
    tr = Tracer(name="io.test", trace_dir=str(tmp_path))
    hook = make_trace_hook(tr)
    feed(hook, [
        {"type": "step_start", "step": 1},
        {"type": "action", "step": 1, "tool": "t", "args": "{}"},
        {"type": "observation", "step": 1, "tool": "t", "result": 1, "error": ""},
        {"type": "final", "step": 2, "answer": "x"},
    ])

    p = trace_path(tr.trace_id, str(tmp_path))
    recs = read_trace(p)
    assert len(recs) == len(tr.spans)
    assert all(r["dur_ms"] is not None for r in recs)
    assert all(r["t_epoch"] is not None for r in recs)


def test_summarize_breaks_down_cost_and_latency():
    tr = make_tracer()
    with tr.span("step.1", kind=KIND_STEP):
        time.sleep(0.02)
        with tr.span("tool.a", kind=KIND_TOOL):
            time.sleep(0.01)
    tr.finish_root(cost_cny=0.00123, usage={"in": 100, "out": 20})

    s = tr.summarize()
    assert s["ok"] is True
    assert s["span_count"] == 3
    assert s["cost_cny"] == 0.00123
    assert "step" in s["breakdown"] and "tool" in s["breakdown"]
    assert s["llm_ms_derived"] > 0
    assert s["llm_ms_derived"] < s["breakdown"]["step"]["ms"]


def test_langfuse_degrades_gracefully():
    """没装 / 没配 key 时，观测层整体行为不变，只是少一个镜像。"""
    exp = LangfuseExporter()
    if not exp.enabled:
        assert exp.reason
        assert langfuse_exporter() is None
        exp.export({"name": "x", "trace_id": "y"})
    else:
        assert langfuse_exporter() is not None


def test_tracer_works_without_any_exporter():
    tr = make_tracer()
    hook = make_trace_hook(tr)
    feed(hook, [{"type": "final", "turn": 1, "answer": "x"}])
    assert tr.root.ended
    assert len(tr.sink.lines) == 1


def test_patch_root_appends_and_merges():
    """根 span 结束后补汇总字段：追加一条 patch，查看时按 span_id 合并。"""
    tr = make_tracer()
    hook = make_trace_hook(tr)
    feed(hook, [{"type": "final", "turn": 1, "answer": "x"}])
    assert tr.root.ended

    tr.patch_root(cost_cny=0.001, usage={"in": 10})
    assert len(tr.sink.lines) == 2

    merged = merge_patches(list(tr.sink.lines))
    assert len(merged) == 1
    assert merged[0]["meta"]["cost_cny"] == 0.001
    assert merged[0]["meta"]["usage"] == {"in": 10}
    assert merged[0]["status"] == "ok"


def test_patch_root_survives_jsonl(tmp_path):
    tr = Tracer(name="patch.io", trace_dir=str(tmp_path))
    hook = make_trace_hook(tr)
    feed(hook, [{"type": "final", "turn": 1, "answer": "x"}])
    tr.patch_root(cost_cny=0.002, latency_s=1.5)

    recs = read_trace(trace_path(tr.trace_id, str(tmp_path)))
    root = [r for r in recs if r["kind"] == KIND_ROOT][0]
    assert root["meta"]["cost_cny"] == 0.002
    assert len(recs) == 1


def test_unknown_event_is_kept_not_dropped():
    """未知事件宁可多存也别丢 —— 静默丢弃等于给自己埋排障盲区。"""
    tr = make_tracer()
    hook = make_trace_hook(tr)
    feed(hook, [
        {"type": "step_start", "step": 1},
        {"type": "some_future_event", "foo": 1},
        {"type": "final", "step": 1, "answer": "x"},
    ])
    steps = [s for s in tr.spans if s.kind == KIND_STEP]
    assert steps[0].meta["extra_events"] == [{"type": "some_future_event", "foo": 1}]


def test_endpoint_writes_trace_only_when_asked(tmp_path, monkeypatch):
    """接 trace **不能**改 Agent 逻辑 —— 这条测试钉死的是「钩子挂对了」。"""
    import os
    import app.main as m
    import app.observe as observe
    from fastapi.testclient import TestClient

    monkeypatch.setattr(observe, "TRACE_DIR", str(tmp_path))

    seen = {"hook": None}

    def fake_run_react(s, task, **kw):
        seen["hook"] = kw.get("on_event")
        if seen["hook"]:
            seen["hook"]({"type": "step_start", "step": 1})
            seen["hook"]({"type": "final", "step": 1, "answer": "ok"})
        return {"ok": True, "answer": "ok", "trace": [], "steps": 1,
                "tool_calls_made": 0, "stop_reason": "final_answer",
                "provider": "p", "model": "m",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "cached_tokens": 0},
                "cost_cny": 0.0001, "latency_s": 0.1}

    monkeypatch.setattr(m, "run_react", fake_run_react)
    client = TestClient(m.app)

    r0 = client.post("/agent/react", json={"question": "q"})
    assert r0.status_code == 200
    assert r0.json()["trace_id"] == ""
    assert seen["hook"] is None

    r1 = client.post("/agent/react", json={"question": "q", "trace": True})
    assert r1.status_code == 200
    tid = r1.json()["trace_id"]
    assert tid
    p = os.path.join(str(tmp_path), tid + ".jsonl")
    assert os.path.exists(p)

    recs = read_trace(p)
    root = [x for x in recs if x["kind"] == KIND_ROOT][0]
    assert root["meta"]["cost_cny"] == 0.0001
    assert any(x["kind"] == KIND_STEP for x in recs)


def test_endpoint_writes_trace_only_when_asked(tmp_path, monkeypatch):
    """接 trace **不能**改 Agent 逻辑 —— 这条测试钉死的是「钩子挂对了」。"""
    import os
    import app.main as m
    import app.observe as observe
    from fastapi.testclient import TestClient

    monkeypatch.setattr(observe, "TRACE_DIR", str(tmp_path))

    seen = {"hook": None}

    def fake_run_react(s, task, **kw):
        seen["hook"] = kw.get("on_event")
        if seen["hook"]:
            seen["hook"]({"type": "step_start", "step": 1})
            seen["hook"]({"type": "final", "step": 1, "answer": "ok"})
        return {"ok": True, "answer": "ok", "trace": [], "steps": 1,
                "tool_calls_made": 0, "stop_reason": "final_answer",
                "provider": "p", "model": "m",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "cached_tokens": 0},
                "cost_cny": 0.0001, "latency_s": 0.1}

    monkeypatch.setattr(m, "run_react", fake_run_react)
    client = TestClient(m.app)

    r0 = client.post("/agent/react", json={"question": "q"})
    assert r0.status_code == 200
    assert r0.json()["trace_id"] == ""
    assert seen["hook"] is None

    r1 = client.post("/agent/react", json={"question": "q", "trace": True})
    assert r1.status_code == 200
    tid = r1.json()["trace_id"]
    assert tid
    p = os.path.join(str(tmp_path), tid + ".jsonl")
    assert os.path.exists(p)

    recs = read_trace(p)
    root = [x for x in recs if x["kind"] == KIND_ROOT][0]
    assert root["meta"]["cost_cny"] == 0.0001
    assert any(x["kind"] == KIND_STEP for x in recs)
