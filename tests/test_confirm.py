"""人工确认（human-in-the-loop）链路的测试 —— W4 块 C 的确认点接流式。"""
from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import agent, tools
from app.confirm import ConfirmationRegistry
from app.main import CONFIRMATIONS, app
from app.tools import TOOL_SCHEMAS


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    """每个测试：干净幂等表 + 临时审计文件 + 空确认表。"""
    tools.reset_actions_for_test()
    monkeypatch.setattr(tools, "_ACTIONS_PATH", str(tmp_path / "actions.jsonl"))
    CONFIRMATIONS.clear()
    yield
    tools.reset_actions_for_test()
    CONFIRMATIONS.clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def tc(name: str, args: dict | str, call_id: str = "call_1") -> dict:
    raw = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": raw}}


def fake_result(content: str = "", tool_calls: list[dict] | None = None, **kw) -> dict:
    base = {
        "content": content, "tool_calls": tool_calls or [],
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "reasoning_content": "",
        "usage": {"in": 20, "out": 30, "cached": 0},
        "latency_s": 0.5, "provider": "deepseek", "model": "deepseek-flash",
        "cost_cny": 0.001,
    }
    base.update(kw)
    return base


def _settings():
    from app.config import Settings
    return Settings()


class FakeRegistry:
    """假登记表：不阻塞，直接给决定 —— 用来测链路，不测等待本身。"""

    def __init__(self, decision: bool = True) -> None:
        self.decision = decision
        self.opened: list[dict] = []

    def open(self, *, tool, args, request_id):  # noqa: A002
        cid = f"cid-{len(self.opened) + 1}"
        self.opened.append({"confirm_id": cid, "tool": tool, "args": args,
                            "request_id": request_id})
        return cid

    def wait(self, cid, timeout_s=None):
        return self.decision


TICKET = {"title": "登录失败", "body": "用户无法登录系统", "request_id": "req-cf-1"}


def test_registry_open_then_approve():
    reg = ConfirmationRegistry(timeout_s=5)
    assert reg.pending_count() == 0
    cid = reg.open(tool="create_ticket", args="{}", request_id="r1")
    assert reg.pending_count() == 1 and cid
    assert reg.resolve(cid, True) is True
    assert reg.wait(cid) is True
    assert reg.pending_count() == 0


def test_registry_reject():
    reg = ConfirmationRegistry(timeout_s=5)
    cid = reg.open(tool="create_ticket", args="{}", request_id="r1")
    reg.resolve(cid, False)
    assert reg.wait(cid) is False


def test_registry_timeout_is_reject():
    """超时按拒绝处理（fail-closed）—— 安全默认值必须是拒绝，不是放行。"""
    reg = ConfirmationRegistry(timeout_s=0.2)
    cid = reg.open(tool="create_ticket", args="{}", request_id="r1")
    t0 = time.time()
    assert reg.wait(cid) is False
    assert time.time() - t0 < 2


def test_registry_unknown_ids():
    reg = ConfirmationRegistry()
    assert reg.resolve("nope", True) is False
    assert reg.wait("nope", timeout_s=0.1) is False


def test_registry_clear_wakes_as_reject():
    reg = ConfirmationRegistry(timeout_s=5)
    cid = reg.open(tool="t", args="{}", request_id="r")
    box: dict = {}

    def _w():
        box["ok"] = reg.wait(cid)

    th = threading.Thread(target=_w)
    th.start()
    time.sleep(0.1)
    reg.clear()
    th.join(timeout=2)
    assert box["ok"] is False and not th.is_alive()


def test_guarded_approve_executes_and_emits(monkeypatch):
    it = iter([fake_result(tool_calls=[tc("create_ticket", TICKET)]),
               fake_result(content="已建单")])
    monkeypatch.setattr(agent, "call_chat", lambda *a, **k: next(it))
    reg = FakeRegistry(decision=True)
    events: list[dict] = []

    out = agent.run_tool_loop(_settings(), "帮我建单", tools=TOOL_SCHEMAS,
                              confirmed=set(), confirm_registry=reg,
                              on_event=events.append, max_turns=3)

    assert out["ok"] is True
    hit = [t for t in out["trace"] if t["tool"] == "create_ticket"]
    assert hit and hit[0]["error"] == "" and hit[0]["result"]
    assert hit[0]["result"]["ticket_id"].startswith("TK")
    kinds = [e["type"] for e in events]
    assert "need_confirm" in kinds
    assert "confirm_result" in kinds
    assert any(e["type"] == "confirm_result" and e["approved"] for e in events)
    assert reg.opened[0]["request_id"] == "req-cf-1"


def test_guarded_reject_does_not_execute(monkeypatch, tmp_path):
    it = iter([fake_result(tool_calls=[tc("create_ticket", TICKET)]),
               fake_result(content="好的，不建了")])
    monkeypatch.setattr(agent, "call_chat", lambda *a, **k: next(it))
    reg = FakeRegistry(decision=False)
    events: list[dict] = []

    out = agent.run_tool_loop(_settings(), "帮我建单", tools=TOOL_SCHEMAS,
                              confirmed=set(), confirm_registry=reg,
                              on_event=events.append, max_turns=3)

    hit = [t for t in out["trace"] if t["tool"] == "create_ticket"]
    assert hit and hit[0]["result"] is None
    assert "未获人工确认" in hit[0]["error"]
    assert any(e["type"] == "confirm_result" and not e["approved"] for e in events)
    p = tmp_path / "actions.jsonl"
    assert (not p.exists()) or p.read_text(encoding="utf-8").strip() == ""


def test_write_blocked_without_registry(monkeypatch):
    """只给 confirmed 空集、不给登记表：写动作被闸门拦下，且不会有 need_confirm。"""
    it = iter([fake_result(tool_calls=[tc("create_ticket", TICKET)]),
               fake_result(content="无法执行")])
    monkeypatch.setattr(agent, "call_chat", lambda *a, **k: next(it))
    events: list[dict] = []

    out = agent.run_tool_loop(_settings(), "帮我建单", tools=TOOL_SCHEMAS,
                              confirmed=set(), on_event=events.append, max_turns=3)

    hit = [t for t in out["trace"] if t["tool"] == "create_ticket"]
    assert hit and hit[0]["result"] is None
    assert "尚未获得人工确认" in hit[0]["error"]
    assert not any(e["type"] == "need_confirm" for e in events)


def test_read_tool_never_asks(monkeypatch):
    """读工具不该弹确认 —— 确认是最高成本的动作，不能滥用。"""
    it = iter([fake_result(tool_calls=[tc("get_current_weather", {"city": "杭州"})]),
               fake_result(content="晴")])
    monkeypatch.setattr(agent, "call_chat", lambda *a, **k: next(it))
    reg = FakeRegistry(decision=True)
    events: list[dict] = []

    out = agent.run_tool_loop(_settings(), "杭州天气", tools=TOOL_SCHEMAS,
                              confirmed=set(), confirm_registry=reg,
                              on_event=events.append, max_turns=3)

    assert not any(e["type"] == "need_confirm" for e in events)
    assert reg.opened == []
    hit = [t for t in out["trace"] if t["tool"] == "get_current_weather"]
    assert hit and hit[0]["error"] == ""


def test_real_registry_blocks_then_resumes(monkeypatch):
    """用真实登记表 + 延迟 0.3s 的外部确认，证明 worker 确实暂停过。"""
    it = iter([fake_result(tool_calls=[tc("create_ticket", TICKET)]),
               fake_result(content="已完成")])
    monkeypatch.setattr(agent, "call_chat", lambda *a, **k: next(it))
    reg = ConfirmationRegistry(timeout_s=5)

    def on_event(ev):
        if ev["type"] == "need_confirm":
            def later():
                time.sleep(0.3)
                reg.resolve(ev["confirm_id"], True)
            threading.Thread(target=later, daemon=True).start()

    t0 = time.time()
    out = agent.run_tool_loop(_settings(), "帮我建单", tools=TOOL_SCHEMAS,
                              confirmed=set(), confirm_registry=reg,
                              on_event=on_event, max_turns=3)
    elapsed = time.time() - t0

    assert elapsed >= 0.3, "worker 应真的等待了外部确认"
    hit = [t for t in out["trace"] if t["tool"] == "create_ticket"]
    assert hit and hit[0]["result"] and hit[0]["result"]["ticket_id"].startswith("TK")


def test_confirm_endpoint_unknown_id(client):
    r = client.post("/agent/confirm", json={"confirm_id": "nosuchid", "approve": True})
    assert r.status_code == 404
    assert "不存在或已过期" in r.json()["detail"]


def test_confirm_endpoint_resolves(client):
    cid = CONFIRMATIONS.open(tool="create_ticket", args="{}", request_id="r1")
    box: dict = {}

    def _w():
        box["ok"] = CONFIRMATIONS.wait(cid, timeout_s=5)

    th = threading.Thread(target=_w)
    th.start()
    time.sleep(0.1)
    r = client.post("/agent/confirm", json={"confirm_id": cid, "approve": True})
    th.join(timeout=5)

    assert r.status_code == 200
    assert r.json()["approved"] is True
    assert box["ok"] is True


def test_confirm_endpoint_reject(client):
    cid = CONFIRMATIONS.open(tool="create_ticket", args="{}", request_id="r1")

    def _w():
        pass

    th = threading.Thread(target=_w)
    r = client.post("/agent/confirm", json={"confirm_id": cid, "approve": False})
    assert r.status_code == 200 and r.json()["approved"] is False
