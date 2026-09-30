"""工具调用的测试。"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import agent, tools
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def tc(name: str, args: dict | str, call_id: str = "call_1") -> dict:
    """造一个 tool_call。arguments 必须是字符串 —— 真实响应就是字符串。"""
    raw = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": raw}}


def fake_result(content: str = "", tool_calls: list[dict] | None = None, **kw) -> dict:
    base = {
        "content": content,
        "tool_calls": tool_calls or [],
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "reasoning_content": "",
        "usage": {"in": 20, "out": 30, "cached": 0},
        "latency_s": 0.5,
        "provider": "deepseek",
        "model": "deepseek-flash",
        "cost_cny": 0.001,
    }
    base.update(kw)
    return base


def _settings():
    from app.config import Settings
    return Settings()


def test_dispatch_ok():
    result, err = tools.dispatch("get_current_weather", '{"city": "杭州"}')
    assert err == ""
    assert result["city"] == "杭州"
    assert result["unit"] == "C"


def test_dispatch_bad_json():
    """参数不是合法 JSON —— 官方明确说模型会这么干。"""
    result, err = tools.dispatch("get_current_weather", '{"city": 杭州')
    assert result is None
    assert "合法 JSON" in err


def test_dispatch_unknown_tool():
    result, err = tools.dispatch("delete_everything", "{}")
    assert result is None
    assert "可用工具" in err


def test_dispatch_bad_param():
    """unit 不在枚举里，必须被 Pydantic 挡住而不是传进函数。"""
    result, err = tools.dispatch("get_current_weather", '{"city":"杭州","unit":"kelvin"}')
    assert result is None
    assert "参数不合法" in err


def test_dispatch_missing_required():
    result, err = tools.dispatch("get_order_detail", "{}")
    assert result is None
    assert "参数不合法" in err


def test_stats_refuses_write():
    """最小权限：模型不该有写库能力。"""
    a = tools.StatsArgs(sql="DELETE FROM orders")
    out = tools.query_orders_stats(a)
    assert "error" in out


def test_stats_runs_select():
    a = tools.StatsArgs(sql="SELECT city, COUNT(*) AS n FROM orders GROUP BY city")
    out = tools.query_orders_stats(a)
    assert out["count"] >= 1


def test_order_detail_unknown_id():
    out = tools.get_order_detail(tools.OrderArgs(order_id="SO99999999"))
    assert "error" in out
    assert "库中现有" in out["error"]


def test_weather_unknown_city():
    out = tools.get_current_weather(tools.WeatherArgs(city="火星"))
    assert "error" in out


def test_loop_single_tool(monkeypatch):
    """第一轮要调工具，第二轮给最终答案。"""
    seq = [fake_result(tool_calls=[tc("get_current_weather", {"city": "杭州"})]),
           fake_result("杭州现在 27 度，多云。")]
    it = iter(seq)
    monkeypatch.setattr(agent, "call_chat", lambda *a, **k: next(it))

    out = agent.run_tool_loop(_settings(), "杭州现在天气怎么样？")

    assert out["ok"] is True
    assert out["turns"] == 2
    assert out["tool_calls_made"] == 1
    assert out["trace"][0]["tool"] == "get_current_weather"
    assert out["trace"][0]["error"] == ""
    assert "27" in out["answer"]


def test_loop_no_tool(monkeypatch):
    """不需要工具时应该直接回答，一轮结束。"""
    monkeypatch.setattr(agent, "call_chat", lambda *a, **k: fake_result("你好！"))
    out = agent.run_tool_loop(_settings(), "你好")
    assert out["turns"] == 1
    assert out["trace"] == []
    assert out["stop_reason"] == "no_tool_calls"


def test_loop_recovers_from_bad_tool(monkeypatch):
    """模型调了一个不存在的工具 —— 应该收到错误消息后改对，而不是崩掉。"""
    seq = [fake_result(tool_calls=[tc("get_weather_forecast", {"city": "杭州"})]),
           fake_result(tool_calls=[tc("get_current_weather", {"city": "杭州"})]),
           fake_result("杭州现在 27 度。")]
    it = iter(seq)
    monkeypatch.setattr(agent, "call_chat", lambda *a, **k: next(it))

    out = agent.run_tool_loop(_settings(), "杭州天气？")

    assert out["ok"] is True
    assert out["trace"][0]["error"]
    assert out["trace"][1]["error"] == ""


def test_loop_fuses_at_max_turns(monkeypatch):
    """模型一直调工具 —— 必须熔断，不能无限烧钱。"""
    monkeypatch.setattr(
        agent, "call_chat",
        lambda *a, **k: fake_result(tool_calls=[tc("get_current_weather", {"city": "杭州"})]),
    )
    out = agent.run_tool_loop(_settings(), "杭州天气？", max_turns=3)

    assert out["ok"] is False
    assert out["turns"] == 3
    assert out["stop_reason"] == "max_turns_reached"
    assert out["answer"] == ""


def test_tool_message_order_and_ids(monkeypatch):
    """红线：tool 消息必须紧跟带 tool_calls 的 assistant 消息，且 id 一一对应。"""
    seen = []

    def fake(s, messages, **k):
        seen.append([dict(m) for m in messages])
        if len(seen) == 1:
            return fake_result(tool_calls=[tc("get_current_weather", {"city": "北京"},
                                              call_id="call_abc")])
        return fake_result("北京 22 度。")

    monkeypatch.setattr(agent, "call_chat", fake)
    agent.run_tool_loop(_settings(), "北京天气？")

    second_round = seen[1]
    assistant = second_round[-2]
    tool_msg = second_round[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_abc"
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_abc"


def test_thinking_rejected_with_tools():
    """调工具时开思考会被服务端 400，与其等它报错不如在本地先挡住。"""
    from app.llm import LLMError, call_chat
    with pytest.raises(LLMError, match="thinking"):
        call_chat(_settings(), [{"role": "user", "content": "x"}],
                  tools=tools.TOOL_SCHEMAS, tool_choice="auto", thinking=True)


def test_agent_endpoint(client, monkeypatch):
    seq = [fake_result(tool_calls=[tc("get_order_detail", {"order_id": "SO20260002"})]),
           fake_result("订单 SO20260002 目前运输中。")]
    it = iter(seq)
    monkeypatch.setattr(agent, "call_chat", lambda *a, **k: next(it))

    r = client.post("/agent", json={"question": "SO20260002 到哪了？"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["trace"][0]["tool"] == "get_order_detail"
    assert body["cost_cny"] > 0


def test_agent_tools_endpoint(client):
    r = client.get("/agent/tools")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(tools.TOOL_SCHEMAS)
    assert body["count"] >= 3
    assert "search_docs" in [t["function"]["name"] for t in body["tools"]]
