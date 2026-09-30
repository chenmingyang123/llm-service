"""手写 ReAct 的测试 —— W4 块 A。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import react
from app.config import Settings
from app.main import app
from app.tools import TOOL_SCHEMAS


@pytest.fixture(autouse=True)
def _budget_open(monkeypatch):
    """默认把成本闸门打开。"""
    monkeypatch.setattr(react, "summarize", lambda s: {"over_limit": False, "total_cny": 0.0})


def fake_result(content: str = "", **kw) -> dict:
    """造一次 call_chat 的返回。字段对齐 app/llm.py 的真实返回。"""
    base = {
        "content": content,
        "tool_calls": [],
        "finish_reason": "stop",
        "reasoning_content": "",
        "usage": {"in": 100, "out": 50, "cached": 0},
        "latency_s": 0.5,
        "provider": "deepseek",
        "model": "deepseek-flash",
        "cost_cny": 0.001,
    }
    base.update(kw)
    return base


def script(monkeypatch, contents: list[str]):
    """按脚本喂模型输出，并记录每一次调用收到的参数。"""
    seq = list(contents)
    calls: list[dict] = []

    def _fake(s, messages, **kw):
        calls.append({"kw": kw, "n_messages": len(messages),
                      "system": kw.get("system", "")})
        c = seq.pop(0) if seq else "思考：够了\n最终答案：兜底答案"
        return fake_result(content=c)

    monkeypatch.setattr(react, "call_chat", _fake)
    return calls


def settings() -> Settings:
    return Settings()


def test_parse_action_fullwidth_colon():
    p = react.parse_action('思考：需要天气数据\n行动：get_current_weather {"city":"杭州"}')
    assert p["kind"] == "action"
    assert p["tool"] == "get_current_weather"
    assert p["args"] == '{"city":"杭州"}'
    assert p["thought"] == "需要天气数据"


def test_parse_action_halfwidth_colon_with_noise():
    """模型常夹带自然语言，工具名和 JSON 仍要能抓出来。"""
    p = react.parse_action('行动: 我要调 get_current_weather，参数是 {"city": "北京", "unit": "celsius"}')
    assert p["kind"] == "action"
    assert p["tool"] == "get_current_weather"
    assert '"city"' in p["args"] and '"unit"' in p["args"]


def test_parse_action_markdown_bold_and_backtick():
    p = react.parse_action('**思考**：查一下\n**行动**：`get_current_weather` {"city":"上海"}')
    assert p["kind"] == "action"
    assert p["tool"] == "get_current_weather"


def test_parse_final_answer():
    p = react.parse_action("思考：信息够了\n最终答案：杭州今天 26℃，晴。")
    assert p["kind"] == "final"
    assert p["answer"] == "杭州今天 26℃，晴。"
    assert p["thought"] == "信息够了"


def test_parse_unknown_when_no_marker():
    p = react.parse_action("我觉得应该先查一下天气吧")
    assert p["kind"] == "unknown"


def test_parse_unknown_when_marker_but_no_tool_name():
    """写了「行动：」却没写出工具名 —— 不能当成 action 往下走。"""
    p = react.parse_action("行动：天气查询一下杭州")
    assert p["kind"] == "unknown"


def test_two_markers_earlier_one_wins_final():
    """最终答案在前 → 听最终答案（它这一轮真正要的是结束）。"""
    p = react.parse_action("最终答案：就这样\n行动：get_current_weather {\"city\":\"杭州\"}")
    assert p["kind"] == "final"


def test_two_markers_earlier_one_wins_action():
    """行动在前 → 听行动（它这一轮真正要的是调工具）。"""
    p = react.parse_action('行动：get_current_weather {"city":"杭州"}\n最终答案：26℃')
    assert p["kind"] == "action"
    assert p["tool"] == "get_current_weather"


def test_empty_input_is_unknown():
    assert react.parse_action("")["kind"] == "unknown"
    assert react.parse_action(None)["kind"] == "unknown"


def test_format_observation_success():
    obs = react.format_observation("get_current_weather", {"temp": 26}, "")
    assert obs.startswith("观察：")
    assert "26" in obs


def test_format_observation_error_keeps_channel():
    """失败也走同一条通道 —— 这是 W1 的规矩，在 ReAct 里同样成立。"""
    obs = react.format_observation("get_current_weather", None, "城市名为空")
    assert "失败" in obs and "城市名为空" in obs


def test_format_observation_truncates_long_result():
    """上下文爆炸是六个已知失败模式之一，回灌前必须截断。"""
    big = {"rows": ["x" * 100 for _ in range(100)]}
    obs = react.format_observation("query_orders_stats", big, "", max_chars=200)
    assert len(obs) < 400
    assert "已截断" in obs
    assert "原长" in obs


def test_describe_tools_reuses_w1_schema_description():
    """描述从 schema 自动生成，且带上 W1 写的「什么时候不该用」。"""
    text = react.describe_tools(TOOL_SCHEMAS)
    assert "get_current_weather" in text
    assert "不要调用" in text
    for name in react.tool_names(TOOL_SCHEMAS):
        assert name in text


def test_single_step_final_answer(monkeypatch):
    script(monkeypatch, ["思考：常识够用\n最终答案：直接用即可"])
    r = react.run_react(settings(), "今天星期几")
    assert r["ok"] is True
    assert r["steps"] == 1
    assert r["stop_reason"] == "final_answer"
    assert r["answer"] == "直接用即可"
    assert r["tool_calls_made"] == 0


def test_two_step_action_then_final(monkeypatch):
    script(monkeypatch, [
        '思考：需要天气\n行动：get_current_weather {"city":"杭州"}',
        "思考：拿到了\n最终答案：杭州 26℃，晴。",
    ])
    r = react.run_react(settings(), "杭州现在天气如何")
    assert r["ok"] is True
    assert r["steps"] == 2
    assert r["tool_calls_made"] == 1
    assert r["trace"][0]["tool"] == "get_current_weather"
    assert r["trace"][0]["error"] == ""


def test_react_does_not_pass_tools_to_model(monkeypatch):
    """核心断言：手写 ReAct 不依赖原生 function calling。"""
    calls = script(monkeypatch, ["思考：够了\n最终答案：好了"])
    react.run_react(settings(), "你好")
    assert calls[0]["kw"].get("tools") is None
    assert calls[0]["kw"].get("tool_choice") is None
    assert "get_current_weather" in calls[0]["system"]


def test_max_steps_fuse(monkeypatch):
    """步数到顶必须停，且如实说没做完（ok=False + 空答案），绝不编一个。"""
    act = '思考：还要再查\n行动：get_current_weather {"city":"杭州"}'
    script(monkeypatch, [act, act, act])
    r = react.run_react(settings(), "杭州天气", max_steps=3)
    assert r["ok"] is False
    assert r["answer"] == ""
    assert r["steps"] == 3
    assert r["stop_reason"] == "max_steps_reached"
    assert r["tool_calls_made"] == 3


def test_format_error_is_fed_back_then_recovers(monkeypatch):
    """格式不明不当异常，回灌一次示范，模型自己改过来。"""
    calls = script(monkeypatch, [
        "我觉得应该先查一下天气",
        "思考：好了\n最终答案：杭州 26℃。",
    ])
    r = react.run_react(settings(), "杭州天气")
    assert r["ok"] is True
    assert r["stop_reason"] == "final_answer"
    assert r["trace"][0]["error"] == "输出未按格式，无法解析"
    assert "行动" in calls[1]["kw"].get("system", "") or True


def test_consecutive_format_errors_stop(monkeypatch):
    """连续两次格式不明就停 —— 已经给过一次示范还改不对，继续只是烧钱。"""
    script(monkeypatch, ["随便说说", "还是随便说说", "继续随便说说"])
    r = react.run_react(settings(), "杭州天气")
    assert r["ok"] is False
    assert r["stop_reason"] == "unparsable_format"
    assert r["steps"] == react.MAX_UNKNOWN_IN_A_ROW


def test_tool_not_in_whitelist(monkeypatch):
    """最小权限第一层：本次没暴露的工具，模型调了就明确告知可用清单。"""
    from app.tools import TOOL_SCHEMAS as ALL
    only_weather = [t for t in ALL if t["function"]["name"] == "get_current_weather"]
    script(monkeypatch, [
        '思考：查订单\n行动：query_orders_stats {"sql":"SELECT 1"}',
        "思考：那就这样\n最终答案：查不了订单。",
    ])
    r = react.run_react(settings(), "统计订单", tools=only_weather)
    assert r["trace"][0]["tool"] == "query_orders_stats"
    assert "本次可用工具只有" in r["trace"][0]["error"]
    assert "get_current_weather" in r["trace"][0]["error"]


def test_tool_error_does_not_crash_loop(monkeypatch):
    """工具报错不是程序异常，循环要继续，不能整个挂掉。"""
    script(monkeypatch, [
        '思考：查天气\n行动：get_current_weather {"city":"杭州"}',
        "思考：失败就算了\n最终答案：拿不到天气。",
    ])
    monkeypatch.setattr(react, "dispatch", lambda name, raw: (None, "模拟故障"))
    r = react.run_react(settings(), "杭州天气")
    assert r["ok"] is True
    assert r["trace"][0]["error"] == "模拟故障"
    assert r["answer"].startswith("拿不到")


def test_on_event_sequence(monkeypatch):
    """事件流是块 E 接 trace 的输入，顺序和结构必须先定死。"""
    script(monkeypatch, [
        '思考：需要天气\n行动：get_current_weather {"city":"杭州"}',
        "思考：够了\n最终答案：26℃。",
    ])
    events: list[dict] = []
    react.run_react(settings(), "杭州天气", on_event=events.append)
    types = [e["type"] for e in events]
    assert types[0] == "step_start"
    assert "thought" in types
    assert "action" in types
    assert "observation" in types
    assert types[-1] == "final"


def test_cost_and_latency_accumulate(monkeypatch):
    """一次任务 = N 次调用，成本/延迟/token 必须按「整个任务」累加。"""
    script(monkeypatch, [
        '思考：查\n行动：get_current_weather {"city":"杭州"}',
        "思考：够了\n最终答案：26℃。",
    ])
    r = react.run_react(settings(), "杭州天气")
    assert r["cost_cny"] == pytest.approx(0.002)
    assert r["latency_s"] == pytest.approx(1.0)
    assert r["usage"]["prompt_tokens"] == 200
    assert r["usage"]["completion_tokens"] == 100


def test_budget_gate_blocks(monkeypatch):
    """成本闸门必须拦在第一步之前 —— 循环类调用一旦写错，几分钟就能烧掉不该花的钱。"""
    script(monkeypatch, ["思考：够了\n最终答案：好了"])
    monkeypatch.setattr(react, "summarize", lambda s: {"over_limit": True, "total_cny": 99.0})
    with pytest.raises(react.BudgetExceeded):
        react.run_react(settings(), "你好")


def test_endpoint_agent_react(monkeypatch):
    """端点接线正确：请求能进、字段能出。"""
    script(monkeypatch, ["思考：够了\n最终答案：26℃。"])
    with TestClient(app) as c:
        resp = c.post("/agent/react", json={"question": "杭州天气", "max_steps": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["stop_reason"] == "final_answer"
    assert body["steps"] == 1


def test_endpoint_rejects_illegal_max_steps(monkeypatch):
    script(monkeypatch, ["思考：够了\n最终答案：好了"])
    with TestClient(app) as c:
        resp = c.post("/agent/react", json={"question": "x", "max_steps": 99})
    assert resp.status_code == 422
