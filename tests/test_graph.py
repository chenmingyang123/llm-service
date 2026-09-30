"""最小状态图引擎 + 受控 workflow 的测试 —— W4 块 B。"""
from __future__ import annotations

import pytest

from app import graph as G
from app.graph import END, CompiledGraph, InMemoryCheckpointer, StateGraph
from app import workflow as W


def _linear():
    g = StateGraph()
    g.add_node("a", lambda st: {"a": 1})
    g.add_node("b", lambda st: {"b": 2})
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile()


def test_linear_graph_runs_in_order():
    out = _linear().invoke({"x": 0})
    assert out["__interrupt__"] is None
    assert out["visited"] == ["a", "b"]
    assert out["state"]["x"] == 0
    assert out["state"]["a"] == 1 and out["state"]["b"] == 2


def test_node_must_return_dict():
    g = StateGraph()
    g.add_node("a", lambda st: "不是 dict")
    g.set_entry_point("a")
    g.add_edge("a", END)
    with pytest.raises(RuntimeError, match="增量 patch"):
        g.compile().invoke({})


def test_conditional_edge_follows_router():
    g = StateGraph()
    g.add_node("start", lambda st: {})
    g.add_node("left", lambda st: {"side": "left"})
    g.add_node("right", lambda st: {"side": "right"})
    g.set_entry_point("start")
    g.add_conditional_edges("start", lambda st: st["go"],
                            {"L": "left", "R": "right"})
    g.add_edge("left", END)
    g.add_edge("right", END)
    out = g.compile().invoke({"go": "R"})
    assert out["visited"] == ["start", "right"]
    assert out["state"]["side"] == "right"


def test_router_unknown_key_raises():
    """路由返回意外值时必须明确报错 —— 悄悄走默认分支会掩盖 bug。"""
    g = StateGraph()
    g.add_node("start", lambda st: {})
    g.add_node("left", lambda st: {})
    g.set_entry_point("start")
    g.add_conditional_edges("start", lambda st: "哪都不是", {"L": "left"})
    g.add_edge("left", END)
    with pytest.raises(RuntimeError, match="不在 mapping 里"):
        g.compile().invoke({})


def test_graph_without_entry_raises():
    g = StateGraph()
    g.add_node("a", lambda st: {})
    with pytest.raises(ValueError, match="没有入口节点"):
        g.compile()


def _interrupt_graph(interrupt_before=("confirm",), with_cp=True, cp=None):
    g = StateGraph()
    g.add_node("plan", lambda st: {"planned": True})
    g.add_node("confirm", lambda st: {"approved": True})
    g.add_node("answer", lambda st: {"answer": "done"})
    g.set_entry_point("plan")
    g.add_edge("plan", "confirm")
    g.add_edge("confirm", "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=cp or (InMemoryCheckpointer() if with_cp else None),
                     interrupt_before=interrupt_before)


def test_interrupt_before_stops_at_node():
    out = _interrupt_graph().invoke({"task": "t"},
                                    {"configurable": {"thread_id": "T1"}})
    assert out["__interrupt__"] == "confirm"
    assert out["visited"] == ["plan"]
    assert out["state"]["planned"] is True


def test_interrupt_requires_checkpointer():
    """没 checkpointer 就没法存状态，resume 无从谈起 —— 跑起来时必须报错。"""
    with pytest.raises(RuntimeError, match="没有 checkpointer"):
        _interrupt_graph(with_cp=False).invoke({"task": "t"})


def test_resume_continues_after_interrupt():
    """中断 → 放行 → 继续，这是人工确认点的主路径。"""
    cg = _interrupt_graph()
    out1 = cg.invoke({"task": "t"}, {"configurable": {"thread_id": "T2"}})
    assert out1["__interrupt__"] == "confirm"

    cg.approve("confirm")
    out2 = cg.invoke(None, {"configurable": {"thread_id": "T2"}})
    assert out2["__interrupt__"] is None
    assert out2["visited"] == ["plan", "confirm", "answer"]
    assert out2["state"]["answer"] == "done"


def test_approved_node_does_not_interrupt_again():
    """放行过的节点不能再拦 —— 否则 resume 后又中断，死循环。"""
    cg = _interrupt_graph()
    cg.invoke({"task": "t"}, {"configurable": {"thread_id": "T3"}})
    cg.approve("confirm")
    out = cg.invoke(None, {"configurable": {"thread_id": "T3"}})
    assert out["__interrupt__"] is None


def test_resume_without_checkpoint_raises():
    cg = _interrupt_graph()
    with pytest.raises(RuntimeError, match="没有 checkpoint"):
        cg.invoke(None, {"configurable": {"thread_id": "不存在的线程"}})


def test_checkpoint_cleared_after_finish():
    """跑完要清掉 checkpoint，避免下次误 resume 到一个已完成的线程。"""
    cp = InMemoryCheckpointer()
    cg = _interrupt_graph(cp=cp)
    cg.invoke({"task": "t"}, {"configurable": {"thread_id": "T4"}})
    assert cp.get("T4") is not None
    cg.approve("confirm")
    cg.invoke(None, {"configurable": {"thread_id": "T4"}})
    assert cp.get("T4") is None


def test_cyclic_graph_has_hard_limit():
    """图也可能写成环 —— 必须有步数上限，和 Agent 的熔断同理。"""
    g = StateGraph()
    g.add_node("spin", lambda st: {"n": st.get("n", 0) + 1})
    g.set_entry_point("spin")
    g.add_edge("spin", "spin")
    out = g.compile().invoke({})
    assert out["visited"].count("spin") == 100


@pytest.mark.parametrize("task,route", [
    ("deepseek 的上下文缓存怎么计费", "retrieve"),
    ("帮我创建一个工单", "act"),
    ("我要投诉并要求退款", "escalate"),
    ("今天天气不错", "answer"),
])
def test_rule_planner_routes(task, route):
    assert W.rule_planner({"task": task})["route"] == route


def _wf(retriever=None, responder=None, planner=None, confirmed=None, ledger=None):
    return W.build_service_workflow(
        planner=planner,
        retriever=retriever or (lambda st: {"docs": [{"text": "资料", "source": "deepseek"}],
                                            "retrieve_error": ""}),
        responder=responder or (lambda st: {"answer": "答：" + st.get("task", "")}),
        confirmed=confirmed,
        ledger=ledger,
    )


def test_workflow_retrieve_path():
    out = _wf().invoke({"task": "计费怎么算"})
    assert out["visited"] == ["plan", "retrieve", "answer"]
    assert out["state"]["answer"].startswith("答：")


def test_workflow_escalate_path():
    out = _wf().invoke({"task": "我要退款"})
    assert out["visited"] == ["plan", "escalate"]
    assert out["state"]["escalated"] is True


def test_workflow_write_action_interrupts_for_human():
    """写操作必须停在人工确认点 —— 这是块 B 最有价值的一条断言。"""
    out = _wf().invoke({"task": "帮我建单"}, {"configurable": {"thread_id": "W1"}})
    assert out["__interrupt__"] == "confirm"
    assert out["visited"] == ["plan"]


def test_workflow_resume_after_approval():
    """两层批准都给齐（流程 approve + 动作令牌 confirmed）时，写动作真正执行。"""
    confirmed: set[str] = set()
    out = W.run_service_workflow(_wf(confirmed=confirmed), "帮我建单",
                                 thread_id="W2", approve=True, confirmed=confirmed)
    assert out["__interrupt__"] is None
    assert out["visited"] == ["plan", "confirm", "answer"]
    assert out["state"]["approved"] is True
    assert out["state"].get("action_result") is not None


def test_workflow_resume_flow_only_still_blocked():
    """只给流程层批准、不给动作令牌 —— 闸门 ③ 仍然拦下写动作。"""
    out = W.run_service_workflow(_wf(), "帮我提交一个数据异常工单",
                                 thread_id="W3", approve=True)
    assert out["__interrupt__"] is None
    assert out["state"]["approved"] is False
    assert "尚未获得人工确认" in out["state"]["action_error"]


def test_workflow_survives_retriever_error():
    """检索失败不能把 workflow 打挂 —— 工具失败是结果不是异常。"""
    def bad(st):
        return {"docs": [], "retrieve_error": "检索链路未就绪"}
    out = _wf(retriever=bad).invoke({"task": "计费怎么算"})
    assert out["__interrupt__"] is None
    assert out["state"]["retrieve_error"] == "检索链路未就绪"
    assert out["state"]["answer"]


def test_llm_planner_falls_back_to_rule_on_bad_json():
    """LLM 路由解析失败时回退规则版，而不是让流程崩掉。"""
    from app import workflow as wf_mod

    class FakeRes:
        def __init__(self): self.content = "这不是 JSON"

    def fake_call(s, messages, **kw):
        return {"content": "这不是 JSON", "tool_calls": [], "finish_reason": "stop",
                "usage": {"in": 1, "out": 1, "cached": 0}, "latency_s": 0.1,
                "model": "m", "cost_cny": 0.0, "provider": "deepseek"}

    wf_mod.__dict__["_llm"] = None
    import app.llm as llm_mod
    orig = llm_mod.call_chat
    llm_mod.call_chat = fake_call
    try:
        planner = W.make_llm_planner(None)
        assert planner({"task": "计费怎么算"})["route"] == "retrieve"
    finally:
        llm_mod.call_chat = orig


def test_llm_planner_rejects_unknown_route():
    import app.llm as llm_mod

    def fake_call(s, messages, **kw):
        return {"content": '{"route": "毁灭世界"}', "tool_calls": [],
                "finish_reason": "stop", "usage": {"in": 1, "out": 1, "cached": 0},
                "latency_s": 0.1, "model": "m", "cost_cny": 0.0, "provider": "deepseek"}

    orig = llm_mod.call_chat
    llm_mod.call_chat = fake_call
    try:
        planner = W.make_llm_planner(None)
        assert planner({"task": "随便"})["route"] == "answer"
    finally:
        llm_mod.call_chat = orig


def test_tools_for_filters_by_risk_level():
    from app.tools import TOOL_RISK, TOOL_SCHEMAS, tools_for

    reads = tools_for(["read"])
    names = [t["function"]["name"] for t in reads]
    assert "search_docs" in names
    assert "get_current_weather" in names
    assert all(TOOL_RISK.get(n, ("read", False))[0] == "read" for n in names)
    assert len(reads) <= len(TOOL_SCHEMAS)


def test_search_docs_is_registered_and_read_only():
    from app.tools import ARG_MODELS, KNOWN, REGISTRY, TOOL_RISK, TOOL_SCHEMAS

    assert "search_docs" in KNOWN
    assert "search_docs" in REGISTRY
    assert "search_docs" in ARG_MODELS
    assert any(t["function"]["name"] == "search_docs" for t in TOOL_SCHEMAS)
    assert TOOL_RISK["search_docs"] == ("read", False)


def test_search_docs_filters_before_truncating(monkeypatch):
    """先放大候选池 → 过滤 → 再截断。"""
    from app import tools

    class FakePipe:
        def __init__(self):
            self.asked_topk = None

        def search(self, query, mode="hybrid", topk=5, pool=50):
            self.asked_topk = topk
            rows = [{"chunk_id": f"c{i}", "source": s, "title": "t", "path": "",
                     "score": 0.9 - i * 0.1, "text": "x"}
                    for i, s in enumerate(["bailian", "bailian", "bailian",
                                           "zhipu", "zhipu"])]
            return {"query": query, "mode": mode, "hits": rows[:topk],
                    "reranked": False, "latency_ms": 1.0}

    fake = FakePipe()
    monkeypatch.setattr(tools, "_PIPELINE", fake)

    res = tools.search_docs(tools.SearchDocsArgs(query="缓存计费", source="zhipu", top_k=2))
    assert fake.asked_topk > 2
    assert [h["source"] for h in res["hits"]] == ["zhipu", "zhipu"]
    assert res["count"] == 2


def test_search_docs_schema_has_when_not_to_use():
    """沿用 W1 的规则 1：描述里必须写「何时不要用」。"""
    from app.tools import TOOL_SCHEMAS

    fn = next(t["function"] for t in TOOL_SCHEMAS
              if t["function"]["name"] == "search_docs")
    assert "何时不要用" in fn["description"]
    assert "术语" in fn["parameters"]["properties"]["query"]["description"]
    assert "留空" in fn["parameters"]["properties"]["source"]["description"]
