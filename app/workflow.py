"""企业服务受控 workflow —— W4 块 B 的交付物。"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from .graph import END, CompiledGraph, InMemoryCheckpointer, StateGraph

_NEED_RETRIEVAL = ("怎么", "如何", "是什么", "多少", "区别", "文档", "参数",
                   "限制", "计费", "是否支持", "查一下", "查询")
_NEED_ACTION = ("创建", "建单", "提交", "发起", "发送", "通知", "修改", "取消")
_NEED_ESCALATE = ("投诉", "退款", "赔", "法务", "人工", "转人工")


def _book(ledger: dict | None, res: dict) -> None:
    """把一次调用的账记进 ledger。"""
    if ledger is None:
        return
    ledger["calls"] = ledger.get("calls", 0) + 1
    ledger["cost_cny"] = round(ledger.get("cost_cny", 0.0) + (res.get("cost_cny") or 0.0), 6)
    ledger["latency_s"] = round(ledger.get("latency_s", 0.0) + (res.get("latency_s") or 0.0), 3)
    u = res.get("usage") or {}
    ledger["prompt_tokens"] = ledger.get("prompt_tokens", 0) + (u.get("in") or 0)
    ledger["completion_tokens"] = ledger.get("completion_tokens", 0) + (u.get("out") or 0)


def rule_planner(state: dict) -> dict:
    """规则版路由：零成本、可预测、可单测。"""
    task = (state.get("task") or "").lower()
    if any(k in task for k in _NEED_ESCALATE):
        route = "escalate"
    elif any(k in task for k in _NEED_ACTION):
        route = "act"
    elif any(k in task for k in _NEED_RETRIEVAL):
        route = "retrieve"
    else:
        route = "answer"
    return {"route": route}


def make_llm_planner(s, provider: str = "deepseek", model: str | None = None,
                     ledger: dict | None = None):
    """造一个用 LLM 判断路由的 planner。"""
    from .llm import call_chat

    system = (
        "你是企业服务工单的路由判断器。只输出一个 JSON，不要别的文字：\n"
        '{"route": "retrieve|act|answer|escalate"}\n'
        "retrieve = 需要查文档/资料才能答；act = 要执行一个写操作（建单、发通知）；\n"
        "escalate = 涉及投诉、退款、赔偿，必须转人工；answer = 直接就能答。"
    )

    def planner(state: dict) -> dict:
        res = call_chat(s, [{"role": "user", "content": state.get("task", "")}],
                        system=system, provider=provider, model=model,
                        thinking=False, max_tokens=64)
        _book(ledger, res)
        text = (res.get("content") or "").strip()
        try:
            route = json.loads(text).get("route", "answer")
        except Exception:  # noqa: BLE001
            route = rule_planner(state)["route"]
        if route not in ("retrieve", "act", "answer", "escalate"):
            route = "answer"
        return {"route": route}

    return planner


def docs_retriever(state: dict) -> dict:
    """检索节点：调 search_docs 工具（真实接 W3 的检索链路）。"""
    from .tools import dispatch

    args = json.dumps({"query": state.get("task", ""), "top_k": 5}, ensure_ascii=False)
    result, error = dispatch("search_docs", args)
    if error:
        return {"docs": [], "retrieve_error": error}
    hits = (result or {}).get("hits", [])
    return {"docs": hits, "retrieve_error": ""}


def make_llm_responder(s, provider: str = "deepseek", model: str | None = None,
                       ledger: dict | None = None):
    """造一个用 LLM 生成最终答案的 responder。"""
    from .llm import call_chat

    system = (
        "你是企业服务助手。只能依据下面提供的资料回答；"
        "资料里没有的，明确说「资料里没有，需要人工确认」，不要编。"
    )

    def responder(state: dict) -> dict:
        docs = state.get("docs") or []
        ctx = "\n\n".join(
            f"[{i + 1}] 来源={d.get('source', '')} 标题={d.get('title', '')}\n{d.get('text', '')}"
            for i, d in enumerate(docs[:5])
        )
        prompt = f"用户问题：{state.get('task', '')}\n\n可用资料：\n{ctx or '（无）'}"
        res = call_chat(s, [{"role": "user", "content": prompt}],
                        system=system, provider=provider, model=model,
                        thinking=False, max_tokens=512)
        _book(ledger, res)
        return {"answer": (res.get("content") or "").strip()}

    return responder


def build_service_workflow(
    *,
    planner: Callable[[dict], dict] | None = None,
    retriever: Callable[[dict], dict] | None = None,
    responder: Callable[[dict], dict] | None = None,
    checkpointer: InMemoryCheckpointer | None = None,
    interrupt_before: tuple[str, ...] = ("confirm",),
    confirmed: set[str] | None = None,
    ledger: dict | None = None,
    allowed_tools: list[str] | None = None,
) -> CompiledGraph:
    """搭出企业服务受控 workflow。"""
    planner = planner or rule_planner
    retriever = retriever or docs_retriever
    responder = responder or (lambda state: {"answer": "(未配置 responder)"})

    def node_plan(state: dict) -> dict:
        return planner(state)

    def node_retrieve(state: dict) -> dict:
        return retriever(state)

    def node_confirm(state: dict) -> dict:
        """真正执行写动作 —— 但先过闸门。"""
        from .guards import execute, gate

        pending = state.get("pending_action") or {}
        tool = pending.get("tool", "")
        raw = pending.get("args", "")
        if not tool:
            return {"approved": True, "action_result": None,
                    "action_error": "没有待执行的动作（pending_action 为空）"}

        gr = gate(tool, raw, allowed=allowed_tools, confirmed=confirmed)
        if not gr.allow:
            return {"approved": False, "action_error": gr.reason,
                    "action_result": None, "need_confirm": gr.need_confirm}

        res = execute(gr)
        return {"approved": True, "action_result": res, "action_error": "",
                "duplicate": bool(gr.duplicate)}

    def node_answer(state: dict) -> dict:
        return responder(state)

    def node_escalate(state: dict) -> dict:
        return {"answer": "该问题已转人工处理。", "escalated": True}

    g = StateGraph()
    g.add_node("plan", node_plan)
    g.add_node("retrieve", node_retrieve)
    g.add_node("confirm", node_confirm)
    g.add_node("answer", node_answer)
    g.add_node("escalate", node_escalate)

    g.set_entry_point("plan")
    g.add_conditional_edges("plan", lambda st: st.get("route", "answer"),
                            {"retrieve": "retrieve", "act": "confirm",
                             "answer": "answer", "escalate": "escalate"})
    g.add_edge("retrieve", "answer")
    g.add_edge("confirm", "answer")
    g.add_edge("answer", END)
    g.add_edge("escalate", END)

    return g.compile(checkpointer=checkpointer or InMemoryCheckpointer(),
                     interrupt_before=interrupt_before)


def default_pending_action(task: str) -> dict:
    """按任务派生一个待执行的写动作（演示与测试用）。"""
    rid = hashlib.md5(task.encode("utf-8")).hexdigest()[:12]
    return {
        "tool": "create_ticket",
        "args": json.dumps({"title": task[:40], "body": task, "request_id": rid},
                            ensure_ascii=False),
        "request_id": rid,
    }


def run_service_workflow(graph: CompiledGraph, task: str, *,
                         thread_id: str = "default",
                         approve: bool = False,
                         pending_action: dict | None = None,
                         confirmed: set[str] | None = None) -> dict:
    """跑一次 workflow，并处理「中断 → 批准 → 继续」这条主路径。"""
    state: dict = {"task": task}
    out = graph.invoke(state, {"configurable": {"thread_id": thread_id}})
    if out.get("__interrupt__") and approve:
        graph.approve(out["__interrupt__"])
        pa = pending_action if pending_action is not None else default_pending_action(task)
        if confirmed is not None and pa.get("request_id"):
            confirmed.add(pa["request_id"])
        ck = graph.checkpointer.get(thread_id) if graph.checkpointer else None
        if ck is not None:
            ck["state"]["pending_action"] = pa
            graph.checkpointer.put(thread_id, ck)
        out = graph.invoke(None, {"configurable": {"thread_id": thread_id}})
    return out


__all__ = ["build_service_workflow", "run_service_workflow", "rule_planner",
           "make_llm_planner", "make_llm_responder", "docs_retriever",
           "default_pending_action", "END", "Any"]
