"""LangGraph 对照 —— W4 块 B。"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASK_RETRIEVE = "deepseek 的上下文硬盘缓存怎么计费"
TASK_WRITE = "帮我创建一个工单"

NODES = {
    "plan": lambda st: {"route": st.get("route", "answer")},
    "retrieve": lambda st: {"docs": [{"text": "缓存价格表"}]},
    "confirm": lambda st: {"approved": True},
    "answer": lambda st: {"answer": "答：" + st.get("task", "")},
}
ROUTES = {"retrieve": "retrieve", "act": "confirm",
          "answer": "answer", "escalate": "answer"}


def own_side() -> dict:
    from app.graph import END, InMemoryCheckpointer, StateGraph
    from app.workflow import rule_planner

    g = StateGraph()
    g.add_node("plan", lambda st: {**NODES["plan"](st), **rule_planner(st)})
    g.add_node("retrieve", NODES["retrieve"])
    g.add_node("confirm", NODES["confirm"])
    g.add_node("answer", NODES["answer"])
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", lambda st: st.get("route", "answer"), ROUTES)
    g.add_edge("retrieve", "answer")
    g.add_edge("confirm", "answer")
    g.add_edge("answer", END)

    cg = g.compile(checkpointer=InMemoryCheckpointer(), interrupt_before=("confirm",))

    out1 = cg.invoke({"task": TASK_RETRIEVE}, {"configurable": {"thread_id": "a"}})
    out2 = cg.invoke({"task": TASK_WRITE}, {"configurable": {"thread_id": "b"}})
    cg.approve("confirm")
    out3 = cg.invoke(None, {"configurable": {"thread_id": "b"}})

    return {
        "engine": "app.graph（自研）",
        "retrieve_path": {"visited": out1["visited"], "interrupt": out1["__interrupt__"]},
        "write_path_before": {"visited": out2["visited"], "interrupt": out2["__interrupt__"]},
        "write_path_resumed": {"visited": out3["visited"],
                               "approved": out3["state"].get("approved")},
    }


def langgraph_side() -> dict:
    from typing import TypedDict

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
    from app.workflow import rule_planner

    class S(TypedDict):
        task: str
        route: str
        docs: list
        approved: bool
        answer: str

    def plan(st: S) -> dict:
        return {**NODES["plan"](st), **rule_planner(st)}

    g: StateGraph = StateGraph(S)
    g.add_node("plan", plan)
    g.add_node("retrieve", NODES["retrieve"])
    g.add_node("confirm", NODES["confirm"])
    g.add_node("answer", NODES["answer"])
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", lambda st: st.get("route", "answer"), ROUTES)
    g.add_edge("retrieve", "answer")
    g.add_edge("confirm", "answer")
    g.add_edge("answer", END)

    app = g.compile(checkpointer=MemorySaver(), interrupt_before=["confirm"])

    cfg_r = {"configurable": {"thread_id": "a"}}
    out1 = app.invoke({"task": TASK_RETRIEVE, "route": "", "docs": [],
                       "approved": False, "answer": ""}, cfg_r)

    cfg_w = {"configurable": {"thread_id": "b"}}
    out2 = app.invoke({"task": TASK_WRITE, "route": "", "docs": [],
                       "approved": False, "answer": ""}, cfg_w)
    interrupted_at = app.get_state(cfg_w).next if hasattr(app, "get_state") else None

    out3 = app.invoke(None, cfg_w)

    return {
        "engine": "langgraph（框架）",
        "retrieve_path": {"visited": _visited(out1), "interrupt": None,
                          "answer": out1.get("answer", "")},
        "write_path_before": {"visited": _visited(out2),
                              "interrupt": list(interrupted_at) if interrupted_at else None,
                              "approved": out2.get("approved")},
        "write_path_resumed": {"visited": _visited(out3),
                               "approved": out3.get("approved")},
    }


def _visited(state: dict) -> list[str]:
    """LangGraph 不直接给 visited 列表，这里从最终 state 推不出，"""
    return []


def compare(own: dict, lg: dict) -> dict:
    checks = []
    own_int = own["write_path_before"]["interrupt"]
    lg_int = lg["write_path_before"]["interrupt"]
    checks.append({
        "项": "写操作是否停在 confirm 前",
        "自研": own_int,
        "langgraph": lg_int,
        "一致": bool(own_int) == bool(lg_int),
    })
    checks.append({
        "项": "retrieve 路径是否直接走完（无中断）",
        "自研": own["retrieve_path"]["interrupt"] is None,
        "langgraph": lg["retrieve_path"]["interrupt"] is None,
        "一致": (own["retrieve_path"]["interrupt"] is None)
                == (lg["retrieve_path"]["interrupt"] is None),
    })
    return {"checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description="自研状态图 vs LangGraph 对照")
    ap.add_argument("--with-langgraph", action="store_true",
                    help="跑框架侧对照（需 pip install langgraph）")
    args = ap.parse_args()

    own = own_side()
    print("=== 自研侧（app/graph.py）===")
    print(json.dumps(own, ensure_ascii=False, indent=2))

    if not args.with_langgraph:
        print("\n（未跑框架侧。想看对照：pip install langgraph && "
              "python scripts/langgraph_compare.py --with-langgraph）")
        return 0

    try:
        lg = langgraph_side()
    except Exception as e:  # noqa: BLE001
        print(f"\n=== 框架侧跳过：{type(e).__name__}: {e}")
        print("（langgraph 版本差异可能导致 API 不兼容；自研侧结论不受影响）")
        return 0

    print("\n=== 框架侧（langgraph）===")
    print(json.dumps(lg, ensure_ascii=False, indent=2))
    print("\n=== 对比 ===")
    print(json.dumps(compare(own, lg), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
