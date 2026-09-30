"""块 C 真实冒烟：四道闸门 + workflow 集成 + on_event + 成本账本。"""
from __future__ import annotations
import sys, json, time
sys.path.insert(0, ".")

import app.workflow as W
from app.graph import InMemoryCheckpointer


def fmt(out: dict) -> None:
    s = out["state"]
    print(f"  中断点   : {out['__interrupt__']}")
    print(f"  轨迹     : {out['visited']}")
    print(f"  批准     : {s.get('approved')}")
    if s.get("action_error"):
        print(f"  动作错误 : {s['action_error']}")
    if s.get("action_result"):
        print(f"  动作结果 : {json.dumps(s['action_result'], ensure_ascii=False)[:120]}")


def main() -> None:
    print("=== 块 C 冒烟：受控 workflow + 四道闸门 ===\n")

    print("[1] 流程 approve=True，但无令牌 → 闸门 ③ 应拦下")
    g1 = W.build_service_workflow(
        retriever=lambda st: {"docs": [{"text": "资料", "source": "deepseek"}], "retrieve_error": ""},
        responder=lambda st: {"answer": "答：" + st.get("task", "")},
    )
    out1 = W.run_service_workflow(g1, "帮我提交一个数据异常工单", thread_id="smoke-1", approve=True)
    fmt(out1)
    assert out1["state"]["approved"] is False, "无令牌时不应执行写动作"
    print("  ✓ 已拦下\n")

    print("[2] 流程 approve + 动作令牌 → 写动作应执行")
    confirmed: set[str] = set()
    g2 = W.build_service_workflow(
        retriever=lambda st: {"docs": [{"text": "资料", "source": "deepseek"}], "retrieve_error": ""},
        responder=lambda st: {"answer": "答：" + st.get("task", "")},
        confirmed=confirmed,
    )
    out2 = W.run_service_workflow(g2, "帮我提交一个数据异常工单", thread_id="smoke-2",
                                  approve=True, confirmed=confirmed)
    fmt(out2)
    assert out2["state"]["approved"] is True, "有令牌时应执行写动作"
    assert out2["state"].get("action_result"), "应拿到动作结果"
    print("  ✓ 已执行\n")

    print("[3] 同一 task 再跑一次 → 幂等命中，不重复执行")
    out3 = W.run_service_workflow(g2, "帮我提交一个数据异常工单", thread_id="smoke-3",
                                  approve=True, confirmed=confirmed)
    fmt(out3)
    assert out3["state"].get("duplicate") is True, "应标记为幂等命中"
    print("  ✓ 幂等生效\n")

    print("[4] on_event 钩子：workflow 是否发出可观测事件")
    events: list[str] = []
    cp = InMemoryCheckpointer()
    g4 = W.build_service_workflow(
        retriever=lambda st: {"docs": [], "retrieve_error": ""},
        responder=lambda st: {"answer": "答"},
        checkpointer=cp,
    )
    out4 = g4.invoke({"task": "计费怎么算"}, None, on_event=lambda e: events.append(e["type"]))
    print(f"  事件序列 : {events}")
    assert "node_start" in events and "end" in events, "应发出 node_start / end 事件"
    print("  ✓ on_event 生效\n")

    print("[5] 成本账本：LLM 路径是否记账")
    from app.config import Settings
    ledger: dict = {"calls": 0, "cost_cny": 0.0, "ms": 0}
    g5 = W.build_service_workflow(
        retriever=lambda st: {"docs": [], "retrieve_error": ""},
        responder=W.make_llm_responder(Settings(), ledger=ledger),
        ledger=ledger,
    )
    g5.invoke({"task": "deepseek 怎么计费"}, None)
    print(f"  账本     : calls={ledger['calls']} cost={ledger['cost_cny']:.6f} ms={ledger['ms']}")
    assert ledger["calls"] >= 1, "responder 走 LLM 时应记账"
    print("  ✓ 账本生效\n")

    print("=== 全部冒烟通过 ===")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n总耗时 {time.time()-t0:.2f}s")
