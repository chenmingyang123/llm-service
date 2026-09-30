"""最小状态图引擎 —— W4 块 B。"""
from __future__ import annotations

from typing import Any, Callable


END = "__end__"


class GraphInterrupt(RuntimeError):
    """图在指定节点前暂停。"""


class InMemoryCheckpointer:
    """内存版 checkpointer（对应 LangGraph 的 MemorySaver）。"""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def put(self, thread_id: str, checkpoint: dict) -> None:
        self._store[thread_id] = checkpoint

    def get(self, thread_id: str) -> dict | None:
        return self._store.get(thread_id)

    def clear(self, thread_id: str | None = None) -> None:
        if thread_id is None:
            self._store.clear()
        else:
            self._store.pop(thread_id, None)


class StateGraph:
    """最小状态图。节点、固定边、条件边，三样就够。"""

    def __init__(self) -> None:
        self.nodes: dict[str, Callable[[dict], dict]] = {}
        self.edges: dict[str, str] = {}
        self.branches: dict[str, tuple[Callable[[dict], Any], dict[Any, str]]] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: Callable[[dict], dict]) -> "StateGraph":
        if name in self.nodes:
            raise ValueError(f"节点重名：{name}")
        self.nodes[name] = fn
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges[src] = dst
        return self

    def add_conditional_edges(self, src: str,
                              router: Callable[[dict], Any],
                              mapping: dict[Any, str]) -> "StateGraph":
        """条件分支：走向由 router(state) 的返回值决定。"""
        self.branches[src] = (router, mapping)
        return self

    def compile(self,
                checkpointer: InMemoryCheckpointer | None = None,
                interrupt_before: tuple[str, ...] = ()) -> "CompiledGraph":
        if self.entry is None:
            raise ValueError("没有入口节点：先调用 set_entry_point")
        for name in interrupt_before:
            if name not in self.nodes:
                raise ValueError(f"interrupt_before 指向了不存在的节点：{name}")
        return CompiledGraph(self, checkpointer=checkpointer,
                             interrupt_before=interrupt_before)


class CompiledGraph:
    """编译后的图：可 invoke，可中断，可恢复。"""

    def __init__(self, g: StateGraph,
                 checkpointer: InMemoryCheckpointer | None,
                 interrupt_before: tuple[str, ...]):
        self.g = g
        self.checkpointer = checkpointer
        self.interrupt_before = set(interrupt_before)
        self._approved: set[str] = set()

    def invoke(self, state: dict | None = None,
               config: dict | None = None,
               on_event: Callable[[dict], None] | None = None) -> dict:
        """跑一次图。"""
        cfg = (config or {}).get("configurable", {})
        thread_id = cfg.get("thread_id") or "default"

        if state is None:
            ck = self.checkpointer.get(thread_id) if self.checkpointer else None
            if ck is None:
                raise RuntimeError(
                    f"thread {thread_id} 没有 checkpoint，无法 resume。"
                    f"要么先带 state 调一次，要么确认 thread_id 没写错")
            state = dict(ck["state"])
            current = ck["next"]
            self._approved.update(ck.get("approved", []))
            visited = list(ck.get("visited", []))
            state.setdefault("__resumed__", True)
        else:
            state = dict(state)
            current = self.g.entry
            visited: list[str] = []
            self._approved = set()

        for _ in range(100):
            if current == END or current is None:
                break

            if current in self.interrupt_before and current not in self._approved:
                if self.checkpointer is None:
                    raise RuntimeError(
                        f"设置了 interrupt_before={sorted(self.interrupt_before)}，"
                        f"但没有 checkpointer —— 没地方存状态，resume 无从谈起")
                self.checkpointer.put(thread_id, {
                    "state": dict(state),
                    "next": current,
                    "approved": sorted(self._approved),
                    "visited": list(visited),
                })
                if on_event:
                    on_event({"type": "interrupt", "node": current,
                              "thread_id": thread_id, "visited": list(visited)})
                return {
                    "__interrupt__": current,
                    "state": state,
                    "visited": visited,
                    "thread_id": thread_id,
                }

            fn = self.g.nodes.get(current)
            if fn is None:
                raise RuntimeError(f"走到了不存在的节点：{current}")

            visited.append(current)
            if on_event:
                on_event({"type": "node_start", "node": current, "step": len(visited)})
            patch = fn(state) or {}
            if not isinstance(patch, dict):
                raise RuntimeError(f"节点 {current} 必须返回 dict（增量 patch），收到 {type(patch).__name__}")
            state = {**state, **patch}
            if on_event:
                on_event({"type": "node_end", "node": current,
                          "step": len(visited), "patch_keys": sorted(patch)})

            current = self._next_node(current, state)

        if self.checkpointer is not None:
            self.checkpointer.clear(thread_id)
        if on_event:
            on_event({"type": "end", "visited": list(visited), "nodes": len(visited)})
        return {"__interrupt__": None, "state": state, "visited": visited,
                "thread_id": thread_id}

    def _next_node(self, current: str, state: dict) -> str | None:
        if current in self.g.branches:
            router, mapping = self.g.branches[current]
            key = router(state)
            if key not in mapping:
                raise RuntimeError(
                    f"节点 {current} 的 router 返回了 {key!r}，"
                    f"不在 mapping 里：{sorted(mapping)}")
            return mapping[key]
        return self.g.edges.get(current)

    def approve(self, node: str) -> None:
        """标记某个确认点已被人工放行。"""
        self._approved.add(node)


__all__ = ["StateGraph", "CompiledGraph", "InMemoryCheckpointer", "GraphInterrupt", "END"]
