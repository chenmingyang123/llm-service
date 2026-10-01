"""W4 块 E｜可观测层：把 on_event 事件流翻成 span 树。"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(ROOT, "data", "traces")

KIND_ROOT = "root"
KIND_STEP = "step"
KIND_TOOL = "tool"
KIND_LLM = "llm"
KIND_CONFIRM = "confirm"


def _wall() -> str:
    return time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time() % 1 * 1000):03d}"


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


@dataclass
class Span:
    """一次有始有终的动作。"""

    trace_id: str
    span_id: str
    name: str
    kind: str = "span"
    parent: str | None = None
    input: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    t0: float = field(default_factory=time.perf_counter)
    start: str = field(default_factory=_wall)
    t_epoch: float = field(default_factory=time.time)

    end_epoch: float | None = None
    end: str | None = None
    dur_ms: float | None = None
    output: Any = None
    error: str | None = None
    status: str = "ok"
    ended: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent": self.parent,
            "name": self.name,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "t_epoch": self.t_epoch,
            "end_epoch": self.end_epoch,
            "dur_ms": self.dur_ms,
            "status": self.status,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "meta": self.meta,
        }


class MemorySink:
    """测试用：写内存。"""

    def __init__(self) -> None:
        self.lines: list[dict] = []

    def write(self, rec: dict) -> None:
        self.lines.append(rec)

    def flush(self) -> None:
        pass


class JSONLSink:
    """生产用：追加写 JSONL。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def flush(self) -> None:
        pass

    def read(self) -> list[dict]:
        """查看器复用同一个 sink 读回来，避免两处各写一遍解析逻辑。"""
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]


class LangfuseExporter:
    """可插拔后端。装了 langfuse 且有 key 才生效，否则整层退化成 no-op。"""

    def __init__(self) -> None:
        self.enabled = False
        self.reason = ""
        self._client = None
        try:
            from langfuse import get_client  # type: ignore
        except Exception as e:  # noqa: BLE001 —— 故意宽收口：缺依赖不算致命
            self.reason = f"未安装 langfuse（{type(e).__name__}）"
            return
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            self.reason = "未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"
            return
        try:
            self._client = get_client()
            self.enabled = True
        except Exception as e:  # noqa: BLE001
            self.reason = f"初始化失败：{type(e).__name__}: {e}"

    def export(self, rec: dict) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            sp = self._client.start_span(
                name=rec["name"],
                trace_id=rec["trace_id"],
                input=rec.get("input"),
                metadata={"kind": rec.get("kind"), **(rec.get("meta") or {})},
            )
            sp.end(output=rec.get("output"))
        except Exception:  # noqa: BLE001 —— 观测后端永远不该把主流程搞挂
            pass


def langfuse_exporter() -> LangfuseExporter | None:
    """返回可用的 Langfuse 导出器；不可用返回 None（调用方直接跳过）。"""
    exp = LangfuseExporter()
    return exp if exp.enabled else None


class Tracer:
    """一棵 span 树的持有者。"""

    def __init__(
        self,
        name: str = "agent.run",
        *,
        trace_id: str | None = None,
        task: str | None = None,
        sink: Any | None = None,
        exporters: list[Any] | None = None,
        trace_dir: str | None = None,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.name = name
        self.sink = sink if sink is not None else JSONLSink(
            os.path.join(trace_dir or TRACE_DIR, f"{self.trace_id}.jsonl"))
        self.exporters = list(exporters or [])
        self._lock = threading.Lock()
        self.spans: list[Span] = []
        self._root = Span(
            trace_id=self.trace_id, span_id=uuid.uuid4().hex[:8],
            name=name, kind=KIND_ROOT, parent=None, input={"task": task},
        )
        self.spans.append(self._root)
        self._closed = False

    @property
    def root(self) -> Span:
        return self._root

    @property
    def root_id(self) -> str:
        return self._root.span_id

    def start_span(
        self,
        name: str,
        kind: str = "span",
        parent: str | None = None,
        input: Any = None,
        **meta: Any,
    ) -> Span:
        sp = Span(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:8],
            name=name,
            kind=kind,
            parent=parent if parent is not None else self.root_id,
            input=input,
            meta=dict(meta),
        )
        with self._lock:
            self.spans.append(sp)
        return sp

    def end_span(
        self,
        span: Span | None,
        output: Any = None,
        error: str | None = None,
        status: str | None = None,
        **meta: Any,
    ) -> Span | None:
        """结束一个 span。**幂等**：已结束的直接返回。"""
        if span is None or span.ended:
            return span
        span.dur_ms = _elapsed_ms(span.t0)
        span.end_epoch = time.time()
        span.end = _wall()
        span.output = output
        span.error = error
        span.status = status or ("error" if error else "ok")
        span.meta.update(meta)
        span.ended = True
        self._emit(span.to_record())
        return span

    @contextmanager
    def span(
        self,
        name: str,
        kind: str = "span",
        parent: str | None = None,
        input: Any = None,
        **meta: Any,
    ) -> Iterator[Span]:
        """手动埋点用。异常会自动按 error 结束并**继续抛出**——"""
        sp = self.start_span(name, kind=kind, parent=parent, input=input, **meta)
        try:
            yield sp
        except Exception as e:  # noqa: BLE001
            self.end_span(sp, error=f"{type(e).__name__}: {e}")
            raise
        else:
            self.end_span(sp)

    def finish_root(self, output: Any = None, error: str | None = None, **meta: Any) -> Span:
        """结束根 span，并兜底关掉所有还开着的子 span。"""
        for sp in list(self.spans):
            if sp is not self._root and not sp.ended:
                self.end_span(sp, error="父级结束时尚未闭合", status="abandoned")
        self.end_span(self._root, output=output, error=error, **meta)
        self._closed = True
        self.flush()
        return self._root

    def patch_root(self, **meta: Any) -> None:
        """根 span **结束后**再补汇总字段（成本 / 延迟 / usage）。"""
        self._root.meta.update(meta)
        rec = self._root.to_record()
        rec["patch"] = True
        self._emit(rec)

    def flush(self) -> None:
        self.sink.flush()

    def summarize(self) -> dict[str, Any]:
        """成本与延迟**分项**。"""
        root = self._root
        total = root.dur_ms or 0.0
        by_kind: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        for sp in self.spans:
            if sp is root:
                continue
            slot = by_kind.setdefault(sp.kind, {"count": 0, "ms": 0.0})
            slot["count"] += 1
            slot["ms"] = round(slot["ms"] + (sp.dur_ms or 0.0), 2)
            if sp.error:
                errors.append(f"{sp.name}: {sp.error}")

        step_ms = by_kind.get(KIND_STEP, {}).get("ms", 0.0)
        tool_ms = by_kind.get(KIND_TOOL, {}).get("ms", 0.0)
        llm_ms = by_kind.get(KIND_LLM, {}).get("ms", 0.0)

        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "status": root.status,
            "ok": root.status == "ok",
            "start": root.start,
            "end": root.end,
            "total_ms": total,
            "span_count": len(self.spans),
            "error_count": len(errors),
            "errors": errors,
            "breakdown": {
                k: {"count": v["count"], "ms": v["ms"],
                    "pct": round(v["ms"] / total * 100, 1) if total else 0.0}
                for k, v in sorted(by_kind.items())
            },
            "llm_ms_measured": llm_ms,
            "llm_ms_derived": round(max(step_ms - tool_ms, 0.0), 2),
            "cost_cny": root.meta.get("cost_cny"),
            "usage": root.meta.get("usage"),
            "stop_reason": root.meta.get("stop_reason"),
        }

    def _emit(self, rec: dict) -> None:
        try:
            self.sink.write(rec)
        except Exception:  # noqa: BLE001 —— 落盘失败不能拖垮主流程
            pass
        for exp in self.exporters:
            try:
                exp.export(rec)
            except Exception:  # noqa: BLE001
                pass


def make_trace_hook(tracer: Tracer, task: str | None = None) -> Callable[[dict], None]:
    """返回一个可直接传给 run_react / run_tool_loop(on_event=...) 的钩子。"""
    if task:
        tracer.root.meta["task"] = task

    state: dict[str, Any] = {"step": None, "tools": {}, "confirm": {}}

    def _parent() -> str:
        sp = state.get("step")
        return sp.span_id if sp is not None else tracer.root_id

    def _close_step() -> None:
        """结束当前 step span。"""
        sp = state.get("step")
        if sp is not None and not sp.ended:
            tracer.end_span(sp)
        state["step"] = None

    def _pop(name: str) -> Span | None:
        """同名工具可能连着调两次，用 FIFO 队列而不是 dict 覆盖。"""
        q = state["tools"].get(name) or []
        return q.pop(0) if q else None

    def on_event(ev: dict) -> None:  # noqa: C901 —— 映射表平铺比拆函数更好读
        t = ev.get("type")

        if t == "step_start":
            _close_step()
            state["step"] = tracer.start_span(
                f"step.{ev.get('step')}", kind=KIND_STEP,
                parent=tracer.root_id, input={"step": ev.get("step")})

        elif t == "thought":
            if state["step"] is not None:
                state["step"].meta["thought"] = (ev.get("thought") or "")[:500]

        elif t == "action":
            sp = tracer.start_span(
                f"tool.{ev.get('tool')}", kind=KIND_TOOL,
                parent=_parent(), input={"args": ev.get("args")})
            state["tools"].setdefault(ev.get("tool") or "?", []).append(sp)

        elif t == "observation":
            sp = _pop(ev.get("tool") or "?")
            tracer.end_span(sp, output=ev.get("result"), error=ev.get("error") or None)

        elif t == "tool_start":
            for name in ev.get("tools") or []:
                sp = tracer.start_span(
                    f"tool.{name}", kind=KIND_TOOL,
                    parent=_parent(), input={"turn": ev.get("turn")})
                state["tools"].setdefault(name or "?", []).append(sp)

        elif t == "tool_done":
            sp = _pop(ev.get("tool") or "?")
            tracer.end_span(sp, output=ev.get("result"), error=ev.get("error") or None)

        elif t == "need_confirm":
            sp = tracer.start_span(
                f"confirm.{ev.get('tool')}", kind=KIND_CONFIRM,
                parent=_parent(),
                input={"args": ev.get("args"), "request_id": ev.get("request_id"),
                       "reason": ev.get("reason")})
            state["confirm"][ev.get("confirm_id")] = sp

        elif t == "confirm_result":
            sp = state["confirm"].pop(ev.get("confirm_id"), None)
            approved = bool(ev.get("approved"))
            tracer.end_span(
                sp, output={"approved": approved},
                status="ok", approved=approved)

        elif t == "final":
            _close_step()
            tracer.finish_root(output={"answer": ev.get("answer")},
                               stop_reason="final_answer")

        elif t == "stop":
            _close_step()
            reason = ev.get("reason") or "unknown"
            tracer.finish_root(
                output=None,
                error=f"未产出最终答案：{reason}",
                status="error", stop_reason=reason)

        else:
            target = state.get("step") or tracer.root
            target.meta.setdefault("extra_events", []).append(ev)

    return on_event


def merge_patches(recs: list[dict]) -> list[dict]:
    """把 patch 记录合并回原 span（查看器 / 测试共用）。"""
    store: dict[str, dict] = {}
    order: list[str] = []
    for r in recs:
        sid = r.get("span_id")
        if r.get("patch") and sid in store:
            merged = dict(store[sid])
            merged.update({k: v for k, v in r.items() if k != "patch"})
            merged["meta"] = {**(store[sid].get("meta") or {}), **(r.get("meta") or {})}
            store[sid] = merged
        else:
            store[sid] = r
            order.append(sid)
    return [store[s] for s in order]


def read_trace(path: str) -> list[dict]:
    """读一条 JSONL trace（查看器 / 测试共用）。已合并 patch。"""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return merge_patches([json.loads(x) for x in f if x.strip()])


def trace_path(trace_id: str, trace_dir: str | None = None) -> str:
    return os.path.join(trace_dir or TRACE_DIR, f"{trace_id}.jsonl")


__all__ = [
    "Tracer", "Span", "MemorySink", "JSONLSink",
    "LangfuseExporter", "langfuse_exporter",
    "make_trace_hook", "read_trace", "merge_patches", "trace_path", "TRACE_DIR",
    "KIND_ROOT", "KIND_STEP", "KIND_TOOL", "KIND_LLM", "KIND_CONFIRM",
]
