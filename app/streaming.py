"""SSE 流式输出 —— W1 第 4 天（9/23）的第一块，也是整个 W1 最难的一个文件。"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Iterator

from .config import Settings
from .cost import append_entry, estimate_cny
from .llm import LLMError, PROVIDER_ENDPOINT, RETRYABLE_CODES, _explain

DONE = "data: [DONE]"


def iter_chat_events(
    s: Settings,
    messages: list[dict],
    *,
    system: str | None = None,
    provider: str = "deepseek",
    model: str | None = None,
    thinking: bool = False,
    max_tokens: int = 1024,
    timeout: float = 120.0,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    record_cost: bool = True,
) -> Iterator[dict]:
    """同步生成器，逐个吐出事件。"""
    if provider not in PROVIDER_ENDPOINT:
        yield {"type": "error", "message": f"未知 provider: {provider}", "retryable": False}
        return

    key_attr, url_attr, default_model = PROVIDER_ENDPOINT[provider]
    api_key = getattr(s, key_attr, "")
    base_url = getattr(s, url_attr, "")
    if not api_key:
        yield {"type": "error", "message": f"{provider} 未配置 API key，检查 .env 里的 {key_attr}",
               "retryable": False}
        return
    model = model or default_model

    payload: dict = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + list(messages),
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if provider == "deepseek" and not thinking:
        payload["thinking"] = {"type": "disabled"}
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Authorization", "Bearer " + api_key)
    req.add_header("Content-Type", "application/json")

    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")[:300]
        yield {"type": "error", "retryable": e.code in RETRYABLE_CODES, "code": e.code,
               "message": f"{provider} 返回 HTTP {e.code}：{_explain(e.code)}　{raw}"}
        return
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "retryable": True, "message": f"{provider} 连接失败：{type(e).__name__} {e}"}
        return

    content_parts: list[str] = []
    tool_acc: dict[int, dict] = {}
    finish_reason = ""
    usage = {"in": 0, "out": 0, "cached": 0}
    first_token_at = None

    try:
        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                u = chunk.get("usage")
                if u:
                    usage = {
                        "in": u.get("prompt_tokens", 0),
                        "out": u.get("completion_tokens", 0),
                        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                    }

                if not chunk.get("choices"):
                    continue

                ch = chunk["choices"][0]
                delta = ch.get("delta") or {}

                text = delta.get("content") or ""
                if text:
                    if first_token_at is None:
                        first_token_at = round(time.time() - t0, 3)
                    content_parts.append(text)
                    yield {"type": "delta", "text": text}

                for frag in delta.get("tool_calls") or []:
                    idx = frag.get("index", 0)
                    slot = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if frag.get("id"):
                        slot["id"] = frag["id"]
                    fn = frag.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                    yield {"type": "tool_delta", "index": idx, "name": slot["name"],
                           "arguments": slot["arguments"]}

                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "retryable": True, "message": f"读取流中断：{type(e).__name__} {e}"}

    latency = round(time.time() - t0, 3)
    tool_calls = [
        {"id": v["id"], "type": "function",
         "function": {"name": v["name"], "arguments": v["arguments"]}}
        for _, v in sorted(tool_acc.items())
    ]

    cost = 0.0
    if record_cost:
        cost = estimate_cny(model, usage["in"], usage["out"], usage["cached"])
        append_entry(s, {
            "provider": provider, "model": model,
            "in": usage["in"], "out": usage["out"], "cached": usage["cached"],
            "latency_s": latency, "cost_cny": cost, "streamed": True,
        })

    yield {"type": "usage", "usage": usage, "cost_cny": cost, "latency_s": latency,
           "first_token_s": first_token_at, "provider": provider, "model": model}
    yield {"type": "done", "content": "".join(content_parts), "tool_calls": tool_calls,
           "finish_reason": finish_reason}


def sse_format(event: dict) -> str:
    """一个事件一帧。"""
    return f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


async def pump_worker_to_sse(worker, request, *, heartbeat_s: float = 15.0):
    """回调式版本：worker(emit) 在线程里跑，需要推送就调 emit。"""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def emit(ev):
        loop.call_soon_threadsafe(q.put_nowait, ev)

    task = asyncio.create_task(asyncio.to_thread(worker, emit))

    async for chunk in _drain(q, request, heartbeat_s=heartbeat_s):
        yield chunk
    task.cancel()


async def _drain(q, request, *, heartbeat_s: float) -> Iterator[str]:
    """队列 → SSE 文本。断开检测与心跳都在这里，两个 pump 共用。"""
    last_send = time.time()
    while True:
        try:
            ev = await asyncio.wait_for(q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if await request.is_disconnected():
                return
            if heartbeat_s and time.time() - last_send > heartbeat_s:
                last_send = time.time()
                yield ": heartbeat\n\n"
            continue
        if ev is None:
            return
        last_send = time.time()
        yield sse_format(ev)
        if await request.is_disconnected():
            return


async def pump_to_sse(gen_factory, request, *, heartbeat_s: float = 15.0) -> Iterator[str]:
    """生成器式版本：把同步生成器搬到异步 SSE 响应里，且能被客户端断开叫停。"""
    q: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            for ev in gen_factory():
                if stop.is_set():
                    return
                loop.call_soon_threadsafe(q.put_nowait, ev)
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(
                q.put_nowait, {"type": "error", "retryable": False,
                               "message": f"{type(e).__name__}: {e}"})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        async for chunk in _drain(q, request, heartbeat_s=heartbeat_s):
            yield chunk
    finally:
        stop.set()
        task.cancel()
