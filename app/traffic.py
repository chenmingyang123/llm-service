"""HTTP 流量日志 —— 每一次 curl 都自动留痕，不用手工抄。"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

RECORD_PREFIXES = ("/chat", "/agent")

MAX_BODY = 200_000


def _log_path() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"traffic-{datetime.now().strftime('%Y%m%d')}.jsonl")


def _enabled() -> bool:
    """测试环境不落盘。"""
    return os.getenv("ENV", "dev") != "test"


def _safe_json(raw: bytes) -> object:
    """能解析成 JSON 就解析，不能就原样留字符串。日志不该因为解析失败而丢。"""
    if not raw:
        return None
    text = raw.decode("utf-8", "replace")[:MAX_BODY]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class TrafficLogMiddleware:
    """纯 ASGI 中间件，和 RequestIdMiddleware 同一套写法，升级不易碎。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not _enabled() or not path.startswith(RECORD_PREFIXES):
            await self.app(scope, receive, send)
            return

        req_chunks: list[bytes] = []
        resp_chunks: list[bytes] = []
        status = [0]
        streamed = [False]

        async def recv():
            msg = await receive()
            if msg.get("type") == "http.request":
                req_chunks.append(msg.get("body", b"") or b"")
            return msg

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status[0] = message.get("status", 0)
                headers = dict(message.get("headers") or [])
                ctype = (headers.get(b"content-type") or b"").decode("latin-1").lower()
                if "text/event-stream" in ctype:
                    streamed[0] = True
            elif message["type"] == "http.response.body" and not streamed[0]:
                resp_chunks.append(message.get("body", b"") or b"")
            await send(message)

        t0 = time.time()
        error = ""
        try:
            await self.app(scope, recv, send_wrapper)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            latency = round(time.time() - t0, 3)
            try:
                entry = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "method": scope.get("method", ""),
                    "path": path,
                    "status": status[0],
                    "latency_s": latency,
                    "request_id": scope.get("request_id", ""),
                    "request": _safe_json(b"".join(req_chunks)),
                    "response": {"streamed": True, "note": "SSE 逐帧输出，未落盘"}
                                 if streamed[0] else _safe_json(b"".join(resp_chunks)),
                    "streamed": streamed[0],
                }
                if error:
                    entry["error"] = error
                with open(_log_path(), "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:  # noqa: BLE001
                pass
