"""成本响应头中间件 —— 让每一次调用的成本在响应头里就看得见。"""
from __future__ import annotations

import json


def _num(v) -> str:
    return str(v if v is not None else 0)


class CostHeaderMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        start_msg: dict | None = None
        body: list[bytes] = []
        is_json = False
        buffering = False

        async def send_wrapper(message):
            nonlocal start_msg, is_json, buffering

            if message["type"] == "http.response.start":
                headers = dict(message.get("headers") or [])
                ctype = (headers.get(b"content-type") or b"").decode("latin-1").lower()
                is_json = "json" in ctype
                if is_json:
                    start_msg = message
                    buffering = True
                    return
                await send(message)
                return

            if message["type"] == "http.response.body":
                if not buffering:
                    await send(message)
                    return
                body.append(message.get("body", b"") or b"")
                if message.get("more_body"):
                    return
                self._attach(start_msg, b"".join(body))
                await send(start_msg)
                await send(message)
                return

            await send(message)

        await self.app(scope, receive, send_wrapper)

    @staticmethod
    def _attach(start_msg: dict, raw: bytes) -> None:
        """从响应体里掏出成本与 token 数，塞进 start 消息的 headers。"""
        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            return
        if not isinstance(data, dict):
            return

        extra: list[tuple[bytes, bytes]] = []
        if data.get("cost_cny") is not None:
            extra.append((b"x-cost-cny", _num(data["cost_cny"]).encode()))
        usage = data.get("usage") or {}
        for key, header in (("prompt_tokens", b"x-tokens-in"),
                            ("completion_tokens", b"x-tokens-out"),
                            ("cached_tokens", b"x-tokens-cached")):
            if usage.get(key) is not None:
                extra.append((header, _num(usage[key]).encode()))
        if data.get("model"):
            extra.append((b"x-llm-model", str(data["model"]).encode()))
        if extra:
            start_msg.setdefault("headers", []).extend(extra)
