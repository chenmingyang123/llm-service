"""LLM 调用层。"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .config import Settings
from .cost import append_entry, estimate_cny

PROVIDER_ENDPOINT = {
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "deepseek-flash"),
    "zhipu": ("ZHIPU_API_KEY", "ZHIPU_BASE_URL", "glm-4.7-flash"),
    "dashscope": ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "qwen-plus"),
}


class LLMError(RuntimeError):
    """调用失败。message 里带人话解释，直接返回给调用方也不丢人。"""

    def __init__(self, message: str, *, retryable: bool = False, code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.code = code


RETRYABLE_CODES = {408, 409, 429, 500, 502, 503, 504}


def _explain(code: int) -> str:
    return {
        401: "密钥无效或过期，回控制台重新生成",
        402: "余额不足，需要充值",
        403: "无权限：百炼需先开通服务；智谱需完成实名认证",
        404: "base_url 不对，或模型名已失效（去查 /models）",
        429: "触发限流，免费模型通常并发为 1，改成串行调用",
    }.get(code, f"HTTP {code}")


def call_chat(
    s: Settings,
    messages: list[dict],
    *,
    system: str | None = None,
    provider: str = "deepseek",
    model: str | None = None,
    thinking: bool = False,
    max_tokens: int = 1024,
    timeout: float = 120.0,
    record_cost: bool = True,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> dict:
    """发一次 chat 调用。"""
    if provider not in PROVIDER_ENDPOINT:
        raise LLMError(f"未知 provider: {provider}，可选 {list(PROVIDER_ENDPOINT)}")

    if tools and thinking:
        raise LLMError("调工具时请关闭 thinking：思考模式下 tool_choice 会被服务端拒绝（400）")

    key_attr, url_attr, default_model = PROVIDER_ENDPOINT[provider]
    api_key = getattr(s, key_attr, "")
    base_url = getattr(s, url_attr, "")
    if not api_key:
        raise LLMError(f"{provider} 未配置 API key，检查 .env 里的 {key_attr}")
    model = model or default_model

    payload: dict = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + list(messages),
        "max_tokens": max_tokens,
        "stream": False,
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")[:400]
        raise LLMError(
            f"{provider} 返回 HTTP {e.code}：{_explain(e.code)}　{raw}",
            retryable=e.code in RETRYABLE_CODES, code=e.code,
        ) from None
    except urllib.error.URLError as e:
        raise LLMError(f"{provider} 网络不可达或超时：{e.reason}", retryable=True) from None
    except TimeoutError as e:
        raise LLMError(f"{provider} 请求超时：{e}", retryable=True) from None
    except Exception as e:  # noqa: BLE001
        raise LLMError(f"{provider} 调用异常：{type(e).__name__} {e}") from None
    latency = round(time.time() - t0, 3)

    try:
        choice = body["choices"][0]
        msg = choice["message"]
        usage = body.get("usage") or {}
    except (KeyError, IndexError):
        raise LLMError(f"{provider} 返回结构异常：{str(body)[:300]}") from None

    tin = usage.get("prompt_tokens", 0)
    tout = usage.get("completion_tokens", 0)
    hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

    result = {
        "content": msg.get("content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason", ""),
        "reasoning_content": msg.get("reasoning_content") or "",
        "usage": {"in": tin, "out": tout, "cached": hit},
        "latency_s": latency,
        "provider": provider,
        "model": model,
    }

    if record_cost:
        cny = estimate_cny(model, tin, tout, hit)
        result["cost_cny"] = cny
        append_entry(s, {
            "provider": provider, "model": model,
            "in": tin, "out": tout, "cached": hit,
            "latency_s": latency, "cost_cny": cny,
        })

    return result
