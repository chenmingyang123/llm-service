"""超时重试与 provider 降级 —— W1 第 4 天的第二块。"""
from __future__ import annotations

import random
import time

from .config import Settings
from .llm import LLMError, PROVIDER_ENDPOINT, call_chat

FALLBACK_CHAIN = ["deepseek", "zhipu", "dashscope"]


def call_with_retry(
    fn,
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.3,
    on_retry=None,
):
    """只重试可重试的错误。不可重试的立刻抛出，一次都不多试。"""
    last: LLMError | None = None
    for i in range(attempts):
        try:
            return fn()
        except LLMError as e:
            last = e
            if not e.retryable or i == attempts - 1:
                raise
            delay = min(base_delay * (2 ** i), max_delay)
            delay = max(delay * (1 + random.uniform(-jitter, jitter)), 0.05)
            if on_retry:
                on_retry(i + 1, e, round(delay, 3))
            time.sleep(delay)
    raise last  # pragma: no cover —— attempts>=1 时不会走到这里


def _has_key(s: Settings, provider: str) -> bool:
    attr = PROVIDER_ENDPOINT.get(provider, ("", "", ""))[0]
    return bool(attr and getattr(s, attr, ""))


def call_with_fallback(
    s: Settings,
    messages: list[dict],
    *,
    providers: list[str] | None = None,
    attempts_per_provider: int = 2,
    system: str | None = None,
    model: str | None = None,
    **kw,
) -> dict:
    """按 FALLBACK_CHAIN 依次尝试，第一个成功的就是结果。"""
    chain = providers or FALLBACK_CHAIN
    attempts: list[dict] = []
    last_err: LLMError | None = None

    for i, p in enumerate(chain):
        if not _has_key(s, p):
            attempts.append({"provider": p, "status": "skipped", "reason": "未配置 API key"})
            continue

        log: list[dict] = []

        def on_retry(n, e, delay, _p=p, _log=log):
            _log.append({"retry": n, "error": str(e)[:160], "delay_s": delay})

        try:
            res = call_with_retry(
                lambda: call_chat(s, messages, system=system, provider=p,
                                  model=(model if i == 0 else None), **kw),
                attempts=attempts_per_provider,
                on_retry=on_retry,
            )
        except LLMError as e:
            last_err = e
            attempts.append({"provider": p, "status": "failed",
                             "error": str(e)[:200], "retryable": e.retryable, "retries": log})
            continue

        attempts.append({"provider": p, "status": "ok", "retries": log})
        return {**res, "degraded": i > 0, "attempts": attempts}

    detail = "; ".join(f"{a['provider']}={a['status']}" for a in attempts)
    raise LLMError(f"全部 provider 都失败了（{detail}）", retryable=False)
