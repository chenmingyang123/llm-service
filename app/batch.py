"""异步并发批量调用 —— W1 第 4 天的第三块。"""
from __future__ import annotations

import asyncio
import time

from .config import Settings
from .llm import LLMError, call_chat
from .resilience import call_with_fallback

DEFAULT_CONCURRENCY = 8


async def run_batch(
    s: Settings,
    prompts: list[str],
    *,
    system: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    use_fallback: bool = False,
    **kw,
) -> dict:
    """并发跑一批 prompt，返回逐条结果与整体统计。"""
    sem = asyncio.Semaphore(max(concurrency, 1))

    async def one(prompt: str) -> dict:
        async with sem:
            t0 = time.time()
            try:
                if use_fallback:
                    res = await asyncio.to_thread(
                        call_with_fallback, s,
                        [{"role": "user", "content": prompt}],
                        system=system, **kw)
                else:
                    res = await asyncio.to_thread(
                        call_chat, s,
                        [{"role": "user", "content": prompt}],
                        system=system, **kw)
            except LLMError as e:
                return {"ok": False, "prompt": prompt, "error": str(e)[:200],
                        "latency_s": round(time.time() - t0, 3)}
            return {
                "ok": True,
                "prompt": prompt,
                "content": res.get("content", ""),
                "degraded": res.get("degraded", False),
                "provider": res.get("provider", ""),
                "model": res.get("model", ""),
                "cost_cny": res.get("cost_cny", 0.0),
                "latency_s": round(time.time() - t0, 3),
            }

    t0 = time.time()
    items = await asyncio.gather(*[one(p) for p in prompts])
    wall = round(time.time() - t0, 3)

    ok_items = [i for i in items if i["ok"]]
    serial = round(sum(i["latency_s"] for i in items), 3)
    return {
        "count": len(items),
        "succeeded": len(ok_items),
        "failed": len(items) - len(ok_items),
        "concurrency": concurrency,
        "wall_s": wall,
        "serial_equivalent_s": serial,
        "speedup": round(serial / wall, 2) if wall > 0 else 0.0,
        "cost_cny": round(sum(i.get("cost_cny", 0.0) for i in ok_items), 6),
        "degraded_count": sum(1 for i in ok_items if i.get("degraded")),
        "items": items,
    }
