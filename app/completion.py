"""把模型输出变成结构化对象 —— /chat 的核心逻辑，W1 第 2 天（9/21）的产出。"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .config import Settings
from .llm import LLMError, call_chat
from .prompts import build_messages, build_repair_message
from .schemas import AnswerOut

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def strip_fence(text: str) -> str:
    """第 1 步：剥掉代码围栏。"""
    t = text.strip()
    m = _FENCE.search(t)
    if m:
        return m.group(1).strip()
    return t


def parse_structured(raw: str) -> AnswerOut:
    """第 2 步：解析并校验。失败就抛，由上层决定要不要重试。"""
    cleaned = strip_fence(raw)

    if not cleaned.startswith("{"):
        i, j = cleaned.find("{"), cleaned.rfind("}")
        if i >= 0 and j > i:
            cleaned = cleaned[i:j + 1]

    return AnswerOut.model_validate_json(cleaned)


class BudgetExceeded(RuntimeError):
    """本地成本闸门已到，拒绝继续调用。"""


def structured_chat(
    s: Settings,
    *,
    message: str,
    system: str | None = None,
    role: str | None = None,
    examples: list[tuple[str, str]] = (),
    use_xml: bool = True,
    thinking: bool = False,
    max_tokens: int = 800,
    provider: str = "deepseek",
    model: str | None = None,
    retries: int = 1,
    enforce_budget: bool = True,
) -> dict:
    """走完上面四个步骤，返回 /chat 需要的字典。"""
    from .cost import summarize

    if enforce_budget and summarize(s)["over_limit"]:
        raise BudgetExceeded(
            f"本地成本已到闸门 ¥{s.COST_LIMIT_CNY}，"
            f"调高 .env 里的 COST_LIMIT_CNY 或先把账本归档"
        )

    sys_text, messages = build_messages(
        message, system=system, role=role, examples=examples, use_xml=use_xml
    )

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    cost_total = 0.0
    latency_total = 0.0
    last_raw = ""
    last_model = ""
    last_error = ""
    used = 0

    for attempt in range(retries + 1):

        extra = []
        if attempt > 0:
            extra.append(build_repair_message(last_error, last_raw))

        res = call_chat(
            s,
            messages + extra,
            system=sys_text,
            provider=provider,
            model=model,
            thinking=thinking,
            max_tokens=max_tokens,
        )

        last_raw = res["content"]
        last_model = res["model"]
        usage_total["prompt_tokens"] += res["usage"]["in"]
        usage_total["completion_tokens"] += res["usage"]["out"]
        usage_total["cached_tokens"] += res["usage"]["cached"]
        cost_total += res.get("cost_cny") or 0.0
        latency_total += res["latency_s"]
        used = attempt

        try:
            data = parse_structured(last_raw)
            return {
                "ok": True,
                "data": data,
                "raw": last_raw,
                "retries_used": used,
                "validation_error": "",
                "provider": provider,
                "model": last_model,
                "usage": usage_total,
                "cost_cny": round(cost_total, 6),
                "latency_s": round(latency_total, 3),
            }
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_error = str(e)[:500]
            if attempt == retries:
                return {
                    "ok": False,
                    "data": None,
                    "raw": last_raw,
                    "retries_used": used,
                    "validation_error": last_error,
                    "provider": provider,
                    "model": last_model,
                    "usage": usage_total,
                    "cost_cny": round(cost_total, 6),
                    "latency_s": round(latency_total, 3),
                }

    raise LLMError("unreachable")
