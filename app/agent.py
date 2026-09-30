"""工具调用主循环 —— W1 第 3 天（9/22）的核心资产，也是整个 W1 最该吃透的一个文件。"""
from __future__ import annotations

import json
from typing import Callable

from .config import Settings
from .cost import summarize
from .guards import GateResult, execute, gate
from .llm import LLMError, call_chat
from .tools import KNOWN, TOOL_SCHEMAS, TOOL_SYSTEM, dispatch


class BudgetExceeded(RuntimeError):
    """本地成本闸门已到，拒绝继续调用。"""


def run_tool_loop(
    s: Settings,
    question: str,
    *,
    system: str | None = None,
    provider: str = "deepseek",
    model: str | None = None,
    max_turns: int = 5,
    tool_choice: str | dict | None = "auto",
    max_tokens: int = 1024,
    enforce_budget: bool = True,
    record_cost: bool = True,
    tools: list[dict] | None = None,
    on_event: Callable[[dict], None] | None = None,
    allowed_tools: list[str] | None = None,
    confirmed: set[str] | None = None,
    confirm_registry=None,
) -> dict:
    """跑一次完整的工具调用循环。"""

    def emit(ev: dict) -> None:
        if on_event:
            on_event(ev)

    if enforce_budget and summarize(s)["over_limit"]:
        raise BudgetExceeded(
            f"本地成本已到闸门 ¥{s.COST_LIMIT_CNY}，"
            f"调高 .env 里的 COST_LIMIT_CNY 或先把账本归档"
        )

    schemas = tools if tools is not None else TOOL_SCHEMAS

    messages: list[dict] = [{"role": "user", "content": question}]
    trace: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    cost_total = 0.0
    latency_total = 0.0
    last_model = ""

    for turn in range(1, max_turns + 1):

        res = call_chat(
            s,
            messages,
            system=system or TOOL_SYSTEM,
            provider=provider,
            model=model,
            thinking=False,
            tools=schemas,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            record_cost=record_cost,
        )

        last_model = res["model"]
        usage["prompt_tokens"] += res["usage"]["in"]
        usage["completion_tokens"] += res["usage"]["out"]
        usage["cached_tokens"] += res["usage"]["cached"]
        cost_total += res.get("cost_cny") or 0.0
        latency_total += res["latency_s"]

        calls = res["tool_calls"]

        assistant_msg: dict = {"role": "assistant", "content": res["content"] or ""}
        if calls:
            assistant_msg["tool_calls"] = calls
        messages.append(assistant_msg)

        if not calls:
            emit({"type": "final", "answer": res["content"], "turn": turn})
            return {
                "ok": True,
                "answer": res["content"],
                "trace": trace,
                "turns": turn,
                "tool_calls_made": len(trace),
                "stop_reason": "no_tool_calls",
                "provider": provider,
                "model": last_model,
                "usage": usage,
                "cost_cny": round(cost_total, 6),
                "latency_s": round(latency_total, 3),
            }

        emit({"type": "tool_start", "turn": turn,
              "tools": [(c.get("function") or {}).get("name", "") for c in calls]})

        for tc in calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw = fn.get("arguments", "")

            use_guards = (confirmed is not None) or (confirm_registry is not None)

            if not use_guards:
                result, error = dispatch(name, raw)
            else:
                gr = gate(name, raw, allowed=allowed_tools, confirmed=confirmed)

                if (not gr.allow) and gr.need_confirm and confirm_registry is not None:
                    cid = confirm_registry.open(tool=name, args=raw,
                                                request_id=gr.request_id)
                    emit({"type": "need_confirm", "turn": turn, "confirm_id": cid,
                          "tool": name, "args": raw, "request_id": gr.request_id,
                          "reason": gr.reason})
                    approved = confirm_registry.wait(cid)
                    emit({"type": "confirm_result", "turn": turn, "confirm_id": cid,
                          "approved": approved, "tool": name})
                    if approved and gr.request_id:
                        if confirmed is None:
                            confirmed = set()
                        confirmed.add(gr.request_id)
                        gr = gate(name, raw, allowed=allowed_tools, confirmed=confirmed)
                    else:
                        gr = GateResult(False, reason=(
                            "该写操作未获人工确认（被拒绝或超时），未产生任何副作用。"
                            "请不要重试同一个动作；如需继续，请让用户重新发起。"))

                if gr.allow:
                    result, error = execute(gr), ""
                else:
                    result, error = None, gr.reason

            item = {"turn": turn, "tool": name, "args": raw, "result": result, "error": error}
            trace.append(item)
            emit({"type": "tool_done", **item})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": json.dumps(
                    result if error == "" else {"error": error, "可用工具": KNOWN},
                    ensure_ascii=False,
                ),
            })

    return {
        "ok": False,
        "answer": "",
        "trace": trace,
        "turns": max_turns,
        "tool_calls_made": len(trace),
        "stop_reason": "max_turns_reached",
        "provider": provider,
        "model": last_model,
        "usage": usage,
        "cost_cny": round(cost_total, 6),
        "latency_s": round(latency_total, 3),
    }


def run_prompt_style(
    s: Settings,
    question: str,
    *,
    provider: str = "deepseek",
    model: str | None = None,
    max_tokens: int = 1024,
) -> dict:
    """提示词版软实现（教程 10.2 的路径）—— 不依赖原生 function calling。"""
    import re

    described = "\n\n".join(
        f"- {t['function']['name']}：{t['function']['description']}\n"
        f"  参数：{json.dumps(t['function']['parameters'], ensure_ascii=False)}"
        for t in TOOL_SCHEMAS
    )
    system = (
        f"{TOOL_SYSTEM}\n\n你有以下工具：\n{described}\n\n"
        "需要调用工具时，只输出下面这段 XML，不要输出别的内容：\n"
        "<tool_request>\n  <tool_name>工具名</tool_name>\n  <parameters>{\"参数\": 值}</parameters>\n</tool_request>\n"
        "不需要工具时，直接正常回答。"
    )

    res = call_chat(
        s, [{"role": "user", "content": question}],
        system=system, provider=provider, model=model,
        thinking=False, max_tokens=max_tokens,
    )
    text = res["content"]

    m = re.search(r"<tool_name>(.*?)</tool_name>\s*<parameters>(.*?)</parameters>", text, re.S)
    if not m:
        return {"ok": True, "answer": text, "tool_call": None,
                "cost_cny": res.get("cost_cny", 0.0), "latency_s": res["latency_s"]}

    name, raw = m.group(1).strip(), m.group(2).strip()
    result, error = dispatch(name, raw)
    return {"ok": True, "answer": text,
            "tool_call": {"tool": name, "args": raw, "result": result, "error": error},
            "cost_cny": res.get("cost_cny", 0.0), "latency_s": res["latency_s"]}


__all__ = ["run_tool_loop", "run_prompt_style", "BudgetExceeded", "LLMError"]
