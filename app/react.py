"""手写 ReAct —— W4 块 A 的核心资产。"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from .config import Settings
from .cost import summarize
from .llm import LLMError, call_chat
from .tools import TOOL_SCHEMAS, dispatch


class BudgetExceeded(RuntimeError):
    """本地成本闸门已到，拒绝继续跑。"""


MAX_UNKNOWN_IN_A_ROW = 2

OBSERVATION_MAX_CHARS = 1500


REACT_SYSTEM = """你是一个可以调用工具完成任务的企业服务助手。

你必须严格按下面的格式输出，一轮只输出其中一种，不要输出任何格式之外的内容。

格式一（还需要更多信息时）：
思考：<一句话说明你下一步要做什么、为什么>
行动：<工具名> {"参数名": 值}

格式二（信息已经够了时）：
思考：<一句话说明为什么够了>
最终答案：<给用户的完整回答>

六条规矩：
1. 一次只调一个工具。拿到「观察」结果后再决定下一步，不要一口气把后面几步都规划完。
2. 工具名和参数必须分开：工具名是英文标识符，参数是一个完整的 JSON 对象，写在花括号里。
3. 工具返回 error 时不要原样重试同一个参数，换参数或换工具；试两次还不行就在最终答案里如实说明拿不到数据。
4. 信息不足时不要猜。要么再查一次，要么在最终答案里说清楚缺什么、需要谁来补。
5. 不要用「最终答案」输出中间过程的话 —— 一旦写了最终答案，任务就结束了。
6. 只能用下面列出的工具，不要发明不存在的工具名。
"""


def describe_tools(schemas: list[dict]) -> str:
    """把工具 schema 翻译成一份给模型看的纯文本清单。"""
    lines = []
    for t in schemas:
        fn = t.get("function", {})
        params = (fn.get("parameters") or {}).get("properties", {})
        param_desc = "，".join(
            f'{k}（{v.get("description", "")}）' for k, v in params.items()
        )
        lines.append(f'- {fn.get("name", "")}：{fn.get("description", "")}')
        if param_desc:
            lines.append(f'    参数：{param_desc}')
    return "\n".join(lines)


def tool_names(schemas: list[dict]) -> list[str]:
    """取工具名列表。用于在提示词和错误消息里告诉模型「你只能选这些」。"""
    return [t.get("function", {}).get("name", "") for t in schemas]


_ACT_RE = re.compile(r"\*{0,2}行动\*{0,2}\s*[:：]\s*", re.S)
_FIN_RE = re.compile(r"\*{0,2}最终答案\*{0,2}\s*[:：]\s*", re.S)
_THOUGHT_RE = re.compile(
    r"\*{0,2}思考\*{0,2}\s*[:：]\s*(.*?)"
    r"(?=\*{0,2}(?:行动|最终答案)\*{0,2}\s*[:：]|\Z)",
    re.S,
)


def parse_action(text: str) -> dict:
    """解析模型的一轮输出，返回三态之一。"""
    text = text or ""

    m_thought = _THOUGHT_RE.search(text)
    thought = (m_thought.group(1).strip() if m_thought else "")

    m_act = _ACT_RE.search(text)
    m_fin = _FIN_RE.search(text)

    if m_act is None and m_fin is None:
        return {"kind": "unknown", "thought": thought, "tool": "", "args": "",
                "answer": "", "raw": text.strip()}

    use_final = m_fin is not None and (m_act is None or m_fin.start() < m_act.start())

    if use_final:
        answer = text[m_fin.end():].strip()
        return {"kind": "final", "thought": thought, "tool": "", "args": "",
                "answer": answer, "raw": text.strip()}

    payload = text[m_act.end():].strip()
    tool, args = _split_tool_call(payload)
    if tool == "":
        return {"kind": "unknown", "thought": thought, "tool": "", "args": "",
                "answer": "", "raw": text.strip()}
    return {"kind": "action", "thought": thought, "tool": tool, "args": args,
            "answer": "", "raw": text.strip()}


def _split_tool_call(payload: str) -> tuple[str, str]:
    """从「行动：」后面的内容里拆出 (工具名, 参数 JSON 字符串)。"""
    if not payload:
        return "", ""

    m = re.search(r"[A-Za-z_][A-Za-z0-9_]*", payload)
    if m is None:
        return "", ""
    name = m.group(0)

    i = payload.find("{")
    j = payload.rfind("}")
    args = payload[i:j + 1] if (i >= 0 and j > i) else ""
    return name, args


def format_observation(tool: str, result: Any, error: str,
                       max_chars: int = OBSERVATION_MAX_CHARS) -> str:
    """把工具结果拼成一条「观察」文本，回灌给模型。"""
    if error:
        return f"观察：调用 {tool} 失败 —— {error}"

    try:
        body = json.dumps(result, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        body = str(result)

    if len(body) > max_chars:
        body = body[:max_chars] + f"…（已截断，原长 {len(body)} 字。如需完整结果请缩小查询范围）"
    return f"观察：{body}"


_FORMAT_HINT = (
    "你上一轮的输出没有按格式写，我无法执行。"
    "请严格按下面两种之一重新输出，不要加别的文字：\n"
    "思考：<一句话>\n行动：<工具名> {\"参数名\": 值}\n"
    "或\n"
    "思考：<一句话>\n最终答案：<回答>"
)


def run_react(
    s: Settings,
    task: str,
    *,
    system: str | None = None,
    tools: list[dict] | None = None,
    max_steps: int = 6,
    provider: str = "deepseek",
    model: str | None = None,
    max_tokens: int = 1024,
    enforce_budget: bool = True,
    record_cost: bool = True,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """跑一次完整的 ReAct 循环（纯文本协议，不传 tools）。"""

    def emit(ev: dict) -> None:
        if on_event:
            on_event(ev)

    if enforce_budget and summarize(s)["over_limit"]:
        raise BudgetExceeded(
            f"本地成本已到闸门 ¥{s.COST_LIMIT_CNY}，"
            f"调高 .env 里的 COST_LIMIT_CNY 或先把账本归档"
        )

    schemas = tools if tools is not None else TOOL_SCHEMAS
    allowed = tool_names(schemas)

    system_text = (
        (system or REACT_SYSTEM)
        + "\n\n你当前可用的工具只有下面这些（不要调用列表之外的名字）：\n"
        + describe_tools(schemas)
    )

    messages: list[dict] = [{"role": "user", "content": task}]
    trace: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    cost_total = 0.0
    latency_total = 0.0
    last_model = ""
    unknown_streak = 0

    for step in range(1, max_steps + 1):
        emit({"type": "step_start", "step": step})

        res = call_chat(
            s,
            messages,
            system=system_text,
            provider=provider,
            model=model,
            thinking=False,
            max_tokens=max_tokens,
            record_cost=record_cost,
        )

        last_model = res.get("model", "")
        usage["prompt_tokens"] += res["usage"]["in"]
        usage["completion_tokens"] += res["usage"]["out"]
        usage["cached_tokens"] += res["usage"]["cached"]
        cost_total += res.get("cost_cny") or 0.0
        latency_total += res.get("latency_s") or 0.0

        text = res.get("content") or ""
        parsed = parse_action(text)
        thought = parsed["thought"]
        if thought:
            emit({"type": "thought", "step": step, "thought": thought})

        messages.append({"role": "assistant", "content": text})

        if parsed["kind"] == "final":
            emit({"type": "final", "step": step, "answer": parsed["answer"]})
            return {
                "ok": True,
                "answer": parsed["answer"],
                "trace": trace,
                "steps": step,
                "tool_calls_made": len([t for t in trace if t.get("tool")]),
                "stop_reason": "final_answer",
                "provider": provider,
                "model": last_model,
                "usage": usage,
                "cost_cny": round(cost_total, 6),
                "latency_s": round(latency_total, 3),
            }

        if parsed["kind"] == "unknown":
            unknown_streak += 1
            trace.append({"step": step, "thought": thought, "tool": "", "args": "",
                          "result": None, "error": "输出未按格式，无法解析"})
            emit({"type": "stop", "step": step, "reason": "unparsable_format"})
            if unknown_streak >= MAX_UNKNOWN_IN_A_ROW:
                return {
                    "ok": False,
                    "answer": "",
                    "trace": trace,
                    "steps": step,
                    "tool_calls_made": len([t for t in trace if t.get("tool")]),
                    "stop_reason": "unparsable_format",
                    "provider": provider,
                    "model": last_model,
                    "usage": usage,
                    "cost_cny": round(cost_total, 6),
                    "latency_s": round(latency_total, 3),
                }
            messages.append({"role": "user", "content": _FORMAT_HINT})
            continue

        unknown_streak = 0
        tool, raw = parsed["tool"], parsed["args"]
        emit({"type": "action", "step": step, "tool": tool, "args": raw})

        if tool not in allowed:
            error = f"没有名为 '{tool}' 的工具。本次可用工具只有：{allowed}。请从中选一个。"
            result: Any = None
        else:
            result, error = dispatch(tool, raw)

        observation = format_observation(tool, result, error)
        trace.append({"step": step, "thought": thought, "tool": tool, "args": raw,
                      "result": result, "error": error})
        emit({"type": "observation", "step": step, "tool": tool,
              "result": result, "error": error})

        messages.append({"role": "user", "content": observation})

    emit({"type": "stop", "step": max_steps, "reason": "max_steps_reached"})
    return {
        "ok": False,
        "answer": "",
        "trace": trace,
        "steps": max_steps,
        "tool_calls_made": len([t for t in trace if t.get("tool")]),
        "stop_reason": "max_steps_reached",
        "provider": provider,
        "model": last_model,
        "usage": usage,
        "cost_cny": round(cost_total, 6),
        "latency_s": round(latency_total, 3),
    }


__all__ = ["run_react", "parse_action", "format_observation", "describe_tools",
           "tool_names", "BudgetExceeded", "MAX_UNKNOWN_IN_A_ROW", "LLMError"]
