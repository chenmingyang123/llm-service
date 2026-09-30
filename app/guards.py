"""护栏（四道闸门）—— W4 块 C 的核心资产。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .tools import ARG_MODELS, TOOL_RISK, TOOL_SCHEMAS, _IDEMPOTENCY


FAILURE_MODES: list[dict] = [
    {"模式": "循环不收敛", "表现": "反复调同一工具，参数每次微调，停不下来",
     "兜底": "max_steps 硬上限 + 相同参数指纹去重"},
    {"模式": "工具选择错误", "表现": "该查订单的去查了文档",
     "兜底": "schema 描述里写清「什么时候不该用」+ 任务级工具子集"},
    {"模式": "参数幻觉", "表现": "编出不存在的参数名 / 非法 JSON",
     "兜底": "Pydantic 校验 + 把错误消息当观察回灌让它自己改"},
    {"模式": "越权动作", "表现": "直接调了写类工具，改了真实数据",
     "兜底": "人工确认令牌 + 工具白名单（闸门 ②③）"},
    {"模式": "重复副作用", "表现": "重试导致建了两张工单、发了两条通知",
     "兜底": "request_id 幂等（闸门 ④）"},
    {"模式": "上下文爆炸", "表现": "工具返回整张表，几轮后窗口塞满，模型「突然变笨」",
     "兜底": "观察回灌前统一截断"},
    {"模式": "成本失控", "表现": "一次任务几十次调用，账单炸了",
     "兜底": "成本闸门 + 单任务预算上限 + 熔断器"},
]


MAX_TASK_CHARS = 2000

_INJECTION_PATTERNS = [
    re.compile(r"忽略(以上|上面|之前)(的)?(所有)?指令", re.I),
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"你现在是|你现在扮演|system\s*:", re.I),
    re.compile(r"不要(告诉|通知)用户", re.I),
]


def validate_input(task: str) -> tuple[bool, str]:
    """闸门 ①：任务描述是否可受理。返回 (ok, reason)。"""
    if task is None or not str(task).strip():
        return False, "任务描述为空"
    if len(task) > MAX_TASK_CHARS:
        return False, f"任务描述过长（{len(task)} > {MAX_TASK_CHARS} 字），请精简后再提交"
    for pat in _INJECTION_PATTERNS:
        if pat.search(task):
            return False, (
                "任务描述包含疑似提示注入的内容（如「忽略以上指令」）。"
                "这类指令不会被当作系统指令执行，请改写为正常描述。"
            )
    return True, ""


@dataclass
class GateResult:
    """闸门判定结果。"""
    allow: bool
    reason: str = ""
    tool: str = ""
    args: Any = None
    result: Any = None
    duplicate: bool = False
    need_confirm: bool = False
    request_id: str = ""


def gate(tool_name: str, raw_arguments: str, *,
         allowed: Iterable[str] | None = None,
         confirmed: set[str] | None = None) -> GateResult:
    """调用前的四道判定。"""
    allowed_set = set(allowed) if allowed is not None else {
        t["function"]["name"] for t in TOOL_SCHEMAS}

    if tool_name not in allowed_set:
        return GateResult(False, reason=(
            f"没有名为 '{tool_name}' 的工具，或它不在本次允许范围内。"
            f"本次可用：{sorted(allowed_set)}。请从中选一个。"))

    model = ARG_MODELS.get(tool_name)
    if model is None:
        return GateResult(False, reason=f"工具 '{tool_name}' 缺少参数模型，无法校验。")
    try:
        raw = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return GateResult(False, reason=f"参数不是合法 JSON（{e}）。格式必须是 {{\"参数名\": 值}}。")
    if not isinstance(raw, dict):
        return GateResult(False, reason=f"参数必须是 JSON 对象，收到的是 {type(raw).__name__}。")
    try:
        args = model.model_validate(raw)
    except Exception as e:  # noqa: BLE001  ValidationError
        return GateResult(False, reason=f"参数不合法：{e}。请按 schema 重新输出。")

    rid = getattr(args, "request_id", "") or ""

    if rid and rid in _IDEMPOTENCY:
        return GateResult(True, tool=tool_name, args=args,
                          result=_IDEMPOTENCY[rid], duplicate=True, request_id=rid,
                          reason="该 request_id 已执行过，直接返回首次结果（未重复执行）")

    level, need_confirm = TOOL_RISK.get(tool_name, ("read", False))
    if need_confirm:
        if not rid:
            return GateResult(False, reason=(
                f"'{tool_name}' 是写操作，必须带 request_id（幂等键），否则无法判断是否重复执行。"),
                need_confirm=True, tool=tool_name)
        if not confirmed or rid not in confirmed:
            return GateResult(False, reason=(
                f"'{tool_name}' 是写操作（不可逆），request_id={rid} 尚未获得人工确认。"
                f"请先让人确认后再调用。"), need_confirm=True, tool=tool_name, args=args,
                request_id=rid)

    return GateResult(True, tool=tool_name, args=args, need_confirm=need_confirm,
                      request_id=rid)


def execute(gr: GateResult) -> dict:
    """闸门放行后真正执行。"""
    if not gr.allow or gr.args is None:
        return {"error": gr.reason or "未通过闸门，拒绝执行"}
    if gr.duplicate:
        base = gr.result if isinstance(gr.result, dict) else {"result": gr.result}
        return {**base, "duplicate": True}
    from .tools import REGISTRY
    fn = REGISTRY.get(gr.tool)
    if fn is None:
        return {"error": f"工具 '{gr.tool}' 未实现"}
    try:
        return fn(gr.args)
    except Exception as e:  # noqa: BLE001
        return {"error": f"工具 '{gr.tool}' 执行时出错：{type(e).__name__}: {e}"}
    return fn(gr.args)


class CircuitBreaker:
    """横向熔断：连续失败或步数超限就跳闸。"""

    def __init__(self, max_failures: int = 3, max_steps: int = 6):
        self.max_failures = max_failures
        self.max_steps = max_steps
        self.failures = 0
        self.steps = 0
        self.tripped = False
        self.reason = ""

    def record_step(self) -> None:
        self.steps += 1
        if self.steps >= self.max_steps:
            self.trip(f"步数达到上限 {self.max_steps}")

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.max_failures:
            self.trip(f"连续失败 {self.failures} 次")

    def record_success(self) -> None:
        """成功要能重置失败计数 —— 否则「成功-失败-成功-失败」也会被误判成连续失败。"""
        self.failures = 0

    def trip(self, reason: str) -> None:
        self.tripped = True
        self.reason = reason

    @property
    def ok(self) -> bool:
        return not self.tripped

    def escalate_payload(self, task: str) -> dict:
        """跳闸后该做什么：转人工，而不是硬编一个答案。"""
        return {"escalated": True, "answer": "该问题已转人工处理。",
                "trip_reason": self.reason, "task": task}


__all__ = ["gate", "execute", "validate_input", "GateResult", "CircuitBreaker",
           "FAILURE_MODES", "MAX_TASK_CHARS"]
