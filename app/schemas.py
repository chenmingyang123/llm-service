"""响应模型。用 Pydantic 定义，OpenAPI 文档自动就有 schema —— 白拿的资产。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthOut(BaseModel):
    """存活探针：只回答「进程还活着吗」，不查任何外部依赖。"""
    status: str = Field("ok", examples=["ok"])
    service: str
    version: str
    env: str
    uptime_s: float
    checked_at: str


class ProviderOut(BaseModel):
    key: str
    name: str
    role: str
    configured: bool
    status: str
    base_url: str
    models: list[str] = []
    latency_s: float | None = None
    detail: str = ""


class CostOut(BaseModel):
    total_cny: float
    calls: int
    limit_cny: float
    remaining_cny: float
    over_limit: bool
    by_provider: dict[str, float] = {}
    by_model: dict[str, float] = {}


class ReadyOut(BaseModel):
    """就绪探针：查依赖。?ping=1 时才真实探活（只打 /models，不花钱）。"""
    status: str = Field(examples=["ready", "degraded", "not_ready"])
    service: str
    version: str
    uptime_s: float
    providers: list[ProviderOut]
    cost: CostOut
    checked_at: str
    probed: bool = False
    detail: Any = ""


class Example(BaseModel):
    """一条 few-shot 样例。第 7 章的核心变量，今晚的实验围绕它做。"""
    input: str = Field(description="用户侧输入")
    output: str = Field(description="期望的 JSON 字符串输出")


class ChatIn(BaseModel):
    message: str = Field(description="用户问题")
    system: str | None = Field(None, description="追加到系统提示的自定义指令")
    role: str | None = Field(None, description="第 3 章：角色提示")
    examples: list[Example] = Field(default_factory=list, description="第 7 章：few-shot 样例")
    use_xml: bool = Field(True, description="第 4 章：是否用 XML 标签包裹用户输入")
    thinking: bool = Field(False, description="打开会让输出 token 成倍增长，默认关")
    max_tokens: int = 800
    provider: str = "deepseek"
    model: str | None = None
    retries: int = Field(1, description="结构化校验失败后的修复重试次数")


class AnswerOut(BaseModel):
    """模型必须满足的输出契约。校验不通过就触发修复重试。"""
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class UsageOut(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class ChatOut(BaseModel):
    ok: bool
    data: AnswerOut | None = None
    raw: str = ""
    retries_used: int = 0
    validation_error: str = ""
    provider: str = ""
    model: str = ""
    usage: UsageOut = UsageOut()
    cost_cny: float = 0.0
    latency_s: float = 0.0


class TraceItem(BaseModel):
    """一次工具调用的全过程。排障和演示都靠它，字段别省。"""
    turn: int
    tool: str
    args: str
    result: Any | None = None
    error: str = ""


class AgentIn(BaseModel):
    question: str = Field(description="用户问题")
    max_turns: int = Field(5, ge=1, le=10, description="循环熔断上限，超过就停")
    tool_choice: str = Field("auto", description="auto / required / none")
    provider: str = "deepseek"
    model: str | None = None
    max_tokens: int = 1024


class BatchIn(BaseModel):
    prompts: list[str] = Field(description="一批 prompt")
    system: str | None = None
    concurrency: int = Field(8, ge=1, le=64, description="并发上限，超出的排队而不是一起冲")
    use_fallback: bool = Field(False, description="是否启用 provider 降级链")
    provider: str = "deepseek"
    max_tokens: int = 512


class AgentOut(BaseModel):
    ok: bool
    answer: str = ""
    trace: list[TraceItem] = Field(default_factory=list)
    turns: int = 0
    tool_calls_made: int = 0
    stop_reason: str = ""
    provider: str = ""
    model: str = ""
    usage: UsageOut = UsageOut()
    cost_cny: float = 0.0
    latency_s: float = 0.0


class ReactTraceItem(BaseModel):
    """ReAct 的一步。"""
    step: int
    thought: str = ""
    tool: str = ""
    args: str = ""
    result: Any | None = None
    error: str = ""


class ConfirmIn(BaseModel):
    """人工确认的回执（配合 /agent/stream 的 need_confirm 事件使用）。"""
    confirm_id: str = Field(description="need_confirm 事件里带回来的编号",
                            min_length=4)
    approve: bool = Field(description="True=放行执行；False=拒绝（不产生任何副作用）")


class ConfirmOut(BaseModel):
    ok: bool
    confirm_id: str = ""
    approved: bool = False
    detail: str = ""


class ReactIn(BaseModel):
    question: str = Field(description="用户任务")
    max_steps: int = Field(6, ge=1, le=10, description="ReAct 步数上限，超过就停并如实说没做完")
    provider: str = "deepseek"
    model: str | None = None
    max_tokens: int = 1024


class ReactOut(BaseModel):
    ok: bool
    answer: str = ""
    trace: list[ReactTraceItem] = Field(default_factory=list)
    steps: int = 0
    tool_calls_made: int = 0
    stop_reason: str = ""
    provider: str = ""
    model: str = ""
    usage: UsageOut = UsageOut()
    cost_cny: float = 0.0
    latency_s: float = 0.0
