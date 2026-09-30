"""RAG 生成端的请求/响应模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..retrieval.schemas import MODES


class CitationOut(BaseModel):
    n: int = Field(description="上下文里的编号，模型引用的就是这个")
    chunk_id: str
    source: str = ""
    title: str = ""
    path: str = ""
    quote: str = Field("", description="被引用的原文前若干字，方便直接核对")


class AskIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: str = Field("hybrid", description="检索模式：" + " / ".join(MODES))
    topk: int = Field(5, ge=1, le=20, description="喂给模型的条数")
    pool: int = Field(50, ge=5, le=200)
    min_score: float | None = Field(
        None, ge=0.0, description="拒答阈值：最高检索分低于它就不再问模型。"
        "不给则用配置里的默认值")
    with_context: bool = Field(False, description="是否在响应里回传喂给模型的原文")


class AskBatchIn(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=100)
    mode: str = Field("hybrid", description="检索模式：" + " / ".join(MODES))
    topk: int = Field(5, ge=1, le=20)
    pool: int = Field(50, ge=5, le=200)
    concurrency: int = Field(4, ge=1, le=16,
                             description="并发上限。打太狠会被限流，反而更慢")


class AskOut(BaseModel):
    ok: bool = True
    query: str
    answer: str = ""
    abstained: bool = Field(False, description="是否选择了拒答")
    abstain_reason: str = Field("", description="拒答原因：检索不够格 / 模型说资料不足")
    citations: list[CitationOut] = Field(default_factory=list)
    used: list[int] = Field(default_factory=list, description="模型声称用到的编号")
    unsupported: list[int] = Field(default_factory=list,
                                  description="标了但编号越界/不存在 —— 说明它在瞎标")
    context: list[str] = Field(default_factory=list)
    top_score: float = Field(0.0, description="最高检索分，用来判断该不该拒答")
    mode: str = ""
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    cost_cny: float = 0.0
