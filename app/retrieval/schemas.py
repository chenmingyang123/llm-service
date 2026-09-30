"""检索相关的请求/响应模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


class RetrieveIn(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="用户问题")
    mode: str = Field("hybrid", description="检索模式：" + " / ".join(MODES))
    topk: int = Field(5, ge=1, le=50, description="最终返回几条")
    pool: int = Field(50, ge=5, le=200,
                      description="粗排候选池大小；rerank 在这个池子里精排")
    with_text: bool = Field(True, description="是否回传正文（关闭可省带宽）")


class HitOut(BaseModel):
    chunk_id: str
    score: float = Field(description="融合后的分数（RRF 分数，不是余弦值）")
    source: str = ""
    title: str = ""
    path: str = Field("", description="章节路径，用于溯源")
    from_parent: bool = Field(False, description="正文取自父块（small-to-big）")
    text: str = ""
    vector_score: float | None = None
    bm25_score: float | None = None


class RetrieveOut(BaseModel):
    ok: bool = True
    query: str
    mode: str
    hits: list[HitOut] = Field(default_factory=list)
    reranked: bool = False
    latency_ms: float = 0.0
    note: str = ""
