"""W3 第一轮优化：三种可插拔的检索策略。"""
from __future__ import annotations

import json
import re
import time

from ..config import Settings, get_settings
from ..llm import LLMError, call_chat
from .pipeline import RetrievalPipeline
from .rerank import build_reranker

SOURCES = ("deepseek", "zhipu", "bailian")

VARIANT_PROMPT = """把下面这个问题改写成 {n} 个不同角度的检索查询，用于在文档库中检索。要求：
1. 每行一个，不要编号
2. 每个改写要聚焦问题的不同侧面（如果是对比题，要分别覆盖被对比的各方）
3. 保持原意，不要引入新问题

问题：{q}"""

HYDE_PROMPT = """假设你在回答下面这个问题，请直接写一段"可能的答案文档片段"。
要求：用文档说明书的口吻写，包含具体术语和参数，不要写"根据文档"，不要提问。

问题：{q}"""

ROUTE_PROMPT = """下面这个问题，是在问哪一家平台的文档？
只回答以下四个词之一，不要解释：deepseek / zhipu / bailian / general
（同时涉及多家、或看不出属于哪家时，回答 general）

问题：{q}"""

SELFQ_PROMPT = """把下面这个问题拆成"语义查询 + 结构化过滤条件"。
只输出一行 JSON，不要解释：
{{"semantic": "改写后的检索语句", "source": "deepseek|zhipu|bailian|null", "has_code": true|false|null}}

问题：{q}"""


def _lines(text: str) -> list[str]:
    out = []
    for line in (text or "").strip().splitlines():
        line = line.strip(" -*•\t")
        if line:
            out.append(line)
    return out


class MultiQueryRetriever:
    """一个查询 → N 个改写 → 各自检索 → 去重合并。"""

    def __init__(self, pipeline: RetrievalPipeline, n: int = 3,
                 provider: str = "deepseek", settings: Settings | None = None):
        self.pipeline = pipeline
        self.n = n
        self.provider = provider
        self.settings = settings or get_settings()
        self.spent = 0.0

    def variants(self, query: str) -> tuple[list[str], float]:
        """返回 (改写列表, 本次改写花的钱)。成本必须透出，否则策略的白花钱看不见。"""
        try:
            res = call_chat(self.settings,
                            [{"role": "user", "content": VARIANT_PROMPT.format(q=query, n=self.n)}],
                            provider=self.provider, thinking=False, max_tokens=200)
            return _lines(res["content"])[: self.n], (res.get("cost_cny") or 0.0)
        except LLMError:
            return [], 0.0

    def search(self, query: str, mode: str = "hybrid",
               topk: int = 5, pool: int = 50) -> dict:
        t0 = time.perf_counter()
        merged: dict[str, dict] = {}
        order: list[str] = []

        variants, cost = self.variants(query)
        self.spent += cost
        queries = [query] + variants

        scored: dict[str, float] = {}
        hitmap: dict[str, dict] = {}
        for q in queries:
            for rank, h in enumerate(self.pipeline.search(q, mode=mode, topk=topk, pool=pool)["hits"], 1):
                cid = h["chunk_id"]
                scored[cid] = scored.get(cid, 0.0) + 1.0 / (60 + rank)
                hitmap.setdefault(cid, h)

        order = sorted(scored, key=lambda c: -scored[c])
        hits = [hitmap[c] for c in order[:topk]]
        return {"query": query, "mode": f"multiquery/{mode}", "hits": hits,
                "variants": len(queries) - 1, "cost_cny": round(cost, 6),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


class HyDERetriever:
    """先让 LLM 写一段假设答案，用它的向量去检索。"""

    def __init__(self, pipeline: RetrievalPipeline, provider: str = "deepseek",
                 settings: Settings | None = None):
        self.pipeline = pipeline
        self.provider = provider
        self.settings = settings or get_settings()
        self.spent = 0.0

    def hypothesize(self, query: str) -> tuple[str, float]:
        try:
            res = call_chat(self.settings,
                            [{"role": "user", "content": HYDE_PROMPT.format(q=query)}],
                            provider=self.provider, thinking=False, max_tokens=300)
            return (res["content"] or "").strip(), (res.get("cost_cny") or 0.0)
        except LLMError:
            return "", 0.0

    def search(self, query: str, mode: str = "hybrid",
               topk: int = 5, pool: int = 50) -> dict:
        t0 = time.perf_counter()
        hypo, cost = self.hypothesize(query)
        self.spent += cost
        used = hypo if hypo else query
        r = self.pipeline.search(used, mode=mode, topk=topk, pool=pool)
        return {"query": query, "mode": f"hyde/{mode}", "hits": r["hits"],
                "hypo": hypo, "used_hypo": bool(hypo), "cost_cny": round(cost, 6),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


class ParentDedupRetriever:
    """父子块去重：同一父块的多个子块只保留最靠前的一条。"""

    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline
        self.spent = 0.0

    def search(self, query: str, mode: str = "hybrid",
               topk: int = 5, pool: int = 50) -> dict:
        t0 = time.perf_counter()
        r = self.pipeline.search(query, mode=mode, topk=topk * 3, pool=pool)
        seen: set[str] = set()
        hits = []
        for h in r["hits"]:
            key = h.get("text", "")[:200]
            if key in seen:
                continue
            seen.add(key)
            hits.append(h)
            if len(hits) >= topk:
                break
        return {"query": query, "mode": f"parentdedup/{mode}", "hits": hits,
                "before": len(r["hits"]), "after": len(hits),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


class LogicalRoutingRetriever:
    """逻辑路由：先判断问题属于哪个平台，再只在那一家里检索。"""

    def __init__(self, pipeline: RetrievalPipeline, provider: str = "deepseek",
                 settings: Settings | None = None):
        self.pipeline = pipeline
        self.provider = provider
        self.settings = settings or get_settings()
        self.spent = 0.0
        self.routed = 0

    def classify(self, query: str) -> str:
        try:
            res = call_chat(self.settings,
                            [{"role": "user", "content": ROUTE_PROMPT.format(q=query)}],
                            provider=self.provider, thinking=False, max_tokens=16)
            self.spent += res.get("cost_cny") or 0.0
            t = (res["content"] or "").strip().lower()
            for s in SOURCES:
                if s in t:
                    return s
            return "general"
        except Exception:  # noqa: BLE001
            return "general"

    def search(self, query: str, mode: str = "hybrid",
               topk: int = 5, pool: int = 50) -> dict:
        t0 = time.perf_counter()
        try:
            target = self.classify(query)
        except Exception:  # noqa: BLE001
            target = "general"
        r = self.pipeline.search(query, mode=mode, topk=pool, pool=pool * 2)
        hits = r["hits"]
        if target != "general":
            self.routed += 1
            keep = [h for h in hits if h.get("source") == target]
            if len(keep) < topk:
                ids = {h["chunk_id"] for h in keep}
                keep += [h for h in hits if h["chunk_id"] not in ids]
            hits = keep
        return {"query": query, "mode": f"route/{mode}", "hits": hits[:topk],
                "routed_to": target, "cost_cny": 0.0,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


class SelfQueryRetriever:
    """自查询：把查询拆成「语义部分 + 结构化过滤条件」。"""

    def __init__(self, pipeline: RetrievalPipeline, provider: str = "deepseek",
                 settings: Settings | None = None):
        self.pipeline = pipeline
        self.provider = provider
        self.settings = settings or get_settings()
        self.spent = 0.0

    def extract(self, query: str) -> dict:
        try:
            res = call_chat(self.settings,
                            [{"role": "user", "content": SELFQ_PROMPT.format(q=query)}],
                            provider=self.provider, thinking=False, max_tokens=120)
            self.spent += res.get("cost_cny") or 0.0
            m = re.search(r"\{.*\}", res["content"] or "", re.S)
            if not m:
                return {}
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except (LLMError, json.JSONDecodeError):
            return {}

    def search(self, query: str, mode: str = "hybrid",
               topk: int = 5, pool: int = 50) -> dict:
        t0 = time.perf_counter()
        sq = self.extract(query)
        semantic = (sq.get("semantic") or query).strip()
        source = sq.get("source")
        if source not in SOURCES:
            source = None

        r = self.pipeline.search(semantic, mode=mode, topk=pool, pool=pool * 2)
        hits = r["hits"]
        if source:
            keep = [h for h in hits if h.get("source") == source]
            if len(keep) < topk:
                ids = {h["chunk_id"] for h in keep}
                keep += [h for h in hits if h["chunk_id"] not in ids]
            hits = keep
        return {"query": query, "mode": f"selfquery/{mode}", "hits": hits[:topk],
                "semantic": semantic, "source_filter": source, "cost_cny": 0.0,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


class RerankRetriever:
    """重排：粗排取 pool 条，再用 LLM 精排出 topk 条。"""

    def __init__(self, pipeline: RetrievalPipeline, provider: str = "deepseek"):
        self.inner = RetrievalPipeline(
            store=pipeline.store, bm25=pipeline.bm25,
            backend=pipeline.backend,
            reranker=build_reranker("llm", provider=provider))
        self.spent = 0.0

    def search(self, query: str, mode: str = "hybrid",
               topk: int = 5, pool: int = 50) -> dict:
        t0 = time.perf_counter()
        before = self.inner.reranker.spent
        r = self.inner.search(query, mode="hybrid_rerank", topk=topk, pool=pool)
        self.spent += self.inner.reranker.spent - before
        return {**r, "mode": f"rerank/{mode}", "reranked": True}


def build_retriever(kind: str, pipeline: RetrievalPipeline, **kw) -> object:
    """按名字构造检索器。baseline 就是原 pipeline 本身。"""
    provider = kw.get("provider", "deepseek")
    if kind == "baseline":
        return pipeline
    if kind == "multiquery":
        return MultiQueryRetriever(pipeline, n=kw.get("n", 3), provider=provider)
    if kind == "hyde":
        return HyDERetriever(pipeline, provider=provider)
    if kind == "parentdedup":
        return ParentDedupRetriever(pipeline)
    if kind == "routing":
        return LogicalRoutingRetriever(pipeline, provider=provider)
    if kind == "selfquery":
        return SelfQueryRetriever(pipeline, provider=provider)
    if kind == "rerank":
        return RerankRetriever(pipeline, provider=provider)
    raise ValueError(f"未知策略：{kind}，可选 baseline / multiquery / hyde / parentdedup "
                     "/ routing / selfquery / rerank")
