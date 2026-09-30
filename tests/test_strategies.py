"""W3 D3/D4 检索策略的纯逻辑测试（不调 API）。"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.retrieval.strategies import (  # noqa: E402
    HyDERetriever, LogicalRoutingRetriever, MultiQueryRetriever,
    ParentDedupRetriever, SelfQueryRetriever, build_retriever,
)


class _FakePipeline:
    """假检索器：返回固定 hits，含重复内容（模拟同父块的多子块）。"""

    def __init__(self, hits):
        self.hits = hits
        self.calls = 0

    def search(self, query, mode="hybrid", topk=5, pool=50):
        self.calls += 1
        return {"query": query, "mode": mode, "hits": self.hits, "latency_ms": 0.0}


def _hits():
    return [
        {"chunk_id": "a1", "text": "相同内容" * 40},
        {"chunk_id": "a2", "text": "相同内容" * 40},
        {"chunk_id": "b1", "text": "另一段内容" * 40},
    ]


def test_parentdedup_removes_duplicates():
    fake = _FakePipeline(_hits())
    r = ParentDedupRetriever(fake).search("q", topk=5)
    texts = [h["text"] for h in r["hits"]]
    assert len(texts) == len(set(texts)), "去重后不应还有重复正文"
    assert len(r["hits"]) == 2, f"3 条里 2 条重复，应剩 2 条，实得 {len(r['hits'])}"
    assert r["after"] <= r["before"]


def test_parentdedup_no_extra_llm_cost():
    fake = _FakePipeline(_hits())
    d = ParentDedupRetriever(fake)
    d.search("q")
    assert d.spent == 0.0, "纯去重不该产生 LLM 开销"


def test_multiquery_fuses_variants():
    """变体的贡献必须能进 topk —— 第一版就是栽在这（原查询填满后变体被截掉）。"""
    fake = _FakePipeline(_hits())
    mq = MultiQueryRetriever(fake, n=2)
    mq.variants = lambda q: (["改写一", "改写二"], 0.0)
    r = mq.search("q", topk=5)
    assert fake.calls == 3, f"应检索 3 次，实得 {fake.calls}"
    assert r["variants"] == 2
    assert len(r["hits"]) <= 5


def test_build_retriever_dispatches_params():
    """HyDE 不能收到 MultiQuery 专属的 n —— 这条是实际炸过一次才补的。"""
    fake = _FakePipeline(_hits())
    assert isinstance(build_retriever("baseline", fake), _FakePipeline)
    assert isinstance(build_retriever("parentdedup", fake), ParentDedupRetriever)
    hyde = build_retriever("hyde", fake, n=3, provider="deepseek")
    assert isinstance(hyde, HyDERetriever), "HyDE 应能构造成功（n 不应传给它）"
    mq = build_retriever("multiquery", fake, n=3, provider="deepseek")
    assert isinstance(mq, MultiQueryRetriever)
    assert mq.n == 3


def _mixed_hits():
    """三家平台的块各一条，用来验证过滤是否生效。"""
    return [
        {"chunk_id": "d1", "source": "deepseek", "text": "D内容" * 20},
        {"chunk_id": "z1", "source": "zhipu", "text": "Z内容" * 20},
        {"chunk_id": "b1", "source": "bailian", "text": "B内容" * 20},
    ]


def test_routing_filters_to_one_source():
    """路由到某家后，第一条应当是那个平台的块。"""
    rt = LogicalRoutingRetriever(_FakePipeline(_mixed_hits()))
    rt.classify = lambda q: "zhipu"
    r = rt.search("q", topk=5)
    assert r["routed_to"] == "zhipu"
    assert r["hits"][0]["source"] == "zhipu"


def test_routing_general_does_not_filter():
    """判为 general 时必须退回全库 —— 宁可不路由，也不能路由错。"""
    rt = LogicalRoutingRetriever(_FakePipeline(_mixed_hits()))
    rt.classify = lambda q: "general"
    r = rt.search("q", topk=5)
    assert len(r["hits"]) == 3, "general 不该过滤"
    assert rt.routed == 0


def test_routing_survives_classify_failure():
    """分类调用挂了也要能检索 —— 路由是增强环节，不能让检索挂掉。"""
    rt = LogicalRoutingRetriever(_FakePipeline(_mixed_hits()))
    def boom(q):
        raise RuntimeError("API 挂了")
    rt.classify = boom
    r = rt.search("q", topk=5)
    assert len(r["hits"]) >= 1


def test_selfquery_uses_semantic_and_filter():
    sq = SelfQueryRetriever(_FakePipeline(_mixed_hits()))
    sq.extract = lambda q: {"semantic": "改写后的问题", "source": "bailian"}
    r = sq.search("原问题", topk=5)
    assert r["semantic"] == "改写后的问题", "应该用改写后的语义去检索"
    assert r["source_filter"] == "bailian"
    assert r["hits"][0]["source"] == "bailian"


def test_selfquery_ignores_unknown_source():
    """抽出的 source 不是三家之一时应当忽略，不能把结果过滤空。"""
    sq = SelfQueryRetriever(_FakePipeline(_mixed_hits()))
    sq.extract = lambda q: {"semantic": "x", "source": "openai"}
    r = sq.search("q", topk=5)
    assert r["source_filter"] is None
    assert len(r["hits"]) == 3
