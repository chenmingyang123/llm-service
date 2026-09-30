"""W2 第 5 天 · 检索子系统的单元测试。"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.retrieval.bm25 import BM25Index  # noqa: E402
from app.retrieval.hybrid import rrf  # noqa: E402
from app.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from app.retrieval.rerank import IdentityReranker, _parse_order  # noqa: E402
from app.retrieval.tokenize import tokenize  # noqa: E402

DOCS = [
    "DeepSeek 的 embedding 模型不支持，请用 embedding-3 或 text-embedding-v4。",
    "自定义 Skill 的目录根下必须有 SKILL.md 文件，description 不能为空。",
    "套餐额度耗尽后，系统不会继续消耗您的账户余额，需要等待下一个周期恢复。",
    "上下文硬盘缓存默认开启，最小命中前缀为 64 token。",
]


def test_tokenize_keeps_identifiers_intact():
    """API 文档里最要命的就是标识符被切碎。"""
    toks = tokenize("用 embedding-3 和 tokenizer_config.json 调 /files/:file_id")
    assert "embedding-3" in toks
    assert "tokenizer_config.json" in toks
    assert "files/:file_id" in toks
    assert "skill.md" in tokenize("根下必须有 SKILL.md 文件")


def test_tokenize_cjk_emits_unigram_and_bigram():
    """只用 bigram 会让单字查询失忆；只用 unigram 会丢词序。两个都要出。"""
    toks = tokenize("上下文硬盘缓存")
    assert "上" in toks and "下" in toks
    assert "上下" in toks and "硬盘" in toks
    assert "存最" not in tokenize("缓存。最小")


def test_tokenize_empty_and_none():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_bm25_puts_exact_term_first():
    """BM25 存在的理由：精确术语必须能命中，这正是向量路的短板。"""
    idx = BM25Index(DOCS)
    top = idx.topk("SKILL.md", k=1)
    assert top and top[0][0] == 1, "讲 SKILL.md 的那条必须排第一"


def test_bm25_no_match_returns_empty():
    """查询必须全用 ASCII 乱码：中文单字（比如"的"）几乎必然跟正文撞上，"""
    idx = BM25Index(DOCS)
    assert idx.topk("xyzzy plugh", k=3) == []


def test_bm25_score_is_nonnegative():
    """IDF 用平滑版就是为了保证这一点 —— 负分会让融合排序出莫名其妙的结果。"""
    idx = BM25Index(DOCS)
    assert all(s >= 0 for s in idx.scores("SKILL.md 缓存 额度"))


def test_bm25_length_normalization_penalizes_long_docs():
    """b=0 时不做长度归一，长文档会靠"词多"霸榜。"""
    docs = ["关键词", "关键词 " + "无关内容 " * 200]
    plain = BM25Index(docs, b=0.0)
    norm = BM25Index(docs, b=0.75)
    assert plain.scores("关键词")[1] > plain.scores("关键词")[0]
    assert norm.scores("关键词")[0] > norm.scores("关键词")[1]


def test_rrf_boosts_docs_present_in_both_paths():
    """在两路都出现的文档，应该压过只在单路排名更靠前的文档。"""
    a = [(2, 0.9), (3, 0.8)]
    b = [(3, 20.0)]
    out = dict(rrf([a, b]))
    assert out[3] > out[2]


def test_rrf_keeps_docs_from_single_path():
    """只在一路出现的文档必须保留 —— 这是"互补"的来源。"""
    out = dict(rrf([[(1, 0.9)], [(2, 5.0)]]))
    assert set(out) == {1, 2}


def test_rrf_ignores_raw_score_scale():
    """量纲无关：BM25 的 20 和余弦的 0.9 混在一起，只看排名就不会打架。"""
    r1 = rrf([[(1, 1.0)], [(2, 1.0)]])
    r2 = rrf([[(1, 1000.0)], [(2, 0.001)]])
    assert [d for d, _ in r1] == [d for d, _ in r2]


def test_rrf_weights_bias_one_path():
    out = dict(rrf([[(1, 1.0), (2, 1.0)], [(2, 1.0), (1, 1.0)]], weights=[1.0, 0.1]))
    assert out[1] > out[2]


def test_parse_order_valid():
    assert _parse_order("[3, 1, 2]", 3) == [3, 1, 2]


def test_parse_order_surrounded_by_prose():
    """模型几乎总会加点解释，必须能从里面抠出数组。"""
    assert _parse_order("根据相关性，重排结果是 [2, 3, 1]。", 3) == [2, 3, 1]


def test_parse_order_rejects_invalid_permutations():
    assert _parse_order("[1, 2]", 3) is None
    assert _parse_order("[1, 1, 2]", 3) is None
    assert _parse_order("[1, 5, 2]", 3) is None
    assert _parse_order("没有数组", 3) is None


def test_identity_reranker_is_a_noop():
    """基线必须真的是"什么都不做"，否则对比出来的提升是假的。"""
    hits = [(0, 0.9), (1, 0.5)]
    assert IdentityReranker().rerank("q", hits, lambda i: "") == hits


class _FakeStore:
    """最小可用的 store：给几条正交的假向量，让排序由数据决定。"""

    def __init__(self):
        self.rows = [
            {"chunk_id": f"c{i}", "source": "test", "title": "t",
             "section_path": ["s"], "role": "normal", "parent_id": None, "text": t}
            for i, t in enumerate(DOCS)
        ]
        self.dim = len(DOCS)

    def vector_search(self, qvec, k=10):
        sims = list(qvec)[: len(self.rows)]
        order = sorted(range(len(self.rows)), key=lambda i: -sims[i])
        return [(i, float(sims[i])) for i in order[:k] if sims[i] > 0]

    def text_of(self, i):
        return self.rows[i]["text"]

    def resolve_parent(self, i):
        return i, self.rows[i]["text"]


@pytest.fixture
def pipe(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr("app.retrieval.pipeline.embed",
                        lambda texts, backend, batch=32: ([[1.0] + [0.0] * 3], {}))
    return RetrievalPipeline(store=store, bm25=BM25Index(DOCS),
                             backend="local", reranker=IdentityReranker())


def test_pipeline_vector_mode(pipe):
    r = pipe.search("任意问题", mode="vector", topk=2)
    assert [h["chunk_id"] for h in r["hits"]] == ["c0"]


def test_pipeline_bm25_mode_ignores_vector(pipe):
    """关键词路要能独立工作 —— 它是向量路的备胎，不能依赖向量。"""
    r = pipe.search("SKILL.md", mode="bm25", topk=1)
    assert r["hits"][0]["chunk_id"] == "c1"


def test_pipeline_hybrid_combines(pipe):
    """向量把 c0 排第一，BM25 把 c1 排第一 —— 融合后两条都该在。"""
    r = pipe.search("SKILL.md", mode="hybrid", topk=2)
    ids = [h["chunk_id"] for h in r["hits"]]
    assert set(ids) == {"c0", "c1"}


def test_pipeline_rejects_unknown_mode(pipe):
    with pytest.raises(ValueError):
        pipe.search("x", mode="不存在")


def test_pipeline_reports_latency_and_source_scores(pipe):
    r = pipe.search("SKILL.md", mode="hybrid", topk=2)
    assert r["latency_ms"] >= 0
    assert any(h["bm25_score"] for h in r["hits"])
    assert any(h["vector_score"] for h in r["hits"])
