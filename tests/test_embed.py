"""W2 第 3 天 · 切块与向量化的单元测试。"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import embed_corpus as ec  # noqa: E402


DOC = {
    "doc_id": "abc123", "source": "deepseek", "title": "上下文硬盘缓存",
    "url": "https://api-docs.deepseek.com/zh-cn/guides/kv_cache",
    "doc_type": "tutorial", "stats": {"n_tables": 0, "n_code_blocks": 0},
    "text": "缓存命中的前提是前缀已落盘。\n\n最小前缀是 64 token，低于此不命中。",
    "sections": [
        {"heading": "缓存规则", "path": ["上下文硬盘缓存", "缓存规则"],
         "text": "缓存命中的前提是前缀已落盘。"},
        {"heading": "最小长度", "path": ["上下文硬盘缓存", "最小长度"],
         "text": "最小前缀是 64 token，低于此不命中。"},
    ],
    "code_blocks": [],
}


def test_cosine_identical_is_one():
    v = [1.0, 2.0, 3.0]
    assert abs(ec.cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert abs(ec.cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_handles_zero_vector():
    assert ec.cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_search_returns_sorted_topk():
    q = [1.0, 0.0]
    rows = [{"id": "a", "vector": [0.0, 1.0]},
            {"id": "b", "vector": [1.0, 0.0]},
            {"id": "c", "vector": [0.9, 0.1]}]
    hits = ec.search(q, rows, k=2)
    assert [h[1]["id"] for h in hits] == ["b", "c"]
    assert hits[0][0] > hits[1][0]


def test_build_chunks_normal_size_not_split():
    chunks = ec.build_chunks([DOC], "fixed", size=500, max_chars=1000)
    assert all(c["role"] == "normal" for c in chunks)
    assert all(c["vectorize"] for c in chunks)


def _big_code_doc() -> dict:
    """参考型文档 + 超长代码块 —— 原子感知会把它整体保留，从而产生超大块。"""
    code = "line_%d = %d\n" % (0, 0) + "".join(f"v{i} = {i}\n" for i in range(600))
    return {**DOC, "doc_type": "reference", "text": "说明\n" + code,
            "sections": [{"heading": "h", "path": ["h"], "text": "说明\n" + code}],
            "code_blocks": [{"code": code}]}


def test_big_code_doc_does_produce_oversized_chunk():
    """先确认测试数据真的能造出超大块 —— 否则后面三个测试都是假的。"""
    chunks = ec.build_chunks([_big_code_doc()], "atomic", size=500, max_chars=1000)
    assert sum(1 for c in chunks if c["role"] == "parent") >= 1


def test_build_chunks_splits_oversized_into_parent_child():
    chunks = ec.build_chunks([_big_code_doc()], "atomic", size=500, max_chars=1000)
    parents = [c for c in chunks if c["role"] == "parent"]
    children = [c for c in chunks if c["role"] == "child"]
    assert len(parents) == 1 and len(children) > 1
    assert not parents[0]["vectorize"]
    assert all(c["vectorize"] for c in children)
    assert all(c["parent_id"] == parents[0]["chunk_id"] for c in children)


def test_parent_keeps_full_text():
    doc = _big_code_doc()
    chunks = ec.build_chunks([doc], "atomic", size=500, max_chars=1000)
    parent = next(c for c in chunks if c["role"] == "parent")
    assert len(parent["text"]) > 1000
    assert parent["text"] in doc["text"]


def test_child_chunks_respect_max_chars():
    chunks = ec.build_chunks([_big_code_doc()], "atomic", size=500, max_chars=1000)
    children = [c for c in chunks if c["role"] == "child"]
    assert children
    assert all(c["n_chars"] <= 1000 for c in children)


def test_resolve_parent_returns_full_text():
    parent = {"chunk_id": "p1", "role": "parent", "text": "完整原文" * 100}
    child = {"chunk_id": "p1-c00", "role": "child", "parent_id": "p1", "text": "片段"}
    chunks = [parent, child]
    assert ec.resolve_parent(child, chunks) is parent
    assert ec.resolve_parent(parent, chunks) is parent


def test_resolve_parent_falls_back_when_missing():
    orphan = {"chunk_id": "x", "role": "child", "parent_id": "不存在", "text": "片段"}
    assert ec.resolve_parent(orphan, [orphan]) is orphan


def test_chunks_carry_section_path():
    chunks = ec.build_chunks([DOC], "atomic", size=500, max_chars=1000)
    assert chunks
    for c in chunks:
        assert "section_path" in c and isinstance(c["section_path"], list)


def test_chunks_carry_source_and_url():
    chunks = ec.build_chunks([DOC], "atomic", size=500, max_chars=1000)
    for c in chunks:
        assert c["source"] == "deepseek"
        assert c["url"].startswith("http")


def test_empty_chunks_dropped():
    doc = {**DOC, "text": "   \n\n   ", "sections": [{"heading": "h", "path": ["h"], "text": "  "}]}
    chunks = ec.build_chunks([doc], "fixed", size=500, max_chars=1000)
    assert all(c["text"].strip() for c in chunks)


def _fake_embed(texts, backend=""):
    """假 embedding：永远返回同一个向量，这样排序由测试数据自己决定。"""
    return [[1.0, 0.0]], {}


def test_eval_counts_hit_at_rank(monkeypatch):
    monkeypatch.setattr(ec, "embed", _fake_embed)
    queries = [{"qid": "q1", "question": "x", "expect_any": ["命中词"]}]
    chunks = [
        {"chunk_id": "a", "role": "normal", "parent_id": None, "vector": [1.0, 0.0],
         "text": "无关内容"},
        {"chunk_id": "b", "role": "normal", "parent_id": None, "vector": [0.9, 0.1],
         "text": "这里有命中词"},
    ]
    r = ec.run_eval(queries, chunks, "zhipu", k=2)
    assert r["n"] == 1
    assert r["hit@2"] == 1.0
    assert r["mrr"] == pytest.approx(0.5)


def test_eval_counts_miss(monkeypatch):
    monkeypatch.setattr(ec, "embed", _fake_embed)
    queries = [{"qid": "q1", "question": "x", "expect_any": ["不存在的词"]}]
    chunks = [{"chunk_id": "a", "role": "normal", "parent_id": None,
               "vector": [1.0, 0.0], "text": "别的内容"}]
    r = ec.run_eval(queries, chunks, "zhipu", k=1)
    assert r["hit@1"] == 0.0 and r["mrr"] == 0.0


def test_eval_looks_into_parent_text(monkeypatch):
    """子块命中时，判分要看父块内容 —— 因为喂给模型的是父块。"""
    monkeypatch.setattr(ec, "embed", _fake_embed)
    queries = [{"qid": "q1", "question": "x", "expect_any": ["只在父块里"]}]
    parent = {"chunk_id": "p", "role": "parent", "parent_id": None, "vector": None,
              "text": "只在父块里出现的答案"}
    child = {"chunk_id": "p-c00", "role": "child", "parent_id": "p",
             "vector": [1.0, 0.0], "text": "片段，没有答案词"}
    r = ec.run_eval(queries, [parent, child], "zhipu", k=1)
    assert r["hit@1"] == 1.0


def test_pick_backend_respects_explicit_choice():
    assert ec.pick_backend("zhipu") == "zhipu"
    assert ec.pick_backend("local") == "local"
    assert ec.pick_backend("bailian") == "bailian"


def test_embed_dispatches_and_rejects_unknown():
    """统一分派入口必须覆盖三个后端，且对未知名字直接炸。"""
    for b in ("bailian", "zhipu", "local"):
        assert hasattr(ec, "embed_" + b), f"缺少后端 {b}"
    with pytest.raises(ValueError):
        ec.embed(["x"], "不存在的后端")


def test_bailian_batch_capped_at_server_limit():
    """百炼单批上限是 10，服务端硬校验（实测超过 10 就 400 InvalidParameter）。"""
    assert ec.BAILIAN_MAX_BATCH == 10
    assert min(32, ec.BAILIAN_MAX_BATCH) == 10


def test_guard_overwrite_refuses_shrinking(tmp_path):
    """小样本试跑不能覆盖整库 —— 这个坑真踩过，赔进去一整轮 embedding 费用。"""
    p = tmp_path / "chunks.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n{"c":3}\n', encoding="utf-8")
    ec._guard_overwrite(str(p), 100)
    with pytest.raises(SystemExit):
        ec._guard_overwrite(str(p), 2)
    ec._guard_overwrite(str(p), 2, force=True)


def test_constants_are_sane():
    assert 500 <= ec.MAX_CHARS <= 2000
    assert 0 <= ec.SPLIT_OVERLAP < ec.MAX_CHARS
