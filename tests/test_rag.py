"""W2 第 6 天 · RAG 生成端的单元测试。"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.rag.abstain import (DEFAULT_MIN_COSINE, extract_cited, gate,  # noqa: E402
                             parse_answer)
from app.rag.prompts import build_context  # noqa: E402
from app.rag.pipeline import AskPipeline  # noqa: E402


def _hit(v=None, kw=None, cid="c1", score=0.5):
    return {"chunk_id": cid, "score": score, "vector_score": v, "bm25_score": kw,
            "source": "s", "title": "t", "path": "p", "text": "正文"}


def test_gate_blocks_low_cosine():
    passed, why = gate([_hit(v=0.20)])
    assert not passed and "低于阈值" in why


def test_gate_passes_high_cosine():
    passed, why = gate([_hit(v=0.80)])
    assert passed and why == ""


def test_gate_uses_cosine_not_fused_score():
    """判据必须是余弦，不是融合分 —— 融合分只有 0.016~0.033 的区间，选不出阈值。"""
    passed, _ = gate([_hit(v=0.45, score=0.032)])
    assert passed
    passed, _ = gate([_hit(v=0.30, score=0.032)])
    assert not passed


def test_gate_falls_back_to_bm25_when_no_vector():
    """纯 bm25 模式没有向量分，不能因此就放行。"""
    assert gate([_hit(kw=30.0)])[0]
    assert not gate([_hit(kw=1.0)])[0]


def test_gate_no_hits():
    passed, why = gate([])
    assert not passed and "没找到" in why


def test_default_threshold_matches_calibration():
    """阈值是校准出来的：域内最低 0.575 放行、域外最高 0.397 拦下。"""
    assert 0.397 < DEFAULT_MIN_COSINE < 0.575


def test_parse_answer_normal():
    abstained, answer, _ = parse_answer("最小前缀是 64 token [1]。")
    assert not abstained and "64 token" in answer


def test_parse_answer_no_answer_marker():
    abstained, answer, reason = parse_answer("NO_ANSWER\n资料里只有价格，没有并发数。")
    assert abstained and answer == "" and "并发数" in reason


def test_parse_answer_marker_tolerates_decoration():
    """模型常给标记加粗或加符号，得容忍。"""
    assert parse_answer("**NO_ANSWER**\n缺数据")[0]
    assert parse_answer("NO_ANSWER")[0] and parse_answer("NO_ANSWER")[2] != ""


def test_parse_answer_empty():
    assert parse_answer("")[0]
    assert parse_answer(None)[0]


def test_extract_cited_valid():
    used, bad = extract_cited("是 64 MiB [2][3]。", n_docs=3)
    assert used == [2, 3] and bad == []


def test_extract_cited_out_of_range_is_flagged():
    """标了不存在的编号 = 瞎标。这是唯一能自动发现的幻觉信号。"""
    used, bad = extract_cited("答案是 [7]。", n_docs=3)
    assert used == [] and bad == [7]


def test_extract_cited_dedupes():
    used, _ = extract_cited("[1] 又 [1] 再 [2]", n_docs=3)
    assert used == [1, 2]


def test_build_context_numbers_from_one():
    """编号从 1 开始（不是 0）—— 模型对 [1] 的理解远比 [0] 稳定。"""
    ctx = build_context(["甲", "乙"])
    assert ctx.startswith("[1] 甲") and "[2] 乙" in ctx


def test_build_context_truncates():
    ctx = build_context(["字" * 500], max_chars=50)
    assert len(ctx) < 80


class _FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, mode="hybrid", topk=5, pool=50):
        return {"hits": self._hits[:topk], "mode": mode}


def _make_ask(hits, monkeypatch, content="答案是 42 [1]。"):
    monkeypatch.setattr("app.rag.pipeline.call_chat",
                        lambda s, ms, **kw: {"content": content, "provider": "p",
                                             "model": "m", "cost_cny": 0.001})
    return AskPipeline(_FakeRetriever(hits))


def test_ask_abstains_without_calling_llm(monkeypatch):
    """闸门①拦下的请求不该产生任何生成费用。"""
    called = []
    monkeypatch.setattr("app.rag.pipeline.call_chat",
                        lambda s, ms, **kw: called.append(1) or {"content": "x"})
    ap = AskPipeline(_FakeRetriever([_hit(v=0.10)]))
    r = ap.ask("随便问")
    assert r["abstained"] and r["cost_cny"] == 0.0 and not called


def test_ask_returns_citations(monkeypatch):
    ap = _make_ask([_hit(v=0.8, cid="c1"), _hit(v=0.7, cid="c2")], monkeypatch)
    r = ap.ask("问", topk=2)
    assert not r["abstained"]
    assert [c.n for c in r["citations"]] == [1]
    assert r["unsupported"] == []


def test_ask_flags_unsupported_citation(monkeypatch):
    ap = _make_ask([_hit(v=0.8)], monkeypatch, content="答案 [9]。")
    r = ap.ask("问", topk=1)
    assert r["unsupported"] == [9]
    assert r["citations"] == []


def test_ask_model_abstain_is_reported(monkeypatch):
    ap = _make_ask([_hit(v=0.8)], monkeypatch, content="NO_ANSWER\n没有这个数")
    r = ap.ask("问", topk=1)
    assert r["abstained"] and "没有这个数" in r["abstain_reason"]


def test_ask_always_has_provider_model_fields(monkeypatch):
    """拒答时也要有 provider/model，字段时有时无比值为空麻烦得多。"""
    ap = _make_ask([_hit(v=0.10)], monkeypatch)
    assert "provider" in ap.ask("问") and "model" in ap.ask("问")


def test_ask_llm_failure_is_not_silently_answered(monkeypatch):
    from app.llm import LLMError

    def boom(s, ms, **kw):
        raise LLMError("网络挂了")
    monkeypatch.setattr("app.rag.pipeline.call_chat", boom)
    ap = AskPipeline(_FakeRetriever([_hit(v=0.8)]))
    r = ap.ask("问")
    assert not r["ok"] and r["abstained"] and "失败" in r["abstain_reason"]
