"""W2 第 7 天 · golden set 与 baseline 的测试。"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import run_baseline as rb  # noqa: E402

GOLDEN = os.path.join(ROOT, "data", "eval", "golden.jsonl")


def _golden():
    return [json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()]


def _chunks_with_vector():
    out = {}
    for l in open(os.path.join(ROOT, "data", "embed", "chunks.jsonl"), encoding="utf-8"):
        if not l.strip():
            continue
        c = json.loads(l)
        if c.get("vector"):
            out[c["chunk_id"]] = c
    return out


def test_golden_has_enough_cases():
    g = _golden()
    assert len(g) >= 50, f"golden set 只有 {len(g)} 条，计划要求 50 条"


def test_golden_contains_hard_types():
    """计划点名要包含难例：规则冲突、适用边界、需跨文档综合。"""
    types = {x["type"] for x in _golden()}
    for t in ("conflict", "boundary", "multi"):
        assert t in types, f"缺少难例类型 {t}"


def test_golden_every_expected_chunk_exists():
    """标了一个不存在的 chunk_id，这条题就永远判不中，而且没人会发现。"""
    ch = _chunks_with_vector()
    for x in _golden():
        for cid in [x["expect_chunk"]] + x.get("expect_chunks", []):
            assert cid in ch, f"{x['qid']} 指向的块 {cid} 不在向量库里"


def test_golden_expected_chunk_really_contains_answer():
    """标准答案块里必须真有答案 —— 这条校验抓出过解析残留的 CSS 垃圾块。"""
    ch = _chunks_with_vector()
    for x in _golden():
        text = ch[x["expect_chunk"]]["text"]
        for kw in x["expect_any"]:
            assert kw in text, f"{x['qid']} 的关键词 {kw!r} 不在标准答案块里"


def test_golden_multi_type_declares_all_required_chunks():
    """跨文档综合题必须列出所有必需来源，只列一个就没法算全中率。"""
    for x in _golden():
        if x["type"] == "multi":
            assert len(x.get("expect_chunks", [])) >= 2, f"{x['qid']} 是 multi 但没列全来源"


def test_golden_qids_unique():
    qids = [x["qid"] for x in _golden()]
    assert len(qids) == len(set(qids)), "qid 有重复"


def test_pct_percentiles():
    xs = list(range(1, 101))
    assert rb.pct(xs, 50) == 50 or abs(rb.pct(xs, 50) - 50) <= 1
    assert rb.pct(xs, 95) == 95 or abs(rb.pct(xs, 95) - 95) <= 1
    assert rb.pct([], 95) == 0.0
    assert rb.pct([7.0], 95) == 7.0


def test_pct_exposes_tail_that_average_hides():
    """均值会被大量快请求拉低，把少数慢请求藏起来。"""
    xs = [300.0] * 90 + [3000.0] * 10
    mean = sum(xs) / len(xs)
    assert rb.pct(xs, 95) > mean, f"p95={rb.pct(xs,95)} 没能暴露长尾（均值 {mean}）"
    assert rb.pct(xs, 50) < mean


def test_faithfulness_floor_catches_fabrication():
    """答案里冒出原文没有的数字 → 下界必须掉下来。"""
    assert rb.faithfulness_floor("是 64 MiB [1]", ["上传单个文件最大 64 MiB"]) == 1.0
    assert rb.faithfulness_floor("是 999 MiB [1]", ["上传单个文件最大 64 MiB"]) < 1.0


def test_faithfulness_floor_ignores_citation_markers():
    """[1] 这种引用编号不能被当成实体 —— 否则忠实度被系统性低估。"""
    assert rb.faithfulness_floor("是 64 MiB [1][2]", ["最大 64 MiB"]) == 1.0
    assert rb.faithfulness_floor("是 64 MiB", ["最大 64 MiB"]) == 1.0


def test_faithfulness_floor_empty_answer():
    assert rb.faithfulness_floor("", ["任意"]) == 1.0
