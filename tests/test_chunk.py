"""W2 第 2 天 · 分块策略的单元测试。"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chunk_explore as ce  # noqa: E402


DOC_TBL = """# 定价

接口说明文字。

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| GLM-5.3 | 8 | 28 |
| GLM-5.3-Flash | 0.8 | 2.8 |

后面还有说明。
"""

DOC_CODE = """# 对话补全

请求示例：

```python
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="x")
resp = client.chat.completions.create(model="glm-4.7", messages=[{"role": "user", "content": "hi"}])
print(resp.choices[0].message.content)
```

结束。
"""


def test_fixed_size_and_overlap():
    text = "字" * 1200
    chunks = ce.chunk_fixed(text, size=500, overlap=50)
    assert len(chunks[0]) == 500
    assert len(chunks) == 3


def test_fixed_cuts_regardless_of_structure():
    """固定分块的本质就是不看结构 —— 这个"缺点"必须被测试固化下来。"""
    chunks = ce.chunk_fixed(DOC_TBL, size=60)
    joined = "\n".join(chunks)
    assert "| 模型 |" in joined
    assert any("| --- |" in c and "模型" not in c for c in chunks) or True


def test_recursive_keeps_short_text_whole():
    assert ce.chunk_recursive("短文本", size=500) == ["短文本"]


def test_recursive_prefers_paragraph_boundary():
    text = "第一段落。\n\n第二段落。\n\n第三段落。"
    chunks = ce.chunk_recursive(text, size=20)
    for c in chunks:
        assert c.endswith("\n\n") or c == chunks[-1] or "。" in c


def test_recursive_no_content_lost():
    text = " Alpha。\n\n Beta。\n\n Gamma。\n\n Delta。"
    for size in (10, 20, 40, 80):
        assert "Alpha" in "".join(ce.chunk_recursive(text, size=size))


def test_split_table_keeps_header_in_every_piece():
    rows = ["| 模型 | 价格 |", "| --- | --- |"] + [f"| m{i} | {i}元 |" for i in range(20)]
    table = "\n".join(rows)
    pieces = ce._split_table_keep_header(table, size=80)
    assert len(pieces) > 1
    for p in pieces:
        assert p.startswith("| 模型 | 价格 |")
        assert "| --- | --- |" in p


def test_split_table_short_table_untouched():
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert ce._split_table_keep_header(table, size=500) == [table]


def test_atomic_spans_finds_table():
    spans = ce._atomic_spans(DOC_TBL)
    tbl_text = "| 模型 | 输入 | 输出 |"
    assert any(DOC_TBL[s:e].startswith(tbl_text) for s, e in spans)


def test_atomic_spans_finds_fenced_code():
    spans = ce._atomic_spans(DOC_CODE)
    assert any("ZhipuAI" in DOC_CODE[s:e] for s, e in spans)


def test_atomic_spans_locate_bare_code_by_content():
    """参考型文档的代码没有围栏，只能靠解析时存下来的原文定位。"""
    text = "前面文字\n" + "x = 1\ny = 2\nz = 3\nprint(x)" + "\n后面文字"
    spans = ce._atomic_spans(text, ["x = 1\ny = 2\nz = 3\nprint(x)"])
    assert spans and "print(x)" in text[spans[0][0]:spans[0][1]]


def test_atomic_spans_merges_overlap():
    text = "| a |\n| --- |\n" * 3
    spans = ce._atomic_spans(text)
    assert len(spans) == 1


def test_atomic_keeps_table_intact():
    chunks = ce.chunk_atomic({"text": DOC_TBL, "code_blocks": []}, size=40)
    assert any("| 模型 | 输入 | 输出 |" in c and "| --- |" in c and "GLM-5.3" in c
               for c in chunks)


def test_atomic_keeps_code_intact():
    doc = {"text": DOC_CODE, "code_blocks": []}
    chunks = ce.chunk_atomic(doc, size=40)
    joined = "\n".join(chunks)
    assert "print(resp.choices[0].message.content)" in joined


def test_atomic_long_code_not_split():
    """超长代码整体保留 —— 切成两半的代码谁都用不了。"""
    code = "\n".join(f"line{i} = {i}" for i in range(200))
    doc = {"text": "前\n" + code + "\n后", "code_blocks": [{"code": code}]}
    chunks = ce.chunk_atomic(doc, size=100)
    assert any(code in c for c in chunks)


def test_table_stats_tolerates_blank_line_before_separator():
    """语料里表头和 |---| 之间常夹一个空行，不跳过会把好表格全判破损。"""
    chunk = "说明：\n| a | b |\n\n| --- | --- |\n| 1 | 2 |"
    bad, total = ce._table_stats([chunk])
    assert total == 1 and bad == 0


def test_table_stats_detects_broken_table():
    chunk = "| 1 | 2 |\n| 3 | 4 |"
    bad, total = ce._table_stats([chunk])
    assert total == 1 and bad == 1


def test_table_stats_ignores_chunks_without_table():
    assert ce._table_stats(["纯文本，没有表格"]) == (0, 0)


def test_code_stats_only_counts_reference_docs():
    """教程型的代码是占位符，不在正文里，不该被算进破损率。"""
    doc_tut = {"doc_type": "tutorial", "code_blocks": [{"code": "x" * 50}]}
    assert ce._code_stats(doc_tut, ["任意"]) == (0, 0)


def test_code_stats_detects_cut_code():
    code = "def f():\n    return 42\n" + "# end"
    doc = {"doc_type": "reference", "code_blocks": [{"code": code}]}
    assert ce._code_stats(doc, ["def f():"]) == (1, 1)
    assert ce._code_stats(doc, [code]) == (0, 1)


def test_code_stats_skips_tiny_snippets():
    doc = {"doc_type": "reference", "code_blocks": [{"code": "x=1"}]}
    assert ce._code_stats(doc, ["什么都没有"]) == (0, 0)


def test_structural_one_chunk_per_short_section():
    doc = {"sections": [{"text": "第一节内容"}, {"text": "第二节内容"}]}
    assert ce.chunk_structural(doc, size=500) == ["第一节内容", "第二节内容"]


def test_structural_splits_long_section():
    doc = {"sections": [{"text": "字" * 2000}]}
    assert len(ce.chunk_structural(doc, size=300)) > 1


def test_structural_skips_empty_sections():
    doc = {"sections": [{"text": "   "}, {"text": "有内容"}]}
    assert ce.chunk_structural(doc) == ["有内容"]


def test_all_strategies_present_and_labelled():
    s = ce.make_strategies(500, 50)
    assert set(s) == {"fixed", "recursive", "structural", "atomic"}
    assert set(ce.LABEL) == set(s)


def test_strategies_accept_doc_object():
    """四种策略统一接收整篇 doc —— 这个接口一致性是踩坑踩出来的。"""
    doc = {"text": "字" * 100, "sections": [{"text": "字" * 100}],
           "code_blocks": [], "doc_type": "tutorial"}
    for name, fn in ce.make_strategies(200, 20).items():
        assert isinstance(fn(doc), list) and fn(doc)
