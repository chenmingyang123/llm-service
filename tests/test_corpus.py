"""W2 第 1 天 · 语料解析的单元测试。"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import parse_corpus as pc  # noqa: E402


MD_SAMPLE = """# 模型概览

正文一段。

## 定价

| 模型 | 输入价格 | 输出价格 |
|---|---|---|
| glm-4.7 | 2元/百万 | 8元/百万 |
| glm-4.5-air | 0.5元/百万 | 2元/百万 |

## 示例

```python
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="x")
```

结束。
"""


def test_markdown_headings():
    items = pc.parse_markdown(MD_SAMPLE)
    hs = [i for i in items if i["type"] == "h"]
    assert [h["level"] for h in hs] == [1, 2, 2]
    assert hs[0]["text"] == "模型概览"


def test_markdown_table_kept():
    items = pc.parse_markdown(MD_SAMPLE)
    tabs = [i for i in items if i["type"] == "table"]
    assert len(tabs) == 1
    rows = tabs[0]["rows"]
    assert rows[0] == ["模型", "输入价格", "输出价格"]
    assert len(rows) == 3


def test_markdown_code_kept():
    items = pc.parse_markdown(MD_SAMPLE)
    codes = [i for i in items if i["type"] == "code"]
    assert len(codes) == 1
    assert codes[0]["lang"] == "python"
    assert "ZhipuAI" in codes[0]["code"]


def test_table_to_markdown_shape():
    md = pc.table_to_markdown([["a", "b"], ["1", "2"]])
    assert md.splitlines()[0] == "| a | b |"
    assert md.splitlines()[1] == "| --- | --- |"
    assert md.splitlines()[2] == "| 1 | 2 |"


def test_table_escapes_pipe_in_cell():
    md = pc.table_to_markdown([["a|b", "c"]])
    assert "a\\|b" in md


def test_table_ragged_rows_padded():
    md = pc.table_to_markdown([["a", "b", "c"], ["1"]])
    assert md.splitlines()[2] == "| 1 |  |  |"


def test_sections_follow_heading_hierarchy():
    items = [
        {"type": "h", "level": 1, "text": "总览"},
        {"type": "p", "text": "总览正文"},
        {"type": "h", "level": 2, "text": "定价"},
        {"type": "p", "text": "定价正文"},
        {"type": "h", "level": 3, "text": "缓存"},
        {"type": "p", "text": "缓存正文"},
    ]
    doc = pc.build_doc(items)
    assert len(doc["sections"]) == 3
    assert doc["sections"][2]["path"] == ["总览", "定价", "缓存"]
    assert doc["sections"][0]["path"] == ["总览"]


def test_heading_same_level_replaces_not_nests():
    items = [
        {"type": "h", "level": 2, "text": "A"},
        {"type": "p", "text": "aa"},
        {"type": "h", "level": 2, "text": "B"},
        {"type": "p", "text": "bb"},
    ]
    doc = pc.build_doc(items)
    assert doc["sections"][1]["path"] == ["B"]


def test_code_block_not_in_body_text():
    """代码块不能进正文 —— 分块时从中间切开就废了，只留占位。"""
    items = [
        {"type": "h", "level": 2, "text": "示例"},
        {"type": "code", "lang": "bash", "code": "pip install openai\nrm -rf /"},
    ]
    doc = pc.build_doc(items)
    assert "rm -rf" not in doc["text"]
    assert doc["code_blocks"][0]["code"].startswith("pip install")
    assert "代码示例" in doc["text"]


def test_table_inlined_into_body_for_retrieval():
    """表格反过来要进正文 —— 不然搜"价格"搜不到表里的数字。"""
    items = [
        {"type": "h", "level": 2, "text": "定价"},
        {"type": "table", "rows": [["模型", "价格"], ["glm-4.7", "2元"]]},
    ]
    doc = pc.build_doc(items)
    assert "glm-4.7" in doc["text"]
    assert doc["tables"][0]["markdown"].startswith("| 模型 | 价格 |")


HTML_SAMPLE = """
<html><head><title>上下文硬盘缓存 | DeepSeek API Docs</title>
<script>var x = "<p>不该出现</p>";</script></head>
<body>
<nav><p>导航噪声</p></nav>
<article>
  <h1>上下文硬盘缓存</h1>
  <p>缓存命中的最小前缀是 64 token。</p>
  <h2>价格</h2>
  <table><tr><th>档位</th><th>价格</th></tr><tr><td>命中</td><td>0.1元</td></tr></table>
  <pre class="prism-code language-python"><code>print("hi")</code></pre>
</article>
<footer><p>页脚</p></footer>
</body></html>
"""


def test_html_skips_script_nav_footer():
    ex = pc.StructureExtractor()
    ex.feed(pc.pick_main_html(HTML_SAMPLE))
    ex.finish()
    texts = " ".join(i.get("text", "") for i in ex.items)
    assert "不该出现" not in texts
    assert "导航噪声" not in texts
    assert "页脚" not in texts


def test_html_extracts_table_and_code():
    ex = pc.StructureExtractor()
    ex.feed(pc.pick_main_html(HTML_SAMPLE))
    ex.finish()
    assert sum(1 for i in ex.items if i["type"] == "table") == 1
    codes = [i for i in ex.items if i["type"] == "code"]
    assert codes and codes[0]["lang"] == "python"
    assert 'print("hi")' in codes[0]["code"]


def test_html_heading_levels():
    ex = pc.StructureExtractor()
    ex.feed(pc.pick_main_html(HTML_SAMPLE))
    ex.finish()
    hs = [i for i in ex.items if i["type"] == "h"]
    assert [h["level"] for h in hs] == [1, 2]
    assert hs[0]["text"] == "上下文硬盘缓存"


def test_pick_main_prefers_aliyun_container():
    doc = '<div><p>n</p></div><div lang="zh" class="icms-help-docs-content"><p>正文</p></div>'
    assert "正文" in pc.pick_main_html(doc)
    assert "n" not in pc.pick_main_html(doc)


def test_pick_main_falls_back_to_article():
    doc = '<div class="x"><p>noise</p></div><article><p>a</p><p>b</p></article>'
    main = pc.pick_main_html(doc)
    assert "a" in main and "b" in main
    assert "noise" not in main


def test_pick_main_keeps_nested_divs():
    """容器里嵌套同名标签时不能提前截断 —— 少一半内容是最难发现的错。"""
    doc = ('<div class="icms-help-docs-content"><p>开头</p>'
           '<div class="code"><pre><code>x=1</code></pre></div>'
           '<p>结尾</p></div><div><p>噪声</p></div>')
    main = pc.pick_main_html(doc)
    assert "开头" in main and "结尾" in main
    assert "噪声" not in main


def test_unescape_nested_restores_quotes():
    escaped = '<div class=\\"icms\\"><p>正文</p></div>'
    out = pc.unescape_nested(escaped)
    assert 'class="icms"' in out
    assert "<p>" in out


def test_unescape_leaves_plain_html_alone():
    plain = '<div class="x"><p>hi</p></div>'
    assert pc.unescape_nested(plain) == plain


def test_short_doc_rejected():
    row = {"doc_id": "x", "source": "zhipu", "url": "http://e", "path": None, "status": 200}
    assert pc.parse_one(row)[1] is not None


def test_failed_fetch_rejected():
    row = {"doc_id": "x", "source": "zhipu", "url": "http://e",
           "path": None, "status": 0, "error": "timeout"}
    _, why = pc.parse_one(row)
    assert why and "抓取失败" in why


def test_est_tokens_positive_and_ordered():
    short = pc.est_tokens("你好")
    long = pc.est_tokens("你好" * 1000)
    assert short > 0
    assert long > short * 100


def test_min_chars_constant_is_sane():
    assert 100 <= pc.MIN_CHARS <= 1000
