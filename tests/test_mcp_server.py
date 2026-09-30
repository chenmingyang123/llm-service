"""MCP Server 的测试 —— W4 块 D。"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp_server import server as M  # noqa: E402


def test_server_metadata():
    assert M.mcp.name == "api-docs"
    assert M.mcp.version == "0.1.0"


def test_tools_listed_and_readonly():
    tools = asyncio.run(M.mcp.list_tools())
    by_name = {t.name: t for t in tools}
    assert {"search_api_docs", "ask_api_docs"} <= set(by_name)
    for t in tools:
        ann = getattr(t, "annotations", None)
        assert ann is not None and ann.read_only_hint is True, f"{t.name} 必须声明只读"
        assert t.description and "何时不要用" in t.description, \
            f"{t.name} 的描述里要写「何时不要用」（沿用 W1 的 schema 规则）"


def test_resource_listed():
    res = asyncio.run(M.mcp.list_resources())
    uris = [str(r.uri) for r in res]
    assert "docs://corpus/stats" in uris


def test_search_query_too_short():
    out = M.search_api_docs("x")
    assert out.startswith("错误：") and "太短" in out


def test_search_bad_source():
    out = M.search_api_docs("上下文缓存", source="openai")
    assert out.startswith("错误：") and "source" in out


def test_search_topk_clamped_or_ok():
    """top_k 传超界值不应该炸（内部夹紧到 1~20）。"""
    out = M.search_api_docs("上下文硬盘缓存", top_k=999)
    assert isinstance(out, str) and out


def test_ask_bad_mode():
    out = M.ask_api_docs("怎么计费", mode="magic")
    assert out.startswith("错误：") and "mode" in out


def test_missing_index_guidance(monkeypatch):
    """索引缺失时必须给**能照着做**的引导，而不是抛异常。"""
    monkeypatch.setattr(M, "index_path", lambda: os.path.join(ROOT, "data", "__nope__.jsonl"))
    assert M.index_ready() is False
    out = M.search_api_docs("上下文硬盘缓存")
    assert "索引不存在" in out and "README" in out
    out2 = M.ask_api_docs("怎么计费")
    assert "索引不存在" in out2
    stats = json.loads(M.corpus_stats())
    assert stats["ready"] is False and stats["read_only"] is True


@pytest.mark.skipif(not M.index_ready(), reason="本机没有索引，跳过真实检索")
def test_search_real_hits():
    out = M.search_api_docs("上下文硬盘缓存", top_k=3)
    assert "找到" in out and "deepseek" in out


def test_corpus_stats_shape():
    stats = json.loads(M.corpus_stats())
    assert stats["server"] == "api-docs"
    assert stats["read_only"] is True
    assert set(stats["sources"]) == {"deepseek", "zhipu", "bailian"}
    assert "hybrid" in stats["modes"]


def test_stdio_roundtrip():
    """起真子进程走 stdio：这是「能被装」的硬证据。"""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def _run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "mcp_server"], cwd=ROOT)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                assert init.server_info.name == "api-docs"

                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert {"search_api_docs", "ask_api_docs"} <= names
                for t in tools.tools:
                    assert getattr(t.annotations, "read_only_hint", None) is True

                res = await session.call_tool("search_api_docs",
                                              {"query": "x"})
                assert res.is_error is False
                text = "\n".join(c.text for c in res.content if getattr(c, "text", None))
                assert text.startswith("错误：") and "太短" in text

                rr = await session.read_resource("docs://corpus/stats")
                body = "".join(getattr(c, "text", "") for c in rr.contents)
                assert json.loads(body)["server"] == "api-docs"

    asyncio.run(_run())
