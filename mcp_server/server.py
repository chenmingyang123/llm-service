"""api-docs MCP Server 的实现。"""
from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from app.rag import AskPipeline
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.store import DEFAULT_CHUNKS

SERVER_NAME = "api-docs"
SERVER_VERSION = "0.1.0"
VALID_SOURCES = ("deepseek", "zhipu", "bailian")
MAX_TOP_K = 20

INSTRUCTIONS = (
    "这是一个 API 文档检索服务，覆盖 DeepSeek / 智谱 / 阿里百炼三家平台的官方 API 文档。"
    "查询时请用文档里可能出现的术语（例如「上下文硬盘缓存」「embedding 维度」），"
    "不要写长句 —— 语料是技术文档，术语匹配比口语改写有效得多。"
    "回答会带来源编号，可以核对；资料里没有的内容它会明确说不知道。"
)

mcp = MCPServer(
    name=SERVER_NAME,
    version=SERVER_VERSION,
    instructions=INSTRUCTIONS,
)

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def index_path() -> str:
    return DEFAULT_CHUNKS


def index_ready() -> bool:
    return os.path.exists(index_path())


def _retriever() -> RetrievalPipeline:
    with _lock:
        if "retriever" not in _cache:
            _cache["retriever"] = RetrievalPipeline()
        return _cache["retriever"]


def _asker() -> AskPipeline:
    with _lock:
        if "asker" not in _cache:
            _cache["asker"] = AskPipeline(_retriever())
        return _cache["asker"]


def _missing_index_msg() -> str:
    return (
        "检索索引不存在，无法查询。\n"
        f"缺少文件：{index_path()}\n"
        "请先在仓库根目录建一次索引（见 README「准备索引」一节），"
        "然后重启本 MCP Server。\n"
        "注意：工具清单与连接本身不受影响，缺的只是检索数据。"
    )


def _fmt_hits(res: dict) -> str:
    hits = res.get("hits") or []
    head = (f"找到 {len(hits)} 条（模式 {res.get('mode')}，"
            f"耗时 {res.get('latency_ms')}ms）")
    if not hits:
        return head + "\n没有命中任何片段。换个术语再试，或确认索引已建好。"
    blocks = []
    for i, h in enumerate(hits, 1):
        path = h.get("path") or ""
        src = h.get("source") or "?"
        title = h.get("title") or "(无标题)"
        text = " ".join((h.get("text") or "").split())
        if len(text) > 400:
            text = text[:400] + "…（已截断）"
        blocks.append(
            f"[{i}] {src} · {title}\n"
            f"    路径：{path}\n"
            f"    相关度：{h.get('score')}（向量 {h.get('vector_score')} / 关键词 {h.get('bm25_score')}）\n"
            f"    片段：{text}"
        )
    return head + "\n\n" + "\n\n".join(blocks)


@mcp.tool(
    name="search_api_docs",
    title="检索 API 文档",
    description=(
        "在三家平台（deepseek / zhipu / bailian）的官方 API 文档里做检索，"
        "返回最相关的片段及其来源与路径。\n"
        "何时用：需要查某个接口参数、计费规则、限制、错误码等**事实**时。\n"
        "何时不要用：需要一段整合好的解释时用 ask_api_docs；"
        "问的不是这三家的 API 时不要用（检索不出来）。\n"
        "查询请写术语（「上下文硬盘缓存」），不要写长句。"
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def search_api_docs(query: str, source: str = "", top_k: int = 5) -> str:
    """检索 API 文档片段。

    Args:
        query: 检索语句，用文档里可能出现的术语，不要写完整句子。
        source: 限定平台，可填 deepseek / zhipu / bailian；留空表示不限（不要猜）。
        top_k: 返回条数，1~20，默认 5。
    """
    q = (query or "").strip()
    if len(q) < 2:
        return "错误：query 太短，请给一个至少 2 个字符的检索词。"
    k = max(1, min(int(top_k or 5), MAX_TOP_K))
    src = (source or "").strip().lower()
    if src and src not in VALID_SOURCES:
        return (f"错误：source 只能是 {', '.join(VALID_SOURCES)} 之一，"
                f"或者留空表示不限；收到的是 {source!r}。")
    if not index_ready():
        return _missing_index_msg()

    try:
        res = _retriever().search(q, mode="hybrid", topk=k)
    except Exception as e:  # noqa: BLE001
        return f"错误：检索失败 —— {type(e).__name__}: {e}"
    if src:
        res = {**res, "hits": [h for h in res.get("hits", []) if h.get("source") == src]}
    return _fmt_hits(res)


@mcp.tool(
    name="ask_api_docs",
    title="问 API 文档（带引用）",
    description=(
        "直接提一个问题，返回基于官方文档的答案，并附来源编号；"
        "资料不足时会明确拒答（不会编）。\n"
        "何时用：需要一个整合好的解释时（「三家平台的 embedding 怎么选」）。\n"
        "何时不要用：只需要原始片段时用 search_api_docs（更快、更省）。"
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def ask_api_docs(query: str, mode: str = "hybrid") -> str:
    """基于文档回答问题，带引用来源。

    Args:
        query: 用户问题，可以是完整句子。
        mode: 检索模式，vector / bm25 / hybrid / hybrid_rerank，默认 hybrid。
    """
    q = (query or "").strip()
    if len(q) < 2:
        return "错误：query 太短，请给一个至少 2 个字符的问题。"
    if mode not in ("vector", "bm25", "hybrid", "hybrid_rerank"):
        return "错误：mode 只能是 vector / bm25 / hybrid / hybrid_rerank。"
    if not index_ready():
        return _missing_index_msg()

    try:
        res = _asker().ask(q, mode=mode, topk=5)
    except Exception as e:  # noqa: BLE001
        return f"错误：问答失败 —— {type(e).__name__}: {e}"

    if res.get("abstained"):
        return (f"资料不足，未作答。\n原因：{res.get('abstain_reason')}\n"
                "（这是设计行为：宁可说不知道，也不编一个看起来像的答案。）")

    lines = [res.get("answer") or "(空答案)"]
    cits = res.get("citations") or []
    if cits:
        lines.append("\n来源：")
        for c in cits:
            lines.append(f"  [{c.get('n')}] {c.get('source')} · {c.get('title')}"
                         f"（{c.get('path')}）")
    unsupported = res.get("unsupported") or []
    if unsupported:
        lines.append(f"\n⚠ 模型标注了不存在的来源编号 {unsupported} —— 这条答案的可信度要打折。")
    return "\n".join(lines)


@mcp.resource(
    "docs://corpus/stats",
    name="语料与检索配置快照",
    description="只读：当前索引的规模、embedding 后端、检索模式等元信息。",
    mime_type="application/json",
)
def corpus_stats() -> str:
    """返回语料的只读快照（索引缺失时也会正常返回，只是 ready=false）。"""
    info: dict[str, Any] = {
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "ready": index_ready(),
        "index_path": index_path(),
        "sources": list(VALID_SOURCES),
        "modes": ["vector", "bm25", "hybrid", "hybrid_rerank"],
        "read_only": True,
    }
    meta_path = os.path.join(os.path.dirname(index_path()), "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                info["index_meta"] = json.load(f)
        except Exception as e:  # noqa: BLE001
            info["index_meta_error"] = f"{type(e).__name__}: {e}"
    if index_ready():
        try:
            info["chunks"] = len(_retriever().store.rows)
        except Exception as e:  # noqa: BLE001
            info["chunks_error"] = f"{type(e).__name__}: {e}"
    return json.dumps(info, ensure_ascii=False, indent=2)


class _StdoutToStderr:
    """把 print() 的输出导向 stderr，但 `.buffer` 仍指向真 stdout。"""

    def __init__(self, real: Any, err: Any) -> None:
        self._real = real
        self._err = err

    def write(self, s: str) -> int:
        return self._err.write(s)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        try:
            self._err.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def buffer(self) -> Any:
        return self._real.buffer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def shield_stdout() -> None:
    """把 print() 从协议流上摘下来（幂等，可重复调用）。"""
    if not isinstance(sys.stdout, _StdoutToStderr):
        sys.stdout = _StdoutToStderr(sys.stdout, sys.stderr)  # type: ignore[assignment]


def main() -> None:
    """以 stdio 启动（默认传输）。"""
    shield_stdout()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
