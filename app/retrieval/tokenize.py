"""检索用的分词：中英混排，且必须保住 API 文档里的标识符。"""
from __future__ import annotations

import re
from functools import lru_cache

_RE_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:\-/]*")

_RE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def tokenize(text: str) -> list[str]:
    """把一段文本切成检索用的 token 列表（不去重，BM25 需要词频）。"""
    if not text:
        return []
    out: list[str] = []

    for m in _RE_WORD.finditer(text):
        w = m.group().strip("./:-_")
        if w:
            out.append(w.lower())

    buf = []
    for ch in text:
        buf.append(ch if _RE_CJK.match(ch) else " ")
    cleaned = "".join(buf)
    for run in cleaned.split():
        if len(run) == 1:
            out.append(run)
            continue
        for i in range(len(run)):
            out.append(run[i])
        for i in range(len(run) - 1):
            out.append(run[i:i + 2])
    return out


@lru_cache(maxsize=200_000)
def _cached(text: str) -> tuple[str, ...]:
    return tuple(tokenize(text))


def tokenize_cached(text: str) -> tuple[str, ...]:
    """带缓存的分词。建索引时同一段文本只切一次，查询时热词也能命中缓存。"""
    return _cached(text)
