"""Rerank：粗排之后再来一次精排。"""
from __future__ import annotations

import json
import re

from ..config import get_settings
from ..llm import LLMError, call_chat

SNIPPET_CHARS = 220

_SYSTEM = (
    "你是检索系统的重排序模块。给你一个用户问题，以及若干候选文档片段。"
    "请按它们对这个问题的有用程度从高到低排序，只输出编号的 JSON 数组，"
    "例如 [3, 1, 2]。不要输出任何解释。"
)

_USER = """问题：{q}

候选片段：
{cands}

请输出重排后的编号数组（形如 [3, 1, 2]），数组长度必须与候选数量相同，且每个编号恰好出现一次。"""


class Reranker:
    """重排器基类。"""

    name = "identity"

    def rerank(self, query: str, hits: list[tuple[int, float]],
               get_text) -> list[tuple[int, float]]:
        raise NotImplementedError


class IdentityReranker(Reranker):
    """原样返回。基线，永远可用，零成本。"""

    name = "identity"

    def rerank(self, query, hits, get_text):
        return list(hits)


class LLMReranker(Reranker):
    """用 LLM 做 listwise 重排。"""

    name = "llm"

    def __init__(self, provider: str = "deepseek", model: str | None = None,
                 max_candidates: int = 20):
        self.provider = provider
        self.model = model
        self.max_candidates = max_candidates
        self.settings = get_settings()
        self.spent = 0.0

    def rerank(self, query, hits, get_text):
        cand = hits[: self.max_candidates]
        if len(cand) < 2:
            return list(hits)

        lines = []
        for n, (doc_id, _s) in enumerate(cand, start=1):
            body = re.sub(r"\s+", " ", get_text(doc_id))[:SNIPPET_CHARS]
            lines.append(f"[{n}] {body}")
        prompt = _USER.format(q=query, cands="\n".join(lines))

        try:
            r = call_chat(self.settings,
                          [{"role": "user", "content": prompt}],
                          system=_SYSTEM,
                          provider=self.provider,
                          model=self.model,
                          thinking=False,
                          max_tokens=256)
        except LLMError:
            return list(hits)
        self.spent += r.get("cost_cny") or 0.0
        order = _parse_order(r.get("content", ""), len(cand))
        if order is None:
            return list(hits)

        seen = set()
        out = []
        for n in order:
            doc_id, score = cand[n - 1]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append((doc_id, score))
        for item in cand:
            if item[0] not in seen:
                seen.add(item[0])
                out.append(item)
        out.extend(hits[self.max_candidates:])
        return out


def _parse_order(text: str, n: int) -> list[int] | None:
    """从模型输出里抠出编号数组。容错要足：模型会加各种装饰。"""
    m = re.search(r"\[\s*[\d\s,]+\]", text)
    if not m:
        return None
    try:
        arr = json.loads(m.group())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(arr, list) or not arr:
        return None
    out = [int(x) for x in arr if isinstance(x, (int, float))]
    if sorted(out) != list(range(1, n + 1)):
        return None
    return out


def build_reranker(kind: str = "identity", **kw) -> Reranker:
    if kind == "identity":
        return IdentityReranker()
    if kind == "llm":
        return LLMReranker(**kw)
    raise ValueError(f"未知 reranker：{kind}")
