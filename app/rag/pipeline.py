"""RAG 生成端管线：检索 → 组装 → 生成 → 校验引用。"""
from __future__ import annotations

import time

from ..config import get_settings
from ..llm import LLMError, call_chat
from ..retrieval.pipeline import RetrievalPipeline
from .abstain import DEFAULT_MIN_COSINE, extract_cited, gate, parse_answer
from .prompts import SYSTEM, USER_TEMPLATE, build_context
from .schemas import CitationOut

QUOTE_CHARS = 120


class AskPipeline:
    def __init__(self, retriever: RetrievalPipeline, provider: str = "deepseek",
                 model: str | None = None, max_context_chars: int = 700):
        self.retriever = retriever
        self.provider = provider
        self.model = model
        self.max_context_chars = max_context_chars
        self.settings = get_settings()

    def ask(self, query: str, mode: str = "hybrid", topk: int = 5,
            pool: int = 50, min_score: float | None = None) -> dict:
        t0 = time.perf_counter()

        r = self.retriever.search(query, mode=mode, topk=topk, pool=pool)
        hits = r["hits"]
        top_score = hits[0]["score"] if hits else 0.0

        base = {
            "query": query, "mode": mode, "top_score": round(top_score, 6),
            "hits_count": len(hits), "provider": "", "model": "",
        }

        passed, why = gate(hits,
                           min_cosine=min_score if min_score is not None else DEFAULT_MIN_COSINE)
        if not passed:
            return {**base, "ok": True, "answer": "",
                    "abstained": True, "abstain_reason": f"检索不足：{why}",
                    "citations": [], "used": [], "unsupported": [],
                    "context": [], "latency_ms": _ms(t0), "cost_cny": 0.0}

        snippets = [h["text"] for h in hits]
        context = build_context(snippets, max_chars=self.max_context_chars)

        try:
            res = call_chat(
                self.settings,
                [{"role": "user", "content": USER_TEMPLATE.format(context=context, query=query)}],
                system=SYSTEM, provider=self.provider, model=self.model,
                thinking=False,
                max_tokens=600)
        except LLMError as e:
            return {**base, "ok": False, "answer": "", "abstained": True,
                    "abstain_reason": f"生成调用失败：{e}",
                    "citations": [], "used": [], "unsupported": [],
                    "context": snippets, "latency_ms": _ms(t0), "cost_cny": 0.0}

        raw = res.get("content", "")

        abstained, answer, reason = parse_answer(raw)

        used, bad = extract_cited(raw, len(hits))
        citations = [
            CitationOut(n=i, chunk_id=hits[i - 1]["chunk_id"],
                        source=hits[i - 1]["source"], title=hits[i - 1]["title"],
                        path=hits[i - 1]["path"],
                        quote=" ".join(hits[i - 1]["text"].split())[:QUOTE_CHARS])
            for i in sorted(set(used)) if not abstained
        ]

        return {**base, "ok": True, "answer": answer,
                "abstained": abstained,
                "abstain_reason": reason if abstained else "",
                "citations": citations, "used": used, "unsupported": bad,
                "context": snippets,
                "provider": res.get("provider", ""), "model": res.get("model", ""),
                "latency_ms": _ms(t0), "cost_cny": round(res.get("cost_cny") or 0.0, 6)}


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
