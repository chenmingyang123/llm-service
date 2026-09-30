"""检索管线：把「向量 / BM25 / 融合 / 重排」串成一条可切换的链路。"""
from __future__ import annotations

import time

from .bm25 import BM25Index
from .embedding import embed, pick_backend
from .hybrid import rrf
from .rerank import IdentityReranker, Reranker
from .store import ChunkStore, get_store


class RetrievalPipeline:
    """一条检索链路。构造一次，反复用。"""

    def __init__(self, store: ChunkStore | None = None,
                 backend: str = "auto",
                 reranker: Reranker | None = None,
                 bm25: BM25Index | None = None,
                 rrf_k: int = 60,
                 rrf_weights: tuple[float, float] | None = None):
        self.store = store or get_store()
        self.backend = pick_backend(backend)
        self.reranker = reranker or IdentityReranker()
        self.bm25 = bm25 or BM25Index([r["text"] for r in self.store.rows])
        self.rrf_k = rrf_k
        self.rrf_weights = list(rrf_weights) if rrf_weights else None

    def vector_hits(self, query: str, k: int) -> list[tuple[int, float]]:
        qvec = embed([query], self.backend)[0][0]
        return self.store.vector_search(qvec, k=k)

    def bm25_hits(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.topk(query, k=k)

    def search(self, query: str, mode: str = "hybrid",
               topk: int = 5, pool: int = 50) -> dict:
        t0 = time.perf_counter()
        vec: list[tuple[int, float]] = []
        kw: list[tuple[int, float]] = []
        fused: list[tuple[int, float]] = []
        reranked = False

        if mode == "vector":
            fused = self.vector_hits(query, pool)[:topk]
            vec = fused
        elif mode == "bm25":
            fused = self.bm25_hits(query, pool)[:topk]
            kw = fused
        elif mode in ("hybrid", "hybrid_rerank"):
            vec = self.vector_hits(query, pool)
            kw = self.bm25_hits(query, pool)
            fused = rrf([vec, kw], k=self.rrf_k, weights=self.rrf_weights)[
                : (pool if mode == "hybrid_rerank" else topk)]
            if mode == "hybrid_rerank":
                fused = self.reranker.rerank(query, fused, self.store.text_of)[:topk]
                reranked = True
        else:
            raise ValueError(f"未知模式：{mode}，可选 vector / bm25 / hybrid / hybrid_rerank")

        vec_map = dict(vec)
        kw_map = dict(kw)
        rows = self.store.rows
        hits = []
        for doc_id, score in fused[:topk]:
            r = rows[doc_id]
            idx, text = self.store.resolve_parent(doc_id)
            hits.append({
                "chunk_id": r["chunk_id"],
                "score": round(float(score), 6),
                "source": r.get("source", ""),
                "title": r.get("title", ""),
                "path": " > ".join(r.get("section_path") or []),
                "from_parent": idx != doc_id or (r.get("role") == "child"),
                "text": text,
                "vector_score": round(float(vec_map.get(doc_id, 0.0)), 4) if vec_map else None,
                "bm25_score": round(float(kw_map.get(doc_id, 0.0)), 4) if kw_map else None,
            })

        return {
            "query": query,
            "mode": mode,
            "hits": hits,
            "reranked": reranked,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
