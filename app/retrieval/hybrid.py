"""混合检索：把向量路和 BM25 路的结果合成一个排序。"""
from __future__ import annotations

RRF_K = 60


def rrf(rank_lists: list[list[tuple[int, float]]], k: int = RRF_K,
        weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion。"""
    if not rank_lists:
        return []
    if weights is None:
        weights = [1.0] * len(rank_lists)
    fused: dict[int, float] = {}
    for w, ranks in zip(weights, rank_lists):
        for rank, (doc_id, _score) in enumerate(ranks, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    out = sorted(fused.items(), key=lambda x: -x[1])
    return [(int(d), float(s)) for d, s in out]


def normalize(scores: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """把一路分数线性压到 [0,1]，仅供展示/调试用。"""
    if not scores:
        return []
    vals = [s for _, s in scores]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [(d, 1.0) for d, _ in scores]
    return [(d, (s - lo) / (hi - lo)) for d, s in scores]
