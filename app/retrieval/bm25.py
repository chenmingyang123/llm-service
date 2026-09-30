"""BM25 —— 关键词检索。跟向量检索互补的那一半。"""
from __future__ import annotations

import math
from collections import Counter

from .tokenize import tokenize_cached

K1 = 1.5
B = 0.75


class BM25Index:
    """倒排 + 词频表。"""

    def __init__(self, docs: list[str], k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.doc_len = [0] * self.n
        self.tf: list[Counter] = []

        df: Counter = Counter()
        for i, text in enumerate(docs):
            toks = tokenize_cached(text)
            c = Counter(toks)
            self.tf.append(c)
            self.doc_len[i] = len(toks)
            df.update(c.keys())

        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.idf = {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(self.tf):
            for t, f in c.items():
                self.postings.setdefault(t, []).append((i, f))

    def scores(self, query: str) -> list[float]:
        """返回每条文档对这条查询的 BM25 分数（未命中为 0）。"""
        out = [0.0] * self.n
        qtok = Counter(tokenize_cached(query))
        avgdl = self.avgdl or 1.0
        for t, qf in qtok.items():
            posts = self.postings.get(t)
            if not posts:
                continue
            idf = self.idf.get(t, 0.0)
            if idf <= 0:
                continue
            for doc_id, f in posts:
                dl = self.doc_len[doc_id]
                norm = 1 - self.b + self.b * dl / avgdl
                out[doc_id] += idf * (f * (self.k1 + 1)) / (f + self.k1 * norm) * qf
        return out

    def topk(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """取分数最高的 k 条。自己写而不全排序：全库几万条时 sort 是浪费。"""
        scored = self.scores(query)
        nz = [(i, s) for i, s in enumerate(scored) if s > 0]
        if not nz:
            return []
        nz.sort(key=lambda x: -x[1])
        return nz[:k]
