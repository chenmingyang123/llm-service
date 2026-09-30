"""向量库封装：把 chunks.jsonl 变成可检索的对象，且只加载一次。"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CHUNKS = os.path.join(ROOT, "data", "embed", "chunks.jsonl")


class ChunkStore:
    """一次加载，多次查询。"""

    def __init__(self, chunks_path: str = DEFAULT_CHUNKS):
        self.path = chunks_path
        self.rows: list[dict] = []
        self.by_id: dict[str, dict] = {}

        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    cut = line.find('{"chunk_id"')
                    if cut <= 0:
                        continue
                    c = json.loads(line[cut:])
                self.by_id[c["chunk_id"]] = c
                if c.get("vector"):
                    self.rows.append(c)

        if not self.rows:
            raise RuntimeError(f"{chunks_path} 里没有带向量的块")

        mat = np.asarray([r["vector"] for r in self.rows], dtype=np.float32)
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        self.mat = mat / norm
        self.dim = mat.shape[1]
        self._by_id_all = self.by_id

    def vector_search(self, qvec: list[float], k: int = 10) -> list[tuple[int, float]]:
        """暴力余弦检索。返回 [(row_index, score), ...]，按分数降序。"""
        q = np.asarray(qvec, dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n == 0:
            return []
        q = q / n
        sims = self.mat @ q
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def text_of(self, i: int) -> str:
        return self.rows[i].get("text", "")

    def resolve_parent(self, i: int) -> tuple[int, str]:
        """small-to-big：命中子块时，真正该喂给模型的是父块原文。"""
        r = self.rows[i]
        if r.get("role") == "child" and r.get("parent_id"):
            p = self.by_id.get(r["parent_id"])
            if p:
                return i, p.get("text", r["text"])
        return i, r.get("text", "")


@lru_cache(maxsize=4)
def get_store(chunks_path: str = DEFAULT_CHUNKS) -> ChunkStore:
    """进程内单例。测试里用 get_store.cache_clear() 重置。"""
    return ChunkStore(chunks_path)
