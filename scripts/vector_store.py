"""W2 第 4 天 · 向量库与纯向量检索 —— 把昨天的向量装进索引，跑出 baseline。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DEFAULT_CHUNKS = os.path.join(ROOT, "data", "embed", "chunks.jsonl")

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def load_vectors(path: str, limit: int | None = None):
    """读出所有带向量的块。父块 vector=None，它们只在命中子块后被返回。"""
    rows = []
    skipped = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            cut = line.find('{"chunk_id"')
            if cut <= 0:
                skipped += 1
                continue
            c = json.loads(line[cut:])
        if c.get("vector"):
            rows.append(c)
    if skipped:
        print(f"  跳过 {skipped} 行无法解析的记录")
    if limit:
        rows = rows[:limit]
    if np is not None:
        mat = np.asarray([r["vector"] for r in rows], dtype=np.float32)
    else:
        mat = [r["vector"] for r in rows]
    return mat, rows


def _normalize(mat):
    """归一化后余弦相似度就等于点积，检索时省掉每次开方。"""
    if np is None:
        return mat
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return mat / norm


class BruteForceIndex:
    """全量比对。召回率恒为 100%，延迟恒为最慢 —— 它就是那把尺子。"""

    name = "brute"

    def __init__(self, mat, rows):
        self.mat = _normalize(mat)
        self.rows = rows

    def search(self, qvec, k=10, **_):
        if np is None:
            scored = sorted(range(len(self.rows)),
                            key=lambda i: -_cos(qvec, self.rows[i]["vector"]))[:k]
            return [(i, _cos(qvec, self.rows[i]["vector"])) for i in scored]
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self.mat @ q
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def stats(self):
        return {"index": self.name, "n": len(self.rows)}


class IVFIndex:
    """倒排索引：先聚类，查询时只搜最近的几个簇。"""

    name = "ivf"

    def __init__(self, mat, rows, nlist=64, nprobe=4, iters=12, seed=42):
        self.rows = rows
        self.nlist = nlist
        self.nprobe = nprobe
        mat = _normalize(mat)

        if np is None:
            raise RuntimeError("IVF 需要 numpy：pip install numpy -i https://pypi.org/simple")

        self.mat = mat

        n = len(mat)
        self.nlist = max(1, min(nlist, n))
        rng = np.random.default_rng(seed)
        init = rng.choice(n, size=self.nlist, replace=False)
        centers = mat[init].copy()

        for _ in range(iters):
            d = (_sq(mat)[:, None] - 2.0 * (mat @ centers.T) + _sq(centers)[None, :])
            labels = d.argmin(axis=1)
            for j in range(self.nlist):
                m = labels == j
                if m.any():
                    centers[j] = mat[m].mean(axis=0)
                else:
                    centers[j] = mat[rng.integers(0, n)]

        self.centers = centers
        self.labels = labels
        self.buckets = [np.flatnonzero(labels == j) for j in range(self.nlist)]

    def search(self, qvec, k=10, nprobe: int | None = None):
        np_ = nprobe or self.nprobe
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)

        d = _sq(self.centers) - 2.0 * (self.centers @ q)
        near = np.argpartition(d, min(np_, self.nlist) - 1)[:np_]

        cand = np.concatenate([self.buckets[j] for j in near if len(self.buckets[j])])
        if cand.size == 0:
            cand = np.arange(len(self.rows))
        sims = self.mat[cand] @ q
        take = min(k, cand.size)
        top = np.argpartition(-sims, take - 1)[:take]
        top = top[np.argsort(-sims[top])]
        return [(int(cand[i]), float(sims[i])) for i in top]

    def stats(self):
        sizes = [len(b) for b in self.buckets]
        return {"index": self.name, "n": len(self.rows), "nlist": self.nlist,
                "nprobe": self.nprobe,
                "bucket_min": min(sizes), "bucket_max": max(sizes),
                "bucket_avg": round(sum(sizes) / len(sizes), 1)}


def _sq(a):
    return (a * a).sum(axis=1)


def perturb(vec, sigma: float = 0.05, seed: int = 0) -> list[float]:
    """把一个块向量变成"近似但不等同"的查询向量。"""
    if np is None or sigma <= 0:
        return list(vec)
    rng = np.random.default_rng(seed)
    v = np.asarray(vec, dtype=np.float64) + rng.normal(0, sigma, size=len(vec))
    v = v / (np.linalg.norm(v) or 1.0)
    return v.astype(np.float32).tolist()


def _cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    return num / (da * db) if da and db else 0.0


class MilvusIndex:
    """Milvus Lite —— 计划里指定的方案，HNSW 索引。"""

    name = "milvus"

    def __init__(self, mat, rows, db_path=None, m=16, ef_construction=200,
                 ef=64, **_):
        from pymilvus import MilvusClient
        from pymilvus.milvus_client.index import IndexParams

        self.rows = rows
        self.db = db_path or os.path.join(ROOT, "data", "milvus.db")
        os.makedirs(os.path.dirname(self.db), exist_ok=True)
        self.client = MilvusClient(uri=self.db)
        dim = len(rows[0]["vector"])

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        import uuid
        self.collection = f"rag_chunks_{dim}_{stamp}_{uuid.uuid4().hex[:6]}"

        self.client.create_collection(
            collection_name=self.collection, dimension=dim,
            metric_type="COSINE", auto_id=True,
        )
        self.ef = ef

        batch = 500
        for i in range(0, len(rows), batch):
            part = rows[i:i + batch]
            self.client.insert(self.collection, [{
                "vector": r["vector"], "chunk_id": r["chunk_id"],
                "source": r["source"], "role": r["role"],
                "parent_id": r["parent_id"] or "",
            } for r in part])

        self.client.load_collection(self.collection)
        self.client.release_collection(self.collection)
        self.client.drop_index(self.collection, index_name="vector")
        ip = IndexParams()
        ip.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                     index_name="hnsw", params={"M": m, "efConstruction": ef_construction})
        self.client.create_index(collection_name=self.collection, index_params=ip)
        self.client.load_collection(self.collection)

    def search(self, qvec, k=10, ef=None):
        res = self.client.search(
            self.collection, data=[list(qvec)], limit=k,
            search_params={"params": {"ef": ef or self.ef}},
            output_fields=["chunk_id"])
        out = []
        for hit in res[0]:
            out.append((self._idx_of(hit["entity"]["chunk_id"]), float(hit["distance"])))
        return out

    def search_batch(self, qvecs, k=10, ef=None):
        """一次调用查多条。批量口径专用，见 _batch_ms 的说明。"""
        res = self.client.search(
            self.collection, data=[list(q) for q in qvecs], limit=k,
            search_params={"params": {"ef": ef or self.ef}},
            output_fields=["chunk_id"])
        out = [[(self._idx_of(h["entity"]["chunk_id"]), float(h["distance"]))
                for h in group] for group in res]
        for group in out:
            group.sort(key=lambda x: -x[1])
        return out

    def _idx_of(self, chunk_id):
        if not hasattr(self, "_map"):
            self._map = {r["chunk_id"]: i for i, r in enumerate(self.rows)}
        return self._map.get(chunk_id, -1)

    def stats(self):
        return {"index": self.name, "n": len(self.rows), "ef": self.ef,
                "db": os.path.relpath(self.db, ROOT).replace("\\", "/")}


def recall_at_k(approx: list[int], exact: list[int], k: int) -> float:
    """索引结果跟暴力结果重合多少。"""
    a, e = set(approx[:k]), set(exact[:k])
    return len(a & e) / k if k else 0.0


def build(name: str, mat, rows, **kw):
    if name == "brute":
        return BruteForceIndex(mat, rows)
    if name == "ivf":
        return IVFIndex(mat, rows, **kw)
    if name == "milvus":
        return MilvusIndex(mat, rows, **kw)
    raise ValueError(f"未知后端：{name}")


def run_sweep(mat, rows, queries, k=10, nlist_grid=(32, 64, 128),
              nprobe_grid=(1, 2, 4, 8, 16)):
    """参数扫描：把"召回 vs 延迟"这条曲线打出来。"""
    base = BruteForceIndex(mat, rows)
    truth = [[i for i, _ in base.search(q, k=k)] for q in queries]
    t0 = time.perf_counter()
    for q in queries:
        base.search(q, k=k)
    base_ms = (time.perf_counter() - t0) / max(1, len(queries)) * 1000

    print(f"{'nlist':>6}{'nprobe':>8}{'召回@' + str(k):>10}{'延迟ms':>10}{'加速':>8}{'簇均大小':>10}")
    print("-" * 56)
    print(f"{'—':>6}{'暴力':>8}{1.0:>10.3f}{base_ms:>10.2f}{1.0:>7.1f}x{'—':>10}")
    out = []
    for nlist in nlist_grid:
        idx = IVFIndex(mat, rows, nlist=nlist, nprobe=1)
        st = idx.stats()
        for nprobe in nprobe_grid:
            rs = []
            t0 = time.perf_counter()
            for q, exact in zip(queries, truth):
                got = [i for i, _ in idx.search(q, k=k, nprobe=nprobe)]
                rs.append(recall_at_k(got, exact, k))
            ms = (time.perf_counter() - t0) / len(queries) * 1000
            rec = sum(rs) / len(rs)
            out.append((nlist, nprobe, rec, ms))
            print(f"{nlist:>6}{nprobe:>8}{rec:>10.3f}{ms:>10.2f}"
                  f"{base_ms / ms if ms else 0:>7.1f}x{st['bucket_avg']:>10}")
    return base_ms, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=DEFAULT_CHUNKS)
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 个向量（验证用）")
    ap.add_argument("--backend", default="ivf", choices=["brute", "ivf", "milvus"])
    ap.add_argument("--nlist", type=int, default=64)
    ap.add_argument("--nprobe", type=int, default=4)
    ap.add_argument("--ef", type=int, default=64, help="Milvus HNSW 查询期参数")
    ap.add_argument("--sweep", action="store_true", help="跑参数扫描")
    ap.add_argument("--compare", action="store_true",
                    help="暴力 / IVF / Milvus 三种索引在同一组查询上的对比")
    ap.add_argument("--warmup", type=int, default=5,
                    help="计时前先跑几条空转查询（Milvus 冷启动首查很慢，不预热会毁掉均值）")
    ap.add_argument("--nq", type=int, default=30, help="自采样查询条数")
    ap.add_argument("--noise", type=float, default=0.05,
                    help="自采样查询向量的噪声强度（0 等于直接拿块向量，召回会虚高）")
    ap.add_argument("--eval", default=None, help="冒烟测试集")
    ap.add_argument("--query", default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", default=None, help="导出 Markdown 报告")
    a = ap.parse_args()

    if not os.path.exists(a.chunks):
        print(f"没有 {a.chunks}，先跑：python scripts/embed_corpus.py")
        sys.exit(1)

    mat, rows = load_vectors(a.chunks, a.limit)
    print(f"载入 {len(rows)} 个向量（维度 {len(rows[0]['vector'])}）")

    qs: list[list[float]] = []
    labels: list[str] = []
    if a.eval or a.query:
        import embed_corpus as ec
        backend = ec.pick_backend("auto")
        texts = []
        if a.query:
            texts.append(a.query)
            labels.append(a.query)
        if a.eval:
            for line in open(os.path.join(ROOT, a.eval), encoding="utf-8"):
                texts.append(json.loads(line)["question"])
                labels.append(json.loads(line)["qid"])
        qs = ec.embed(texts, backend)[0]

    if not qs and (a.compare or a.sweep):
        import random
        random.seed(0)
        idxs = random.sample(range(len(rows)), min(a.nq, len(rows)))
        qs = [perturb(rows[i]["vector"], a.noise) for i in idxs]
        labels = [f"自采样#{i}" for i in idxs]

    if a.compare:
        _compare(mat, rows, qs, a.topk, a.nlist, a.nprobe, a.warmup)
        return

    if a.sweep:
        base_ms, table = run_sweep(mat, rows, qs, k=a.topk)
        print("\n怎么读：召回越接近 1.0 越好，延迟越低越好。两者拉扯，选一个点。")
        if a.out:
            _write_report(a.out, len(rows), base_ms, table, a.topk)
        return

    t0 = time.perf_counter()
    idx = build(a.backend, mat, rows, nlist=a.nlist, nprobe=a.nprobe, ef=a.ef)
    build_s = time.perf_counter() - t0
    st = idx.stats()
    print(f"建索引 {a.backend} 用时 {build_s:.2f}s　{st}")

    if qs:
        print("\n" + "=" * 62)
        for lab, q in zip(labels, qs):
            hits = idx.search(q, k=a.topk)
            print(f"\n{lab}")
            for rank, (i, score) in enumerate(hits[:3], 1):
                r = rows[i]
                path = " > ".join(r.get("section_path") or []) or r["title"]
                print(f"  {rank}. [{score:.4f}] {r['source']} · {path[:44]}")
                if r["role"] == "child":
                    p = next((x for x in rows if x["chunk_id"] == r["parent_id"]), None)
                    if p:
                        print(f"       （子块，返回父块原文 {len(p['text'])} 字）")

        base = BruteForceIndex(mat, rows)
        recs = [recall_at_k([i for i, _ in idx.search(q, k=a.topk)],
                            [i for i, _ in base.search(q, k=a.topk)], a.topk) for q in qs]
        t0 = time.perf_counter()
        for q in qs:
            idx.search(q, k=a.topk)
        ms = (time.perf_counter() - t0) / len(qs) * 1000
        t0 = time.perf_counter()
        for q in qs:
            base.search(q, k=a.topk)
        bms = (time.perf_counter() - t0) / len(qs) * 1000
        print("\n" + "-" * 62)
        print(f"召回@{a.topk}  {sum(recs) / len(recs):.3f}　（相对暴力检索）")
        print(f"延迟      {ms:.2f} ms/次　暴力 {bms:.2f} ms/次　加速 {bms / ms if ms else 0:.1f}×")


def _batch_ms(idx, queries, k) -> float:
    """批量口径：一次调用塞进去一批查询，算单条均摊耗时。"""
    data = [list(q) for q in queries[:20]]
    if not data or not hasattr(idx, "search_batch"):
        return 0.0
    idx.search_batch(data[:1], k=k)
    t0 = time.perf_counter()
    idx.search_batch(data, k=k)
    return (time.perf_counter() - t0) / len(data) * 1000


def _compare(mat, rows, queries, k, nlist=64, nprobe=4, warmup=5):
    """三种索引拉到同一组查询上公平对比。"""
    base = BruteForceIndex(mat, rows)
    truth = [[i for i, _ in base.search(q, k=k)] for q in queries]

    def timed(idx, warmup: int = 0):
        """返回 (平均召回, 单次延迟ms)。延迟只计 search，不计建索引。"""
        for q in queries[:warmup]:
            idx.search(q, k=k)
        recs = []
        t0 = time.perf_counter()
        for q, exact in zip(queries, truth):
            got = [i for i, _ in idx.search(q, k=k)]
            recs.append(recall_at_k(got, exact, k))
        ms = (time.perf_counter() - t0) / len(queries) * 1000
        return sum(recs) / len(recs), ms

    def timed_build(cls, **kw):
        t0 = time.perf_counter()
        idx = cls(mat, rows, **kw) if kw else cls(mat, rows)
        return idx, time.perf_counter() - t0

    print(f"\n{'索引':<16}{'召回@' + str(k):>10}{'延迟ms':>10}{'加速':>9}{'建索引s':>9}")
    print("-" * 58)
    b_rec, b_ms = timed(base)
    print(f"{'暴力全量':<16}{b_rec:>10.3f}{b_ms:>10.2f}{1.0:>8.1f}x{0.0:>9.2f}")

    label = f"IVF nlist={nlist} nprobe={nprobe}"
    ivf, t_build = timed_build(IVFIndex, nlist=nlist, nprobe=nprobe)
    r, ms = timed(ivf)
    print(f"{label:<16}{r:>10.3f}{ms:>10.2f}{b_ms / ms if ms else 0:>8.1f}x{t_build:>9.2f}")

    try:
        mv, t_build = timed_build(MilvusIndex)
        r, ms_cold = timed(mv)
        _, ms = timed(mv, warmup=warmup)
        batch_ms = _batch_ms(mv, queries, k)
        print(f"{'Milvus (HNSW)':<16}{r:>10.3f}{ms:>10.2f}"
              f"{b_ms / ms if ms else 0:>8.1f}x{t_build:>9.2f}"
              f"　冷启动首查 {ms_cold:.0f}ms")
        print(f"{'　 └ 批量口径':<16}{'':>10}{batch_ms:>10.2f}"
              f"{b_ms / batch_ms if batch_ms else 0:>8.1f}x{'一次调用':>9}")
    except Exception as e:
        print(f"{'Milvus (HNSW)':<16}{'不可用':>10}{'-':>10}{'-':>9}{'-':>9}")
        print(f"  原因：{type(e).__name__}: {str(e)[:80]}")

    print("\n建索引时间单独一列，不算进延迟 —— 索引是一次性成本，查询是每次都要付的。")
    print("看加速比时别把建索引时间摊进去，否则建得越大的索引看起来越亏（其实不是）。")


def _write_report(path, n, base_ms, table, k) -> None:
    L = ["# 向量索引参数扫描（W2 第 4 天）", "",
         f"- 向量数：{n}　维度：2048",
         f"- 基准：暴力检索 {base_ms:.2f} ms/次，召回恒为 1.0",
         f"- 指标：召回@{k} = 索引 top-{k} 与暴力 top-{k} 的重合比例",
         "", "| nlist | nprobe | 召回 | 延迟(ms) | 加速 |", "|---|---|---|---|---|",
         "| — | 暴力 | 1.000 | %.2f | 1.0x |" % base_ms]
    for nlist, nprobe, rec, ms in table:
        L.append(f"| {nlist} | {nprobe} | {rec:.3f} | {ms:.2f} | {base_ms / ms if ms else 0:.1f}x |")
    L += ["", "## 怎么选", "",
          "nprobe 越大越接近暴力（召回↑ 延迟↑）。"
          "业务能接受的召回下限决定 nprobe 的下限，",
          "实时性要求决定它的上限。两个约束之间能取到值，这组参数就是可行的。"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n已写入 {path}")


if __name__ == "__main__":
    main()
