"""W2 第 3 天 · 分块落地 + Embedding 向量化。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

CORPUS = os.path.join(ROOT, "data", "parsed", "corpus.jsonl")
LEDGER = os.path.join(ROOT, ".cost_ledger.jsonl")

import chunk_explore as ce  # noqa: E402  （必须在 sys.path.insert 之后）

from app.retrieval.embedding import (  # noqa: E402
    BAILIAN_MAX_BATCH, BAILIAN_MODEL, BAILIAN_URL, ZHIPU_MODEL, ZHIPU_URL,
    embed, embed_bailian, embed_local, embed_zhipu, pick_backend,
)


MAX_CHARS = 1000

SPLIT_OVERLAP = 100


def _locate_section(doc: dict, chunk: str) -> list[str]:
    """反查这个块属于哪一节。为了 W2D7 的引用溯源，现在就得记下来。"""
    head = chunk[:30].strip()
    if not head:
        return []
    for sec in doc["sections"]:
        if head in sec["text"]:
            return list(sec["path"])
    return []


def build_chunks(docs: list[dict], strategy: str, size: int,
                 max_chars: int = MAX_CHARS) -> list[dict]:
    """把文档切成带元数据的块。超大块拆成父子结构。"""
    fn = ce.make_strategies(size, size // 10)[strategy]
    chunks: list[dict] = []
    n_parent = n_child = 0

    for d in docs:
        raw = fn(d)
        for i, text in enumerate(raw):
            t = text.strip()
            if not t:
                continue
            cid = f"{d['doc_id']}-{i:04d}"
            base = {
                "chunk_id": cid, "doc_id": d["doc_id"], "source": d["source"],
                "title": d["title"], "url": d["url"],
                "doc_type": d["doc_type"],
                "section_path": _locate_section(d, t),
                "strategy": strategy,
            }

            if len(t) <= max_chars:
                chunks.append({**base, "role": "normal", "parent_id": None,
                               "text": t, "n_chars": len(t), "vectorize": True})
                continue

            n_parent += 1
            chunks.append({**base, "role": "parent", "parent_id": None,
                           "text": t, "n_chars": len(t), "vectorize": False})
            step = max(1, max_chars - SPLIT_OVERLAP)
            for j, s in enumerate(range(0, len(t), step)):
                piece = t[s:s + max_chars]
                if not piece.strip():
                    continue
                n_child += 1
                chunks.append({**base, "chunk_id": f"{cid}-c{j:02d}",
                               "role": "child", "parent_id": cid,
                               "text": piece, "n_chars": len(piece), "vectorize": True})

    print(f"  切块：{len(chunks)} 块（其中父块 {n_parent}、子块 {n_child}）")
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。零向量返回 0 而不是抛异常 —— 空块不该让整轮检索挂掉。"""
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def search(query_vec, rows, k=3):
    """暴力全量比对后取 top-k。"""
    scored = [(cosine(query_vec, r["vector"]), r) for r in rows]
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


def _guard_overwrite(chunks_path: str, n_new: int, force: bool = False) -> None:
    """别让一次小样本试跑把整库向量冲掉（这是真实踩过的坑）。"""
    if not os.path.exists(chunks_path):
        return
    n_old = 0
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n_old += 1
    if n_old <= n_new:
        return
    if force:
        print(f"  [warn] --force：覆盖已有向量库（旧 {n_old} 块 → 新 {n_new} 块）")
        return
    sys.exit(f"已有向量库 {n_old} 块，这次只生成 {n_new} 块。\n"
             f"想覆盖必须显式 --force，否则原地实验会把整库冲掉。"
             f"（看错了就用 --out data/embed_test 换个目录）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--strategy", default="atomic",
                    choices=["fixed", "recursive", "structural", "atomic"])
    ap.add_argument("--size", type=int, default=500, help="目标块大小（字）")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS, help="超过就切子块")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "local", "zhipu", "bailian"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=os.path.join("data", "embed"))
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖已有的向量库（默认会拒绝把大的覆盖成小的）")
    ap.add_argument("--query", default=None, help="向量化后跑一次检索验证")
    ap.add_argument("--load", default=None,
                    help="从已有的 chunks.jsonl 加载做检索，不重新向量化（省钱）")
    ap.add_argument("--eval", default=None,
                    help="用冒烟测试集跑一轮，回答'管线通不通'（不是正式评测）")
    ap.add_argument("--topk", type=int, default=3)
    a = ap.parse_args()

    if a.load and (a.query or a.eval):
        path = a.load if os.path.isabs(a.load) else os.path.join(ROOT, a.load)
        chunks = [json.loads(l) for l in open(path, encoding="utf-8")]
        backend = pick_backend(a.backend)
        print(f"已加载 {len(chunks)} 块（其中有向量 {sum(1 for c in chunks if c['vector'])} 块）")
        if a.query:
            run_query(a.query, chunks, backend, a.topk)
        if a.eval:
            qs = [json.loads(l) for l in open(os.path.join(ROOT, a.eval), encoding="utf-8")]
            r = run_eval(qs, chunks, backend, k=a.topk)
            print("\n" + "=" * 66)
            print(f"冒烟测试　{a.eval}")
            print("-" * 66)
            for qid, rank in r["detail"]:
                print(f"  {qid}  {'命中 第' + str(rank) + '位' if rank else '未命中'}")
            print("-" * 66)
            print(f"  命中率@{a.topk}  {r['hit@%d' % a.topk] * 100:.1f}%   "
                  f"MRR {r['mrr']:.3f}   （{r['n']} 条）")
            print("  这是冒烟测试，不是正式评测 —— 别拿它当召回率汇报")
        return

    if not os.path.exists(CORPUS):
        print("没有语料，先跑：python scripts/parse_corpus.py")
        sys.exit(1)

    docs = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    if a.limit:
        docs = docs[:a.limit]
    print(f"语料 {len(docs)} 篇　策略 {a.strategy}　目标 {a.size} 字　上限 {a.max_chars} 字")

    chunks = build_chunks(docs, a.strategy, a.size, a.max_chars)
    todo = [c for c in chunks if c["vectorize"]]
    print(f"  待向量化 {len(todo)} 块，跳过父块 {len(chunks) - len(todo)} 块（只在检索命中后返回）")

    backend = pick_backend(a.backend)
    print(f"  后端 {backend}\n")
    vecs, stats = embed([c["text"] for c in todo], backend, batch=a.batch)

    assert len(vecs) == len(todo), f"向量数 {len(vecs)} != 块数 {len(todo)}"
    dim = len(vecs[0]) if vecs else 0

    it = iter(vecs)
    for c in chunks:
        c["vector"] = next(it).copy() if c["vectorize"] else None

    outdir = os.path.join(ROOT, a.out)
    os.makedirs(outdir, exist_ok=True)
    chunks_path = os.path.join(outdir, "chunks.jsonl")
    _guard_overwrite(chunks_path, len(chunks), a.force)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_docs": len(docs), "strategy": a.strategy, "size": a.size,
        "max_chars": a.max_chars,
        "n_chunks": len(chunks),
        "n_vectorized": len(todo),
        "n_parents": sum(1 for c in chunks if c["role"] == "parent"),
        "n_children": sum(1 for c in chunks if c["role"] == "child"),
        "dim": dim, **stats,
    }
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if stats["cost_cny"] > 0:
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": meta["created_at"], "provider": "zhipu", "model": ZHIPU_MODEL,
                "tokens": stats["tokens"], "cost_cny": stats["cost_cny"],
                "note": f"embed {len(todo)} chunks"}, ensure_ascii=False) + "\n")

    print()
    print("=" * 66)
    print(f"块总数      {meta['n_chunks']}")
    print(f"已向量化    {meta['n_vectorized']}　（父块 {meta['n_parents']} / 子块 {meta['n_children']}）")
    print(f"向量维度    {dim}")
    print(f"后端        {stats['backend']} / {stats.get('model')}")
    print(f"耗时        {stats['seconds']}s")
    print(f"花费        ¥{stats['cost_cny']}")
    print("=" * 66)
    print(f"chunks → {a.out}/chunks.jsonl")
    print(f"meta   → {a.out}/meta.json")

    if a.query:
        run_query(a.query, chunks, backend)


def resolve_parent(r: dict, chunks: list[dict]) -> dict:
    """命中子块时，真正喂给模型的是父块原文 —— small-to-big 的关键一步。"""
    if r["role"] != "child" or not r.get("parent_id"):
        return r
    p = next((c for c in chunks if c["chunk_id"] == r["parent_id"]), None)
    return p or r


def run_eval(queries: list[dict], chunks: list[dict], backend: str, k: int = 3) -> dict:
    """冒烟测试：这批问题的答案我凭 W1 的实测就知道，所以能自动判对错。"""
    rows = [c for c in chunks if c["vector"]]
    hit = mrr = 0.0
    detail = []
    for q in queries:
        qv = embed([q["question"]], backend)[0]
        hits = search(qv[0], rows, k=k)
        rank = 0
        for i, (_, r) in enumerate(hits, 1):
            full = resolve_parent(r, chunks)
            if any(kw in full["text"] for kw in q["expect_any"]):
                rank = i
                break
        if rank:
            hit += 1
            mrr += 1 / rank
        detail.append((q["qid"], rank))
    n = len(queries) or 1
    return {"hit@%d" % k: hit / n, "mrr": mrr / n, "detail": detail, "n": len(queries)}


def run_query(q: str, chunks: list[dict], backend: str, k: int = 3) -> None:
    """跑一次真实检索，证明向量是有意义的 —— 不然今天等于白做。"""
    rows = [c for c in chunks if c["vector"]]
    qv, _ = embed([q], backend)
    hits = search(qv[0], rows, k=k)
    print("\n" + "=" * 66)
    print(f"检索验证：「{q}」")
    print("-" * 66)
    for score, r in hits:
        head = " > ".join(r["section_path"]) or r["title"]
        print(f"\n[{score:.4f}] {r['source']} · {head}")
        if r["role"] == "child":
            parent = next((c for c in chunks if c["chunk_id"] == r["parent_id"]), None)
            print(f"           （子块，命中后返回父块 {r['parent_id']}，{parent['n_chars'] if parent else '?'} 字）")
        print("  " + r["text"][:160].replace("\n", " ") + "…")


if __name__ == "__main__":
    main()
