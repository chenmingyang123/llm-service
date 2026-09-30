"""W2 第 5 天 · 在 golden set 上对比四种检索模式。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.retrieval import BM25Index, RetrievalPipeline, get_store  # noqa: E402
from app.retrieval.rerank import build_reranker  # noqa: E402

DEFAULT_GOLDEN = os.path.join(ROOT, "data", "eval", "golden.jsonl")
MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def load_golden(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def evaluate(pipe: RetrievalPipeline, golden: list[dict], mode: str,
             pool: int = 50, topk: int = 5) -> dict:
    """跑一个模式，返回总体与分类型的指标。"""
    hits = {1: 0, 3: 0, 5: 0}
    mrr = 0.0
    by_type: dict[str, dict] = {}
    lat = []
    detail = []

    for g in golden:
        t0 = time.perf_counter()
        res = pipe.search(g["question"], mode=mode, topk=topk, pool=pool)
        lat.append((time.perf_counter() - t0) * 1000)

        ids = [h["chunk_id"] for h in res["hits"]]
        rank = ids.index(g["expect_chunk"]) + 1 if g["expect_chunk"] in ids else 0
        for k in hits:
            if rank and rank <= k:
                hits[k] += 1
        mrr += (1.0 / rank) if rank else 0.0

        t = g.get("type", "?")
        b = by_type.setdefault(t, {"n": 0, "hit1": 0, "hit5": 0, "mrr": 0.0})
        b["n"] += 1
        b["hit1"] += 1 if rank == 1 else 0
        b["hit5"] += 1 if (rank and rank <= 5) else 0
        b["mrr"] += (1.0 / rank) if rank else 0.0

        detail.append({"qid": g["qid"], "rank": rank, "type": t})

    n = len(golden)
    for t, b in by_type.items():
        b["hit1"] = b["hit1"] / b["n"]
        b["hit5"] = b["hit5"] / b["n"]
        b["mrr"] = b["mrr"] / b["n"]
    return {
        "mode": mode, "n": n,
        "hit@1": hits[1] / n, "hit@3": hits[3] / n, "hit@5": hits[5] / n,
        "mrr": mrr / n,
        "latency_ms": sum(lat) / len(lat),
        "by_type": by_type, "detail": detail,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--pool", type=int, default=50, help="粗排候选池")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--rerank", default="identity", choices=["identity", "llm"],
                    help="hybrid_rerank 模式用哪种重排器；identity=不重排（基线）")
    ap.add_argument("--skip", default="", help="跳过某些模式，逗号分隔")
    ap.add_argument("--out", default=None, help="导出 Markdown 报告")
    a = ap.parse_args()

    golden = load_golden(os.path.join(ROOT, a.golden) if not os.path.isabs(a.golden) else a.golden)
    print(f"golden set {len(golden)} 条　粗排池 {a.pool}　topk {a.topk}\n")

    store = get_store()
    print(f"向量库 {len(store.rows)} 条 × {store.dim} 维")
    t0 = time.perf_counter()
    bm25 = BM25Index([r["text"] for r in store.rows])
    print(f"BM25 索引 {time.perf_counter() - t0:.2f}s\n")

    pipe = RetrievalPipeline(store=store, bm25=bm25, backend="auto",
                             reranker=build_reranker(a.rerank))

    skip = {s for s in a.skip.split(",") if s}
    results = []
    for mode in MODES:
        if mode in skip:
            continue
        r = evaluate(pipe, golden, mode, pool=a.pool, topk=a.topk)
        results.append(r)
        print(f"{mode:<14} hit@1 {r['hit@1']*100:5.1f}%  hit@3 {r['hit@3']*100:5.1f}%  "
              f"hit@5 {r['hit@5']*100:5.1f}%  MRR {r['mrr']:.3f}  "
              f"{r['latency_ms']:.0f} ms/次")

    print("\n" + "=" * 74)
    print("按问题类型拆开（term=含精确术语 / semantic=同义改写）")
    print("=" * 74)
    print(f"{'模式':<14}{'term hit@1':>12}{'term MRR':>11}{'sem hit@1':>12}{'sem MRR':>11}")
    print("-" * 74)
    for r in results:
        tm = r["by_type"].get("term", {})
        sm = r["by_type"].get("semantic", {})
        print(f"{r['mode']:<14}{(tm.get('hit1', 0))*100:>11.1f}%"
              f"{tm.get('mrr', 0):>11.3f}"
              f"{(sm.get('hit1', 0))*100:>11.1f}%{sm.get('mrr', 0):>11.3f}")

    base = next((r for r in results if r["mode"] == "vector"), None)
    hyb = next((r for r in results if r["mode"] == "hybrid"), None)
    if base and hyb:
        print("\n" + "-" * 74)
        print(f"混合 vs 纯向量：hit@1 {hyb['hit@1']*100:.1f}% vs {base['hit@1']*100:.1f}% "
              f"（{(hyb['hit@1']-base['hit@1'])*100:+.1f} 个百分点）"
              f"　MRR {hyb['mrr']:.3f} vs {base['mrr']:.3f}（{hyb['mrr']-base['mrr']:+.3f}）")
    rr = next((r for r in results if r["mode"] == "hybrid_rerank"), None)
    if hyb and rr:
        print(f"重排 vs 混合  ：hit@1 {rr['hit@1']*100:.1f}% vs {hyb['hit@1']*100:.1f}% "
              f"（{(rr['hit@1']-hyb['hit@1'])*100:+.1f}）　"
              f"延迟 {rr['latency_ms']:.0f} vs {hyb['latency_ms']:.0f} ms")

    print("\n这是 golden set 上的正式评测，不是冒烟测试 —— 但 29 条仍然偏小，")
    print("差 1 条就是 3.4 个百分点，别对小差距下结论。")

    if a.out:
        _write_report(os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out,
                      results, len(golden), a.pool, a.topk)


def _write_report(path: str, results: list[dict], n: int, pool: int, topk: int) -> None:
    L = ["# 检索模式评测（W2 第 5 天）", "",
         f"- golden set：{n} 条（term {sum(r['by_type'].get('term', {}).get('n', 0) for r in results[:1])}"
         f" / semantic {sum(r['by_type'].get('semantic', {}).get('n', 0) for r in results[:1])}）",
         f"- 粗排候选池 {pool}，最终返回 {topk}",
         "- 指标：hit@k = 标准答案块是否进 top-k；MRR = 首次命中位置的倒数均值",
         "", "| 模式 | hit@1 | hit@3 | hit@5 | MRR | 延迟 ms |", "|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['mode']} | {r['hit@1']*100:.1f}% | {r['hit@3']*100:.1f}% | "
                 f"{r['hit@5']*100:.1f}% | {r['mrr']:.3f} | {r['latency_ms']:.0f} |")
    L += ["", "## 按问题类型拆开", "",
          "| 模式 | term hit@1 | term MRR | semantic hit@1 | semantic MRR |", "|---|---|---|---|---|"]
    for r in results:
        tm = r["by_type"].get("term", {})
        sm = r["by_type"].get("semantic", {})
        L.append(f"| {r['mode']} | {tm.get('hit1',0)*100:.1f}% | {tm.get('mrr',0):.3f} | "
                 f"{sm.get('hit1',0)*100:.1f}% | {sm.get('mrr',0):.3f} |")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n已写入 {path}")


if __name__ == "__main__":
    main()
