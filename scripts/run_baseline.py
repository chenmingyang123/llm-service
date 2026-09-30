"""W2 第 7 天 · 跑出 baseline 四项指标。"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.rag import AskPipeline  # noqa: E402
from app.retrieval import BM25Index, RetrievalPipeline, get_store  # noqa: E402
from app.retrieval.rerank import build_reranker  # noqa: E402

DEFAULT_GOLDEN = os.path.join(ROOT, "data", "eval", "golden.jsonl")
MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")
ENT = __import__("re").compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,}|\d+(?:\.\d+)?%?")
CITE = __import__("re").compile(r"\[\d{1,2}\]")


def faithfulness_floor(answer: str, texts: list[str]) -> float:
    """实体重叠算的忠实度**下界**。"""
    cleaned = CITE.sub(" ", answer)
    ents = set(ENT.findall(cleaned))
    if not ents:
        return 1.0
    blob = " ".join(CITE.sub(" ", t) for t in texts)
    return sum(1 for e in ents if e in blob) / len(ents)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def run_mode(ask: AskPipeline, retriever, golden: list[dict], mode: str,
             topk: int, pool: int) -> dict:
    recalls, faiths = [], []
    lats, costs = [], []
    multi_all, multi_n = 0, 0
    abstain_n = 0

    for g in golden:
        t0 = time.perf_counter()
        r = ask.ask(g["question"], mode=mode, topk=topk, pool=pool)
        lats.append((time.perf_counter() - t0) * 1000)
        costs.append(r["cost_cny"])

        rr = retriever.search(g["question"], mode=mode, topk=topk, pool=pool)
        ids = [h["chunk_id"] for h in rr["hits"]]
        recalls.append(1.0 if g["expect_chunk"] in ids else 0.0)

        need = g.get("expect_chunks")
        if need:
            multi_n += 1
            if all(c in ids for c in need):
                multi_all += 1

        if r["abstained"]:
            abstain_n += 1
            continue
        if r["citations"]:
            texts = [h for h in r.get("context", [])[: max(r["used"] or [1])]]
            faiths.append(faithfulness_floor(r["answer"], texts or [""]))

    n = len(golden)
    return {
        "mode": mode, "n": n,
        "recall@%d" % topk: sum(recalls) / n,
        "faithfulness_floor": (sum(faiths) / len(faiths)) if faiths else 0.0,
        "abstain_rate": abstain_n / n,
        "multi_all_hit": (multi_all / multi_n) if multi_n else None,
        "lat_p50_ms": pct(lats, 50), "lat_p95_ms": pct(lats, 95),
        "lat_max_ms": max(lats) if lats else 0.0,
        "cost_avg_cny": sum(costs) / n,
        "cost_total_cny": sum(costs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--modes", default="vector,bm25,hybrid")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--rerank", default="identity", choices=["identity", "llm"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    golden = [json.loads(l) for l in open(a.golden, encoding="utf-8") if l.strip()]
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    print(f"golden set {len(golden)} 条　topk={a.topk}　pool={a.pool}　"
          f"重排={a.rerank}\n")

    store = get_store()
    bm25 = BM25Index([r["text"] for r in store.rows])
    retriever = RetrievalPipeline(store=store, bm25=bm25, backend="auto",
                                  reranker=build_reranker(a.rerank))
    ask = AskPipeline(retriever)

    rows = []
    for m in modes:
        r = run_mode(ask, retriever, golden, m, a.topk, a.pool)
        rows.append(r)
        extra = ""
        if r["multi_all_hit"] is not None:
            extra = f"　跨文档全中 {r['multi_all_hit']*100:.0f}%"
        print(f"{m:<14} 召回 {r['recall@%d' % a.topk]*100:5.1f}%　"
              f"忠实度下界 {r['faithfulness_floor']:.3f}　"
              f"p50 {r['lat_p50_ms']:.0f}ms　p95 {r['lat_p95_ms']:.0f}ms　"
              f"¥{r['cost_avg_cny']:.5f}/次{extra}")

    print("\n" + "=" * 78)
    print(f"{'模式':<14}{'召回@'+str(a.topk):>9}{'忠实度':>9}{'p50ms':>8}"
          f"{'p95ms':>8}{'最慢ms':>8}{'单次成本':>10}{'拒答率':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['mode']:<14}{r['recall@%d' % a.topk]*100:>8.1f}%"
              f"{r['faithfulness_floor']:>9.3f}{r['lat_p50_ms']:>8.0f}"
              f"{r['lat_p95_ms']:>8.0f}{r['lat_max_ms']:>8.0f}"
              f"{r['cost_avg_cny']:>10.5f}{r['abstain_rate']*100:>7.1f}%")

    base = next((r for r in rows if r["mode"] == "hybrid"), rows[-1])
    print("\n" + "-" * 78)
    print(f"baseline 取 {base['mode']}：召回 {base['recall@%d' % a.topk]*100:.1f}%　"
          f"忠实度下界 {base['faithfulness_floor']:.3f}　"
          f"p95 {base['lat_p95_ms']:.0f}ms　单次 ¥{base['cost_avg_cny']:.5f}")
    print("\n两个必须记住的口径：")
    print("  忠实度是**下界**（实体重叠算的），不是严格忠实度 —— 要拿出去讲得补 LLM-as-judge")
    print(f"  {len(golden)} 条仍然偏小，差 1 条 = {100/len(golden):.1f} 个百分点")

    if a.out:
        path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "golden": len(golden), "topk": a.topk, "pool": a.pool,
                       "rerank": a.rerank, "rows": rows},
                      f, ensure_ascii=False, indent=1)
        print(f"\n已写入 {path}")


if __name__ == "__main__":
    main()
