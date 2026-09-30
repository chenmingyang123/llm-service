"""W3 第 1 天 · 五指标评测：一次跑出检索 + 生成 + 工程的完整画像。"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.evals import answer_relevancy, context_recall, faithfulness  # noqa: E402
from app.rag import AskPipeline  # noqa: E402
from app.retrieval import BM25Index, RetrievalPipeline, get_store  # noqa: E402
from app.retrieval.rerank import build_reranker  # noqa: E402
from app.retrieval.strategies import build_retriever  # noqa: E402

DEFAULT_GOLDEN = os.path.join(ROOT, "data", "eval", "golden.jsonl")


def pct(xs: list[float], p: float) -> float:
    """第 p 百分位。用最近秩法（和 run_baseline.py 一致）。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def ground_truth_of(store, g: dict) -> str:
    """取期望 chunk 的全文作为标准答案（context_recall 用）。"""
    cid = g.get("expect_chunk")
    if cid and cid in store.by_id:
        return store.by_id[cid].get("text", "")
    return ""


def eval_one(ask: AskPipeline, retriever, store, g: dict, *, mode: str,
             topk: int, pool: int, judge: bool, concurrency: int,
             provider: str, model: str | None, skip_gen: bool = False) -> dict:
    """跑一条题，返回这条题的完整指标。"""
    q = g["question"]
    spent0 = getattr(retriever, "spent", 0.0)

    rr = retriever.search(q, mode=mode, topk=topk, pool=pool)
    ids = [h["chunk_id"] for h in rr["hits"]]
    recall = 1.0 if g["expect_chunk"] in ids else 0.0
    multi_all = None
    if g.get("expect_chunks"):
        multi_all = 1.0 if all(c in ids for c in g["expect_chunks"]) else 0.0

    if skip_gen:
        return {"qid": g["qid"], "type": g.get("type", ""),
                "recall": recall, "multi_all": multi_all,
                "abstained": False, "latency_ms": rr.get("latency_ms", 0.0),
                "cost_retrieve_cny": round(getattr(retriever, "spent", 0.0) - spent0, 6),
                "cost_ask_cny": 0.0, "cost_judge_cny": 0.0,
                "cost_cny": round(getattr(retriever, "spent", 0.0) - spent0, 6)}

    r = ask.ask(q, mode=mode, topk=topk, pool=pool)
    contexts = r.get("context") or []
    cost_ask = r.get("cost_cny") or 0.0
    cost_retrieve = getattr(retriever, "spent", 0.0) - spent0
    cost_judge = 0.0

    row: dict = {
        "qid": g["qid"], "type": g.get("type", ""),
        "recall": recall, "multi_all": multi_all,
        "abstained": r.get("abstained", False),
        "latency_ms": r.get("latency_ms", 0.0),
        "cost_retrieve_cny": round(cost_retrieve, 6),
        "cost_ask_cny": round(cost_ask, 6),
        "cost_judge_cny": 0.0,
    }

    if judge:
        if not r.get("abstained") and r.get("answer"):
            f = faithfulness(r["answer"], contexts, provider=provider,
                             model=model, concurrency=concurrency)
            a = answer_relevancy(q, r["answer"], provider=provider,
                                 model=model, concurrency=concurrency)
            row["faithfulness"] = f["score"]
            row["answer_relevancy"] = a["score"]
            cost_judge += f["cost_cny"] + a["cost_cny"]
        else:
            row["faithfulness"] = None
            row["answer_relevancy"] = None

        gt = ground_truth_of(store, g)
        if gt:
            c = context_recall(gt, contexts, provider=provider,
                               model=model, concurrency=concurrency)
            row["context_recall"] = c["score"]
            cost_judge += c["cost_cny"]
        else:
            row["context_recall"] = None

    row["cost_judge_cny"] = round(cost_judge, 6)
    row["cost_cny"] = round(cost_retrieve + cost_ask + cost_judge, 6)
    return row


def aggregate(rows: list[dict], n_total: int) -> dict:
    """把逐题明细聚合成五指标。"""
    def avg(key: str) -> float | None:
        xs = [r[key] for r in rows if r.get(key) is not None]
        return sum(xs) / len(xs) if xs else None

    lats = [r["latency_ms"] for r in rows]
    costs = [r["cost_cny"] for r in rows]
    multi = [r["multi_all"] for r in rows if r.get("multi_all") is not None]
    return {
        "recall_at_5": sum(r["recall"] for r in rows) / n_total,
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_recall": avg("context_recall"),
        "abstain_rate": sum(1 for r in rows if r["abstained"]) / n_total,
        "multi_all_hit": (sum(multi) / len(multi)) if multi else None,
        "lat_p50_ms": pct(lats, 50), "lat_p95_ms": pct(lats, 95),
        "lat_max_ms": max(lats) if lats else 0.0,
        "cost_avg_cny": sum(costs) / n_total if costs else 0.0,
        "cost_total_cny": sum(costs),
        "cost_retrieve_avg_cny": sum(r.get("cost_retrieve_cny", 0.0) for r in rows) / n_total,
        "cost_ask_avg_cny": sum(r.get("cost_ask_cny", 0.0) for r in rows) / n_total,
        "cost_judge_avg_cny": sum(r.get("cost_judge_cny", 0.0) for r in rows) / n_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid", "hybrid_rerank"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量），先跑 5 条验证逻辑")
    ap.add_argument("--skip-judge", action="store_true", help="只跑检索指标，跳过 LLM-as-judge（省钱快速验证）")
    ap.add_argument("--concurrency", type=int, default=4, help="judge 判定并发度")
    ap.add_argument("--strategy", default="baseline",
                    choices=["baseline", "multiquery", "hyde", "parentdedup",
                             "routing", "selfquery", "rerank"],
                    help="检索策略（W3 两轮优化）")
    ap.add_argument("--n-variants", type=int, default=3, help="MultiQuery 的改写条数")
    ap.add_argument("--skip-gen", action="store_true",
                    help="只跑检索指标，不生成不 judge —— 策略筛选阶段用这个："
                         "几秒出结果，成本几乎为零")
    ap.add_argument("--types", default="",
                    help="只跑指定类型（逗号分隔），如 multi,boundary,conflict。"
                         "策略必须在其对症的题型上测，否则会得出错误结论")
    ap.add_argument("--rrf-k", type=int, default=60,
                    help="RRF 融合的 k（默认 60）。k 越小，名次靠前的优势越大")
    ap.add_argument("--weights", default="",
                    help="向量/BM25 两路的权重，逗号分隔，如 1.0,0.5。"
                         "零成本调参：不增加任何 LLM 调用")
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    w = None
    if a.weights:
        try:
            w = tuple(float(x) for x in a.weights.split(","))
            assert len(w) == 2
        except (ValueError, AssertionError):
            raise SystemExit("--weights 要写成两个数字，如 1.0,0.5")

    golden = [json.loads(l) for l in open(a.golden, encoding="utf-8") if l.strip()]
    if a.types:
        want = {t.strip() for t in a.types.split(",") if t.strip()}
        golden = [g for g in golden if g.get("type") in want]
    if a.limit:
        golden = golden[: a.limit]
    judge = (not a.skip_judge) and (not a.skip_gen)
    print(f"golden {len(golden)} 条　策略={a.strategy}　mode={a.mode}　topk={a.topk}　"
          f"pool={a.pool}　生成={'关(仅检索)' if a.skip_gen else '开'}　"
          f"judge={'关' if not judge else '开'}　并发={a.concurrency}\n")

    store = get_store()
    bm25 = BM25Index([r["text"] for r in store.rows])
    base = RetrievalPipeline(store=store, bm25=bm25, backend="auto",
                             reranker=build_reranker("identity"),
                             rrf_k=a.rrf_k, rrf_weights=w)
    retriever = build_retriever(a.strategy, base, n=a.n_variants, provider=a.provider)
    ask = AskPipeline(retriever)

    rows = []
    for i, g in enumerate(golden, 1):
        row = eval_one(ask, retriever, store, g, mode=a.mode, topk=a.topk, pool=a.pool,
                       judge=judge, concurrency=a.concurrency,
                       provider=a.provider, model=a.model, skip_gen=a.skip_gen)
        rows.append(row)
        if a.skip_gen and i % 20 == 0:
            print(f"  ...{i}/{len(golden)}")
        elif not a.skip_gen:
            flag = "拒" if row["abstained"] else "答"
            print(f"  [{i:>3}/{len(golden)}] {row['qid']} {flag} "
                  f"召回{'✓' if row['recall'] else '×'} "
                  f"faith={_fmt(row.get('faithfulness'))} "
                  f"rel={_fmt(row.get('answer_relevancy'))} "
                  f"crec={_fmt(row.get('context_recall'))} "
                  f"{row['latency_ms']:.0f}ms ¥{row['cost_cny']:.5f}")

    m = aggregate(rows, len(golden))

    print("\n" + "=" * 72)
    print("五指标汇总")
    print("=" * 72)
    print(f"  召回@5            {m['recall_at_5']*100:.1f}%")
    print(f"  忠实度 faithfulness  {_pct(m['faithfulness'])}   ← 真·忠实度（LLM-as-judge）")
    print(f"  答案相关性 relevancy {_pct(m['answer_relevancy'])}")
    print(f"  上下文召回 recall    {_pct(m['context_recall'])}")
    if m["multi_all_hit"] is not None:
        print(f"  跨文档全中          {m['multi_all_hit']*100:.1f}%")
    print(f"  拒答率            {m['abstain_rate']*100:.1f}%")
    print(f"  p50 {m['lat_p50_ms']:.0f}ms  p95 {m['lat_p95_ms']:.0f}ms  最慢 {m['lat_max_ms']:.0f}ms")
    print(f"  单次 ¥{m['cost_avg_cny']:.5f}　总 ¥{m['cost_total_cny']:.5f}")
    print(f"    ├ 检索/策略 ¥{m['cost_retrieve_avg_cny']:.5f}")
    print(f"    ├ 生成     ¥{m['cost_ask_avg_cny']:.5f}   ← 系统运行成本，和 W2 baseline 比这个")
    print(f"    └ 评测     ¥{m['cost_judge_avg_cny']:.5f}   ← judge 开销，不上生产")

    if a.out:
        path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "golden": len(golden), "strategy": a.strategy, "mode": a.mode,
                   "topk": a.topk, "pool": a.pool,
                   "judge": not a.skip_judge, "metrics": m, "per_question": rows}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"\n已写入 {path}")


def _fmt(v: float | None) -> str:
    return "  - " if v is None else f"{v:.2f}"


def _pct(v: float | None) -> str:
    return "  - " if v is None else f"{v*100:.1f}%"


if __name__ == "__main__":
    main()
