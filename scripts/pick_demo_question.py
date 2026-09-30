"""为 demo 短片 ① 挑选演示题目（跨文档题 + 拒答题）。"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.rag import AskPipeline  # noqa: E402
from app.retrieval import BM25Index, RetrievalPipeline, get_store  # noqa: E402
from app.retrieval.rerank import build_reranker  # noqa: E402
from app.retrieval.strategies import build_retriever  # noqa: E402

ABSTAIN_CANDIDATES = [
    "企业版的年费是多少",
    "如果我的 API Key 泄露了，应该怎么找回",
    "你们支持私有化部署吗",
    "有没有面向教育行业的优惠政策",
    "开通服务需要签署合同吗",
]


def build_ask() -> AskPipeline:
    """构造和 D4 评测口径完全一致的链路。"""
    store = get_store()
    base = RetrievalPipeline(
        store=store,
        bm25=BM25Index([r["text"] for r in store.rows]),
        backend="auto",
        reranker=build_reranker("identity"),
    )
    return AskPipeline(build_retriever("rerank", base, provider="deepseek"))


def _f(obj, name: str, default=None):
    """兼容 dict 与 pydantic 对象取值。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def sources_of(res: dict) -> list[str]:
    """答案实际引用了哪几家来源（去重，保持出现顺序）。"""
    out: list[str] = []
    for c in res.get("citations") or []:
        s = _f(c, "source") or "?"
        if s not in out:
            out.append(s)
    return out


def score(row: dict) -> tuple[float, list[str]]:
    """给一条候选打分，返回 (分数, 理由)。分数只用来排序，不看绝对值。"""
    if row["abstained"]:
        return 0.0, ["拒答了，不能用来演示跨文档"]

    srcs = row["cited_sources"]
    if len(srcs) < 2:
        return 0.0, [f"只引用了 {len(srcs)} 家（{srcs}），跨文档主张不成立"]

    if row["unsupported"]:
        return 0.0, [f"有瞎标引用 {row['unsupported']}，不能上镜"]

    pts, why = 0.0, []
    n = len(row["answer"])

    if 150 <= n <= 600:
        pts += 3.0
        why.append(f"长度合适（{n} 字）")
    elif n < 150:
        pts += 1.0
        why.append(f"偏短（{n} 字）")
    else:
        pts += 1.5
        why.append(f"偏长（{n} 字，终端要拉高一点）")

    digits = sum(1 for ch in row["answer"] if ch.isdigit())
    if digits >= 4:
        pts += 2.0
        why.append(f"带 {digits} 个数字，画面有信息点")
    else:
        why.append("数字偏少")

    ncite = len(row["citations"])
    if 2 <= ncite <= 4:
        pts += 2.0
        why.append(f"引用 {ncite} 条，展开时画面不乱")
    else:
        pts += 0.5
        why.append(f"引用 {ncite} 条")

    per: dict[str, int] = {}
    for c in row["citations"]:
        s = _f(c, "source") or "?"
        per[s] = per.get(s, 0) + 1
    if per and min(per.values()) >= 1 and len(per) >= 2:
        pts += 1.5
        why.append("两家都有实质引用（" + "、".join(f"{k}×{v}" for k, v in per.items()) + "）")

    return pts, why


def score_abstain(runs: list[dict]) -> tuple[float, list[str]]:
    """给一条拒答候选打分。输入是它的 N 次重复结果，不是单次。"""
    if not runs:
        return 0.0, ["没有结果"]

    n = len(runs)
    n_abstain = sum(1 for r in runs if r["abstained"])
    if n_abstain < n:
        return 0.0, [f"{n} 次里只有 {n_abstain} 次拒答 —— 会录到「它答了」的画面，不能用"]

    gates = ["检索不足" not in (r["abstain_reason"] or "") for r in runs]
    n_model = sum(gates)
    pts, why = 0.0, []

    if n_model == n:
        pts += 4.0
        why.append(f"{n}/{n} 次都走模型闸门 ★ 稳定且有说服力")
    elif n_model == 0:
        pts += 1.5
        why.append(f"{n}/{n} 次都走检索闸门（稳定，但只能证明「找不到」）")
    else:
        pts += 0.5
        why.append(f"闸门不稳定（模型 {n_model} / 检索 {n - n_model}）—— 录屏有风险")

    avg_len = sum(len(r["abstain_reason"] or "") for r in runs) / n
    if avg_len >= 20:
        pts += 1.5
        why.append(f"拒答理由具体（平均 {avg_len:.0f} 字，说明真读了资料）")
    else:
        why.append(f"拒答理由偏短（平均 {avg_len:.0f} 字）")

    return pts, why


def run_ask(ask: AskPipeline, question: str, qid: str = "?") -> dict:
    """跑一条题，抽出评分需要的字段。"""
    t0 = time.perf_counter()
    res = ask.ask(question, mode="hybrid", topk=5, pool=50)
    cites = res.get("citations") or []
    return {
        "qid": qid,
        "question": question,
        "abstained": bool(res.get("abstained")),
        "abstain_reason": res.get("abstain_reason") or "",
        "answer": res.get("answer") or "",
        "citations": [{"n": _f(c, "n"), "source": _f(c, "source"),
                       "title": _f(c, "title"), "chunk_id": _f(c, "chunk_id")}
                      for c in cites],
        "cited_sources": sources_of(res),
        "used": res.get("used") or [],
        "unsupported": res.get("unsupported") or [],
        "top_score": res.get("top_score"),
        "latency_ms": res.get("latency_ms") or round((time.perf_counter() - t0) * 1000, 1),
        "cost_cny": res.get("cost_cny") or 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="为 demo 短片 ① 挑演示题")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条 multi 题（冒烟用）")
    ap.add_argument("--abstain-repeat", type=int, default=3,
                    help="每条拒答候选重复跑几次（看闸门稳定性，默认 3）")
    ap.add_argument("--golden", default=os.path.join(ROOT, "data/eval/golden.jsonl"))
    ap.add_argument("--out", default=None, help="把完整结果存成 JSON")
    a = ap.parse_args()

    golden = [json.loads(l) for l in open(a.golden, encoding="utf-8") if l.strip()]
    multi = [g for g in golden if g.get("type") == "multi"]
    if a.limit:
        multi = multi[: a.limit]

    ask = build_ask()
    print(f"挑题：{len(multi)} 条 multi 题 + {len(ABSTAIN_CANDIDATES)} 条拒答候选")
    print("配置：rerank 策略 · hybrid_rerank · topk=5 · pool=50（与 D4 评测口径一致）\n")

    rows: list[dict] = []
    print("=" * 78)
    print("第一部分：跨文档题候选")
    print("=" * 78)
    for i, g in enumerate(multi, 1):
        try:
            row = run_ask(ask, g["question"], g["qid"])
        except Exception as e:                                    # noqa: BLE001
            print(f"  [{i:>2}/{len(multi)}] {g['qid']} 跑失败：{type(e).__name__}: {e}")
            continue
        row["score"], row["why"] = score(row)
        row["expect_chunks"] = g.get("expect_chunks") or []
        row["expect_any"] = g.get("expect_any") or []
        rows.append(row)
        flag = "拒答" if row["abstained"] else f"{len(row['cited_sources'])}家"
        print(f"  [{i:>2}/{len(multi)}] {g['qid']:<5} {flag:<5} "
              f"引用={len(row['citations'])}条 得分={row['score']:.1f}  "
              f"{row['latency_ms']:.0f}ms  ¥{row['cost_cny']:.5f}")

    rows.sort(key=lambda r: -r["score"])
    ok = [r for r in rows if r["score"] > 0]

    print(f"\n跨两家且无瞎标的候选：{len(ok)} / {len(rows)} 条")
    print(f"\nTop {min(5, len(ok))}（按录屏观感排序）：")
    for r in ok[:5]:
        print(f"\n  【{r['qid']}】得分 {r['score']:.1f}")
        print(f"  问题：{r['question']}")
        print(f"  引用来源：{r['cited_sources']}")
        print(f"  引用明细：" + " | ".join(
            f"[{c['n']}] {c['source']} · {(c['title'] or '')[:22]}" for c in r["citations"]))
        print(f"  理由：{'；'.join(r['why'])}")
        print(f"  答案：{r['answer'][:150]}{'…' if len(r['answer']) > 150 else ''}")

    print("\n" + "=" * 78)
    print(f"第二部分：拒答题候选（每条跑 {a.abstain_repeat} 次 —— 看闸门稳不稳定）")
    print("=" * 78)
    abstain_rows: list[dict] = []
    for q in ABSTAIN_CANDIDATES:
        runs: list[dict] = []
        for _ in range(a.abstain_repeat):
            try:
                runs.append(run_ask(ask, q, "abstain"))
            except Exception as e:                                # noqa: BLE001
                print(f"  跑失败：{type(e).__name__}: {e}")
                break
        if not runs:
            continue

        row = {
            "question": q,
            "abstained": all(r["abstained"] for r in runs),
            "abstain_reason": runs[-1]["abstain_reason"],
            "top_score": runs[-1]["top_score"],
            "answer": runs[-1]["answer"],
            "latency_ms": runs[-1]["latency_ms"],
            "cost_cny": sum(r["cost_cny"] for r in runs),
            "runs": [{"abstained": r["abstained"],
                      "reason": (r["abstain_reason"] or "")[:60],
                      "top_score": r["top_score"]} for r in runs],
        }
        row["score"], row["why"] = score_abstain(runs)
        abstain_rows.append(row)

        gates = "".join("模" if "检索不足" not in (r["abstain_reason"] or "") else "检"
                        for r in runs if r["abstained"])
        flag = f"拒答 {len(gates)}/{len(runs)}" if len(gates) == len(runs) else f"答了 {len(runs) - len(gates)} 次"
        print(f"  {flag:<9} 闸门={gates or '—':<5} 得分={row['score']:.1f}　{q}")
        if row["abstained"]:
            print(f"      理由：{row['abstain_reason'][:58]}")
        else:
            print(f"      答成了：{row['answer'][:70]}")

    abstain_rows.sort(key=lambda r: -r["score"])
    good_abstain = [r for r in abstain_rows if r["score"] > 0]

    print("\n" + "=" * 78)
    print("拍摄单（短片 ① 用的两条题）")
    print("=" * 78)

    if ok:
        best = ok[0]
        print(f"\n【第一段 0–40s】跨文档问答 → 用 {best['qid']}")
        print(f"  问题：{best['question']}")
        print(f"  预期画面：{len(best['citations'])} 条引用，来源 {' + '.join(best['cited_sources'])}")
        for c in best["citations"]:
            print(f"      [{c['n']}] {c['source']:<9} {(c['title'] or '')[:34]}")
        print(f"  答案 {len(best['answer'])} 字 · {best['latency_ms']:.0f}ms · ¥{best['cost_cny']:.5f}")
        print(f"  命令：python scripts/ask_demo.py \"{best['question']}\"")
        print(f"\n  答案全文（旁白念大意即可，措辞每次会变）：\n  "
              + best["answer"].replace(chr(10), chr(10) + "  "))
    else:
        print("\n跨文档题 → 没有合格候选，需要检查 rerank 配置或换题")

    if good_abstain:
        pick = good_abstain[0]
        gate = "检索闸门" if "检索不足" in (pick["abstain_reason"] or "") else "模型闸门"
        print(f"\n【第二段 40–65s】拒答演示 → 用「{pick['question']}」")
        print(f"  走向：{gate}　({pick['why'][0]})")
        print(f"  top_score={pick['top_score']}")
        print(f"  拒答理由（旁白念大意，不要逐字念 —— 措辞每次会变）：")
        for ln in pick["abstain_reason"].split("。"):
            if ln.strip():
                print(f"      {ln.strip()}。")
        print(f"  命令：python scripts/ask_demo.py \"{pick['question']}\"")
        print("\n  备选（如果想演示「检索闸门」那一种）：")
        for r in good_abstain[1:3]:
            g = "".join("模" if "检索不足" not in (x["reason"] or "") else "检"
                        for x in r["runs"] if x["abstained"])
            print(f"      · {r['question']}　[闸门 {g}]")
    else:
        print("\n拒答题 → 候选全被答了，需要另造更「文档外」的问题")

    if a.out:
        path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "cross_doc": rows, "abstain": abstain_rows},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n完整结果已存：{path}")


if __name__ == "__main__":
    main()
