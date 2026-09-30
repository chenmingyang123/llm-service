"""W2 第 6 天 · 生成端评测：拒答准不准、引用靠不靠谱、答案忠不忠实。"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.rag import AskPipeline  # noqa: E402
from app.retrieval import BM25Index, RetrievalPipeline, get_store  # noqa: E402

DEFAULT_ABSTAIN = os.path.join(ROOT, "data", "eval", "abstain.jsonl")
DEFAULT_GOLDEN = os.path.join(ROOT, "data", "eval", "golden.jsonl")

_ENT = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,}|\d+(?:\.\d+)?%?")


def faithfulness_floor(answer: str, cited_texts: list[str]) -> float:
    """答案里的实体有多少能在被引用的原文里找到。"""
    ents = set(_ENT.findall(answer))
    if not ents:
        return 1.0
    blob = " ".join(cited_texts)
    hit = sum(1 for e in ents if e in blob)
    return hit / len(ents)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstain", default=DEFAULT_ABSTAIN)
    ap.add_argument("--golden", default=DEFAULT_GOLDEN, help="顺便跑一遍引用校验")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--skip-golden", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cases = [json.loads(l) for l in open(a.abstain, encoding="utf-8") if l.strip()]
    store = get_store()
    ask = AskPipeline(RetrievalPipeline(
        store=store, bm25=BM25Index([r["text"] for r in store.rows]), backend="auto"))

    print(f"拒答评测 {len(cases)} 条（域外 {sum(1 for c in cases if c['kind']=='域外')} / "
          f"域内无答案 {sum(1 for c in cases if c['kind']=='域内无答案')} / "
          f"域内有答案 {sum(1 for c in cases if c['kind']=='域内有答案')}）\n")

    rows = []
    for c in cases:
        r = ask.ask(c["question"], topk=a.topk)
        got = "abstain" if r["abstained"] else "answer"
        ok = got == c["expect"]
        rows.append({**c, "got": got, "ok": ok, "reason": r["abstain_reason"],
                     "latency_ms": r["latency_ms"], "cost_cny": r["cost_cny"],
                     "unsupported": r["unsupported"]})
        flag = "OK " if ok else "×  "
        print(f"  {flag}{c['qid']} [{c['kind']:<6}] 期望 {c['expect']:<8} 实际 {got:<8} "
              f"{r['latency_ms']:.0f}ms ¥{r['cost_cny']:.5f}")
        if not ok:
            print(f"       理由：{r['abstain_reason'][:70]}")
            if got == "answer":
                print(f"       回答：{r['answer'][:70]}")

    def rate(kind=None, expect=None):
        sub = [x for x in rows
               if (kind is None or x["kind"] == kind)
               and (expect is None or x["expect"] == expect)]
        return (sum(1 for x in sub if x["ok"]) / len(sub), len(sub)) if sub else (0.0, 0)

    print("\n" + "=" * 68)
    print("拒答准确率")
    print("=" * 68)
    for kind in ("域外", "域内无答案", "域内有答案"):
        acc, n = rate(kind=kind)
        print(f"  {kind:<10} {acc * 100:5.1f}%  （{n} 条）")
    acc_all, n_all = rate()
    print(f"  {'合计':<10} {acc_all * 100:5.1f}%  （{n_all} 条）")

    cheap = [x for x in rows if x["got"] == "abstain" and x["cost_cny"] == 0.0]
    print(f"\n  其中 {len(cheap)} 条被闸门①拦下，没花生成钱（省了 {len(cheap)} 次 LLM 调用）")

    if not a.skip_golden:
        gold = [json.loads(l) for l in open(a.golden, encoding="utf-8") if l.strip()]
        print("\n" + "=" * 68)
        print(f"引用校验（golden set {len(gold)} 条）")
        print("=" * 68)
        bad_cite = 0
        fsum = 0.0
        fcount = 0
        for g in gold:
            r = ask.ask(g["question"], topk=a.topk)
            if r["unsupported"]:
                bad_cite += 1
                print(f"  瞎标 {g['qid']}: 引用了不存在的编号 {r['unsupported']}")
            if not r["abstained"] and r["citations"]:
                texts = [c.quote for c in r["citations"]]
                full = [h for h in r.get("context", [])[:len(r["citations"])]]
                fsum += faithfulness_floor(r["answer"], full or texts)
                fcount += 1
        print(f"  瞎标率        {bad_cite}/{len(gold)} = {bad_cite / len(gold) * 100:.1f}%")
        if fcount:
            print(f"  忠实度下界    {fsum / fcount:.3f}  （{fcount} 条有引用的答案）")
        print("  注意：这个忠实度是实体重叠算出来的下界，不是严格忠实度。")

    if a.out:
        path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"\n已写入 {path}")


if __name__ == "__main__":
    main()
