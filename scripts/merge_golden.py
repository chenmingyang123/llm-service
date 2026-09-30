"""把出题草稿合并进 golden set：去重 → 按配额裁剪 → 校验 → 写入（先备份）。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.retrieval import get_store  # noqa: E402

DEFAULT_GOLDEN = os.path.join(ROOT, "data", "eval", "golden.jsonl")
TARGET = {"term": 50, "semantic": 30, "multi": 15, "boundary": 15, "conflict": 10}


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm(q: str) -> str:
    """题目归一化（用于去重）：去空白与常见标点差异。"""
    return "".join(ch for ch in q if not ch.isspace()).strip("?？.。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--drafts", nargs="+", required=True, help="草稿文件（可多个）")
    ap.add_argument("--backup", default=None, help="备份路径（默认 golden_v2_<n>.jsonl）")
    ap.add_argument("--out", default=None, help="输出路径（默认覆盖 --golden）")
    a = ap.parse_args()

    old = load(a.golden)
    print(f"现有 golden {len(old)} 条")

    bak = a.backup or os.path.join(os.path.dirname(a.golden),
                                   f"golden_v2_{len(old)}.jsonl")
    if not os.path.exists(bak):
        shutil.copy(a.golden, bak)
        print(f"已备份 → {os.path.basename(bak)}")

    drafts: list[dict] = []
    for p in a.drafts:
        rows = load(p)
        print(f"  草稿 {os.path.basename(p)}：{len(rows)} 条")
        drafts += rows

    seen = {norm(g["question"]) for g in old}
    by_type: dict[str, list[dict]] = {}
    dup = 0
    for d in drafts:
        k = norm(d["question"])
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        by_type.setdefault(d.get("type", "term"), []).append(d)
    print(f"\n去重：剔除 {dup} 条重复题（与现有或草稿内部撞车）")

    have = {}
    for g in old:
        have[g.get("type", "term")] = have.get(g.get("type", "term"), 0) + 1

    picked: list[dict] = []
    print("\n配额（现有 → 目标 / 补入）：")
    for t in TARGET:
        need = max(TARGET[t] - have.get(t, 0), 0)
        avail = by_type.get(t, [])
        take = avail[:need]
        picked += take
        print(f"  {t:<9} {have.get(t,0):>3} → {TARGET[t]:>3}　补 {len(take):>3}"
              f"（草稿可用 {len(avail)}）")

    store = get_store()
    used_qid = {g["qid"] for g in old}
    bad = []
    merged = list(old)
    for i, d in enumerate(picked, 1):
        qid = f"m{i:03d}"
        while qid in used_qid:
            i += 1
            qid = f"m{i:03d}"
        used_qid.add(qid)

        cid = d["expect_chunk"]
        if cid not in store.by_id:
            bad.append((qid, f"expect_chunk 不存在：{cid}"))
            continue
        for c in d.get("expect_chunks") or []:
            if c not in store.by_id:
                bad.append((qid, f"expect_chunks 里的块不存在：{c}"))
        if not d.get("expect_any"):
            bad.append((qid, "expect_any 为空，没法校验命中"))

        merged.append({"qid": qid, "question": d["question"],
                       "expect_chunk": cid,
                       "expect_source": d.get("expect_source", ""),
                       "type": d.get("type", "term"),
                       "expect_any": d.get("expect_any", []),
                       **({"expect_chunks": d["expect_chunks"]}
                          if d.get("expect_chunks") else {})})

    if bad:
        print(f"\n⚠ {len(bad)} 条校验未通过，已跳过：")
        for qid, why in bad[:10]:
            print(f"    {qid}: {why}")

    out = a.out or a.golden
    with open(out, "w", encoding="utf-8") as f:
        for g in merged:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(f"\ngolden set {len(old)} → {len(merged)} 条（新增 {len(merged)-len(old)}）")
    print(f"已写入 {out}")
    print("\n下一步：跑 python scripts/golden_stats.py 看分布是否达标。")


if __name__ == "__main__":
    main()
