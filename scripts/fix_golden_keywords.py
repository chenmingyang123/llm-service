"""修复 golden set 里"关键词不在标准答案块里"的问题。"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.retrieval import get_store  # noqa: E402

DEFAULT_GOLDEN = os.path.join(ROOT, "data", "eval", "golden.jsonl")

DASH = "-~～—–"
COMMA = ",，"


def fuzzy_find(kw: str, text: str) -> str | None:
    """在 text 里模糊找 kw，命中则返回**块里的真实写法**，否则 None。"""
    if not kw:
        return None
    if kw in text:
        return kw
    parts = []
    for c in kw:
        if c in DASH:
            parts.append("[" + re.escape(DASH) + "]")
        elif c in COMMA:
            parts.append("[" + re.escape(COMMA) + "]?")
        else:
            parts.append(re.escape(c))
    pattern = r"\s*".join(parts)
    m = re.search(pattern, text)
    return m.group(0) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--backup", default=None)
    ap.add_argument("--apply", action="store_true", help="真的写回（默认只报告）")
    a = ap.parse_args()

    store = get_store()
    rows = [json.loads(l) for l in open(a.golden, encoding="utf-8") if l.strip()]

    fixed_kw, dropped_kw, dropped_q = 0, 0, []
    out = []
    for x in rows:
        text = store.by_id[x["expect_chunk"]].get("text", "")
        new_kw = []
        for kw in x.get("expect_any", []):
            if kw in text:
                new_kw.append(kw)
                continue
            real = fuzzy_find(kw, text)
            if real:
                new_kw.append(real)
                fixed_kw += 1
                print(f"  [修] {x['qid']}  {kw!r} → {real!r}")
            else:
                dropped_kw += 1
                print(f"  [删] {x['qid']}  关键词 {kw!r} 块里没有，剔除")
        if not new_kw:
            dropped_q.append(x["qid"])
            print(f"  [剔题] {x['qid']} 所有关键词都无效，整题剔除")
            continue
        x["expect_any"] = new_kw
        out.append(x)

    print(f"\n关键词修复 {fixed_kw} 个　剔除 {dropped_kw} 个　整题剔除 {len(dropped_q)} 条")
    print(f"题数 {len(rows)} → {len(out)}")

    if a.apply:
        bak = a.backup or a.golden.replace(".jsonl", "_bak_keywords.jsonl")
        if not os.path.exists(bak):
            shutil.copy(a.golden, bak)
            print(f"已备份 → {os.path.basename(bak)}")
        with open(a.golden, "w", encoding="utf-8") as f:
            for x in out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"已写回 {a.golden}")
    else:
        print("\n（只报告，未写回。加 --apply 生效）")


if __name__ == "__main__":
    main()
