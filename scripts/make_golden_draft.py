"""golden set 半自动扩展：从语料出题，人工确认后合并。"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.config import get_settings  # noqa: E402
from app.llm import LLMError, call_chat  # noqa: E402
from app.retrieval import get_store  # noqa: E402

DEFAULT_GOLDEN = os.path.join(ROOT, "data", "eval", "golden.jsonl")
DEFAULT_OUT = os.path.join(ROOT, "data", "eval", "golden_draft.jsonl")
SOURCES = ("deepseek", "zhipu", "bailian")

BASE = """根据下面这段文档片段，出一道"用户查资料时可能会问"的问题。要求：
1. 必须是一个**完整的疑问句**，以问号结尾
2. 答案必须就在这段片段里（不能问片段里没有的内容）
3. 优先问数字、限制条件、机制、开关这类"查表型"问题，答案唯一、可判定对错
4. **不要照抄文档片段**，不要出现代码片段、CSS 类名、HTML 标签、URL 路径
5. 不要问"这段文档说了什么"这种没有信息量的问题

文档片段：
{text}

只输出一行 JSON，不要任何解释：
{{"question": "...", "expect_any": ["关键词1", "关键词2"]}}"""

MULTI = """下面有**两段来自不同平台**的文档片段。请出一道必须**同时用到两段**才能答全的问题
（例如对比两家的做法、或者各问一半）。要求：
1. 必须是一个完整的疑问句，以问号结尾
2. 只拿到其中一段会答不全 —— 这是关键
3. 不要照抄片段，不要出现代码片段或 URL

片段 A（{src_a}）：
{text_a}

片段 B（{src_b}）：
{text_b}

只输出一行 JSON：
{{"question": "...", "expect_any": ["关键词1", "关键词2"]}}"""

BOUNDARY = """根据下面这段文档片段，出一道问**适用边界 / 什么时候不成立**的问题。
例如"什么情况下不会命中""哪一类场景不适用""超出什么限制就会失败"。要求：
1. 必须是一个完整的疑问句，以问号结尾
2. 答案必须能从片段里推出（或明确说出限制条件）
3. 不要照抄片段，不要出现代码片段

文档片段：
{text}

只输出一行 JSON：
{{"question": "...", "expect_any": ["关键词1", "关键词2"]}}"""

CONFLICT = """下面有**两段来自不同平台**的文档片段，讲的是同一类事情但说法/规则不同。
请出一道问"两家有什么不同"的问题（认错一家就答错）。要求：
1. 必须是一个完整的疑问句，以问号结尾
2. 必须明确涉及两家的差异
3. 不要照抄片段

片段 A（{src_a}）：
{text_a}

片段 B（{src_b}）：
{text_b}

只输出一行 JSON：
{{"question": "...", "expect_any": ["关键词1", "关键词2"]}}"""

PROMPTS = {"term": BASE, "semantic": BASE, "boundary": BOUNDARY,
           "multi": MULTI, "conflict": CONFLICT}


CODE_RE = re.compile(r"(className|<div|</?[a-z]+>|function\s*\(|=>|margin-|padding-|#[0-9a-fA-F]{6})")
ASK_RE = re.compile(r"(什么|如何|怎么|多少|哪|是否|会不会|有哪些|为什么|能否|几)")
ASK_END = ("？", "?")


def quality_ok(q: str, kw: list, max_len: int = 60) -> tuple[bool, str]:
    """自动过滤一道题。返回 (是否合格, 不合格原因)。"""
    q = (q or "").strip()
    if not q:
        return False, "空问题"
    if len(q) < 10:
        return False, f"过短（{len(q)} 字）"
    if len(q) > max_len:
        return False, f"过长（{len(q)} 字 > {max_len}）"
    if not q.endswith(ASK_END):
        return False, "不是疑问句"
    if not ASK_RE.search(q):
        return False, "没有疑问词"
    if CODE_RE.search(q):
        return False, "含代码特征"
    if len(kw or []) < 2:
        return False, f"关键词不足（{len(kw or [])} 个）"
    return True, ""


def extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def pick_basic(cand: list[dict], n: int, seed: int) -> list[dict]:
    """基础题：按 source 分层轮询抽样。"""
    by: dict[str, list] = {}
    for r in cand:
        by.setdefault(r.get("source", "?"), []).append(r)
    for s in by:
        random.Random(seed).shuffle(by[s])
    out = []
    while len(out) < n and any(by.values()):
        for s in list(by):
            if len(out) >= n:
                break
            if by[s]:
                out.append(by[s].pop())
    return out


def pick_pairs(cand: list[dict], n: int, seed: int) -> list[tuple[dict, dict]]:
    """难例题：抽**两个不同平台**的块组成一对（multi / conflict 用）。"""
    by: dict[str, list] = {}
    for r in cand:
        by.setdefault(r.get("source", "?"), []).append(r)
    for s in by:
        random.Random(seed).shuffle(by[s])
    rng = random.Random(seed + 1)
    pairs, used = [], set()
    srcs = [s for s in SOURCES if by.get(s)]
    guard = 0
    while len(pairs) < n and guard < n * 40:
        guard += 1
        if len(srcs) < 2:
            break
        a_src, b_src = rng.sample(srcs, 2)
        if not by.get(a_src) or not by.get(b_src):
            continue
        a, b = by[a_src][-1], by[b_src][-1]
        if a["chunk_id"] in used or b["chunk_id"] in used:
            rng.shuffle(by[a_src]); rng.shuffle(by[b_src]); continue
        used.add(a["chunk_id"]); used.add(b["chunk_id"])
        pairs.append((a, b))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--n", type=int, default=40, help="想要多少道**合格**的题")
    ap.add_argument("--kind", default="term", choices=list(PROMPTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--max-tries", type=int, default=0,
                    help="最多尝试多少块（0=自动按 n 的 3 倍）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    covered = set()
    if os.path.exists(a.golden):
        for l in open(a.golden, encoding="utf-8"):
            if l.strip():
                g = json.loads(l)
                covered.add(g["expect_chunk"])
                covered.update(g.get("expect_chunks") or [])

    store = get_store()
    cand = [r for r in store.rows
            if r.get("role") != "child" and r["chunk_id"] not in covered
            and 80 <= len(r.get("text", "")) <= 1200]

    kind = a.kind
    is_pair = kind in ("multi", "conflict")
    tries = a.max_tries or (a.n * 6 if is_pair else a.n * 3)

    if is_pair:
        items = pick_pairs(cand, min(tries, a.n * 6), a.seed)
    else:
        items = [(r, None) for r in pick_basic(cand, tries, a.seed)]

    print(f"候选块 {len(cand)}（排除子块与已覆盖 {len(covered)}）")
    print(f"题型 {kind}　目标 {a.n} 条合格　最多试 {len(items)} 组\n")

    settings = get_settings()
    rows, rejected = [], 0
    for i, (ra, rb) in enumerate(items, 1):
        if len(rows) >= a.n:
            break
        if rb is None:
            prompt = PROMPTS[kind].format(text=ra["text"])
            ids, srcs = [ra["chunk_id"]], [ra.get("source", "")]
        else:
            prompt = PROMPTS[kind].format(
                text_a=ra["text"], text_b=rb["text"],
                src_a=ra.get("source", ""), src_b=rb.get("source", ""))
            ids, srcs = [ra["chunk_id"], rb["chunk_id"]], [ra.get("source", ""), rb.get("source", "")]

        try:
            res = call_chat(settings, [{"role": "user", "content": prompt}],
                            provider=a.provider, thinking=False, max_tokens=256)
            obj = extract_json(res.get("content", ""))
        except LLMError:
            continue
        if not obj or not obj.get("question"):
            rejected += 1
            continue

        q, kw = obj["question"], obj.get("expect_any", [])
        ok, why = quality_ok(q, kw, max_len=110 if is_pair else 60)
        if not ok:
            rejected += 1
            print(f"  [剔] {why}：{q[:40]}")
            continue

        row = {"qid": f"auto_{kind}_{len(rows)+1:03d}", "question": q,
               "expect_chunk": ids[0], "expect_source": srcs[0],
               "type": kind, "expect_any": kw}
        if len(ids) > 1:
            row["expect_chunks"] = ids
        rows.append(row)
        print(f"  [{len(rows):>3}] {srcs[0]:<9}{' + '+srcs[1] if len(srcs)>1 else ''} {q[:46]}")

    if not rows:
        print("\n没有产出合格草稿。")
        return

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已写 {len(rows)} 条合格草稿（剔除 {rejected} 条不合格）→ {a.out}")
    print("下一步（人工）：核对每道题答案确实在其 expect_chunk 里，确认后再合并进 golden.jsonl。")


if __name__ == "__main__":
    main()
