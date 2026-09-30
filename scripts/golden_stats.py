"""golden set 分布体检：偏没偏、差多少、补什么。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_GOLDEN = os.path.join(ROOT, "data", "eval", "golden.jsonl")

TARGET_TYPE = {"term": 50, "semantic": 30, "multi": 15, "boundary": 15, "conflict": 10}
HARD = {"multi", "boundary", "conflict"}
TARGET_TOTAL = sum(TARGET_TYPE.values())
SOURCES = ("deepseek", "zhipu", "bailian")


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def alarm_hard_ratio(cnt: Counter, total: int) -> str | None:
    """难例占比：低于 30% 就没有分辨力。"""
    hard = sum(cnt.get(t, 0) for t in HARD)
    r = hard / total if total else 0.0
    if r < 0.30:
        return f"难例只占 {r*100:.0f}%（{hard}/{total}），低于 30% → 天花板效应，测了也白测"
    return None


def alarm_type_dominant(cnt: Counter, total: int) -> list[str]:
    """单类型占比 > 50%：这一类型已经测不出差异。"""
    out = []
    for t, n in cnt.most_common():
        if total and n / total > 0.50:
            out.append(f"{t} 占 {n/total*100:.0f}%（{n}/{total}）> 50% → 该类型已饱和，再加无收益")
    return out


def alarm_source_skew(src: Counter, total: int) -> list[str]:
    """平台覆盖不均：过低是盲区，过高是偏斜。"""
    out = []
    for s in SOURCES:
        n = src.get(s, 0)
        r = n / total if total else 0.0
        if r < 0.25:
            out.append(f"{s} 只占 {r*100:.0f}%（{n}/{total}）< 25% → 该平台覆盖不足，评测分数不代表真实水平")
        elif r > 0.45:
            out.append(f"{s} 占 {r*100:.0f}%（{n}/{total}）> 45% → 偏斜，总分会被这一家主导")
    return out


def alarm_missing_cells(golden: list[dict]) -> list[str]:
    """平台 × 类型 的空格：某些组合一条都没有。"""
    have = {(g.get("expect_source", "?"), g.get("type", "?")) for g in golden}
    out = []
    for s in SOURCES:
        miss = [t for t in TARGET_TYPE if (s, t) not in have]
        if miss:
            out.append(f"{s} 缺 {len(miss)} 种题型：{', '.join(miss)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--target", type=int, default=TARGET_TOTAL, help="目标总条数（默认 120）")
    a = ap.parse_args()

    g = load(a.golden)
    total = len(g)
    cnt = Counter(x.get("type", "?") for x in g)
    src = Counter(x.get("expect_source", "?") for x in g)

    print(f"golden set {total} 条　目标 {a.target} 条\n")

    print("【类型分布】")
    for t in TARGET_TYPE:
        n = cnt.get(t, 0)
        r = n / total * 100 if total else 0
        bar = "█" * int(r / 2)
        print(f"  {t:<9} {n:>3} 条  {r:>5.1f}%  {bar}")

    print("\n【来源分布】")
    for s in SOURCES:
        n = src.get(s, 0)
        r = n / total * 100 if total else 0
        bar = "█" * int(r / 2)
        print(f"  {s:<9} {n:>3} 条  {r:>5.1f}%  {bar}")

    print("\n【平台 × 类型】")
    for s in SOURCES:
        sub = [x for x in g if x.get("expect_source") == s]
        inner = Counter(x.get("type", "?") for x in sub)
        print(f"  {s:<9} {len(sub):>3} 条   " +
              "  ".join(f"{t}:{inner.get(t,0)}" for t in TARGET_TYPE))

    print("\n【偏斜告警】")
    alarms: list[str] = []
    if a2 := alarm_hard_ratio(cnt, total):
        alarms.append(a2)
    alarms += alarm_type_dominant(cnt, total)
    alarms += alarm_source_skew(src, total)
    alarms += alarm_missing_cells(g)

    if alarms:
        for x in alarms:
            print(f"  ⚠ {x}")
    else:
        print("  无告警，分布达标")

    print(f"\n【到 {a.target} 条的缺口】")
    scale = a.target / TARGET_TOTAL
    need_total = 0
    for t in TARGET_TYPE:
        target = round(TARGET_TYPE[t] * scale)
        n = cnt.get(t, 0)
        need = max(target - n, 0)
        need_total += need
        if need:
            print(f"  {t:<9} 现有 {n:>3} → 目标 {target:>3}　还差 {need:>3} 条")
        else:
            print(f"  {t:<9} 现有 {n:>3} → 目标 {target:>3}　已达标")
    print(f"\n  合计还差 {need_total} 条")

    per = round(a.target / len(SOURCES))
    print(f"\n【来源缺口（目标每家 {per} 条）】")
    for s in SOURCES:
        n = src.get(s, 0)
        need = max(per - n, 0)
        print(f"  {s:<9} 现有 {n:>3} → 目标 {per:>3}　还差 {need:>3} 条")

    print("\n提示：难例（multi/boundary/conflict）通常需要手工写，")
    print("      make_golden_draft.py 只擅长出 term/semantic 基础题。")


if __name__ == "__main__":
    main()
