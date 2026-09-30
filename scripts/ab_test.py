# -*- coding: utf-8 -*-
"""A/B 对照测试：一次跑完所有组合，直接输出可粘贴进实验记录的表格。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.completion import structured_chat  # noqa: E402
from app.config import get_settings  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

QUESTIONS = [
    "小猫怎么叫？",
    "RAG 能解决什么问题？",
    "圣诞老人会在圣诞节给我带礼物吗？",
]

GROUPS = {
    "a": {
        "name": "A 基线（qa 模式 + XML）",
        "kwargs": {"use_xml": True, "mode": "qa"},
    },
    "b": {
        "name": "B 关掉 XML",
        "kwargs": {"use_xml": False},
    },
    "c": {
        "name": "C 加角色（资深工程师）",
        "kwargs": {"use_xml": True, "role": "资深大模型应用工程师"},
    },
    "d": {
        "name": "D 加 few-shot（迎合型示例）",
        "kwargs": {
            "use_xml": True,
            "examples": [(
                "牙仙真的存在吗？",
                '{"answer":"当然存在，亲爱的。把牙齿放枕头下，明天会有惊喜","confidence":0.9,"tags":["牙仙"]}',
            )],
        },
    },
    "e": {
        "name": "E RAG 模式（验证 bug）",
        "kwargs": {"use_xml": True, "mode": "rag"},
    },
}


def run_one(settings, q: str, kwargs: dict) -> dict:
    t0 = time.time()
    try:
        out = structured_chat(settings, message=q, **kwargs)
        return {
            "ok": out["ok"],
            "answer": (out["data"].answer if out["data"] else out["raw"])[:70],
            "conf": out["data"].confidence if out["data"] else 0,
            "tags": ",".join(out["data"].tags) if out["data"] else "",
            "tok": out["usage"]["completion_tokens"],
            "lat": out["latency_s"],
            "cost": out["cost_cny"],
            "retry": out["retries_used"],
            "err": out["validation_error"][:40],
        }
    except Exception as e:
        return {"ok": False, "answer": f"ERROR: {type(e).__name__} {e}"[:70], "conf": 0,
                "tags": "", "tok": 0, "lat": round(time.time() - t0, 2), "cost": 0.0,
                "retry": 0, "err": str(e)[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    keys = [k.strip() for k in args.groups.split(",") if k.strip()]
    total = len(keys) * len(QUESTIONS)
    print(f"将跑 {len(keys)} 组 × {len(QUESTIONS)} 题 = {total} 次调用（约 ¥{total * 0.001:.3f}）\n")

    if args.dry:
        for k in keys:
            print(f"  [{k}] {GROUPS[k]['name']}  {GROUPS[k]['kwargs']}")
        return

    settings = get_settings()
    started = time.time()
    rows = []

    for qi, q in enumerate(QUESTIONS, 1):
        print(f"── 问题 {qi}/{len(QUESTIONS)}：{q}")
        for k in keys:
            g = GROUPS[k]
            r = run_one(settings, q, g["kwargs"])
            r.update({"group": k, "gname": g["name"], "q": q})
            rows.append(r)
            flag = "OK " if r["ok"] else "!! "
            print(f"   {flag}[{k}] conf={r['conf']:<4} tok={r['tok']:<4} "
                  f"{r['lat']:.2f}s  {r['answer'][:44]}")
        print()

    print("=" * 78)
    print("汇总（可直接粘进 docs/实验记录.md）\n")
    for q in QUESTIONS:
        print(f"**问题：{q}**\n")
        print("| 组 | 配置 | answer | conf | tags | tok | 延迟 | 成本 | 重试 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["q"] != q:
                continue
            ans = r["answer"].replace("|", "/").replace("\n", " ")
            print(f"| {r['group'].upper()} | {r['gname']} | {ans} | {r['conf']} | "
                  f"{r['tags']} | {r['tok']} | {r['lat']}s | ¥{r['cost']:.5f} | {r['retry']} |")
        print()

    total_cost = sum(r["cost"] for r in rows)
    print(f"本次共 {len(rows)} 次调用，花费 ¥{total_cost:.5f}，耗时 {time.time() - started:.1f}s")
    print("\n看什么：")
    print("  · A vs B  → XML 标签对 DeepSeek 到底有没有用（第 4 章结论）")
    print("  · A vs C  → 角色提示改变了什么（第 3 章）")
    print("  · A vs D  → few-shot 是有害还是有益，看 conf 掉没掉（第 7 章）")
    print("  · E 组    → 如果回答'文档中没有相关信息'就是正常的；")
    print("              如果回答'根据文档无法回答'说明 bug 还在")


if __name__ == "__main__":
    main()
