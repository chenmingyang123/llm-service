"""Prompt 缓存实测：命中了吗？多久失效？—— W1 第 5 天上午第三件事。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings  # noqa: E402
from app.llm import LLMError, call_chat  # noqa: E402
from app.pricing import MIN_CACHE_PREFIX, breakeven  # noqa: E402

BASE = ("你是一个严谨的技术文档助手。以下是本项目的背景资料，请熟记并在后续回答中引用。"
        "资料内容涉及检索增强生成、向量检索、重排序、评测指标、成本控制、延迟优化、"
        "降级策略、并发限流、结构化输出、工具调用、流式响应、可观测性等多个方面。")

FILLER = ("RAG 系统的评测要同时看召回质量与生成忠实度，前者靠 golden set，后者靠 NLI 判据；"
          "成本控制的核心杠杆是输出长度而非输入长度，因为输出单价通常是输入的四倍；"
          "延迟的长尾往往来自上游排队而不是本地服务，这一点必须用 p95 而不是 p50 来判断。")


def make_prefix(chars: int) -> str:
    out = BASE
    while len(out) < chars:
        out += FILLER
    return out[:chars]


def one_call(s, prefix: str, question: str):
    """只有 messages 的内容参与前缀匹配 —— 没有专门的缓存开关要设。"""
    return call_chat(
        s,
        [{"role": "user", "content": f"{prefix}\n\n问题：{question}"}],
        provider="deepseek", model="deepseek-flash", max_tokens=16,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-chars", type=int, default=6000, help="固定前缀的字符数")
    ap.add_argument("--gaps", type=int, nargs="+", default=[0, 60, 300], help="各次之间的间隔秒数")
    ap.add_argument("--jitter", choices=["none", "once", "all"], default="none",
                    help="模拟「前缀里混进了变化的内容」：once=只在第 1 次；all=每次都变")
    ap.add_argument("--jitter-pos", choices=["head", "tail"], default="head",
                    help="变化内容放在前缀的哪里。head=开头（最常见的踩坑写法："
                         "系统提示最前面塞当前时间）；tail=末尾")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    s = get_settings()
    prefix = make_prefix(a.prefix_chars)
    print(f"前缀 {len(prefix)} 字符；间隔序列 {a.gaps} 秒；jitter={a.jitter}")
    print(f"提示：DeepSeek 自动缓存最小前缀 {MIN_CACHE_PREFIX} token，低于此值不缓存\n")

    rows = []
    prev = None
    for i, gap in enumerate(a.gaps):
        if gap and prev is not None:
            wait = max(gap - (time.time() - prev), 0)
            if wait > 0:
                print(f"等待 {wait:.0f} 秒…", flush=True)
                time.sleep(wait)
        q = ["第一项是什么？", "第二项是什么？", "第三项是什么？", "第四项是什么？"][i % 4]

        used_prefix = prefix
        if a.jitter == "all" or (a.jitter == "once" and i == 1):
            stamp = f"[当前时间 {datetime.now().isoformat(timespec='seconds')}]"
            used_prefix = f"{stamp}\n{prefix}" if a.jitter_pos == "head" else f"{prefix}\n{stamp}"

        try:
            r = one_call(s, used_prefix, q)
        except LLMError as e:
            print(f"[{i}] 失败：{e}")
            rows.append({"i": i, "gap_s": gap, "ok": False, "error": str(e)[:160]})
            continue
        prev = time.time()
        tin, cached = r["usage"]["in"], r["usage"]["cached"]
        rows.append({"i": i, "gap_s": gap, "ok": True, "prompt_tokens": tin,
                     "cached_tokens": cached,
                     "hit": cached > 0, "jittered": used_prefix != prefix,
                     "cost_cny": r.get("cost_cny", 0.0)})
        tag = "（本次前缀含随机内容）" if used_prefix != prefix else ""
        print(f"[{i}] 间隔 {gap:>4}s → prompt={tin:5d}  cached={cached:5d}  "
              f"{'命中' if cached else '未命中'}  本次 ¥{r.get('cost_cny', 0):.6f}{tag}")

    hits = [r for r in rows if r.get("hit")]
    md = ["# Prompt 缓存实测", "",
          f"- 固定前缀：{len(prefix)} 字符（约 {len(prefix)//2} token 量级，以各家返回为准）",
          f"- 动态问题放在前缀之后 —— 这是缓存能命中的前提", "",
          "| 次序 | 与前次间隔 | prompt tokens | cached tokens | 命中 | 本次成本 |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        if not r["ok"]:
            md.append(f"| {r['i']} | {r['gap_s']}s | — | — | 失败 | {r['error'][:40]} |")
            continue
        md.append(f"| {r['i']} | {r['gap_s']}s | {r['prompt_tokens']} | {r['cached_tokens']} | "
                  f"{'是' if r['hit'] else '否'} | ¥{r['cost_cny']:.6f} |")

    md += ["", "## 结论", ""]
    if hits:
        first = hits[0]
        md.append(f"- **缓存生效**：第 {first['i']} 次（间隔 {first['gap_s']}s）起命中，"
                  f"cached={first['cached_tokens']}。")
        misses = [r for r in rows if r["ok"] and not r["hit"] and r["i"] > first["i"]]
        if misses:
            md.append(f"- **存在失效**：第 {[m['i'] for m in misses]} 次又未命中，"
                      f"说明缓存有 TTL —— 把这个间隔记下来，它就是你的业务能否吃到缓存的分界线。")
        else:
            md.append(f"- 本次最长间隔 {max(a.gaps)}s 内仍然命中，TTL 至少大于这个值。")
    else:
        md.append("- **一次都没命中**。先查前缀是否 >= 64 token、以及是否每次都完全一字不差。")

    md += ["", "## 对应的盈亏平衡", ""]
    be = breakeven(prefix_tokens=max((r.get("cached_tokens") or 0) for r in rows if r["ok"]) or 2000,
                   dynamic_tokens=50, calls=100)
    md.append(f"按本次实测的前缀规模，复用 100 次可省 ¥{be['saved_cny']:.6f}"
              f"（省 {be['saved_pct']}%）。")
    md.append("")
    md.append("**判断标准**：省下来的钱如果抵不过「把 prompt 重排成固定前缀在前」"
              "这件事的工程代价，就别做 —— 缓存不是免费的，它收的是工程复杂度。")

    out = "\n".join(md)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"\n已写入 {a.out}")
    else:
        print()
        print(out)

    if "--json" in sys.argv:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
