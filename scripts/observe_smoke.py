"""W4 块 E｜可观测冒烟：跑一次带 trace 的 Agent → 落 JSONL → 打印成本/延迟分项。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.config import Settings                       # noqa: E402
from app.observe import (                             # noqa: E402
    Tracer, make_trace_hook, langfuse_exporter, trace_path, read_trace, TRACE_DIR,
)
from app.react import run_react, tool_names           # noqa: E402
from app.tools import TOOL_SCHEMAS                    # noqa: E402

TASK = "上海今天天气怎么样？顺便看一下近 7 天的订单情况和订单 A1001 的详情。"

SCRIPT = [
    "思考：需要三样信息，先查天气。\n"
    "行动：get_current_weather\n"
    '参数：{"city": "上海"}',

    "思考：天气拿到了，再看近 7 天的订单统计。\n"
    "行动：query_orders_stats\n"
    '参数：{"sql": "SELECT status, COUNT(*) FROM orders GROUP BY status", "limit": 10}',

    "思考：最后看一下订单 A1001 的详情。\n"
    "行动：get_order_detail\n"
    '参数：{"order": "A1001"}',

    "思考：上一轮参数写坏了，订单详情拿不到，天气和统计已经够了，直接给结论。\n"
    "最终答案：上海今天晴，26℃。近 7 天订单 128 单，其中已完成 119 单。"
    "订单 A1001 的详情这次没拿到（参数格式错误），需要重新发起查询。",
]

FAKE_TOOLS = {"get_current_weather", "query_orders_stats", "get_order_detail"}


def make_fake_chat(script: list[str], llm_sleep: float = 0.15):
    """把 app.react.call_chat 换成脚本化实现。"""
    box = {"i": 0}

    def fake_call_chat(s, messages, **kw):
        i = min(box["i"], len(script) - 1)
        box["i"] += 1
        time.sleep(llm_sleep)
        return {
            "content": script[i],
            "usage": {"in": 800 + 60 * i, "out": 120 + 20 * i, "cached": 640},
            "cost_cny": 0.00042 + 0.00008 * i,
            "latency_s": round(llm_sleep, 3),
            "model": "fake-scripted-model",
            "provider": kw.get("provider", "fake"),
        }

    return fake_call_chat


def run_event_trace(real: bool, use_langfuse: bool) -> Tracer:
    s = Settings()
    exp = [langfuse_exporter()] if use_langfuse else []
    if use_langfuse and not exp:
        print("  ⚠ Langfuse 不可用（未安装或没配 key），本次只落本地 JSONL")

    tr = Tracer(name="agent.react", task=TASK, exporters=exp)
    hook = make_trace_hook(tr, task=TASK)

    if real:
        print("  · 真实模式：调真模型，会产生费用")
        out = run_react(s, TASK, on_event=hook, max_steps=5)
    else:
        import app.react as react_mod
        react_mod.call_chat = make_fake_chat(SCRIPT)
        schemas = [t for t in TOOL_SCHEMAS
                   if (t.get("function") or {}).get("name") in FAKE_TOOLS]
        print(f"  · 合成模式：假模型 + 本地工具 {sorted(tool_names(schemas))}")
        out = run_react(s, TASK, tools=schemas, on_event=hook,
                        max_steps=5, enforce_budget=False, record_cost=False)

    tr.patch_root(**{
        "cost_cny": out.get("cost_cny"),
        "latency_s": out.get("latency_s"),
        "usage": out.get("usage"),
        "stop_reason": out.get("stop_reason"),
        "ok": out.get("ok"),
    })
    return tr


def run_manual_trace() -> Tracer:
    """演示上下文管理器埋点，以及**异常时 span 会按 error 收尾且不吞异常**。"""
    tr = Tracer(name="demo.manual_spans")

    with tr.span("retrieval.search", kind="retrieval", query="上下文硬盘缓存") as sp:
        time.sleep(0.08)
        tr.end_span(sp, output={"hits": 3, "top": "cache.md#ctx"})

    with tr.span("llm.call", kind="llm", model="deepseek-flash") as sp:
        time.sleep(0.12)
        tr.end_span(sp, output="缓存命中率 82%", cost_cny=0.00031,
                    tokens={"in": 1200, "out": 96})

    try:
        with tr.span("tool.write", kind="tool", tool="create_ticket") as sp:
            time.sleep(0.03)
            raise RuntimeError("演示：下游 500")
    except RuntimeError as e:
        print(f"  · 异常已如实上抛（观测层没吞）：{e}")

    tr.finish_root(output={"demo": "manual spans"})
    return tr


def prune(keep: int) -> None:
    if keep <= 0 or not os.path.isdir(TRACE_DIR):
        return
    files = sorted(
        (os.path.join(TRACE_DIR, f) for f in os.listdir(TRACE_DIR) if f.endswith(".jsonl")),
        key=os.path.getmtime, reverse=True)
    for p in files[keep:]:
        os.remove(p)
        html_side = os.path.splitext(p)[0] + ".html"
        if os.path.exists(html_side):
            os.remove(html_side)
        print(f"  · 清理旧 trace：{os.path.basename(p)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="用真模型跑（会花钱）")
    ap.add_argument("--langfuse", action="store_true", help="同时镜像到 Langfuse")
    ap.add_argument("--view", action="store_true", help="跑完渲染 HTML 查看器")
    ap.add_argument("--keep", type=int, default=0, help="只保留最近 N 条 trace")
    args = ap.parse_args()

    print("=== W4 块 E · 可观测冒烟 ===\n")

    print("[1/3] 用法 A · 事件流自动翻 span（Agent 逻辑零改动）")
    tr1 = run_event_trace(args.real, args.langfuse)
    s1 = tr1.summarize()
    p1 = trace_path(tr1.trace_id)
    print(f"  trace_id : {tr1.trace_id}")
    print(f"  span 数  : {s1['span_count']}（其中出错 {s1['error_count']}）")
    print(f"  总耗时   : {s1['total_ms']} ms")
    for k, v in s1["breakdown"].items():
        print(f"    - {k:<9} {v['count']} 次  {v['ms']:>8.2f} ms  {v['pct']:>5.1f}%")
    print(f"  模型净耗时：实测 {s1['llm_ms_measured']} ms / 推导 {s1['llm_ms_derived']} ms")
    print(f"  成本     : ¥{s1['cost_cny']}   停止原因：{s1['stop_reason']}")
    for e in s1["errors"]:
        print(f"    ✗ {e}")
    print(f"  → {p1}\n")

    print("[2/3] 用法 B · 手动埋点（含异常路径）")
    tr2 = run_manual_trace()
    s2 = tr2.summarize()
    p2 = trace_path(tr2.trace_id)
    print(f"  trace_id : {tr2.trace_id}   span 数：{s2['span_count']}"
          f"   出错：{s2['error_count']}   总耗时：{s2['total_ms']} ms")
    print(f"  → {p2}\n")

    print("[3/3] 校验：JSONL 落盘可回读")
    recs = read_trace(p1)
    kinds = {}
    for r in recs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"  回读 {len(recs)} 行，kind 分布 {kinds}")
    if not recs:
        print("  ✗ 落盘为空"); return 1
    if any(r.get("dur_ms") is None for r in recs):
        print("  ✗ 存在未闭合 span"); return 1
    print("  ✓ 每行都有 dur_ms，无悬空 span\n")

    if args.view:
        out_html = os.path.join(ROOT, "data", "traces", f"{tr1.trace_id}.html")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from trace_view import render
        render(recs, out_html)
        print(f"  HTML 查看器：{out_html}\n")

    print("=== 冒烟通过 ===")
    print("下一步：python scripts/trace_view.py "
          f"{p1}    # 渲染成 HTML 看瀑布图")
    if args.keep:
        prune(args.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
