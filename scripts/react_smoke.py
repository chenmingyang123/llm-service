"""ReAct 真实冒烟 —— 证明「提示词」这一半也是对的。"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings          # noqa: E402
from app.cost import summarize           # noqa: E402
from app.react import run_react          # noqa: E402

DEFAULT_TASK = (
    "帮我查一下杭州现在的天气，再统计订单表里杭州的订单数和总金额，"
    "最后告诉我杭州今天适不适合做户外推广活动。"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="ReAct 真实冒烟")
    ap.add_argument("--task", default=DEFAULT_TASK, help="任务描述")
    ap.add_argument("--steps", type=int, default=6, help="步数上限")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    s = Settings()
    acc = summarize(s)
    print(f"成本闸门：已花 ¥{acc['total_cny']:.4f} / 上限 ¥{acc['limit_cny']:.2f}"
          f"（余 ¥{acc['remaining_cny']:.4f}）")

    events: list[dict] = []
    r = run_react(
        s,
        task=args.task,
        max_steps=args.steps,
        max_tokens=args.max_tokens,
        on_event=events.append,
    )

    print(f"\n任务：{args.task}\n" + "=" * 72)
    for ev in events:
        t = ev["type"]
        if t == "thought":
            print(f"  [思考] step{ev['step']}: {ev['thought']}")
        elif t == "action":
            print(f"  [行动] step{ev['step']}: {ev['tool']} {ev['args']}")
        elif t == "observation":
            body = ev["result"] if ev["error"] == "" else f"ERROR: {ev['error']}"
            txt = json.dumps(body, ensure_ascii=False, default=str)
            print(f"  [观察] step{ev['step']}: {txt[:220]}")
        elif t == "final":
            print(f"  [最终答案] step{ev['step']}: {ev['answer']}")
        elif t == "stop":
            print(f"  [停止] step{ev['step']}: reason={ev['reason']}")

    print("=" * 72)
    print(f"ok            = {r['ok']}")
    print(f"steps         = {r['steps']}（工具调用 {r['tool_calls_made']} 次）")
    print(f"stop_reason   = {r['stop_reason']}")
    print(f"tokens        = in {r['usage']['prompt_tokens']} / out {r['usage']['completion_tokens']}")
    print(f"成本          = ¥{r['cost_cny']:.6f}")
    print(f"延迟          = {r['latency_s']:.2f}s（{r['steps']} 步累加）")
    if r["answer"]:
        print(f"\n最终答案：\n{r['answer']}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
