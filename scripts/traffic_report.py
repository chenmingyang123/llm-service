"""把 logs/traffic-*.jsonl 渲染成 Markdown 实验记录。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.traffic import LOG_DIR  # noqa: E402


def load(date: str, path_filter: str | None) -> list[dict]:
    p = os.path.join(LOG_DIR, f"traffic-{date}.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if path_filter and not e.get("path", "").startswith(path_filter):
                continue
            out.append(e)
    return out


def block(obj) -> str:
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


def render(entries: list[dict]) -> str:
    if not entries:
        return "_（这一天没有记录）_"

    lines: list[str] = []
    total_cost = 0.0
    for i, e in enumerate(entries, 1):
        resp = e.get("response") or {}
        if isinstance(resp, str):
            resp = {}
        cost = resp.get("cost_cny") or 0.0
        total_cost += cost

        lines.append(f"## #{i} · {e.get('ts', '')} · {e.get('method','')} {e.get('path','')} "
                     f"· HTTP {e.get('status','')} · {e.get('latency_s','')}s")
        lines.append("")

        lines.append("**请求**")
        lines.append(block(e.get("request")))
        lines.append("")

        lines.append("**结果摘要**")
        lines.append("")
        lines.append("| 项 | 值 |")
        lines.append("|---|---|")
        for k in ("ok", "turns", "tool_calls_made", "stop_reason", "provider", "model",
                  "cost_cny", "latency_s"):
            if k in resp:
                lines.append(f"| {k} | `{resp[k]}` |")
        answer = resp.get("answer") or resp.get("data") or ""
        if answer:
            lines.append("")
            lines.append(f"**answer**：{answer}")
        lines.append("")

        trace = resp.get("trace")
        if trace:
            lines.append("**调用轨迹**")
            lines.append("")
            lines.append("| 轮次 | 工具 | 参数 | 结果 | 错误 |")
            lines.append("|---|---|---|---|---|")
            for t in trace:
                res = t.get("result")
                res_s = "-" if res is None else ("`" + json.dumps(res, ensure_ascii=False) + "`")
                lines.append(f"| {t.get('turn','')} | `{t.get('tool','')}` | "
                             f"`{t.get('args','')}` | {res_s} | {t.get('error') or '-'} |")
            lines.append("")

        lines.append("<details><summary>原始响应</summary>")
        lines.append("")
        lines.append(block(e.get("response")))
        lines.append("")
        lines.append("</details>")
        lines.append("")

    head = [
        f"# HTTP 流量实验记录 · {entries[0].get('ts','')[:10]}",
        "",
        f"- 记录条数：**{len(entries)}**",
        f"- 累计成本：**¥{round(total_cost, 6)}**",
        f"- 数据来源：`logs/traffic-*.jsonl`（服务自动落盘，非手工誊抄）",
        "",
        "---",
        "",
    ]
    return "\n".join(head + lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--path", default=None, help="只看某个前缀，如 /agent")
    ap.add_argument("--out", default=None, help="写入文件；不给就打印到屏幕")
    a = ap.parse_args()

    md = render(load(a.date, a.path))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"已写入 {a.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
