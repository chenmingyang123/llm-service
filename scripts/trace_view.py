"""W4 块 E｜trace 查看器：把 data/traces/*.jsonl 渲染成单文件 HTML。"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.observe import TRACE_DIR, read_trace, merge_patches  # noqa: E402

KIND_COLOR = {
    "root": "#64748b",
    "step": "#2563eb",
    "llm": "#7c3aed",
    "tool": "#059669",
    "retrieval": "#0891b2",
    "confirm": "#d97706",
}
FALLBACK_COLOR = "#94a3b8"


def build_rows(recs: list[dict]) -> list[tuple[int, dict]]:
    """按 parent 还原成树，再摊平成 (depth, record) 列表。"""
    by_parent: dict[str, list[dict]] = {}
    roots: list[dict] = []
    for r in recs:
        p = r.get("parent")
        if p is None:
            roots.append(r)
        else:
            by_parent.setdefault(p, []).append(r)
    for v in by_parent.values():
        v.sort(key=lambda r: r.get("t_epoch") or 0.0)
    roots.sort(key=lambda r: r.get("t_epoch") or 0.0)

    rows: list[tuple[int, dict]] = []

    def walk(r: dict, depth: int) -> None:
        rows.append((depth, r))
        for c in by_parent.get(r.get("span_id"), []):
            walk(c, depth + 1)

    for r in roots:
        walk(r, 0)
    return rows


def summarize(recs: list[dict]) -> dict:
    root = next((r for r in recs if r.get("kind") == "root"), recs[0] if recs else {})
    t0 = min((r.get("t_epoch") or 0.0) for r in recs) if recs else 0.0
    t1 = max((r.get("end_epoch") or 0.0) for r in recs) if recs else 0.0
    total_ms = (t1 - t0) * 1000

    by_kind: dict[str, dict] = {}
    for r in recs:
        if r.get("kind") == "root":
            continue
        slot = by_kind.setdefault(r.get("kind") or "span", {"count": 0, "ms": 0.0})
        slot["count"] += 1
        slot["ms"] += r.get("dur_ms") or 0.0

    errs = [r for r in recs if r.get("error")]
    return {
        "root": root,
        "t0": t0,
        "total_ms": round(total_ms, 2),
        "span_count": len(recs),
        "error_count": len(errs),
        "by_kind": {k: {"count": v["count"], "ms": round(v["ms"], 2),
                        "pct": round(v["ms"] / total_ms * 100, 1) if total_ms else 0.0}
                    for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]["ms"])},
        "cost_cny": (root.get("meta") or {}).get("cost_cny"),
        "usage": (root.get("meta") or {}).get("usage"),
        "stop_reason": (root.get("meta") or {}).get("stop_reason"),
    }


def _preview(v: object, n: int = 180) -> str:
    if v is None:
        return ""
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return html.escape(s[:n] + ("…" if len(s) > n else ""))


def render(recs: list[dict], out_html: str) -> str:
    recs = merge_patches(recs)
    if not recs:
        raise ValueError("trace 为空，没什么可渲染的")

    info = summarize(recs)
    rows = build_rows(recs)
    t0 = info["t0"]
    total_ms = info["total_ms"] or 1.0

    root_meta = info["root"].get("meta") or {}
    usage = info["usage"] or {}
    status = info["root"].get("status", "?")
    ok_badge = ("<span class='badge ok'>OK</span>" if status == "ok"
                else f"<span class='badge err'>{html.escape(str(status))}</span>")

    cards = [
        ("总耗时", f"{info['total_ms']:.0f} ms"),
        ("Span 数", str(info["span_count"])),
        ("出错", str(info["error_count"])),
        ("成本", f"¥{info['cost_cny']}" if info["cost_cny"] is not None else "—"),
        ("Token", f"{usage.get('prompt_tokens', usage.get('in', '—'))} in / "
                  f"{usage.get('completion_tokens', usage.get('out', '—'))} out"
         if usage else "—"),
        ("停止原因", str(info["stop_reason"] or "—")),
    ]
    card_html = "".join(
        f"<div class='card'><div class='k'>{html.escape(k)}</div>"
        f"<div class='v'>{html.escape(v)}</div></div>" for k, v in cards)

    segs, legend = [], []
    for k, v in info["by_kind"].items():
        color = KIND_COLOR.get(k, FALLBACK_COLOR)
        segs.append(f"<div class='seg' style='width:{max(v['pct'], 0.6):.2f}%;"
                    f"background:{color}' title='{html.escape(k)} "
                    f"{v['ms']:.0f}ms'></div>")
        legend.append(f"<span class='lg'><i style='background:{color}'></i>"
                      f"{html.escape(k)} · {v['count']} 次 · {v['ms']:.0f} ms "
                      f"({v['pct']}%)</span>")
    bar_html = (f"<div class='bar'>{''.join(segs)}</div>"
                f"<div class='legend'>{''.join(legend)}</div>")

    lines = []
    for depth, r in rows:
        off = ((r.get("t_epoch") or t0) - t0) * 1000
        dur = r.get("dur_ms") or 0.0
        left = off / total_ms * 100
        width = max(dur / total_ms * 100, 0.4)
        color = KIND_COLOR.get(r.get("kind") or "", FALLBACK_COLOR)
        err = r.get("error")
        err_cls = " row-err" if err else ""
        name = html.escape(r.get("name") or "?")
        meta_bits = []
        th = (r.get("meta") or {}).get("thought")
        inp, out = r.get("input"), r.get("output")
        if th:
            meta_bits.append("思考：" + _preview(th, 140))
        if inp not in (None, "", {}):
            meta_bits.append("入参：" + _preview(inp, 120))
        if out not in (None, "", {}):
            meta_bits.append("出参：" + _preview(out, 160))
        if err:
            meta_bits.append("<span class='errmsg'>错误：" + _preview(err, 200) + "</span>")
        meta_html = ("<div class='meta'>" + " ｜ ".join(meta_bits) + "</div>"
                     if meta_bits else "")

        lines.append(
            f"<div class='row{err_cls}'>"
            f"<div class='nm' style='padding-left:{depth * 16}px'>"
            f"{'└ ' if depth else ''}{name}"
            f"<span class='ms'>{dur:.0f}ms</span></div>"
            f"<div class='track'><div class='fill' style='left:{left:.2f}%;"
            f"width:{width:.2f}%;background:{color}'></div></div>"
            f"{meta_html}</div>")

    task = _preview(root_meta.get("task"), 200)
    tid = html.escape(str(info["root"].get("trace_id") or ""))

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Trace {tid}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;padding:28px;background:#f8fafc;color:#0f172a;
 font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
h1{{font-size:19px;margin:0 0 4px}}
.sub{{color:#64748b;font-size:13px;margin-bottom:18px}}
.badge{{display:inline-block;padding:2px 9px;border-radius:10px;font-size:12px;
 font-weight:600;vertical-align:middle;margin-left:8px}}
.badge.ok{{background:#dcfce7;color:#15803d}}
.badge.err{{background:#fee2e2;color:#b91c1c}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
 gap:12px;margin-bottom:18px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px}}
.card .k{{color:#64748b;font-size:12px}}
.card .v{{font-size:17px;font-weight:600;margin-top:3px}}
h2{{font-size:14px;margin:22px 0 10px;color:#334155}}
.bar{{display:flex;height:16px;border-radius:8px;overflow:hidden;background:#eef2f7}}
.seg{{height:100%}}
.legend{{margin-top:8px;color:#475569;font-size:12px;display:flex;flex-wrap:wrap;gap:14px}}
.lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}
.rows{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden}}
.row{{display:grid;grid-template-columns:290px 1fr;gap:12px;align-items:center;
 padding:8px 12px;border-bottom:1px solid #f1f5f9}}
.row:last-child{{border-bottom:none}}
.row-err{{background:#fff7f7}}
.nm{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.ms{{color:#94a3b8;margin-left:6px;font-size:11px}}
.track{{position:relative;height:16px;background:#f1f5f9;border-radius:4px}}
.fill{{position:absolute;top:0;height:100%;border-radius:4px}}
.meta{{grid-column:1 / -1;color:#64748b;font-size:12px;padding-left:6px}}
.errmsg{{color:#b91c1c}}
.foot{{margin-top:18px;color:#94a3b8;font-size:12px}}
</style></head><body>
<h1>Agent Trace {ok_badge}</h1>
<div class="sub">trace_id <code>{tid}</code> ｜ {len(recs)} span ｜ 任务：{task}</div>
<div class="cards">{card_html}</div>
<h2>时间都花在哪了</h2>
{bar_html}
<h2>瀑布图（缩进 = 父子关系，条的位置 = 开始时刻）</h2>
<div class="rows">{''.join(lines)}</div>
<div class="foot">由 scripts/trace_view.py 从本地 JSONL 生成 ｜
 数据真相来源：data/traces/{tid}.jsonl</div>
</body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_html


def latest_trace() -> str:
    if not os.path.isdir(TRACE_DIR):
        raise SystemExit(f"还没有 trace 目录：{TRACE_DIR}\n"
                         f"先跑：python scripts/observe_smoke.py")
    files = [os.path.join(TRACE_DIR, f) for f in os.listdir(TRACE_DIR)
             if f.endswith(".jsonl")]
    if not files:
        raise SystemExit(f"{TRACE_DIR} 下没有 .jsonl，先跑：python scripts/observe_smoke.py")
    return max(files, key=os.path.getmtime)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="trace 的 .jsonl 路径；省略则用最新一条")
    ap.add_argument("-o", "--out", help="输出 HTML 路径（默认与 jsonl 同名 .html）")
    ap.add_argument("--open", action="store_true", help="渲染完用浏览器打开")
    args = ap.parse_args()

    src = args.path or latest_trace()
    if not os.path.exists(src):
        raise SystemExit(f"找不到：{src}")
    out = args.out or (os.path.splitext(src)[0] + ".html")

    recs = read_trace(src)
    render(recs, out)
    print(f"已渲染 {len(recs)} 个 span → {out}")
    if args.open:
        webbrowser.open("file:///" + os.path.abspath(out).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
