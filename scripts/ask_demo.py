"""短片 ① 的演示脚本：把 /ask 的响应打印成「录屏友好」的样子。"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import unicodedata

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

with contextlib.redirect_stdout(io.StringIO()):
    from app.rag import AskPipeline                                    # noqa: E402
    from app.retrieval import BM25Index, RetrievalPipeline, get_store   # noqa: E402
    from app.retrieval.rerank import build_reranker                    # noqa: E402
    from app.retrieval.strategies import build_retriever               # noqa: E402

WIDTH = 78
SEP = "─" * WIDTH
VERBOSE = False


@contextlib.contextmanager
def quiet():
    """吞掉子模块往 stdout 打的进度信息。"""
    if VERBOSE:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def dwidth(s: str) -> int:
    """字符串的显示宽度（中日韩全角算 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WFA" else 1 for c in s)


def _is_wide(ch: str) -> bool:
    """是否是"可以单独作为断点"的宽字符。"""
    return unicodedata.east_asian_width(ch) in ("W", "F", "A")


NO_LINE_START = set("，。、；：！？％）】》」』〉…—·")


def _tokens(seg: str) -> list[str]:
    """把一段话切成"不可断单元"：连续 ASCII 词算一个，宽字符各算一个。"""
    out: list[str] = []
    buf = ""
    for ch in seg:
        if _is_wide(ch) or ch == " ":
            if buf:
                out.append(buf)
                buf = ""
            if ch in NO_LINE_START and out:
                out[-1] += ch
            else:
                out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def wrap(text: str, width: int = WIDTH) -> list[str]:
    """按显示宽度折行。"""
    lines: list[str] = []
    for seg in text.split("\n"):
        cur, w = "", 0
        for tok in _tokens(seg):
            tw = dwidth(tok)
            if tw > width:
                for ch in tok:
                    cw = dwidth(ch)
                    if w + cw > width and cur.strip():
                        lines.append(cur.rstrip())
                        cur, w = "", 0
                    cur += ch
                    w += cw
                continue
            if w + tw > width and cur.strip():
                lines.append(cur.rstrip())
                cur, w = "", 0
                if tok == " ":
                    continue
            cur += tok
            w += tw
        lines.append(cur.rstrip())
    return lines


def _f(obj, name: str, default=None):
    """兼容 dict 与 pydantic 对象取值（AskPipeline 的 citations 是 CitationOut 对象）。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_ask() -> AskPipeline:
    """和评测口径一致的链路：rerank 策略 + hybrid_rerank。"""
    with quiet():
        store = get_store()
        base = RetrievalPipeline(
            store=store,
            bm25=BM25Index([r["text"] for r in store.rows]),
            backend="auto",
            reranker=build_reranker("identity"),
        )
        return AskPipeline(build_retriever("rerank", base, provider="deepseek"))


def gate_info() -> dict:
    """拒答阈值的校准依据。与 GET /ask/gate 返回同一份数据。"""
    with contextlib.redirect_stdout(io.StringIO()):
        from app.rag.abstain import DEFAULT_MIN_COSINE
    return {
        "min_cosine": DEFAULT_MIN_COSINE,
        "calibrated_on": "29 条 golden set + 5 条域外问题，2026-09-25 实测",
        "in_domain_min": 0.575,
        "out_of_domain_range": [0.236, 0.397],
        "cannot_block": ("域内但答错 —— 命中组 p50 0.775 vs 未命中组 p50 0.749，"
                         "余弦在高度同质化的 API 文档语料上分不开这两者"),
        "second_gate": "模型输出 NO_ANSWER 行，兜住第一道闸门漏掉的",
    }


def print_gate() -> None:
    g = gate_info()
    print(SEP)
    print("拒答阈值是怎么定出来的")
    print(SEP)
    print()
    print(f"  阈值 = {g['min_cosine']}（最高向量分低于它就直接拒答，不问模型）")
    print(f"  依据 = {g['calibrated_on']}")
    print()
    print("  为什么取这个值：")
    print(f"    · 域内问题的最低分    {g['in_domain_min']}")
    print(f"    · 域外问题的分数区间  {g['out_of_domain_range'][0]} ~ {g['out_of_domain_range'][1]}")
    print(f"    · 阈值 {g['min_cosine']} 卡在两者中间的空档里")
    print()
    print("  它挡不住什么（这段是主动交代的）：")
    for ln in wrap(g["cannot_block"], WIDTH - 4):
        print(f"    {ln}")
    print()
    print(f"  兜底 = {g['second_gate']}")
    print()


def render(res: dict, question: str, brief: bool = False) -> None:
    """把一条问答打印出来。"""
    abstained = bool(res.get("abstained"))
    cites = res.get("citations") or []
    unsupported = res.get("unsupported") or []

    print(SEP)
    for i, ln in enumerate(wrap(f"问：{question}", WIDTH)):
        print(ln if i == 0 else f"    {ln}")
    print(SEP)
    print()

    if abstained:
        reason = res.get("abstain_reason") or ""
        gate = "检索闸门" if "检索不足" in reason else "模型闸门"
        gate_note = ("最高检索分低于阈值，没问模型就拒了" if gate == "检索闸门"
                     else "检索到了相关内容，但模型判断这些资料回答不了这个问题")
        print("【拒答】")
        for ln in wrap(reason, WIDTH - 4):
            print(f"    {ln}")
        print()
        print(f"    └ 走向：{gate}（{gate_note}）")
    else:
        print("答：")
        ans = (res.get("answer") or "").strip()
        if brief and len(ans) > 220:
            ans = ans[:220] + "……"
        for ln in wrap(ans, WIDTH - 4):
            print(f"    {ln}")
        print()
        print("依据（每一条都能点回原文核对）：")
        if cites:
            for c in cites:
                n, src, title = _f(c, "n"), _f(c, "source") or "?", _f(c, "title") or ""
                print(f"    [{n}] {src:<9} {title[:44]}")
                quote = (_f(c, "quote") or "").strip().replace("\n", " ")
                if quote:
                    for ln in wrap(f"“{quote[:64]}…”", WIDTH - 10)[:2]:
                        print(f"        {ln}")
        else:
            print("    （没有引用）")

    print()
    metrics = (f"  {res.get('mode', '?')} · {res.get('latency_ms', 0):.0f}ms · "
               f"¥{res.get('cost_cny', 0):.5f}")
    if not abstained:
        metrics += f" · {len(cites)} 条依据"
        metrics += " · 无瞎标引用" if not unsupported else f" · ⚠ 瞎标引用 {unsupported}"
    print(metrics)
    print()


def load_short1() -> tuple[dict, dict]:
    """读 pick_demo_question.py 的产出，拿短片 ① 的两条题。"""
    path = os.path.join(ROOT, "data/eval/demo_pick.json")
    if not os.path.exists(path):
        print("找不到 data/eval/demo_pick.json，先跑：")
        print("  python scripts/pick_demo_question.py --out data/eval/demo_pick.json")
        sys.exit(1)
    d = json.load(open(path, encoding="utf-8"))
    cross = [r for r in d.get("cross_doc", []) if r.get("score", 0) > 0]
    abstain = [r for r in d.get("abstain", []) if r.get("score", 0) > 0]
    if not cross or not abstain:
        print("demo_pick.json 里没有合格候选，重跑 pick_demo_question.py")
        sys.exit(1)
    cross.sort(key=lambda r: -r["score"])
    abstain.sort(key=lambda r: -r["score"])
    return cross[0], abstain[0]


def main() -> None:
    global VERBOSE

    ap = argparse.ArgumentParser(description="短片 ① 的录屏演示脚本")
    ap.add_argument("question", nargs="*", help="要问的问题")
    ap.add_argument("--gate", action="store_true", help="只打印拒答阈值（不调用 API，不花钱）")
    ap.add_argument("--short1", action="store_true",
                    help="录前热身：跑短片 ① 的两条题（跨文档 + 拒答）")
    ap.add_argument("--list", action="store_true", help="只列题，不跑")
    ap.add_argument("--brief", action="store_true", help="答案只显示前 220 字（一屏放得下）")
    ap.add_argument("--verbose", action="store_true", help="放出声明的进度日志（排查用）")
    a = ap.parse_args()
    VERBOSE = a.verbose

    if a.gate:
        print_gate()
        return

    if a.short1:
        cross, abstain = load_short1()
        if a.list:
            reason = abstain.get("abstain_reason") or ""
            gate = "检索闸门" if "检索不足" in reason else "模型闸门"
            print("短片 ① 的两条题（来自 data/eval/demo_pick.json）")
            print()
            print(f"  【第一段 · 跨文档】{cross['qid']}")
            print(f"    {cross['question']}")
            print(f"    引用：{' + '.join(cross['cited_sources'])}"
                  f"（{len(cross['citations'])} 条）")
            print()
            print(f"  【第二段 · 拒答】{abstain['question']}")
            print(f"    走向：{gate}")
            print(f"    理由：{reason[:60]}")
            return
        ask = build_ask()
        print()
        print("短片 ① 热身跑（两条题，约 ¥0.004）")
        print()
        for q in (cross["question"], abstain["question"]):
            with quiet():
                res = ask.ask(q, mode="hybrid", topk=5, pool=50)
            render(res, q, brief=a.brief)
        print_gate()
        return

    if not a.question:
        ap.print_help()
        return

    q = " ".join(a.question)
    ask = build_ask()
    with quiet():
        res = ask.ask(q, mode="hybrid", topk=5, pool=50)
    render(res, q, brief=a.brief)


if __name__ == "__main__":
    main()
