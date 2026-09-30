"""W2 第 1 天 · 语料解析 —— 把 data/raw/ 里的 HTML / Markdown 统一成结构化 JSON。"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PARSED = os.path.join(ROOT, "data", "parsed")

MIN_CHARS = 300
SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside"}


class StructureExtractor(HTMLParser):
    """把一段 HTML 抽成 (标题, 段落, 表格, 代码块) 的线性序列。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict] = []
        self._skip_depth = 0
        self._in_pre = False
        self._pre_lang = ""
        self._pre_buf: list[str] = []
        self._in_table = False
        self._in_cell = False
        self._cell_buf: list[str] = []
        self._rows: list[list[str]] = []
        self._row: list[str] = []
        self._in_heading = 0
        self._head_buf: list[str] = []
        self._in_p = False
        self._p_buf: list[str] = []

    @staticmethod
    def _clean(s: str) -> str:
        s = html.unescape(s)
        s = re.sub(r"[\u200b-\u200f\ufeff\u00ad]", "", s)
        return re.sub(r"\s+", " ", s).strip()

    def _flush_p(self) -> None:
        t = self._clean("".join(self._p_buf))
        self._p_buf.clear()
        if t:
            self.items.append({"type": "p", "text": t})

    def _flush_heading(self) -> None:
        t = self._clean("".join(self._head_buf))
        self._head_buf.clear()
        if t:
            self.items.append({"type": "h", "level": self._in_heading, "text": t})

    def _flush_cell(self) -> None:
        self._row.append(self._clean("".join(self._cell_buf)))
        self._cell_buf.clear()

    def _flush_table(self) -> None:
        self._flush_cell()
        if self._row:
            self._rows.append(self._row)
            self._row = []
        rows = [r for r in self._rows if any(c for c in r)]
        if rows:
            self.items.append({"type": "table", "rows": rows})
        self._rows = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if tag == "pre":
            self._in_pre = True
            self._pre_lang = ""
            cls = (d.get("class") or "") + " " + (d.get("outputclass") or "")
            m = re.search(r"language-([\w+#-]+)", cls)
            if m:
                self._pre_lang = m.group(1)
            self._pre_buf = []
            return
        if tag == "table":
            self._in_table = True
            self._rows = []
            self._row = []
            return
        if tag == "tr" and self._in_table:
            self._row = []
            return
        if tag in ("td", "th") and self._in_table:
            self._in_cell = True
            self._cell_buf = []
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._flush_p()
            self._in_heading = int(tag[1])
            self._head_buf = []
            return
        if tag == "p":
            self._in_p = True
            self._p_buf = []
            return
        if tag in ("br", "li"):
            if self._in_p:
                self._p_buf.append(" ")
            return

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return

        if tag == "pre" and self._in_pre:
            self._in_pre = False
            code = "".join(self._pre_buf).strip("\n")
            self._pre_buf = []
            if code.strip():
                self.items.append({"type": "code", "lang": self._pre_lang, "code": code})
            return
        if tag == "table" and self._in_table:
            self._in_table = False
            self._flush_table()
            return
        if tag == "tr" and self._in_table:
            self._flush_cell()
            if self._row:
                self._rows.append(self._row)
            self._row = []
            return
        if tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            self._flush_cell()
            return
        if re.fullmatch(r"h[1-6]", tag) and self._in_heading:
            self._flush_heading()
            self._in_heading = 0
            return
        if tag == "p" and self._in_p:
            self._in_p = False
            self._flush_p()
            return

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_pre:
            self._pre_buf.append(data)
        elif self._in_cell:
            self._cell_buf.append(data)
        elif self._in_heading:
            self._head_buf.append(data)
        elif self._in_p:
            self._p_buf.append(data)

    def finish(self) -> None:
        self._flush_p()
        self._flush_heading()


def _balanced_slice(doc: str, start: int, tag: str) -> str:
    """从 start 处的开标签之后起，按同名标签配平，切出完整容器内容。"""
    pat = re.compile(rf"</?{tag}\b[^>]*>", flags=re.I)
    depth, i, n = 1, start, len(doc)
    while i < n:
        m = pat.search(doc, i)
        if not m:
            return doc[start:]
        if m.group(0).lower().startswith(f"</{tag}"):
            depth -= 1
            if depth == 0:
                return doc[start:m.start()]
        else:
            depth += 1
        i = m.end()
    return doc[start:]


def pick_main_html(doc: str) -> str:
    """从整页 HTML 里圈出正文区域。"""
    m = re.search(r"<div[^>]*icms-help-docs-content[^>]*>", doc, flags=re.I)
    if m:
        return _balanced_slice(doc, m.end(), "div")

    for tag in ("article", "main"):
        m = re.search(rf"<{tag}\b[^>]*>", doc, flags=re.I)
        if m:
            body = _balanced_slice(doc, m.end(), tag)
            if body.count("<p") >= 2:
                return body

    blocks = re.split(r"(?=<div\b[^>]*>)", doc, flags=re.I)
    if len(blocks) > 1:
        best = max(blocks, key=lambda b: b.count("<p"))
        if best.count("<p") >= 3:
            return best
    return doc


def unescape_nested(doc: str) -> str:
    """百炼的正文整段塞在 JS 字符串里，引号被转义成 \"，先还原成正常标签。"""
    if '\\"' not in doc and "\\n" not in doc:
        return doc
    out = (doc.replace('\\"', '"')
              .replace("\\n", "\n")
              .replace("\\t", "\t")
              .replace("\\/", "/")
              .replace("\\\\", "\\"))
    return out


def parse_markdown(md: str) -> list[dict]:
    """智谱给的是标准 Markdown，结构天然完整，只需按块类型切出来。"""
    md = re.sub(r"^>\s+.*$", "", md, flags=re.M)
    md = re.sub(r"^>\s*$", "", md, flags=re.M)
    items: list[dict] = []
    lines = md.split("\n")
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            items.append({"type": "h", "level": len(m.group(1)), "text": m.group(2).strip()})
            i += 1
            continue

        if line.strip().startswith("```"):
            lang = line.strip()[3:].strip()
            buf, i = [], i + 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            items.append({"type": "code", "lang": lang, "code": "\n".join(buf)})
            i += 1
            continue

        if line.strip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            rows, i2 = [], i
            while i2 < n and lines[i2].strip().startswith("|"):
                cells = [c.strip() for c in lines[i2].strip().strip("|").split("|")]
                if not re.match(r"^[\s:|-]+$", "".join(cells)):
                    rows.append(cells)
                i2 += 1
            if rows:
                items.append({"type": "table", "rows": rows})
            i = i2
            continue

        if line.count("](#") >= 2:
            i += 1
            continue

        if line.strip():
            items.append({"type": "p", "text": line.strip()})
        i += 1
    return items


def table_to_markdown(rows: list[list[str]]) -> str:
    """把解析出来的表格还原成 Markdown 表格。"""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    def fmt(r: list[str]) -> str:
        cells = [c.replace("|", "\\|") for c in r] + [""] * (width - len(r))
        return "| " + " | ".join(cells[:width]) + " |"
    head = fmt(rows[0])
    sep = "| " + " | ".join(["---"] * width) + " |"
    return "\n".join([head, sep] + [fmt(r) for r in rows[1:]])


def build_doc(items: list[dict], code_in_text: bool = False) -> dict:
    """把线性序列整理成 章节 / 表格 / 代码块 三段式。"""
    sections: list[dict] = []
    tables: list[dict] = []
    codes: list[dict] = []
    path: list[str] = []
    level_of: dict[str, int] = {}

    def cur_heading() -> str:
        return path[-1] if path else ""

    buf: list[str] = []

    def flush_section() -> None:
        nonlocal buf
        t = " ".join(buf).strip()
        buf = []
        if t:
            sections.append({"heading": cur_heading(), "path": list(path), "text": t})

    for it in items:
        if it["type"] == "h":
            flush_section()
            lv = it["level"]
            level_of[it["text"]] = lv
            while path and level_of.get(path[-1], 0) >= lv:
                path.pop()
            path.append(it["text"])
        elif it["type"] == "p":
            buf.append(it["text"])
        elif it["type"] == "table":
            tables.append({"heading": cur_heading(), "markdown": table_to_markdown(it["rows"])})
            buf.append(table_to_markdown(it["rows"]))
        elif it["type"] == "code":
            codes.append({"heading": cur_heading(), "lang": it.get("lang", ""), "code": it["code"]})
            tok = (it.get("lang") or "").lstrip("`").replace("/", " ").split()
            buf.append(it["code"] if code_in_text else f"[代码示例 {tok[0] if tok else ''}]")

    flush_section()
    text = "\n\n".join(s["text"] for s in sections)
    return {"sections": sections, "tables": tables, "code_blocks": codes, "text": text}


def est_tokens(text: str) -> int:
    """粗估 token 数。中文约 1 字 0.6 token，英文约 1 词 1.3 token。"""
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]+", text))
    return int(zh * 0.6 + en * 1.3)


def parse_one(row: dict) -> tuple[dict | None, str | None]:
    """返回 (文档, 拒绝原因)。拒绝原因非 None 表示这篇没进语料库。"""
    if row.get("status") != 200 or not row.get("path"):
        return None, f"抓取失败：{row.get('error') or row.get('status')}"

    path = os.path.join(ROOT, row["path"])
    if not os.path.exists(path):
        return None, "文件缺失"
    raw = open(path, encoding="utf-8", errors="replace").read()

    source = row["source"]
    if source == "zhipu" or path.endswith(".md"):
        items = parse_markdown(raw)
        title = next((i["text"] for i in items if i["type"] == "h" and i["level"] == 1), "")
    else:
        doc = unescape_nested(raw) if source == "bailian" else raw
        main = pick_main_html(doc)
        ex = StructureExtractor()
        try:
            ex.feed(main)
            ex.finish()
        except Exception as e:  # noqa: BLE001
            return None, f"HTML 解析异常：{type(e).__name__}: {e}"
        items = ex.items
        m = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.S | re.I)
        title = ""
        if m:
            title = re.sub(r"\s*[-|_|]\s*(阿里云|DeepSeek|智谱).*$", "", html.unescape(m.group(1))).strip()
        if not title:
            h1 = next((i["text"] for i in items if i["type"] == "h" and i["level"] == 1), "")
            title = h1

    built = build_doc(items)
    text = built["text"]
    doc_type = "tutorial"
    if len(re.sub(r"\s+", "", text)) < MIN_CHARS:
        alt = build_doc(items, code_in_text=True)
        if len(re.sub(r"\s+", "", alt["text"])) >= MIN_CHARS:
            built, text, doc_type = alt, alt["text"], "reference"
        else:
            return None, f"正文过短（{len(text)} 字 < {MIN_CHARS}）"

    doc = {
        "doc_id": row["doc_id"],
        "source": source,
        "url": row["url"],
        "title": title or row["url"].rsplit("/", 1)[-1],
        "doc_type": doc_type,
        "n_chars": len(text),
        "n_tokens_est": est_tokens(text),
        "stats": {
            "n_sections": len(built["sections"]),
            "n_tables": len(built["tables"]),
            "n_code_blocks": len(built["code_blocks"]),
        },
        "sections": built["sections"],
        "tables": built["tables"],
        "code_blocks": built["code_blocks"],
        "text": text,
    }
    return doc, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--show", default=None, help="打印某家一篇的解析结果")
    a = ap.parse_args()

    man = os.path.join(RAW, "manifest.jsonl")
    if not os.path.exists(man):
        print("没有 manifest，先跑：python scripts/fetch_corpus.py")
        sys.exit(1)

    rows = [json.loads(l) for l in open(man, encoding="utf-8")]
    if a.source:
        rows = [r for r in rows if r["source"] == a.source]
    if a.limit:
        rows = rows[:a.limit]

    docs: list[dict] = []
    rejected: list[dict] = []
    seen_digest: dict[str, str] = {}
    for r in rows:
        d, why = parse_one(r)
        if d is None:
            rejected.append({"doc_id": r["doc_id"], "source": r["source"],
                             "url": r["url"], "reason": why})
            continue

        digest = hashlib.md5(re.sub(r"\s+", "", d["text"]).encode("utf-8")).hexdigest()
        if digest in seen_digest:
            rejected.append({"doc_id": d["doc_id"], "source": d["source"],
                             "url": d["url"],
                             "reason": f"与 {seen_digest[digest]} 正文重复"})
            continue
        seen_digest[digest] = d["url"]
        docs.append(d)

    os.makedirs(PARSED, exist_ok=True)
    out = os.path.join(PARSED, "corpus.jsonl")
    rej = os.path.join(PARSED, "rejected.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(rej, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 68)
    print(f"{'来源':<10}{'入库':>6}{'拒绝':>6}{'平均字':>8}{'表格':>7}{'代码块':>7}{'token':>8}")
    print("-" * 68)
    for s in sorted({r["source"] for r in rows}):
        ds = [d for d in docs if d["source"] == s]
        rj = [r for r in rejected if r["source"] == s]
        if not ds:
            print(f"{s:<10}{0:>6}{len(rj):>6}        —— 全部被拒 ——")
            continue
        print(f"{s:<10}{len(ds):>6}{len(rj):>6}"
              f"{sum(d['n_chars'] for d in ds) // len(ds):>8}"
              f"{sum(d['stats']['n_tables'] for d in ds):>7}"
              f"{sum(d['stats']['n_code_blocks'] for d in ds):>7}"
              f"{sum(d['n_tokens_est'] for d in ds):>8}")
    print("-" * 68)
    tot_t = sum(d["stats"]["n_tables"] for d in docs)
    tot_c = sum(d["stats"]["n_code_blocks"] for d in docs)
    print(f"{'合计':<10}{len(docs):>6}{len(rejected):>6}"
          f"{(sum(d['n_chars'] for d in docs) // len(docs)) if docs else 0:>8}"
          f"{tot_t:>7}{tot_c:>7}{sum(d['n_tokens_est'] for d in docs):>8}")
    print()
    print(f"corpus   → {os.path.relpath(out, ROOT).replace(chr(92), '/')}")
    print(f"rejected → {os.path.relpath(rej, ROOT).replace(chr(92), '/')}")

    if rejected:
        from collections import Counter
        print(f"\n拒绝 {len(rejected)} 篇，原因分布：")
        for reason, n in Counter(r["reason"].split("（")[0] for r in rejected).most_common():
            print(f"  {n:4d}  {reason}")

    if docs:
        ts = sorted(d["n_chars"] for d in docs)
        p = lambda q: ts[min(len(ts) - 1, int(len(ts) * q))]  # noqa: E731
        print(f"\n正文长度分布：p10={p(.10)}  p50={p(.50)}  p90={p(.90)}  max={ts[-1]}")
        n_ref = sum(1 for d in docs if d["doc_type"] == "reference")
        print(f"文档类型：教程型 {len(docs) - n_ref}　参考型 {n_ref}（代码即正文，靠回退才救回来）")
        print(f"预估总 token：{sum(d['n_tokens_est'] for d in docs):,}")

    if a.show:
        ds = [d for d in docs if d["source"] == a.show]
        if not ds:
            print(f"\n{a.show} 没有入库文档")
            return
        d = max(ds, key=lambda x: x["stats"]["n_tables"]) if any(
            x["stats"]["n_tables"] for x in ds) else max(ds, key=lambda x: x["n_chars"])
        print("\n" + "=" * 68)
        print(f"样例 [{d['source']}] {d['title']}\n{d['url']}")
        print(f"字符 {d['n_chars']}  章节 {d['stats']['n_sections']}  "
              f"表格 {d['stats']['n_tables']}  代码块 {d['stats']['n_code_blocks']}")
        print("-" * 68)
        for s in d["sections"][:3]:
            hd = s["heading"] or "(无标题)"
            print(f"\n## {hd}   [path: {' > '.join(s['path']) or '-'}]")
            print(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))
        if d["tables"]:
            print("\n--- 表格（转 Markdown 后）---")
            print(d["tables"][0]["markdown"][:400])
        if d["code_blocks"]:
            print(f"\n--- 代码块 [{d['code_blocks'][0]['lang']}] ---")
            print(d["code_blocks"][0]["code"][:300])


if __name__ == "__main__":
    main()
