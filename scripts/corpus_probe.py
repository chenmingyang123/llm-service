"""语料可得性探测 —— 9/25 定方向时先用它验一遍，别到 9/26 才发现抓不到。"""
from __future__ import annotations

import argparse
import os
import re
import ssl
import sys
import urllib.request
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

SOURCES = {
    "deepseek": {
        "home": "https://api-docs.deepseek.com/zh-cn/",
        "prefix": "https://api-docs.deepseek.com/zh-cn/",
        "link": r'href="(/zh-cn/[^"#?]*\.?[^"#?]*)"',
        "bad": (".css", ".js", ".png", ".jpg", ".svg", ".ico", ".woff"),
    },
    "zhipu": {
        "home": "https://docs.bigmodel.cn/",
        "prefix": "https://docs.bigmodel.cn/cn/",
        "link": r'href="(/cn/[^"#?]*)"',
        "bad": (".css", ".js", ".png", ".jpg", ".svg", ".ico", ".woff"),
    },
    "bailian": {
        "home": "https://help.aliyun.com/zh/model-studio/",
        "prefix": "https://help.aliyun.com/zh/model-studio/",
        "link": r'href="(/zh/model-studio/[^"#?]*)"',
        "bad": (".css", ".js", ".png", ".jpg", ".svg", ".ico", ".woff"),
    },
}


def fetch(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read().decode("utf-8", "replace")


def to_text(html: str) -> str:
    """粗略抽正文。够用来判断"这一页有没有内容"，不是最终解析方案。"""
    t = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def probe(name: str, cfg: dict, max_pages: int) -> dict:
    seen: set[str] = set()
    queue = deque([cfg["home"]])
    good: list[tuple[str, int]] = []

    while queue and len(seen) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        try:
            html = fetch(url)
        except Exception as e:  # noqa: BLE001
            continue

        text = to_text(html)
        if len(text) >= 500:
            good.append((url, len(text)))

        for href in re.findall(cfg["link"], html):
            if any(href.endswith(b) for b in cfg["bad"]):
                continue
            full = urllib.parse.urljoin(cfg["home"], href)
            if full.startswith(cfg["prefix"]) and full not in seen:
                queue.append(full)

    return {"source": name, "crawled": len(seen), "usable": len(good),
            "avg_chars": (sum(c for _, c in good) // len(good)) if good else 0,
            "sample": [u for u, _ in good[:6]]}


def main() -> None:
    import urllib.parse  # noqa: F401  (probe 里用得到)

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=40, help="每家最多爬多少页")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for name, cfg in SOURCES.items():
        try:
            r = probe(name, cfg, a.max_pages)
        except Exception as e:  # noqa: BLE001
            rows.append({"source": name, "crawled": 0, "usable": 0, "avg_chars": 0,
                         "sample": [], "error": f"{type(e).__name__}: {e}"})
            print(f"[{name:9}] 探测失败")
            continue
        rows.append(r)
        print(f"[{name:9}] 爬到 {r['crawled']:3d} 页 → 有效文档 {r['usable']:3d} 篇 "
              f"（平均 {r['avg_chars']} 字）")

    total = sum(r["usable"] for r in rows)
    print(f"\n合计有效文档页：{total} 篇")

    md = ["# 语料可得性探测", "",
          f"- 探测方式：从各文档站首页 BFS 两层，每家上限 {a.max_pages} 页",
          "- 有效判定：正文 ≥ 500 字符（低于此基本是导航页或空壳）", "",
          "| 来源 | 爬到 | 有效文档 | 平均字数 | 状态 |", "|---|---|---|---|---|"]
    for r in rows:
        status = "可达" if r["usable"] > 0 else ("失败：" + r.get("error", "")) if r.get("error") else "无有效页"
        md.append(f"| {r['source']} | {r['crawled']} | {r['usable']} | {r['avg_chars']} | {status} |")
    md += ["", f"**合计：{total} 篇。**", "",
           "## 判定", "",
           "- 合计 ≥ 200 篇 → 语料规模达标，9/26 可直接开抓。",
           "- 合计 100–200 篇 → 够用但偏薄，需要加第二层语料补足。",
           "- 合计 < 100 篇 → 换方案，别硬凑。"]
    if total >= 200:
        md.append(f"\n本次结果：**{total} 篇，达标。**")
    elif total >= 100:
        md.append(f"\n本次结果：**{total} 篇，偏薄** —— 建议加第二层语料补足到 300 篇左右。")
    else:
        md.append(f"\n本次结果：**{total} 篇，不达标** —— 换备选语料。")

    out = "\n".join(md)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"已写入 {a.out}")
    else:
        print()
        print(out)


if __name__ == "__main__":
    main()
