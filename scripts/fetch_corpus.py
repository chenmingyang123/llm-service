"""W2 第 1 天 · 语料抓取 —— 把三家 LLM 平台文档落到本地 data/raw/。"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

SOURCES: dict[str, dict] = {
    "deepseek": {
        "index": "https://api-docs.deepseek.com/zh-cn/sitemap.xml",
        "kind": "sitemap",
        "keep": lambda u: "/zh-cn/" in u and u.count("/") >= 5,
        "ext": "html",
        "delay": (0.4, 0.9),
    },
    "zhipu": {
        "index": "https://docs.bigmodel.cn/llms.txt",
        "kind": "llms",
        "keep": lambda u: (u.startswith("https://docs.bigmodel.cn/")
                           and "/en/" not in u),
        "ext": "md",
        "delay": (0.3, 0.7),
    },
    "bailian": {
        "index": "https://help.aliyun.com/zh/model-studio/llms.txt",
        "kind": "llms",
        "keep": lambda u: u.startswith("https://help.aliyun.com/zh/model-studio/"),
        "rewrite": lambda u: u[:-3] if u.endswith(".md") else u,
        "ext": "html",
        "delay": (0.5, 1.1),
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_url(url: str) -> str:
    """把 URL 路径里的中文做 percent-encode。"""
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((
        p.scheme, p.netloc, urllib.parse.quote(p.path, safe="/%~"),
        p.query, p.fragment))


def http_get(url: str, timeout: float = 30.0, retries: int = 2) -> tuple[int, bytes]:
    """带重试的 GET。返回 (status, body)。失败抛最后一次的异常。"""
    url = safe_url(url)
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return r.status, r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1) + random.random() * 0.4)
    raise last  # type: ignore[misc]


def build_manifest(name: str, cfg: dict) -> list[dict]:
    """从索引文件解析出待抓 URL 列表。"""
    status, body = http_get(cfg["index"])
    text = body.decode("utf-8", "replace")
    log(f"  [{name:8}] 索引 {cfg['index']} → {status}, {len(text)} 字符")

    if cfg["kind"] == "sitemap":
        urls = re.findall(r"<loc>(.*?)</loc>", text)
    else:
        urls = re.findall(r"\((https?://[^)\s]+)\)", text)

    out, seen = [], set()
    for u in urls:
        u = urllib.parse.urldefrag(u)[0]
        if cfg.get("rewrite"):
            u = cfg["rewrite"](u)
        if not cfg["keep"](u) or u in seen:
            continue
        seen.add(u)
        out.append({"source": name, "url": u,
                    "doc_id": hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]})
    return out


def target_path(item: dict, cfg: dict) -> str:
    """落盘路径。文件名用 URL 的 sha1 前 12 位，不用序号。"""
    return os.path.join(RAW, item["source"], f"{item['doc_id']}.{cfg['ext']}")


def fetch_one(item: dict, cfg: dict, force: bool) -> dict:
    """抓一篇，返回写回 manifest 的那一行。"""
    path = target_path(item, cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) > 0 and not force:
        item.update(status=200, bytes=os.path.getsize(path),
                    path=os.path.relpath(path, ROOT).replace("\\", "/"),
                    cached=True, fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        return item

    lo, hi = cfg["delay"]
    time.sleep(random.uniform(lo, hi))
    try:
        status, body = http_get(item["url"])
        with open(path, "wb") as f:
            f.write(body)
        item.update(status=status, bytes=len(body),
                    path=os.path.relpath(path, ROOT).replace("\\", "/"),
                    cached=False, error=None,
                    fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    except Exception as e:  # noqa: BLE001
        item.update(status=0, bytes=0, path=None, cached=False,
                    error=f"{type(e).__name__}: {e}",
                    fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return item


def run_source(name: str, cfg: dict, limit: int | None, workers: int, force: bool) -> list[dict]:
    """抓完一家。并发数由 workers 控制，每家的请求间隔由 cfg["delay"] 控制。"""
    items = build_manifest(name, cfg)
    if limit:
        items = items[:limit]
    log(f"  [{name:8}] 待抓 {len(items)} 篇，并发 {workers}")

    done, t0 = [], time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_one, it, cfg, force) for it in items]
        for n, fut in enumerate(cf.as_completed(futs), 1):
            done.append(fut.result())
            if n % 25 == 0 or n == len(items):
                ok = sum(1 for d in done if d.get("status") == 200)
                log(f"  [{name:8}] {n}/{len(items)}  成功 {ok}  "
                    f"用时 {time.time() - t0:.0f}s")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="只抓某一家：deepseek / zhipu / bailian")
    ap.add_argument("--limit", type=int, default=None, help="每家最多抓多少篇（先小样本验证）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="忽略已存在的文件，重新抓")
    a = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    sources = {k: v for k, v in SOURCES.items()
               if a.source is None or k == a.source}
    if not sources:
        log(f"未知 source: {a.source}（可选 {list(SOURCES)}）")
        sys.exit(1)

    all_rows: list[dict] = []
    for name, cfg in sources.items():
        all_rows += run_source(name, cfg, a.limit, a.workers, a.force)

    man = os.path.join(RAW, "manifest.jsonl")
    old: list[dict] = []
    if os.path.exists(man):
        old = [json.loads(l) for l in open(man, encoding="utf-8")
               if l.strip() and json.loads(l)["source"] not in sources]
    with open(man, "w", encoding="utf-8") as f:
        for r in old + all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print("=" * 62)
    print(f"{'来源':<10}{'总数':>6}{'成功':>7}{'失败':>6}{'平均字节':>10}")
    print("-" * 62)
    for name in sources:
        rs = [r for r in all_rows if r["source"] == name]
        ok = [r for r in rs if r.get("status") == 200]
        bad = [r for r in rs if r.get("status") != 200]
        avg = sum(r["bytes"] for r in ok) // len(ok) if ok else 0
        print(f"{name:<10}{len(rs):>6}{len(ok):>7}{len(bad):>6}{avg:>10}")
    total_ok = sum(1 for r in all_rows if r.get("status") == 200)
    print("-" * 62)
    print(f"{'合计':<10}{len(all_rows):>6}{total_ok:>7}"
          f"{len(all_rows) - total_ok:>6}")
    print()
    print(f"manifest → {os.path.relpath(man, ROOT).replace(chr(92), '/')}")

    bad = [r for r in all_rows if r.get("status") != 200]
    if bad:
        print(f"\n失败 {len(bad)} 条（已如实记录，前 5 条原因）：")
        for r in bad[:5]:
            print(f"  {r['url']}\n    {r.get('error')}")

    print()
    if total_ok >= 200:
        print(f"判定：{total_ok} 篇，达标（≥200），可以进解析。")
    elif total_ok >= 100:
        print(f"判定：{total_ok} 篇，偏薄（100–200），建议加第二层语料。")
    else:
        print(f"判定：{total_ok} 篇，不达标（<100），先修抓取别硬凑。")


if __name__ == "__main__":
    main()
