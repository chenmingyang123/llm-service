"""压测脚本 —— W1 第 4 天晚间那 1 小时要交的东西。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime

import httpx

DEFAULT_URL = "http://127.0.0.1:8000"


async def one(client: httpx.AsyncClient, target: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        t0 = time.time()
        try:
            if target == "echo":
                r = await client.get("/health", timeout=30.0)
            else:
                r = await client.post(
                    "/agent", timeout=120.0,
                    json={"question": "杭州现在天气怎么样？", "max_turns": 3},
                )
            ok = r.status_code == 200
            return {"ok": ok, "status": r.status_code, "latency_s": round(time.time() - t0, 3)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "status": 0, "latency_s": round(time.time() - t0, 3),
                    "error": f"{type(e).__name__}"}


async def run_level(url: str, target: str, concurrency: int, n: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, limits=limits) as client:
        t0 = time.time()
        results = await asyncio.gather(*[one(client, target, sem) for _ in range(n)])
        wall = round(time.time() - t0, 3)

    lat = sorted(r["latency_s"] for r in results)
    ok = [r for r in results if r["ok"]]

    def pct(p: float) -> float:
        if not lat:
            return 0.0
        k = min(int(round(p / 100 * (len(lat) - 1))), len(lat) - 1)
        return round(lat[k], 3)

    return {
        "target": target,
        "concurrency": concurrency,
        "requests": n,
        "succeeded": len(ok),
        "failed": n - len(ok),
        "wall_s": wall,
        "rps": round(n / wall, 2) if wall > 0 else 0.0,
        "min_s": round(lat[0], 3) if lat else 0,
        "p50_s": pct(50),
        "p95_s": pct(95),
        "p99_s": pct(99),
        "max_s": round(lat[-1], 3) if lat else 0,
        "mean_s": round(statistics.fmean(lat), 3) if lat else 0,
        "errors": sorted({r.get("error", f"HTTP {r['status']}") for r in results if not r["ok"]}),
    }


def to_markdown(rows: list[dict], url: str) -> str:
    lines = [
        f"# 压测记录 · {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"- 目标服务：`{url}`",
        f"- 客户端：`httpx.AsyncClient`，信号量限流（不是无限并发）",
        "",
        "| target | 并发 | 请求数 | 成功 | 失败 | 墙钟(s) | RPS | p50(s) | p95(s) | p99(s) | max(s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['concurrency']} | {r['requests']} | {r['succeeded']} | "
            f"{r['failed']} | {r['wall_s']} | {r['rps']} | {r['p50_s']} | {r['p95_s']} | "
            f"{r['p99_s']} | {r['max_s']} |"
        )
    lines.append("")
    for r in rows:
        if r["errors"]:
            lines.append(f"- `{r['target']}` 并发 {r['concurrency']} 的错误：{r['errors']}")
    lines.append("")
    lines.append("## 怎么读这张表")
    lines.append("")
    lines.append("- **echo 是框架基线**（只打 /health，不碰 LLM）。agent 的 p95 减去 echo 的 p95，"
                 "才是模型那一跳的真实耗时。")
    lines.append("- **并发翻倍而 RPS 没有接近翻倍**，说明瓶颈已经出现，继续加并发只会堆队列、拉高 p95。")
    lines.append("- **失败数变多**通常不是对方挂了，是你自己把限流打爆了 —— 这时候该减并发，不是加。")
    lines.append("")
    return "\n".join(lines)


async def main_async(args) -> list[dict]:
    rows = []
    for c in args.concurrency:
        row = await run_level(args.url, args.target, c, args.n)
        rows.append(row)
        print(f"[{args.target}] 并发 {c:>3} → p50={row['p50_s']}s p95={row['p95_s']}s "
              f"RPS={row['rps']} 失败={row['failed']}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--target", default="echo", choices=["echo", "agent"])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[10, 50])
    ap.add_argument("--n", type=int, default=50, help="每档并发下的总请求数")
    ap.add_argument("--out", default=None, help="写入 Markdown 文件")
    a = ap.parse_args()

    if a.target == "agent":
        print(f"提示：将发起 {len(a.concurrency) * a.n} 次真实 LLM 调用，注意成本与限流。")

    rows = asyncio.run(main_async(a))
    md = to_markdown(rows, a.url)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"已写入 {a.out}")
    else:
        print()
        print(md)

    if "--json" in sys.argv:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
