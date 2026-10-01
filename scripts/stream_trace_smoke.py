"""块 E · /agent/stream 接 trace 的真实冒烟 —— 起真 uvicorn，走真 socket（不花真钱）。"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

sys.path.insert(0, ".")

import requests  # noqa: E402
import uvicorn  # noqa: E402
from requests.structures import CaseInsensitiveDict  # noqa: E402

from app import agent  # noqa: E402
from app.main import CONFIRMATIONS, app  # noqa: E402
from app.observe import TRACE_DIR, read_trace  # noqa: E402

PORT = 8098
BASE = f"http://127.0.0.1:{PORT}"


def _watchdog(seconds: int) -> None:
    def _boom():
        print(f"\n[watchdog] 超过 {seconds}s 未完成，强制退出")
        os._exit(2)
    t = threading.Timer(seconds, _boom)
    t.daemon = True
    t.start()


def tc(name: str, args: dict, cid: str = "call-1") -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)}}


def fake_result(content: str = "", tool_calls=None) -> dict:
    return {"content": content, "tool_calls": tool_calls or [],
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "reasoning_content": "", "usage": {"in": 20, "out": 30, "cached": 0},
            "latency_s": 0.5, "provider": "deepseek", "model": "deepseek-flash",
            "cost_cny": 0.001}


def _etype(line: str) -> str:
    import re
    m = re.search(r'"type"\s*:\s*"([a-z_]+)"', line)
    return m.group(1) if m else ""


def _cid(line: str) -> str:
    import re
    m = re.search(r'"confirm_id"\s*:\s*"([0-9a-f]+)"', line)
    return m.group(1) if m else ""


def tid_of(line: str) -> str:
    import re
    m = re.search(r'"trace_id"\s*:\s*"([0-9a-f]+)"', line)
    return m.group(1) if m else ""


def _start_server() -> None:
    cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()
    for _ in range(100):
        try:
            if requests.get(f"{BASE}/health", timeout=1).status_code == 200:
                return
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError("server 起不来")


def main(view: bool = True) -> int:
    _watchdog(60)
    CONFIRMATIONS.clear()

    script = [
        fake_result(tool_calls=[tc("create_ticket", {
            "title": "登录失败", "body": "用户无法登录系统，需要排查",
            "request_id": "req-trace-smoke-1"})]),
        fake_result(content="已按你的要求建单完毕。"),
    ]
    it = iter(script)
    agent.call_chat = lambda *a, **k: next(it)  # type: ignore[assignment]

    print("=== 块 E · /agent/stream 接 trace 的真机冒烟 ===\n")
    _start_server()
    print(f"① server 就绪 {BASE}")

    collected: list[str] = []
    headers: CaseInsensitiveDict = CaseInsensitiveDict()
    done = threading.Event()

    def run_stream() -> None:
        try:
            with requests.post(f"{BASE}/agent/stream",
                               json={"question": "帮我建单：登录失败，需要排查",
                                     "trace": True},
                               stream=True, timeout=30) as r:
                headers.update(r.headers)
                for raw in r.iter_lines(decode_unicode=True):
                    if raw:
                        collected.append(raw)
        except Exception as e:  # noqa: BLE001
            collected.append(f"[stream error] {type(e).__name__}: {e}")
        finally:
            done.set()

    th = threading.Thread(target=run_stream, daemon=True)
    th.start()

    tid_header = ""
    for _ in range(100):
        if headers:
            tid_header = headers.get("X-Trace-Id", "")
            break
        time.sleep(0.05)
    print(f"② 通道一 · 响应头 X-Trace-Id = {tid_header or '(没拿到)'}")
    if not tid_header:
        print(f"   诊断：实际收到的头 = {sorted(headers.keys())}")

    confirm_line = ""
    for _ in range(200):
        time.sleep(0.05)
        hit = [l for l in list(collected) if _etype(l) == "need_confirm"]
        if hit:
            confirm_line = hit[0]
            break
    if not confirm_line:
        print("✗ 没等到 need_confirm 事件")
        print("  收到的流内容：", collected[:8])
        return 1
    cid = _cid(confirm_line)
    print(f"③ 流里收到 need_confirm，confirm_id = {cid}")

    resp = requests.post(f"{BASE}/agent/confirm",
                         json={"confirm_id": cid, "approve": True}, timeout=10)
    print(f"④ POST /agent/confirm → {resp.status_code}")

    done.wait(timeout=15)
    th.join(timeout=5)

    kinds = [_etype(l) for l in collected if _etype(l)]
    print(f"⑤ 流事件序列：{kinds}")

    tid_stream = ""
    for l in collected:
        if _etype(l) == "trace_started":
            tid_stream = tid_of(l)
            break
    print(f"⑥ 通道二 · 流内 trace_started → trace_id = {tid_stream or '(没拿到)'}")

    ok = True
    if not tid_header:
        print("✗ 响应头缺 X-Trace-Id"); ok = False
    if not tid_stream:
        print("✗ 流里缺 trace_started 事件"); ok = False
    if tid_header and tid_stream and tid_header != tid_stream:
        print(f"✗ 两个通道的 trace_id 不一致：{tid_header} vs {tid_stream}"); ok = False
    if "confirm_result" not in kinds:
        print("✗ 缺 confirm_result"); ok = False

    tid = tid_header or tid_stream
    path = os.path.join(TRACE_DIR, f"{tid}.jsonl")
    if not os.path.exists(path):
        print(f"✗ trace 没落盘：{path}"); return 1

    recs = read_trace(path)
    conf_spans = [x for x in recs if x["kind"] == "confirm"]
    print(f"⑦ 落盘 {path}")
    print(f"   span 数 {len(recs)}，其中 confirm span {len(conf_spans)}")
    root = [x for x in recs if x["kind"] == "root"][0]
    print(f"   根 span：dur_ms={root.get('dur_ms')} "
          f"cost_cny={root['meta'].get('cost_cny')} "
          f"status={root.get('status')}")

    if not conf_spans:
        print("✗ confirm 没有单独成 span（会把「人没点」误判成「系统慢」）")
        ok = False
    if not root.get("dur_ms"):
        print("✗ 根 span 未闭合（悬空）"); ok = False

    if view and ok:
        from scripts.trace_view import render  # noqa: E402
        out = os.path.join(TRACE_DIR, f"{tid}.html")
        render(recs, out)
        print(f"⑧ 瀑布图：{out}")

    print("\n" + ("✓ 全通过" if ok else "✗ 有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(view="--no-view" not in sys.argv))
