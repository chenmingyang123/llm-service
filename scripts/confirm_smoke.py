"""块 C · 确认点接流式的真实冒烟 —— 起真 uvicorn，走真 socket（不花真钱）。"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time

sys.path.insert(0, ".")

import requests  # noqa: E402
import uvicorn  # noqa: E402

from app import agent, tools  # noqa: E402
from app.main import CONFIRMATIONS, app  # noqa: E402

PORT = 8099
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
    m = re.search(r'"type"\s*:\s*"([a-z_]+)"', line)
    return m.group(1) if m else ""


def _cid(line: str) -> str:
    m = re.search(r'"confirm_id"\s*:\s*"([0-9a-f]+)"', line)
    return m.group(1) if m else ""


def _start_server() -> uvicorn.Server:
    cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            if requests.get(f"{BASE}/health", timeout=1).status_code == 200:
                return server
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError("server 起不来")


def main(approve: bool) -> int:
    tmp = tempfile.mkdtemp(prefix="confirm-smoke-")
    audit = os.path.join(tmp, "actions.jsonl")
    tools._ACTIONS_PATH = audit
    tools.reset_actions_for_test()
    CONFIRMATIONS.clear()

    script = [
        fake_result(tool_calls=[tc("create_ticket", {
            "title": "登录失败", "body": "用户无法登录系统，需要排查",
            "request_id": "req-smoke-1"})]),
        fake_result(content="已按你的要求处理完毕。"),
    ]
    it = iter(script)
    agent.call_chat = lambda *a, **k: next(it)  # type: ignore[assignment]

    print(f"=== 块 C 确认点冒烟（approve={approve}）===\n")
    _start_server()
    print(f"① server 就绪 {BASE}")

    collected: list[str] = []
    done = threading.Event()

    def run_stream() -> None:
        try:
            with requests.post(f"{BASE}/agent/stream",
                               json={"question": "帮我建单：登录失败，需要排查"},
                               stream=True, timeout=30) as r:
                for raw in r.iter_lines(decode_unicode=True):
                    if raw:
                        collected.append(raw)
        except Exception as e:  # noqa: BLE001
            collected.append(f"[stream error] {type(e).__name__}: {e}")
        finally:
            done.set()

    th = threading.Thread(target=run_stream, daemon=True)
    th.start()

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
    print(f"② 流里收到 need_confirm，confirm_id = {cid}")
    print(f"   事件内容：{confirm_line[:170]}")

    resp = requests.post(f"{BASE}/agent/confirm",
                         json={"confirm_id": cid, "approve": approve}, timeout=10)
    print(f"③ POST /agent/confirm → {resp.status_code} {resp.text[:120]}")

    done.wait(timeout=10)
    th.join(timeout=10)

    kinds = [_etype(l) for l in collected if _etype(l)]
    print(f"④ 流事件序列：{kinds}")
    wrote = os.path.exists(audit) and open(audit, encoding="utf-8").read().strip() != ""
    print(f"⑤ 审计文件是否写入：{'是' if wrote else '否'}")

    ok = True
    if "need_confirm" not in kinds or "confirm_result" not in kinds:
        print("✗ 缺 need_confirm / confirm_result"); ok = False
    if "summary" not in kinds:
        print("✗ 流没正常收尾（缺 summary）"); ok = False
    if approve and not wrote:
        print("✗ 批准后应落盘却没有"); ok = False
    if (not approve) and wrote:
        print("✗ 拒绝后不应落盘却写了"); ok = False
    print("\n=== 冒烟", "通过 ===" if ok else "失败 ===")
    return 0 if ok else 1


if __name__ == "__main__":
    _watchdog(60)
    sys.exit(main(approve=("--reject" not in sys.argv)))
