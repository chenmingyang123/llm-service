"""MCP Server 真实协议冒烟：起子进程，走完整 JSON-RPC 握手。"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

RUN_PY = os.path.join(ROOT, "mcp_server", "run.py")


def _watchdog(seconds: int) -> None:
    def _boom():
        print(f"\n[watchdog] 超过 {seconds}s 未完成，强制退出")
        os._exit(2)
    t = threading.Timer(seconds, _boom)
    t.daemon = True
    t.start()


def _text(result) -> str:
    return "\n".join(getattr(c, "text", "") for c in (result.content or [])
                     if getattr(c, "text", None))


async def _roundtrip(label: str, params: StdioServerParameters) -> bool:
    """跑一次完整往返，返回是否全部通过。"""
    ok = True
    print(f"--- {label} ---")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = init.server_info
            print(f"① initialize OK —— {info.name} v{info.version} "
                  f"（协议 {init.protocol_version}）")
            caps = init.capabilities
            print(f"   能力：tools={bool(getattr(caps, 'tools', None))} "
                  f"resources={bool(getattr(caps, 'resources', None))}")

            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"② tools/list → {names}")
            for t in tools.tools:
                ro = getattr(getattr(t, "annotations", None), "read_only_hint", None)
                if ro is not True:
                    print(f"   ✗ {t.name} 没声明只读"); ok = False
            if {"search_api_docs", "ask_api_docs"} - set(names):
                print("   ✗ 缺少预期工具"); ok = False

            res = await session.list_resources()
            ruris = [str(r.uri) for r in res.resources]
            print(f"③ resources/list → {ruris}")
            if not ruris:
                print("   ✗ 没有 resource"); ok = False

            call = await session.call_tool(
                "search_api_docs", {"query": "上下文硬盘缓存", "top_k": 3})
            text = _text(call)
            print(f"④ tools/call → {text.replace(chr(10), ' ')[:150]}")
            if call.is_error:
                print("   ✗ 工具返回 is_error=True"); ok = False
            if "找到" not in text and "索引不存在" not in text:
                print("   ✗ 返回内容不符合预期"); ok = False

            rr = await session.read_resource("docs://corpus/stats")
            body = "".join(getattr(c, "text", "") for c in (rr.contents or []))
            print(f"⑤ resources/read → {body.replace(chr(10), ' ')[:130]}")
            if '"server"' not in body:
                print("   ✗ resource 内容不符合预期"); ok = False
    print(f"    → {'通过' if ok else '失败'}\n")
    return ok


async def run() -> int:
    py = sys.executable
    other_cwd = tempfile.mkdtemp(prefix="mcp-smoke-cwd-")

    print("=== MCP Server 真实协议冒烟 ===\n")
    print(f"解释器：{py}")
    print(f"仓库根：{ROOT}\n")

    a = await _roundtrip(
        "入口 A · python -m mcp_server（cwd=仓库根）",
        StdioServerParameters(command=py, args=["-m", "mcp_server"], cwd=ROOT))

    b = await _roundtrip(
        f"入口 B · run.py 绝对路径（cwd={other_cwd}）",
        StdioServerParameters(command=py, args=[RUN_PY], cwd=other_cwd))

    ok = a and b
    print("=== 冒烟", "通过 ===" if ok else "失败 ===")
    return 0 if ok else 1


if __name__ == "__main__":
    _watchdog(120)
    sys.exit(asyncio.run(run()))
