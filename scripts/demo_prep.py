"""demo 录屏的一键准备脚本（10/8 执行）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, ".")

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
TRACE_DIR = os.path.join("data", "traces")

SEGMENTS = [
    {
        "name": "① Agent 多步任务（75s）",
        "claim": "它真的能自己决定走几步，不是写死的流程",
        "cmd": ('curl -N %s/agent/stream -H "Content-Type: application/json" '
                '-d \'{"question":"查一下北京天气，再看近7天订单统计",'
                '"max_turns":4,"trace":true}\'' % BASE),
        "say": "开场 10s 说清场景别介绍背景；中间看它走了几步；结尾报步数/耗时/成本。",
    },
    {
        "name": "② 出错与兜底（90s）★",
        "claim": "出错时不崩溃、不编造，写操作会停下来等人",
        "cmd": ('curl -N %s/agent/stream -H "Content-Type: application/json" '
                '-d \'{"question":"帮我建一张工单：登录失败需要排查","trace":true}\''
                % BASE),
        "say": ("先明说「这段是故意让它出错的」；看它换参数而不是死循环；"
                "走到写操作时**停在确认点**，然后**拒绝**它，看它优雅退出。"),
        "then": ('# 拿到流里的 confirm_id 后，拒绝一次（这是本片的重点）：\n'
                 'curl -s %s/agent/confirm -H "Content-Type: application/json" '
                 '-d \'{"confirm_id":"<换成真实的>","approve":false}\'' % BASE),
    },
    {
        "name": "③ MCP 被装（80s）★",
        "claim": "别人能直接装上用，不是只有我机器上能跑",
        "cmd": "python scripts/mcp_smoke.py",
        "say": "录 GitHub README 的一行配置 → 在真客户端里跑通 → 强调只读、无副作用。",
    },
]


def check() -> bool:
    """环境自检：把「录到一半才发现」的问题提前暴露。"""
    ok = True

    env = ".env"
    if not os.path.exists(env):
        print("✗ 缺 .env（复制 .env.example 后至少填一个 API key）")
        ok = False
    else:
        txt = open(env, encoding="utf-8", errors="replace").read()
        keys = [k for k in ("DEEPSEEK_API_KEY", "ZHIPU_API_KEY", "BAILIAN_API_KEY")
                if k in txt]
        if not keys:
            print("✗ .env 里没看到任何 API key")
            ok = False
        else:
            print(f"✓ .env 已配置：{', '.join(keys)}")

    chunks = os.path.join("data", "embed", "chunks.jsonl")
    if os.path.exists(chunks):
        n = sum(1 for _ in open(chunks, encoding="utf-8", errors="replace"))
        print(f"✓ 语料已就位：{chunks}（{n} 块）")
    else:
        print(f"⚠ 没找到 {chunks} —— RAG 相关演示会空转，Agent 工具演示不受影响")

    if os.path.isdir(TRACE_DIR):
        old = [f for f in os.listdir(TRACE_DIR) if f.endswith(".jsonl")]
        if old:
            print(f"⚠ 有 {len(old)} 条旧 trace，建议先 --clean（否则演示画面里会混进上次的）")
    else:
        os.makedirs(TRACE_DIR, exist_ok=True)
        print(f"✓ 已建 {TRACE_DIR}")

    import socket
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            print(f"⚠ 端口 {PORT} 已被占用 —— 要么先停掉，要么 --serve 换个端口")
        else:
            print(f"✓ 端口 {PORT} 空闲")

    return ok


def clean() -> None:
    """清掉旧 trace 与它渲染出的 html，保证演示从干净状态开始。"""
    if not os.path.isdir(TRACE_DIR):
        print("没有 trace 目录，无需清理")
        return
    n = 0
    for f in os.listdir(TRACE_DIR):
        if f.endswith((".jsonl", ".html")):
            try:
                os.remove(os.path.join(TRACE_DIR, f))
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"  · 跳过 {f}：{type(e).__name__}: {e}")
    print(f"✓ 清理 {n} 个文件（{TRACE_DIR}）")


def serve() -> None:
    import uvicorn
    from app.main import app
    print(f"起服务 {BASE} —— 录完 Ctrl+C 停")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def show() -> None:
    print("\n=== 三条片段 · 照着跑 ===\n")
    for seg in SEGMENTS:
        print(f"{seg['name']}  主张：{seg['claim']}")
        print(f"  {seg['cmd']}")
        if seg.get("then"):
            for line in seg["then"].split("\n"):
                print(f"  {line}")
        print(f"  旁白要点：{seg['say']}\n")
    print("收尾：python scripts/trace_view.py --open   # 打开刚那条 trace 的瀑布图")
    print("\n纪律：单条硬上限 2 分钟；出错不要重录整条，只补那一段。")


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if "--check" in args:
        sys.exit(0 if check() else 1)
    elif "--clean" in args:
        clean()
    elif "--serve" in args:
        serve()
    else:
        show()
