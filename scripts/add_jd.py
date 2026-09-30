"""从剪贴板存一份 JD —— 你在 BOSS 上复制，一条命令存成文件，不用切回来粘贴。"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "jd")


def read_clipboard() -> str:
    """从剪贴板读文本。Windows 走 PowerShell 的 Get-Clipboard，不需要装任何包。"""
    if sys.platform != "win32":
        raise RuntimeError("这个脚本目前只支持 Windows（其它平台直接在对话里粘贴给我就行）")
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
        capture_output=True,
    )
    if out.returncode != 0:
        raise RuntimeError("读取剪贴板失败，直接在对话里粘贴给我吧")
    return out.stdout.decode("utf-8", "replace").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="", help="可选：给这份 JD 起个名字，方便以后回看")
    a = ap.parse_args()

    text = read_clipboard()
    if len(text) < 20:
        print(f"剪贴板里只有 {len(text)} 个字符，是不是没复制上？")
        return

    os.makedirs(JD_DIR, exist_ok=True)
    existing = [f for f in os.listdir(JD_DIR) if f.endswith(".txt")]
    n = len(existing) + 1
    name = f"{n:02d}_{a.tag}.txt" if a.tag else f"{n:02d}.txt"
    path = os.path.join(JD_DIR, name)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"已存 {name}　{len(text)} 字符")
    print(f"当前共 {n} 份。攒够 20 份后跑：")
    print("  python scripts/jd_keywords.py --dir data/jd --out docs/JD关键词对照表.md")


if __name__ == "__main__":
    main()
