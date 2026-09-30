"""不依赖 cwd 的入口脚本 —— Host 配置里推荐用这个。"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp_server.server import main  # noqa: E402

if __name__ == "__main__":
    main()
