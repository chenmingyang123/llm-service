"""配置：从 .env 与环境变量读取，集中一处，别在业务代码里散落 os.getenv。"""
from __future__ import annotations

import os
from functools import lru_cache

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path=None):
    """手写一个极简 .env 解析器，避免依赖 python-dotenv。"""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return True
load_env()

class Settings:
    """极简配置对象。用 pydantic-settings 更省事，但多一个依赖；这里够用。"""

    SERVICE_NAME: str = "llm-service"
    VERSION: str = "0.1.0"
    ENV: str = os.getenv("ENV", "dev")

    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    DEEPSEEK_THINKING: str = os.getenv("DEEPSEEK_THINKING", "disabled").strip()

    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "").strip()
    ZHIPU_BASE_URL: str = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip()

    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "").strip()
    DASHSCOPE_BASE_URL: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip()

    COST_LIMIT_CNY: float = float(os.getenv("COST_LIMIT_CNY", "20"))
    COST_LEDGER: str = os.getenv("COST_LEDGER", ".cost_ledger.jsonl")

    @property
    def is_dev(self) -> bool:
        return self.ENV in ("dev", "local", "test")


@lru_cache
def get_settings() -> Settings:
    return Settings()
