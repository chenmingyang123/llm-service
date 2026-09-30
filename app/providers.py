"""三家 provider 的元信息与就绪状态。"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .config import Settings

PROVIDERS = [
    {
        "key": "deepseek",
        "name": "DeepSeek",
        "role": "primary",
        "base_url": "DEEPSEEK_BASE_URL",
        "api_key": "DEEPSEEK_API_KEY",
        "models": ["deepseek-flash", "deepseek-v4-pro"],
    },
    {
        "key": "zhipu",
        "name": "智谱 GLM",
        "role": "fallback",
        "base_url": "ZHIPU_BASE_URL",
        "api_key": "ZHIPU_API_KEY",
        "models": ["glm-4.7-flash", "glm-4.5-flash", "glm-4.5-air"],
    },
    {
        "key": "dashscope",
        "name": "阿里百炼 / 通义",
        "role": "fallback",
        "base_url": "DASHSCOPE_BASE_URL",
        "api_key": "DASHSCOPE_API_KEY",
        "models": ["qwen-plus", "qwen-turbo", "qwen3-max"],
    },
]


def _ping(base_url: str, api_key: str, timeout: float = 8.0) -> tuple[bool, float, str]:
    """真实探活：只打 /models，不生成任何 token，所以不花钱。"""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", "Bearer " + api_key)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return True, round(time.time() - t0, 3), ""
    except urllib.error.HTTPError as e:
        return False, round(time.time() - t0, 3), f"HTTP {e.code}"
    except Exception as e:
        return False, round(time.time() - t0, 3), type(e).__name__


def provider_status(p: dict, s: Settings, do_ping: bool = False) -> dict:
    api_key = getattr(s, p["api_key"], "")
    base_url = getattr(s, p["base_url"], "")

    if not api_key:
        return {
            "key": p["key"], "name": p["name"], "role": p["role"],
            "configured": False, "status": "unconfigured",
            "base_url": base_url, "models": p["models"], "detail": "未配置 API key",
        }

    if not do_ping:
        return {
            "key": p["key"], "name": p["name"], "role": p["role"],
            "configured": True, "status": "configured",
            "base_url": base_url, "models": p["models"], "detail": "key 已配置（未探活）",
        }

    ok, latency, err = _ping(base_url, api_key)
    return {
        "key": p["key"], "name": p["name"], "role": p["role"],
        "configured": True,
        "status": "ready" if ok else "unreachable",
        "base_url": base_url, "models": p["models"],
        "latency_s": latency,
        "detail": "" if ok else err,
    }


def all_provider_status(s: Settings, do_ping: bool = False) -> list[dict]:
    return [provider_status(p, s, do_ping) for p in PROVIDERS]
