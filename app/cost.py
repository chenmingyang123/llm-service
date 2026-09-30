"""成本账本。每次 LLM 调用都记一行到 .cost_ledger.jsonl，这里负责估价、写入、汇总。"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .config import Settings

PRICE_USD = {
    "deepseek-flash":  {"in_hit": (0.003, 0.006), "in_miss": (0.15, 0.30), "out": (0.60, 1.20)},
    "deepseek-v4-pro": {"in_hit": (0.022, 0.044), "in_miss": (0.66, 1.32), "out": (1.98, 3.96)},
}
FREE_MODELS = {"glm-4.7-flash", "glm-4.5-flash", "glm-4.5-air", "glm-4.6v-flash"}
DEFAULT_USD_CNY = 7.10


def is_peak_hour(now: datetime | None = None) -> bool:
    """DeepSeek 高峰档：UTC 01:00-04:00 与 06:00-10:00，周一至周五（中国法定节假日除外）。"""
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    h = now.hour
    return (1 <= h < 4) or (6 <= h < 10)


def estimate_cny(model: str, tin: int, tout: int, cached: int = 0) -> float:
    """单次调用花了多少钱。未知模型返回 0.0（宁可少记，不要瞎报）。"""
    if model in FREE_MODELS:
        return 0.0
    p = PRICE_USD.get(model)
    if not p:
        return 0.0
    idx = 1 if is_peak_hour() else 0
    miss = max(tin - cached, 0)
    usd = (cached * p["in_hit"][idx] + miss * p["in_miss"][idx] + tout * p["out"][idx]) / 1_000_000
    return round(usd * DEFAULT_USD_CNY, 6)


def ledger_path(s: Settings) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), s.COST_LEDGER)


def append_entry(s: Settings, entry: dict) -> None:
    """记一行。用 append 而不是重写，进程崩了也不丢历史。"""
    entry = dict(entry)
    entry.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    with open(ledger_path(s), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def summarize(s: Settings) -> dict:
    path = ledger_path(s)
    total, calls = 0.0, 0
    per_provider: dict[str, float] = {}
    per_model: dict[str, float] = {}

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = e.get("cost_cny") or 0.0
                total += c
                calls += 1
                per_provider[e.get("provider", "?")] = per_provider.get(e.get("provider", "?"), 0.0) + c
                per_model[e.get("model", "?")] = per_model.get(e.get("model", "?"), 0.0) + c

    return {
        "total_cny": round(total, 6),
        "calls": calls,
        "limit_cny": s.COST_LIMIT_CNY,
        "remaining_cny": round(max(s.COST_LIMIT_CNY - total, 0.0), 6),
        "over_limit": total >= s.COST_LIMIT_CNY,
        "by_provider": {k: round(v, 6) for k, v in per_provider.items()},
        "by_model": {k: round(v, 6) for k, v in per_model.items()},
    }


def cache_stats(s: Settings) -> dict:
    """从账本统计缓存命中率，以及命中实际省了多少钱。"""
    from .pricing import MODEL_PRICES

    path = ledger_path(s)
    tin = cached = 0
    by_model: dict[str, dict] = {}

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                i, c = e.get("in") or 0, e.get("cached") or 0
                tin += i
                cached += c
                m = e.get("model", "?")
                slot = by_model.setdefault(m, {"in": 0, "cached": 0})
                slot["in"] += i
                slot["cached"] += c

    saved = 0.0
    for m, slot in by_model.items():
        p = MODEL_PRICES.get(m)
        if not p or p.get("in_hit") is None:
            continue
        diff = (p["in_miss"] - p["in_hit"]) * (DEFAULT_USD_CNY if p["currency"] == "USD" else 1.0)
        saved += slot["cached"] * diff / 1_000_000

    return {
        "prompt_tokens": tin,
        "cached_tokens": cached,
        "hit_rate_pct": round(cached / tin * 100, 2) if tin else 0.0,
        "saved_cny": round(saved, 6),
        "by_model": by_model,
    }
