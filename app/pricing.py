"""价格与盈亏平衡 —— W1 第 5 天的核心资产。"""
from __future__ import annotations

from .cost import DEFAULT_USD_CNY, is_peak_hour

MODEL_PRICES: dict[str, dict] = {
    "deepseek-flash": {
        "provider": "deepseek", "currency": "USD", "peak_multiplier": 2.0,
        "in_miss": 0.15, "in_hit": 0.003, "out": 0.60,
        "note": "高峰档（工作日 09-12、14-18 点）单价翻倍",
    },
    "deepseek-v4-pro": {
        "provider": "deepseek", "currency": "USD", "peak_multiplier": 2.0,
        "in_miss": 0.66, "in_hit": 0.022, "out": 1.98,
        "note": "旗舰档，输出单价是 flash 的 3.3 倍",
    },
    "glm-4.7-flash": {
        "provider": "zhipu", "currency": "CNY", "peak_multiplier": 1.0,
        "in_miss": 0.0, "in_hit": 0.0, "out": 0.0, "free": True,
        "note": "官方完全免费，但并发为 1，只能串行",
    },
    "qwen-plus": {
        "provider": "dashscope", "currency": "CNY", "peak_multiplier": 1.0,
        "in_miss": 0.80, "in_hit": None, "out": 2.00,
        "note": "人民币直标；缓存命中价官方未公示，故不参与盈亏平衡计算",
    },
}

MIN_CACHE_PREFIX = 64


def _fx(p: dict) -> float:
    return DEFAULT_USD_CNY if p["currency"] == "USD" else 1.0


def unit_cny(model: str, kind: str, *, peak: bool | None = None) -> float | None:
    """某一项单价，折算成人民币 / 百万 token。未知模型或官方未公示则返回 None。"""
    p = MODEL_PRICES.get(model)
    if not p:
        return None
    v = p.get(kind)
    if v is None:
        return None
    if peak is None:
        peak = is_peak_hour()
    return round(v * (p["peak_multiplier"] if peak else 1.0) * _fx(p), 4)


def price_table(*, peak: bool | None = None, include_free: bool = True) -> list[dict]:
    """模型价格对比表的数据。这张表本身就是作品 —— 面试时甩出来很有说服力。"""
    rows = []
    for model, p in MODEL_PRICES.items():
        if p.get("free") and not include_free:
            continue
        rows.append({
            "model": model,
            "provider": p["provider"],
            "free": bool(p.get("free")),
            "input_cny_per_m": unit_cny(model, "in_miss", peak=peak),
            "cached_cny_per_m": unit_cny(model, "in_hit", peak=peak),
            "output_cny_per_m": unit_cny(model, "out", peak=peak),
            "out_in_ratio": (round(p["out"] / p["in_miss"], 1) if p["in_miss"] else None),
            "note": p.get("note", ""),
        })
    return rows


def breakeven(
    prefix_tokens: int,
    dynamic_tokens: int,
    calls: int,
    *,
    model: str = "deepseek-flash",
    peak: bool | None = None,
) -> dict:
    """Prompt 缓存的盈亏平衡：同样 N 次调用，缓存能省多少。"""
    p = MODEL_PRICES.get(model)
    if not p:
        raise KeyError(f"未知模型 {model}，可选 {list(MODEL_PRICES)}")
    if peak is None:
        peak = is_peak_hour()
    m = p["peak_multiplier"] if peak else 1.0
    fx = _fx(p)
    in_miss = p["in_miss"] * m * fx
    unknown_cache_price = p.get("in_hit") is None
    in_hit = (p["in_hit"] if not unknown_cache_price else p["in_miss"]) * m * fx

    total = prefix_tokens + dynamic_tokens
    no_cache = calls * total * in_miss / 1_000_000
    with_cache = (total * in_miss
                  + (calls - 1) * (dynamic_tokens * in_miss + prefix_tokens * in_hit)) / 1_000_000
    saved = no_cache - with_cache

    eligible = prefix_tokens >= MIN_CACHE_PREFIX and not unknown_cache_price
    return {
        "model": model, "peak": peak,
        "prefix_tokens": prefix_tokens, "dynamic_tokens": dynamic_tokens, "calls": calls,
        "eligible_for_cache": eligible,
        "cache_price_unknown": unknown_cache_price,
        "min_prefix_required": MIN_CACHE_PREFIX,
        "no_cache_cny": round(no_cache, 6),
        "with_cache_cny": round(with_cache, 6),
        "saved_cny": round(saved, 6),
        "saved_pct": round(saved / no_cache * 100, 2) if no_cache else 0.0,
        "unit_in_miss_cny_per_m": round(in_miss, 4),
        "unit_in_hit_cny_per_m": round(in_hit, 4),
    }


def min_calls_for_saving(
    prefix_tokens: int, target_cny: float, *, model: str = "deepseek-flash", peak: bool | None = None
) -> dict:
    """反解：想省下 target_cny，这个前缀得被复用多少次？"""
    p = MODEL_PRICES.get(model)
    if not p:
        raise KeyError(f"未知模型 {model}")
    if peak is None:
        peak = is_peak_hour()
    if p.get("in_hit") is None:
        return {"feasible": False, "model": model,
                "reason": "该模型缓存命中价官方未公示，无法估算节省额"}
    m = p["peak_multiplier"] if peak else 1.0
    fx = _fx(p)
    diff = (p["in_miss"] - p["in_hit"]) * m * fx

    per_call_saving = prefix_tokens * diff / 1_000_000
    if per_call_saving <= 0:
        return {"feasible": False, "model": model, "reason": "前缀过短或该模型无缓存折扣"}
    calls = int(target_cny / per_call_saving) + 1
    return {
        "feasible": True, "model": model, "prefix_tokens": prefix_tokens,
        "target_cny": target_cny, "saved_per_call_cny": round(per_call_saving, 8),
        "calls_needed": calls,
    }
