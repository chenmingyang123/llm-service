"""价格与盈亏平衡的测试。全 mock，不花钱。"""
from __future__ import annotations

import pytest

from app import pricing


def test_price_table_covers_all_models():
    rows = pricing.price_table()
    assert {r["model"] for r in rows} == set(pricing.MODEL_PRICES)


def test_output_costs_more_than_input():
    """降本第一杠杆是让模型少说话，不是把 prompt 写短。这条必须成立。"""
    for model in ("deepseek-flash", "deepseek-v4-pro", "qwen-plus"):
        i = pricing.unit_cny(model, "in_miss", peak=False)
        o = pricing.unit_cny(model, "out", peak=False)
        assert o > i, f"{model} 输出竟然不比输入贵"
    assert pricing.unit_cny("deepseek-flash", "out", peak=False) == pytest.approx(
        pricing.unit_cny("deepseek-flash", "in_miss", peak=False) * 4, rel=0.01)


def test_cached_input_is_orders_of_magnitude_cheaper():
    miss = pricing.unit_cny("deepseek-flash", "in_miss", peak=False)
    hit = pricing.unit_cny("deepseek-flash", "in_hit", peak=False)
    assert miss / hit > 10, "缓存折扣不到一个数量级，就不值得为它重构 prompt"


def test_peak_doubles_deepseek_price():
    idle = pricing.unit_cny("deepseek-flash", "in_miss", peak=False)
    peak = pricing.unit_cny("deepseek-flash", "in_miss", peak=True)
    assert peak == pytest.approx(idle * 2, rel=0.01)


def test_free_model_is_zero():
    assert pricing.unit_cny("glm-4.7-flash", "in_miss", peak=False) == 0.0


def test_unknown_price_returns_none_not_guess():
    """官方没公示的价格就返回 None，绝不能编一个数 —— 宁可少记，不要瞎报。"""
    assert pricing.unit_cny("qwen-plus", "in_hit") is None
    assert pricing.unit_cny("不存在的模型", "in_miss") is None


def test_breakeven_saves_money_with_long_prefix():
    b = pricing.breakeven(8000, 200, 100, peak=False)
    assert b["eligible_for_cache"] is True
    assert b["saved_cny"] > 0
    assert b["with_cache_cny"] < b["no_cache_cny"]
    assert 0 < b["saved_pct"] < 100


def test_short_prefix_is_not_cacheable():
    """前缀低于 64 token 时 DeepSeek 不缓存。算出来再好看也是零，"""
    b = pricing.breakeven(30, 200, 100, peak=False)
    assert b["eligible_for_cache"] is False
    assert b["prefix_tokens"] < pricing.MIN_CACHE_PREFIX


def test_more_calls_means_more_saving():
    a = pricing.breakeven(2000, 200, 10, peak=False)
    b = pricing.breakeven(2000, 200, 100, peak=False)
    assert b["saved_cny"] > a["saved_cny"]


def test_min_calls_scales_with_prefix():
    small = pricing.min_calls_for_saving(500, 1.0, peak=False)
    large = pricing.min_calls_for_saving(8000, 1.0, peak=False)
    assert large["calls_needed"] < small["calls_needed"], "前缀越长，达到同样节省所需次数越少"


def test_min_calls_infeasible_when_price_unknown():
    r = pricing.min_calls_for_saving(2000, 1.0, model="qwen-plus")
    assert r["feasible"] is False
