"""重试与降级的测试。全 mock，不花钱。"""
from __future__ import annotations

import pytest

from app import resilience
from app.config import Settings
from app.llm import LLMError, PROVIDER_ENDPOINT


def err(msg: str, retryable: bool = True, code: int | None = None) -> LLMError:
    return LLMError(msg, retryable=retryable, code=code)


def test_retry_then_success():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise err("HTTP 429 限流", retryable=True, code=429)
        return {"ok": True}

    out = resilience.call_with_retry(fn, attempts=3, base_delay=0.001)
    assert out == {"ok": True}
    assert len(calls) == 3


def test_no_retry_on_non_retryable():
    """401 重试一万次结果一样。必须一次就抛，不许浪费时间。"""
    calls = []

    def fn():
        calls.append(1)
        raise err("密钥无效", retryable=False, code=401)

    with pytest.raises(LLMError):
        resilience.call_with_retry(fn, attempts=3, base_delay=0.001)
    assert len(calls) == 1


def test_retry_exhausted():
    calls = []

    def fn():
        calls.append(1)
        raise err("HTTP 503", retryable=True, code=503)

    with pytest.raises(LLMError):
        resilience.call_with_retry(fn, attempts=3, base_delay=0.001)
    assert len(calls) == 3


def test_fallback_switches_provider(monkeypatch):
    """主用挂了，备胎顶上，并且要如实标记 degraded。"""
    used = []

    def fake(s, messages, **kw):
        used.append(kw.get("provider"))
        if kw.get("provider") == "deepseek":
            raise err("HTTP 502", retryable=False, code=502)
        return {"content": "备胎回答", "provider": kw.get("provider"), "model": "x"}

    monkeypatch.setattr(resilience, "call_chat", fake)
    monkeypatch.setattr(resilience, "_has_key", lambda s, p: True)

    out = resilience.call_with_fallback(Settings(), [{"role": "user", "content": "hi"}],
                                        providers=["deepseek", "zhipu"])
    assert out["degraded"] is True
    assert out["provider"] == "zhipu"
    assert used == ["deepseek", "zhipu"]


def test_fallback_model_only_for_first(monkeypatch):
    """拿 DeepSeek 的模型名去打智谱必然 404，所以备胎一律用各自默认值。"""
    seen = []

    def fake(s, messages, **kw):
        seen.append((kw.get("provider"), kw.get("model")))
        raise err("HTTP 500", retryable=False, code=500)

    monkeypatch.setattr(resilience, "call_chat", fake)
    monkeypatch.setattr(resilience, "_has_key", lambda s, p: True)

    with pytest.raises(LLMError):
        resilience.call_with_fallback(Settings(), [{"role": "user", "content": "hi"}],
                                      providers=["deepseek", "zhipu"], model="deepseek-flash")
    assert seen[0] == ("deepseek", "deepseek-flash")
    assert seen[1] == ("zhipu", None)


def test_fallback_skips_provider_without_key(monkeypatch):
    monkeypatch.setattr(resilience, "_has_key", lambda s, p: p == "zhipu")

    def fake(s, messages, **kw):
        return {"content": "ok", "provider": kw.get("provider"), "model": "m"}

    monkeypatch.setattr(resilience, "call_chat", fake)
    out = resilience.call_with_fallback(Settings(), [{"role": "user", "content": "hi"}],
                                        providers=["deepseek", "zhipu"])
    assert any(a["status"] == "skipped" for a in out["attempts"])
    assert out["provider"] == "zhipu"


def test_fallback_all_fail_raises(monkeypatch):
    monkeypatch.setattr(resilience, "_has_key", lambda s, p: True)
    monkeypatch.setattr(resilience, "call_chat",
                        lambda *a, **k: (_ for _ in ()).throw(err("HTTP 500", retryable=False)))
    with pytest.raises(LLMError, match="全部 provider"):
        resilience.call_with_fallback(Settings(), [{"role": "user", "content": "hi"}],
                                      providers=["deepseek", "zhipu"])


def test_chain_matches_endpoints():
    """降级链里的每个名字都必须在 PROVIDER_ENDPOINT 里有定义，否则是配置错误。"""
    assert set(resilience.FALLBACK_CHAIN) <= set(PROVIDER_ENDPOINT)
