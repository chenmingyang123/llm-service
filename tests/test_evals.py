"""评测 judge 的纯逻辑测试。"""
import pytest
from unittest import mock

from app.evals.judge import (
    _mean, _parse_verdict, _split_lines, context_recall, faithfulness,
)


@pytest.mark.parametrize("text,expect", [
    ("YES", True),
    ("No", False),
    ("yes.", True),
    ("NO", False),
    ("是", True),
    ("否", False),
    ("对", True),
    ("不支持", False),
    ("随便说点别的", None),
    ("", None),
])
def test_parse_verdict(text, expect):
    assert _parse_verdict(text) is expect


def test_split_lines_strips_numbering():
    assert _split_lines("1. 苹果\n2. 香蕉\n- 橙子\n\n") == ["苹果", "香蕉", "橙子"]
    assert _split_lines("苹果\n香蕉") == ["苹果", "香蕉"]
    assert _split_lines("") == []


@pytest.mark.parametrize("flags,expect", [
    ([True, False], (0.5, 2, 0)),
    ([True, False, None], (0.5, 2, 1)),
    ([None, None], (0.0, 0, 2)),
    ([], (0.0, 0, 0)),
])
def test_mean(flags, expect):
    assert _mean(flags) == expect


def test_faithfulness_aggregation():
    responses = iter([
        {"content": "苹果是红的\n香蕉是黄的", "cost_cny": 0.001},
        {"content": "YES", "cost_cny": 0.0005},
        {"content": "NO", "cost_cny": 0.0005},
    ])
    with mock.patch("app.evals.judge.call_chat",
                    side_effect=lambda *a, **k: next(responses)):
        r = faithfulness("苹果通常是红色的，而香蕉一般是黄色的，这点在成熟之后会更明显。",
                         ["苹果是红的"], concurrency=1)

    assert r["score"] == 0.5
    assert r["n"] == 2
    assert r["n_failed"] == 0
    assert len(r["items"]) == 2
    assert r["items"][0]["supported"] is True
    assert r["items"][1]["supported"] is False


def test_faithfulness_empty_answer():
    """空答案必须**短路**，一次 LLM 都不该调。"""
    with mock.patch("app.evals.judge.call_chat") as m:
        r = faithfulness("", ["上下文"], concurrency=1)
    m.assert_not_called()
    assert r["score"] == 0.0
    assert r["n"] == 0
    assert r["items"] == []


def test_faithfulness_short_answer_keeps_sign():
    """短答案直接当一条陈述，不走 LLM 拆解 —— 保住符号。"""
    responses = iter([{"content": "YES", "cost_cny": 0.0005}])
    with mock.patch("app.evals.judge.call_chat",
                    side_effect=lambda *a, **k: next(responses)) as m:
        r = faithfulness("-9999.0 [1][2]", ["该 token 的对数概率为 -9999.0"], concurrency=1)
    assert r["score"] == 1.0
    assert r["items"][0]["statement"] == "-9999.0", "引用编号应被剥掉，负号必须保留"
    assert m.call_count == 1, "短答案不该再调一次拆解"


def test_faithfulness_long_answer_still_splits():
    """长答案仍然走 LLM 拆解（短路只针对短答案，不能把拆解整个废掉）。"""
    responses = iter([
        {"content": "苹果是红的\n香蕉是黄的", "cost_cny": 0.001},
        {"content": "YES", "cost_cny": 0.0005},
        {"content": "YES", "cost_cny": 0.0005},
    ])
    with mock.patch("app.evals.judge.call_chat",
                    side_effect=lambda *a, **k: next(responses)) as m:
        r = faithfulness("苹果是红色的；香蕉是黄色的；橙子是橙色的。", ["苹果红，香蕉黄"], concurrency=1)
    assert m.call_count == 3, "长答案应是 1 次拆解 + 2 次判定"
    assert r["n"] == 2


def test_context_recall_aggregation():
    responses = iter([
        {"content": "支持 64K 上下文\n支持 function calling", "cost_cny": 0.001},
        {"content": "YES", "cost_cny": 0.0005},
        {"content": "NO", "cost_cny": 0.0005},
    ])
    with mock.patch("app.evals.judge.call_chat",
                    side_effect=lambda *a, **k: next(responses)):
        r = context_recall("DeepSeek 支持 64K 上下文且支持 function calling",
                           ["DeepSeek 支持 64K"], concurrency=1)
    assert r["score"] == 0.5
    assert len(r["items"]) == 2
    assert r["items"][0]["covered"] is True
    assert r["items"][1]["covered"] is False
