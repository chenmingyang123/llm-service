"""golden set 体检脚本的告警逻辑测试。"""
from __future__ import annotations

import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import golden_stats as gs  # noqa: E402


def test_hard_ratio_low_alarms():
    cnt = Counter({"term": 25, "semantic": 15, "multi": 3, "boundary": 5, "conflict": 2})
    assert gs.alarm_hard_ratio(cnt, 50) is not None


def test_hard_ratio_ok_no_alarm():
    cnt = Counter({"term": 50, "multi": 15, "boundary": 15, "conflict": 10})
    assert gs.alarm_hard_ratio(cnt, 90) is None


def test_type_exactly_half_no_alarm():
    cnt = Counter({"term": 25, "semantic": 25})
    assert gs.alarm_type_dominant(cnt, 50) == []


def test_type_over_half_alarms():
    cnt = Counter({"term": 60, "semantic": 40})
    assert len(gs.alarm_type_dominant(cnt, 100)) == 1


def test_source_low_alarms():
    src = Counter({"deepseek": 12, "zhipu": 20, "bailian": 18})
    assert any("deepseek" in a for a in gs.alarm_source_skew(src, 50))


def test_source_balanced_no_alarm():
    src = Counter({"deepseek": 40, "zhipu": 40, "bailian": 40})
    assert gs.alarm_source_skew(src, 120) == []


def test_source_over_45_alarms():
    src = Counter({"deepseek": 10, "zhipu": 60, "bailian": 30})
    assert any("zhipu" in a for a in gs.alarm_source_skew(src, 100))


def test_missing_cells_detected():
    golden = [{"expect_source": "deepseek", "type": "term"},
              {"expect_source": "zhipu", "type": "semantic"}]
    out = gs.alarm_missing_cells(golden)
    assert len(out) == 3
    assert any("bailian" in a for a in out)
