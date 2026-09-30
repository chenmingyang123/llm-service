"""拒答（no-answer）判定。"""
from __future__ import annotations

import re

DEFAULT_MIN_COSINE = 0.40
DEFAULT_MIN_BM25 = 8.0

NO_ANSWER_MARK = "NO_ANSWER"


def gate(hits: list[dict], min_cosine: float = DEFAULT_MIN_COSINE,
         min_bm25: float = DEFAULT_MIN_BM25) -> tuple[bool, str]:
    """闸门 ①：检索结果够不够格拿去问模型。"""
    if not hits:
        return False, "检索没找到任何内容"

    top = hits[0]
    cos = top.get("vector_score")
    kw = top.get("bm25_score")

    if cos is not None:
        if cos < min_cosine:
            return False, f"最相关的一条向量分 {cos:.3f}，低于阈值 {min_cosine}"
        return True, ""

    if kw is not None:
        if kw < min_bm25:
            return False, f"最相关的一条关键词分 {kw:.2f}，低于阈值 {min_bm25}"
        return True, ""
    return True, ""


def parse_answer(text: str) -> tuple[bool, str, str]:
    """解析模型输出，拆出 (是否拒答, 正文, 缺什么的说明)。"""
    if not text:
        return True, "", "模型没有输出"
    lines = [l.rstrip() for l in text.strip().split("\n")]
    if lines and lines[0].strip().strip("*·` ") == NO_ANSWER_MARK:
        reason = "\n".join(lines[1:]).strip()
        return True, "", reason or "资料中没有相关内容"
    return False, text.strip(), ""


def extract_cited(text: str, n_docs: int) -> tuple[list[int], list[int]]:
    """抠出答案里标的所有 [n]，分成「有效」和「对不上」两组。"""
    nums = [int(m) for m in re.findall(r"\[(\d{1,2})\]", text)]
    seen, used, bad = set(), [], []
    for n in nums:
        if n in seen:
            continue
        seen.add(n)
        (used if 1 <= n <= n_docs else bad).append(n)
    return used, bad
