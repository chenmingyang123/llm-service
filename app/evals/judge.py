"""LLM-as-judge 三指标：忠实度 / 答案相关性 / 上下文召回。"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

from ..config import Settings, get_settings
from ..llm import LLMError, call_chat


EXTRACT_STATEMENTS = """把下面这段回答拆成独立的原子事实陈述。要求：
1. 每行一条，不要编号、不要项目符号
2. 只保留事实性陈述，删掉"根据资料""综上所述"这类套话
3. 一条陈述只包含一个事实；但**并列句（A、B 或 C 这种）要保持在一起**，拆开会改变原意
4. 只提取回答本身陈述的事实，**不要输出"某内容出现在引用[1]中"这类关于回答自身的元陈述**
5. **数字、符号、单位必须原样照抄**（例：-9999.0 不能写成 9999.0）——
   改一个负号，判定就会从"能推出"变成"推不出"

回答：
{answer}"""

JUDGE_SUPPORT = """判断下面这条陈述能否完全由给定上下文推出（即上下文里有没有足够的信息支持它成立）。

上下文：
{context}

陈述：
{statement}

只能回答 YES 或 NO，不要解释。"""

GEN_QUESTIONS = """根据下面这段回答，写出它可能在回答的问题。要求：
1. 每行一个问题，不要编号，**最多 3 个**
2. 只问回答里**明确提到**的信息，不要推测，不要问"为什么""依据是什么"这类回答里没有的内容
3. 忽略 [1] 这种引用编号，它不是信息点

回答：
{answer}"""

JUDGE_SAME_QUESTION = """判断下面两个问题在问的是不是同一个信息点。

问题 A：{question_a}
问题 B：{question_b}

只回答以下三个词之一，不要解释：
SAME      两个问题问的是同一件事（措辞不同也算）
PARTIAL   问的是同一个信息点，但其中一个的范围更宽或更窄（比如少了产品名、少了限定条件）
DIFFERENT 问的不是同一件事"""

EXTRACT_CLAIMS = """从下面这段标准答案中抽取关键事实断言（用于验证信息是否被覆盖）。要求：
1. 每行一条断言，不要编号
2. 只抽取有信息量的关键事实（数字、限制、机制、结论），跳过铺垫和客套

标准答案：
{ground_truth}"""

JUDGE_COVERED = """判断下面这条断言所表达的信息，是否出现在给定上下文中。

上下文：
{context}

断言：
{claim}

只能回答 YES 或 NO，不要解释。"""


def _split_lines(text: str) -> list[str]:
    """把 LLM「每行一条」的输出拆成干净的非空列表，剥掉编号/项目符号。"""
    out = []
    for line in text.strip().splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d{1,2}[.、)]|[\u2460-\u24ff])\s*", "", line).strip()
        if line:
            out.append(line)
    return out


def _parse_verdict(text: str) -> bool | None:
    """解析 YES/NO（含中文兜底）。解析不出来返回 None，由调用方按失败处理。"""
    t = text.strip().lower()
    if t.startswith(("yes", "true", "是", "对", "支持")):
        return True
    if t.startswith(("no", "false", "否", "错", "不支持")):
        return False
    if "否" in t or t.startswith("不") or "不能" in t:
        return False
    if "是" in t or "对" in t or "支持" in t:
        return True
    return None


def _parse_relevancy(text: str) -> float | None:
    """解析 SAME / PARTIAL / DIFFERENT → 1.0 / 0.5 / 0.0。"""
    t = text.strip().upper()
    if t.startswith("SAME"):
        return 1.0
    if t.startswith("PARTIAL"):
        return 0.5
    if t.startswith("DIFFERENT"):
        return 0.0
    if "DIFFERENT" in t:
        return 0.0
    if "PARTIAL" in t:
        return 0.5
    if "SAME" in t:
        return 1.0
    return None


def _run_prompts(settings: Settings, prompts: list[str], *, provider: str,
                 model: str | None, concurrency: int, max_tokens: int,
                 ) -> tuple[list[str], float]:
    """并发发一批判定调用，返回 (原始输出列表, 累计成本)。"""
    def one(prompt: str) -> tuple[str, float]:
        try:
            res = call_chat(settings, [{"role": "user", "content": prompt}],
                            provider=provider, model=model, thinking=False,
                            max_tokens=max_tokens, record_cost=True)
            return (res["content"] or ""), (res.get("cost_cny") or 0.0)
        except LLMError:
            return "", 0.0

    if concurrency <= 1 or len(prompts) <= 1:
        results = [one(p) for p in prompts]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(one, prompts))
    return [r[0] for r in results], sum(r[1] for r in results)


def _mean(flags: list[bool | None]) -> tuple[float, int, int]:
    """把 [True/False/None] 聚合成 (分数, 有效数, 失败数)。None 不进分母。"""
    valid = [f for f in flags if f is not None]
    n_failed = len(flags) - len(valid)
    score = sum(valid) / len(valid) if valid else 0.0
    return score, len(valid), n_failed


def faithfulness(answer: str, contexts: list[str], *, settings: Settings | None = None,
                 provider: str = "deepseek", model: str | None = None,
                 concurrency: int = 4) -> dict:
    """忠实度：答案里每条陈述，能否完全由上下文推出。"""
    settings = settings or get_settings()
    ctx = "\n\n".join(contexts)
    t0 = time.perf_counter()
    cost = 0.0

    stripped = re.sub(r"\[\d{1,2}\]", "", answer or "").strip()
    if not stripped:
        return {"score": 0.0, "n": 0, "n_failed": 0, "items": [],
                "cost_cny": 0.0, "latency_s": round(time.perf_counter() - t0, 3)}

    if len(stripped) <= 20:
        statements = [stripped]
    else:
        raw, c1 = _run_prompts(settings, [EXTRACT_STATEMENTS.format(answer=answer)],
                               provider=provider, model=model, concurrency=1, max_tokens=256)
        cost += c1
        statements = _split_lines(raw[0])

    if not statements:
        return {"score": 0.0, "n": 0, "n_failed": 0, "items": [],
                "cost_cny": round(cost, 6), "latency_s": round(time.perf_counter() - t0, 3)}

    prompts = [JUDGE_SUPPORT.format(statement=s, context=ctx) for s in statements]
    texts, c2 = _run_prompts(settings, prompts, provider=provider, model=model,
                             concurrency=concurrency, max_tokens=16)
    cost += c2
    flags = [_parse_verdict(t) for t in texts]
    score, n_valid, n_failed = _mean(flags)
    items = [{"statement": s, "supported": f} for s, f in zip(statements, flags)]

    return {"score": round(score, 4), "n": n_valid, "n_failed": n_failed, "items": items,
            "cost_cny": round(cost, 6), "latency_s": round(time.perf_counter() - t0, 3)}


def answer_relevancy(question: str, answer: str, *, settings: Settings | None = None,
                     provider: str = "deepseek", model: str | None = None,
                     concurrency: int = 4, max_q: int = 3) -> dict:
    """答案相关性：从答案反推的问题里，有多少与原问题同义。"""
    settings = settings or get_settings()
    t0 = time.perf_counter()
    cost = 0.0

    raw, c1 = _run_prompts(settings, [GEN_QUESTIONS.format(answer=answer)],
                           provider=provider, model=model, concurrency=1, max_tokens=256)
    cost += c1
    questions = _split_lines(raw[0])[:max_q]
    if not questions:
        return {"score": 0.0, "n": 0, "n_failed": 0, "items": [],
                "cost_cny": round(cost, 6), "latency_s": round(time.perf_counter() - t0, 3)}

    prompts = [JUDGE_SAME_QUESTION.format(question_a=question, question_b=q)
               for q in questions]
    texts, c2 = _run_prompts(settings, prompts, provider=provider, model=model,
                             concurrency=concurrency, max_tokens=32)
    cost += c2
    vals = [_parse_relevancy(t) for t in texts]
    valid = [v for v in vals if v is not None]
    n_failed = len(vals) - len(valid)
    score = sum(valid) / len(valid) if valid else 0.0
    items = [{"question": q, "relevancy": v} for q, v in zip(questions, vals)]

    return {"score": round(score, 4), "n": len(valid), "n_failed": n_failed, "items": items,
            "cost_cny": round(cost, 6), "latency_s": round(time.perf_counter() - t0, 3)}


def context_recall(ground_truth: str, contexts: list[str], *, settings: Settings | None = None,
                   provider: str = "deepseek", model: str | None = None,
                   concurrency: int = 4) -> dict:
    """上下文召回：标准答案的关键断言，被检索到的上下文覆盖了多少。"""
    settings = settings or get_settings()
    ctx = "\n\n".join(contexts)
    t0 = time.perf_counter()
    cost = 0.0

    raw, c1 = _run_prompts(settings, [EXTRACT_CLAIMS.format(ground_truth=ground_truth)],
                           provider=provider, model=model, concurrency=1, max_tokens=256)
    cost += c1
    claims = _split_lines(raw[0])
    if not claims:
        return {"score": 0.0, "n": 0, "n_failed": 0, "items": [],
                "cost_cny": round(cost, 6), "latency_s": round(time.perf_counter() - t0, 3)}

    prompts = [JUDGE_COVERED.format(claim=c, context=ctx) for c in claims]
    texts, c2 = _run_prompts(settings, prompts, provider=provider, model=model,
                             concurrency=concurrency, max_tokens=16)
    cost += c2
    flags = [_parse_verdict(t) for t in texts]
    score, n_valid, n_failed = _mean(flags)
    items = [{"claim": c, "covered": f} for c, f in zip(claims, flags)]

    return {"score": round(score, 4), "n": n_valid, "n_failed": n_failed, "items": items,
            "cost_cny": round(cost, 6), "latency_s": round(time.perf_counter() - t0, 3)}
