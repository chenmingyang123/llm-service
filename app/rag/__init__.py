"""RAG 生成端（W2 第 6 天）。"""
from .abstain import DEFAULT_MIN_COSINE, extract_cited, gate, parse_answer
from .pipeline import AskPipeline
from .prompts import SYSTEM, USER_TEMPLATE, build_context
from .schemas import AskBatchIn, AskIn, AskOut, CitationOut

__all__ = [
    "AskBatchIn", "AskIn", "AskOut", "AskPipeline", "CitationOut",
    "DEFAULT_MIN_COSINE",
    "SYSTEM", "USER_TEMPLATE", "build_context", "extract_cited", "gate",
    "parse_answer",
]
