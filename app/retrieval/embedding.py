"""Embedding 客户端：三个后端一个入口。"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
ZHIPU_MODEL = "embedding-3"
ZHIPU_PRICE_PER_M = 0.5


def embed_zhipu(texts: list[str], batch: int = 32) -> tuple[list[list[float]], dict]:
    """智谱 embedding-3。返回 (向量列表, 用量统计)。"""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:  # noqa: BLE001
        pass
    key = os.getenv("ZHIPU_API_KEY") or ""
    if not key:
        raise RuntimeError("没有 ZHIPU_API_KEY，检查 .env")

    vecs: list[list[float]] = []
    tokens = 0
    t0 = time.time()

    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        body = json.dumps({"model": ZHIPU_MODEL, "input": part}).encode("utf-8")
        req = urllib.request.Request(
            ZHIPU_URL, data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            detail = ""
            if hasattr(e, "read"):
                try:
                    detail = e.read().decode("utf-8", "replace")[:200]
                except Exception:  # noqa: BLE001
                    pass
            raise RuntimeError(f"第 {i // batch} 批失败：{type(e).__name__} {detail}") from e

        got = sorted(d.get("data", []), key=lambda x: x.get("index", 0))
        vecs += [g["embedding"] for g in got]
        tokens += d.get("usage", {}).get("total_tokens", 0)
        n = i + len(part)
        if n % (batch * 10) < batch or n == len(texts):
            print(f"    {n}/{len(texts)}  用时 {time.time() - t0:.0f}s")

    cost = tokens / 1_000_000 * ZHIPU_PRICE_PER_M
    stats = {"backend": "zhipu", "model": ZHIPU_MODEL, "tokens": tokens,
             "cost_cny": round(cost, 6), "seconds": round(time.time() - t0, 1)}
    return vecs, stats


BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
BAILIAN_MODEL = "text-embedding-v4"
BAILIAN_PRICE_PER_M = 0.7


BAILIAN_MAX_BATCH = 10

def embed_bailian(texts: list[str], batch: int = BAILIAN_MAX_BATCH) -> tuple[list[list[float]], dict]:
    """阿里百炼 text-embedding-v4。返回 (向量列表, 用量统计)。"""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:  # noqa: BLE001
        pass
    key = os.getenv("DASHSCOPE_API_KEY") or ""
    if not key:
        raise RuntimeError("没有 DASHSCOPE_API_KEY，检查 .env")

    vecs: list[list[float]] = []
    tokens = 0
    t0 = time.time()

    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        body = json.dumps({"model": BAILIAN_MODEL, "input": part}).encode("utf-8")
        req = urllib.request.Request(
            BAILIAN_URL, data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            detail = ""
            if hasattr(e, "read"):
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
            raise RuntimeError(f"第 {i // batch} 批失败：{type(e).__name__} {detail}") from e

        got = sorted(d.get("data", []), key=lambda x: x.get("index", 0))
        if len(got) != len(part):
            raise RuntimeError(f"第 {i // batch} 批返回 {len(got)} 条，期望 {len(part)} 条")
        vecs += [g["embedding"] for g in got]
        tokens += d.get("usage", {}).get("total_tokens", 0)
        n = i + len(part)
        if n % (batch * 10) < batch or n == len(texts):
            print(f"    {n}/{len(texts)}  用时 {time.time() - t0:.0f}s")

    cost = tokens / 1_000_000 * BAILIAN_PRICE_PER_M
    stats = {"backend": "bailian", "model": BAILIAN_MODEL, "tokens": tokens,
             "cost_cny": round(cost, 6), "seconds": round(time.time() - t0, 1)}
    return vecs, stats


def embed_local(texts: list[str], model_name: str = "BAAI/bge-small-zh-v1.5") -> tuple[list[list[float]], dict]:
    """本地 SentenceTransformer（bge-small-zh-v1.5）。零成本、可离线。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "本地后端需要 sentence-transformers + torch：\n"
            "  pip install numpy torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install sentence-transformers -i https://pypi.org/simple\n"
            "装不上就用 --backend zhipu") from e

    t0 = time.time()
    model = SentenceTransformer(model_name)
    load_s = time.time() - t0
    vecs = model.encode(texts, batch_size=16, normalize_embeddings=True,
                        show_progress_bar=False).tolist()
    stats = {"backend": "local", "model": model_name, "tokens": 0, "cost_cny": 0.0,
             "load_seconds": round(load_s, 1),
             "seconds": round(time.time() - t0, 1)}
    return vecs, stats


def pick_backend(want: str) -> str:
    """auto：有 torch 就走本地（零成本），否则回退云端 API。不假装有什么。"""
    if want != "auto":
        return want
    try:
        import sentence_transformers  # noqa: F401
        return "local"
    except Exception:  # noqa: BLE001
        print("  未装 sentence-transformers → 回退到百炼 text-embedding-v4 API")
        return "bailian"


def embed(texts: list[str], backend: str, batch: int = 32) -> tuple[list[list[float]], dict]:
    """按后端名分派。三个后端都要从这里走，别在各处 if/else。"""
    if backend == "local":
        return embed_local(texts)
    if backend == "bailian":
        return embed_bailian(texts, batch=min(batch, BAILIAN_MAX_BATCH))
    if backend == "zhipu":
        return embed_zhipu(texts, batch=batch)
    raise ValueError(f"未知后端：{backend}")
