"""各家 tokenizer 差异实测 —— W1 第 5 天上午第一件事。"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings  # noqa: E402
from app.llm import LLMError, call_chat  # noqa: E402
from app.pricing import MODEL_PRICES, unit_cny  # noqa: E402

SAMPLE = """检索增强生成（RAG, Retrieval-Augmented Generation）是一种让大模型在回答前
先检索外部知识库的技术。它的核心流程是：query -> embedding -> vector search -> rerank -> generate。

```python
def retrieve(query: str, top_k: int = 5) -> list[Chunk]:
    vec = embed(query)
    return index.search(vec, top_k)
```

2026 年，RAG 系统的评测指标通常包括 faithfulness、answer_relevancy、context_recall 三项，
单次调用成本约 ¥0.0014，p95 延迟 3.33 秒。"""

PROBES = [
    ("deepseek", "deepseek-flash"),
    ("zhipu", "glm-4.7-flash"),
    ("dashscope", "qwen-plus"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", default=None, help="用指定文件的文本做样本")
    ap.add_argument("--out", default=None, help="写入 Markdown 文件")
    a = ap.parse_args()

    text = open(a.text_file, encoding="utf-8").read() if a.text_file else SAMPLE
    s = get_settings()

    rows = []
    for provider, model in PROBES:
        try:
            r = call_chat(s, [{"role": "user", "content": text}],
                          provider=provider, model=model, max_tokens=16)
        except LLMError as e:
            rows.append({"provider": provider, "model": model, "ok": False,
                         "error": str(e)[:160]})
            print(f"[{provider:9}] 失败：{str(e)[:80]}")
            continue
        tin = r["usage"]["in"]
        rows.append({"provider": provider, "model": model, "ok": True,
                     "prompt_tokens": tin, "chars": len(text),
                     "chars_per_token": round(len(text) / tin, 2) if tin else 0,
                     "input_cny_per_m": unit_cny(model, "in_miss"),
                     "cost_per_1k_calls_cny": round(
                         (unit_cny(model, "in_miss") or 0) * tin / 1_000_000 * 1000, 4)})
        print(f"[{provider:9}] {model:16} prompt_tokens={tin:5d}  "
              f"每 token {len(text)/tin if tin else 0:.2f} 字符  "
              f"千次调用输入成本 ¥{(unit_cny(model,'in_miss') or 0)*tin/1e6*1000:.4f}")

    ok = [r for r in rows if r["ok"]]
    md = ["# tokenizer 差异实测", "",
          f"- 样本文本：{len(text)} 字符（中英混排 + 代码块 + 数字）",
          f"- 数据来源：各家返回的 `usage.prompt_tokens`，不是本地估算", "",
          "| provider | 模型 | prompt tokens | 字符/token | 输入单价 ¥/百万 | 千次调用输入成本 |",
          "|---|---|---|---|---|---|"]
    for r in ok:
        md.append(f"| {r['provider']} | `{r['model']}` | {r['prompt_tokens']} | "
                  f"{r['chars_per_token']} | {r['input_cny_per_m']} | ¥{r['cost_per_1k_calls_cny']} |")
    for r in rows:
        if not r["ok"]:
            md.append("")
            md.append(f"- `{r['provider']}` 未跑通：{r['error']}")

    if ok:
        mx = max(r["prompt_tokens"] for r in ok)
        mn = min(r["prompt_tokens"] for r in ok)
        md += ["", f"**最大差异：{mx} vs {mn}，相差 {(mx-mn)/mn*100:.1f}%。**", "",
               "这意味着按一家估的成本，换到另一家会偏这么多 —— "
               "跨 provider 做成本对比时必须用各家自己的 token 数，不能用一个数折算。"]

    out = "\n".join(md)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"\n已写入 {a.out}")
    else:
        print()
        print(out)

    if "--json" in sys.argv:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
