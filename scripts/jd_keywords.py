"""JD 关键词对照表 —— W1 第 6 天下午那 3.5 小时的主产出。"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


KEYWORDS: dict[str, list[str]] = {
    "LLM 应用": ["LLM", "大模型", "Prompt", "提示词", "提示工程", "Function Calling",
              "工具调用", "Tool Use", "结构化输出", "JSON", "流式输出", "SSE", "流式",
              "多轮对话", "上下文", "指令微调", "Token"],
    "RAG": ["RAG", "检索增强", "向量库", "向量检索", "Embedding", "向量化", "知识库",
            "混合检索", "BM25", "Rerank", "重排序", "分块", "Chunk", "文档解析",
            "Milvus", "pgvector", "Faiss", "语义检索"],
    "Agent": ["Agent", "智能体", "ReAct", "规划", "任务编排", "多智能体", "MCP",
              "LangGraph", "AutoGPT", "工具集", "自主决策"],
    "评测": ["评测", "评估", "Eval", "评测集", "golden set", "基准测试", "RAGAS",
             "召回率", "忠实度", "准确率", "A/B", "指标体系", "效果评估"],
    "工程化": ["FastAPI", "Docker", "微服务", "异步", "asyncio", "并发", "高并发",
               "缓存", "降级", "熔断", "限流", "重试", "可观测", "监控", "tracing",
               "Langfuse", "CI/CD", "K8s", "Kubernetes", "部署"],
    "成本": ["成本", "成本优化", "Token 成本", "计费", "降本", "模型路由", "小模型分流",
             "缓存命中", "Prompt 缓存", "性价比"],
    "基础": ["Python", "Java", "SQL", "Linux", "Git", "数据结构", "算法", "MySQL", "Redis"],
}


MY_SKILLS: dict[str, tuple[str, str]] = {
    "LLM": ("strong", "llm.py 调用层，三家 provider"),
    "大模型": ("strong", "同上"),
    "Prompt": ("strong", "prompts.py + 09-21 三组对照实验"),
    "提示词": ("strong", "同上"),
    "Function Calling": ("strong", "tools.py 三个 schema + agent.py 主循环"),
    "工具调用": ("strong", "同上"),
    "结构化输出": ("strong", "completion.py Pydantic 契约 + 修复重试"),
    "JSON": ("strong", "同上"),
    "流式输出": ("strong", "streaming.py SSE + 中断处理"),
    "SSE": ("strong", "同上"),
    "流式": ("strong", "同上"),
    "多轮对话": ("some", "agent.py 多轮 tool loop"),
    "上下文": ("some", "消息装配，未做压缩/裁剪"),
    "Token": ("strong", "pricing.py 计费精算 + 成本账本"),
    "成本": ("strong", "pricing.py + /cost 系列端点"),
    "成本优化": ("strong", "缓存盈亏平衡分析"),
    "缓存命中": ("strong", "cache_probe.py 实测 91% 降幅"),
    "缓存": ("strong", "cost.py 命中统计 + cache_probe.py 对照"),
    "高并发": ("some", "batch.py 信号量并发，未做压测到极限"),
    "Prompt 缓存": ("strong", "同上"),
    "降级": ("strong", "resilience.py 降级链"),
    "重试": ("strong", "resilience.py 指数退避 + 抖动"),
    "并发": ("strong", "batch.py 信号量 + asyncio.gather"),
    "异步": ("strong", "同上"),
    "asyncio": ("strong", "同上"),
    "FastAPI": ("strong", "整个服务"),
    "评测": ("some", "方法论强（六年风控），工具链未做"),
    "评估": ("some", "同上"),
    "指标体系": ("some", "同上"),
    "Python": ("strong", "十年"),
    "Java": ("strong", "十年"),
    "SQL": ("strong", "tools.py sqlite + 十年"),
    "Linux": ("some", "日常开发"),
    "Git": ("some", "日常开发"),
    "算法": ("strong", "六年算法工程师"),
    "数据结构": ("strong", "同上"),

    "RAG": ("none", "W2 开始"),
    "检索增强": ("none", "W2 开始"),
    "向量库": ("none", "W2 开始"),
    "Embedding": ("none", "W2 开始"),
    "知识库": ("none", "W2 开始"),
    "Milvus": ("none", "W2"),
    "pgvector": ("none", "W2"),
    "BM25": ("none", "W2 混合检索"),
    "Rerank": ("none", "W2 重排序"),
    "Agent": ("none", "W4"),
    "智能体": ("none", "W4"),
    "MCP": ("none", "W4 发布 MCP Server"),
    "LangGraph": ("none", "W4"),
    "RAGAS": ("none", "W3"),
    "召回率": ("none", "W2 起评测"),
    "Docker": ("none", "W5 部署"),
    "K8s": ("none", "不做"),
    "Kubernetes": ("none", "不做"),
    "Langfuse": ("none", "W4 可观测"),
}

LEVEL_ORDER = {"strong": 2, "some": 1, "none": 0}


def load_jds(path: str) -> list[tuple[str, str]]:
    out = []
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if fn.lower().endswith((".txt", ".md")):
                p = os.path.join(root, fn)
                out.append((fn, open(p, encoding="utf-8", errors="replace").read()))
    return out


DEMO_JDS = [
    ("A_AI应用工程师.txt", """岗位职责：1. 负责大模型应用的设计与开发，包括 Prompt 工程、Function Calling、
    结构化输出；2. 搭建 RAG 检索增强系统，负责知识库构建、向量库选型（Milvus/pgvector）、
    Embedding 与混合检索、Rerank 重排序；3. 建设评测体系，定义评测集与指标；
    4. 优化成本与延迟，做模型路由与缓存。要求：Python 扎实，熟悉 FastAPI，有 Docker 部署经验。"""),
    ("B_LLM后端工程师.txt", """1. 基于 LLM 开发智能客服与 Agent，熟悉工具调用、多轮对话管理；
    2. 负责服务工程化：异步并发、降级熔断、限流重试、可观测 tracing；
    3. 有大模型应用评测经验者优先；4. 熟悉 Java 或 Python，有高并发系统经验。"""),
    ("C_RAG算法工程师.txt", """1. 负责 RAG 全链路：文档解析、分块策略、向量检索、BM25 混合检索、重排序；
    2. 构建评测集与自动化评测（RAGAS 或自研），跟踪召回率与忠实度；
    3. 成本优化：Token 计费分析、Prompt 缓存、小模型分流；4. Python，SQL 熟练。"""),
]


def analyze(jds: list[tuple[str, str]]) -> dict:
    n = len(jds) or 1
    rows = []
    for cat, words in KEYWORDS.items():
        for w in words:
            hits = sum(1 for _, t in jds if w.lower() in t.lower())
            if hits == 0:
                continue
            level, evidence = MY_SKILLS.get(w, ("none", "未覆盖"))
            rows.append({"category": cat, "keyword": w, "jd_count": hits,
                         "jd_pct": round(hits / n * 100, 1), "level": level, "evidence": evidence})
    rows.sort(key=lambda r: (-r["jd_pct"], r["category"]))

    total = len(rows)
    covered = sum(1 for r in rows if r["level"] != "none")
    strong = sum(1 for r in rows if r["level"] == "strong")
    w_total = sum(r["jd_pct"] for r in rows) or 1
    w_covered = sum(r["jd_pct"] for r in rows if r["level"] != "none")

    gaps = [r for r in rows if r["level"] == "none" and r["jd_pct"] >= 30]
    weak = [r for r in rows if r["level"] == "some"]
    return {
        "jd_files": [fn for fn, _ in jds], "jd_num": len(jds),
        "rows": rows, "total": total, "covered": covered, "strong": strong,
        "coverage_pct": round(covered / total * 100, 1) if total else 0,
        "weighted_coverage_pct": round(w_covered / w_total * 100, 1),
        "gaps": gaps, "weak": weak,
    }


def render(r: dict) -> str:
    L = [f"# JD 关键词对照表", "",
         f"- 样本：**{r['jd_num']} 份 JD**",
         f"- 命中关键词：**{r['total']} 个**，其中我具备 **{r['covered']}** 个（强项 {r['strong']} 个）",
         f"- **简单覆盖率 {r['coverage_pct']}%　加权覆盖率 {r['weighted_coverage_pct']}%**"
         f"（加权按出现频率计，更能反映真实匹配度）", "",
         "## 全量对照", "",
         "| 类别 | 关键词 | 出现在 | 占比 | 我的程度 | 证据 |",
         "|---|---|---|---|---|---|"]
    lv = {"strong": "**强**", "some": "部分", "none": "**无**"}
    for x in r["rows"]:
        L.append(f"| {x['category']} | {x['keyword']} | {x['jd_count']}/{r['jd_num']} | "
                 f"{x['jd_pct']}% | {lv[x['level']]} | {x['evidence']} |")

    L += ["", "## 缺口（高频但我没有 —— 这就是 W2–W6 的优先级）", ""]
    if r["gaps"]:
        L += ["| 关键词 | 占比 | 什么时候补 |", "|---|---|---|"]
        for x in r["gaps"]:
            L.append(f"| {x['keyword']} | {x['jd_pct']}% | {x['evidence']} |")
    else:
        L.append("_没有高频缺口。_")

    if r["weak"]:
        L += ["", "## 半桶水（有方法论但没落地成工具 —— 最容易被追问穿）", ""]
        for x in r["weak"]:
            L.append(f"- **{x['keyword']}**（{x['jd_pct']}%）—— {x['evidence']}")

    L += ["", "## 怎么读这张表", "",
          "- **加权覆盖率比简单覆盖率重要**：一个出现 100% 的词没掌握，比五个出现 10% 的词没掌握严重得多。",
          "- **缺口栏按占比排序**，从上往下补就是性价比最高的顺序。",
          "- **半桶水那一栏最危险**：面试时被追问「你用什么工具做的」会答不上来。"
          "要么补成能演示的，要么就别在简历上写。",
          "", "---", "",
          f"样本文件：{', '.join(r['jd_files']) or '（示例数据）'}"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/jd", help="JD 文件目录（txt/md，一个文件一份）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", action="store_true", help="用内置示例 JD 跑一遍看结构")
    a = ap.parse_args()

    if a.demo or not os.path.isdir(a.dir):
        if not a.demo:
            print(f"[提示] {a.dir} 不存在，先用内置示例跑一遍看结构。")
            print(f"       把真实 JD 存成 txt/md 放进 {a.dir}/ 再跑一次。\n")
        jds = DEMO_JDS
    else:
        jds = load_jds(a.dir)
        if not jds:
            print(f"{a.dir} 里没有 txt/md 文件。")
            return

    r = analyze(jds)
    md = render(r)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"已写入 {a.out}")
    else:
        print(md)

    print(f"\n简单覆盖率 {r['coverage_pct']}%　加权覆盖率 {r['weighted_coverage_pct']}%　"
          f"高频缺口 {len(r['gaps'])} 个", file=sys.stderr)


if __name__ == "__main__":
    main()
