# llm-service

> 从 **LLM 接入层** 到 **RAG 全链路 + 评测体系**。
> 6 周冲刺（2026.09–10）的主项目，也是作品集里的第一个可演示工程。

一句话概括这个项目在做什么：**不只是"把 RAG 跑起来"，而是建一套能判断"改动到底有没有用"的评测体系，然后用它做决策。**

---

## 核心成果（W1–W3）

### 1. RAG 全链路（自建，未依赖 LangChain）

```
官方文档抓取 → 解析 → 原子感知分块 → 向量化 → 混合检索 → 生成端（引用 + 拒答）
                                                    ↑
                                          评测体系贯穿每一环
```

| 环节 | 关键设计 | 实测效果 |
|---|---|---|
| 语料 | 三家大模型平台官方 API 文档 | 342 篇 / 约 40 万 token |
| 解析 | 表格转 Markdown 保结构、代码块"排除后太短就回退" | 避免静默丢一半语料 |
| 分块 | **原子感知**（表格 + 代码块整体保留） | 表格破损率 83.3%→**28.5%**、代码破损率 83.2%→**0%** |
| 检索 | 混合检索（向量 + BM25 经 **RRF** 融合） | 召回@5 **69.2%** |
| 生成 | 引用溯源 + **两道拒答闸门** | 忠实度 **96.6%** |

### 2. 评测体系（核心差异化）

- **120 条 golden set**：按 term / semantic / multi / boundary / conflict 五类难度标注，难例占 **33%**；
  另写体检脚本量化评测集自身的来源与难度偏斜（修复 4 处偏斜：难例不足、平台覆盖不均、结构性空格）
- **手写 LLM-as-judge 五指标**：召回@5、忠实度、答案相关性、上下文召回、p95 延迟与单次成本
  —— 不依赖 RAGAS / LangChain，能讲清每一处实现
- **成本三分项**：检索 / 生成 / 评测分开统计。单次总 ¥0.00528 中，**生成只占 ¥0.00047**
  —— 不分开就会误以为"系统变贵了"

### 3. 优化迭代：6 种策略全部实测，只有 1 种留下

| 策略 | 结论 | 依据 |
|---|---|---|
| HyDE | **否决** | 术语精确型语料上召回反降 19pp |
| Logical Routing | **否决** | 跨平台题被截断 |
| Self-Query | **否决** | 过滤更激进，跨文档全中掉到 8.3% |
| ParentDedup | **否决** | 去重误伤答案块（大样本下才暴露） |
| MultiQuery | 待定 | 难例召回 +10pp，但延迟 3 倍 |
| **Rerank** | **采用** | **跨文档综合题全中率 12.5% → 37.5%** |
| **RRF k=100** | **采用** | 零成本 +1.6pp |

**最能说明「评测驱动决策」的一处**：Rerank 的整体召回是从 69.2% **降到** 62.5% 的，
单看这个数就该砍掉它。但按题型拆开：

| 题型 | baseline | Rerank | 差值 |
|---|---|---|---|
| term | 74.0% | 64.0% | -10.0pp |
| semantic | 73.3% | 66.7% | -6.7pp |
| **multi** | 73.3% | **80.0%** | **+6.7pp** |
| boundary | 80.0% | 66.7% | -13.3pp |
| conflict | 10.0% | 10.0% | 0.0pp |

→ **Rerank 是"难例题专用工具"**：它只伤简单题、只救跨文档题。整体平均值把这个结构完全掩盖了。

### 4. 评测端治理（一件反直觉的事）

原计划优化生成端 Prompt，动手前先看失败案例，却发现 **faithfulness 的低分几乎全部来自 judge 自身的缺陷**，不是生成端幻觉：

| 缺陷 | 现象 | 修法 |
|---|---|---|
| 空答案污染 | 模型对空输入回"你还没发那段回答呢"，被当成陈述 | 短路，不调 LLM |
| 并列句拆散 | "从 A、B 或 C 起算，以较晚者为准"被拆成 3 条独立陈述 | prompt 约束 |
| 短答案丢符号 | `-9999.0` 被改写成 `9999.0`（负号没了） | 短路，不让 LLM 改写 |

修完：**忠实度 92.4% → 96.6%**，同时**评测成本下降 57%**（短路省掉 27 条拒答题的无效调用）。

---

## 当前进度

- [x] **W1（9/20–9/25）LLM 服务** — 多 provider、流式、降级、格式化输出、成本回传
- [x] **W2（9/26–10/2）RAG 全链路** — 语料 → 解析 → 分块 → 向量 → 混合检索 → 生成端；baseline 四指标
- [x] **W3（10/3–10/9）评测体系 + 策略优化** — 120 条 golden set、五指标 judge、6 种策略实测、迭代曲线
- [~] **W4 Agent + MCP Server + 可观测** — 块 A 手写 ReAct ✅ / 块 B 受控 workflow ✅ / 块 C 四道护栏 + 确认点接流式 ✅ / 块 D MCP Server ✅ / 块 E 可观测（进行中）
- [ ] W5 生产化、成本优化、作品集包装

---

## MCP Server（可被独立安装）

`mcp_server/` 是一个标准 MCP Server，把本仓库的检索与问答能力暴露成 MCP tools，
供 Claude Desktop / Cursor 等 Host 直接调用。**只读，无副作用。**

```json
{
  "mcpServers": {
    "api-docs": {
      "command": "D:\\AIWorkspace\\workbuddy\\llm-service\\.venv\\Scripts\\python.exe",
      "args": ["D:\\AIWorkspace\\workbuddy\\llm-service\\mcp_server\\run.py"]
    }
  }
}
```

- 工具：`search_api_docs`（检索片段）、`ask_api_docs`（带引用问答）
- 资源：`docs://corpus/stats`（只读快照）
- 验证：`python scripts/mcp_smoke.py` —— 起真子进程走真 stdio，跑
  `initialize → tools/list → tools/call → resources/read`

细节见 [`mcp_server/README.md`](mcp_server/README.md)。

---

## 快速开始

```bash
# 1) 虚拟环境
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2) 装依赖（卡住就加官方源 -i https://pypi.org/simple）
pip install -r requirements.txt

# 3) 配置密钥
cp .env.example .env            # Windows: copy .env.example .env
# 编辑 .env，填入三个 key

# 4) 起服务
uvicorn app.main:app --reload --port 8000

# 5) 验证
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/health/ready?ping=1"
```

打开 http://127.0.0.1:8000/docs 看自动生成的交互文档。

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活探针。只看进程活着没活，不碰网络 |
| GET | `/health/ready` | 就绪探针。查三家 provider 状态 + 成本账本汇总 |
| POST | `/chat` | 结构化输出（Pydantic 校验 + 解析失败重试） |
| POST | `/ask` | **RAG 问答**：检索 → 生成 → 带引用 → 敢说不知道 |
| POST | `/ask/batch` | 并发批量问答（评测与压测用，信号量限流） |
| GET | `/ask/gate` | 拒答阈值、怎么定的、挡不住什么 |
| POST | `/retrieve` | 只要检索不要生成（对照实验用） |

## 评测与复现

```bash
# 快速筛选：只跑检索指标（约 1 分钟，成本几乎为零）
python scripts/run_eval.py --skip-gen

# 完整五指标（含 LLM-as-judge）
python scripts/run_eval.py --strategy baseline --out data/eval/baseline.json

# 换个策略比
python scripts/run_eval.py --strategy rerank --types multi,boundary,conflict

# 评测集体检（来源/难度偏斜）
python scripts/golden_stats.py
```

**当前迭代曲线（120 条 golden set，hybrid）**：

| 阶段 | 召回@5 | 跨文档全中 | 忠实度 | p95 |
|---|---|---|---|---|
| W3 baseline | 69.2% | 12.5% | 92.4% | 1755ms |
| + RRF k=100 | **70.8%** | 12.5% | — | ~350ms |
| + Rerank | 62.5% | **33.3%** | — | 1597ms |
| Rerank · 难例 40 条 | 60.0% | **37.5%** | 89.2% | 3016ms |
| judge 修复后 | 69.2% | 12.5% | **96.6%** | 1751ms |

> 口径说明：部分数字来自不同样本（120 条 vs 难例 40 条），不可直接横比；
> W2 的「忠实度 0.988」是实体重叠**下界**，与上表的 96.6%（LLM-as-judge 真指标）不是同一把尺子。

## 目录结构

```
app/
  main.py         FastAPI 应用、路由、请求 ID 中间件
  config.py       集中读取环境变量与 .env
  providers.py    三家 provider 元信息与就绪探活
  cost.py         成本账本读取与汇总
  llm.py          LLM 调用层（标准库 urllib，含错误人话化与重试判定）
  rag/            RAG 生成端：两道拒答闸门、引用校验、提示词
  retrieval/      检索子系统：embedding / tokenize / bm25 / hybrid / rerank
                  / store / pipeline / strategies（6 种可插拔策略）
  evals/          评测子系统：手写 LLM-as-judge 三指标
scripts/
  run_eval.py          五指标评测（含 --skip-gen 廉价筛选）
  golden_stats.py      golden set 分布体检
  make_golden_draft.py 半自动出题（5 种题型 + 质量过滤）
  merge_golden.py      草稿合并（去重 + 配额 + 校验）
  run_baseline.py      W2 的四指标 baseline
tests/                 224 passed
```

## 测试

```bash
python -m pytest -p no:warnings      # 224 passed
```

## 成本纪律

- 每次调用记账到 `.cost_ledger.jsonl`，累计超 `COST_LIMIT_CNY`（默认 ¥20）脚本自动中止
- DeepSeek 高峰档 = 工作日 09:00–12:00 / 14:00–18:00，**跑批与全量评测挪到晚上或周末**，成本减半
- 调试期显式关闭思考模式：`extra_body={"thinking": {"type": "disabled"}}`
- 全量评测跑之前，**先用 5 条样本验证逻辑**
- **评测开销与系统运行成本分开统计**——否则会误以为系统变贵了
