# 通用企业服务 Agent + 开源 MCP Server

> 一个可运行的 LLM 应用工程样本：多 provider 接入、RAG 检索问答、带护栏的 Agent、标准 MCP Server、可回放的 trace。
>
> 定位**通用企业服务场景**（工单 / 通知 / 订单查询 / 文档问答），不做行业垂直定制。

---

## 一、一句话

**让企业已有的数据和系统，既能被「问」，也能被「办」，且整个过程可观测、可审计、有护栏。**

---

## 二、能做什么

| 能力 | 说明 | 入口 |
|---|---|---|
| **文档问答（RAG）** | 3700 块 API 文档语料，混合检索 + 重排，答案带引用编号，证据不足会拒答 | `POST /ask` |
| **多步任务 Agent** | 手写 ReAct（纯文本协议）+ 原生 Function Calling 双路径，工具调用看得见，写操作要人确认 | `POST /agent/react` / `POST /agent/stream` |
| **人工确认点** | 写类工具暂停等人，决定走独立端点，超时按拒绝（fail-closed） | `POST /agent/confirm` |
| **标准 MCP Server** | 2 个 tool + 1 个 resource，stdio 传输，能被 Claude Desktop / Cursor 直接装 | `mcp_server/` |
| **可观测** | 每次运行落一条 JSONL trace，可渲染成瀑布图（步数 / 每步耗时 / 成本） | `scripts/trace_view.py` |

---

## 三、快速开始

```bash
# 1. 装依赖（Python >= 3.10）
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"        # Windows
# source .venv/bin/pip install -e ".[dev]"   # Linux / macOS

# 2. 配 key（只放本地，不进仓库）
cp .env.example .env
# 至少填一个：DEEPSEEK_API_KEY / ZHIPU_API_KEY / BAILIAN_API_KEY

# 3. 起服务
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

# 4. 验证
curl http://127.0.0.1:8000/health
```

**成本闸门**：默认累计 ¥20 抛异常中止（`COST_LIMIT_CNY`）。跑批前先看 `/cost`。

---

## 四、三分钟演示

### ① 带思考过程的 Agent（手写 ReAct）

```bash
curl -s http://127.0.0.1:8000/agent/react \
  -H "Content-Type: application/json" \
  -d '{"question":"查一下北京天气，再看近7天订单统计","max_steps":4,"trace":true}'
```

返回里每步都带 `thought`，`trace_id` 可用于回放。

### ② 流式 + 人工确认点（最有演示价值）

```bash
curl -N http://127.0.0.1:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"帮我建一张工单：登录失败需要排查","trace":true}'
```

你会看到流里推来 `need_confirm`，然后**暂停等人**：

```bash
curl -s http://127.0.0.1:8000/agent/confirm \
  -H "Content-Type: application/json" \
  -d '{"confirm_id":"<上面事件里的>","approve":true}'
```

不点确认就**不会执行**（超时 120s 按拒绝）。

### ③ 看这条运行的瀑布图

```bash
.venv/Scripts/python.exe scripts/trace_view.py --open
```

### ④ MCP Server 冒烟（不花真钱）

```bash
.venv/Scripts/python.exe scripts/mcp_smoke.py
```

---

## 五、架构

```
客户端 → FastAPI 端点 → 自研核心（无 Agent 框架）→ 框架依赖
                          ├ react.py      手写 ReAct（纯文本协议）
                          ├ agent.py      原生 FC 循环
                          ├ guards.py     四道闸门
                          ├ observe.py    span 树 → JSONL
                          └ streaming.py  SSE + 确认点
                                 ↓
                   pydantic(校验) / httpx(LLM) / numpy(向量) / mcp(协议)
                                 ↓
                   本地 JSONL = 真相来源（后端只是查看器）
```

**关键取舍：Agent 核心不依赖任何 Agent 框架。**
LangGraph / Langfuse 都**没有进主依赖**（做成对照脚本与可选导出器）。

| 文件 | 看什么 |
|---|---|
| `app/react.py` | 手写 ReAct：解析三态、格式纠错回灌、三个终止出口 |
| `app/guards.py` | 四道闸门 + 两级批准（流程层 ≠ 动作层） |
| `app/observe.py` | 事件流 → span 树，JSONL 落盘与 patch 合并 |
| `mcp_server/server.py` | MCPServer + 2 tool + 1 resource |
| `scripts/` | 各类冒烟脚本（默认零成本） |

---

## 六、工程上的几个判断

| 决策 | 理由 |
|---|---|
| **手写 ReAct，不用框架** | 事件自己发 → 接 trace 时 Agent 代码改 0 行。框架抽象层吃掉的价值 > 带来的 |
| **纯文本协议为主** | 不传 `tools`/`tool_choice`，最差的模型也能跑（降级路径） |
| **熔断返回空答案** | 目标是「不烧钱 + 不骗人」，不是「一定要有答案」 |
| **流程层批准 ≠ 动作层批准** | 两套令牌独立。混用会出现「批准了 A、B 也被放行」 |
| **幂等做双层** | 单层挡不住并发；靠「检查」不行，得靠写入时约束 |
| **本地 JSONL 是真相来源** | 判据：删掉后端，我的历史一条不少 |
| **成本必须三分项** | 检索 / 生成 / 评测分开记，否则分不清「贵在哪」 |
| **观测层不吞异常** | 记 error 后继续抛出 |

---

## 七、真实数据（来自本机实测）

| 指标 | 数值 | 说明 |
|---|---|---|
| RAG 召回@5 | **69.17%** | 120 条评测集 |
| 答案忠实度 | **96.58%** | 修复评测端缺陷前是 92.39% |
| 上下文召回 | 76.90% | |
| p95 延迟 | **1751 ms** | 120 条 |
| 单次成本 | **¥0.0053** | 其中 judge 占 90.6%，真实服务只 ¥0.0005 |
| 检索成本 | **¥0** | 本地暴力全量检索，不调 API |
| 测试 | **355 passed / 2 skipped** | |

> 这些数字来自 `data/eval/*.json`，可复现。
> 跨文档类「全中率」只有 33.33%（40 条难题子集 37.5%）——**没达标，但它是真的，我没有为了好看去改评测集。**

---

## 八、已知边界（诚实列出）

- **Langfuse 导出器未实测**（本机未安装、无 key），只做接口对齐；未安装时优雅退化到本地 JSONL。
- **`llm_ms_derived` 是推导值**（step 耗时 − tool 耗时），不是实测值。
- **MCP Server 依赖主项目**（`app.rag` / `app.retrieval`），**不能单独发仓库**，需整仓发布。
- **LangGraph 对照脚本依赖未安装的 langgraph**，需额外 `pip install langgraph` 才能跑。
- **生产更推荐 workflow**：Agent 的错误会累积（十步每步 95% 正确，端到端只剩 60%）。本项目的 Agent 主要作能力展示。

---

## 九、License 与状态

状态：**进行中**（持续迭代，欢迎提 issue）。
