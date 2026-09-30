# api-docs MCP Server

把三家平台（DeepSeek / 智谱 / 阿里百炼）官方 API 文档的**检索**与**问答**能力，
封装成标准 MCP tools。任何支持 MCP 的 Host（Claude Desktop、Cursor、各种 IDE 插件…）
配一行就能用，不需要知道背后的向量库、分块策略、rerank 是什么。

> **只读**。没有任何写工具，不产生副作用。工具清单里只有 `search_api_docs` 和
> `ask_api_docs`，两者都带 `read_onlyHint` 标记。

---

## 一行配置

把下面这段贴进 Host 的 MCP 配置里（路径按你的实际位置改）：

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

`run.py` 用的是**绝对路径**，所以不依赖 Host 的工作目录 —— 这是刻意设计的：
Host 拉起子进程时 cwd 不由你控制，配置一旦依赖 cwd 就会出现「我这儿能跑、别人装了不行」。

如果你确定 Host 的工作目录是仓库根，也可以用模块入口：

```json
{ "command": "<仓库的 python 绝对路径>", "args": ["-m", "mcp_server"] }
```

---

## 安装

```bash
git clone <repo>
cd llm-service
python -m venv .venv
.venv/Scripts/python -m pip install -e .          # Windows
# source .venv/bin/activate && pip install -e .   # macOS / Linux
```

依赖里已经包含 `mcp>=2.0`。另需一份 `.env`（检索的 query 向量化与问答生成要调模型）：

```
DEEPSEEK_API_KEY=...
BAILIAN_API_KEY=...      # embedding 默认走百炼 text-embedding-v4
```

## 准备索引

检索需要一个 chunk 索引：`data/embed/chunks.jsonl`（本项目约 87 MB，**不进仓库**）。

```bash
# 建一次即可，之后反复用
python -m scripts.build_index       # 具体命令见仓库根 README「构建索引」一节
```

**索引缺失不会让 server 挂掉**：进程照常启动、工具清单照常列出（这是别人 clone
后的第一体验），只是调 `search_api_docs` 会返回一条「请先建索引」的引导。
先跑通连接、再补数据 —— 这两件事不该互相阻塞。

---

## 工具与资源

| 类型 | 名称 | 说明 |
|---|---|---|
| tool | `search_api_docs(query, source?, top_k?)` | 检索文档片段，返回来源与路径。用术语查询（「上下文硬盘缓存」），不要写长句。 |
| tool | `ask_api_docs(query, mode?)` | 基于文档回答，带引用编号；资料不足会明确拒答，不编。 |
| resource | `docs://corpus/stats` | 只读快照：索引是否就绪、chunk 数、embedding 后端、支持的检索模式。 |

`source` 可选 `deepseek` / `zhipu` / `bailian`，留空表示不限。
`mode` 可选 `vector` / `bm25` / `hybrid`（默认）/ `hybrid_rerank`。

---

## 已验证清单

- [x] `initialize` 握手成功（协议版本 2025-11-25，协议能力 tools + resources）
- [x] `tools/list` 返回 2 个工具，且都带 `readOnlyHint=true`
- [x] `resources/list` 返回 `docs://corpus/stats`
- [x] `tools/call search_api_docs` 真实检索成功（hybrid，约 380–420 ms）
- [x] `resources/read` 返回合法 JSON
- [x] 两个入口（`-m mcp_server` 与 `run.py` 绝对路径）均通过，含**从非仓库目录启动**
- [x] 索引缺失时给出可照做的引导，服务不挂

复现：`python scripts/mcp_smoke.py`（起真子进程走真 stdio）。

---

## 两个坑（踩过，写下来省得别人再踩）

**1. mcp 2.x 把 FastMCP 改名成了 MCPServer。**
网上绝大多数教程还是 1.x 的写法：

```python
from mcp.server.fastmcp import FastMCP        # 1.x
from mcp.server.mcpserver import MCPServer    # 2.x ← 现在用这个
```

**2. stdio 传输下，stdout 只能走协议帧，一个多余字节都不能有。**
本项目检索链路里有进度打印（`print("未装 sentence-transformers…")`），
在 Windows 上会用控制台编码（GBK）写出**非 UTF-8 字节**，直接污染协议流 ——
客户端读到就抛 `UnicodeDecodeError`，表现为「工具调不动」。

更阴的是：**这种错只在真的装进 Host、真的连上时才暴露**，函数级测试完全看不到。
修法见 `server.py` 的 `shield_stdout()`：让 `print()` 走 stderr，
但把 `.buffer` 留给 SDK 发协议帧。

---

## 架构一览

```
Host ──(stdio / JSON-RPC)──> mcp_server
                               ├─ search_api_docs ─┐
                               ├─ ask_api_docs ────┤
                               └─ docs://corpus/stats
                                                     ↓
                                        app.retrieval.RetrievalPipeline   （W2/W3 检索内核）
                                        app.rag.AskPipeline               （检索→生成→引用校验）
```

检索内核与仓库里的 `/ask`、以及 Agent 的 `search_docs` 工具**是同一套**，
MCP 这一层只是把它换个协议暴露出去 —— 不是另写一份。
