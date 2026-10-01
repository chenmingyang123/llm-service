# demo 录屏操作步骤

> 分镜沿用 W4 手册附录（三条片段 75s / 90s / 80s，**单条硬上限 2 分钟**）。
> 本文补的是手册里没有的：**每一步具体敲什么、说什么、怎么收尾**。

---

## 〇、录之前先跑一遍自检

```bash
.venv/Scripts/python.exe scripts/demo_prep.py --check
```

它会检查四件事：`.env` 有没有 key、语料在不在（3700 块）、有没有旧 trace 混进来、端口占没占。

**四件全绿再开录。** 录到一半发现 key 没配，是最浪费时间的翻车方式。

```bash
# 清掉旧 trace —— 否则画面里会混进上一次的运行，讲解会乱
.venv/Scripts/python.exe scripts/demo_prep.py --clean

# 起服务（另开一个终端，录完 Ctrl+C）
.venv/Scripts/python.exe scripts/demo_prep.py --serve
```

---

## 片段 ① Agent 多步任务（75s）

**主张：它真的能自己决定走几步，不是写死的流程。**

```bash
curl -N http://127.0.0.1:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"查一下北京天气，再看近7天订单统计","max_turns":4,"trace":true}'
```

| 时间 | 画面 | 说什么 |
|---|---|---|
| 0–10s | 终端 + 一句场景说明 | 「一个多步任务，我让它先查天气再查订单统计。」<br>**别介绍项目背景**——老板没耐心听 |
| 10–50s | SSE 实时滚动 | 「注意它是自己决定先查哪个、要不要再查一次，步数不是我写死的。」 |
| 50–75s | summary + trace_id | 「四步走完，成本 ¥X，p95 Y 毫秒。这个 trace_id 可以回放每一步。」 |

---

## 片段 ② 出错与兜底（90s）★ 面试分最高

**主张：出错时不崩溃、不编造，写操作会停下来等人。**

```bash
curl -N http://127.0.0.1:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"帮我建一张工单：登录失败需要排查","trace":true}'
```

| 时间 | 画面 | 说什么 |
|---|---|---|
| 0–10s | 说明 | **「下面这段是我故意让它出错的。」** —— 先说这句，否则观众会以为你不熟练 |
| 10–40s | 工具报错 → 换参数 | 「工具返回错了，它换了参数重试，没有死循环。」 |
| 40–75s | 流里推来 `need_confirm` 后**停住** | 「写操作不直接执行，它停在这里等人。SSE 是单向的，所以决定走另一个端点。」 |
| 75–90s | 拒绝确认 → 优雅退出 | **「我拒绝它。」** 然后看它如实退出、不产生副作用 |

拒绝那条命令（把 confirm_id 换成流里的真值）：

```bash
curl -s http://127.0.0.1:8000/agent/confirm \
  -H "Content-Type: application/json" \
  -d '{"confirm_id":"<换成真实的>","approve":false}'
```

> **这段是本片重点。** 「会停」比「会做」更能证明工程能力——
> 因为前者意味着你知道什么情况下**不该**让它做。

---

## 片段 ③ MCP 被装（80s）★

**主张：别人能直接装上用，不是只有我机器上能跑。**

```bash
.venv/Scripts/python.exe scripts/mcp_smoke.py
```

| 时间 | 画面 | 说什么 |
|---|---|---|
| 0–20s | README 的一行配置 | 「把这段贴进 Host 配置，路径按实际位置改。」 |
| 20–60s | 真客户端跑通 initialize → list → call | 「这是官方 SDK 的 client，跟 Claude Desktop 内部那个是同一套 API。」 |
| 60–80s | 返回结果 | 「两个 tool 都是只读，标了 readOnlyHint，没有副作用。」 |

> ⚠️ **录制前确认 GitHub 仓库已发布**（这步需要你手动 push，我没凭证）。
> 没发布就先不录这段——「我说它能被装」和「我演示它被装」差一个量级。

---

## 收尾（每条都做）

```bash
.venv/Scripts/python.exe scripts/trace_view.py --open
```

打开刚那条 trace 的瀑布图，指着说：**「每步多少毫秒、多少钱、有没有失败，都能看到。」**

---

## 录制纪律

- **单条硬上限 2 分钟。** 超过就掉人——中小公司老板的耐心比你想的短。
- **一条视频只证明一个主张。** 不要录一条长片，录三条短的，分别贴进 README 对应小节。
- **出错不要重录整条**，只补那一段。重录整条的代价是你最后干脆不录了。
- **不要介绍背景，直接进场景。** 前三秒决定观众看不看下去。
- 终端字号调大（16px 以上），录屏分辨率 1280×720 足够。

---

## 一键拿全部命令

```bash
.venv/Scripts/python.exe scripts/demo_prep.py
```

会把三条片段的命令、旁白要点、收尾命令全部打印出来，照着跑就行。
