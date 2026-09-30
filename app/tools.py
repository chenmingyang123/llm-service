"""工具定义与执行 —— W1 第 3 天（Function Calling）的核心资产。"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError


class WeatherArgs(BaseModel):
    city: str = Field(description="城市名")
    unit: str = Field("celsius", pattern="^(celsius|fahrenheit)$")


class StatsArgs(BaseModel):
    sql: str = Field(description="只读 SELECT 语句")
    limit: int = Field(100, ge=1, le=1000)


class OrderArgs(BaseModel):
    order_id: str = Field(description="订单号", min_length=4)


class SearchDocsArgs(BaseModel):
    """检索 API 文档的参数。"""
    query: str = Field(description="检索语句，用文档里可能出现的术语，不要写完整句子",
                       min_length=2)
    source: str | None = Field(
        None, description="限定平台：deepseek / zhipu / bailian。不确定时留空（不要猜）")
    top_k: int = Field(5, ge=1, le=20, description="返回条数，默认 5")


class CreateTicketArgs(BaseModel):
    """建工单。"""
    title: str = Field(description="工单标题", min_length=2, max_length=80)
    body: str = Field(description="工单正文：要说明用户遇到了什么、期望什么", min_length=2)
    priority: str = Field("normal", pattern="^(low|normal|high)$")
    request_id: str = Field(
        description="本次动作的唯一标识（可用 任务摘要+时间戳 生成）。"
                    "**同一个 request_id 只会生效一次**，重复调用返回首次结果，不会再建一单。",
        min_length=4)


class SendNoticeArgs(BaseModel):
    channel: str = Field(description="通知渠道", pattern="^(email|sms|webhook)$")
    to: str = Field(description="接收方（邮箱 / 手机号 / webhook 名）", min_length=2)
    content: str = Field(description="通知正文", min_length=2, max_length=500)
    request_id: str = Field(description="幂等键，同一 request_id 只发送一次", min_length=4)


class EscalateArgs(BaseModel):
    reason: str = Field(description="转人工的原因，要说清卡在哪", min_length=4)
    request_id: str = Field(description="幂等键", min_length=4)


ARG_MODELS: dict[str, type[BaseModel]] = {
    "get_current_weather": WeatherArgs,
    "query_orders_stats": StatsArgs,
    "get_order_detail": OrderArgs,
    "search_docs": SearchDocsArgs,
    "create_ticket": CreateTicketArgs,
    "send_notice": SendNoticeArgs,
    "escalate_to_human": EscalateArgs,
}


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "查询某个城市当前的天气实况，返回温度、天气现象、风力。"
                "仅用于用户问'现在/此刻/当前'的天气。"
                "如果用户问的是未来某天或未来几天的天气预报、或是历史天气，"
                "不要调用这个工具，直接说明你无法获取预报数据。"
                "如果用户没有给出城市，不要猜测，先向用户追问城市名。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名，不带'市'字后缀，例如'杭州'、'北京'。用户只给区县时请换算成所属地级市。",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位。用户未明确指定时使用 celsius。",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_orders_stats",
            "description": (
                "对订单库做只读的聚合统计查询：计数、求和、分组、排序、筛选。"
                "只能执行 SELECT；任何插入、更新、删除都不允许，遇到就拒绝。"
                "如果用户要的是某一笔订单的完整明细（物流节点、收货地址等），"
                "不要调用这个工具，应改用 get_order_detail，并先拿到订单号。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "一条只读的 SELECT 语句，表名必须是 orders。可用列：order_id, city, amount, status, created_at。",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回行数上限，默认 100，最大 1000。",
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_detail",
            "description": (
                "按订单号查询单笔订单的完整明细：状态、金额、收货地址、物流节点。"
                "必须要有订单号才能调用。"
                "如果用户只描述了订单特征（比如'我上周买的那单'）而没有给出订单号，"
                "不要凭空猜测订单号，先用 query_orders_stats 查到订单号，或向用户追问。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，形如 'SO20260001'。区分大小写，必须完整。",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "在三家国内大模型平台（DeepSeek / 智谱 / 阿里百炼）的官方 API 文档中检索片段。"
                "用于查参数含义、计费口径、调用限制、适用范围、平台间差异。"
                "何时不要用：需要精确数值计算（比如算一次调用多少钱）时，"
                "先检索拿到单价再自己算，不要指望文档里有现成答案；"
                "也不要用它查天气、订单这类业务数据 —— 那是另外三个工具的事。"
                "本工具只读，不产生任何副作用，可以放心调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "检索语句。**用文档里可能出现的术语，不要写完整句子**"
                            "（好例子：'deepseek 上下文缓存 计费'；坏例子：'请问 deepseek 的缓存怎么收费呢'）。"
                            "这个语料是技术文档，术语匹配比口语改写有效得多。"
                        ),
                    },
                    "source": {
                        "type": "string",
                        "enum": ["deepseek", "zhipu", "bailian"],
                        "description": (
                            "限定只查某一家平台的文档。"
                            "**只在用户明确点名某家平台时才填**，不确定就留空 —— "
                            "猜错会把正确答案过滤掉，而多召回几条的代价很低。"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回条数，默认 5。最多 20 —— 拿太多会把上下文撑爆。",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": (
                "创建一个客服工单。**这是写操作，不可撤销**。"
                "何时不要用：① 用户只是来问问题、没有要求处理时，不要擅自建单；"
                "② 已经用同一个 request_id 建过单时，绝对不要再建一次；"
                "③ 信息不全（说不清遇到了什么）时先追问，不要拿猜测填工单。"
                "调用前必须已经获得人工确认。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "工单标题，20 字以内为佳"},
                    "body": {"type": "string",
                             "description": "工单正文：用户遇到了什么、期望什么、已经排查过什么"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high"],
                                 "description": "紧急程度，默认 normal"},
                    "request_id": {
                        "type": "string",
                        "description": (
                            "幂等键。**同一个 request_id 只会生效一次**，"
                            "重复调用不会重复建单。建议用「任务摘要+时间戳」生成，"
                            "同一个任务在重试时必须复用同一个 request_id。"
                        ),
                    },
                },
                "required": ["title", "body", "request_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notice",
            "description": (
                "发一条通知（邮件 / 短信 / webhook）。**这是写操作，发出去收不回**。"
                "何时不要用：收件人不明确时不要猜；内容还没确认时不要发。"
                "调用前必须已经获得人工确认。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "enum": ["email", "sms", "webhook"],
                                "description": "通知渠道"},
                    "to": {"type": "string", "description": "接收方，必须是明确给出的，不要猜"},
                    "content": {"type": "string", "description": "通知正文，500 字以内"},
                    "request_id": {"type": "string",
                                   "description": "幂等键，同一 request_id 只发送一次"},
                },
                "required": ["channel", "to", "content", "request_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "把这件事转给人工处理。"
                "**这个工具永远可以调用，不需要额外确认** —— "
                "当你判断信息不足、超出能力范围、或涉及投诉/退款/赔偿时，"
                "应该优先用它，而不是硬着头皮给一个不确定的答案。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string",
                               "description": "转人工的原因，要说清卡在哪、已经查过什么"},
                    "request_id": {"type": "string", "description": "幂等键"},
                },
                "required": ["reason", "request_id"],
            },
        },
    },
]


_WEATHER = {
    "杭州": {"temp_c": 27, "phenomenon": "多云", "wind": "东南风 3 级"},
    "北京": {"temp_c": 22, "phenomenon": "晴", "wind": "西北风 2 级"},
    "上海": {"temp_c": 29, "phenomenon": "阴", "wind": "东风 4 级"},
}

_ORDERS = [
    ("SO20260001", "杭州", 1299.0, "已签收", "2026-09-01", "杭州市西湖区文三路 100 号", "已签收"),
    ("SO20260002", "北京", 349.5, "运输中", "2026-09-03", "北京市海淀区中关村大街 1 号", "已到达北京分拨中心"),
    ("SO20260003", "杭州", 89.0, "已签收", "2026-09-05", "杭州市余杭区未来科技城 8 号", "已签收"),
    ("SO20260004", "上海", 2199.0, "待发货", "2026-09-08", "上海市浦东新区世纪大道 500 号", "仓库备货中"),
    ("SO20260005", "杭州", 459.0, "运输中", "2026-09-10", "杭州市拱墅区运河东路 22 号", "已发出，运输中"),
]


def _build_db() -> sqlite3.Connection:
    """内存 sqlite。今天用它代替真实数据库，验证的是编排而不是数据源。"""
    con = sqlite3.connect(":memory:", check_same_thread=False)
    con.execute(
        "CREATE TABLE orders (order_id TEXT, city TEXT, amount REAL, "
        "status TEXT, created_at TEXT, address TEXT, logistics TEXT)"
    )
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", _ORDERS)
    return con


_DB = _build_db()


def get_current_weather(a: WeatherArgs) -> dict:
    w = _WEATHER.get(a.city)
    if w is None:
        return {"error": f"没有 {a.city} 的天气数据，已知城市：{list(_WEATHER)}"}
    t = w["temp_c"]
    if a.unit == "fahrenheit":
        return {"city": a.city, "temp": round(t * 9 / 5 + 32, 1), "unit": "F",
                "phenomenon": w["phenomenon"], "wind": w["wind"]}
    return {"city": a.city, "temp": t, "unit": "C",
            "phenomenon": w["phenomenon"], "wind": w["wind"]}


def query_orders_stats(a: StatsArgs) -> dict:
    """只放行 SELECT。这是最小权限原则在工具层的落地：模型不该有写库的能力。"""
    sql = a.sql.strip().rstrip(";")
    head = sql.lstrip("(").split()[0].upper() if sql.split() else ""
    if head not in ("SELECT", "WITH"):
        return {"error": f"只允许 SELECT 查询，收到的是 {head or '空语句'}"}
    if "orders" not in sql.lower():
        return {"error": "只允许查询 orders 表"}
    for bad in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "ATTACH", "PRAGMA"):
        if bad in sql.upper():
            return {"error": f"语句包含不允许的关键字 {bad}"}
    try:
        cur = _DB.execute(f"{sql} LIMIT {a.limit}")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except sqlite3.Error as e:
        return {"error": f"SQL 执行失败：{e}"}
    return {"columns": cols, "rows": rows, "count": len(rows)}


def get_order_detail(a: OrderArgs) -> dict:
    cur = _DB.execute(
        "SELECT order_id, city, amount, status, created_at, address, logistics "
        "FROM orders WHERE order_id = ?",
        (a.order_id,),
    )
    row = cur.fetchone()
    if row is None:
        known = [r[0] for r in _DB.execute("SELECT order_id FROM orders")]
        return {"error": f"没有订单号为 {a.order_id} 的订单，库中现有：{known}"}
    cols = ["order_id", "city", "amount", "status", "created_at", "address", "logistics"]
    return dict(zip(cols, row))


_PIPELINE = None


def search_docs(a: SearchDocsArgs) -> dict:
    """在三家平台的官方 API 文档里检索 —— 真实接 W3 的检索链路。"""
    global _PIPELINE
    if _PIPELINE is None:
        try:
            from .retrieval.pipeline import RetrievalPipeline
            _PIPELINE = RetrievalPipeline()
        except Exception as e:  # noqa: BLE001
            return {"error": f"检索链路未就绪：{type(e).__name__}: {e}",
                    "hits": [], "count": 0}

    need = a.top_k if not a.source else min(a.top_k * 4, 50)
    res = _PIPELINE.search(a.query, mode="hybrid", topk=need)
    hits = res.get("hits", [])
    if a.source:
        hits = [h for h in hits if h.get("source") == a.source]
    hits = hits[: a.top_k]

    slim = [{"chunk_id": h["chunk_id"], "source": h.get("source", ""),
             "title": h.get("title", ""), "path": h.get("path", ""),
             "score": h["score"], "text": h.get("text", "")[:800]}
            for h in hits]
    return {"query": a.query, "mode": res.get("mode", "hybrid"),
            "count": len(slim), "hits": slim,
            "latency_ms": res.get("latency_ms", 0.0)}


_IDEMPOTENCY: dict[str, dict] = {}
_ACTIONS_PATH: str | None = None


def _actions_path() -> str:
    """审计日志路径。放在 data/ 下，和向量库、评测数据同级。"""
    global _ACTIONS_PATH
    if _ACTIONS_PATH is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _ACTIONS_PATH = os.path.join(root, "data", "actions.jsonl")
    return _ACTIONS_PATH


def _commit(kind: str, request_id: str, payload: dict) -> dict:
    """落盘 + 记幂等。"""
    if request_id in _IDEMPOTENCY:
        return {"ok": True, "duplicate": True, "skipped": True,
                "result": _IDEMPOTENCY[request_id]}

    record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind,
              "request_id": request_id, **payload}
    os.makedirs(os.path.dirname(_actions_path()), exist_ok=True)
    with open(_actions_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    out = {"ok": True, "duplicate": False, **payload}
    _IDEMPOTENCY[request_id] = out
    return out


def _stable_id(prefix: str, request_id: str) -> str:
    """从 request_id 稳定派生一个业务号。"""
    return prefix + hashlib.md5(request_id.encode("utf-8")).hexdigest()[:8].upper()


def create_ticket(a: CreateTicketArgs) -> dict:
    """创建一个客服工单。会真实落盘到 data/actions.jsonl。"""
    return _commit("create_ticket", a.request_id, {
        "ticket_id": _stable_id("TK", a.request_id),
        "title": a.title, "body": a.body, "priority": a.priority,
    })


def send_notice(a: SendNoticeArgs) -> dict:
    """发一条通知。会真实落盘（不真发，避免把消息发到真实收件人）。"""
    return _commit("send_notice", a.request_id, {
        "channel": a.channel, "to": a.to, "content": a.content,
        "note": "演示环境：只落盘不实际投递",
    })


def escalate_to_human(a: EscalateArgs) -> dict:
    """转人工。永远允许 —— 它是「答不出来」时的第一选择，比硬编一个答案强得多。"""
    return _commit("escalate_to_human", a.request_id, {
        "reason": a.reason, "queue": "human-review",
    })


def reset_actions_for_test() -> None:
    """清空幂等表（仅测试用）。"""
    _IDEMPOTENCY.clear()


REGISTRY: dict[str, Callable[[BaseModel], dict]] = {
    "get_current_weather": get_current_weather,
    "query_orders_stats": query_orders_stats,
    "get_order_detail": get_order_detail,
    "search_docs": search_docs,
    "create_ticket": create_ticket,
    "send_notice": send_notice,
    "escalate_to_human": escalate_to_human,
}


TOOL_RISK: dict[str, tuple[str, bool]] = {
    "get_current_weather": ("read", False),
    "query_orders_stats": ("read", False),
    "get_order_detail": ("read", False),
    "search_docs": ("read", False),
    "create_ticket": ("write", True),
    "send_notice": ("write", True),
    "escalate_to_human": ("escalate", False),
}


def tools_for(levels) -> list[dict]:
    """按风险级别筛出工具 schema 子集 —— **最小权限的第一层**。"""
    keep = set(levels)
    return [t for t in TOOL_SCHEMAS
            if TOOL_RISK.get(t["function"]["name"], ("read", False))[0] in keep]


KNOWN = list(REGISTRY)


def dispatch(name: str, raw_arguments: str) -> tuple[Any, str]:
    """执行一个工具调用，把四种失败都收口成一条人话错误消息。"""
    try:
        raw = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return None, f"参数不是合法 JSON（{e}）。请重新输出，格式必须是 {{\"参数名\": 值}}。"

    if not isinstance(raw, dict):
        return None, f"参数必须是一个 JSON 对象，收到的是 {type(raw).__name__}。"


    model = ARG_MODELS.get(name)
    fn = REGISTRY.get(name)
    if model is None or fn is None:
        return None, f"没有名为 '{name}' 的工具。可用工具只有：{KNOWN}。请从里面选一个。"

    try:
        args = model.model_validate(raw)
    except ValidationError as e:
        return None, f"参数不合法：{e.errors()[0]['msg']}（字段 {e.errors()[0]['loc']}）。请按 schema 重新输出。"

    try:
        return fn(args), ""
    except Exception as e:  # noqa: BLE001
        return None, f"工具执行出错：{type(e).__name__}: {e}。可以换个参数再试。"


TOOL_SYSTEM = """你是一个可以调用工具的助手。

规则：
1. 需要外部数据（天气、订单统计、订单明细）时才调用工具；能用常识直接回答的就不要调用。
2. 一次可以调用多个工具，但只在它们互相独立时。
3. 信息不足（比如没给城市、没给订单号）时，先追问用户，不要猜参数。
4. 拿到工具结果后，用中文自然回答。不要复述 JSON，不要提"工具"两个字。
5. 工具返回了 error 就按提示修正参数重试；重试两次仍失败，就如实告诉用户拿不到数据。"""
