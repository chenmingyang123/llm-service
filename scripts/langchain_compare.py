"""LangChain 三节速览 + 「不用框架我自己怎么实现」对照 —— 9/25 上午那 3 小时。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings  # noqa: E402

QUESTION = "杭州的订单总金额是多少？其中金额最高的一单现在物流到哪了？"


def own_side(s) -> dict:
    """我自己的实现：agent.run_tool_loop。"""
    from app.agent import run_tool_loop

    t0 = time.time()
    try:
        out = run_tool_loop(s, QUESTION, max_turns=4, max_tokens=400)
        return {"ok": out["ok"], "answer": out["answer"][:120], "turns": out["turns"],
                "tool_calls": out["tool_calls_made"], "cost_cny": out["cost_cny"],
                "latency_s": round(time.time() - t0, 3),
                "trace_visible": bool(out["trace"])}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def langchain_side(s) -> dict:
    """LangChain 侧。没装就如实返回未安装，不假装有结果。"""
    try:
        from langchain_core.tools import tool  # noqa: F401
    except ImportError:
        return {"installed": False,
                "hint": "pip install langchain langchain-core（纯新增安装，别 upgrade 已有包）"}

    try:
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool as lc_tool
        from langchain_openai import ChatOpenAI
    except ImportError:
        return {"installed": False, "hint": "还需要 pip install langchain-openai"}

    @lc_tool
    def get_current_weather(city: str) -> str:
        """查询某个城市当前的天气实况。"""
        from app.tools import get_current_weather as fn, WeatherArgs
        return json.dumps(fn(WeatherArgs(city=city)), ensure_ascii=False)

    @lc_tool
    def query_orders_stats(sql: str, limit: int = 100) -> str:
        """对订单库做只读聚合查询，只能 SELECT。"""
        from app.tools import query_orders_stats as fn, StatsArgs
        return json.dumps(fn(StatsArgs(sql=sql, limit=limit)), ensure_ascii=False)

    @lc_tool
    def get_order_detail(order_id: str) -> str:
        """按订单号查询单笔订单完整明细。"""
        from app.tools import get_order_detail as fn, OrderArgs
        return json.dumps(fn(OrderArgs(order_id=order_id)), ensure_ascii=False)

    try:
        llm = ChatOpenAI(model="deepseek-flash", api_key=s.DEEPSEEK_API_KEY,
                         base_url=s.DEEPSEEK_BASE_URL, temperature=0)
        llm_with_tools = llm.bind_tools([get_current_weather, query_orders_stats, get_order_detail])
        t0 = time.time()
        ai = llm_with_tools.invoke([HumanMessage(content=QUESTION)])
        return {"installed": True, "ok": True, "turns": 1,
                "tool_calls": len(getattr(ai, "tool_calls", []) or []),
                "latency_s": round(time.time() - t0, 3),
                "note": "只跑了第一轮（选工具）。完整循环要用 LangGraph 或手写 agent loop。"}
    except Exception as e:  # noqa: BLE001
        return {"installed": True, "ok": False, "error": f"{type(e).__name__}: {e}"}


COMPARISON = [
    ("调用层", "langchain_openai.ChatOpenAI 封装 SDK",
     "app/llm.py，纯标准库 urllib",
     "SDK 帮你处理重试与序列化；自己写省一个依赖、且错误码映射能按你的业务说话"),
    ("结构化输出", "PydanticOutputParser + 重试链",
     "app/completion.py：剥 fence → Pydantic 校验 → 把错误喂回去自修",
     "框架的 parser 通用但报错抽象；自己写能把「上一次错在哪」原话回传，修复成功率高"),
    ("工具调用", "@tool 装饰器自动生成 schema",
     "app/tools.py 手写 schema + Pydantic 参数模型",
     "装饰器省事但描述质量不可控；手写能写清「什么时候不该调用」，这是稳定调用的关键"),
    ("循环编排", "LangGraph 状态图 / AgentExecutor",
     "app/agent.py 手写 while 循环 + max_turns 熔断",
     "框架适合复杂分支；单链路工具调用手写更短、更可控、更好排障"),
    ("成本与可观测", "callback / tracer",
     "cost.py 账本 + X-Cost-CNY 响应头 + trace 结构",
     "框架要额外接；自己写的从第一天就在响应头里"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-langchain", action="store_true", help="同时跑 LangChain 侧")
    ap.add_argument("--skip-own", action="store_true", help="跳过自研侧（省一次调用）")
    a = ap.parse_args()

    s = get_settings()
    own = None if a.skip_own else own_side(s)
    lc = langchain_side(s) if a.with_langchain else {"installed": False, "skipped": True}

    print("=" * 72)
    print(f"对照任务：{QUESTION}")
    print("=" * 72)

    if own:
        print("\n【我的实现】app/agent.py")
        print(json.dumps(own, ensure_ascii=False, indent=2))

    print("\n【LangChain】")
    print(json.dumps(lc, ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print("三节速览对照（面试时被问「为什么不用框架」就用这张表）")
    print("=" * 72)
    for topic, lc_way, my_way, note in COMPARISON:
        print(f"\n■ {topic}")
        print(f"  LangChain：{lc_way}")
        print(f"  我写的  ：{my_way}")
        print(f"  差别    ：{note}")

    print("\n" + "=" * 72)
    print("一句话结论")
    print("=" * 72)
    print("框架解决的是「从零到一」，我这个项目已经过了那个阶段：")
    print("要的是可控、可观测、能报成本。这些恰恰是框架封装掉、出了问题最难查的部分。")
    print("但——如果明天要做多分支、多智能体、带人工确认的复杂流程，就该上 LangGraph，")
    print("别为了「自己写的」而自己写。知道什么时候该用框架，比不用框架更值钱。")


if __name__ == "__main__":
    main()
