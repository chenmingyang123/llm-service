"""llm-service —— W1 的 FastAPI 骨架。"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .agent import BudgetExceeded as AgentBudgetExceeded
from .agent import run_prompt_style, run_tool_loop
from .batch import run_batch
from .completion import BudgetExceeded, structured_chat
from .config import get_settings
from .confirm import ConfirmationRegistry
from .cost import cache_stats, summarize
from .cost_header import CostHeaderMiddleware
from .llm import LLMError
from .observe import Tracer, langfuse_exporter, make_trace_hook
from .providers import all_provider_status
from .resilience import FALLBACK_CHAIN, call_with_fallback
from .pricing import MODEL_PRICES, breakeven, min_calls_for_saving, price_table
from .rag import AskBatchIn, AskIn, AskOut, AskPipeline
from .rag.abstain import DEFAULT_MIN_COSINE
from .react import BudgetExceeded as ReactBudgetExceeded
from .react import run_react
from .retrieval.schemas import MODES, RetrieveIn, RetrieveOut
from .schemas import (AgentIn, AgentOut, BatchIn, ChatIn, ChatOut, ConfirmIn,
                      ConfirmOut, HealthOut, ReactIn, ReactOut, ReadyOut)
from .streaming import iter_chat_events, pump_to_sse, pump_worker_to_sse
from .tools import TOOL_SCHEMAS
from .traffic import TrafficLogMiddleware
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("llm-service")

START_TIME = time.time()
settings = get_settings()

CONFIRMATIONS = ConfirmationRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting %s v%s env=%s", settings.SERVICE_NAME, settings.VERSION, settings.ENV)
    yield
    logger.info("shutting down")


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.VERSION,
    description="LLM 接入层：多 provider、可降级、能报成本。6 周冲刺 W1 项目。",
    lifespan=lifespan,
)


class RequestIdMiddleware:
    """纯 ASGI 中间件，不依赖 Starlette 的高层 API，升级不易碎。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        rid = scope.get("headers") and dict(scope["headers"]).get(b"x-request-id")
        rid = rid.decode() if rid else uuid.uuid4().hex[:16]
        scope["request_id"] = rid

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(CostHeaderMiddleware)
app.add_middleware(TrafficLogMiddleware)
app.add_middleware(RequestIdMiddleware)


@app.get("/", summary="服务元信息")
def root():
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "ready": "/health/ready",
    }


@app.get("/health", response_model=HealthOut, summary="存活探针（不查外部依赖）")
def health():
    """K8s liveness 用的那种：进程活着就 200，不碰网络，永远不会因外部故障而假死。"""
    return HealthOut(
        status="ok",
        service=settings.SERVICE_NAME,
        version=settings.VERSION,
        env=settings.ENV,
        uptime_s=round(time.time() - START_TIME, 3),
        checked_at=datetime.now().isoformat(timespec="seconds"),
    )


@app.get("/health/ready", response_model=ReadyOut, summary="就绪探针（查 provider 与成本账本）")
def ready(ping: bool = Query(False, description="true 时真实探活各家 /models（只列模型，不产生 token 费用）")):
    providers = all_provider_status(settings, do_ping=ping)

    configured = [p for p in providers if p["configured"]]
    if not configured:
        status = "not_ready"
        detail = "没有任何 provider 配置了 API key，检查 .env"
    elif ping:
        reachable = [p for p in configured if p["status"] == "ready"]
        if not reachable:
            status = "not_ready"
            detail = "所有已配置的 provider 都连不上"
        elif len(reachable) < len(configured):
            status = "degraded"
            detail = f"{len(reachable)}/{len(configured)} 家可用，降级链路已就绪"
        else:
            status = "ready"
            detail = f"{len(reachable)}/{len(configured)} 家可用"
    else:
        status = "ready"
        detail = f"{len(configured)}/{len(providers)} 家已配置（未探活，加 ?ping=1 真实探测）"

    cost = summarize(settings)
    if cost["over_limit"]:
        status = "degraded" if status == "ready" else status
        detail = f"本地成本账本已达闸门 ¥{cost['limit_cny']}，禁止继续调用"

    return ReadyOut(
        status=status,
        service=settings.SERVICE_NAME,
        version=settings.VERSION,
        uptime_s=round(time.time() - START_TIME, 3),
        providers=providers,
        cost=cost,
        checked_at=datetime.now().isoformat(timespec="seconds"),
        probed=ping,
        detail=detail,
    )


@app.post("/chat", response_model=ChatOut, summary="结构化输出对话（含校验与修复重试）")
def chat(body: ChatIn):
    """一次调用返回：结构化结果 + 用了几次修复 + 这次花了多少钱、多快。"""
    try:
        return structured_chat(
            settings,
            message=body.message,
            system=body.system,
            role=body.role,
            examples=[(e.input, e.output) for e in body.examples],
            use_xml=body.use_xml,
            thinking=body.thinking,
            max_tokens=body.max_tokens,
            provider=body.provider,
            model=body.model,
            retries=body.retries,
        )
    except BudgetExceeded as e:
        raise HTTPException(status_code=429, detail=str(e)) from None
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None


@app.get("/agent/tools", summary="当前可用的工具 schema")
def agent_tools():
    """把给模型看的那份 schema 原样返回。改描述之前先看看现在长什么样。"""
    return {"count": len(TOOL_SCHEMAS), "tools": TOOL_SCHEMAS}


@app.post("/agent", response_model=AgentOut, summary="工具调用循环（原生 function calling）")
def agent(body: AgentIn):
    """一次请求返回：最终答案 + 完整调用轨迹 + 轮数 + 成本 + 延迟。"""
    try:
        return run_tool_loop(
            settings,
            question=body.question,
            max_turns=body.max_turns,
            tool_choice=body.tool_choice,
            provider=body.provider,
            model=body.model,
            max_tokens=body.max_tokens,
        )
    except AgentBudgetExceeded as e:
        raise HTTPException(status_code=429, detail=str(e)) from None
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None


@app.post("/agent/prompt-style", summary="提示词版工具调用（不依赖原生 function calling）")
def agent_prompt_style(body: AgentIn):
    """教程 10.2 的路径：把工具描述写进系统提示，让模型输出 XML，你解析后执行。"""
    try:
        return run_prompt_style(
            settings,
            question=body.question,
            provider=body.provider,
            model=body.model,
            max_tokens=body.max_tokens,
        )
    except AgentBudgetExceeded as e:
        raise HTTPException(status_code=429, detail=str(e)) from None
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None


@app.post("/agent/react", response_model=ReactOut,
          summary="手写 ReAct：纯文本协议的多步工具循环（不依赖原生 function calling）")
def agent_react(body: ReactIn):
    """与 /agent 的区别不是「多了一个思考」，是**协议换了**。"""
    tracer = None
    if body.trace:
        exp = langfuse_exporter()
        tracer = Tracer(name="agent.react", task=body.question,
                        exporters=[exp] if exp else [])

    try:
        out = run_react(
            settings,
            task=body.question,
            max_steps=body.max_steps,
            provider=body.provider,
            model=body.model,
            max_tokens=body.max_tokens,
            on_event=make_trace_hook(tracer, task=body.question) if tracer else None,
        )
    except ReactBudgetExceeded as e:
        raise HTTPException(status_code=429, detail=str(e)) from None
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None

    if tracer is not None:
        tracer.patch_root(
            cost_cny=out.get("cost_cny"),
            latency_s=out.get("latency_s"),
            usage=out.get("usage"),
            stop_reason=out.get("stop_reason"),
            ok=out.get("ok"),
        )
        out["trace_id"] = tracer.trace_id
    return out


@app.post("/chat/stream", summary="SSE 流式对话")
async def chat_stream(body: ChatIn, request: Request):
    """逐 token 推送。事件类型：delta / usage / done / error。"""
    def factory():
        return iter_chat_events(
            settings,
            [{"role": "user", "content": body.message}],
            system=body.system,
            provider=body.provider,
            model=body.model,
            thinking=body.thinking,
            max_tokens=body.max_tokens,
        )

    return StreamingResponse(
        pump_to_sse(factory, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/agent/stream", summary="SSE 流式工具调用（看得见它在查什么 + 人工确认点）")
async def agent_stream(body: AgentIn, request: Request):
    """把每一轮的 tool_start / tool_done 实时推出来，最后给 summary。"""
    confirmed: set[str] = set()

    tracer = None
    hook = None
    if body.trace:
        exp = langfuse_exporter()
        tracer = Tracer(name="agent.tool_loop", task=body.question,
                        exporters=[exp] if exp else [])
        hook = make_trace_hook(tracer, task=body.question)

    def worker(emit):
        def fanout(ev):
            """一个事件、两个消费者：SSE 给客户端看，trace 给事后复盘看。"""
            if ev is not None and hook is not None:
                hook(ev)
            emit(ev)

        try:
            if tracer is not None:
                emit({"type": "trace_started", "trace_id": tracer.trace_id})
            res = run_tool_loop(
                settings, body.question,
                max_turns=body.max_turns, tool_choice=body.tool_choice,
                provider=body.provider, model=body.model, max_tokens=body.max_tokens,
                on_event=fanout,
                allowed_tools=[t["function"]["name"] for t in TOOL_SCHEMAS],
                confirmed=confirmed,
                confirm_registry=CONFIRMATIONS,
            )
            if tracer is not None:
                tracer.patch_root(
                    cost_cny=res.get("cost_cny"),
                    latency_s=res.get("latency_s"),
                    usage=res.get("usage"),
                    turns=res.get("turns"),
                    ok=res.get("ok"),
                )
            emit({"type": "summary", "ok": res["ok"], "turns": res["turns"],
                  "tool_calls_made": res["tool_calls_made"], "cost_cny": res["cost_cny"],
                  "latency_s": res["latency_s"], "trace": res["trace"],
                  "trace_id": tracer.trace_id if tracer else ""})
        except Exception as e:  # noqa: BLE001
            if tracer is not None:
                tracer.finish_root(error=f"{type(e).__name__}: {e}", status="error")
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            emit(None)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    if tracer is not None:
        headers["X-Trace-Id"] = tracer.trace_id
    return StreamingResponse(
        pump_worker_to_sse(worker, request),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/agent/confirm", response_model=ConfirmOut,
          summary="人工确认回执（配合 /agent/stream 的 need_confirm 事件）")
def agent_confirm(body: ConfirmIn):
    """把人对某个待确认动作的决定送回服务端，解除 /agent/stream 的暂停。"""
    found = CONFIRMATIONS.resolve(body.confirm_id, body.approve)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"确认请求 {body.confirm_id} 不存在或已过期（可能已被处理或超时）")
    return ConfirmOut(ok=True, confirm_id=body.confirm_id, approved=body.approve,
                      detail="已收到决定" if body.approve else "已拒绝，不会产生副作用")


@app.post("/chat/batch", summary="并发跑一批 prompt（信号量限流）")
async def chat_batch(body: BatchIn):
    """返回逐条结果 + 整体统计。最有价值的数字是 speedup（串行耗时 / 墙钟时间）。"""
    return await run_batch(
        settings,
        body.prompts,
        system=body.system,
        concurrency=body.concurrency,
        use_fallback=body.use_fallback,
        provider=body.provider,
        max_tokens=body.max_tokens,
    )


@app.post("/chat/fallback", summary="带降级的单次调用")
def chat_fallback(body: ChatIn):
    """按 FALLBACK_CHAIN 依次尝试。返回体里的 attempts 是排障依据："""
    try:
        return call_with_fallback(
            settings,
            [{"role": "user", "content": body.message}],
            system=body.system,
            model=body.model,
            thinking=body.thinking,
            max_tokens=body.max_tokens,
        )
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None


@app.get("/resilience/chain", summary="当前降级链与各 provider 是否有 key")
def resilience_chain():
    from .resilience import _has_key
    return {"chain": FALLBACK_CHAIN,
            "configured": {p: _has_key(settings, p) for p in FALLBACK_CHAIN}}


_PIPE = None


def _pipeline():
    """拿到检索管线（进程内单例）。"""
    global _PIPE
    if _PIPE is None:
        from .retrieval import BM25Index, RetrievalPipeline, get_store
        store = get_store()
        _PIPE = RetrievalPipeline(store=store,
                                  bm25=BM25Index([r["text"] for r in store.rows]),
                                  backend="auto")
    return _PIPE


@app.post("/retrieve", response_model=RetrieveOut, summary="检索：向量 / BM25 / 混合 / 混合+重排")
def retrieve(body: RetrieveIn):
    """四种模式可切换 —— 留着 baseline 是为了让改进可被证明。"""
    if body.mode not in MODES:
        raise HTTPException(400, f"未知模式：{body.mode}，可选 {' / '.join(MODES)}")
    try:
        res = _pipeline().search(body.query, mode=body.mode,
                                 topk=body.topk, pool=body.pool)
    except RuntimeError as e:
        raise HTTPException(503, f"检索不可用：{e}") from e
    except LLMError as e:
        raise HTTPException(502, f"重排调用失败：{e}") from e
    if not body.with_text:
        for h in res["hits"]:
            h["text"] = ""
    return RetrieveOut(ok=True, **res)


_ASK = None


def _ask() -> AskPipeline:
    """生成端单例，复用检索管线（库只加载一次）。"""
    global _ASK
    if _ASK is None:
        _ASK = AskPipeline(_pipeline())
    return _ASK


@app.post("/ask", response_model=AskOut, summary="RAG 问答：检索 → 生成 → 带引用，且敢说不知道")
def ask(body: AskIn):
    """完整 RAG 链路。"""
    if body.mode not in MODES:
        raise HTTPException(400, f"未知模式：{body.mode}，可选 {' / '.join(MODES)}")
    res = _ask().ask(body.query, mode=body.mode, topk=body.topk,
                     pool=body.pool, min_score=body.min_score)
    if not body.with_context:
        res.pop("context", None)
    return AskOut(**res)


@app.post("/ask/batch", summary="并发跑一批问答（评测与压测用）")
async def ask_batch(body: AskBatchIn):
    """一批问题并发问。"""
    import asyncio

    sem = asyncio.Semaphore(body.concurrency)
    pipe = _ask()

    def one(q: str) -> dict:
        return pipe.ask(q, mode=body.mode, topk=body.topk, pool=body.pool)

    async def guarded(q: str) -> dict:
        async with sem:
            return await asyncio.to_thread(one, q)

    t0 = time.time()
    results = await asyncio.gather(*[guarded(q) for q in body.queries])
    return {"ok": True, "n": len(results),
            "results": [AskOut(**r) for r in results],
            "wall_s": round(time.time() - t0, 2)}


@app.get("/ask/gate", summary="拒答阈值是多少、它是怎么定出来的")
def ask_gate():
    """阈值不是拍脑袋定的，这里把校准依据摊开。"""
    return {
        "min_cosine": DEFAULT_MIN_COSINE,
        "calibrated_on": "29 条 golden set + 5 条域外问题，2026-09-25 实测",
        "in_domain_min": 0.575,
        "out_of_domain_range": [0.236, 0.397],
        "cannot_block": ("域内但答错 —— 命中组 p50 0.775 vs 未命中组 p50 0.749，"
                         "余弦在高度同质化的 API 文档语料上分不开这两者"),
        "second_gate": "模型输出 NO_ANSWER 行，兜住第一道闸门漏掉的",
    }


@app.get("/retrieve/modes", summary="四种检索模式分别在解决什么")
def retrieve_modes():
    """把这个端点当文档看 —— 它解释了为什么要有四种，而不是一种。"""
    return {
        "modes": [
            {"name": "vector", "desc": "纯向量。语义强，对精确术语（SKILL.md / 200万次）弱"},
            {"name": "bm25", "desc": "纯关键词。精确术语强，换个说法就找不到"},
            {"name": "hybrid", "desc": "两路 RRF 融合。默认方案"},
            {"name": "hybrid_rerank", "desc": "融合后再过一次 LLM 精排，慢且花钱，需评测证明值得"},
        ],
        "default": "hybrid",
    }


@app.get("/cost", summary="成本账本汇总 + 缓存命中统计")
def cost_summary():
    """total/remaining 是花了多少还剩多少；cache 是命中率与折算成钱的节省额。"""
    return {"ledger": summarize(settings), "cache": cache_stats(settings)}


@app.get("/cost/models", summary="模型价格对比表（本身就是作品）")
def cost_models(peak: bool | None = Query(None, description="true=按高峰档算；不给=按当前时段")):
    rows = price_table(peak=peak)
    return {"peak": peak, "cny_per_million_tokens": True, "models": rows}


@app.get("/cost/breakeven", summary="Prompt 缓存的盈亏平衡")
def cost_breakeven(
    prefix: int = Query(2000, ge=0, description="固定前缀 token 数"),
    dynamic: int = Query(200, ge=0, description="每次变动部分的 token 数"),
    calls: int = Query(100, ge=1, description="复用次数"),
    model: str = Query("deepseek-flash"),
    peak: bool | None = Query(None),
):
    """同样 N 次调用，缓存能省多少、省几个百分点。"""
    try:
        return breakeven(prefix, dynamic, calls, model=model, peak=peak)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


@app.get("/cost/min-calls", summary="想省下 X 元，前缀得复用多少次")
def cost_min_calls(
    prefix: int = Query(2000, ge=1),
    target: float = Query(1.0, gt=0, description="想省下多少人民币"),
    model: str = Query("deepseek-flash"),
    peak: bool | None = Query(None),
):
    """这个问题的答案通常很反直觉 —— 算出来你就知道该不该为缓存重构 prompt。"""
    try:
        return min_calls_for_saving(prefix, target, model=model, peak=peak)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


@app.exception_handler(Exception)
async def unhandled(request, exc):
    """兜底：别把栈追踪裸奔给调用方，但日志里要留全。"""
    logger.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)[:200]})
