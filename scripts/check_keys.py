# -*- coding: utf-8 -*-
"""check_keys.py —— 零依赖的三家 LLM API 密钥体检脚本（只用标准库）"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path=None):
    """手写一个极简 .env 解析器，避免依赖 python-dotenv。"""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return True


USD_CNY = float(os.environ.get("USD_CNY", "7.10"))

PRICE_USD = {
    "deepseek-flash":    {"in_hit": (0.003, 0.006), "in_miss": (0.15, 0.30), "out": (0.60, 1.20)},
    "deepseek-v4-pro":   {"in_hit": (0.022, 0.044), "in_miss": (0.66, 1.32), "out": (1.98, 3.96)},
    "deepseek-v4-flash": {"in_hit": (0.003, 0.006), "in_miss": (0.15, 0.30), "out": (0.60, 1.20)},
}
FREE_MODELS = {
    "glm-4.7-flash", "glm-4.5-flash", "glm-4.5-air", "glm-4.6v-flash",
    "glm-4-flash", "glm-4-flashx",
}


def is_peak_hour(now=None):
    """DeepSeek 高峰时段：UTC 01:00-04:00 与 06:00-10:00，周一至周五（中国法定节假日除外）。"""
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    h = now.hour
    return (1 <= h < 4) or (6 <= h < 10)


PROVIDERS = [
    {
        "key": "deepseek",
        "name": "DeepSeek（主用）",
        "env_key": "DEEPSEEK_API_KEY",
        "env_url": "DEEPSEEK_BASE_URL",
        "default_url": "https://api.deepseek.com",
        "home": "https://platform.deepseek.com",
        "candidates": ["deepseek-flash", "deepseek-v4-pro"],
        "grep": ("flash", "pro", "deepseek"),
    },
    {
        "key": "zhipu",
        "name": "智谱 GLM（备用 1）",
        "env_key": "ZHIPU_API_KEY",
        "env_url": "ZHIPU_BASE_URL",
        "default_url": "https://open.bigmodel.cn/api/paas/v4",
        "home": "https://open.bigmodel.cn",
        "candidates": ["glm-4.7-flash", "glm-4.5-flash", "glm-4.5-air"],
        "grep": ("glm",),
    },
    {
        "key": "dashscope",
        "name": "阿里百炼 / 通义（备用 2）",
        "env_key": "DASHSCOPE_API_KEY",
        "env_url": "DASHSCOPE_BASE_URL",
        "default_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "home": "https://bailian.console.aliyun.com",
        "candidates": ["qwen-plus", "qwen-turbo", "qwen3-max", "qwen-flash"],
        "grep": ("qwen",),
    },
]


def http(url, api_key, method="GET", payload=None, timeout=60):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + api_key)
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return resp.status, json.loads(body), time.time() - t0
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body), time.time() - t0
        except Exception:
            return e.code, {"raw": body[:400]}, time.time() - t0
    except Exception as e:
        return 0, {"raw": str(e)}, time.time() - t0


def explain(code, body, prov):
    """把错误码翻译成人话 —— 申请当天最容易卡在这几个。"""
    msg = ""
    if isinstance(body, dict):
        msg = (body.get("error") or {}).get("message", "") or body.get("message", "") or body.get("raw", "")
    msg = str(msg)[:200]
    table = {
        401: "密钥无效或已过期 → 回控制台重新生成，注意 key 只显示一次",
        402: "余额不足 → 需要充值（DeepSeek 必须先完成实名认证才能充值）",
        403: "无权限 → 百炼需先「开通服务」；智谱需完成实名认证",
        404: f"地址不对 → 检查 base_url，正确值是 {prov['default_url']}",
        429: "触发限流 → 免费模型并发通常为 1，串行调用即可",
    }
    return f"{msg}｜{table.get(code, '')}" if msg else table.get(code, f"HTTP {code}")


def ledger_path():
    return os.path.join(ROOT, os.environ.get("COST_LEDGER", ".cost_ledger.jsonl"))


def ledger_add(entry):
    with open(ledger_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ledger_sum():
    p = ledger_path()
    if not os.path.exists(p):
        return 0.0, 0, {}
    total, n = 0.0, 0
    per = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            c = e.get("cost_cny") or 0.0
            total += c
            n += 1
            per[e.get("provider", "?")] = per.get(e.get("provider", "?"), 0.0) + c
    return total, n, per


def check_budget():
    limit = float(os.environ.get("COST_LIMIT_CNY", "20"))
    total, n, _ = ledger_sum()
    if total >= limit:
        print(f"\n[中止] 本地累计花费 ¥{total:.4f} 已达闸门 ¥{limit:.2f}（{n} 次调用）")
        print("      这是防烧钱兜底。确认无误后调高 .env 里的 COST_LIMIT_CNY 再跑。")
        sys.exit(2)


def list_models(prov, api_key, base):
    code, body, dt = http(base.rstrip("/") + "/models", api_key, timeout=30)
    if code != 200:
        return None, code, body, dt
    ids = [m.get("id", "") for m in body.get("data", []) if isinstance(m, dict)]
    keys = prov["grep"]
    picked = [i for i in ids if any(k in i.lower() for k in keys)]
    return (picked or ids), code, body, dt


def try_chat(prov, api_key, base, model, thinking="disabled"):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a terse assistant. Reply in one short sentence."},
            {"role": "user", "content": "用一句话说明什么是 RAG。"},
        ],
        "max_tokens": 300,
        "stream": False,
    }
    if prov["key"] == "deepseek":
        payload["extra_body"] = {"thinking": {"type": thinking}}

    code, body, dt = http(base.rstrip("/") + "/chat/completions", api_key, "POST", payload, timeout=120)
    if code != 200:
        return None, code, body, dt

    usage = body.get("usage") or {}
    tin = usage.get("prompt_tokens", 0)
    tout = usage.get("completion_tokens", 0)
    answer = ""
    try:
        answer = body["choices"][0]["message"].get("content", "")
    except Exception:
        pass
    detail = usage.get("prompt_tokens_details") or {}
    hit = detail.get("cached_tokens", 0) if isinstance(detail, dict) else 0
    return {"in": tin, "out": tout, "hit": hit, "answer": answer.strip()}, code, body, dt


def estimate_cny(model, tin, tout, hit=0):
    if model in FREE_MODELS:
        return 0.0, "免费额度"
    p = PRICE_USD.get(model)
    if not p:
        return None, "未收录价格"
    idx = 1 if is_peak_hour() else 0
    miss = max(tin - hit, 0)
    usd = (hit * p["in_hit"][idx] + miss * p["in_miss"][idx] + tout * p["out"][idx]) / 1_000_000
    tag = "高峰" if idx else "空闲"
    return usd * USD_CNY, tag


def run_provider(prov, only_models=False):
    api_key = os.environ.get(prov["env_key"], "").strip()
    base = os.environ.get(prov["env_url"], "").strip() or prov["default_url"]

    print("=" * 68)
    print(f"{prov['name']}")
    print(f"  base_url : {base}")

    if not api_key:
        print("  [跳过] .env 里没有填 " + prov["env_key"])
        print(f"         去 {prov['home']} 申请后填进来")
        return False

    print(f"  key      : {api_key[:8]}...{api_key[-4:]}（已加载）")

    ids, code, body, dt = list_models(prov, api_key, base)
    if ids is None:
        print(f"  [失败] GET /models → {code}　{explain(code, body, prov)}")
        return False
    print(f"  模型列表 : {len(ids)} 个可用（{dt:.2f}s）")
    print("            " + ", ".join(ids[:12]) + (" ..." if len(ids) > 12 else ""))

    if only_models:
        return True

    ok = False
    for model in prov["candidates"]:
        if model not in ids:
            continue
        res, code, body, dt = try_chat(prov, api_key, base, model,
                                       os.environ.get("DEEPSEEK_THINKING", "disabled"))
        if res is None:
            print(f"  [调用失败] {model} → {code}　{explain(code, body, prov)}")
            continue
        cny, tag = estimate_cny(model, res["in"], res["out"], res["hit"])
        cost_txt = "¥%.5f" % cny if cny is not None else "-"
        print(f"  [OK] {model}")
        print(f"       延迟 {dt:.2f}s　输入 {res['in']} tok（缓存命中 {res['hit']}）　输出 {res['out']} tok")
        print(f"       本次成本 {cost_txt}　计价档 {tag}")
        print(f"       回答：{res['answer'][:80]}")
        ledger_add({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "provider": prov["key"], "model": model,
            "in": res["in"], "out": res["out"], "hit": res["hit"],
            "latency_s": round(dt, 3), "cost_cny": cny,
        })
        ok = True
        break

    if not ok:
        matched = [m for m in prov["candidates"] if m in ids]
        if not matched:
            print("  [注意] 预设的候选模型都不在列表里 —— 模型名已更新，请从上面的列表里手动挑一个")
            print("         挑好后改 scripts/check_keys.py 里 PROVIDERS 的 candidates")
        return False
    return True


def main():
    args = sys.argv[1:]
    if not load_env():
        print("没有找到 .env。请先：cp .env.example .env  （Windows: copy .env.example .env）")
        print("然后把申请到的三个 key 填进去。")

    if "--budget" in args:
        total, n, per = ledger_sum()
        print(f"本地累计：¥{total:.4f}　{n} 次调用　闸门 ¥{os.environ.get('COST_LIMIT_CNY','20')}")
        for k, v in per.items():
            print(f"   {k:<12} ¥{v:.4f}")
        return

    check_budget()

    only = None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1] if i + 1 < len(args) else None
    only_models = "--models" in args

    print(f"运行时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　"
          f"当前计价档：{'高峰' if is_peak_hour() else '空闲'}　汇率 {USD_CNY}")
    print("提示：DeepSeek 高峰 = 北京时间周一至周五 09:00-12:00 与 14:00-18:00，其余时间半价。\n")

    results = []
    for prov in PROVIDERS:
        if only and prov["key"] != only:
            continue
        results.append((prov["name"], run_provider(prov, only_models=only_models)))

    print("\n" + "=" * 68)
    for name, ok in results:
        print(f"  {'通过' if ok else '未通过'}　{name}")

    total, n, per = ledger_sum()
    print(f"\n本地累计花费：¥{total:.4f}（{n} 次）　闸门 ¥{os.environ.get('COST_LIMIT_CNY','20')}")
    if results and all(ok for _, ok in results):
        print("\n下一步：把主用模型名记下来，9/20 的第 1 小时到此完成。")
    else:
        print("\n有项目未通过 —— 按上面的提示回控制台处理，然后重跑本脚本。")


if __name__ == "__main__":
    main()
