"""生成模型价格对比表 —— 这张表本身就是作品。"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pricing import MODEL_PRICES, breakeven, min_calls_for_saving, price_table  # noqa: E402


def render(peak: bool) -> str:
    rows = price_table(peak=peak)
    lines = [
        "# 模型价格对比表",
        "",
        f"- 计价口径：**人民币 / 百万 token**（DeepSeek 官方为美元，按 1 USD = 7.10 CNY 折算）",
        f"- 档位：**{'高峰档（工作日 09–12、14–18 点）' if peak else '闲时'}**"
        f"{'　单价 ×2' if peak else ''}",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| 模型 | 厂商 | 输入（未命中） | 输入（缓存命中） | 输出 | 输出/输入 倍数 | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fmt = lambda v: "免费" if r["free"] else (f"¥{v:.2f}" if v is not None else "未公示")  # noqa: E731
        lines.append(
            f"| `{r['model']}` | {r['provider']} | {fmt(r['input_cny_per_m'])} | "
            f"{fmt(r['cached_cny_per_m'])} | {fmt(r['output_cny_per_m'])} | "
            f"{r['out_in_ratio'] or '—'} | {r['note']} |"
        )

    lines += ["", "## 三件这张表逼你正视的事", "",
              "1. **输出比输入贵得多。** deepseek-flash 输出是输入的 4 倍，v4-pro 是 3 倍。"
              "所以降本第一杠杆从来不是「把 prompt 写短」，而是「让模型少说废话」。",
              "2. **缓存命中价是未命中的 1/50。** 这是最大的单项折扣，"
              "但只在「长前缀 + 高频复用」时兑现（见下面盈亏平衡）。",
              "3. **高峰档翻倍。** 同样的调用，工作日 09–12、14–18 点花两倍的钱。"
              "跑批和全量评测挪到晚上或周末，既便宜又稳（p95 低 85%）。"]

    lines += ["", "## 缓存盈亏平衡（deepseek-flash）", "",
              "| 前缀 token | 动态 token | 复用次数 | 不缓存 | 用缓存 | 省下 | 省幅 |",
              "|---|---|---|---|---|---|---|"]
    for prefix, dynamic, calls in [(500, 100, 100), (2000, 200, 100),
                                   (8000, 200, 100), (8000, 200, 10)]:
        b = breakeven(prefix, dynamic, calls, peak=peak)
        flag = "" if b["eligible_for_cache"] else " ⚠ 前缀过短不缓存"
        lines.append(f"| {prefix} | {dynamic} | {calls} | ¥{b['no_cache_cny']:.4f} | "
                     f"¥{b['with_cache_cny']:.4f} | ¥{b['saved_cny']:.4f} | "
                     f"{b['saved_pct']}%{flag} |")

    lines += ["", "## 反直觉的那个数字", "",
              "| 前缀 token | 想省下 | 需要复用多少次 |",
              "|---|---|---|"]
    for prefix, target in [(500, 1.0), (2000, 1.0), (8000, 1.0), (8000, 100.0)]:
        m = min_calls_for_saving(prefix, target, peak=peak)
        if m["feasible"]:
            lines.append(f"| {prefix} | ¥{target} | **{m['calls_needed']:,} 次** |")
        else:
            lines.append(f"| {prefix} | ¥{target} | 不可行（{m['reason']}） |")
    lines += ["",
              "**读法**：想省 ¥1，前缀 500 token 要复用近 2000 次，8000 token 只要 120 次 —— "
              "**短前缀基本不值得为它重构 prompt**。",
              "",
              "而想省 ¥100，就算是 8000 token 的长前缀也要复用约 1.2 万次。"
              "这说明缓存是**规模化之后才兑现**的收益：早期别为它过度设计，"
              "等业务量上来、前缀稳定了再排，顺序反了就是纯亏。"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak", type=int, default=None, help="1=高峰档 0=闲时；不给按当前时段")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from app.cost import is_peak_hour
    peak = bool(a.peak) if a.peak is not None else is_peak_hour()
    md = render(peak)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"已写入 {a.out}（{'高峰档' if peak else '闲时'}）")
    else:
        print(md)


if __name__ == "__main__":
    main()
