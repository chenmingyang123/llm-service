# -*- coding: utf-8 -*-
"""把 golden set 从 29 条扩到 50 条，加上计划里点名的三类难例。"""
import io
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "data", "eval", "golden.jsonl")
BAK = os.path.join(ROOT, "data", "eval", "golden_v1_29.jsonl")

NEW = [
    ("h01", "DeepSeek 的硬盘缓存是怎么触发的，百炼的免费额度用完即停又是什么表现",
     "09ef360777ad-0001", ["09ef360777ad-0001", "f64fae639142-0003"], "deepseek", "multi", ["硬盘缓存"]),
    ("h02", "智谱的速率限制和百炼的限流条件，分别是按什么维度限制的",
     "afb23b5c7d34-0000", ["afb23b5c7d34-0000", "24154652d880-0054"], "zhipu", "multi", ["速率限制"]),
    ("h03", "百炼的免费额度有效期是多久，智谱的套餐额度耗尽后要等多久恢复",
     "f64fae639142-0000", ["f64fae639142-0000", "c3089cfb6d10-0003"], "bailian", "multi", ["90 天"]),
    ("h04", "开启百炼的免费额度用完即停后，额度耗尽会返回什么",
     "f64fae639142-0003", None, "bailian", "boundary", ["403"]),
    ("h05", "什么情况下 DeepSeek 的硬盘缓存不会命中",
     "09ef360777ad-0001", None, "deepseek", "boundary", ["前缀"]),
    ("h06", "百炼的免费额度有效期从哪天开始算",
     "f64fae639142-0000", None, "bailian", "boundary", ["90 天"]),
    ("h07", "百炼的文档解析服务里，哪一档不保留原始图片",
     "980579feb312-0011", None, "bailian", "boundary", ["不保留图片"]),
    ("h08", "免费额度用完之后，百炼和智谱的处理方式有什么不同",
     "f64fae639142-0003", ["f64fae639142-0003", "c3089cfb6d10-0003"], "bailian", "conflict", ["停止响应"]),
    ("h09", "百炼的限流除了 RPM 和 TPM，还可能按什么粒度执行",
     "24154652d880-0001", None, "bailian", "conflict", ["RPS"]),
    ("h10", "智谱 Batch 调用里 GLM-4-Flash 的额度是多少次",
     "f05c8ec30d12-0002", None, "zhipu", "term", ["1000 万次"]),
    ("h11", "智谱为什么要对 API 调用做速率限制",
     "afb23b5c7d34-0000", None, "zhipu", "semantic", ["稳定性"]),
    ("h12", "怎么避免百炼账户欠费停服",
     "661882d5cf94-0002", None, "bailian", "semantic", ["费用告警"]),
    ("h13", "百炼支持用哪些方式充值",
     "f64fae639142-0006", None, "bailian", "term", ["支付宝"]),
    ("h14", "在哪里能查到百炼各模型还剩多少免费额度",
     "f64fae639142-0002", None, "bailian", "semantic", ["余量"]),
    ("h15", "不同用户在 DeepSeek 的缓存会不会互相看到",
     "09ef360777ad-0002", None, "deepseek", "semantic", ["独立"]),
    ("h16", "智谱模型访问量过大时会得到什么提示",
     "afb23b5c7d34-0004", None, "zhipu", "term", ["访问量过大"]),
    ("h17", "触发百炼限流后，第一步应该做什么",
     "24154652d880-0001", None, "bailian", "semantic", ["判断"]),
    ("h18", "DeepSeek 硬盘缓存上线后要不要改代码",
     "3c38c79b63bb-0000", None, "deepseek", "boundary", ["无需修改代码"]),
    ("h19", "百炼千问模型的限流条件是超出什么就触发",
     "24154652d880-0054", None, "bailian", "term", ["限流条件"]),
    ("h20", "智谱套餐额度耗尽后，需要等多久才能恢复",
     "c3089cfb6d10-0003", None, "zhipu", "term", ["5 小时"]),
    ("h21", "百炼新人免费额度是怎么发放的",
     "f64fae639142-0000", None, "bailian", "semantic", ["自动"]),
]


def main():
    if not os.path.exists(BAK):
        shutil.copy(GOLD, BAK)
        print("已备份旧版 →", os.path.basename(BAK))

    old = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    have = {x["qid"] for x in old}
    add = []
    for qid, q, chunk, chunks, src, typ, kw in NEW:
        if qid in have:
            continue
        row = {"qid": qid, "question": q, "expect_chunk": chunk,
               "expect_source": src, "type": typ, "expect_any": kw}
        if chunks:
            row["expect_chunks"] = chunks
        add.append(row)

    with open(GOLD, "w", encoding="utf-8") as f:
        for r in old + add:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"golden set {len(old)} → {len(old) + len(add)} 条（新增 {len(add)}）")


if __name__ == "__main__":
    main()
