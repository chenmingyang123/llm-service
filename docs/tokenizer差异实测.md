# tokenizer 差异实测

- 样本文本：368 字符（中英混排 + 代码块 + 数字）
- 数据来源：各家返回的 `usage.prompt_tokens`，不是本地估算

| provider | 模型 | prompt tokens | 字符/token | 输入单价 ¥/百万 | 千次调用输入成本 |
|---|---|---|---|---|---|
| deepseek | `deepseek-flash` | 136 | 2.71 | 1.065 | ¥0.1448 |
| zhipu | `glm-4.7-flash` | 140 | 2.63 | 0.0 | ¥0.0 |
| dashscope | `qwen-plus` | 151 | 2.44 | 0.8 | ¥0.1208 |

**最大差异：151 vs 136，相差 11.0%。**

这意味着按一家估的成本，换到另一家会偏这么多 —— 跨 provider 做成本对比时必须用各家自己的 token 数，不能用一个数折算。