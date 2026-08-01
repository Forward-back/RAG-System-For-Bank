# Safety Module — 三层事实核查架构

## 设计原则

**不能用生成模型审核自己的输出。** 如果 Generator 是 DeepSeek，那 FactChecker 就不能只有 DeepSeek。

业界 2025-2026 年的共识是**多层次防御**：不同模型、不同机制、层层递进。本模块实现了三层验证：

```
生成答案
    │
    ▼
┌──────────────────────────────────────────────┐
│ Layer 1: NumericGuard   规则引擎  0ms   0 API │
│   - 数值 vs SQL 结果比对                        │
│   - 绝对化表述检测（"免费""无"）                    │
│   - 孤立数字检测（答案中的数字是否在原文中）            │
└──────────────────────────────────────────────┘
    │ 全部通过? → 跳过 L3
    │ 有标记?   → 继续 L2
    ▼
┌──────────────────────────────────────────────┐
│ Layer 2: NLIVerifier    Cross-encoder  <100ms │
│   模型: bge-reranker-v2-m3 (本地, ≠ DeepSeek)  │
│   - 拆解答案为原子 claims（正则分句，不用 LLM）       │
│   - 每条 claim 对每个 source doc 打分              │
│   - 最高分 < 阈值 → 标记为 uncertain              │
└──────────────────────────────────────────────┘
    │ unsupported_ratio ≤ 0.3 → 替换 uncertain claims, 返回
    │ unsupported_ratio > 0.3 → 继续 L3
    ▼
┌──────────────────────────────────────────────┐
│ Layer 3: FactChecker    LLM DeepSeek  ~500ms  │
│   仅在 L1+L2 都标记了大量问题时触发                 │
│   - LLM 深度语义推理                              │
│   - 多句子蕴含关系判断                              │
│   - 整个答案不可靠 → 全量替换为转人工                   │
└──────────────────────────────────────────────┘
```

## 为什么这样设计

| 决策 | 理由 |
|------|------|
| L1 用纯规则 | 数值幻觉是银行场景最危险的错误。规则 100% 确定、零延迟、零成本 |
| L2 用 bge-reranker | 和 Generator (DeepSeek) 是不同的模型架构和参数，打破自查循环。且复用已有实例，不增加内存 |
| L2 用正则分句 | FactScore 论文表明句子粒度足以验证；用 LLM 分句反而引入新的不确定性和延迟 |
| L3 仅在必要时调用 | FAQ 查询大多在 L1+L2 解决，不产生额外 API 调用 |

## 配置

```bash
# .env
ENABLE_NUMERIC_GUARD=true     # Layer 1: 规则引擎
ENABLE_NLI_VERIFIER=true      # Layer 2: Cross-encoder 验证
ENABLE_FACT_CHECK=true        # Layer 3: LLM 兜底 (仅在 L1+L2 标记问题时触发)
```

## 延迟预期

| 场景 | L1 | L2 | L3 | 总延迟 |
|------|----|----|-----|--------|
| FAQ 类查询（L1+L2 通过） | 0ms | <100ms | — | <100ms |
| 少量不确定 claims（L2 改写） | 0ms | <100ms | — | <100ms |
| 大量标记问题（触发 L3） | 0ms | <100ms | ~500ms | ~600ms |

## 文件结构

```
src/safety/
├── README.md              # 本文档
├── __init__.py             # 模块导出
├── numeric_guard.py        # Layer 1: 规则引擎
├── nli_verifier.py         # Layer 2: Cross-encoder 验证
└── fact_checker.py         # Layer 3: LLM 深度复核
```

## 参考资料

- SelfCheckGPT (Manakul et al., EMNLP 2023): 多采样一致性检测幻觉
- FactScore (Min et al., 2023): 原子事实分解 + 知识库验证
- HHEM-2.1 + MiniCheck-Flan-T5 (2025): 双 NLI ensemble 等价于 Claude Sonnet 4.6，成本 1/250
- BGE-reranker-v2-m3 (BAAI): 多语言 cross-encoder，支持中英文 zero-shot NLI
