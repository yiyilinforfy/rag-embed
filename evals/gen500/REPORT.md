# 检索质量报告（400 题）

只测检索。题集：`evals/gen500/dataset.json`。

## 对照总表

| 轮次 | 配置 | JSON |
|------|------|------|
| A 基线 | hybrid，无路由 | `run-hybrid.json` |
| B 路由 | + `ROUTE_MODE=rules` | `run-hybrid-routed-hybrid.json` |
| C 坏扩写 | B + `QUERY_EXPAND_MODE=llm`（qwen3:1.7b 改坏 query） | `run-hybrid-expand-hybrid.json` |
| **D 规则扩写** | B + `QUERY_EXPAND_MODE=rules` + `RERANK_MIN_SCORE=0.02` + 本地 embed/DB | `run-hybrid-expand-rules-hybrid.json` |

| 指标 | A | B | C | **D** |
|------|--:|--:|--:|------:|
| hit@5 | 0.960 | 0.968 | 0.677 | **0.974** |
| hit@1 | 0.824 | 0.833 | 0.501 | **0.833** |
| context_hit | 0.902 | 0.916 | 0.622 | **0.951** |
| OOS blocked | 0.962 | 0.962 | 0.943 | **0.962** |
| 误拒答 | — | 0 | 0 | **0** |
| 均耗时 | 3.5s | 19s | 0.77s | **0.79s** |

分题型 hit@5 / context_hit（B → D）：

| 题型 | B hit@5 | D hit@5 | B context_hit | D context_hit |
|------|--------:|--------:|--------------:|--------------:|
| direct | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_term | 0.991 | 0.991 | 0.991 | 0.991 |
| english | 0.970 | 0.970 | 0.939 | **0.970** |
| paraphrase | 0.902 | **0.984** | 0.770 | **0.984** |
| multi_doc | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_turn | **0.938** | 0.875 | **0.750** | 0.708 |
| out_of_scope | blocked 0.962 | 0.962 | — | — |

## D 轮结论

1. **规则口语扩写 + 略降 rerank 门闸**后，大盘优于仅路由的 B；**paraphrase 是最大收益**（context_hit 0.770→0.984，61 题无一门闸清空）。
2. C 轮崩盘已排除：换 `rules` 扩写后恢复并超过 B。
3. `multi_turn` 比 B 略回落（0.938→0.875 / 0.750→0.708），仍高于最早无路由基线的 context_hit 0.646；可再单独盯改写。
4. 本地 embed 下整场约 **6 分钟**，均 0.79s/题。
