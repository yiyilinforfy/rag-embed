# VKube 文档 RAG — 架构与流程复习稿

> 面向本人复习 / 面试口述。以当前代码为准（手写链路，不用 LangChain）。  
> 数据规模约：中文文档数十篇 → **~267 chunk**。

---

## 1. 一句话定位

把 VKube 中文文档切块、向量化入库；用户提问时 **混合召回 + 精排 + 门闸**，再让 LLM **只根据检索到的原文**回答，并带来源链接。

---

## 2. 两条主线

### 2.1 写路径（离线建索引）

```
GitHub 文档仓库 (docs/zh)
  → sync：git fetch / 定时 / webhook
  → chunker：按 Markdown 标题切分（对齐锚点）
  → embedding：bge-m3 → 1024 维向量
  → ingest：写入 PostgreSQL + pgvector
       documents / chunks
  →（可选）重建内存 BM25 索引
```

增量三层：commit 未变跳过 → 文件 hash 未变跳过 → chunk 文本未变则复用旧向量。

### 2.2 读路径（在线问答）

```
Vue 前端
  → POST /api/ask/stream（X-Wallet-Address）
  → auth + 限流
  → trim_history
  →（多轮）LLM query rewrite → 独立检索问句
  → search：
        ① 问题 embed
        ② 向量 Top-20  ∪  BM25 Top-20（按 chunk_id 去重）
        ③ 内容去重；（rerank 时通常不做章节去重）
        ④ Cross-Encoder Rerank 重排
        ⑤ 截断 Top-K（默认 5）
  → apply_score_gates（进 LLM 前）：
        MIN_SCORE（向量余弦）→ RERANK_MIN_SCORE（rerank 分）
        特例：余弦会整题清空，但有高 rerank（纯 BM25 Top-K）时可放行
  → 无上下文 → NO_ANSWER（不调生成 LLM）
  → 有上下文 → 拼 prompt → 流式生成
  → SSE：先 sources，再 token，再 end
```

**粗排 = 多路召回（向量 + BM25 并集）**  
**精排 = bge-reranker-v2-m3（Cross-Encoder）**

---

## 3. 架构示意

```
                    ┌───────────── 离线 ─────────────┐
  文档 ──切块──► embed(bge-m3) ──► PG/pgvector
                      │                    ▲
                      └──► 内存 BM25 语料 ──┘（与 chunks 同步）

                    ┌───────────── 在线 ─────────────┐
  问题 ──改写──► embed ──► 向量 Top-20 ┐
                 BM25 Top-20 ──────────┼─并集─► Rerank ─► Top-K
                                       │              │
                                       │         分数门闸
                                       │              ▼
                                       │         LLM 生成 + 引用
                                       └──────────► SSE → 前端
```

线上大致：`Docker(VKube)` + `Supabase(pgvector)` + `硅基流动(embedding/rerank)` + `LLM(Ollama 或云端)`。

---

## 4. 关键配置（默认量级）

| 配置 | 典型值 | 作用 |
|------|--------|------|
| `TOP_K` | 5 | 最终进 LLM 的条数上限 |
| `RERANK_CANDIDATES` / 向量 | 20 | 稠密召回条数 |
| `BM25_CANDIDATES` | 20 | 稀疏召回条数 |
| `MIN_SCORE` | 0.45 | Top-K 后比 **向量余弦**；问题级全不过则拒答 |
| `RERANK_MIN_SCORE` | 0.05 | Top-K 后比 **rerank 分**；0=关闭 |
| Embedding | bge-m3 / 1024 维 | 建库与查询必须同一模型 |
| Rerank | `BAAI/bge-reranker-v2-m3` | 精排，不是 embedding |

**注意：**  
- Top-20 **取候选时不做** MIN_SCORE（保召回）。  
- 门闸用的 cos 是召回/合并时就算好的，**不是** rerank 后再算一遍。  
- BM25-only chunk 也会补 cos，只是往往偏低。

---

## 5. 为什么并集、不加权 / 不上 RRF

- 余弦、BM25、rerank **分数尺度不同**，直接 6:4 相加难解释、要调参。  
- 当前策略：**并集扩候选 → Rerank 拍板**。  
- RRF（倒数名次融合）适合无精排或多路只要名次融合；已有 Cross-Encoder 时优先级靠后。

---

## 6. 切分策略（本项目）

- **按 Markdown 标题路径**切（不是父子块 parent-child）。  
- 好处：结构清晰、URL 锚点可对齐文档站。  
- 长节可再拆 part；评测用 `(source, heading)` 标识 chunk，避免 DB id 重建后失效。

---

## 7. 评测怎么测

- 集：`evals/dataset.json`（约 100+ 题）：direct / paraphrase / **exact_term** / multi_doc / english / multi_turn / **out_of_scope**。  
- 命令示例：

```bash
uv run python -m evals.run --strategies rerank,hybrid --retrieval-only
```

- 检索指标：hit@k、ndcg、recall、context_recall、blocked_rate_oos…  
- 生成指标（可选跑答案）：correctness、faithfulness、refusal_rate_oos、pass_rate。

### 近期对比印象（复习用，以 runs 里 JSON 为准）

| 对比 | 结论 |
|------|------|
| rerank vs hybrid | hybrid 抬 exact_term / 关键词；hit@5 明显升；延迟略增 |
| 仅 MIN_SCORE vs +RERANK_MIN | OOS 系统拦截约 25%→50%；少数弱分能答题会误拒；可救回余弦清空的 BM25 题 |
| 生成侧 | 检索强于答题忠实度；小模型易扩写 |

典型 BM25 救回例子：`refundVsys`、`postwosign`、`refundToken`、`preventDefault`。

---

## 8. 拒答两层

1. **系统拒答**：门闸后无上下文 → 固定 NO_ANSWER（不调生成 LLM）。  
2. **模型拒答**：有资料但不够 → prompt 要求说「文档未覆盖」。

OOS 里「听起来很产品」（如 Windows 容器）可能 cos/rerank 都高 → 分数门拦不住，要靠生成拒答或问题路由。

---

## 9. 模块地图（读代码顺序）

`config` → `models` → `chunker` → `embedding` → `ingest` → `bm25_index` / `tokenizer` → `search` → `rerank` → `rag` → `llm` → `main` → `auth` / `ratelimit` → `sync`

---

## 10. 已知短板（面试可主动说）

- 多轮 rewrite 仍会 miss。  
- 生成 faithfulness / pass_rate 弱于检索。  
- OOS 高分假相关未彻底解决。  
- 数据量小，尚未依赖 HNSW；未上 ES、未上 RRF、未做父子块。  
- 主回答路径未用 JSON Schema 强约束（评测 judge 有 JSON）。

---

## 11. 口述 60 秒版

> 我做了个 VKube 文档问答 RAG：离线按标题切块，用 bge-m3 打成 1024 维进 pgvector，并建内存 BM25。在线先改写多轮问题，再向量和 BM25 各取 Top-20 并集，用 bge-reranker 精排取 Top-5，再用余弦和 rerank 分做门闸，过了才给 LLM，SSE 流式返回。评测上混合检索明显抬了精确词命中；拒答靠门闸加 prompt。没做分数加权是因为尺度不同，排序交给 cross-encoder。
