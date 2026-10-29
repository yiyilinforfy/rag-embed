# VKube 文档 RAG — 架构与流程复习稿

> 面向本人复习 / 面试口述。以当前代码为准（手写链路，不用 LangChain）。  
> 数据规模约：中文文档数十篇 → **~279 chunk**（本地/Supabase 同步后以 health 为准）。  
> 配套：[`rag-interview-notes.md`](./rag-interview-notes.md) · [`observability-design.md`](./observability-design.md) · `evals/`

---

## 1. 一句话定位

把 VKube 中文文档切块、向量化入库；在线先做 **查询路由**，再 **口语扩写 / 多轮改写 → 混合召回 + 精排 + 门闸**，让 LLM **只根据检索到的原文**回答；每问落库可观测，带赞踩反馈。

仍是 **增强版 Pipeline RAG**（固定流水线），不是 Agentic RAG（模型不在循环里自己决定再搜几轮）。

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
Vue 前端（/agent）
  → POST /api/ask/stream（X-Wallet-Address）
  → auth + 限流
  → trim_history
  → router：DIRECT | RETRIEVE | MULTI | REFUSE
       DIRECT/REFUSE → 固定文案，不检索、不调回答 LLM
  →（有历史或 MULTI）aux LLM rewrite → 独立检索问句
  → QUERY_EXPAND：off | rules（关键词提示）| llm（再改写，本机小模型易改坏）
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
  → 有上下文 → 拼 prompt → 流式生成（可附媒体）
  → SSE：meta(ask_id) → route → sources → token* → end
  → ask_events 落库；可选 POST /api/feedback 赞踩
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
  问题 ──router──► DIRECT/REFUSE（短路）
              │
              └── RETRIEVE/MULTI
                    │
         rewrite? + expand(rules|llm)
                    │
           embed ──► 向量 Top-20 ┐
           BM25 Top-20 ──────────┼─并集─► Rerank ─► Top-K
                                 │              │
                                 │         分数门闸
                                 │              ▼
                                 │         LLM 生成 + 引用
                                 └──────────► SSE → 前端
                                              │
                                         ask_events / feedback
```

部署常见组合：`API 进程` + `Supabase(pgvector)` 或本地 Docker PG；embedding/rerank 可用硅基流动；回答可用 DeepSeek；改写/路由可用本机 Ollama（冷启动会拖延迟）。

---

## 4. 查询路由（`app/router.py`）

| 策略 | 行为 | 典型触发 |
|------|------|----------|
| `DIRECT` | 固定问候，不检索 | 「你好」 |
| `RETRIEVE` | 正常检索生成 | 「怎么创建安全组」 |
| `MULTI` | **强制 rewrite** 再检索 | 「那怎么删除它」 |
| `REFUSE` | 固定拒答，不检索 | 明确域外 / 注入 |

- 默认 `ROUTE_MODE=rules`：规则高置信优先；可开 LLM 路由（贵且本机易抖）。  
- **产品词压过问候**：「你好，安全组怎么建」→ RETRIEVE，不是 DIRECT。  
- 作用：事前 triage，少打无意义 embedding；补门闸拦不住的「听起来很产品」OOS（部分）。

---

## 5. 查询改写 vs 口语扩写（别混）

| | **Rewrite（多轮）** | **Expand（口语→文档词）** |
|--|---------------------|---------------------------|
| 何时 | 有历史，或 MULTI | 每题检索前（可关） |
| 目的 | 消解「它/那个」成独立问句 | 把白话补成文档用语 |
| 实现 | aux LLM | `off` / `rules`（正则提示词）/ `llm` |
| 坑 | 小模型慢、偶发改偏 | **llm + 弱本地模型曾把 hit@5 打崩到 ~0.68**；生产用 **rules** |

当前推荐：`QUERY_EXPAND_MODE=rules`，`RERANK_MIN_SCORE=0.02`（paraphrase 友好；0.05 偏严）。

---

## 6. 关键配置（默认量级）

| 配置 | 典型值 | 作用 |
|------|--------|------|
| `TOP_K` | 5 | 最终进 LLM 的条数上限 |
| `RERANK_CANDIDATES` / 向量 | 20 | 稠密召回条数 |
| `BM25_CANDIDATES` | 20 | 稀疏召回条数 |
| `MIN_SCORE` | 0.45 | Top-K 后比 **向量余弦**；问题级全不过则拒答 |
| `RERANK_MIN_SCORE` | **0.02** | Top-K 后比 **rerank 分**；0=关闭 |
| `ROUTE_MODE` | rules | off / rules / llm |
| `QUERY_EXPAND_MODE` | rules | off / rules / llm |
| Embedding | bge-m3 / 1024 维 | 建库与查询必须同一模型 |
| Rerank | `BAAI/bge-reranker-v2-m3` | 精排，不是 embedding |

**注意：**  
- Top-20 **取候选时不做** MIN_SCORE（保召回）。  
- 门闸用的 cos 是召回/合并时就算好的，**不是** rerank 后再算一遍。  
- BM25-only chunk 也会补 cos，只是往往偏低。

---

## 7. 为什么并集、不加权 / 不上 RRF

- 余弦、BM25、rerank **分数尺度不同**，直接 6:4 相加难解释、要调参。  
- 当前策略：**并集扩候选 → Rerank 拍板**。  
- RRF（倒数名次融合）适合无精排或多路只要名次融合；已有 Cross-Encoder 时优先级靠后。

---

## 8. 切分策略（本项目）

- **按 Markdown 标题路径**切（不是父子块 parent-child）。  
- 好处：结构清晰、URL 锚点可对齐文档站。  
- 长节可再拆 part；评测用 `(source, heading)` 标识 chunk，避免 DB id 重建后失效。

---

## 9. 评测体系与实验路径

### 9.1 怎么测

- 正式集：`evals/dataset.json`；大规模检索：`evals/gen500/`（约 400 题）；生成小样：`evals/gen50/`。  
- 题型：direct / paraphrase / exact_term / multi_doc / english / multi_turn / out_of_scope。  
- 命令：

```bash
uv run python -m evals.run --retrieval-only
uv run python -m evals.run --strategies baseline,rerank,hybrid --retrieval-only
uv run python -m evals.run                  # + 生成 + LLM-as-judge
uv run python -m evals.run --compare evals/runs/<上次>.json
```

- 检索：hit@k、ndcg、context_hit/context_recall、blocked_rate_oos…  
- 生成：correctness、faithfulness、pass_rate、refusal_rate_oos（LLM judge，相对对比可信，辅以要点标注）。

### 9.2 实验路径（推荐口述顺序）

```
① baseline（仅向量）
② + rerank
③ + hybrid（向量∪BM25 → rerank）     ← exact_term 明显抬头
④ + ROUTE_MODE=rules                  ← OOS/闲聊短路；检索大盘略升
⑤ 试 QUERY_EXPAND=llm（弱模型）       ← 负例：paraphrase/大盘崩
⑥ QUERY_EXPAND=rules + RERANK_MIN=0.02 ← 正例：paraphrase 拉回
```

### 9.3 检索对照（gen500 / ~400 题，明细见 `evals/gen500/REPORT.md`）

| 轮次 | 配置 | hit@5 | context_hit | 备注 |
|------|------|------:|------------:|------|
| A | hybrid，无路由 | 0.960 | 0.902 | 基线 |
| B | + rules 路由 | 0.968 | 0.916 | 大盘略升 |
| C | B + **llm 扩写**（弱本地） | **0.677** | 0.622 | **反例：扩写改坏 query** |
| **D** | B + **rules 扩写** + RERANK_MIN=0.02 | **0.974** | **0.951** | 当前方向 |

**paraphrase（B→D）：** hit@5 `0.902→0.984`；context_hit `0.770→0.984`（最大收益）。  
**multi_turn：** D 比 B 略回落（改写仍是弱项），仍高于更早无路由基线。  
典型 BM25 救回：`refundVsys`、`postwosign`、`refundToken` 等。

### 9.4 生成小评测（gen50，见 `evals/gen50/REPORT.md`）

| 指标 | 约值 | 解读 |
|------|-----:|------|
| pass_rate | 0.55 | 可答题综合及格率 |
| correctness | 0.80 | 要点覆盖尚可 |
| faithfulness | 0.71 | 仍有幻觉空间 |
| refusal_rate_oos | 1.00 | 域外拒答好 |
| false_refusal_rate | ~0.11 | 约一成误拒 |
| multi_turn pass | ~0.17 | **端到端最弱** |

结论：**检索强于生成**；下一刀优先多轮 + 忠实度，而不是再堆召回路子。

---

## 10. 拒答两层 + 路由

1. **路由拒答**：REFUSE → 固定文案，不检索。  
2. **系统拒答**：门闸后无上下文 → NO_ANSWER（不调生成 LLM）。  
3. **模型拒答**：有资料但不够 → prompt 要求说「文档未覆盖」。

OOS「听起来很产品」可能 cos/rerank 都高 → 仅分数门不够，靠路由 + 生成拒答。

---

## 11. 可观测性（中团队向）

- **业务事实：** `ask_events`（每问 UUID、路由、检索策略、分段耗时、配置快照）+ `ask_feedback`。  
- **看数：** Grafana → Postgres（本地或 Supabase）；不必先上 Prometheus。  
- **SLO 体感：** 热路径总时长常约 **~5s**；冷启动（本机 Ollama embed/改写、BM25 首建）可到 **10s～30s**。瓶颈常见：`vector_ms`（embedding）> `generate_ms` > `rewrite_ms`。  
- 设计说明：[`observability-design.md`](./observability-design.md)。

---

## 12. 模块地图（读代码顺序）

`config` → `models` → `chunker` → `embedding` → `ingest` → `bm25_index` / `tokenize` → `router` → `search` → `rerank` → `rag` → `ask_log` → `llm` → `main` → `auth` / `ratelimit` → `sync`

---

## 13. 已知短板（面试可主动说）

- 多轮 rewrite / multi_turn 端到端仍弱。  
- 生成 faithfulness / pass_rate 弱于检索。  
- 本机 embedding/改写冷启动延迟差。  
- 数据量小，尚未依赖 HNSW；未上 ES、未上 RRF、未做父子块、未做 Agentic 多轮检索。  
- 主回答路径未用 JSON Schema 强约束（评测 judge 有 JSON）。

---

## 14. 口述 60 秒版

> 我做了个 VKube 文档问答 RAG：离线按标题切块，bge-m3 打成 1024 维进 pgvector，并建内存 BM25。在线先用规则路由分流闲聊/拒答/检索/多轮；检索前对口语做规则扩写、对追问做改写；再向量和 BM25 各 Top-20 并集，用 bge-reranker 精排取 Top-5，余弦和 rerank 双门闸后才给 LLM，SSE 流式返回，每问落库可观测。评测上 hybrid 抬精确词；rules 扩写把 paraphrase 的 context_hit 从约 0.77 拉到 0.98；弱模型 llm 扩写会毁掉检索——所以扩写用规则。没做分数加权是因为尺度不同，排序交给 cross-encoder。这是增强 Pipeline RAG，还不是 Agentic。
