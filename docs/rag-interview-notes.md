# RAG 面试题 × 知识点复习稿

> 把网上常见题 + 自己追问过的概念，融成「用本项目怎么答」。  
> **原则：按自己的架构讲，别背别人的 Milvus / 6:4 / 512 维 / 父子块。**

配套架构文：[`rag-architecture.md`](./rag-architecture.md) · 可观测：[`observability-design.md`](./observability-design.md)

---

## A. 用本项目直接能答的题

### A1. 讲讲 RAG 完整流程 / 召回读写各环节

**写：** 文档 → 切块 → bge-m3(1024) → pgvector；可选重建 BM25。  
**读：** **路由** →（多轮）rewrite → **口语 expand** → 向量 Top-20 ∪ BM25 Top-20 → Rerank → Top-K → 门闸 → LLM → SSE（带 `ask_id`）→ 落库。  
**数量级：** 候选约 20～35；精排后默认 5；热路径常见总时长 ~5s；冷启动时 embedding/改写可拖到 10s+。

### A2. 优化了哪些指标？怎么测的？

- 检索：hit@5、ndcg、**context_hit**、blocked_rate_oos；分题型（exact_term / paraphrase / multi_turn）。  
- 生成：correctness、faithfulness、pass_rate、refusal_rate_oos（LLM-as-judge + key_points）。  
- 方法：`evals/dataset.json` / gen500 / gen50 + `evals.run --strategies …` + `--compare` 回归。  
- **数字口径（复习用，以 REPORT 为准）：**  
  - hybrid 抬 exact_term；大盘 hit@5 约 **0.96→0.97** 量级。  
  - rules 扩写 + RERANK_MIN=0.02：paraphrase **context_hit ~0.77→0.98**。  
  - llm 扩写（弱本地）负例：hit@5 **崩到 ~0.68**。  
  - gen50：pass_rate ~0.55，OOS 拒答 ~1.0，multi_turn 最弱。  
- 诚实：检索强于生成忠实度。

### A3. 粗排 / 精排分别做什么？

| | 本项目 |
|--|--------|
| **粗排** | 向量语义召回 + BM25 关键词召回，**并集**扩候选 |
| **精排** | Cross-Encoder（`bge-reranker-v2-m3`）对候选重打分，取 Top-K |

不要说「粗排=向量、精排=BM25」。

### A4. 「语义相关但不是用户想要的」怎么解决？

- 例子：OOS「AWS 价格」余弦够、rerank 很低 → `RERANK_MIN_SCORE` 拦。  
- 「Windows 容器」cos/rerank 都高 → 分数门不够，靠 **prompt 拒答 + 查询路由 REFUSE**（已做轻量路由，不是纯未来）。  
- 进 LLM 前两道闸：`MIN_SCORE`（余弦）+ `RERANK_MIN_SCORE`（现常用 **0.02**）。

### A5. 向量和 BM25 权重 6:4？为什么？

**我们没有 6:4。**  
答法：分数尺度不同，不做加权；并集 + Rerank 拍板，免调权重。  
若问 RRF：名次融合公式 \(1/(k+\mathrm{rank})\)；适合无精排时；我们有 rerank 故未上。

### A6. 召回不准的例子？

- `refundVsys` / `postwosign` / `refundToken`：纯向量弱，BM25 救回。  
- 口语 paraphrase：靠 **rules expand**（补「安全组」等文档词），不是靠再堆向量。  
- 多轮题：改写失败/偏题仍会导致 miss（暴露短板也加分）。  
- **反例：** 用弱本地模型做 llm expand，会把好检索改坏。

### A7. 为什么父子块？提升多少？

**本项目未用父子块。**  
改答：按 **Markdown 标题**切，对齐文档结构与锚点。  
了解父子块：子块精召回、返回父块保上下文——文档已是清晰标题树，未引入。

### A8. 向量库 vs MySQL？为什么不用普通表存向量？

- 用的是 **PostgreSQL + pgvector**，不是 Milvus。  
- 要做高维相似度（`<=>` 余弦距离），普通 B-tree/KV 不适合近邻检索。  
- 小数据可全表扫；大规模再上 **HNSW** 等 ANN 索引。

### A9. 库里存什么？为什么是 1024 维不是 512？

- chunks：content、source、title、heading、url、metadata、**embedding(1024)**。  
- 另有 **ask_events / ask_feedback**（每问事实 + 赞踩）。  
- 维数跟 **bge-m3 默认输出**走；建库查询必须同模型。  
- 更高维≠一定更准；小语料 1024 够用。勿说 512。

### A10. 系统延迟怎么算？

- 服务端分段进 `ask_events`：`route_ms` / `rewrite_ms` / `expand_ms` / `vector_ms` / `bm25_ms` / `rerank_ms` / `generate_ms` / `first_token_ms` / `total_ms`。  
- Grafana 可画 P50/P95；用户体感看 **首 token**，不是只看总时长。  
- 经验：热路径 ~5s 可用；冷启动时 `vector_ms`（本机 embed）常是第一凶手，其次 rewrite、generate。  
- 前端渲染通常不是主瓶颈。

### A11. 有没有结构化输出 / JSON Schema？

- **主回答：自然语言 + `[n]` 引用**，未用 Schema 强约束。  
- **评测 judge / 路由 LLM**：JSON。  
- 概念：Prompt 要求形状 + Schema/Zod **校验**；不合格可重试。

### A12. 部署？数据扩大 100 倍？

- 现：API + Supabase（或本地 PG）+ 硅基流动 embedding/rerank + DeepSeek / Ollama。  
- 百倍：向量 HNSW；BM25→ES/OpenSearch 或分片；API 多副本；先找瓶颈再扩。  
- **了解分布式，未上集群**——诚实。

### A13. 查询路由是什么？算 Agent 吗？

- **四级：** DIRECT / RETRIEVE / MULTI / REFUSE；默认 rules，可 LLM。  
- MULTI 强制 rewrite；DIRECT/REFUSE 跳过检索。  
- **不是 Agent：** 仍是固定 pipeline，模型不在循环里反复选工具。  
- Agentic RAG = 检索可多轮、由 LLM 决定「要不要再搜」。我们还没做。

### A14. Rewrite 和 Expand 有什么区别？

| | Rewrite | Expand |
|--|---------|--------|
| 场景 | 多轮指代 | 单轮口语≠文档词 |
| 推荐 | aux LLM（有历史时） | **rules**（快、稳） |
| 坑 | 慢、改偏 | llm+弱模型可毁掉 hit@5 |

### A15. LangChain 默认链会不会更好？

- **大概率不会。** 默认常是「向量 Top-K → 生成」；我们多了 hybrid、rerank、门闸、路由、扩写、拒答与评测。  
- LangChain 赢在搭得快；效果取决于你怎么配。我们手写是为了可控与可讲清楚。

### A16. 用 LLM 评回答质量靠谱吗？

- **靠谱，且业界常用**（RAGAS 等也是这条思路），但是 **有偏差的相对打分器**。  
- 我们用法：固定要点 `key_points` + judge 打 correctness/faithfulness；适合回归对比。  
- 正经配套：固定 judge 模型与 prompt、抽样人工复核、检索指标仍用标注硬指标。

---

## B. 概念速记（自己问过的）

### B1. Embedding 维度与模型

- 维度主要由模型（及是否 Matryoshka 截断）决定。  
- **bge-m3 → 通常 1024**；可截短维但是进阶用法，默认全维。  
- 换模型必须 **全库重嵌入**。  
- 常见：bge-m3、e5、OpenAI embedding-3 等。

### B2. Rerank = Cross-Encoder？

**是。**  
- Bi-encoder：问/文各自向量再比（快，粗召回）→ bge-m3。  
- Cross-encoder：问+文一起进模型打分（准，慢）→ bge-reranker-v2-m3。  
- **不是**同一个「bge-m3-rerank」模型干两件事。

### B3. 为何不在向量 Top-20 做 MIN_SCORE？

- Top-N 要 **召回宽**；门闸要 **进 LLM 前变严**。  
- 过早过滤会砍掉「cos 一般但关键词/rerank 会救」的候选。  
- 门闸用的是候选上 **已有** cos，不是 rerank 后再生成。

### B4. BM25「被清空」？

- BM25-only 也会补 cos，常 **&lt; 0.45**。  
- 问题级规则：Top-K **全员** cos 不过 → 整题清空。  
- 修复：高 rerank 时可覆盖放行（如 `postwosign`）。

### B5. OOS / 拒答 / 幻觉

| 词 | 含义 |
|----|------|
| OOS | 文档范围外的题 |
| 路由拒答 | REFUSE，不检索 |
| 系统拒答 | 门闸空上下文 → NO_ANSWER |
| 模型拒答 | LLM 说资料不足 |
| 幻觉 | 写出资料没有的具体说法（faithfulness） |

痛点：检索已较强；生成忠实度 + 多轮 + 沾边 OOS 是下一刀。

### B6. BM25 vs ES/OpenSearch

- **BM25**：打分公式。  
- **ES/OpenSearch**：搜索引擎产品，大规模全文检索，**常用 BM25 打分**。  
- 不是「初级 BM25 / 高级 ES」，是「算法 vs 带倒排/分片的引擎」。

### B7. ANN / HNSW / 和 SQL Top-K

| | 含义 |
|--|------|
| SQL `ORDER BY dist LIMIT K` | 小库精确（或引擎实现的）排序截断 |
| **ANN** | Approximate Nearest Neighbor，**近似**近邻检索方法，**不是**某个生成模型名 |
| **HNSW** | 一种 ANN **索引**：向量当点，近邻连边，多层图上跳着查 |

ANN ≠ 必须多机；多机分片是扩容。节点一般 = 一个向量 ≈ 一个 chunk。

### B8. Top-K 与堆

- 海量取 Top-K 常用 **大小为 K 的堆**（要最大 K 个常用小根堆维护）。  
- 堆 ≈ 完全二叉树的数组实现。  
- 向量库里更多是索引 + LIMIT，不必手写堆。

### B9. 分布式 ≠ CDN

- **CDN**：静态资源边缘缓存。  
- **服务分布式**：多实例扛流量、数据分片扛容量、拆服务。  
- 前端类比：多副本 ≈ 多台相同 API + 负载均衡；分片 ≈ 数据切开并行查。

### B10. Zod / JSON Schema

- 都是校验「输出/数据长什么样」。  
- Zod：TS 生态；JSON Schema：跨语言描述。  
- 和「Prompt 要求 JSON」是配套：生成 + 校验。

### B11. Citation 对齐

- 检查答案里的 `[n]` 是否对应上下文、断言是否真在原文。  
- 可先在评测集离线做；非当前主路径必做项。

### B12. P95 / 可观测分层

- **P95**：排序后第 95% 延迟，比平均值更能反映大部分用户体感。  
- 中团队：业务事实进 Postgres；指标可用 Grafana SQL（我们已做 Ask Overview）；Prometheus 偏 QPS/存活，不替代每问明细。  
- 宝塔看机器 ≠ 业务可观测。

### B13. Pipeline RAG vs Agentic RAG

- **Pipeline：** 步骤写死（我们现在）。  
- **Agentic：** LLM 循环选工具（可多次检索/改写/校验）。  
- 路由只是分流，**不等于** Agent。

---

## C. 通用基础题（转岗也要会）

### C1. URL 访问全流程

输入 URL → DNS → TCP →（HTTPS 则 TLS）→ HTTP 请求 → 服务器响应 → 浏览器解析 → 可能再 fetch/SSE 调 API → 渲染。

### C2. HTTP vs HTTPS

HTTPS = HTTP + TLS；加密、完整性、证书身份；有握手成本，现代默认应使用。

### C3. 海量数据取 Top-K

- 通用算法：堆 / 快速选择。  
- 向量场景：ANN（如 HNSW），不是扫全表精排。

---

## D. 被追问时的标准短答

**为什么没加权 / RRF？**  
> 尺度不同不宜相加；已有 cross-encoder 精排，用并集扩召回即可。RRF 是备选降级方案。

**bge 够不够？**  
> 中文文档 + 评测数字说明够用；瓶颈在生成忠实度/多轮/延迟，不在维数。

**已经大厂水准了吗？**  
> 不是平台级（缺大规模索引、Agent 闭环、严格线上 SLO 常态化），但是可答辩的中级全链路 RAG；评测与可观测已按中小团队规范在做。

**扩写为什么不用 LLM？**  
> 弱本地模型做过负例，hit@5 掉到 0.68；rules 扩写把 paraphrase context_hit 拉到 ~0.98，又几乎零延迟。

---

## E. 面试前 5 分钟自检

1. 能画：路由 → expand/rewrite → 并集粗排 → Rerank → 双门闸 → LLM → 落库。  
2. 能报：hybrid 对 exact_term；paraphrase 扩写收益；llm 扩写负例；gen50 生成弱于检索。  
3. 能举：一个 BM25 救回 + 一个仍失败（多轮）+ 冷启动延迟故事。  
4. 能澄清：pgvector 不是 Milvus；1024 不是 512；无 6:4；无父子块；有路由但不是 Agentic。  
5. 能说下一步：多轮改写质量 / 生成 faithfulness / 热路径延迟，而不是再堆召回路子。

---

## F. 推荐复习顺序

1. 本文件 **B 概念**扫一遍（含 B12/B13）  
2. **A 项目题**对照 [`rag-architecture.md`](./rag-architecture.md)（尤其 A13–A16）  
3. 口头练 A1 + A3 + A5 + A6 + A13（最常连问）  
4. 扫一眼 `evals/gen500/REPORT.md` / `evals/gen50/REPORT.md` 数字  
5. 补 C1–C3 基础
