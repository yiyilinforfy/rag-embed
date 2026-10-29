# 用 AI 测召回质量（不上生成 LLM）— Prompt + 操作手册

目标：让另一名 AI **自己出约 500 题**，只评 **检索到的 chunk 好不好**（不跑 `/ask` 生成答案）。  
工作目录：`rag-backend/`。当前推荐策略：**hybrid = 向量∪BM25 → Rerank → Top-5 → 门闸**。

---

## 0. 给你要测的那名 AI 的系统 Prompt（整段复制）

```text
你是 VKube 文档 RAG 的「检索质量评测员」。只评召回，不调用生成回答接口。

## 任务
1. 基于仓库里真实文档 chunk 出约 500 道用户问题。
2. 对每题跑混合检索，拿到 Top-5（及门闸后的 context）。
3. 判断召回质量并汇总指标与坏例。

## 硬性约束
- 题目必须能对应到至少一个真实 chunk（用 source + heading 标识）。禁止凭空编造文档里没有的 API/参数名。
- 不要评「最终自然语言答案」；只评 chunk 是否相关、是否进 Top-5 / context。
- 不要修改业务代码除非为跑通评测所必需；优先写 JSON 结果到 evals/ 下。
- 检索策略固定为 hybrid（向量 Top-20 ∪ BM25 Top-20 → rerank → top_k=5），与线上目标一致。
- 已有 evals/dataset.json 约 100 题可作格式参考，但 500 题应新建文件，勿直接覆盖生产评测集，除非用户明确要求合并。

## 题型配比（合计约 500）
- direct 100：用词接近标题/小节名
- paraphrase 100：同义改写，少用文档原词
- exact_term 120：命令、字段、API、环境变量、菜单英文原句（如 refundVsys、portSpecification）
- multi_doc 40：需要 ≥2 个不同 source 或相距很远的 heading
- english 40：英文问，答案在中文文档
- multi_turn 40：带 1～2 轮 history 的追问（代词/「那个」）
- out_of_scope 40：文档明确没有的（天气、股价、AWS 实时价、做菜等）；期望 blocked 或 Top 明显不相关

## 出题流程（必须遵守）
1. 先导出 chunk 目录：`uv run python -m evals.label chunks`
2. 按文档分批阅读 `evals.label chunks <关键词> --full`，从 chunk 内容出题，并写下预期 relevant：
   {"source":"zh/...md","heading":"完整标题路径","grade":2或1}
3. 每题 id 唯一，如 gen-exact-001；question 简洁像真实用户。
4. out_of_scope：relevant=[]，不编造 heading。
5. 写完后 `uv run python -m evals.label check`（若接入 dataset 校验）确保 heading 未失效。

## 检索与判定
对每题：
1. 用 hybrid 检索 Top-5（见下方操作命令）。
2. 判定（检索侧）：
   - hit@5：Top-5 的 (source,heading) 是否命中任一 grade≥1 的标注
   - hit@1：第 1 名是否命中
   - context_hit：过门闸后的 context 是否仍含 grade=2（若被门闸清空则记 blocked）
3. 对 out_of_scope：记录 blocked=true/false；若未 blocked，记录 Top-1 的 heading 与分数（说明「沾边噪声」）。
4. 不要用「看起来相关」替代标注：先有 gold relevant，再比检索结果。

## 交付物
1. evals/gen500/dataset.json — 500 题（格式同现有 dataset）
2. evals/gen500/run-hybrid.json — 每题 query_keys、scores、rerank_scores、from_bm25、blocked、hit@5 等
3. evals/gen500/REPORT.md — 总表：各 category 的 hit@5 / hit@1 / blocked_oos；最差 20 题；结论 5 条

## 质量红线
- exact_term 故意用稀有标识符，避免全是「怎么部署」这种过宽问题。
- 同一 chunk 不要衍生超过 5 道几乎相同的题。
- 若 hybrid 依赖 RERANK_API_KEY 缺失，停止并报告，不要默默退化成 baseline 还声称测了 hybrid。
```

---

## 1. 你这边先准备环境（操作细节）

在 `rag-backend/`：

```bash
# 1) DB + 文档索引可用
docker compose up -d   # 若本地 PG
# 或已连 Supabase 则跳过

# 2) .env 确认（测「当前 RAG」必须开）
# RERANK_ENABLED=true
# RERANK_API_KEY=...
# RERANK_MODEL=BAAI/bge-reranker-v2-m3
# HYBRID_ENABLED=true          # 或评测时用 --strategies hybrid
# RERANK_MIN_SCORE=0.05
# MIN_SCORE=0.45
# EMBEDDING_* 可用

# 3) 冒烟
uv run python -m evals.label chunks | head
uv run python -m evals.run --strategies hybrid --retrieval-only --ids cli-install,dev-refundvsys-fn,oos-weather
```

若 `--strategies hybrid` 报缺 key，先修好再让 AI 开测。

---

## 2. 建议 AI 分阶段做（500 题一次做完易翻车）

| 阶段 | 做什么 | 规模 |
|------|--------|------|
| P0 | 读 chunk 目录，定文档覆盖清单 | — |
| P1 | 先出 **100** 题 + 标注 + 跑通 hybrid 报告 | 100 |
| P2 | 扩到 **300**（补 exact_term / OOS / multi_turn） | +200 |
| P3 | 补满 **500**，出最终 REPORT | +200 |

你对 AI 说：「严格按 P0→P1，P1 报告我过目后再 P2」。

---

## 3. 出题时 AI 应用的命令

```bash
# 列出所有 chunk key
uv run python -m evals.label chunks

# 读某篇全文（出题+标 gold 用）
uv run python -m evals.label chunks lua_scripts --full
uv run python -m evals.label chunks Full-V-Kube --full

# 单题看检索（默认走 Settings；要 hybrid 请用 evals.run）
uv run python -m evals.label search "refundVsys 参数是什么" -k 5
```

**批量评测（推荐）**：把题写入 JSON 后：

```bash
mkdir -p evals/gen500
# 若 AI 把题写成独立 dataset，可临时改 DATASET 或合并进评测脚本；
# 最简单：合并进 evals/dataset.json 的副本再跑：

cp evals/dataset.json evals/gen500/dataset.backup.json
# AI 维护 evals/gen500/dataset.json 后，用下面方式跑（需小改或设路径——见 §5）

uv run python -m evals.run --retrieval-only --strategies hybrid \
  --out evals/gen500/run-hybrid.json
```

> 现有 `evals.run` 默认读 `evals/dataset.json`。500 题建议：  
> **方案 A**：AI 生成 `evals/gen500/dataset.json`，你临时替换/软链到 `evals/dataset.json` 再跑；跑完还原。  
> **方案 B**：让 AI 给 `evals/run.py` 加 `--dataset` 参数（一次性小改，更干净）。

### 方案 A 临时替换示例

```bash
cp evals/dataset.json evals/dataset.prod.json
cp evals/gen500/dataset.json evals/dataset.json
uv run python -m evals.label check
uv run python -m evals.run --retrieval-only --strategies hybrid --out evals/gen500/run-hybrid.json
mv evals/dataset.prod.json evals/dataset.json
```

---

## 4. 单题 JSON 格式（必须）

```json
{
  "id": "gen-exact-001",
  "category": "exact_term",
  "question": "refundToken 的三个参数是什么",
  "history": [],
  "relevant": [
    {
      "source": "zh/developers/lua_scripts.md",
      "heading": "lua脚本 > Lua 脚本功能使用",
      "grade": 2
    }
  ],
  "key_points": []
}
```

检索-only 时 `key_points` 可空数组；仍建议留 1 句便于以后测生成。

`multi_turn` 示例：

```json
{
  "id": "gen-mt-001",
  "category": "multi_turn",
  "question": "那个过期了还能用吗",
  "history": [
    {"role": "user", "content": "Access Token 怎么创建"},
    {"role": "assistant", "content": "……（可短）"}
  ],
  "relevant": [ ... ],
  "key_points": []
}
```

---

## 5. 你给 AI 的「用户消息」模板（开测时粘贴）

```text
请按 docs/retrieval-ai-eval-prompt.md 里的系统 Prompt 执行。

仓库：rag-backend
目标：约 500 题，只测召回（hybrid），不上生成 LLM。
先做 P0+P1（100 题），输出：
- evals/gen500/dataset.json（先 100 题）
- 用方案 A 跑 hybrid --retrieval-only
- evals/gen500/REPORT-p1.md

P1 通过后再扩到 500。
约束：heading 必须来自 `evals.label chunks`；exact_term 至少 25% 用稀有标识符。
```

---

## 6. REPORT 里应有的表（验收标准）

```markdown
| category | n | hit@1 | hit@5 | context_recall | blocked(仅OOS) |
| ... |

坏例 Top20：id / question / gold / 实际 Top3 / 原因猜测（向量弱/改写烂/门闸误杀）

结论：
- hybrid 相对弱点（题型）
- 是否建议调 RERANK_MIN_SCORE / 补 BM25 分词
- 不建议改动的部分
```

**及格线（经验，可调）：**  
- 全量 hit@5 ≥ 0.90  
- exact_term hit@5 ≥ 0.92  
- OOS blocked ≥ 0.40（有门闸时）  
- 误杀能答题（answerable 且 blocked）单独列表，越少越好  

---

## 7. 注意（避免测歪）

1. **出题必须先读 chunk**，否则 gold 乱标，数字无意义。  
2. **不要用「当前检索 Top1」当唯一 gold**（会自我循环夸大）。应以文档内容为准标 relevant。  
3. 500 题 API 费用：每题 1 次 embed + 1 次 rerank；控制并发、分阶段。  
4. 测的是「找 chunk」，不是「答得对」；生成质量另开任务。  
5. `scripts.search` / `evals.label search` 走默认 Settings；对比策略请用 `evals.run --strategies`。

---

## 8. 最小自检（你自己 2 分钟）

```bash
uv run python -m evals.label check
uv run python -m evals.run --strategies hybrid --retrieval-only --categories exact_term,out_of_scope
```

若现有 exact_term / OOS 基线都跑不通，先别让 AI 出 500 题。
