# RAG 评测体系

目标：不只知道"有没有检索到"，还知道"答案对不对"。改 chunker、换模型、调 prompt 之后，用同一套题算出数字来对比。

## 怎么跑

```bash
# 数据库要在跑（docker compose up -d），embedding / LLM 按 .env 配置好
uv run python -m evals.label check          # 标注是否还和当前切分对得上
uv run python -m evals.run --retrieval-only # 只测检索，不花钱
uv run python -m evals.run                  # 检索 + 生成 + 判分（每题约 2 次 LLM 调用）
uv run python -m evals.run --compare evals/runs/<上次>.json

# 对比 baseline / rerank / hybrid（向量∪BM25 → Reranker）
# 需要 RERANK_API_KEY；本地 .env 没有的话可临时从 test.env 注入：
#   export RERANK_API_KEY=$(grep ^EMBEDDING_API_KEY= test.env | cut -d= -f2-)
#   export RERANK_BASE_URL=$(grep ^EMBEDDING_BASE_URL= test.env | cut -d= -f2-)
uv run python -m evals.run --strategies baseline,rerank,hybrid --retrieval-only
# 只比当前方案 vs 混合：
uv run python -m evals.run --strategies rerank,hybrid --retrieval-only
```

每次运行会在 `evals/runs/` 写一份带时间戳的 JSON（每题的检索结果、回答、判语都在里），控制台打印摘要。

## 题型（`category`）

| 题型 | 在测什么 |
|------|----------|
| `direct` | 用词接近文档标题，基线 |
| `paraphrase` | 换个说法，考语义匹配 |
| `exact_term` | 命令名 / 参数名 / 接口名，向量检索的弱项 |
| `multi_doc` | 答案分散在两篇以上文档 |
| `english` | 英文提问（文档只有中文） |
| `multi_turn` | 带对话历史的追问，考 query rewriting |
| `out_of_scope` | 文档里没有答案，正确行为是拒答 |

## 一道题长什么样

```json
{
  "id": "svc-sg-create",
  "category": "direct",
  "question": "怎么创建安全组",
  "history": [],
  "relevant": [
    {"source": "zh/VKservice/securityGroup.md", "heading": "安全组 > 创建安全组", "grade": 2}
  ],
  "key_points": [
    "在安全组页面点击创建",
    "创建后要把服务添加到安全组里"
  ]
}
```

- `relevant`：Chunk 级相关性标注。`grade` 2 = 直接回答了问题；1 = 有帮助的背景。用 `(source, heading)` 标识，不用数据库 id（重新入库 id 会变）。
- `key_points`：参考答案的要点，每条一个可以单独判断对错的事实。

## 指标怎么读

**检索（不花钱）**

- `doc_hit@k` / `doc_mrr`：文档级，兼容旧的 22 题评测。
- `hit@k` / `recall@k` / `precision@k` / `mrr` / `ndcg@k`：Chunk 级。`ndcg` 同时看排名和相关程度（grade 2 比 grade 1 值钱）。
- `context_recall` / `context_precision`：真正送进 LLM 的那几条（过了 `MIN_SCORE`）里，相关内容占多少。`context_recall` 低 → LLM 根本拿不到答案。
- `blocked_rate_oos`：超范围题有没有被阈值拦住（没调 LLM）。

**答案质量（花钱）**

- `correctness`：参考要点讲到了几条。
- `faithfulness`：有没有编造检索资料里没有的具体说法。
- `pass_rate`：能回答的题里，"没拒答 + 忠实 + 正确性 ≥ 2/3" 的比例。
- `refusal_rate_oos` / `false_refusal_rate`：超范围题的拒答率，以及能回答的题被误拒的比例。

## 怎么加新题

```bash
uv run python -m evals.label chunks securityGroup --full   # 读某篇文档的所有 chunk
uv run python -m evals.label search "怎么创建安全组"         # 看当前系统检索到什么（仅供参考）
# 编辑 evals/dataset.json，贴上 relevant / key_points
uv run python -m evals.label check
```

标注时不要只从检索结果里挑：检索没找到的相关 chunk 才是最该标的，否则评测集会偏向"当前系统已经能找到的内容"。

## 回归

`--compare` 会按 `evals/run.py` 里的 `REGRESSION_RULES` 对比关键指标。掉得超过阈值就以退出码 1 结束，适合改完代码本地验证，也适合以后接到 CI。

## 目录

```
evals/
  dataset.json      正式题库
  dataset.py        读取 / 校验
  metrics.py        指标（纯函数）
  judge.py          LLM-as-judge
  run.py            评测入口
  label.py          标注辅助
  runs/             每次运行的原始结果（已 gitignore）
  drafts/           出题时的草稿（已 gitignore）
  README.md
```
