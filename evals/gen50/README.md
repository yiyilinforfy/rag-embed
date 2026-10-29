# 生成 / 拒答小评测（50 题）

从 `gen500` 分层抽取，与 400 题检索评测并列：

| 题型 | 题数 |
|------|-----:|
| direct | 8 |
| paraphrase | 8 |
| exact_term | 8 |
| multi_doc | 4 |
| english | 4 |
| multi_turn | 6 |
| out_of_scope | 12 |

可答题均有 `key_points`；抽取时故意混入一部分检索 `context_hit=0` / OOS 未拦住的题，方便暴露生成与拒答问题。

```bash
# 只用 .env（本地 embed/DB）；不要 source test.env 覆盖
set -a && source .env && set +a
uv run python -m evals.label check --dataset evals/gen50/dataset.json
uv run python -m evals.run --dataset evals/gen50/dataset.json --out evals/gen50/run-answer.json
```

关注：`pass_rate` / `correctness` / `faithfulness` / `refusal_rate_oos` / `false_refusal_rate`。
