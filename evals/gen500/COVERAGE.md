# P0 文档覆盖（检索评测 gen500）

chunk 目录来自 `uv run python -m evals.label chunks`：当前切分 **206** 个 `(source, heading)`，分布在 `data/vcloud-doc/docs/zh` 的 **39** 篇正文（另有仓库根 README，未进 zh 子目录切分）。

检索策略固定为 hybrid：向量 Top-20 ∪ BM25 Top-20 → `BAAI/bge-reranker-v2-m3` → Top-5，再过 `MIN_SCORE=0.45` 与 `RERANK_MIN_SCORE=0.05`。冒烟三题（`cli-install`、`dev-refundvsys-fn`、`oos-weather`）已确认策略名为 hybrid，rerank 有耗时，没有退回向量。

## 已出题（400）

在最初 100 题之后又加了 300 题，合计 400。新增部分多补了精确词、文档里没有的问题，以及带上一轮对话的追问，并覆盖了购买流程、用户 CLI、VKubefile、平台 API、商品模板、日志和 Topic 订阅这些原先没出过的段落。

400 题的 hybrid 结果见 `REPORT.md`。离原先规划的 500 题还差约 100。

## 最初 100 题的配比（已被上面的 400 题取代）

| 题型 | 题数 | 500 题目标里的比例 |
|------|------|-------------------|
| direct | 21 | 100 |
| paraphrase | 21 | 100 |
| exact_term | 26 | 120 |
| multi_doc | 8 | 40 |
| english | 8 | 40 |
| multi_turn | 8 | 40 |
| out_of_scope | 8 | 40 |

P1 标注碰到的 source（约 30 篇）：VKservice 的 Example、FAQ、Full-V-Kube、accessToken、assetsDetail、configuration、getting-started、securityGroup、ssh；developers 的 lua_scripts、platform_api、sdk_api；guide 的 advance-termination、service-expired、vcloud-skills；log-hosting 的介绍、购买、Grafana 数据源；node-watcher 的介绍、Topic、充值；supplier 的 README、advanced、config-sdk-service、duration-services、manage-product、price-auditing、refund；vkube_cli 的 get-started、workflows。

## P2/P3 尚未出题的文档

购买流程、用户 CLI 安装（`guide/getting-started.md`）、`use-service`、日志绑定容器、vkube `configureFile`、商品 yaml 模板、平台 API 里订单查询/创建服务商等大段、SDK 的 User Service 列表字段、工作流具体示例、各 README 短文。这些留给 P2（补 exact_term / OOS / multi_turn）和 P3。
