# 可观测性与 Ask 日志设计（中团队向）

## P95 是什么

把一段时间里每次请求的耗时从小到大排序，**第 95% 那个点**叫 P95。  
例如 100 次请求里，95 次都 ≤ 3.2s，则 P95≈3.2s。  
看平均值会被少数超慢请求骗；看 P95/P99 才像「大部分用户真实体感」。

## 中团队怎么监控（不是宝塔）

宝塔偏「机器/网站托管面板」。RAG/API 中团队更常见是：

| 层 | 存什么 | 典型工具 |
|----|--------|----------|
| **业务事实** | 每问一条（本设计的 `ask_events`）+ 用户反馈 | PostgreSQL |
| **指标 Metrics** | QPS、P95 延迟、拒答率、rerank fallback 率 | Prometheus + Grafana（或云监控） |
| **日志 Logs** | 异常堆栈、依赖失败细节 | Loki / ELK / 云日志；与 `ask_id` 关联 |
| **告警** | 破 SLO 发通知 | Grafana Alert / PagerDuty / 飞书机器人 |

推荐路径：

1. **真相在 Postgres**（ask_events / ask_feedback）——产品与排障都查它  
2. **Grafana** 用 SQL 或 Prometheus 画：P95 total_ms、拒答率、差评率  
3. 应用 stdout 仍打 JSON 一行（方便接入 Loki），`ask_id` 必带  

宝塔可以顺带看 CPU/磁盘，**不能替代**上述业务可观测。

## 数据模型原则

- **一次提问一个 `ask_id`（UUID）**：SSE、反馈、日志、指标用同一 ID 关联  
- **事件不可变，反馈可追加**：`ask_events` 只插入；`ask_feedback` 独立表  
- **写日志失败不能影响回答**（best-effort）  
- **配置快照**：事后知道当时门闸/扩写模式，避免「代码改了旧日志对不上」  
- **耗时拆段**：route / rewrite / expand / vector / bm25 / rerank / generate / first_token / total  

## 表

见 `app/models.py`：`AskEvent`、`AskFeedback`。

## API

- 回答响应 / SSE `meta` 事件带 `ask_id`  
- `POST /api/feedback`：`{ ask_id, helpful, reason?, comment? }`  

## SLO 示例（可后配告警）

- P95 `first_token_ms` < 4000（本地 Docker + DeepSeek 可再调）  
- P95 `total_ms` < 8000  
- `rerank_fallback` 率 < 5%（按小时）  
- 差评率日环比异常（人工先看）  

## Grafana / SQL 起步查询

```sql
-- 近 24h P95 总耗时 & 首 token
SELECT
  percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms) AS p95_total_ms,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY first_token_ms) AS p95_ttfb_ms,
  count(*) AS asks
FROM ask_events
WHERE created_at > now() - interval '24 hours'
  AND total_ms IS NOT NULL;

-- 拒答 / 门闸空结果率
SELECT
  count(*) FILTER (WHERE refused) AS refused,
  count(*) FILTER (WHERE gate_blocked) AS gate_blocked,
  count(*) AS total
FROM ask_events
WHERE created_at > now() - interval '24 hours';

-- 差评关联（最近）
SELECT e.id, e.question, e.route_strategy, e.total_ms, f.helpful, f.reason, f.comment
FROM ask_feedback f
JOIN ask_events e ON e.id = f.ask_id
WHERE f.helpful = false
ORDER BY f.created_at DESC
LIMIT 50;
```

配置项：`ASK_LOG_ENABLED` / `ASK_LOG_STORE_ANSWER` / `ASK_LOG_STDOUT`（见 `app/config.py`）。

## 本地 Grafana（已配好，不用先学 Prometheus）

```bash
cd rag-backend
docker compose up -d postgres grafana
```

1. 浏览器打开 http://127.0.0.1:3000  
2. 登录：`admin` / `admin`（首次会让你改密码，本地可 Skip）  
3. 左侧 **Dashboards → RAG → Ask Overview**（或首页就是它）  
4. 右上角时间范围默认「过去 24 小时」，刷新 10s  

面板包含：提问次数、P95 总耗时 / 首 Token、拒答率、门闸空结果率、差评率、时间序列、最近提问表、差评明细。

停掉：`docker compose stop grafana`（Postgres 可继续跑着给后端用）。
