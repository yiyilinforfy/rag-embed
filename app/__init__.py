"""VKube 文档 RAG 后端 —— 学习总览

推荐阅读顺序：
    本文件 → config → models → chunker → embedding → ingest → search → rag → llm
    → main → auth / ratelimit → sync → Dockerfile / docker/entrypoint.sh
每个模块开头都有一段「学习笔记」：它在流程里的位置、为什么这样设计、踩过的坑、可以继续进阶的方向。


一、RAG 是什么
    大模型不知道 VKube 文档里写了什么。RAG（Retrieval-Augmented Generation，检索增强生成）的做法是：
    先从文档里检索出和问题最相关的几段原文，放进 prompt，让大模型"看着资料回答"。
    好处：不用训练模型；文档更新后重新入库即可；回答能附上原文链接，用户可以核对。
    本项目刻意不用 LangChain / LlamaIndex，每一步都手写，方便看清原理。


二、整体流程（两条线）

    【离线：建索引】文档 → 向量库
        GitHub 私有仓库 virtualeconomy/vcloud-doc（dev 分支，只索引 docs/zh）
          → sync.py       git fetch 拉取最新文档
          → chunker.py    Markdown 按标题切成 chunk（纯 Python，不需要模型）
          → embedding.py  每个 chunk 用 bge-m3 转成 1024 维向量
          → ingest.py     写入 PostgreSQL + pgvector（documents / chunks 两张表）

    【在线：回答问题】问题 → 答案
        Vue 前端页面（请求头带钱包地址 X-Wallet-Address，请求体带最近几轮对话）
          → main.py       FastAPI 接口 POST /api/ask/stream
          → auth.py       地址在不在 users 表（401 / 403）
          → ratelimit.py  按地址限流（429）
          → rag.py        截断历史；有历史时让 LLM 把追问改写成独立问题
          → search.py     问题转向量，pgvector 按余弦距离取 Top-K
          → rag.py        分数都低于阈值 → 直接回复"没找到"（不调 LLM）
                          否则拼 prompt：历史 + 检索到的资料 + 问题
          → llm.py        DeepSeek 流式生成
          → main.py       SSE 推给前端：先推参考来源，再逐字推回答


三、关键概念速查
    chunk        文档切出来的一小段，检索的最小单位。太大 → 噪声多、向量语义被稀释；太小 → 缺上下文。
                 本项目上限约 1000 字。
    embedding    把文字映射成一串数字（这里 1024 个），语义相近的文字，向量方向也相近。
                 建索引和查询必须用同一个模型，否则两边的向量没法比较。
    余弦距离     pgvector 的 `<=>` 运算符：0 = 方向相同，1 = 无关，2 = 相反。本项目 score = 1 - distance。
    Top-K        取最相似的 K 个 chunk（默认 5）。
    MIN_SCORE    相似度阈值 0.45。实测 bge-m3 在这些文档上：相关问题 0.64~0.77，无关问题 0.31~0.42。
    两层兜底     ① 分数不够 → 不调 LLM，直接说没找到；
                 ② 分数够但资料答不了 → prompt 规则要求 LLM 如实说"文档没有提供足够信息"。
    分工         bge-m3 只负责"文字 → 向量"；比较和排序是数据库（pgvector）做的；DeepSeek 只负责"读资料写答案"。
    向量检索     优势是"更准"（能匹配同义表达），不是"更快"。267 行是全表扫描，约 2ms；
                 数据量到几万行以上才需要 HNSW 近似索引。


四、数据
    39 篇中文文档 → 267 个 chunk。每个 chunk 存：content、source、title、heading、url（带锚点）、metadata、embedding。
    增量更新有三层：git 提交号没变 → 没有新内容；文件 content_hash 没变 → 整篇跳过；chunk 文本没变 → 复用旧向量。
    检索评测（scripts/eval_search.py，22 道题）：本地 hit@1 = 21/22，hit@5 = 22/22，MRR = 0.970；
    线上组合（Supabase + 硅基流动）hit@1 = 20/22，hit@5 = 22/22。


五、项目是怎么一步步搭起来的（也是推荐的学习路线）
     1  项目结构：uv 管理 Python 3.12 + FastAPI
     2  Docker Compose 启动 PostgreSQL + pgvector（本地开发用，docker-compose.yml）
     3  数据模型：documents / chunks（models.py）
     4  Markdown Chunker：按标题切 + 子树合并 + 长代码截断 + 对齐 VuePress 锚点规则（chunker.py）
     5  Embedding + 入库：本地 Ollama bge-m3（embedding.py、ingest.py）
     6  向量检索 + 评测 + 阈值校准（search.py、scripts/eval_search.py）
     7  RAG Pipeline：prompt 模板、阈值兜底、参考来源（rag.py、llm.py）
     8  FastAPI 接口 + SSE 流式 + 接入 Vue 前端，前端用 localStorage 缓存对话（main.py）
    之后的进阶：
     9  文档同步：git fetch + 三层增量；7 天定时 / webhook / 手动接口（sync.py）
    10  多轮对话：前端带历史、后端截断、LLM 改写追问 query rewriting（rag.py）
    11  鉴权 + 限流：钱包地址白名单 + 令牌桶 / 每日上限 / 并发数（auth.py、ratelimit.py）
    12  部署：Docker 镜像 → Docker Hub → VKube container service；
        数据库换成 Supabase，embedding 换成硅基流动（Dockerfile、scripts/push_db.sh、scripts/check_embedding.py）


六、线上架构
    VKube container service（北美，1 个实例）运行镜像 tommylyy/vkube-rag
      ├─ Supabase（加拿大）      托管 PostgreSQL + pgvector，存文档、向量、用户
      ├─ 硅基流动 BAAI/bge-m3    问题转向量。免费（带 Pro/ 前缀的才收费）；
      │                          和本地 Ollama bge-m3 实测相似度 0.9999，本地算好的向量直接迁移复用
      ├─ DeepSeek                生成回答，整套系统里唯一花钱的部分
      └─ GitHub 私有仓库         文档来源，容器里用只读 token 克隆
    前端：Vue 页面通过 VUE_APP_AGENT_API_URL 调用后端。
    配置：本地开发用 .env（Docker 里的 PostgreSQL + 本机 Ollama），线上配置在 .env.production，都不提交到 git。


七、已知局限和下一步（适合继续学习的方向）
    检索     混合检索（向量 + 全文检索 BM25，用 RRF 合并排名）；Reranker（bge-reranker-v2-m3 重新打分）
    切分     父子 chunk（用小块检索、返回所在的大块）；给小块补充上下文
    生成     prompt 要求 [n] 引用标注、回答语言规则、防 prompt 注入
    评测     扩大题库；评测回答质量（是否忠于资料、有没有答对）
    鉴权     从"地址白名单"升级为"钱包签名验证"（地址是公开的，现在可以被冒充）
    扩容     多实例时限流计数换成 Redis；数据量大时给向量建 HNSW 索引
    Agent    Agentic RAG：检索效果不好时让 Agent 改写重搜、拆分问题，或调用 VKube API 查询真实状态
"""
