"""旧的文档级检索评测（22 题）。新的完整评测见 evals/：

    uv run python -m evals.run --retrieval-only
    uv run python -m evals.run

本脚本保留是为了和历史数字对比（本地 hit@1 = 21/22，MRR = 0.970）。
"""

from app.search import search

TOP_K = 5

# (question, files that count as a correct answer)
QUESTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("怎么用工作流自动部署我的服务", ("zh/vkube_cli/workflows.md",)),
    ("如何通过ssh远程连接到容器", ("zh/VKservice/ssh.md",)),
    ("怎么创建安全组", ("zh/VKservice/securityGroup.md",)),
    ("访问令牌怎么创建和使用", ("zh/VKservice/accessToken.md",)),
    ("如何批量更新服务里的配置文件", ("zh/VKservice/configuration.md",)),
    ("日志查询要怎么添加数据源", ("zh/log-hosting/view-logs.md",)),
    ("怎么把服务绑定到日志抓取", ("zh/log-hosting/bind-containers.md",)),
    ("node watcher 的 topic 名称是什么结构", ("zh/node-watcher/sub-topic.md",)),
    ("node watcher 服务怎么充值", ("zh/node-watcher/topup.md",)),
    ("订单可以提前解约吗", ("zh/guide/advance-termination.md",)),
    ("服务到期时间是怎么计算的", ("zh/guide/service-expired.md",)),
    ("什么情况下可以退款", ("zh/supplier/refund.md",)),
    ("怎么在 Claude Code 里安装 vcloud skill", ("zh/guide/vcloud-skills.md",)),
    ("怎么安装 vkube 命令行工具", ("zh/vkube_cli/get-started.md",)),
    ("What is VKube?", ("zh/VKservice/FAQ.md",)),
    ("怎么让几个服务之间只能走内网互相访问", ("zh/VKservice/securityGroup.md",)),
    ("我往 GitHub 推代码之后能自动发布新版本吗", ("zh/vkube_cli/workflows.md",)),
    ("VKube 支持哪些操作系统", ("zh/VKservice/FAQ.md",)),
    ("资源单位是什么意思", ("zh/VKservice/FAQ.md", "zh/VKservice/assetsDetail.md")),
    ("怎么给容器绑定我自己的域名", ("zh/VKservice/Full-V-Kube.md",)),
    ("能不能在浏览器里直接打开容器的终端", ("zh/VKservice/Full-V-Kube.md", "zh/VKservice/ssh.md")),
    ("钱包余额不够了怎么续费", ("zh/node-watcher/topup.md", "zh/VKservice/FAQ.md", "zh/guide/buying-process.md")),
]


def main() -> None:
    hits_at_1 = hits_at_k = 0
    reciprocal_ranks = []

    for question, expected in QUESTIONS:
        sources = [r.source for r in search(question, TOP_K)]
        rank = next((i for i, s in enumerate(sources, start=1) if s in expected), None)

        hits_at_1 += rank == 1
        hits_at_k += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0)

        mark = f"rank {rank}" if rank else "MISS  "
        print(f"[{mark}] {question}")
        if rank != 1:
            print(f"          expected {list(expected)}, got {sources[:3]}")

    n = len(QUESTIONS)
    print(f"\nhit@1 = {hits_at_1}/{n}   hit@{TOP_K} = {hits_at_k}/{n}   MRR = {sum(reciprocal_ranks) / n:.3f}")
    print("(完整评测：uv run python -m evals.run --retrieval-only)")


if __name__ == "__main__":
    main()
