#!/bin/sh
# 学习笔记：容器启动脚本，先做准备工作，再执行 Dockerfile 里的 CMD（uvicorn）。
# - 容器的文件系统每次启动都是全新的，文档不在镜像里，所以启动时克隆一份（--depth 1 只拉最新提交，速度快）。
# - 文档仓库是私有的：DOCS_REPO_TOKEN 通过 git 凭据助手传给 git，不会写进 URL 或日志。
# - 克隆失败不会让容器退出：已经入库的数据在 Supabase 里，问答照常工作，以后的同步会重试。
#   ingest 里有保护：一篇文档都没有时拒绝入库，不会把索引清空。
# - exec "$@"：用 CMD 替换当前 shell 进程，uvicorn 成为 1 号进程，能直接收到停止信号，正常退出。
set -e

REPO_DIR=/app/data/vcloud-doc
REPO_URL=${DOCS_REPO_URL:-https://github.com/virtualeconomy/vcloud-doc.git}
BRANCH=${DOCS_BRANCH:-dev}

# Private repo: a read-only GitHub token, handed to git through a credential helper so it never lands
# in the remote URL, .git/config or error messages. Also used by later fetches from /api/sync.
if [ -n "${DOCS_REPO_TOKEN:-}" ]; then
    git config --global credential.helper \
        '!f() { echo username=x-access-token; echo "password=$DOCS_REPO_TOKEN"; }; f'
fi

# The container filesystem is fresh on every start: fetch the docs so sync / ingest have files to read.
# Unchanged files are skipped by content hash, so this never re-embeds what's already in the database.
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "cloning $REPO_URL ($BRANCH)"
    git clone --quiet --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR" \
        || echo "WARNING: docs clone failed; answering still works, sync will retry later"
fi

exec "$@"
