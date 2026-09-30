#!/bin/sh
# Copy documents, chunks (with vectors) and users from the local Docker database to a fresh remote one.
#
#   TARGET_DATABASE_URL='postgresql+psycopg://postgres.xxxx:PASSWORD@aws-0-xx.pooler.supabase.com:5432/postgres?sslmode=require' \
#       sh scripts/push_db.sh
#
# 学习笔记：一次性数据迁移（本地 Docker PostgreSQL → Supabase）
# - 为什么迁移而不是在线上重新入库：向量是花时间算出来的，而且已经验证两边的模型兼容（check_embedding），直接搬走即可。
# - 步骤：在目标库建表（init_db）→ 目标库已经有数据就拒绝执行（防止重复导入）
#   → pg_dump 只导出数据 → 通过管道交给 psql 写入目标库。
# - pg_dump / psql 借用本地 rag-postgres 容器里的，本机不用安装 PostgreSQL 客户端。
# - Supabase 要用 Session pooler 的地址（支持 IPv4，端口 5432）；直连地址只支持 IPv6，6543 端口是事务模式。
set -eu

: "${TARGET_DATABASE_URL:?set TARGET_DATABASE_URL to the remote database}"
LOCAL_CONTAINER=${LOCAL_CONTAINER:-rag-postgres}
# psql wants plain postgresql://, SQLAlchemy wants postgresql+psycopg://
PSQL_URL=$(printf '%s' "$TARGET_DATABASE_URL" | sed 's#^postgresql+psycopg://#postgresql://#')

echo "1/3 creating the vector extension and tables on the target"
DATABASE_URL="$TARGET_DATABASE_URL" uv run python -m scripts.init_db > /dev/null

existing=$(docker exec "$LOCAL_CONTAINER" psql "$PSQL_URL" -tAc "select count(*) from documents")
if [ "$existing" != "0" ]; then
    echo "target already has $existing documents; refusing to copy over it" >&2
    exit 1
fi

echo "2/3 copying data (the local container's psql talks to the target)"
docker exec "$LOCAL_CONTAINER" pg_dump -U rag -d rag --data-only --no-owner --no-privileges \
    -t users -t documents -t chunks \
    | docker exec -i "$LOCAL_CONTAINER" psql "$PSQL_URL" -q -v ON_ERROR_STOP=1 > /dev/null

echo "3/3 row counts on the target"
docker exec "$LOCAL_CONTAINER" psql "$PSQL_URL" -tAc \
    "select 'documents', count(*) from documents union all
     select 'chunks', count(*) from chunks union all
     select 'users', count(*) from users"
