#!/bin/sh
# Copy documents, chunks (with vectors) and users from the local Docker database to a fresh remote one.
#
#   TARGET_DATABASE_URL='postgresql+psycopg://postgres.xxxx:PASSWORD@aws-0-xx.pooler.supabase.com:5432/postgres?sslmode=require' \
#       sh scripts/push_db.sh
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
