#!/bin/sh
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
