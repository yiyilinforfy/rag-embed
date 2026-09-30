"""Pull the docs repo from GitHub and re-index whatever changed.

Triggered by scripts/sync_docs.py (cron), the periodic task in app.main, POST /api/sync, or the GitHub webhook.
"""

import hashlib
import hmac
import subprocess
import threading
import time
from pathlib import Path

from app.config import get_settings
from app.ingest import ingest

# One ingest at a time per process: a webhook arriving during the periodic sync must not index twice.
_lock = threading.Lock()


class SyncError(Exception):
    pass


class SyncBusy(SyncError):
    pass


def _last_sync_file() -> Path:
    return get_settings().docs_repo_dir.parent / ".last_sync"


def sync_is_due(interval_minutes: int) -> bool:
    """Compare against the last successful sync on disk, so API restarts don't reset a 7-day schedule."""
    try:
        last = float(_last_sync_file().read_text())
    except (FileNotFoundError, ValueError):
        return True
    return time.time() - last >= interval_minutes * 60


def _git(*args: str) -> str:
    repo = get_settings().docs_repo_dir
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise SyncError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def reindex(force: bool = False) -> dict:
    """Re-index the working tree as it is (no git pull)."""
    if not _lock.acquire(blocking=False):
        raise SyncBusy("another sync / ingest is running")
    try:
        return ingest(force=force)
    finally:
        _lock.release()


def sync_docs(force: bool = False) -> dict:
    settings = get_settings()
    if not _lock.acquire(blocking=False):
        raise SyncBusy("another sync / ingest is running")
    try:
        before = _git("rev-parse", "HEAD")
        _git("fetch", "--quiet", "origin", settings.docs_branch)
        after = _git("rev-parse", "FETCH_HEAD")

        changed_files: list[str] = []
        if before != after:
            changed_files = _git(
                "diff", "--name-only", before, after, "--", f"docs/{settings.docs_subdir}"
            ).splitlines()
            # The clone belongs to this service and is never edited by hand, so moving it is safe.
            _git("checkout", "--quiet", "-B", settings.docs_branch, after)

        # Always run: it's cheap when nothing changed (hash compare only) and it recovers from
        # a previous run that moved the checkout but failed half-way through embedding.
        stats = ingest(force=force)
        _last_sync_file().write_text(str(time.time()))
        return {"from": before[:7], "to": after[:7], "changed_files": changed_files, **stats}
    finally:
        _lock.release()


def verify_github_signature(secret: str, body: bytes, signature: str) -> bool:
    """Check the X-Hub-Signature-256 header GitHub sends with every webhook delivery."""
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")
