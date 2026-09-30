"""文档同步：从 GitHub 拉取最新文档，再做增量入库。

学习笔记
- 为什么需要：文档会更新，索引要跟着更新。三层增量（提交号 / 文件哈希 / chunk 文本）让同步很便宜：
  没有变化时一次同步只要几秒，不产生任何 embedding 调用。
- 流程：git rev-parse HEAD（当前版本）→ git fetch origin dev → 对比 FETCH_HEAD → 有新提交就切过去 → ingest()。
  即使没有新提交也会跑一次 ingest：没变化时很快，还能补上"上次拉了代码但入库失败"的情况。
- 四种触发方式（最终都调用 sync_docs）：
  1. 定时：main.py 里的后台任务，每小时检查一次距上次成功同步是否已满 SYNC_INTERVAL_MINUTES（线上是 7 天）。
     上次成功的时间记在 data/.last_sync 文件里，所以进程重启不会重新计时。
  2. 手动：POST /api/sync（需要 X-Admin-Token 请求头），或者命令行 scripts/sync_docs.py。
  3. GitHub webhook：仓库一有推送，GitHub 主动 POST 到 /api/webhooks/github；
     用 HMAC-SHA256 签名（verify_github_signature）确认请求确实来自 GitHub，而不是别人伪造的。
  4. GitHub Actions：在文档仓库里加一个工作流，推送后用 curl 调用 /api/sync（效果和 webhook 一样，还能按目录过滤）。
  webhook 和 Actions 都要求后端有公网地址。webhook 不属于 CI/CD，它只是"事件发生时通知你"的机制。
- 锁：同一进程里同一时间只允许一个同步或入库在跑，避免定时任务和 webhook 撞在一起重复入库（撞上时返回 409）。
- 文档仓库是私有的：容器里靠 DOCS_REPO_TOKEN（只读 token，通过 git 凭据助手传入，见 docker/entrypoint.sh）。
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
