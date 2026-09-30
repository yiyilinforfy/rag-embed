"""Per-user limits for the ask endpoints: token bucket (rate + burst), daily quota, concurrent streams.

In-memory, so it only holds within one process. With several workers / machines, move the counters to Redis.
"""

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from app.config import get_settings

# A stream slot is reclaimed after this long even if release() never ran (e.g. the client vanished mid-stream).
SLOT_TTL_SECONDS = 300


class RateLimited(Exception):
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.message = message
        self.retry_after = max(1, math.ceil(retry_after))


@dataclass
class _Bucket:
    tokens: float
    updated: float
    day: date
    used_today: int = 0
    slots: dict[int, float] = field(default_factory=dict)  # slot id -> acquired at


class RateLimiter:
    def __init__(
        self,
        per_minute: int,
        burst: int,
        daily: int,
        max_concurrent: int,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ):
        self.rate = per_minute / 60  # tokens per second
        self.burst = burst
        self.daily = daily
        self.max_concurrent = max_concurrent
        self.clock = clock
        self.now = now
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._next_slot = 0

    def acquire(self, key: str) -> Callable[[], None]:
        """Take one request from `key`'s budget or raise RateLimited. Call the returned function when done."""
        with self._lock:
            t = self.clock()
            today = self.now().date()
            bucket = self._buckets.setdefault(key, _Bucket(tokens=self.burst, updated=t, day=today))

            bucket.slots = {i: at for i, at in bucket.slots.items() if t - at < SLOT_TTL_SECONDS}
            if bucket.day != today:
                bucket.day, bucket.used_today = today, 0
            bucket.tokens = min(self.burst, bucket.tokens + (t - bucket.updated) * self.rate)
            bucket.updated = t

            # All checks before any deduction: a rejected request must not use up quota.
            if len(bucket.slots) >= self.max_concurrent:
                raise RateLimited(f"同时进行的回答最多 {self.max_concurrent} 个，请等当前回答结束", 5)
            if bucket.used_today >= self.daily:
                midnight = datetime.combine(today + timedelta(days=1), datetime.min.time())
                raise RateLimited(f"今日提问次数已达上限（{self.daily} 次），请明天再试", (midnight - self.now()).total_seconds())
            if bucket.tokens < 1:
                wait = (1 - bucket.tokens) / self.rate
                raise RateLimited(f"提问太频繁，请 {math.ceil(wait)} 秒后再试", wait)

            bucket.tokens -= 1
            bucket.used_today += 1
            self._next_slot += 1
            slot = self._next_slot
            bucket.slots[slot] = t

        def release() -> None:
            with self._lock:
                bucket.slots.pop(slot, None)

        return release


def _from_settings() -> RateLimiter:
    s = get_settings()
    return RateLimiter(s.rate_limit_per_minute, s.rate_limit_burst, s.rate_limit_daily, s.max_concurrent_streams)


ask_limiter = _from_settings()
