"""限流：按钱包地址限制 /api/ask 的调用（令牌桶 + 每日上限 + 并发数）。

学习笔记
- 为什么要限流：每次提问都要调用 DeepSeek（花钱）和 embedding，没有限流的话，一个脚本就能刷光余额。
- 三条规则（默认值见 config.py）：
  1. 令牌桶：桶容量 3（允许连发 3 次），每分钟补充 10 个令牌。短时间连问几句没问题，持续刷会被拦住。
     对比其他算法：固定窗口在窗口交界处可能放进 2 倍流量；滑动窗口更平滑；
     令牌桶允许短时间突发、长期限制速度，最适合聊天场景。
  2. 每日上限：每个地址每天 200 次，零点重置。
  3. 并发数：同一个地址同时最多 2 个流式回答（长连接），回答结束时释放；
     客户端中途断开、没能释放的名额，5 分钟后自动回收。
- 先检查全部规则再扣减额度：被拒绝的请求不消耗额度。
  超限返回 429，并带 Retry-After 响应头（秒数），前端据此提示"N 秒后再试"。
- 局限：计数存在进程内存里，进程重启就清零；多进程或多实例时各算各的，限流会失效，到那时换成 Redis。
  所以线上只跑 1 个实例、1 个 worker。正式上线时还可以在 Nginx 或网关层再加一道限流。
- 测试：tests/test_ratelimit.py 用假时钟验证突发、补充、并发回收、跨天重置。
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
