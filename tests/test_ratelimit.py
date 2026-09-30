from datetime import datetime, timedelta

import pytest

from app.ratelimit import SLOT_TTL_SECONDS, RateLimited, RateLimiter


class FakeTime:
    def __init__(self):
        self.t = 1000.0
        self.day = datetime(2026, 9, 30, 12, 0, 0)

    def clock(self) -> float:
        return self.t

    def now(self) -> datetime:
        return self.day + timedelta(seconds=self.t - 1000.0)


def make(fake, per_minute=10, burst=3, daily=100, max_concurrent=5) -> RateLimiter:
    return RateLimiter(per_minute, burst, daily, max_concurrent, clock=fake.clock, now=fake.now)


def test_burst_then_refill_at_rate():
    fake = FakeTime()
    limiter = make(fake)  # 10/min = one token every 6s, burst 3
    for _ in range(3):
        limiter.acquire("A")()

    with pytest.raises(RateLimited) as exc:
        limiter.acquire("A")
    assert exc.value.retry_after == 6

    fake.t += 6
    limiter.acquire("A")()


def test_users_have_separate_buckets():
    fake = FakeTime()
    limiter = make(fake, burst=1)
    limiter.acquire("A")()
    with pytest.raises(RateLimited):
        limiter.acquire("A")
    limiter.acquire("B")()


def test_concurrent_slots_are_released_or_expire():
    fake = FakeTime()
    limiter = make(fake, per_minute=600, burst=10, max_concurrent=2)
    release_1 = limiter.acquire("A")
    limiter.acquire("A")  # never released: the client disappeared

    with pytest.raises(RateLimited, match="同时进行"):
        limiter.acquire("A")

    release_1()
    release_3 = limiter.acquire("A")
    release_3()

    fake.t += SLOT_TTL_SECONDS
    limiter.acquire("A")  # the leaked slot has expired
    limiter.acquire("A")


def test_daily_quota_resets_next_day_and_rejections_cost_nothing():
    fake = FakeTime()
    limiter = make(fake, per_minute=600, burst=10, daily=2)
    limiter.acquire("A")()
    limiter.acquire("A")()

    for _ in range(3):
        with pytest.raises(RateLimited, match="今日"):
            limiter.acquire("A")

    fake.t += 13 * 3600  # past midnight
    limiter.acquire("A")()
    limiter.acquire("A")()
