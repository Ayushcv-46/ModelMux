"""
Day 34 — Pure Python Token Bucket Unit Tests

Proves:
1. Burst behavior: full bucket allows `capacity` immediate requests at t=0.
2. Empty behavior: once empty, requests are rejected with accurate `retry_after`.
3. Refill behavior: advancing injected clock refills tokens at `refill_rate`.
4. Cap behavior: tokens never exceed `capacity` even after long idle periods.
5. Fractional refill accumulation and backward-clock safety.
6. Parameter validation and clock_fn injection.
"""

import pytest

from app.ratelimit.token_bucket import TokenBucket


def test_burst_up_to_capacity():
    """A fresh bucket starts full and allows `capacity` requests in an instant burst."""
    bucket = TokenBucket(capacity=5, refill_rate=1.0)

    for _ in range(5):
        allowed, retry_after = bucket.allow(now=100.0)
        assert allowed is True
        assert retry_after == 0.0

    assert bucket.remaining == 0


def test_empty_bucket_rejects_with_retry_after():
    """Once the bucket is empty, the next call is rejected with exact retry_after seconds."""
    bucket = TokenBucket(capacity=3, refill_rate=2.0)  # 2 tokens/sec -> 0.5s per token

    # Drain all 3 tokens at t=10.0
    for _ in range(3):
        allowed, _ = bucket.allow(now=10.0)
        assert allowed is True

    # 4th call at the same instant is rejected; needs 1.0 token at 2.0 tokens/sec = 0.5s
    allowed, retry_after = bucket.allow(now=10.0)
    assert allowed is False
    assert retry_after == pytest.approx(0.5)


def test_refill_over_time():
    """Advancing time refills tokens at `refill_rate` and allows requests again."""
    bucket = TokenBucket(capacity=4, refill_rate=2.0)  # 2 tokens/sec

    # Drain all 4 tokens at t=0.0
    for _ in range(4):
        assert bucket.allow(now=0.0)[0] is True

    # Empty at t=0.0
    assert bucket.allow(now=0.0)[0] is False

    # Advance 0.25s -> 0.5 tokens refilled (still < 1.0, rejected, 0.25s left)
    allowed, retry_after = bucket.allow(now=0.25)
    assert allowed is False
    assert retry_after == pytest.approx(0.25)

    # Advance another 0.25s (total 0.5s -> 1.0 token refilled) -> allowed!
    allowed, retry_after = bucket.allow(now=0.50)
    assert allowed is True
    assert retry_after == 0.0
    assert bucket.remaining == 0


def test_tokens_capped_at_capacity():
    """Long idle periods must not refill tokens past `capacity`."""
    bucket = TokenBucket(capacity=5, refill_rate=10.0)

    # Drain 2 tokens at t=0.0 (3 left)
    bucket.allow(now=0.0)
    bucket.allow(now=0.0)
    assert bucket.remaining == 3

    # Wait 1000 seconds -> would be 10,000 tokens, but capped at 5
    allowed, _ = bucket.allow(now=1000.0)
    assert allowed is True
    assert bucket.remaining == 4  # 5 capped - 1 consumed = 4


def test_clock_fn_injection():
    """Bucket works seamlessly with an injected `clock_fn` when `now` is omitted."""
    fake_time = 50.0
    bucket = TokenBucket(capacity=2, refill_rate=1.0, clock_fn=lambda: fake_time)

    assert bucket.allow()[0] is True
    assert bucket.allow()[0] is True
    assert bucket.allow()[0] is False

    fake_time += 1.0
    assert bucket.allow()[0] is True


def test_backward_clock_does_not_subtract_tokens():
    """If clock skews slightly backward, tokens must not become negative or decrease."""
    bucket = TokenBucket(capacity=3, refill_rate=1.0)
    assert bucket.allow(now=100.0)[0] is True  # 2 tokens left

    # Clock goes backward to 95.0 -> elapsed clamped to 0.0
    allowed, _ = bucket.allow(now=95.0)
    assert allowed is True
    assert bucket.remaining == 1


def test_invalid_parameters_raise_value_error():
    """Negative or zero capacity, refill_rate, or cost must raise ValueError."""
    with pytest.raises(ValueError, match="capacity"):
        TokenBucket(capacity=0, refill_rate=1.0)

    with pytest.raises(ValueError, match="refill_rate"):
        TokenBucket(capacity=5, refill_rate=0)

    bucket = TokenBucket(capacity=5, refill_rate=1.0)
    with pytest.raises(ValueError, match="cost"):
        bucket.allow(now=0.0, cost=0)
