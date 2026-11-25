"""
Day 35 — Standalone script test showing atomic Token Bucket decisions from live Redis.

Prerequisites:
    docker compose up -d redis

Usage:
    .\venv\Scripts\python scripts/test_redis_limiter.py
"""

import asyncio
import sys
from pathlib import Path

import redis.asyncio as aioredis

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.ratelimit import RateLimiter


async def main() -> None:
    redis_url = "redis://localhost:6379/0"
    client = aioredis.from_url(redis_url, decode_responses=True)

    try:
        await client.ping()
        print(f"Connected to Redis at {redis_url}")
    except Exception as exc:
        print(f"Could not connect to Redis at {redis_url}: {exc}")
        print("Start Redis first with: docker compose up -d redis")
        await client.aclose()
        return

    test_key = "demo_script_key"
    await client.delete(f"ratelimit:{test_key}")

    limiter = RateLimiter(redis_client=client, default_capacity=3, default_refill_rate=1.0)

    print("\n--- Testing Burst (capacity=3, refill=1/sec) ---")
    for i in range(1, 5):
        allowed, retry_after = await limiter.check(test_key)
        print(f"Request {i}: allowed={allowed}, retry_after={retry_after:.3f}s, remaining={limiter.last_remaining}")

    print("\n--- Sleeping 1.1s for 1 token refill ---")
    await asyncio.sleep(1.1)
    allowed, retry_after = await limiter.check(test_key)
    print(f"Request 5 (after refill): allowed={allowed}, retry_after={retry_after:.3f}s, remaining={limiter.last_remaining}")

    stored = await client.hgetall(f"ratelimit:{test_key}")
    print(f"\nRedis Hash state at 'ratelimit:{test_key}': {stored}")

    await client.delete(f"ratelimit:{test_key}")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
