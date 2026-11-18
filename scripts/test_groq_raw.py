import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from app.config import settings


async def main():
    api_key = settings.groq_api_key
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "When did the Titanic sink?"}],
        "max_tokens": 512,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.post(url, headers=headers, json=payload)

    print("Status:", res.status_code)
    data = res.json()
    print("Full choice message:")
    print(json.dumps(data["choices"][0]["message"], indent=2))
    print("Usage:", data.get("usage"))


if __name__ == "__main__":
    asyncio.run(main())
