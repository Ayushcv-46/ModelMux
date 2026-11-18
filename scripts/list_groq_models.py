import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from app.config import settings


async def main():
    api_key = settings.groq_api_key
    if not api_key:
        print("[!] GROQ_API_KEY is missing from .env")
        return

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    print("[*] Fetching available models from Groq...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.get(url, headers=headers)

    print(f"Status: {res.status_code}")
    if res.status_code != 200:
        print("Error:", res.text)
        return

    data = res.json()
    models = sorted([m["id"] for m in data.get("data", [])])
    print(f"\n[+] Found {len(models)} models available on your Groq key:")
    for m in models:
        print(f"  - {m}")


if __name__ == "__main__":
    asyncio.run(main())
