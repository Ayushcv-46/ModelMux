import asyncio
import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from app.config import settings

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)


async def main():
    api_key = settings.gemini_api_key

    if not api_key:
        print("\n[!] ERROR: GEMINI_API_KEY is not set.")
        print("Steps:")
        print("1. Get a free API key at: https://aistudio.google.com/app/apikey")
        print("2. Create a file named .env in the project root containing:")
        print("   GEMINI_API_KEY=your_actual_api_key_here\n")
        return

    # Payload matching Gemini's REST API specifications
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "when is dune 3 going to be released."}],
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 100,
        },
    }

    print(f"[*] Sending raw request to Gemini ({GEMINI_MODEL})...")
    start_time = time.perf_counter()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GEMINI_URL,
            params={"key": api_key},
            json=payload,
        )

    latency_ms = (time.perf_counter() - start_time) * 1000

    print(f"[*] Response Status Code: {response.status_code}")

    if response.status_code != 200:
        print("[!] Error response from Gemini:")
        print(response.text)
        if response.status_code == 404:
            print("\n[*] Fetching available models for your API key...")
            async with httpx.AsyncClient(timeout=10.0) as client:
                m_res = await client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": api_key},
                )
                if m_res.status_code == 200:
                    models = [
                        m["name"].replace("models/", "")
                        for m in m_res.json().get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    print("Available models you can use in GEMINI_MODEL:")
                    for m in models[:10]:
                        print(f"  - {m}")
        return

    data = response.json()
    print("\n--- Raw Gemini Response JSON ---")
    print(json.dumps(data, indent=2))

    # Extract response text and token counts
    candidate = data.get("candidates", [{}])[0]
    content = candidate.get("content", {})
    parts = content.get("parts", [{}])
    generated_text = parts[0].get("text", "")

    usage = data.get("usageMetadata", {})
    tokens_in = usage.get("promptTokenCount", 0)
    tokens_out = usage.get("candidatesTokenCount", 0)

    print("\n--- Parsed Fields (Mapping to ModelMux ChatResponse) ---")
    print(f"Text       : {generated_text.strip()}")
    print(f"Provider   : gemini")
    print(f"Model      : {GEMINI_MODEL}")
    print(f"Tokens In  : {tokens_in}")
    print(f"Tokens Out : {tokens_out}")
    print(f"Latency    : {latency_ms:.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
