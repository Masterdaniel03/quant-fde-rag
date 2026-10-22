import asyncio
import os

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError

MODEL_NAME = "gemini-3.8-flash"
GENERATION_TIMEOUT_SECONDS = 30

api_key = os.environ.get("GEMINI_API_KEY")
client = (
    genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GENERATION_TIMEOUT_SECONDS * 1000),
    )
    if api_key
    else None
)


async def generate_draft(
    prompt: str,
) -> tuple[str | None, dict[str, int], str | None]:
    if client is None:
        return None, {}, "missing_api_key"

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=prompt,
            ),
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
    except (APIError, httpx.RequestError, TimeoutError) as exc:
        return None, {}, f"generation_failed:{type(exc).__name__}"

    if not response.text or not response.text.strip():
        return None, {}, "empty_response"

    usage = response.usage_metadata
    usage_metrics = {
        "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
        "output_tokens": (usage.candidates_token_count or 0) if usage else 0,
    }
    return response.text.strip(), usage_metrics, None