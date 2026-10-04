"""Shared LLM call helper for the procedural memory tier.

MemMachine's LanguageModel interface (OpenAIResponses, OpenAIChatCompletions,
AmazonBedrock, LiteLLM) exposes ``generate_response(system_prompt=, user_prompt=)``
returning ``(text, tool_outputs)``. Earlier procedural code assumed a raw
OpenAI client (``.chat.completions``) or a ``.generate()`` method, neither of
which MemMachine models have -- so every call silently fell through.
"""

from __future__ import annotations

import json
import re
from typing import Any


async def llm_complete(llm: Any, prompt: str) -> str:
    """Return the text completion for ``prompt`` from any supported client.

    Raises TypeError for an unsupported client so callers never mistake an
    interface mismatch for a legitimate empty/negative result.
    """
    if hasattr(llm, "generate_response"):
        # MemMachine LanguageModel (primary path)
        text, _ = await llm.generate_response(user_prompt=prompt)
        return text or ""
    if hasattr(llm, "chat") and hasattr(llm.chat, "completions"):
        # Raw OpenAI-compatible client (kept for external callers)
        response = await llm.chat.completions.create(
            model=getattr(llm, "model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    raise TypeError(f"Unsupported LLM client type: {type(llm).__name__}")


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract and parse the first JSON object in ``raw`` (handles prose wrapping)."""
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group(0) if match else raw)
