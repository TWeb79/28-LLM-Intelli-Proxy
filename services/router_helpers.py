"""
Utility helpers extracted from the router to keep the main file small.

Contains lightweight helpers used by the IntelligentRouter during warmup
and model categorization. Kept intentionally minimal and dependency free
so they can be tested independently.
"""
import logging
from typing import List


async def warm_up_models_async(client, models: List[str], base_url: str, timeout: int = 30):
    """Warm up a list of models by calling the provider's generate endpoint.

    Args:
        client: an async HTTP client with a .post() coroutine (httpx.AsyncClient)
        models: list of model names to warm up
        base_url: base URL of the provider (e.g. AirLLM or Ollama)
        timeout: request timeout in seconds
    """
    for model in models:
        try:
            await client.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": "warmup", "stream": False},
                timeout=timeout,
            )
            logging.info(f"🔥 Warmed up model: {model}")
        except Exception:
            logging.debug(f"Failed to warm up model: {model}", exc_info=True)


def categorize_models_by_keyword(model_names: List[str]):
    """Simple categorizer used during discovery to group models.

    Returns a dict of category -> list of model names.
    """
    categories = {"code": [], "vision": [], "reasoning": [], "general": []}
    keywords = {
        "code": ["coder", "code"],
        "vision": ["llava", "vision"],
        "reasoning": ["deepseek", "r1"],
    }

    for name in model_names:
        placed = False
        nl = name.lower()
        for cat, words in keywords.items():
            if any(w in nl for w in words):
                categories[cat].append(name)
                placed = True
                break
        if not placed:
            categories["general"].append(name)

    return categories
