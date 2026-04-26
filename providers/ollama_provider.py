"""
Ollama provider adapter implementing minimal LLMProvider contract.

This adapter is intentionally lightweight: list_models() normalizes the /api/tags
response into a list of dicts compatible with the registry upsert call.
forward_request() proxies generate/chat requests to the configured Ollama base URL.
"""
from typing import List, Dict, Any, Optional
import os
import httpx
from time import time

DEFAULT_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


class OllamaProvider:
    def __init__(self, name: str = "ollama", base_url: Optional[str] = None, refresh_interval_minutes: int = 15):
        self.name = name
        self.base_url = (base_url or DEFAULT_BASE).rstrip("/")
        self.refresh_interval_minutes = refresh_interval_minutes

    async def list_models(self) -> List[Dict[str, Any]]:
        """Call /api/tags and normalize results."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", []) if isinstance(data, dict) else []
            result = []
            for m in models:
                result.append({
                    "id": m.get("name"),
                    "provider": self.name,
                    "source_url": self.base_url,
                    "category": None,
                    "description": m.get("description") or None,
                    "context_window": m.get("context_window") or None,
                })
            return result

    async def forward_request(self, model_id: str, payload: Dict[str, Any], stream: bool = False, endpoint: str = "/api/generate") -> Dict[str, Any]:
        """Forward generate/chat request to Ollama. Endpoint can be '/api/generate' or '/api/chat'.

        Returns JSON response body.
        """
        url = f"{self.base_url}{endpoint}"
        body = dict(payload or {})
        # ensure the model key is present when required
        if 'model' not in body:
            body['model'] = model_id

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return resp.json()
