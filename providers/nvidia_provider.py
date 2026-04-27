"""
NVIDIA NIM Provider adapter for LLM IntelliProxy.

Integrates with NVIDIA's NIM (NVIDIA Inference Microservices) API,
which provides OpenAI-compatible endpoints for NVIDIA-hosted models.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import httpx

from providers.base_provider import LLMProvider

# Default NVIDIA NIM endpoint
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_CATALOG_URL = "https://build.nvidia.com/models"


class NvidiaProvider(LLMProvider):
    """NVIDIA NIM provider adapter.

    Implements the LLMProvider interface for NVIDIA's hosted models.
    Uses OpenAI-compatible API format with NVIDIA API key authentication.
    """

    def __init__(
        self,
        name: str = "nvidia",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_catalog_url: Optional[str] = None,
        refresh_interval_minutes: int = 15
    ):
        """Initialize NVIDIA NIM provider.

        Args:
            name: Provider name (default: 'nvidia')
            base_url: API base URL (default: https://integrate.api.nvidia.com/v1)
            api_key: NVIDIA API key (default: from NVIDIA_API_KEY env var)
            model_catalog_url: URL for model catalog (default: https://build.nvidia.com/models)
            refresh_interval_minutes: Model list refresh interval
        """
        self._name = name
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self._model_catalog_url = model_catalog_url or DEFAULT_CATALOG_URL
        self._refresh_interval_minutes = refresh_interval_minutes
        self._cached_models: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Provider name."""
        return self._name

    @property
    def base_url(self) -> str:
        """API base URL."""
        return self._base_url

    @property
    def api_key(self) -> Optional[str]:
        """NVIDIA API key."""
        return self._api_key or None

    @property
    def model_list_endpoint(self) -> Optional[str]:
        """Model catalog URL for discovery."""
        return self._model_catalog_url

    @property
    def refresh_interval_minutes(self) -> int:
        """Model list refresh interval in minutes."""
        return self._refresh_interval_minutes

    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with NVIDIA API key authentication."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        return headers

    async def list_models(self) -> List[Dict[str, Any]]:
        """Fetch available models from NVIDIA NIM.

        Returns a list of models from the NVIDIA model catalog.
        Falls back to cached models if the catalog is unavailable.

        Returns:
            List of model entries with id, provider, source_url, etc.
        """
        # Try to fetch from model catalog
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(self._model_catalog_url)
                resp.raise_for_status()
                data = resp.json()

            models = []
            # NVIDIA catalog returns models in a specific format
            # Parse based on actual API response structure
            if isinstance(data, dict):
                # Handle different catalog formats
                model_list = data.get("models", []) or data.get("data", [])
                for m in model_list:
                    if isinstance(m, dict):
                        model_id = m.get("name") or m.get("id") or m.get("model")
                        if model_id:
                            models.append({
                                "id": model_id,
                                "provider": self._name,
                                "source_url": self._base_url,
                                "category": self._infer_category(model_id),
                                "description": m.get("description") or m.get("short_description") or None,
                                "context_window": m.get("context_window") or m.get("max_tokens") or None,
                            })

            self._cached_models = models
            return models

        except httpx.TimeoutException:
            # Handle timeout specifically
            logging.warning(f"NVIDIA model catalog timeout: {self._model_catalog_url}")
            if self._cached_models:
                return self._cached_models
            return self._get_default_models()
        except httpx.ConnectError as e:
            # Handle connection errors specifically
            logging.warning(f"NVIDIA model catalog connection error: {e}")
            if self._cached_models:
                return self._cached_models
            return self._get_default_models()
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors specifically
            logging.warning(f"NVIDIA model catalog HTTP error {e.response.status_code}: {e}")
            if self._cached_models:
                return self._cached_models
            return self._get_default_models()
        except Exception as e:
            # Catch-all for other exceptions
            logging.warning(f"NVIDIA model catalog error: {e}")
            if self._cached_models:
                return self._cached_models
            return self._get_default_models()

    def _infer_category(self, model_id: str) -> Optional[str]:
        """Infer model category from model ID.

        Args:
            model_id: Model identifier

        Returns:
            Inferred category or None
        """
        model_id_lower = model_id.lower()
        if any(x in model_id_lower for x in ["code", "codellama", "starcoder", "deepseek-coder"]):
            return "coding"
        elif any(x in model_id_lower for x in ["math", "reasoning", "r1", "nemotron"]):
            return "reasoning"
        elif any(x in model_id_lower for x in ["vision", "llava", "vision"]):
            return "vision"
        elif any(x in model_id_lower for x in ["image", "sdxl", "stable-diffusion"]):
            return "image"
        elif any(x in model_id_lower for x in ["embedding", "embed"]):
            return "embedding"
        else:
            return "chat"

    def _get_default_models(self) -> List[Dict[str, Any]]:
        """Return default NVIDIA models as fallback.

        These are commonly available NVIDIA NIM models.
        """
        return [
            {
                "id": "nvidia/llama-3.1-nemotron-70b-instruct",
                "provider": self._name,
                "source_url": self._base_url,
                "category": "chat",
                "description": "High-quality general-purpose chat model",
                "context_window": 128000,
            },
            {
                "id": "nvidia/llama-3.3-nemotron-70b-instruct",
                "provider": self._name,
                "source_url": self._base_url,
                "category": "chat",
                "description": "Latest Nemotron instruction-following model",
                "context_window": 128000,
            },
            {
                "id": "mistralai/mixtral-8x7b-instruct-v0.1",
                "provider": self._name,
                "source_url": self._base_url,
                "category": "chat",
                "description": "Efficient mixture-of-experts model",
                "context_window": 32000,
            },
            {
                "id": "google/gemma-2-27b-instruct",
                "provider": self._name,
                "source_url": self._base_url,
                "category": "chat",
                "description": "Google's instruction-tuned Gemma model",
                "context_window": 8192,
            },
        ]

    async def forward_request(
        self,
        model_id: str,
        payload: Dict[str, Any],
        stream: bool = False,
        endpoint: str = "/chat/completions"
    ) -> Dict[str, Any]:
        """Forward a chat completion request to NVIDIA NIM.

        Args:
            model_id: Target model identifier
            payload: Request payload with messages, etc.
            stream: Whether to enable streaming
            endpoint: API endpoint (default: /chat/completions)

        Returns:
            Response dictionary from NVIDIA NIM
        """
        url = f"{self._base_url}{endpoint}"
        body = dict(payload or {})

        # Ensure model is set
        if "model" not in body:
            body["model"] = model_id

        # Set streaming if requested
        if stream:
            body["stream"] = True

        headers = self.get_headers()

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()

            if stream:
                # For streaming, return the response for SSE handling
                return {"stream": True, "content": resp.text}

            return resp.json()

    async def health_check(self) -> bool:
        """Check if NVIDIA NIM is reachable.

        Returns:
            True if API is accessible, False otherwise
        """
        if not self._api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Try a simple models list request
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers=self.get_headers()
                )
                return resp.status_code == 200
        except Exception:
            return False