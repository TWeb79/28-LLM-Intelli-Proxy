"""
Abstract base class for LLM providers.

Defines the contract that all provider adapters must implement,
enabling pluggable provider support (Ollama, NVIDIA NIM, Anthropic, etc.).
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All provider adapters must implement these methods to ensure
    consistent behavior across different LLM backends.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'ollama', 'nvidia', 'anthropic')."""
        pass
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base URL for the provider API."""
        pass
    
    @property
    def api_key(self) -> Optional[str]:
        """API key for authentication (None if not required)."""
        return None
    
    @property
    def model_list_endpoint(self) -> Optional[str]:
        """Optional endpoint for model discovery (falls back to static list)."""
        return None
    
    @property
    def refresh_interval_minutes(self) -> int:
        """How often to refresh the model list (default: 15 minutes)."""
        return 15
    
    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """Fetch available models from the provider.
        
        Returns:
            List of model entries with keys:
            - id: Model identifier
            - provider: Provider name
            - source_url: Provider base URL
            - category: Optional category tag
            - description: Optional description
            - context_window: Optional max context length
        """
        pass
    
    @abstractmethod
    async def forward_request(
        self,
        model_id: str,
        payload: Dict[str, Any],
        stream: bool = False,
        endpoint: str = "/chat/completions"
    ) -> Dict[str, Any]:
        """Forward a request to the provider.
        
        Args:
            model_id: Target model identifier
            payload: Request payload (messages, prompt, etc.)
            stream: Whether to enable streaming
            endpoint: API endpoint to use
            
        Returns:
            Response dictionary from the provider
        """
        pass
    
    async def health_check(self) -> bool:
        """Check if the provider is reachable.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            models = await self.list_models()
            return len(models) >= 0
        except Exception:
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for requests to this provider.
        
        Returns:
            Dictionary of headers including auth if needed
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class ProviderRegistry:
    """Registry for managing provider instances."""
    
    _providers: Dict[str, LLMProvider] = {}
    
    @classmethod
    def register(cls, provider: LLMProvider) -> None:
        """Register a provider instance.
        
        Args:
            provider: Provider instance to register
        """
        cls._providers[provider.name] = provider
    
    @classmethod
    def get(cls, name: str) -> Optional[LLMProvider]:
        """Get a provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider instance or None if not found
        """
        return cls._providers.get(name)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names.
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def get_all(cls) -> List[LLMProvider]:
        """Get all registered provider instances.
        
        Returns:
            List of provider instances
        """
        return list(cls._providers.values())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (useful for testing)."""
        cls._providers.clear()