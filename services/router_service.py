"""
Router service: encapsulates routing logic separated from the FastAPI app.

This module exposes a route_and_execute function that delegates to the
IntelligentRouter from the service layer, providing a stable interface
for the API layer.
"""
import os
from typing import Optional, Dict, Any

# Global router reference; set during application startup
_router = None


def get_router():
    """Get or create the global router instance.

    Returns:
        IntelligentRouter instance
    """
    global _router
    if _router is None:
        from services.router import IntelligentRouter
        from services.config_loader import load_config
        from services.decision_engine import DecisionEngine

        config = load_config()
        decision_engine = DecisionEngine(
            decision_model=config.get("decision", {}).get("model"),
            fallback_model=config.get("proxy", {}).get("fallback_model")
        )
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:8128")
        _router = IntelligentRouter(
            ollama_base_url=ollama_base,
            decision_engine=decision_engine,
            classifier_model=config.get("decision", {}).get("model")
        )
    return _router


def set_router(router) -> None:
    """Set the global router instance."""
    global _router
    _router = router


async def route_and_execute(
    prompt: str,
    stream: bool = False,
    requested_model: Optional[str] = None,
    override_model: Optional[str] = None
) -> Dict[str, Any]:
    """Forward request to the underlying IntelligentRouter.

    Args:
        prompt: User prompt
        stream: Whether to stream (not yet supported)
        requested_model: Explicitly requested model
        override_model: Override from X-LLMProxy-Model header

    Returns:
        Router execution result
    """
    router = get_router()
    return await router.route_and_execute(prompt, stream, requested_model, override_model)
