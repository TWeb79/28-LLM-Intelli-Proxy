"""
Router implementation shim.

This module re-exports the IntelligentRouter class from the service layer
and provides a factory function for creating router instances.
"""
from typing import Optional

try:
    from services.router import IntelligentRouter as _IntelligentRouter
except ImportError:
    _IntelligentRouter = None


IntelligentRouter = _IntelligentRouter


def create_router(
    ollama_url: Optional[str] = None,
    classifier_model: Optional[str] = None,
    decision_model: Optional[str] = None,
    fallback_model: Optional[str] = None
):
    """Create a new IntelligentRouter instance with proper dependencies.

    Args:
        ollama_url: Ollama base URL
        classifier_model: Optional classifier model for legacy compatibility
        decision_model: Decision engine model override
        fallback_model: Fallback model override

    Returns:
        Configured IntelligentRouter
    """
    if _IntelligentRouter is None:
        raise RuntimeError("IntelligentRouter implementation not available")

    from services.decision_engine import DecisionEngine
    from services.config_loader import load_config

    config = load_config()
    decision_model = decision_model or config.get("decision", {}).get("model")
    fallback_model = fallback_model or config.get("proxy", {}).get("fallback_model")

    decision_engine = DecisionEngine(
        decision_model=decision_model,
        fallback_model=fallback_model
    )

    return _IntelligentRouter(
        ollama_base_url=ollama_url or "http://localhost:8128",
        decision_engine=decision_engine,
        classifier_model=classifier_model
    )

    return _IntelligentRouter(
        ollama_base_url=ollama_url or "http://localhost:8128",
        decision_engine=decision_engine,
        classifier_model=classifier_model
    )
