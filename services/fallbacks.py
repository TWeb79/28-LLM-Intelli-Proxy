"""
Fallback configuration store.

Provides a mutable global store for model fallback mappings.
Both the router and configuration endpoints can access and modify this.
"""
from typing import Dict

# Mutable fallback mapping: model_name -> fallback_model_name
MODEL_FALLBACKS: Dict[str, str] = {}

# Timeout for fallback attempts (seconds)
FALLBACK_TIMEOUT = 30


def get_fallbacks() -> Dict[str, str]:
    """Get current fallback mapping."""
    return MODEL_FALLBACKS.copy()


def set_fallbacks(fallbacks: Dict[str, str]) -> None:
    """Replace the entire fallback mapping."""
    global MODEL_FALLBACKS
    MODEL_FALLBACKS = dict(fallbacks)


def get_fallback_for_model(model: str) -> str:
    """Get the fallback model for a given model, or None."""
    return MODEL_FALLBACKS.get(model)


def set_timeout(seconds: int) -> None:
    """Set fallback timeout."""
    global FALLBACK_TIMEOUT
    FALLBACK_TIMEOUT = seconds
