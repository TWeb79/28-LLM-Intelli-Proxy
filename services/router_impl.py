"""
Router implementation shim.

During the incremental refactor this module re-exports the IntelligentRouter
class from the compatibility shim (ollama_router.py). Once the full split is
completed the class implementation can be moved here and imports updated.
"""
from typing import Optional

try:
    from ollama_router import IntelligentRouter as _IntelligentRouter  # type: ignore
except Exception:
    _IntelligentRouter = None  # type: ignore


IntelligentRouter = _IntelligentRouter


def create_router(ollama_url: Optional[str], classifier_model: Optional[str]):
    """Create a new router instance.

    This is a simple factory wrapper around the compatibility shim's
    IntelligentRouter. It keeps call sites stable while we migrate code.
    """
    if _IntelligentRouter is None:
        raise RuntimeError("IntelligentRouter implementation not available")
    return _IntelligentRouter(ollama_url, classifier_model)
