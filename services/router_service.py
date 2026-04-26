"""
Router service: encapsulates routing logic separated from the FastAPI app.
This module exposes the same route_and_execute function as the IntelligentRouter
but in a service object that the API layer can call.
"""
from typing import Optional, Dict, Any

# This service delegates to the main router instance exposed by the
# compatibility shim (ollama_router.py) when available. During the
# incremental refactor we keep ollama_router.py as the canonical place
# that wires everything together; router_service simply forwards calls
# so the smaller api/app.py can import a stable service interface.

try:
    # Prefer the router instance from the compatibility shim
    from ollama_router import router as _router  # type: ignore
except Exception:
    # Fallback: lazy import of IntelligentRouter if shim not present (tests)
    from ollama_router import IntelligentRouter, CONFIG, OLLAMA_TARGET, CLASSIFIER_MODEL  # type: ignore
    _router = IntelligentRouter(OLLAMA_TARGET.get('base_url'), CONFIG.get('proxy', {}).get('classifier_model') or CLASSIFIER_MODEL)


async def route_and_execute(prompt: str, stream: bool = False, requested_model: Optional[str] = None, override_model: Optional[str] = None) -> Dict[str, Any]:
    """Forward request to the underlying IntelligentRouter instance.

    Signature matches IntelligentRouter.route_and_execute.
    """
    return await _router.route_and_execute(prompt, stream, requested_model, override_model)


def get_router():
    return _router
