"""
Router service: encapsulates routing logic separated from the FastAPI app.
This module exposes the same route_and_execute function as the IntelligentRouter
but in a service object that the API layer can call.
"""
from typing import Optional, Dict, Any
from ollama_router import IntelligentRouter, CONFIG, OLLAMA_TARGET, CLASSIFIER_MODEL

_router = IntelligentRouter(OLLAMA_TARGET.get('base_url'), CONFIG.get('proxy', {}).get('classifier_model') or CLASSIFIER_MODEL)

async def route_and_execute(prompt: str, stream: bool = False, requested_model: Optional[str] = None, override_model: Optional[str] = None) -> Dict[str, Any]:
    return await _router.route_and_execute(prompt, stream, requested_model, override_model)

def get_router():
    return _router
