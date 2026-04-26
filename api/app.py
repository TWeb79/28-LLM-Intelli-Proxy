"""
API entrypoint that re-exports the FastAPI app from ollama_router.

This module provides backward compatibility for running the API server via:
    uvicorn api.app:api_app

All route implementations are defined in the service layer (routes/, services/).
"""
from ollama_router import api_app, web_app

__all__ = ['api_app', 'web_app']
