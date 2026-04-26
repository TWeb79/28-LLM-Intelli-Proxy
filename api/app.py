"""
API entrypoint that registers routes and delegates implementation to the core module.

This module allows running the FastAPI apps from a smaller entrypoint while keeping
the existing core logic in ollama_router.py during an incremental refactor.
"""
from fastapi import FastAPI, HTTPException, Response
from services import registry as model_registry
from services import router_service
import ollama_router as core  # compatibility shim providing web_app and other exports


api_app = FastAPI(title="IntelliProxy API Gateway")


@api_app.get("/models")
async def models():
    # Return the same view as the compatibility shim
    return await core.list_models()


@api_app.get("/v1/models")
async def v1_models():
    return await core.v1_models()


@api_app.post("/api/generate")
async def generate(request: core.GenerateRequest):
    # delegate to router service for routing
    return await router_service.route_and_execute(request.prompt or "", request.stream, requested_model=request.model)


@api_app.post("/api/chat")
async def chat(request: core.ChatRequest):
    return await router_service.route_and_execute("\n".join([f"{m.role}: {m.content}" for m in request.messages]), request.stream, requested_model=request.model)


@api_app.get("/health")
async def health():
    return await core.health_check()


# Mount the dashboard app from the compatibility shim
web_app = core.web_app
