"""
Request/Response Pydantic schemas for LLM IntelliProxy API.

These schemas define the data models for client requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# Request Models
# ============================================================================

class TaskRequest(BaseModel):
    """Simple task routing request."""
    prompt: str
    stream: bool = False


class GenerateRequest(BaseModel):
    """Ollama-compatible generate request."""
    model: Optional[str] = None  # If None, router selects automatically
    prompt: Optional[str] = None
    images: Optional[List[str]] = None
    stream: bool = False
    options: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    """Single chat message for chat completions."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Ollama-compatible chat request."""
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    options: Optional[Dict[str, Any]] = None


class PerformanceTestRequest(BaseModel):
    """Performance testing request."""
    prompt: str
    mode: str = "direct"  # "direct", "intelliproxy", "all"


class SetOllamaTargetRequest(BaseModel):
    """Request to update Ollama target configuration."""
    host: Optional[str] = None
    port: Optional[int] = None


class SetFallbacksRequest(BaseModel):
    """Request to update fallback configuration."""
    fallbacks: Optional[Dict[str, str]] = None
    timeout: Optional[int] = None


# ============================================================================
# Response Models (used internally, can be returned as dict)
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    overall_status: str
    proxy: Dict[str, Any]
    ollama: Dict[str, Any]
    performance: Dict[str, Any]


class StatsResponse(BaseModel):
    """Usage statistics response."""
    requests: Dict[str, Any]
    cache: Dict[str, Any]
    models: Dict[str, Any]


class ModelInfo(BaseModel):
    """Model registry entry."""
    id: str
    provider: str
    source_url: str
    category: Optional[str] = None
    description: Optional[str] = None
    context_window: Optional[int] = None
    last_seen: Optional[str] = None
    enabled: bool = True
    assessed: bool = False


class ModelsListResponse(BaseModel):
    """Response for /models endpoint."""
    total: int
    models: Dict[str, Any]  # model_name -> metadata
    categories: Dict[str, List[str]]


class V1ModelsResponse(BaseModel):
    """OpenAI-compatible models list response."""
    object: str = "list"
    data: List[Dict[str, Any]]


class ConfigResponse(BaseModel):
    """Current configuration response."""
    ollama: Dict[str, Any]
    router: Dict[str, Any]


class DecisionLogEntry(BaseModel):
    """Decision backlog entry."""
    id: int
    timestamp: str
    prompt_preview: str
    selected_model: str
    provider: Optional[str]
    reason: str
    latency_ms: int
    token_count: int
    routing_mode: str = "auto"
