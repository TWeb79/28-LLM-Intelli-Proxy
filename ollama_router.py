#!/usr/bin/env python3
"""
LLM IntelliProxy - Unified Model Registry & Provider System

Entry point that wires together modular services and provides FastAPI apps.
"""

import os
import io
import csv
import json
import time
import asyncio
import logging
import sqlite3
import yaml
import httpx
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config() -> dict:
    config = {
        "proxy": {
            "port": int(os.getenv("PROXY_PORT", "8130")),
            "host": os.getenv("PROXY_HOST", "0.0.0.0"),
            "log_level": os.getenv("LOG_LEVEL", "info"),
            "fallback_model": os.getenv("FALLBACK_MODEL", "qwen2.5:8b"),
        },
        "decision": {
            "model": os.getenv("DECISION_MODEL", ""),
            "refresh_registry_on_startup": True,
        },
        "storage": {
            "type": "sqlite",
            "path": os.path.join(os.getenv("DATA_DIR", "/data"), "llmproxy.db"),
        },
        "providers": [],
        "server": {
            "port": int(os.getenv("DASHBOARD_PORT", os.getenv("WEB_PORT", "3000"))),
        }
    }

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                for key in config:
                    if key in file_config:
                        if isinstance(config[key], dict):
                            config[key].update(file_config[key])
                        else:
                            config[key] = file_config[key]
        except Exception as e:
            logging.warning(f"Could not load config.yaml: {e}")

    return config

CONFIG = load_config()

# Ensure data directory exists
data_dir = os.path.dirname(CONFIG["storage"]["path"])
os.makedirs(data_dir, exist_ok=True)

# Configure structured logging
from services.logging_config import setup_logging
setup_logging(CONFIG.get('proxy', {}).get('log_level', 'info'))

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

OLLAMA_TARGET = {
    "host": os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_SERVICE_HOST", "localhost")),
    "port": int(os.getenv("OLLAMA_PORT", os.getenv("OLLAMA_SERVICE_PORT", "8128"))),
}
OLLAMA_TARGET["base_url"] = f"http://{OLLAMA_TARGET['host']}:{OLLAMA_TARGET['port']}"

PROXY_HOST = CONFIG.get('proxy', {}).get('host', '0.0.0.0')
PROXY_PORT = CONFIG.get('proxy', {}).get('port', 8130)
WEB_HOST = os.getenv('WEB_HOST', '0.0.0.0')
WEB_PORT = CONFIG.get('server', {}).get('port', 3000)

CLASSIFIER_MODEL = CONFIG.get('decision', {}).get('model') or os.getenv('CLASSIFIER_MODEL', '')
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '120'))
MODEL_FALLBACKS = {}
FALLBACK_TIMEOUT = int(os.getenv('FALLBACK_TIMEOUT', '30'))

# ============================================================================
# DATABASE
# ============================================================================

from services.database import set_db_path, ensure_tables

DB_PATH = CONFIG["storage"]["path"]
set_db_path(DB_PATH)
ensure_tables()

# ============================================================================
# PROVIDERS & SERVICES
# ============================================================================

from providers.ollama_provider import OllamaProvider
from services.decision_engine import DecisionEngine
from services.router import IntelligentRouter, get_http_client
from services.fallbacks import set_fallbacks, set_timeout

OLLAMA_PROVIDER = OllamaProvider(name="ollama", base_url=OLLAMA_TARGET.get("base_url"))
DECISION_ENGINE = DecisionEngine(
    decision_model=CONFIG.get("decision", {}).get("model"),
    fallback_model=CONFIG.get("proxy", {}).get("fallback_model")
)

# ============================================================================
# ROUTER
# ============================================================================

router = IntelligentRouter(
    ollama_base_url=OLLAMA_TARGET["base_url"],
    decision_engine=DECISION_ENGINE,
    classifier_model=CLASSIFIER_MODEL
)

# ============================================================================
# FASTAPI APPS
# ============================================================================

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

api_app = FastAPI(
    title="IntelliProxy API",
    description="OpenAI-compatible LLM routing proxy",
    version="3.0"
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

web_app = FastAPI(title="IntelliProxy Dashboard", version="3.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")
web_app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================================
# ROUTE REGISTRATION
# ============================================================================

# Import services
from services import registry as model_registry
from services.scheduler import get_scheduler

# Import route handler functions
from routes.api import (
    list_models, v1_models, process_task, generate, chat,
    get_stats, performance_test, health_check, classify_only,
    get_config as get_config_handler, get_fallbacks as get_fallbacks_handler,
    set_ollama_target as set_ollama_handler, set_fallbacks as set_fallbacks_handler,
    get_requests, clear_requests, initialize as initialize_api
)

from routes.web import (
    dashboard as web_dashboard,
    web_stats as web_stats_handler,
    web_health as web_health_handler,
    get_fallbacks_config as web_fallbacks,
    initialize as initialize_web
)

api_app.get("/")(lambda: {"status": "running", "service": "IntelliProxy API"})
api_app.get("/models")(list_models)
api_app.get("/v1/models")(v1_models)
api_app.post("/task")(process_task)
api_app.post("/api/generate")(generate)
api_app.post("/api/chat")(chat)
api_app.get("/stats")(get_stats)
api_app.post("/performance-test")(performance_test)
api_app.get("/health/full")(health_check)
api_app.get("/health")(health_check)
@api_app.get("/classify")
async def classify(prompt: str):
    return await classify_only(prompt)
api_app.get("/config")(get_config_handler)
api_app.get("/config/fallbacks")(get_fallbacks_handler)
api_app.post("/config/ollama")(set_ollama_handler)
api_app.post("/config/fallbacks")(set_fallbacks_handler)
api_app.get("/requests")(get_requests)
api_app.post("/requests/clear")(clear_requests)

@api_app.get("/api/registry")
async def get_registry():
    return {"total": len(model_registry.list_models()), "models": model_registry.list_models()}

@api_app.post("/api/registry/refresh")
async def refresh_registry():
    scheduler = get_scheduler()
    scheduler.refresh_now()
    return {"status": "refresh triggered"}

@api_app.patch("/api/registry/{provider}/{model_id}")
async def update_model(provider: str, model_id: str, request: dict):
    models = model_registry.list_models()
    for m in models:
        if m.get("provider") == provider and m.get("id") == model_id:
            model_registry.upsert_model(
                provider=provider, model_id=model_id,
                source_url=m.get("source_url", OLLAMA_TARGET["base_url"]),
                category=m.get("category"), description=m.get("description"),
                context_window=m.get("context_window"),
                enabled=request.get("enabled", m.get("enabled", True)),
                assessed=m.get("assessed", False)
            )
            return {"status": "updated", "model": model_id}
    raise HTTPException(status_code=404, detail="Model not found")

@api_app.get("/api/decisions")
async def get_decisions(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    if not os.path.exists(DB_PATH):
        return {"total": 0, "decisions": []}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    query = "SELECT * FROM decision_backlog WHERE 1=1"
    params = []
    if model:
        query += " AND selected_model = ?"
        params.append(model)
    if provider:
        query += " AND provider = ?"
        params.append(provider)
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    cur.execute(query, params)
    rows = cur.fetchall()

    count_query = "SELECT COUNT(*) FROM decision_backlog WHERE 1=1"
    count_params = []
    if model:
        count_query += " AND selected_model = ?"
        count_params.append(model)
    if provider:
        count_query += " AND provider = ?"
        count_params.append(provider)
    cur.execute(count_query, count_params)
    total = cur.fetchone()[0]
    conn.close()

    decisions = []
    for row in rows:
        decisions.append({
            "id": row["id"], "timestamp": row["timestamp"],
            "prompt_preview": row["prompt_preview"], "selected_model": row["selected_model"],
            "provider": row["provider"], "reason": row["reason"],
            "latency_ms": row["latency_ms"], "token_count": row["token_count"],
            "routing_mode": row["routing_mode"],
        })
    return {"total": total, "decisions": decisions, "limit": limit, "offset": offset}

@api_app.get("/api/decisions/export")
async def export_decisions(format: str = "json"):
    if not os.path.exists(DB_PATH):
        return {"error": "no data"}, 404

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM decision_backlog ORDER BY timestamp DESC")
    rows = cur.fetchall()
    conn.close()

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "prompt_preview", "selected_model", "provider", "reason", "latency_ms", "token_count", "routing_mode"])
        for row in rows:
            writer.writerow([
                row["timestamp"], row["prompt_preview"], row["selected_model"],
                row["provider"], row["reason"], row["latency_ms"],
                row["token_count"], row["routing_mode"],
            ])
        return {"format": "csv", "data": output.getvalue()}
    else:
        decisions = []
        for row in rows:
            decisions.append({
                "timestamp": row["timestamp"], "prompt_preview": row["prompt_preview"],
                "selected_model": row["selected_model"], "provider": row["provider"],
                "reason": row["reason"], "latency_ms": row["latency_ms"],
                "token_count": row["token_count"], "routing_mode": row["routing_mode"],
            })
        return {"format": "json", "data": decisions}

@api_app.get("/api/live-feed")
async def live_feed():
    async def event_generator():
        yield f"data: {json.dumps({'type': 'connected', 'timestamp': str(datetime.utcnow())})}\n\n"
        while True:
            await asyncio.sleep(30)
            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': str(datetime.utcnow())})}\n\n"
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@api_app.get("/api/scheduler/status")
async def scheduler_status():
    return get_scheduler().stats

# ============================================================================
# WEB ROUTES
# ============================================================================

web_app.get("/")(web_dashboard)
web_app.get("/api/stats")(web_stats_handler)
web_app.get("/api/health")(web_health_handler)
web_app.get("/api/config/fallbacks")(web_fallbacks)

# ============================================================================
# STARTUP
# ============================================================================

@api_app.on_event("startup")
async def startup_event():
    logging.info("=" * 70)
    logging.info("LLM IntelliProxy starting...")
    logging.info(f"Ollama: {OLLAMA_TARGET['base_url']}")
    logging.info(f"API: http://{PROXY_HOST}:{PROXY_PORT}")
    logging.info(f"Dashboard: http://{WEB_HOST}:{WEB_PORT}")
    logging.info("=" * 70)
    logging.info("System ready")
    logging.info("=" * 70)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'CONFIG', 'OLLAMA_TARGET', 'PROXY_HOST', 'PROXY_PORT',
    'WEB_HOST', 'WEB_PORT', 'CLASSIFIER_MODEL', 'REQUEST_TIMEOUT',
    'MODEL_FALLBACKS', 'FALLBACK_TIMEOUT', 'DB_PATH',
    'OLLAMA_PROVIDER', 'DECISION_ENGINE', 'router',
    'api_app', 'web_app',
]
