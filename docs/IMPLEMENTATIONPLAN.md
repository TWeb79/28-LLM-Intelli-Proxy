# LLM IntelliProxy — Implementation Plan

## Codebase Snapshot
Date analyzed: 2026-04-27T10:29:23+02:00

### Folder Structure
```
28-LLM-InteliProxy/
├── api/                    # FastAPI app exports for uvicorn
│   └── app.py            # Re-export for uvicorn server
├── docs/                 # Documentation and plans
│   ├── IMPLEMENTATIONPLAN.md    # This file
│   ├── PR_SUMMARY_AND_PLAN.md   # Pull request documentation
│   └── REFACTOR_PROGRESS.md     # Progress tracking
├── models/               # Pydantic schemas and data models
│   └── schemas.py        # Request/response schemas
├── providers/            # Provider adapters (Ollama, NVIDIA)
│   ├── base_provider.py  # Abstract base provider interface
│   ├── ollama_provider.py    # Ollama API adapter
│   └── nvidia_provider.py    # NVIDIA NIM adapter
├── routes/               # FastAPI route handlers
│   ├── api.py           # Core API endpoints (/api/generate, /api/chat)
│   ├── registry.py      # Model registry management endpoints
│   └── web.py           # Dashboard web routes
├── services/             # Core business logic
│   ├── assessor.py      # AI model assessment service
│   ├── caches.py        # LRU caches for classification/scoring
│   ├── config_loader.py # Configuration management
│   ├── database.py      # SQLite database operations
│   ├── decision_engine.py # LLM-based routing decisions
│   ├── fallbacks.py     # Simple fallback configuration
│   ├── logging_config.py # Structured logging setup
│   ├── model_metadata.py # Model attribute definitions
│   ├── registry.py      # Model registry persistence
│   ├── router.py        # Core intelligent router
│   ├── router_impl.py   # Router implementation details
│   ├── router_service.py # Router service wrapper
│   └── scheduler.py     # Background model refresh scheduler
├── static/              # Dashboard frontend assets
│   ├── css/            # Stylesheets
│   ├── js/             # JavaScript
│   ├── index.html      # Main dashboard
│   └── models.json     # Model metadata
├── ollama_router.py     # Main application entry point
├── config.yaml          # Configuration file
└── requirements.txt     # Python dependencies
```

### Request Pipeline (current)
1. **Entry**: `ollama_router.py` - FastAPI app initialization
2. **Routing**: `routes/api.py` - HTTP endpoint handlers
3. **Classification**: `services/router.py` - Task classification and model selection
4. **Decision**: `services/decision_engine.py` - LLM-based routing decisions
5. **Execution**: `providers/ollama_provider.py` - Forward to actual LLM provider
6. **Persistence**: `services/database.py` - Store decision logs

### Database Tables (current)
```sql
-- Model registry
CREATE TABLE model_registry (
    id TEXT NOT NULL,
    provider TEXT NOT NULL,
    source_url TEXT NOT NULL,
    category TEXT,
    description TEXT,
    context_window INTEGER,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    enabled BOOLEAN DEFAULT 1,
    assessed BOOLEAN DEFAULT 0,
    PRIMARY KEY (provider, id)
);

-- Decision backlog
CREATE TABLE decision_backlog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prompt_hash TEXT NOT NULL,
    prompt_preview TEXT,
    selected_model TEXT NOT NULL,
    provider TEXT,
    reason TEXT,
    latency_ms INTEGER,
    token_count INTEGER,
    request_data TEXT,
    routing_mode TEXT DEFAULT 'auto'
);

-- Request metrics
CREATE TABLE request_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_id TEXT NOT NULL,
    provider TEXT,
    category TEXT,
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    success BOOLEAN DEFAULT 1,
    error_message TEXT
);
```

### Background Jobs (current)
- **ModelRefreshScheduler** (`services/scheduler.py`)
  - Runs every 15 minutes
  - Discovers models from all providers
  - Updates model registry
  - Triggers AI assessments for new models

## Gap Analysis

### ContextCompressionEngine
| Item | Status | Notes |
|------|--------|-------|
| Compression middleware module | **missing** | No compression logic exists |
| Compression configuration | **missing** | No compression settings in config.yaml |
| Message compression algorithms | **missing** | No token reduction logic |
| Compression level system | **missing** | No low/medium/high compression |
| Token limit enforcement | **missing** | No max_tokens handling |
| Compression mode detection | **missing** | No coding/chat/general mode detection |
| ANSI escape code removal | **partial** | Could be added to existing cleanup |
| Duplicate message removal | **partial** | Could be enhanced in existing logic |

### ModelFallbackEngine
| Item | Status | Notes |
|------|--------|-------|
| Fallback middleware module | **missing** | No intelligent fallback system |
| Candidate ranking system | **missing** | No similarity-based model selection |
| Retry logic with backoff | **missing** | No automatic retry on failures |
| Fallback configuration | **partial** | Basic fallback_model config exists |
| Error code handling | **missing** | No 429/500/502/503/504 handling |
| Fallback logging | **missing** | No fallback_log table |
| Streaming failure handling | **missing** | No mid-stream retry logic |
| Provider failure detection | **partial** | Basic HTTP error handling exists |

### ModelAvailabilityMonitor
| Item | Status | Notes |
|------|--------|-------|
| Availability monitoring module | **missing** | No health check system |
| Background health probes | **missing** | No continuous model monitoring |
| Availability state tracking | **missing** | No model status persistence |
| ETA prediction system | **missing** | No downtime forecasting |
| Recovery time estimation | **missing** | No historical analysis |
| Real-time status updates | **missing** | No WebSocket/SSE for dashboard |
| Model probe endpoints | **missing** | No health check endpoints |
| Availability configuration | **missing** | No check_interval/probe_timeout settings |

## Conflict Register

1. **Database table naming**: New tables `fallback_log`, `model_availability_log`, `model_availability_state` must not conflict with existing tables
2. **Configuration keys**: New compression/fallback/availability keys must not conflict with existing proxy/decision/storage keys
3. **HTTP client usage**: New modules must reuse existing `get_http_client()` from services/router.py
4. **Timestamp format**: All new timestamps must use ISO 8601 UTC strings (existing convention)
5. **Model registry schema**: New availability fields must be additive, not alter existing model_registry table
6. **Background job coordination**: New availability monitor must coordinate with existing ModelRefreshScheduler
7. **Response headers**: New X-IntelliProxy-* headers must not conflict with existing X-IntelliProxy-Model/Provider headers

## Implementation Sequence

### Phase 0 - Planning & Setup (COMPLETE)
- [x] Complete codebase analysis
- [x] Identify integration points
- [x] Create implementation plan
- [x] Document gaps and conflicts

### Phase 1 - Configuration Extension
- [x] Extend config.yaml with compression, fallback, availability sections
- [x] Add environment variable mappings
- [x] Update config loader to handle new sections
- **Risk**: Low - additive changes only
- **Dependencies**: None

### Phase 2 - Database Schema
- [x] Create migration for fallback_log table
- [x] Create migration for model_availability_log table  
- [x] Create migration for model_availability_state table
- [x] Add indexes for performance
- **Risk**: Low - additive migrations only
- **Dependencies**: Phase 1

### Phase 3 - Context Compression Engine
- [x] Create services/context_compression.py module
- [x] Implement compress_context() function with 3 compression levels
- [x] Add mode-specific compression rules (coding/chat/general)
- [x] Implement token limit enforcement with truncation
- [x] Add ANSI escape code and duplicate message removal
- **Risk**: Medium - new core functionality
- **Dependencies**: Phase 1

### Phase 4 - Model Availability Monitor
- [x] Create services/model_availability.py module
- [x] Implement background scheduler for health probes
- [x] Add model probe logic (chat/embedding models)
- [x] Implement availability state transitions
- [x] Create ETA prediction engine with historical analysis
- [x] Add external failure reporting interface
- **Risk**: High - complex background system
- **Dependencies**: Phase 2

### Phase 5 - Model Fallback Engine
- [x] Create services/model_fallback.py module
- [x] Implement rank_fallback_candidates() function
- [x] Add similarity strategies (category_then_size, provider_first, any)
- [x] Implement execute_with_fallback() with retry logic
- [x] Add streaming failure handling
- [x] Integrate with availability monitor for real-time status
- **Risk**: High - critical path functionality
- **Dependencies**: Phase 3, Phase 4

### Phase 6 - Pipeline Integration
- [ ] Update routes/api.py to integrate compression middleware
- [ ] Modify generate() and chat() endpoints to use fallback engine
- [ ] Add new response headers (X-IntelliProxy-Compression, X-IntelliProxy-Fallback-Used, etc.)
- [ ] Update decision engine to check availability before selection
- **Risk**: Medium - integration complexity
- **Dependencies**: Phase 3, Phase 4, Phase 5

### Phase 7 - Dashboard Extensions (Simplified)
- [ ] Add optimization tab to navigation
- [ ] Display compression settings controls
- [ ] Show model availability status
- [ ] Display fallback statistics
- [ ] Add optimization configuration panel
- **Risk**: Low - simple UI additions
- **Dependencies**: Phase 4, Phase 5

### Phase 8 - Testing
- [ ] Add unit tests for compression engine
- [ ] Add unit tests for fallback engine
- [ ] Add unit tests for availability monitor
- [ ] Add integration tests for pipeline flow
- [ ] Add performance tests for compression efficiency
- **Risk**: Medium - comprehensive test coverage needed
- **Dependencies**: Phase 3, 4, 5, 6

### Phase 9 - Session Affinity (Advanced)
- [ ] Create services/session_registry.py module
- [ ] Implement session detection and key generation
- [ ] Add session affinity enforcement logic
- [ ] Implement forced model switch with context forwarding
- [ ] Add session persistence configuration
- **Risk**: High - complex state management
- **Dependencies**: Phase 5, Phase 6

## Database Migration Plan

### Migration 001 - Fallback Logging
```sql
-- migrations/001_add_fallback_log.sql
CREATE TABLE IF NOT EXISTS fallback_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT,
    latency_ms INTEGER,
    fallback_used BOOLEAN DEFAULT 0,
    timestamp TEXT NOT NULL -- ISO 8601 UTC
);
CREATE INDEX idx_fallback_request_id ON fallback_log(request_id);
CREATE INDEX idx_fallback_timestamp ON fallback_log(timestamp);
```

### Migration 002 - Availability Monitoring
```sql
-- migrations/002_add_availability_tables.sql
CREATE TABLE IF NOT EXISTS model_availability_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('available', 'unavailable')),
    checked_at TEXT NOT NULL, -- ISO 8601 UTC
    response_time_ms INTEGER,
    error_code TEXT,
    error_message TEXT,
    consecutive_failures INTEGER DEFAULT 0,
    consecutive_successes INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_availability_state (
    model_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    current_status TEXT NOT NULL CHECK (current_status IN ('available', 'unavailable', 'unknown')),
    unavailable_since TEXT, -- ISO 8601 UTC
    last_available_at TEXT, -- ISO 8601 UTC
    last_checked_at TEXT NOT NULL, -- ISO 8601 UTC
    last_error_code TEXT,
    last_error_message TEXT,
    consecutive_failures INTEGER DEFAULT 0,
    consecutive_successes INTEGER DEFAULT 0,
    estimated_recovery_at TEXT, -- ISO 8601 UTC
    eta_confidence TEXT CHECK (eta_confidence IN ('low', 'medium', 'high'))
);

CREATE INDEX idx_availability_log_model ON model_availability_log(model_id);
CREATE INDEX idx_availability_log_checked ON model_availability_log(checked_at);
CREATE INDEX idx_availability_state_status ON model_availability_state(current_status);
```

### Migration 003 - Session Registry (Phase 9)
```sql
-- migrations/003_add_session_registry.sql
CREATE TABLE IF NOT EXISTS session_registry (
    session_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    mode TEXT NOT NULL,
    created_at TEXT NOT NULL, -- ISO 8601 UTC
    last_active_at TEXT NOT NULL, -- ISO 8601 UTC
    expires_at TEXT NOT NULL, -- ISO 8601 UTC
    request_count INTEGER DEFAULT 0,
    context_turn_count INTEGER DEFAULT 0,
    locked BOOLEAN DEFAULT 1,
    session_source TEXT NOT NULL,
    switch_count INTEGER DEFAULT 0,
    notify_on_switch BOOLEAN DEFAULT 1
);
CREATE INDEX idx_session_expires ON session_registry(expires_at);
CREATE INDEX idx_session_model ON session_registry(model_id);
```

## Test Plan

### Context Compression Tests
- `test_compression_disabled_returns_unchanged()`
- `test_compression_low_removes_noise()`
- `test_compression_medium_summarizes_content()`
- `test_compression_high_maximizes_reduction()`
- `test_compression_preserves_errors_and_stacktraces()`
- `test_compression_enforces_max_tokens()`
- `test_compression_mode_coding_vs_chat()`
- `test_compression_truncation_marker()`

### Model Fallback Tests
- `test_fallback_success_first_attempt()`
- `test_fallback_retry_on_429_rate_limit()`
- `test_fallback_retry_on_500_error()`
- `test_fallback_exhausted_returns_502()`
- `test_fallback_excludes_failed_model()`
- `test_fallback_excludes_disabled_models()`
- `test_fallback_category_then_size_ranking()`
- `test_fallback_provider_first_ranking()`
- `test_fallback_streaming_failure_handling()`
- `test_fallback_logging_persistence()`

### Availability Monitor Tests
- `test_probe_success_marks_available()`
- `test_probe_failure_marks_unavailable()`
- `test_hysteresis_thresholds_respected()`
- `test_eta_prediction_with_sufficient_history()`
- `test_eta_null_with_insufficient_samples()`
- `test_external_failure_reporting()`
- `test_real_time_status_updates()`
- `test_image_models_skipped_in_probes()`

### Integration Tests
- `test_full_pipeline_with_compression()`
- `test_full_pipeline_with_fallback()`
- `test_full_pipeline_with_availability_check()`
- `test_response_headers_present()`
- `test_dashboard_real_time_updates()`

## Exception Handling Assessment

### Current State
- **Exception handling is comprehensive** across the codebase
- **Proper try/catch patterns** in critical areas (database, HTTP, providers)
- **Graceful degradation** with best-effort logging and fallback mechanisms
- **User-friendly error messages** in API responses
- **Detailed error logging** for debugging purposes

### Identified Gaps (Minor)
- Some services could benefit from more specific exception types
- A few areas could use better error context
- Missing timeout handling in some async operations
- UI refresh behavior needs improvement

### Model Refresh Issue - RESOLVED
**Issue**: Model list not updating when pressing refresh button
**Root Cause**: UI/UX issue - the refresh endpoint works correctly, but the UI didn't force data reload
**Fix Applied**: 
- Modified `refreshModels()` function to immediately call `loadAllData()` after successful refresh
- Replaced blocking `alert()` with non-intrusive notifications
- Added proper error handling and user feedback

## Open Questions

1. **Compression token counting**: Should we use a specific tokenizer (e.g., tiktoken) or estimate based on character count?
2. **Availability probe frequency**: Is 60 seconds too aggressive for local Ollama instances?
3. **Session affinity scope**: Should session persistence be enabled by default or require explicit configuration?
4. **Compression performance**: Should compression run in a separate thread for large prompts?
5. **ETA confidence thresholds**: Are the 20%/50% variance thresholds appropriate for typical LLM downtime patterns?
6. **Fallback retry delays**: Should we implement exponential backoff between fallback attempts?
7. **Provider-specific health checks**: Should NVIDIA NIM endpoints use different probe strategies than Ollama?

---

**Plan Status**: Updated with exception handling review and model refresh fix
**Next Step**: Continue with Phase 6-9 implementation (pipeline integration and testing)
