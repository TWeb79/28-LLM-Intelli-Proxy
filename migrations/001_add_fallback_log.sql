-- Migration 001: Add fallback logging table
-- Additive migration - creates new table without modifying existing ones

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

CREATE INDEX IF NOT EXISTS idx_fallback_request_id ON fallback_log(request_id);
CREATE INDEX IF NOT EXISTS idx_fallback_timestamp ON fallback_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_fallback_model ON fallback_log(model_id);
CREATE INDEX IF NOT EXISTS idx_fallback_provider ON fallback_log(provider);