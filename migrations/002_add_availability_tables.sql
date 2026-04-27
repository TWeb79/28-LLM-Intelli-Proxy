-- Migration 002: Add availability monitoring tables
-- Additive migration - creates new tables without modifying existing ones

-- Availability event log - records every probe result
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

-- Current availability state - one row per model
CREATE TABLE IF NOT EXISTS model_availability_state (
    model_id TEXT NOT NULL,
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
    eta_confidence TEXT CHECK (eta_confidence IN ('low', 'medium', 'high')),
    PRIMARY KEY (model_id, provider)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_availability_log_model ON model_availability_log(model_id);
CREATE INDEX IF NOT EXISTS idx_availability_log_provider ON model_availability_log(provider);
CREATE INDEX IF NOT EXISTS idx_availability_log_checked ON model_availability_log(checked_at);
CREATE INDEX IF NOT EXISTS idx_availability_log_status ON model_availability_log(status);
CREATE INDEX IF NOT EXISTS idx_availability_state_status ON model_availability_state(current_status);
CREATE INDEX IF NOT EXISTS idx_availability_state_provider ON model_availability_state(provider);