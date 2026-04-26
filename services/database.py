"""
Database operations for LLM IntelliProxy.

Provides context-managed SQLite connections and schema initialization
for the model registry, decision backlog, and request metrics.
"""
import sqlite3
import logging
from contextlib import contextmanager
from typing import Iterator, Dict, List, Any
from datetime import datetime

from services.model_metadata import MODEL_ATTRIBUTES

# Global database path (set from config)
DB_PATH = "/data/llmproxy.db"


def set_db_path(path: str) -> None:
    """Set the global database path."""
    global DB_PATH
    DB_PATH = path


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """Context manager for database connections.

    Yields a sqlite3.Connection with row_factory set to sqlite3.Row.
    Commits on successful exit, rolls back on exception.
    """
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database schema with all required tables and indexes."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Model registry table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
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
            )
        """)

        # Decision backlog table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_backlog (
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
            )
        """)

        # Request metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS request_metrics (
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
            )
        """)

        # Create indexes for efficient queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_last_seen ON model_registry(last_seen)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decision_timestamp ON decision_backlog(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON request_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_model ON request_metrics(model_id)")

        logging.info("Database initialized successfully")

        # Add missing columns (additive migrations)
        _add_missing_columns(cursor)


def _add_missing_columns(cursor: sqlite3.Cursor) -> None:
    """Add missing columns to existing tables (additive migrations only)."""
    # Ensure decision_backlog has routing_mode column
    cursor.execute("PRAGMA table_info(decision_backlog)")
    cols = [r[1] for r in cursor.fetchall()]
    if 'routing_mode' not in cols:
        cursor.execute("ALTER TABLE decision_backlog ADD COLUMN routing_mode TEXT DEFAULT 'auto'")
        logging.info("Added routing_mode column to decision_backlog")


def ensure_tables() -> None:
    """Ensure all required tables exist (wrapper for init_db)."""
    init_db()
