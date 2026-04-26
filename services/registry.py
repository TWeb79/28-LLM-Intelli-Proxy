"""
Minimal Model Registry service for LLM IntelliProxy

Provides simple DB upsert/list operations against the same SQLite DB used by the router.
This module is intentionally small to avoid invasive refactors of the existing code.
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Optional

# DB path should match the main application DATA_DIR/llmproxy.db
DB_PATH = os.getenv("DATA_DIR", "/data")
DB_PATH = os.path.join(DB_PATH, "llmproxy.db") if os.path.isdir(os.path.dirname(DB_PATH)) else os.getenv("DB_PATH", "/data/llmproxy.db")


@contextmanager
def get_db():
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


def upsert_model(provider: str, model_id: str, source_url: str, category: Optional[str] = None,
                 description: Optional[str] = None, context_window: Optional[int] = None,
                 enabled: bool = True, assessed: bool = False) -> None:
    """Insert or update a model entry by (provider, id)."""
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO model_registry (id, provider, source_url, category, description, context_window, last_seen, enabled, assessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, id) DO UPDATE SET
                source_url=excluded.source_url,
                last_seen=excluded.last_seen,
                category=COALESCE(excluded.category, model_registry.category),
                description=COALESCE(excluded.description, model_registry.description),
                context_window=COALESCE(excluded.context_window, model_registry.context_window),
                enabled=excluded.enabled,
                assessed=COALESCE(excluded.assessed, model_registry.assessed)
            """,
            (model_id, provider, source_url, category, description, context_window, now, 1 if enabled else 0, 1 if assessed else 0)
        )


def list_models() -> List[Dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM model_registry ORDER BY last_seen DESC")
        rows = cur.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r["id"],
                "provider": r["provider"],
                "source_url": r["source_url"],
                "category": r["category"],
                "description": r["description"],
                "context_window": r["context_window"],
                "last_seen": r["last_seen"],
                "enabled": bool(r["enabled"]),
                "assessed": bool(r["assessed"]),
            })
        return result


def mark_assessed(provider: str, model_id: str, category: str, description: str) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE model_registry SET assessed=1, category=?, description=? WHERE provider=? AND id=?",
            (category, description, provider, model_id)
        )
