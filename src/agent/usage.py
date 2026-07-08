"""Token usage tracking for the agent's LLM calls (Module 11: LLMOps Monitoring).

Every Gemini call the agent makes (router classification, ReAct loop iterations,
search_web) is logged to SQLite here, so usage can be reported both per turn and
cumulatively — useful for watching a free-tier daily request quota, not just cost.

Scope: only covers src/agent/'s own calls. The plain-RAG fallback path in
src/api.py (src/rag/answerer.py) and the embedding calls in scripts/ingest.py
are separate call sites this module doesn't see.
"""

import sqlite3
from datetime import datetime, timezone

from src import config
from src.schemas import TokenUsage


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            model TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def extract_usage(response) -> TokenUsage:
    """Pull token counts off a Gemini response. Zeros if unavailable."""
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=metadata.prompt_token_count or 0,
        completion_tokens=metadata.candidates_token_count or 0,
        total_tokens=metadata.total_token_count or 0,
    )


def record_usage(model: str, usage: TokenUsage, session_id: str | None = None) -> None:
    if usage.total_tokens == 0 and usage.prompt_tokens == 0 and usage.completion_tokens == 0:
        return
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO token_usage
                (session_id, model, prompt_tokens, completion_tokens, total_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_usage_summary(since: datetime | None = None, session_id: str | None = None) -> dict:
    conn = _get_connection()
    try:
        query = "SELECT * FROM token_usage WHERE 1=1"
        params: list[str] = []
        if since is not None:
            query += " AND created_at >= ?"
            params.append(since.isoformat())
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()
    return {
        "request_count": len(rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "total_tokens": sum(r["total_tokens"] for r in rows),
    }


def get_usage_today() -> dict:
    # Approximation: uses UTC day boundaries. Google's free-tier daily quota
    # reset time isn't UTC midnight, so this is for visibility, not an exact
    # match to the quota window.
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return get_usage_summary(since=start_of_day)


def get_usage_all_time() -> dict:
    return get_usage_summary()
