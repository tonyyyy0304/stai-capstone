"""Short-term session memory (Module 5: Memory), SQLite-backed. Replaces the
in-process _SESSION_HISTORY dict that used to live in src/api.py — that dict
lost everything on server restart; this survives it.

Full history persists here indefinitely. get_history() returns only the last
MEMORY_TRIM_TURNS turns — the in-context window handed to the agent — so a
long-running conversation doesn't grow the prompt (and cost) unboundedly.
Turns older than the window aren't discarded, they're handed to
src/memory/persistent.py to fold into a rolling summary.
"""

import sqlite3
from datetime import datetime, timezone

from src import config


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_turns_session_id ON session_turns(session_id)"
    )
    return conn


def append_turn(session_id: str, role: str, content: str) -> None:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_index "
            "FROM session_turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.execute(
            """
            INSERT INTO session_turns (session_id, turn_index, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, row["next_index"], role, content, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def count_turns(session_id: str) -> int:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM session_turns WHERE session_id = ?", (session_id,)
        ).fetchone()
    finally:
        conn.close()
    return row["n"]


def get_history(session_id: str, limit: int = config.MEMORY_TRIM_TURNS) -> list[dict]:
    """Last `limit` turns, oldest first — the in-context window for the agent."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content FROM session_turns WHERE session_id = ? "
            "ORDER BY turn_index DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    finally:
        conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_full_history(session_id: str) -> list[dict]:
    """Entire persisted history, oldest first, with turn_index — used by the
    summarizer to know exactly which turns have already fallen out of the
    trimmed window and haven't been folded into the summary yet."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT turn_index, role, content FROM session_turns WHERE session_id = ? "
            "ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [{"turn_index": r["turn_index"], "role": r["role"], "content": r["content"]} for r in rows]
