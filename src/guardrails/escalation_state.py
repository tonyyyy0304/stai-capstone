from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from src import config
from src.schemas import EscalationFlowState


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS escalation_sessions (
            session_id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            miss_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def get_state(session_id: str) -> EscalationFlowState:
    """Pure lookup, no side effects. Missing row -> NORMAL."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT state FROM escalation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return EscalationFlowState.NORMAL
    return EscalationFlowState(row["state"])


def set_state(session_id: str, state: EscalationFlowState) -> None:
    """Transition a session into a non-NORMAL flow state. Setting NORMAL
    explicitly is equivalent to clear_state() -- both remove the row.
    miss_count resets to 0 on every transition, since a fresh state deserves
    a fresh grace period."""
    if state == EscalationFlowState.NORMAL:
        clear_state(session_id)
        return

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO escalation_sessions (session_id, state, miss_count, updated_at)
            VALUES (?, ?, 0, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                state = excluded.state,
                miss_count = 0,
                updated_at = excluded.updated_at
            """,
            (session_id, state.value, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def record_form_miss(session_id: str) -> int:
    """The employee typed free chat instead of submitting the rendered form.
    Increments and returns the new miss_count so the caller can decide
    whether to remind again (first miss) or fail-safe escalate (repeated
    misses) rather than escalating on the very first stray message."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            UPDATE escalation_sessions
            SET miss_count = miss_count + 1, updated_at = ?
            WHERE session_id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), session_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT miss_count FROM escalation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    return row["miss_count"] if row is not None else 0


def clear_state(session_id: str) -> None:
    """Return a session to NORMAL by deleting its row outright."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM escalation_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
