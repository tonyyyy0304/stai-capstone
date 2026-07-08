"""Long-term / cross-session memory (Module 5: Memory) — a rolling per-session
summary, extended incrementally as turns fall out of the short-term trim
window, plus cross-session recall keyed by employee_id.

Design is hybrid trim + overflow-summarize, deliberately NOT summarize-every-
turn — that would cost an LLM call per chat turn, which directly conflicts
with this project's #1 documented constraint (Gemini's 20 requests/day free
tier, already exhausted mid-testing multiple times; see PLAN.md §2.1, §8).
Summarization only fires once config.MEMORY_SUMMARY_BATCH_SIZE turns have
overflowed the window since the last summary, and each call extends the
existing summary (one incremental call) rather than re-summarizing the whole
conversation from scratch.

Scope: only ever reads from session_turns (src/memory/session.py), which only
holds general FAQ/chat message text. It never reads complaint-form PII
payloads — those live in the separate `tickets` table (src/agent/tools.py),
a different table entirely, so there's no path for this summarizer to see
ComplaintTicket fields (parties_involved, description, etc.).
"""

import sqlite3
from datetime import datetime, timezone

from src import config
from src.agent import prompts
from src.memory import session
from src.schemas import SessionSummary


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_summaries (
            session_id TEXT PRIMARY KEY,
            employee_id TEXT,
            summary_text TEXT NOT NULL,
            summarized_through_turn INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def _get_summary_row(session_id: str) -> sqlite3.Row | None:
    conn = _get_connection()
    try:
        return conn.execute(
            "SELECT * FROM session_summaries WHERE session_id = ?", (session_id,)
        ).fetchone()
    finally:
        conn.close()


def get_summary(session_id: str) -> str | None:
    row = _get_summary_row(session_id)
    return row["summary_text"] if row else None


def get_latest_summary_for_employee(employee_id: str) -> str | None:
    """Most recently updated summary across any of this employee's past
    sessions — used to seed context on a brand-new session."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT summary_text FROM session_summaries WHERE employee_id = ? "
            "ORDER BY updated_at DESC LIMIT 1",
            (employee_id,),
        ).fetchone()
    finally:
        conn.close()
    return row["summary_text"] if row else None


def _save_summary(
    session_id: str, employee_id: str | None, summary_text: str, summarized_through_turn: int
) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO session_summaries
                (session_id, employee_id, summary_text, summarized_through_turn, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                summary_text = excluded.summary_text,
                employee_id = COALESCE(excluded.employee_id, session_summaries.employee_id),
                summarized_through_turn = excluded.summarized_through_turn,
                updated_at = excluded.updated_at
            """,
            (
                session_id,
                employee_id,
                summary_text,
                summarized_through_turn,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def maybe_update_summary(session_id: str, employee_id: str | None = None, client=None) -> None:
    """Call after appending a turn. No-op unless a full batch
    (config.MEMORY_SUMMARY_BATCH_SIZE) of turns has overflowed the trim
    window since the last summary — this is what keeps summarization rare
    instead of per-turn."""
    total_turns = session.count_turns(session_id)
    overflow_target = total_turns - config.MEMORY_TRIM_TURNS
    if overflow_target <= 0:
        return

    row = _get_summary_row(session_id)
    already_through = row["summarized_through_turn"] if row else 0
    pending = overflow_target - already_through
    if pending < config.MEMORY_SUMMARY_BATCH_SIZE:
        return

    full_history = session.get_full_history(session_id)
    newly_evicted = [
        t for t in full_history if already_through <= t["turn_index"] < overflow_target
    ]
    if not newly_evicted:
        return

    existing_summary = row["summary_text"] if row else "(no prior summary)"
    summary = _summarize(existing_summary, newly_evicted, client=client)
    _save_summary(session_id, employee_id, summary.summary_text, overflow_target)


def _summarize(existing_summary: str, evicted_turns: list[dict], client=None) -> SessionSummary:
    from google.genai import types

    client = client or config.get_llm_client()
    turns_text = "\n".join(f"{t['role']}: {t['content']}" for t in evicted_turns)
    response = client.models.generate_content(
        model=config.ACTIVE_CHAT_MODEL,
        contents=prompts.SESSION_SUMMARY_PROMPT.format(
            existing_summary=existing_summary, new_turns=turns_text
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SessionSummary,
            temperature=0.0,
        ),
    )
    result: SessionSummary | None = response.parsed
    if result is None:  # fail closed: keep the existing summary rather than losing it
        return SessionSummary(summary_text=existing_summary)
    return result


def get_context(session_id: str, employee_id: str | None = None) -> list[dict]:
    """The history list to hand to the agent: trimmed raw turns, prefixed
    with a rolling summary of anything older. For a brand-new session with a
    known employee_id and no summary of its own yet, seeds from their most
    recent summary out of any prior session (cross-session recall)."""
    raw_turns = session.get_history(session_id)
    summary_text = get_summary(session_id)

    if summary_text is None and not raw_turns and employee_id:
        summary_text = get_latest_summary_for_employee(employee_id)

    if not summary_text:
        return raw_turns
    return [
        {"role": "assistant", "content": f"[Summary of earlier conversation: {summary_text}]"}
    ] + raw_turns
