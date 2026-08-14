"""Idempotent send-state for the HR handoff email (Component 14 follow-on).

Same SQLite shape as src/memory/onboarding_status.py:27 — inline
CREATE TABLE IF NOT EXISTS on every connect, no migrations, explicit
commit()/close() via try/finally.

packet_hash is the sha256 of the sorted source_hashes of the documents in
the packet (computed by src/notifications/hr_packet.py), so: re-uploading
the SAME two documents never resends (same packet_hash, already `sent`);
replacing a document with a genuinely different file changes its
source_hash, which changes packet_hash, which is treated as a new,
legitimate packet. `last_error` is contractually PII-free, same discipline
as RuleResult.detail in src/schemas.py.
"""

import sqlite3
from datetime import datetime, timezone

from src import config
from src.schemas import EmailSendResult, NotificationStatus


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hr_notifications (
            employee_id  TEXT NOT NULL,
            packet_hash  TEXT NOT NULL,
            status       TEXT NOT NULL,
            provider_id  TEXT,
            attempts     INTEGER NOT NULL DEFAULT 0,
            last_error   TEXT,
            sent_at      TEXT,
            PRIMARY KEY (employee_id, packet_hash)
        )
        """
    )
    return conn


def already_sent(employee_id: str, packet_hash: str) -> bool:
    """The idempotency check. Callers must check this BEFORE composing/
    sending — a true result means no send should be attempted."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT status FROM hr_notifications WHERE employee_id = ? AND packet_hash = ?",
            (employee_id, packet_hash),
        ).fetchone()
    finally:
        conn.close()
    return row is not None and row["status"] == NotificationStatus.SENT.value


def record_attempt(employee_id: str, packet_hash: str) -> int:
    """Upserts a `queued` row and increments `attempts`; returns the new
    attempt count. Called once per send attempt, before calling the email
    provider, so a crash mid-send still leaves an attempts trail."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO hr_notifications (employee_id, packet_hash, status, attempts)
            VALUES (?, ?, ?, 1)
            ON CONFLICT (employee_id, packet_hash) DO UPDATE SET
                attempts = attempts + 1,
                status = excluded.status
            """,
            (employee_id, packet_hash, NotificationStatus.QUEUED.value),
        )
        conn.commit()
        row = conn.execute(
            "SELECT attempts FROM hr_notifications WHERE employee_id = ? AND packet_hash = ?",
            (employee_id, packet_hash),
        ).fetchone()
    finally:
        conn.close()
    return row["attempts"]


def list_for_employee(employee_id: str) -> list[dict]:
    """Newest-first, for GET /hr-notifications/{employee_id} (the UI's
    "Sent to HR" indicator)."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT packet_hash, status, sent_at FROM hr_notifications "
            "WHERE employee_id = ? ORDER BY sent_at DESC, ROWID DESC",
            (employee_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def record_result(employee_id: str, packet_hash: str, result: EmailSendResult) -> None:
    """Records the outcome of a send attempt. Never raises — mirrors
    onboarding_status.record_result()'s convention so a caller can call this
    unconditionally after email_client returns."""
    conn = _get_connection()
    try:
        sent_at = datetime.now(timezone.utc).isoformat() if result.status == NotificationStatus.SENT else None
        conn.execute(
            """
            UPDATE hr_notifications SET
                status = ?,
                provider_id = ?,
                last_error = ?,
                sent_at = COALESCE(?, sent_at)
            WHERE employee_id = ? AND packet_hash = ?
            """,
            (result.status.value, result.provider_id or None, result.error or None, sent_at, employee_id, packet_hash),
        )
        conn.commit()
    finally:
        conn.close()
