"""Durable per-employee onboarding checklist state (Component 14,
CV_INTEGRATION.md §2.8). SQLite-backed, same _get_connection() shape as
src/memory/session.py:18-36 — inline CREATE TABLE IF NOT EXISTS on every
connect, no migrations, explicit commit()/close() via try/finally.

Deliberately carries no extracted field value (no name, no reference
number) in onboarding_documents — only status/outcome/source_hash. This is
what keeps a name or ID number from ever reaching a persisted table via this
path. source_hash is the deletion key for withdrawing consent: deleting the
image, the ground truth, the OCR cache entry, and this row are all findable
by that one hash.
"""

import sqlite3
from datetime import datetime, timezone

from src import config
from src.schemas import ChecklistStatus, DocStatus, DocType, OnboardingDocument, ValidationOutcome, ValidationResult

_OUTCOME_TO_STATUS = {
    ValidationOutcome.ACCEPTED: DocStatus.VALIDATED,
    ValidationOutcome.REJECTED: DocStatus.REJECTED,
    ValidationOutcome.NEEDS_REVIEW: DocStatus.NEEDS_REVIEW,
}


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS onboarding_documents (
            employee_id TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            status TEXT NOT NULL,
            outcome TEXT,
            validated_at TEXT,
            source_hash TEXT,
            PRIMARY KEY (employee_id, doc_type)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS onboarding_profile (
            employee_id TEXT PRIMARY KEY,
            faculty_class TEXT
        )
        """
    )
    return conn


def record_result(employee_id: str, doc_type: DocType, result: ValidationResult, source_hash: str) -> None:
    """Upserts the checklist row for (employee_id, doc_type) from a
    ValidationResult. Never stores an extracted field value — only the
    outcome shape."""
    status = _OUTCOME_TO_STATUS[result.outcome]
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO onboarding_documents (employee_id, doc_type, status, outcome, validated_at, source_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (employee_id, doc_type) DO UPDATE SET
                status = excluded.status,
                outcome = excluded.outcome,
                validated_at = excluded.validated_at,
                source_hash = excluded.source_hash
            """,
            (
                employee_id,
                doc_type.value,
                status.value,
                result.outcome.value,
                datetime.now(timezone.utc).isoformat(),
                source_hash,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_document(employee_id: str, doc_type: DocType) -> OnboardingDocument | None:
    """Single-row lookup — the sibling-lookup half of the cross-document
    check (CV_INTEGRATION.md §2.7): given the doc_type just uploaded, look up
    whether the OTHER required doc_type already has a source_hash on file,
    for a cache-hit re-extraction."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT doc_type, status, outcome, validated_at, source_hash FROM onboarding_documents "
            "WHERE employee_id = ? AND doc_type = ?",
            (employee_id, doc_type.value),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_document(employee_id, row)


def get_status(employee_id: str) -> ChecklistStatus:
    """One row per doc_type in config.REQUIRED_ONBOARDING_DOCS, always —
    a doc_type with no submission yet still gets a DocStatus.MISSING entry,
    so the UI checklist panel and the chat reply path always see the full
    set, not just what's been uploaded so far. `missing` lists every
    doc_type not yet at DocStatus.VALIDATED (submitted-but-rejected or
    needs-review still counts as missing from a complete checklist)."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT doc_type, status, outcome, validated_at, source_hash FROM onboarding_documents "
            "WHERE employee_id = ?",
            (employee_id,),
        ).fetchall()
        faculty_class = _get_faculty_class(conn, employee_id)
    finally:
        conn.close()

    by_type = {r["doc_type"]: r for r in rows}
    documents: list[OnboardingDocument] = []
    missing: list[str] = []
    for doc_type_str in config.REQUIRED_ONBOARDING_DOCS:
        row = by_type.get(doc_type_str)
        if row is None:
            documents.append(
                OnboardingDocument(employee_id=employee_id, doc_type=DocType(doc_type_str), status=DocStatus.MISSING)
            )
            missing.append(doc_type_str)
            continue
        documents.append(_row_to_document(employee_id, row))
        if DocStatus(row["status"]) != DocStatus.VALIDATED:
            missing.append(doc_type_str)

    return ChecklistStatus(employee_id=employee_id, documents=documents, missing=missing, faculty_class=faculty_class)


def set_faculty_class(employee_id: str, faculty_class: str) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO onboarding_profile (employee_id, faculty_class) VALUES (?, ?)
            ON CONFLICT (employee_id) DO UPDATE SET faculty_class = excluded.faculty_class
            """,
            (employee_id, faculty_class),
        )
        conn.commit()
    finally:
        conn.close()


def get_faculty_class(employee_id: str) -> str | None:
    conn = _get_connection()
    try:
        result = _get_faculty_class(conn, employee_id)
    finally:
        conn.close()
    return result


def _get_faculty_class(conn: sqlite3.Connection, employee_id: str) -> str | None:
    row = conn.execute(
        "SELECT faculty_class FROM onboarding_profile WHERE employee_id = ?", (employee_id,)
    ).fetchone()
    return row["faculty_class"] if row else None


def _row_to_document(employee_id: str, row: sqlite3.Row | None) -> OnboardingDocument | None:
    if row is None:
        return None
    return OnboardingDocument(
        employee_id=employee_id,
        doc_type=DocType(row["doc_type"]),
        status=DocStatus(row["status"]),
        outcome=ValidationOutcome(row["outcome"]) if row["outcome"] else None,
        validated_at=row["validated_at"],
        source_hash=row["source_hash"] or "",
    )
