"""Pure composition of the HR handoff packet (Component 14 follow-on,
CV integration's CLOSE-THE-LOOP feature).

No I/O — takes already-in-memory extraction results plus the checklist that
gated the send, returns an HrPacket. This is what makes it trivially
unit-testable and snapshot-testable (see tests/test_hr_email.py); the caller
(src/api.py's /upload-doc background task) is responsible for the cache
lookups (src/ocr/extractor.load_cached_result) and the actual send
(src/notifications/email_client.py).

Name-discrepancy values are the one place this module would reach past the
redacted OnboardingDocument/RuleResult shapes into raw extracted names — same
precedent the deleted Midterm emailer set for ComplaintTicket
(git show 23417c9~1:src/agent/tools.py, _build_email_bodies's docstring: "the
one place that intentionally reaches past EscalationEvent into the ticket").
Not exercised by the current gate (a cross-document mismatch escalates to
needs_review, which never reaches this module — see api.py's gate), kept as
a documented, unused extension point (HrPacket.discrepancy) rather than
built out further.
"""

from __future__ import annotations

import hashlib
from datetime import date

from dateutil.relativedelta import relativedelta

from src import config
from src.schemas import (
    ChecklistStatus,
    DocStatus,
    DocType,
    HrCorrectionNotice,
    HrPacket,
    HrReviewAlert,
    IdExtractionResult,
    NbiExtractionResult,
    OutstandingItem,
    ValidationResult,
    VerifiedDocSummary,
)


def _parse_flexible_date(value: str | None) -> date | None:
    """Bare ISO date or ISO datetime — same shape as
    doc_validation._parse_flexible_date, kept local for the same reason that
    module gives (a date-parsing concern local to this module's own job, not
    a shared registry)."""
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        pass
    try:
        from datetime import datetime

        return datetime.fromisoformat(value.strip()).date()
    except ValueError:
        return None


def _resubmit_by(nbi_result: NbiExtractionResult | None) -> tuple[str, str]:
    """Returns (printed_valid_until, employer_resubmit_by) as ISO strings
    (empty when unavailable). Mirrors doc_validation._rule_validity_window_nbi's
    effective_deadline = min(printed valid_until, date_printed +
    NBI_VALIDITY_MONTHS) — same rule, shown to HR as two separate dates
    rather than collapsed into one, per config.NBI_VALIDITY_MONTHS's
    docstring: the six-month window is an employer freshness policy layered
    ON TOP of (not instead of) the document's own printed validity."""
    if nbi_result is None:
        return "", ""
    printed = nbi_result.valid_until.value or ""
    date_printed = _parse_flexible_date(nbi_result.date_printed.value)
    valid_until = _parse_flexible_date(nbi_result.valid_until.value)
    if date_printed is None:
        return printed, ""
    employer_deadline = date_printed + relativedelta(months=config.NBI_VALIDITY_MONTHS)
    effective = min(valid_until, employer_deadline) if valid_until is not None else employer_deadline
    return printed, effective.isoformat()


def _packet_hash(checklist: ChecklistStatus) -> str:
    """sha256 of the sorted source_hashes of the documents in the packet —
    the idempotency key (src/memory/hr_notifications.py). Sorted so upload
    order never changes the hash."""
    hashes = sorted(doc.source_hash for doc in checklist.documents if doc.source_hash)
    return hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()


def compose_packet(
    nbi_result: NbiExtractionResult | None,
    id_result: IdExtractionResult | None,
    checklist: ChecklistStatus,
    faculty_class: str | None,
) -> HrPacket:
    """(nbi_result, id_result, checklist, faculty_class) -> HrPacket.

    nbi_result/id_result are None when the sibling OCR cache entry was
    unavailable (container restart between the two uploads — see
    api.py:309-310's existing silent-skip path for the cross-document check,
    which this deliberately does NOT replicate: a validated hire reaching HR
    with less detail beats not reaching HR at all, so reduced_detail=True
    degrades the packet instead of skipping the send).
    """
    verified_docs = [
        VerifiedDocSummary(doc_type=doc.doc_type, outcome=doc.outcome, validated_at=doc.validated_at or "")
        for doc in checklist.documents
        if doc.status == DocStatus.VALIDATED
    ]
    verified_at = max((doc.validated_at for doc in verified_docs if doc.validated_at), default="")

    outstanding_rows = config.PREEMPLOYMENT_CHECKLIST.get(faculty_class or "", ())
    outstanding = [
        OutstandingItem(item=row["item"], source_doc=config.PREEMPLOYMENT_CHECKLIST_SOURCE, section=row["section"])
        for row in outstanding_rows
    ]

    printed_valid_until, resubmit_by = _resubmit_by(nbi_result)

    return HrPacket(
        employee_id=checklist.employee_id,
        faculty_class=faculty_class,
        faculty_class_label=config.AUDIENCE_LABELS.get(faculty_class or "", faculty_class or "Unspecified"),
        verified_at=verified_at,
        verified_docs=verified_docs,
        outstanding=outstanding,
        nbi_printed_valid_until=printed_valid_until,
        nbi_resubmit_by=resubmit_by,
        reduced_detail=nbi_result is None or id_result is None,
        discrepancy=None,
        packet_hash=_packet_hash(checklist),
    )


def compose_correction_notice(
    doc_type: DocType,
    validation: ValidationResult,
    checklist: ChecklistStatus,
    faculty_class: str | None,
) -> HrCorrectionNotice:
    """(doc_type, validation, checklist, faculty_class) -> HrCorrectionNotice.

    Called only when a document that was VALIDATED regresses to something
    else AFTER HR was already sent a packet for this employee (the gate
    lives in api.py, not here — this function just composes, same "pure,
    no I/O" contract as compose_packet). still_verified reuses compose_packet's
    exact VerifiedDocSummary construction so the two summaries never drift.
    notice_hash reuses _packet_hash unmodified: a regression always changes
    the regressed document's source_hash, which is enough on its own to
    produce a hash distinct from the original "verified" send's packet_hash
    — no separate idempotency mechanism needed."""
    still_verified = [
        VerifiedDocSummary(doc_type=doc.doc_type, outcome=doc.outcome, validated_at=doc.validated_at or "")
        for doc in checklist.documents
        if doc.status == DocStatus.VALIDATED
    ]
    # Read the persisted status back off the checklist rather than re-deriving
    # it from validation.outcome here -- onboarding_status.record_result()
    # already applied the outcome->status mapping when it wrote this row;
    # this just reuses that, one source of truth.
    current_doc = next((doc for doc in checklist.documents if doc.doc_type == doc_type), None)
    new_status = current_doc.status if current_doc is not None else DocStatus.NEEDS_REVIEW

    return HrCorrectionNotice(
        employee_id=checklist.employee_id,
        faculty_class_label=config.AUDIENCE_LABELS.get(faculty_class or "", faculty_class or "Unspecified"),
        doc_type=doc_type,
        new_outcome=validation.outcome,
        new_status=new_status,
        message=validation.message,
        still_verified=still_verified,
        notice_hash=_packet_hash(checklist),
    )


def compose_review_alert(
    doc_type: DocType,
    validation: ValidationResult,
    checklist: ChecklistStatus,
    faculty_class: str | None,
    source_hash: str,
) -> HrReviewAlert:
    """(doc_type, validation, checklist, faculty_class, source_hash) ->
    HrReviewAlert. Called whenever THIS upload's outcome is NEEDS_REVIEW —
    independent of compose_correction_notice's regression condition, see
    HrReviewAlert's docstring for why the two can legitimately both fire on
    one request.

    review_hash keys off (doc_type, source_hash) rather than the whole
    checklist (unlike packet_hash/notice_hash) — this alert is about ONE
    document's own content, not about what else happens to be on file for
    this employee, so its idempotency shouldn't shift just because the
    sibling document changes."""
    return HrReviewAlert(
        employee_id=checklist.employee_id,
        faculty_class_label=config.AUDIENCE_LABELS.get(faculty_class or "", faculty_class or "Unspecified"),
        doc_type=doc_type,
        message=validation.message,
        review_hash=hashlib.sha256(f"{doc_type.value}:{source_hash}".encode("utf-8")).hexdigest(),
    )
