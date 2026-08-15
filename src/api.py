"""FastAPI interface for the assistant (branding/scope from src/config.py).

Member 4 owns this boundary: HTTP contracts, request validation, sanitized
monitoring, and API-shaped responses for the Streamlit UI. When Member 2's
agent orchestrator is present it can be plugged in without changing clients;
until then the endpoint serves grounded FAQ answers through the RAG module.

POST /upload-doc / GET /onboarding-status/{employee_id} (Component 14, Phase 6)
run the CV pipeline synchronously and separately from /chat's agent loop — raw
image bytes never cross into the ReAct loop (CV_INTEGRATION.md §1.5). No HR
record source exists anywhere in this codebase, so full_name/date_of_birth are
supplied by the uploader alongside employee_id, not looked up.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.guardrails.doc_validation import (
    apply_cross_document_result,
    validate_cross_document,
    validate_document,
    validate_id_document,
)
from src.memory import hr_notifications, onboarding_status
from src.monitoring import chat_trace, configure_mlflow, doc_trace, email_trace
from src.notifications.email_client import send_email, send_packet
from src.notifications.hr_packet import compose_correction_notice, compose_packet, compose_review_alert
from src.notifications.templates import (
    build_correction_email_bodies,
    build_review_alert_email_bodies,
    correction_subject_line,
    review_alert_subject_line,
)
from src.ocr.extractor import extract_document, load_cached_result
from src.rag.answerer import answer_question
from src.rag.retriever import RetrievedChunk
from src.schemas import (
    ChecklistStatus,
    Citation,
    DocStatus,
    DocType,
    IdExtractionResult,
    NbiExtractionResult,
    NotificationStatus,
    RuleResult,
    TokenUsage,
    ValidationOutcome,
    ValidationResult,
    WebCitation,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_mlflow()
    yield


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Client-generated chat session ID")
    message: str = Field(min_length=1, max_length=4000)
    employee_id: str | None = Field(default=None, description="Optional stable employee ID")
    category: str | None = Field(
        default=None,
        description="Optional retrieval filter: onboarding|conduct|leave|benefits",
    )


class SourceResponse(BaseModel):
    chunk_id: str
    title: str
    section_path: str
    similarity: float
    effective_date: str = ""
    version: str = ""
    page: int = 0
    preview: str


class ActionResponse(BaseModel):
    type: str
    label: str
    status: Literal["completed", "pending", "unavailable"] = "completed"


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourceResponse] = Field(default_factory=list)
    web_citations: list[WebCitation] = Field(default_factory=list)
    actions: list[ActionResponse] = Field(default_factory=list)
    insufficient_context: bool = False
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    chroma_dir_exists: bool
    manifest_exists: bool
    gemini_api_key_configured: bool
    mlflow_tracking_uri: str


class UploadDocResponse(BaseModel):
    validation: ValidationResult
    checklist: ChecklistStatus


class HrNotificationResponse(BaseModel):
    sent: bool
    sent_at: str | None = None


class UsageResponse(BaseModel):
    today: dict[str, int]
    all_time: dict[str, int]
    note: str = (
        "Covers the agent's own LLM calls only (router, ReAct loop, search_web), "
        "combined across whichever backend(s) LLM_PROVIDER has pointed at over "
        "time — Gemini and Ollama usage are logged with different model labels in "
        "the underlying token_usage table but summed together here. Does not "
        "include the plain-RAG fallback path or ingestion embedding calls, which "
        "always use Gemini."
    )


app = FastAPI(
    title=f"{config.ASSISTANT_NAME} API",
    version="0.1.0",
    description=f"REST API for grounded answers about {config.SCOPE_PHRASE}.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _source_from_chunk(chunk: RetrievedChunk) -> SourceResponse:
    preview = " ".join(chunk.text.split())
    if len(preview) > 360:
        preview = preview[:357].rstrip() + "..."
    return SourceResponse(
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        section_path=chunk.section_path,
        similarity=round(chunk.similarity, 4),
        effective_date=chunk.effective_date,
        version=chunk.version,
        page=chunk.page_start,
        preview=preview,
    )


def _try_agent_orchestrator(request: ChatRequest, session_id: str) -> ChatResponse | None:
    """Use the agent orchestrator when it exists.

    Supported shape: handle_message(message=..., session_id=..., employee_id=...)
    returning either ChatResponse, dict, or object with response-like
    attributes. `history` is deliberately not passed — leaving it unset tells
    handle_message() to manage session/long-term memory itself via
    src/memory/ (SQLite-backed, survives a restart), rather than api.py
    maintaining its own in-process copy.
    """
    try:
        from src.agent.orchestrator import handle_message
    except Exception:
        return None

    result = handle_message(
        message=request.message,
        session_id=session_id,
        employee_id=request.employee_id,
    )
    if isinstance(result, ChatResponse):
        return result
    if isinstance(result, dict):
        return ChatResponse.model_validate(result)
    return ChatResponse.model_validate(result.model_dump())


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid4())
    with chat_trace(session_id=session_id, message=request.message) as trace:
        agent_response = _try_agent_orchestrator(request, session_id)
        if agent_response is not None:
            trace["metrics"] = {
                "citation_count": len(agent_response.citations),
                "source_count": len(agent_response.sources),
                "web_citation_count": len(agent_response.web_citations),
                "action_count": len(agent_response.actions),
                "prompt_tokens": agent_response.token_usage.prompt_tokens,
                "completion_tokens": agent_response.token_usage.completion_tokens,
                "total_tokens": agent_response.token_usage.total_tokens,
            }
            trace["tags"] = {
                "route": "agent",
                "insufficient_context": agent_response.insufficient_context,
            }
            return agent_response

        answer, chunks = answer_question(request.message, category=request.category)
        response = ChatResponse(
            session_id=session_id,
            reply=answer.answer,
            citations=answer.citations,
            sources=[_source_from_chunk(chunk) for chunk in chunks],
            insufficient_context=answer.insufficient_context,
        )
        trace["metrics"] = {
            "citation_count": len(response.citations),
            "source_count": len(response.sources),
            "action_count": len(response.actions),
        }
        trace["tags"] = {"route": "rag", "insufficient_context": response.insufficient_context}
        return response


_OTHER_DOC_TYPE = {
    "nbi_clearance": DocType.GOVERNMENT_ID,
    "government_id": DocType.NBI_CLEARANCE,
}


def _sniff_image_mime(data: bytes) -> str | None:
    """Magic-byte sniff, not the filename/declared content-type — a renamed
    file shouldn't be able to claim a MIME type it isn't."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return None


def _rejected_before_processing(detail: str, message: str) -> ValidationResult:
    """A reject that never reached extraction — nothing to record against the
    checklist, since there's no real document attempt behind it (oversized
    file, wrong MIME)."""
    return ValidationResult(
        outcome=ValidationOutcome.REJECTED,
        rules=[RuleResult(rule="fail_safe", passed=False, detail=detail)],
        composite_confidence=0.0,
        message=message,
    )


def _persist_upload(raw_bytes: bytes, mime_type: str, source_hash: str) -> None:
    """Only called when config.EMAIL_ATTACH_ORIGINALS is on — otherwise raw
    bytes never outlive the request (config.PERSIST_UPLOADS). Keyed by
    source_hash so the HR-email attachment loader can find either sibling's
    bytes regardless of which one is "current" this request."""
    ext = "png" if mime_type == "image/png" else "jpg"
    try:
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        (config.UPLOADS_DIR / f"{source_hash}.{ext}").write_bytes(raw_bytes)
    except OSError:
        logger.warning("hr_email_attachment_persist_failed source_hash=%s", source_hash)


def _load_attachments(checklist: ChecklistStatus) -> list[tuple[str, bytes, str]]:
    attachments: list[tuple[str, bytes, str]] = []
    for doc in checklist.documents:
        if not doc.source_hash:
            continue
        matches = sorted(config.UPLOADS_DIR.glob(f"{doc.source_hash}.*"))
        if not matches:
            continue
        path = matches[0]
        mime_type = "image/png" if path.suffix == ".png" else "image/jpeg"
        attachments.append((f"{doc.doc_type.value}{path.suffix}", path.read_bytes(), mime_type))
    return attachments


def _send_hr_packet(
    nbi_result: NbiExtractionResult | None,
    id_result: IdExtractionResult | None,
    checklist: ChecklistStatus,
) -> None:
    """FastAPI BackgroundTasks target: runs after /upload-doc's response has
    already been sent, so a slow or unreachable email provider can never
    degrade the upload endpoint's latency or 200 response (CLAUDE.md
    fail-safe convention; same reasoning as email_client.send_packet's own
    never-raise contract). Wrapped in try/except as a last-resort backstop —
    send_packet() itself already never raises, but composition/attachment-
    loading here could."""
    try:
        packet = compose_packet(nbi_result, id_result, checklist, checklist.faculty_class)
        if hr_notifications.already_sent(checklist.employee_id, packet.packet_hash):
            return
        attempts = hr_notifications.record_attempt(checklist.employee_id, packet.packet_hash)
        attachments = _load_attachments(checklist) if config.EMAIL_ATTACH_ORIGINALS else None
        with email_trace() as trace:
            result = send_packet(packet, attachments=attachments)
            trace["metrics"] = {"attempt_count": attempts}
            trace["tags"] = {
                "email_status": result.status.value,
                "email_provider": "dry_run" if config.EMAIL_DRY_RUN else config.EMAIL_PROVIDER,
            }
        hr_notifications.record_result(checklist.employee_id, packet.packet_hash, result)
    except Exception:
        logger.exception("hr_email_gate_failed employee_id=%s", checklist.employee_id)


def _send_correction_notice(
    doc_type: DocType,
    validation: ValidationResult,
    checklist: ChecklistStatus,
) -> None:
    """FastAPI BackgroundTasks target, sibling to _send_hr_packet -- fires
    when a document that was VALIDATED regresses to something else AFTER HR
    was already sent at least one packet for this employee (the gate is in
    upload_doc below; this just composes+sends, same pattern as
    _send_hr_packet). Reuses the same hr_notifications idempotency table
    keyed by notice_hash instead of packet_hash -- a regression always
    changes the regressed document's source_hash, so the hash is naturally
    distinct from the original "verified" send and already_sent() works
    unmodified, no schema change needed."""
    try:
        notice = compose_correction_notice(doc_type, validation, checklist, checklist.faculty_class)
        if hr_notifications.already_sent(checklist.employee_id, notice.notice_hash):
            return
        attempts = hr_notifications.record_attempt(checklist.employee_id, notice.notice_hash)
        with email_trace() as trace:
            plain_text, html_body = build_correction_email_bodies(notice)
            result = send_email(
                correction_subject_line(notice), plain_text, html_body,
                checklist.employee_id, notice.notice_hash,
            )
            trace["metrics"] = {"attempt_count": attempts}
            trace["tags"] = {
                "email_status": result.status.value,
                "email_provider": "dry_run" if config.EMAIL_DRY_RUN else config.EMAIL_PROVIDER,
                "email_kind": "correction",
            }
        hr_notifications.record_result(checklist.employee_id, notice.notice_hash, result)
    except Exception:
        logger.exception("hr_correction_email_gate_failed employee_id=%s", checklist.employee_id)


def _send_review_alert(
    doc_type: DocType,
    validation: ValidationResult,
    checklist: ChecklistStatus,
    source_hash: str,
) -> None:
    """FastAPI BackgroundTasks target, sibling to _send_hr_packet /
    _send_correction_notice -- fires whenever THIS upload's outcome is
    NEEDS_REVIEW (the gate is in upload_doc below). Idempotency keyed by
    review_hash = hash(doc_type, source_hash), not the whole checklist's
    packet_hash -- this alert is about one document's own content, so
    resubmitting the SAME bytes must not re-alert, independent of whatever
    the sibling document is doing."""
    try:
        alert = compose_review_alert(doc_type, validation, checklist, checklist.faculty_class, source_hash)
        if hr_notifications.already_sent(checklist.employee_id, alert.review_hash):
            return
        attempts = hr_notifications.record_attempt(checklist.employee_id, alert.review_hash)
        with email_trace() as trace:
            plain_text, html_body = build_review_alert_email_bodies(alert)
            result = send_email(
                review_alert_subject_line(alert), plain_text, html_body,
                checklist.employee_id, alert.review_hash,
            )
            trace["metrics"] = {"attempt_count": attempts}
            trace["tags"] = {
                "email_status": result.status.value,
                "email_provider": "dry_run" if config.EMAIL_DRY_RUN else config.EMAIL_PROVIDER,
                "email_kind": "review",
            }
        hr_notifications.record_result(checklist.employee_id, alert.review_hash, result)
    except Exception:
        logger.exception("hr_review_email_gate_failed employee_id=%s", checklist.employee_id)


@app.post("/upload-doc", response_model=UploadDocResponse)
async def upload_doc(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    employee_id: str = Form(...),
    doc_type: Literal["nbi_clearance", "government_id"] = Form(...),
    full_name: str = Form(...),
    date_of_birth: str = Form(...),
    faculty_class: str | None = Form(default=None),
) -> UploadDocResponse:
    """Runs the full CV pipeline synchronously (quality gate -> extraction ->
    validation -> checklist record -> cross-document check against any
    sibling document already on file), outside the agent's ReAct loop
    entirely (CV_INTEGRATION.md §1.5). Degrades toward needs_review/rejected
    rather than a 500 on any processing failure, matching
    _try_agent_orchestrator's no-HTTPException convention.

    When both REQUIRED_ONBOARDING_DOCS reach validated, schedules the HR
    handoff email as a BackgroundTasks job (see _send_hr_packet) so a slow
    or down email provider never adds latency to this endpoint.
    """
    if faculty_class:
        onboarding_status.set_faculty_class(employee_id, faculty_class)

    raw_bytes = await file.read()

    with doc_trace(doc_type=doc_type) as trace:
        if len(raw_bytes) > config.MAX_UPLOAD_BYTES:
            validation = _rejected_before_processing(
                "file exceeds maximum upload size", "This file is too large. Please upload a smaller image."
            )
            trace["tags"] = {"validation_outcome": validation.outcome.value}
            return UploadDocResponse(validation=validation, checklist=onboarding_status.get_status(employee_id))

        mime_type = _sniff_image_mime(raw_bytes)
        if mime_type is None or mime_type not in config.ALLOWED_IMAGE_MIME:
            validation = _rejected_before_processing(
                "unsupported file type", "Please upload a JPEG or PNG image."
            )
            trace["tags"] = {"validation_outcome": validation.outcome.value}
            return UploadDocResponse(validation=validation, checklist=onboarding_status.get_status(employee_id))

        doc_type_enum = DocType(doc_type)
        faculty_record = {"employee_id": employee_id, "full_name": full_name, "date_of_birth": date_of_birth}

        extracted, quality_report = None, None
        try:
            extracted, quality_report = extract_document(raw_bytes, mime_type, expected_doc_type=doc_type_enum)
        except Exception:
            logger.exception("upload_doc_extraction_failed employee_id=%s doc_type=%s", employee_id, doc_type)

        if quality_report is None:
            validation = ValidationResult(
                outcome=ValidationOutcome.NEEDS_REVIEW,
                rules=[RuleResult(rule="fail_safe", passed=False, detail="processing error")],
                composite_confidence=0.0,
                message="Something went wrong processing this document. It has been sent for manual review.",
            )
        elif doc_type_enum == DocType.NBI_CLEARANCE:
            validation = validate_document(extracted, quality_report, faculty_record)
        else:
            validation = validate_id_document(extracted, quality_report, faculty_record)

        source_hash = hashlib.sha256(raw_bytes).hexdigest()
        # Captured BEFORE record_result overwrites the row -- the only way to
        # later tell whether this upload regressed an already-VALIDATED,
        # possibly-already-emailed document (see the correction-notice gate
        # near the end of this function).
        previous_doc = onboarding_status.get_document(employee_id, doc_type_enum)
        onboarding_status.record_result(employee_id, doc_type_enum, validation, source_hash=source_hash)

        # Persisted here (every upload, not just whichever one happens to
        # complete the checklist) so BOTH documents' bytes are available
        # once the HR-email gate below fires on the second one — the first
        # upload's raw bytes would otherwise never be written, since at that
        # point the checklist is still incomplete.
        if config.EMAIL_ATTACH_ORIGINALS:
            _persist_upload(raw_bytes, mime_type, source_hash)

        # Cross-document check: only runs when a sibling document is already on
        # file, via a cache-hit re-extraction (extractor.load_cached_result) —
        # zero new API calls, no new image bytes (CV_INTEGRATION.md §2.7).
        # nbi_result/id_result are also what the HR-email gate below composes
        # the handoff packet from — computed once here, reused there, rather
        # than a second cache lookup.
        other_type = _OTHER_DOC_TYPE[doc_type]
        sibling = onboarding_status.get_document(employee_id, other_type)
        sibling_result = load_cached_result(sibling.source_hash, other_type) if sibling and sibling.source_hash else None
        nbi_result = extracted if doc_type_enum == DocType.NBI_CLEARANCE else sibling_result
        id_result = sibling_result if doc_type_enum == DocType.NBI_CLEARANCE else extracted

        if extracted is not None and sibling_result is not None:
            cross_rule = validate_cross_document(nbi_result, id_result)
            validation = apply_cross_document_result(validation, cross_rule)
            onboarding_status.record_result(employee_id, doc_type_enum, validation, source_hash=source_hash)

        # Sanitized shape only — never a field value (blur_score/skew_deg from
        # the deterministic quality gate; extraction_confidence/fields_* from
        # the validation outcome, not from any ExtractedField.value).
        trace["metrics"] = {
            "extraction_confidence": validation.composite_confidence,
            "fields_extracted": sum(1 for r in validation.rules if r.rule == "completeness" and r.passed),
            "fields_missing": sum(1 for r in validation.rules if r.rule == "completeness" and not r.passed),
        }
        trace["tags"] = {"validation_outcome": validation.outcome.value}
        if quality_report is not None:
            trace["metrics"]["blur_score"] = quality_report.blur_score
            trace["metrics"]["skew_deg"] = quality_report.skew_deg
            trace["tags"]["quality_verdict"] = quality_report.verdict.value

        checklist = onboarding_status.get_status(employee_id)
        if not checklist.missing:
            try:
                background_tasks.add_task(_send_hr_packet, nbi_result, id_result, checklist)
            except Exception:
                logger.exception("hr_email_schedule_failed employee_id=%s", employee_id)

        # Correction gate: this upload's own doc_type just regressed away
        # from VALIDATED (a re-upload of an already-verified document), and
        # HR was already told SOMETHING about this employee before -- so
        # that earlier email is now stale. Mutually exclusive with the
        # "not checklist.missing" branch above in practice (a regression
        # means checklist.missing is non-empty this request).
        current_doc = next((doc for doc in checklist.documents if doc.doc_type == doc_type_enum), None)
        regressed = (
            previous_doc is not None and previous_doc.status == DocStatus.VALIDATED
            and current_doc is not None and current_doc.status != DocStatus.VALIDATED
        )
        if regressed and hr_notifications.has_ever_sent(employee_id):
            try:
                background_tasks.add_task(_send_correction_notice, doc_type_enum, validation, checklist)
            except Exception:
                logger.exception("hr_correction_email_schedule_failed employee_id=%s", employee_id)

        # Review-alert gate: independent of the two gates above -- fires
        # whenever THIS upload's own outcome is NEEDS_REVIEW, regardless of
        # what the document's previous status was. REJECTED is deliberately
        # excluded (deterministic, self-service retry, not a human-judgment
        # case); can legitimately co-fire with the correction gate above on
        # the same request (see HrReviewAlert's docstring).
        if validation.outcome == ValidationOutcome.NEEDS_REVIEW:
            try:
                background_tasks.add_task(_send_review_alert, doc_type_enum, validation, checklist, source_hash)
            except Exception:
                logger.exception("hr_review_email_schedule_failed employee_id=%s", employee_id)

        return UploadDocResponse(validation=validation, checklist=checklist)


@app.get("/onboarding-status/{employee_id}", response_model=ChecklistStatus)
def onboarding_status_endpoint(employee_id: str) -> ChecklistStatus:
    return onboarding_status.get_status(employee_id)


@app.get("/hr-notifications/{employee_id}", response_model=HrNotificationResponse)
def hr_notifications_endpoint(employee_id: str) -> HrNotificationResponse:
    """Backs the UI's "Sent to HR" checklist indicator."""
    rows = hr_notifications.list_for_employee(employee_id)
    sent_rows = [row for row in rows if row["status"] == NotificationStatus.SENT.value]
    if not sent_rows:
        return HrNotificationResponse(sent=False)
    return HrNotificationResponse(sent=True, sent_at=sent_rows[0]["sent_at"])


@app.get("/usage", response_model=UsageResponse)
def usage() -> UsageResponse:
    try:
        from src.agent import usage as usage_tracker
    except Exception:
        empty = {"request_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return UsageResponse(today=empty, all_time=empty)

    return UsageResponse(
        today=usage_tracker.get_usage_today(),
        all_time=usage_tracker.get_usage_all_time(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    gemini_key = bool(__import__("os").environ.get("GEMINI_API_KEY"))
    chroma_exists = config.CHROMA_DIR.exists()
    manifest_exists = config.MANIFEST_PATH.exists()
    return HealthResponse(
        status="ok" if chroma_exists and manifest_exists and gemini_key else "degraded",
        chroma_dir_exists=chroma_exists,
        manifest_exists=manifest_exists,
        gemini_api_key_configured=gemini_key,
        mlflow_tracking_uri=config.MLFLOW_TRACKING_URI,
    )
