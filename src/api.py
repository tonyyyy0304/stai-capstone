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

from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.guardrails.doc_validation import (
    apply_cross_document_result,
    validate_cross_document,
    validate_document,
    validate_id_document,
)
from src.memory import onboarding_status
from src.monitoring import chat_trace, configure_mlflow, doc_trace
from src.ocr.extractor import extract_document, load_cached_result
from src.rag.answerer import answer_question
from src.rag.retriever import RetrievedChunk
from src.schemas import (
    ChecklistStatus,
    Citation,
    DocType,
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


@app.post("/upload-doc", response_model=UploadDocResponse)
async def upload_doc(
    file: UploadFile,
    employee_id: str = Form(...),
    doc_type: Literal["nbi_clearance", "government_id"] = Form(...),
    full_name: str = Form(...),
    date_of_birth: str = Form(...),
) -> UploadDocResponse:
    """Runs the full CV pipeline synchronously (quality gate -> extraction ->
    validation -> checklist record -> cross-document check against any
    sibling document already on file), outside the agent's ReAct loop
    entirely (CV_INTEGRATION.md §1.5). Degrades toward needs_review/rejected
    rather than a 500 on any processing failure, matching
    _try_agent_orchestrator's no-HTTPException convention.
    """
    raw_bytes = await file.read()

    with doc_trace(session_id=employee_id, doc_type=doc_type) as trace:
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
        onboarding_status.record_result(employee_id, doc_type_enum, validation, source_hash=source_hash)

        # Cross-document check: only runs when a sibling document is already on
        # file, via a cache-hit re-extraction (extractor.load_cached_result) —
        # zero new API calls, no new image bytes (CV_INTEGRATION.md §2.7).
        other_type = _OTHER_DOC_TYPE[doc_type]
        sibling = onboarding_status.get_document(employee_id, other_type)
        if extracted is not None and sibling is not None and sibling.source_hash:
            sibling_result = load_cached_result(sibling.source_hash, other_type)
            if sibling_result is not None:
                nbi_result = extracted if doc_type_enum == DocType.NBI_CLEARANCE else sibling_result
                id_result = sibling_result if doc_type_enum == DocType.NBI_CLEARANCE else extracted
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

        return UploadDocResponse(validation=validation, checklist=onboarding_status.get_status(employee_id))


@app.get("/onboarding-status/{employee_id}", response_model=ChecklistStatus)
def onboarding_status_endpoint(employee_id: str) -> ChecklistStatus:
    return onboarding_status.get_status(employee_id)


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
