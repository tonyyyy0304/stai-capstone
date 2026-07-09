"""Tools the ReAct agent can call (Module 8: Tool Use).

- search_kb    -> internal RAG, delegates to answer_question() in src/rag/answerer.py
- search_web   -> fallback for DOLE/labor-law questions the internal KB doesn't cover;
                  Tavily search (domain-restricted, provider-agnostic) + one structured
                  Gemini call to shape results into a GroundedAnswer
- file_complaint / get_ticket_status -> SQLite-backed ticket tools, exposed to the model
- escalate_to_hr -> deterministic hand-off, code-only; never exposed as a callable
  tool. The should_escalate/danger_scan rules it relies on live in
  src/guardrails/ (Module 6), so tools.py itself carries no escalation *logic* --
  only execution (SQLite write + notification hand-off).
"""

import json
import logging
import smtplib
import sqlite3
import ssl
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from os import environ

from src import config
from src.agent import prompts, usage
from src.rag.answerer import answer_question
from src.rag.retriever import RetrievedChunk
from src.schemas import (
    AnswerSource,
    ComplaintTicket,
    EscalationEvent,
    GroundedAnswer,
    WebCitation,
)

logger = logging.getLogger(__name__)

NO_WEB_ANSWER = (
    "I couldn't find a reliable DOLE/official source for this either, so I don't want "
    "to guess. I can route your question to the HR team instead — would you like that?"
)


# --- search_kb (internal RAG) -------------------------------------------------

def search_kb(
    question: str, category: str | None = None
) -> tuple[GroundedAnswer, list[RetrievedChunk]]:
    """Internal HR-policy RAG tool. Thin wrapper so the orchestrator has a single
    tool-call surface; all retrieval/grounding logic lives in src/rag/answerer.py."""
    return answer_question(question, category=category)


# --- search_web (DOLE/labor-law fallback) -------------------------------------

def search_web(
    question: str, client=None, session_id: str | None = None, tavily_client=None
) -> GroundedAnswer:
    """Fallback for questions the internal KB doesn't cover (e.g. DOLE labor law).

    Tavily does the actual searching (domain-restricted, works the same regardless
    of which LLM serves chat), then one Gemini call with response_schema shapes the
    results into a typed GroundedAnswer.
    """
    from google.genai import types

    results = _tavily_search(question, tavily_client=tavily_client)
    if not results:
        return no_web_answer()

    client = client or config.get_llm_client()
    shape_response = client.models.generate_content(
        model=config.ACTIVE_CHAT_MODEL,
        contents=prompts.WEB_ANSWER_SHAPE_PROMPT.format(
            question=question, search_results=_format_tavily_results(results)
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GroundedAnswer,
            temperature=0.0,
        ),
    )
    usage.record_usage(config.ACTIVE_CHAT_MODEL, usage.extract_usage(shape_response), session_id=session_id)
    answer: GroundedAnswer | None = shape_response.parsed
    if answer is None or answer.insufficient_context:  # fail closed
        return no_web_answer()

    web_citations = [
        WebCitation(url=r["url"], title=r.get("title") or r["url"], snippet=(r.get("content") or "")[:300])
        for r in results
    ]
    return answer.model_copy(
        update={"source": AnswerSource.WEB, "web_citations": web_citations, "citations": []}
    )


def _tavily_search(question: str, tavily_client=None) -> list[dict]:
    """Domain-restricted Tavily search. Fails closed (empty list) on API errors
    rather than raising — a search outage shouldn't crash the whole agent turn."""
    from tavily.errors import (
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        UsageLimitExceededError,
    )
    from tavily.errors import TimeoutError as TavilyTimeoutError

    tavily_client = tavily_client or config.get_tavily_client()
    try:
        response = tavily_client.search(
            query=question,
            include_domains=list(config.DOLE_ALLOWED_DOMAINS),
            max_results=config.TAVILY_MAX_RESULTS,
        )
    except (
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        UsageLimitExceededError,
        TavilyTimeoutError,
    ) as exc:
        logger.warning("tavily_search_error question=%r error=%s", question, exc)
        return []
    return response.get("results") or []


def _format_tavily_results(results: list[dict]) -> str:
    parts = []
    for r in results:
        parts.append(f"[url: {r['url']} | title: {r.get('title', '')}]\n{r.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def no_web_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer=NO_WEB_ANSWER,
        citations=[],
        source=AnswerSource.NONE,
        web_citations=[],
        insufficient_context=True,
    )


# --- Ticket tools (SQLite) ----------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT NOT NULL,
            parties_involved TEXT NOT NULL,
            incident_date TEXT,
            desired_outcome TEXT,
            status TEXT NOT NULL,
            escalated INTEGER NOT NULL DEFAULT 0,
            trigger_rule TEXT,
            sla_deadline TEXT,
            redacted_summary TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def file_complaint(ticket: ComplaintTicket) -> str:
    """Writes a validated ComplaintTicket to SQLite, returns the new ticket_id."""
    ticket_id = str(uuid.uuid4())
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO tickets
                (ticket_id, category, severity, description, parties_involved,
                 incident_date, desired_outcome, status, escalated, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'open', 0, ?)
            """,
            (
                ticket_id,
                ticket.category.value,
                ticket.severity.value,
                ticket.description,
                json.dumps(ticket.parties_involved),
                ticket.incident_date,
                ticket.desired_outcome,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return ticket_id


def get_ticket_status(ticket_id: str) -> dict | None:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


# --- Escalation handoff (Module 6: Guardrails, Module 8: Tool Use) -----------

def _send_email_notification(event: EscalationEvent) -> bool:
    """Attempt a real SMTP send. Returns True on success, False on anything
    that should fall back to the mock channel -- never raises, since a
    misconfigured or unreachable mail server must not fail the employee's
    chat turn."""
    smtp_host = environ.get("SMTP_HOST", "")
    to_addr = environ.get("HR_ESCALATION_EMAIL_TO", "")
    if not smtp_host or not to_addr:
        return False  # not configured -> caller falls back to the mock channel

    from_addr = environ.get("HR_ESCALATION_EMAIL_FROM", "hr-agent@localhost")
    smtp_port = int(environ.get("SMTP_PORT", "587"))
    smtp_username = environ.get("SMTP_USERNAME", "")
    smtp_password = environ.get("SMTP_PASSWORD", "")

    message = EmailMessage()
    message["Subject"] = f"[HR Escalation] {event.category.value} ({event.severity.value})"
    message["From"] = from_addr
    message["To"] = to_addr
    message.set_content(
        "An HR complaint has been escalated for human review.\n\n"
        f"Ticket ID: {event.ticket_id}\n"
        f"Category: {event.category.value}\n"
        f"Severity: {event.severity.value}\n"
        f"Trigger rule: {event.trigger_rule.value}\n"
        f"SLA deadline: {event.sla_deadline.isoformat()}\n\n"
        f"Summary: {event.redacted_summary}\n"
    )

    try:
        if smtp_port == 465:
            # Implicit TLS from the first byte (Gmail/most providers' "SSL" port).
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10, context=ssl.create_default_context()) as server:
                if smtp_username:
                    server.login(smtp_username, smtp_password)
                server.send_message(message)
        else:
            # STARTTLS: connect in plaintext, then upgrade (587 is the near-universal default).
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls(context=ssl.create_default_context())
                if smtp_username:
                    server.login(smtp_username, smtp_password)
                server.send_message(message)
        return True
    except (smtplib.SMTPException, OSError, TimeoutError, ssl.SSLError) as exc:
        logger.warning("smtp_escalation_send_failed ticket=%s error=%s", event.ticket_id, exc)
        return False


def _write_mock_outbox(event: EscalationEvent) -> None:
    """Local fallback notification channel: appends a JSON line so an
    escalation stays inspectable during dev/demo without a real mail server."""
    outbox_path = config.DATA_DIR / "escalation_outbox.jsonl"
    outbox_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ticket_id": event.ticket_id,
        "category": event.category.value,
        "severity": event.severity.value,
        "trigger_rule": event.trigger_rule.value,
        "sla_deadline": event.sla_deadline.isoformat(),
        "redacted_summary": event.redacted_summary,
        "created_at": event.created_at.isoformat(),
    }
    with open(outbox_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def escalate_to_hr(event: EscalationEvent) -> dict:
    """Notify HR of an escalation and flag the ticket.

    event is an EscalationEvent (non-PII) built by
    src/guardrails/form_pii.py::to_escalation_event -- this function never
    receives the employee's free-text description, named parties, or contact
    info, only category/severity/trigger metadata.
    """
    conn = _get_connection()
    try:
        conn.execute(
            """
            UPDATE tickets
            SET escalated = 1, status = 'escalated', trigger_rule = ?,
                sla_deadline = ?, redacted_summary = ?
            WHERE ticket_id = ?
            """,
            (
                event.trigger_rule.value,
                event.sla_deadline.isoformat(),
                event.redacted_summary,
                event.ticket_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    delivered = _send_email_notification(event)
    channel = "smtp" if delivered else "mock"
    if not delivered:
        _write_mock_outbox(event)

    logger.warning(
        "ESCALATION ticket=%s trigger=%s severity=%s channel=%s",
        event.ticket_id,
        event.trigger_rule.value,
        event.severity.value,
        channel,
    )
    return {
        "ticket_id": event.ticket_id,
        "escalated": True,
        "channel": channel,
        "sla_deadline": event.sla_deadline.isoformat(),
    }
