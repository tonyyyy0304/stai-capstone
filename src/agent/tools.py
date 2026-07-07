"""Tools the ReAct agent can call (Module 8: Tool Use).

- search_kb    -> internal RAG, delegates to answer_question() in src/rag/answerer.py
- search_web   -> fallback for DOLE/labor-law questions the internal KB doesn't
                  cover; Gemini native google_search grounding, domain-restricted
- file_complaint / get_ticket_status -> SQLite-backed ticket tools, exposed to the model
- escalate_to_hr / should_escalate -> deterministic, code-only; never exposed as a
  callable tool, so escalation for harassment/safety/legal is never an LLM decision
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone

from src import config
from src.agent import prompts
from src.rag.answerer import answer_question
from src.rag.retriever import RetrievedChunk
from src.schemas import (
    AnswerSource,
    ComplaintCategory,
    ComplaintTicket,
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

def search_web(question: str, client=None) -> GroundedAnswer:
    """Fallback for questions the internal KB doesn't cover (e.g. DOLE labor law).

    Two Gemini calls, because google_search grounding and response_schema cannot
    be combined in a single call:
      1. Grounded search restricted to config.DOLE_ALLOWED_DOMAINS, no response_schema.
      2. Reshape the grounded text + grounding URLs into a typed GroundedAnswer.
    """
    from google.genai import types

    client = client or config.get_gemini_client()

    search_response = client.models.generate_content(
        model=config.CHAT_MODEL,
        contents=prompts.WEB_SEARCH_PROMPT.format(
            domains=", ".join(config.DOLE_ALLOWED_DOMAINS), question=question
        ),
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
        ),
    )
    grounded_text = (search_response.text or "").strip()
    web_citations = _extract_web_citations(search_response)
    if not grounded_text or not web_citations:
        return no_web_answer()

    shape_response = client.models.generate_content(
        model=config.CHAT_MODEL,
        contents=prompts.WEB_ANSWER_SHAPE_PROMPT.format(
            question=question, grounded_text=grounded_text
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GroundedAnswer,
            temperature=0.0,
        ),
    )
    answer: GroundedAnswer | None = shape_response.parsed
    if answer is None or answer.insufficient_context:  # fail closed
        return no_web_answer()

    return answer.model_copy(
        update={"source": AnswerSource.WEB, "web_citations": web_citations, "citations": []}
    )


def _extract_web_citations(response) -> list[WebCitation]:
    """Pull grounding URLs off a google_search response, restricted to the DOLE allowlist."""
    citations: list[WebCitation] = []
    for candidate in getattr(response, "candidates", None) or []:
        metadata = getattr(candidate, "grounding_metadata", None)
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if not web or not web.uri:
                continue
            if not any(domain in web.uri for domain in config.DOLE_ALLOWED_DOMAINS):
                continue
            citations.append(WebCitation(url=web.uri, title=web.title or web.uri))
    return citations


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


ESCALATION_CATEGORIES = frozenset(
    {
        ComplaintCategory.HARASSMENT,
        ComplaintCategory.DISCRIMINATION,
        ComplaintCategory.SAFETY,
        ComplaintCategory.LEGAL,
    }
)


def should_escalate(category: ComplaintCategory) -> bool:
    """Deterministic escalation rule — never delegated to the model."""
    return category in ESCALATION_CATEGORIES


def escalate_to_hr(ticket_id: str, reason: str) -> dict:
    """Mocked HR escalation channel (email/webhook) for dev; flags the ticket."""
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE tickets SET escalated = 1, status = 'escalated' WHERE ticket_id = ?",
            (ticket_id,),
        )
        conn.commit()
    finally:
        conn.close()
    logger.warning("ESCALATION ticket=%s reason=%s", ticket_id, reason)
    return {"ticket_id": ticket_id, "escalated": True, "reason": reason}
