"""Tools the ReAct agent can call (Module 8: Tool Use).

- search_kb    -> internal RAG, delegates to answer_question() in src/rag/answerer.py
- search_web   -> fallback for DOLE/labor-law questions the internal KB doesn't cover;
                  Tavily search (domain-restricted, provider-agnostic) + one structured
                  Gemini call to shape results into a GroundedAnswer
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
from src.agent import prompts, usage
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
        model=config.CHAT_MODEL,
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
    """Mocked HR escalation channel (email/webhook) for dev; flags the ticket.

    TODO: swap for the real HR-side escalation handoff once that's built (owned by
    another member — intake form on HR's end, escalation guardrails, notification
    channel). This stub only flags the ticket row and logs a warning.
    """
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
