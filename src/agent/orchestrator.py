"""ReAct agent loop (Module 7: ReAct Agent).

run_turn() is the single entry point Member 4's FastAPI layer should call:
    run_turn(session_id, message, history) -> AgentResponse

Flow per turn:
    classify_intent()
      -> AMBIGUOUS / low confidence  : ask the clarifying question, no tool call
      -> OUT_OF_SCOPE                : decline, no tool call
      -> FAQ                         : search_kb() -> insufficient? -> search_web()
      -> COMPLAINT                   : slot-fill ComplaintDraft -> file_complaint()
                                        -> should_escalate() -> escalate_to_hr()

Input/output guardrail hooks are stubbed pass-through until Member 3 ships
src/guardrails/ — swap _check_input_stub for the real check_input import then.
Session/long-term memory is likewise stubbed: history is passed in by the
caller each turn rather than persisted here, until Member 3 ships src/memory/.
"""

from dataclasses import dataclass, field

from src import config
from src.agent import prompts, tools
from src.agent.router import classify_intent, format_history, needs_clarification
from src.schemas import (
    Citation,
    ComplaintDraft,
    ComplaintTicket,
    Intent,
    IntentClassification,
    REQUIRED_COMPLAINT_FIELDS,
    WebCitation,
)

OUT_OF_SCOPE_REPLY = (
    "I can only help with company policy, DOLE labor law questions, and complaint "
    "filing. For anything else, please reach out to the right team directly."
)

FIELD_PROMPTS = {
    "category": "What kind of issue is this — harassment, safety, payroll, etc.?",
    "severity": "How serious would you say this is (low, medium, high, or critical)?",
    "description": "Can you describe what happened, in your own words?",
}


@dataclass
class AgentStep:
    thought: str
    tool: str | None
    tool_args: dict
    observation: str


@dataclass
class AgentResponse:
    reply: str
    citations: list[Citation] = field(default_factory=list)
    web_citations: list[WebCitation] = field(default_factory=list)
    ticket_id: str | None = None
    escalated: bool = False
    steps: list[AgentStep] = field(default_factory=list)


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""


def _check_input_stub(message: str) -> GuardrailResult:
    """TEMPORARY pass-through until src/guardrails/input_checks.py ships
    (Member 3): topic filter, prompt-injection heuristics, PII detection."""
    return GuardrailResult(allowed=True)


def run_turn(
    session_id: str,
    message: str,
    history: list[dict[str, str]] | None = None,
    client=None,
) -> AgentResponse:
    client = client or config.get_gemini_client()
    history = history or []
    steps: list[AgentStep] = []

    guardrail_result = _check_input_stub(message)
    if not guardrail_result.allowed:
        return AgentResponse(reply=guardrail_result.reason, steps=steps)

    classification = classify_intent(message, history, client=client)
    steps.append(
        AgentStep(
            thought=f"classified intent={classification.intent.value} "
            f"confidence={classification.confidence:.2f}",
            tool=None,
            tool_args={},
            observation=classification.model_dump_json(),
        )
    )

    if needs_clarification(classification):
        return AgentResponse(
            reply=classification.clarifying_question or FALLBACK_CLARIFYING_TEXT,
            steps=steps,
        )

    if classification.intent == Intent.OUT_OF_SCOPE:
        return AgentResponse(reply=OUT_OF_SCOPE_REPLY, steps=steps)

    if classification.intent == Intent.FAQ:
        return _handle_faq(message, classification, steps, client)

    if classification.intent == Intent.COMPLAINT:
        return _handle_complaint(message, history, steps, client)

    # Unreachable given the Intent enum, but fail closed rather than crash mid-conversation.
    return AgentResponse(reply=OUT_OF_SCOPE_REPLY, steps=steps)


FALLBACK_CLARIFYING_TEXT = "Could you clarify what you need help with?"


def _handle_faq(
    message: str,
    classification: IntentClassification,
    steps: list[AgentStep],
    client,
) -> AgentResponse:
    answer, chunks = tools.search_kb(message, category=classification.category)
    steps.append(
        AgentStep(
            thought="tried internal KB",
            tool="search_kb",
            tool_args={"query": message, "category": classification.category},
            observation=f"insufficient_context={answer.insufficient_context}, chunks={len(chunks)}",
        )
    )

    if answer.insufficient_context:
        answer = tools.search_web(message, client=client)
        steps.append(
            AgentStep(
                thought="internal KB insufficient, tried web fallback",
                tool="search_web",
                tool_args={"query": message},
                observation=f"insufficient_context={answer.insufficient_context}",
            )
        )

    return AgentResponse(
        reply=answer.answer,
        citations=answer.citations,
        web_citations=answer.web_citations,
        steps=steps,
    )


def _handle_complaint(
    message: str,
    history: list[dict[str, str]],
    steps: list[AgentStep],
    client,
) -> AgentResponse:
    draft = _extract_complaint_draft(message, history, client)
    steps.append(
        AgentStep(
            thought="extracted complaint draft",
            tool=None,
            tool_args={},
            observation=draft.model_dump_json(),
        )
    )

    missing = _missing_required_fields(draft)
    if missing:
        return AgentResponse(reply=FIELD_PROMPTS[missing[0]], steps=steps)

    ticket = ComplaintTicket(
        category=draft.category,
        severity=draft.severity,
        description=draft.description,
        parties_involved=draft.parties_involved,
        incident_date=draft.incident_date,
        desired_outcome=draft.desired_outcome,
    )
    ticket_id = tools.file_complaint(ticket)
    steps.append(
        AgentStep(
            thought="filed complaint",
            tool="file_complaint",
            tool_args={"category": ticket.category.value},
            observation=ticket_id,
        )
    )

    escalated = tools.should_escalate(ticket.category)
    if escalated:
        tools.escalate_to_hr(ticket_id, reason=f"category={ticket.category.value}")
        steps.append(
            AgentStep(
                thought="escalated per deterministic rule",
                tool="escalate_to_hr",
                tool_args={"ticket_id": ticket_id},
                observation="escalated=True",
            )
        )

    return AgentResponse(
        reply=_complaint_confirmation(ticket_id, escalated),
        ticket_id=ticket_id,
        escalated=escalated,
        steps=steps,
    )


def _extract_complaint_draft(
    message: str, history: list[dict[str, str]], client
) -> ComplaintDraft:
    """Re-extracts the draft from full history each turn (stateless). Once
    Member 3's session memory (src/memory/) ships, this should read/write an
    incrementally-updated draft instead of re-deriving it every call."""
    from google.genai import types

    response = client.models.generate_content(
        model=config.CHAT_MODEL,
        contents=prompts.COMPLAINT_EXTRACTION_PROMPT.format(
            history=format_history(history), message=message
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ComplaintDraft,
            temperature=0.0,
        ),
    )
    return response.parsed or ComplaintDraft()


def _missing_required_fields(draft: ComplaintDraft) -> list[str]:
    return [f for f in REQUIRED_COMPLAINT_FIELDS if getattr(draft, f) is None]


def _complaint_confirmation(ticket_id: str, escalated: bool) -> str:
    base = f"I've filed your complaint (ticket {ticket_id})."
    if escalated:
        return base + " Given what you've described, I've also escalated this to HR directly."
    return base + " HR will follow up on it."
