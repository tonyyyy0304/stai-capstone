"""ReAct agent loop (Module 7: ReAct Agent).

run_turn() is the core entry point:
    run_turn(session_id, message, history) -> AgentResponse

handle_message() adapts run_turn() to src/api.py's ChatResponse contract
(Member 4's _try_agent_orchestrator looks for this exact name) and returns a
plain dict rather than importing api.py's models, to avoid a circular import.
It's also where session/long-term memory (src/memory/) is wired in — run_turn()
itself stays a pure function over an explicit history list, so existing tests
that call it directly don't gain surprise DB side effects.

Per turn: classify_intent() (Module 4: Disambiguation) gates the conversation —
ambiguous or low-confidence input gets a clarifying question, out-of-scope input
gets declined, neither reaches a tool. Everything else enters the ReAct loop:
Gemini picks a tool, tools.py executes it, the result is fed back as an
observation, and the model repeats until it answers in plain text or
MAX_REACT_ITERATIONS is hit.

Guardrails (Module 6, src/guardrails/) run in three stages of run_turn():
1. Pre-router (_check_input): free, deterministic prompt-injection regex —
   runs before classify_intent() so obvious attacks never reach an LLM call.
2. Post-classification (_check_semantic_guardrails): reads the is_toxic/
   is_injection_attempt fields classify_intent() already filled in, as a
   semantic backstop for whatever the deterministic layer missed, plus a
   complaint-aware toxicity check that only trusts the semantic signal for
   Intent.COMPLAINT (a harassment complaint quoting abusive language isn't
   itself abusive — see check_toxicity_with_context's docstring). Runs after
   classification because both of these need it, at zero incremental LLM
   cost since the router call already happened.
3. Output (check_grounding): re-verifies citations before the final
   AgentResponse is returned.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import ValidationError

from src import config
from src.agent import prompts, tools, usage
from src.agent.router import classify_intent, needs_clarification
from src.guardrails.grounding import check_grounding
from src.guardrails.input_checks import check_injection_semantic, check_topic_and_injection
from src.guardrails.toxicity import check_toxicity_with_context
from src.rag.retriever import RetrievedChunk
from src.schemas import (
    Citation,
    ComplaintCategory,
    ComplaintTicket,
    GuardrailResult,
    Intent,
    IntentClassification,
    Severity,
    TokenUsage,
    WebCitation,
)

OUT_OF_SCOPE_REPLY = (
    "I can only help with company policy, DOLE labor law questions, and complaint "
    "filing. For anything else, please reach out to the right team directly."
)
FALLBACK_CLARIFYING_TEXT = "Could you clarify what you need help with?"
MAX_ITERATIONS_REPLY = (
    "I wasn't able to finish handling this in the usual number of steps. "
    "I've flagged it for HR to follow up on directly."
)
API_ERROR_REPLY = (
    "I'm having trouble reaching the assistant service right now. Please try again "
    "in a moment, or reach out to HR directly if this is urgent."
)

NON_LABOR_LAW_CATEGORIES = tuple(c for c in config.CATEGORIES if c != "labor_law")

logger = logging.getLogger(__name__)


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
    chunks: list[RetrievedChunk] = field(default_factory=list)
    ticket_id: str | None = None
    escalated: bool = False
    insufficient_context: bool = False
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    steps: list[AgentStep] = field(default_factory=list)


@dataclass
class _RunState:
    """Side effects collected across ReAct iterations for the final AgentResponse."""

    citations: list[Citation] = field(default_factory=list)
    web_citations: list[WebCitation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    ticket_id: str | None = None
    escalated: bool = False
    insufficient_context: bool = False


def _check_input(message: str) -> GuardrailResult:
    """Deterministic, no-LLM-call, pre-router guardrail: prompt-injection
    heuristics only. Topic restriction's authoritative enforcement is the
    intent router's out_of_scope classification further down in run_turn()
    (already a hard block) — see src/guardrails/input_checks.py's docstring
    for why this doesn't duplicate that with keyword topic-matching.

    Toxicity is deliberately NOT checked here — it needs the classification
    result to distinguish an employee's own hostile language from a
    complaint quoting abuse said TO them (see check_toxicity_with_context's
    docstring), so it runs after classify_intent() below instead."""
    return check_topic_and_injection(message)


def _check_semantic_guardrails(message: str, classification: IntentClassification) -> GuardrailResult:
    """Runs after classify_intent() returns — reads the router's is_toxic/
    is_injection_attempt fields (filled in by that same already-mandatory
    call, so this costs no extra LLM request) as a backstop for whatever the
    deterministic pre-router checks missed."""
    result = check_injection_semantic(classification)
    if not result.allowed:
        return result
    return check_toxicity_with_context(message, classification)


def run_turn(
    session_id: str,
    message: str,
    history: list[dict[str, str]] | None = None,
    client=None,
) -> AgentResponse:
    from google.genai.errors import APIError

    from src.agent.llm_client import LLMBackendError

    client = client or config.get_llm_client()
    history = history or []
    steps: list[AgentStep] = []
    turn_started_at = datetime.now(timezone.utc)

    logger.info("session=%s turn_start", session_id)
    guardrail_result = _check_input(message)
    if not guardrail_result.allowed:
        return AgentResponse(reply=guardrail_result.reason, steps=steps)

    try:
        classification = classify_intent(message, history, client=client, session_id=session_id)
        steps.append(
            AgentStep(
                thought=f"classified intent={classification.intent.value} "
                f"confidence={classification.confidence:.2f}",
                tool=None,
                tool_args={},
                observation=classification.model_dump_json(),
            )
        )

        semantic_result = _check_semantic_guardrails(message, classification)
        if not semantic_result.allowed:
            return AgentResponse(
                reply=semantic_result.reason,
                steps=steps,
                token_usage=_turn_token_usage(turn_started_at, session_id),
            )

        if needs_clarification(classification):
            return AgentResponse(
                reply=classification.clarifying_question or FALLBACK_CLARIFYING_TEXT,
                steps=steps,
                token_usage=_turn_token_usage(turn_started_at, session_id),
            )

        if classification.intent == Intent.OUT_OF_SCOPE:
            return AgentResponse(
                reply=OUT_OF_SCOPE_REPLY,
                steps=steps,
                token_usage=_turn_token_usage(turn_started_at, session_id),
            )

        reply, run_state = _run_tool_loop(
            message, history, classification, steps, client, session_id
        )
    except (APIError, LLMBackendError) as exc:
        logger.warning("session=%s llm_api_error status=%s", session_id, exc.code)
        return AgentResponse(
            reply=API_ERROR_REPLY,
            steps=steps,
            token_usage=_turn_token_usage(turn_started_at, session_id),
        )

    # Output guardrail: hard-verify (not just prompt-instruct) that every
    # citation actually maps to a chunk retrieved this turn, before the reply
    # goes back to the employee.
    verified_citations, insufficient_context = check_grounding(
        run_state.citations, run_state.chunks, run_state.insufficient_context
    )

    return AgentResponse(
        reply=reply,
        citations=verified_citations,
        web_citations=run_state.web_citations,
        chunks=run_state.chunks,
        ticket_id=run_state.ticket_id,
        escalated=run_state.escalated,
        insufficient_context=insufficient_context,
        token_usage=_turn_token_usage(turn_started_at, session_id),
        steps=steps,
    )


def _turn_token_usage(turn_started_at: datetime, session_id: str) -> TokenUsage:
    """Sums the usage log rows this turn's calls just wrote, rather than
    threading an accumulator through every call site."""
    summary = usage.get_usage_summary(since=turn_started_at, session_id=session_id)
    return TokenUsage(
        prompt_tokens=summary["prompt_tokens"],
        completion_tokens=summary["completion_tokens"],
        total_tokens=summary["total_tokens"],
    )


def _run_tool_loop(
    message: str,
    history: list[dict[str, str]],
    classification: IntentClassification,
    steps: list[AgentStep],
    client,
    session_id: str,
) -> tuple[str, _RunState]:
    from google.genai import types

    run_state = _RunState()
    contents = _build_initial_contents(history, message, classification)
    agent_tools = types.Tool(function_declarations=_function_declarations())

    for iteration in range(config.MAX_REACT_ITERATIONS):
        response = client.models.generate_content(
            model=config.ACTIVE_CHAT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=prompts.REACT_SYSTEM_PROMPT,
                tools=[agent_tools],
                temperature=0.2,
            ),
        )
        usage.record_usage(config.ACTIVE_CHAT_MODEL, usage.extract_usage(response), session_id=session_id)
        candidate_content = response.candidates[0].content
        function_calls = [
            part.function_call for part in candidate_content.parts if part.function_call
        ]

        if not function_calls:
            return response.text or FALLBACK_CLARIFYING_TEXT, run_state

        contents.append(candidate_content)
        response_parts = []
        for call in function_calls:
            args = dict(call.args or {})
            observation = _execute_tool(call.name, args, client, run_state, session_id)
            steps.append(
                AgentStep(
                    thought=f"iteration {iteration + 1}: calling {call.name}",
                    tool=call.name,
                    tool_args=args,
                    observation=json.dumps(observation),
                )
            )
            response_parts.append(
                types.Part.from_function_response(name=call.name, response=observation)
            )
        contents.append(types.Content(role="user", parts=response_parts))

    return MAX_ITERATIONS_REPLY, run_state


def _build_initial_contents(
    history: list[dict[str, str]], message: str, classification: IntentClassification
):
    from google.genai import types

    contents = []
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))
    hint = f"[router: intent={classification.intent.value}, category={classification.category}]\n{message}"
    contents.append(types.Content(role="user", parts=[types.Part(text=hint)]))
    return contents


def _function_declarations() -> list:
    from google.genai import types

    return [
        types.FunctionDeclaration(
            name="search_kb",
            description="Search the internal company policy knowledge base "
            "(Code of Conduct, leave, benefits, payroll, onboarding, etc.).",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "question": types.Schema(
                        type="STRING", description="The employee's question"
                    ),
                    "category": types.Schema(
                        type="STRING",
                        enum=list(NON_LABOR_LAW_CATEGORIES),
                        description="Policy category to filter by, if known",
                    ),
                },
                required=["question"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_web",
            description="Search official Philippine government sources for DOLE/labor "
            "law questions not covered by company policy.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "question": types.Schema(type="STRING", description="The labor-law question")
                },
                required=["question"],
            ),
        ),
        types.FunctionDeclaration(
            name="file_complaint",
            description="File a formal HR complaint. Only call once category, severity, "
            "and a description are known.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "category": types.Schema(type="STRING", enum=[c.value for c in ComplaintCategory]),
                    "severity": types.Schema(type="STRING", enum=[s.value for s in Severity]),
                    "description": types.Schema(
                        type="STRING", description="What happened, in the employee's words"
                    ),
                    "parties_involved": types.Schema(
                        type="ARRAY", items=types.Schema(type="STRING")
                    ),
                    "incident_date": types.Schema(type="STRING"),
                    "desired_outcome": types.Schema(type="STRING"),
                },
                required=["category", "severity", "description"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_ticket_status",
            description="Look up the status of a previously filed complaint.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"ticket_id": types.Schema(type="STRING")},
                required=["ticket_id"],
            ),
        ),
    ]


def _execute_tool(
    name: str, args: dict, client, run_state: _RunState, session_id: str
) -> dict:
    if name == "search_kb":
        answer, chunks = tools.search_kb(args["question"], category=args.get("category"))
        run_state.citations = answer.citations
        run_state.web_citations = answer.web_citations
        run_state.chunks = chunks
        run_state.insufficient_context = answer.insufficient_context
        return {
            "answer": answer.answer,
            "insufficient_context": answer.insufficient_context,
            "chunks_found": len(chunks),
        }

    if name == "search_web":
        answer = tools.search_web(args["question"], client=client, session_id=session_id)
        run_state.citations = answer.citations
        run_state.web_citations = answer.web_citations
        run_state.insufficient_context = answer.insufficient_context
        return {"answer": answer.answer, "insufficient_context": answer.insufficient_context}

    if name == "file_complaint":
        try:
            ticket = ComplaintTicket(
                category=args["category"],
                severity=args["severity"],
                description=args["description"],
                parties_involved=args.get("parties_involved", []),
                incident_date=args.get("incident_date"),
                desired_outcome=args.get("desired_outcome"),
            )
        except (ValidationError, KeyError) as exc:
            return {"error": f"could not file complaint, invalid fields: {exc}"}

        ticket_id = tools.file_complaint(ticket)
        escalated = tools.should_escalate(ticket.category)
        if escalated:
            tools.escalate_to_hr(ticket_id, reason=f"category={ticket.category.value}")
        run_state.ticket_id = ticket_id
        run_state.escalated = escalated
        return {"ticket_id": ticket_id, "escalated": escalated}

    if name == "get_ticket_status":
        status = tools.get_ticket_status(args["ticket_id"])
        return status or {"error": "no ticket found with that id"}

    return {"error": f"unknown tool: {name}"}


def _source_dict_from_chunk(chunk: RetrievedChunk) -> dict:
    preview = " ".join(chunk.text.split())
    if len(preview) > 360:
        preview = preview[:357].rstrip() + "..."
    return {
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "section_path": chunk.section_path,
        "similarity": round(chunk.similarity, 4),
        "effective_date": chunk.effective_date,
        "version": chunk.version,
        "preview": preview,
    }


def handle_message(
    message: str,
    session_id: str | None = None,
    employee_id: str | None = None,
    history: list[dict[str, str]] | None = None,
    client=None,
) -> dict:
    """Adapter for src/api.py's ChatResponse contract. Also where session and
    long-term memory (src/memory/) are wired in — run_turn() itself stays a
    pure function over an explicit history list.

    If `history` is passed explicitly (tests, callers that manage their own
    context), it's used as-is and no memory read/write happens — matches the
    prior behavior so existing callers aren't surprised by new DB writes.
    Otherwise history is loaded from src/memory/persistent.get_context()
    (trimmed recent turns + rolling summary, seeded from the employee's most
    recent summary on a brand-new session), and the turn is appended back
    afterward.
    """
    from src.memory import persistent as memory_persistent
    from src.memory import session as memory_session

    session_id = session_id or str(uuid.uuid4())
    manage_memory = history is None
    if manage_memory:
        history = memory_persistent.get_context(session_id, employee_id)

    result = run_turn(session_id, message, history=history, client=client)

    if manage_memory:
        memory_session.append_turn(session_id, "user", message)
        memory_session.append_turn(session_id, "assistant", result.reply)
        memory_persistent.maybe_update_summary(session_id, employee_id, client=client)

    actions = []
    if result.ticket_id:
        label = "Complaint filed"
        if result.escalated:
            label += " and escalated to HR"
        actions.append(
            {
                "type": "complaint_filed",
                "label": label,
                "status": "completed",
                "ticket_id": result.ticket_id,
            }
        )

    return {
        "session_id": session_id,
        "reply": result.reply,
        "citations": result.citations,
        "sources": [_source_dict_from_chunk(chunk) for chunk in result.chunks],
        "web_citations": result.web_citations,
        "actions": actions,
        "insufficient_context": result.insufficient_context,
        "token_usage": result.token_usage,
    }
