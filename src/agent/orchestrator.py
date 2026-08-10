"""Turn orchestration (Module 7: agent).

run_turn() is the core entry point:
    run_turn(session_id, message, history) -> AgentResponse

handle_message() adapts run_turn() to src/api.py's ChatResponse contract
(_try_agent_orchestrator looks for this exact name) and returns a plain dict
rather than importing api.py's models, to avoid a circular import. It's also
where session/long-term memory (src/memory/) is wired in — run_turn() itself
stays a pure function over an explicit history list, so existing tests that
call it directly don't gain surprise DB side effects.

Per turn: classify_intent() (Module 4: Disambiguation) gates the conversation —
ambiguous or low-confidence input gets a clarifying question, out-of-scope input
gets declined, neither reaches a tool. Everything else enters _react_loop()
(Module 7: Agent): a model-driven ReAct loop that reasons step by step, choosing
search_kb / search_web / finish each iteration (bounded by MAX_REACT_ITERATIONS).
The model plans retrieval only — it can decompose a multi-part question into
several search_kb queries and reformulate a query that came back insufficient —
and the final answer is synthesized afterward over the union of gathered chunks
via generate_grounded_answer(), so the grounding guardrail still verifies every
citation. This buys robustness on hard multi-hop/ambiguous questions at the cost
of ~1 planning call + 1 tool call per hop.

Guardrails (Module 6, src/guardrails/) run at three stages of run_turn():
1. Pre-router (_check_input): deterministic prompt-injection regex and a
   plain toxicity wordlist, then one LLM-as-judge call
   (src/guardrails/llm_judge.py) covering toxicity/PII/injection/off-topic/
   jailbreak in a single structured request — the deterministic checks
   short-circuit blatant cases for free (Gemini quota is the #1 constraint,
   PLAN.md §2.1/§8) before the judge call runs.
2. Post-router (_check_semantic_guardrails): once classify_intent() has run,
   check_injection_semantic() and check_toxicity_semantic()
   (src/guardrails/input_checks.py, toxicity.py) re-check using the
   classification's is_toxic/is_injection_attempt signal — free, since that
   LLM call already happened. Catches paraphrased abuse/injection the
   pre-router deterministic layer misses.
3. Output (check_grounding): re-verifies citations before the final
   AgentResponse is returned.

Intent.DOCUMENT_UPLOAD / Intent.DOCUMENT_STATUS (Component 14, Phase 6) short-
circuit before _react_loop() the same way OUT_OF_SCOPE already does, rather than
becoming new ReActAction members — a checklist lookup is a deterministic read,
not something that benefits from the model's planning loop, and this keeps
_react_loop()/ReActAction untouched. Raw document bytes never reach this file at
all; POST /upload-doc (src/api.py) runs that pipeline synchronously and
separately (CV_INTEGRATION.md §1.5).
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src import config
from src.agent import prompts, tools, usage
from src.agent.router import classify_intent, format_history, needs_clarification
from src.guardrails.grounding import check_grounding
from src.guardrails.input_checks import (
    check_injection_semantic,
    check_jailbreak_semantic,
    check_topic_and_injection,
)
from src.guardrails.llm_judge import check_input_llm
from src.guardrails.toxicity import check_toxicity, check_toxicity_semantic
from src.rag.retriever import RetrievedChunk
from src.schemas import (
    Citation,
    GuardrailResult,
    Intent,
    IntentClassification,
    ReActAction,
    ReActStep,
    TokenUsage,
    WebCitation,
)

OUT_OF_SCOPE_REPLY = (
    f"I can only help with {config.SCOPE_PHRASE}. For anything else, please reach out to "
    f"{config.HELP_CONTACT} directly."
)
DOCUMENT_UPLOAD_REPLY = (
    "You can submit your document using the upload option in this app. Once it's processed, "
    "just ask me and I can tell you its status here."
)
NO_EMPLOYEE_ID_REPLY = (
    "I don't have your employee ID for this session, so I can't look up your document status. "
    "Please provide it and ask again."
)
FALLBACK_CLARIFYING_TEXT = "Could you clarify what you need help with?"
API_ERROR_REPLY = (
    "I'm having trouble reaching the assistant service right now. Please try again "
    f"in a moment, or contact {config.HELP_CONTACT} directly if this is urgent."
)


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
    insufficient_context: bool = False
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    steps: list[AgentStep] = field(default_factory=list)
    # Dict-shaped to match api.py's ActionResponse (type/label/status) without
    # importing that Pydantic model here — api.py adapts orchestrator output
    # to the HTTP contract, not the other way around. Only ever populated by
    # the DOCUMENT_UPLOAD/DOCUMENT_STATUS short-circuits below, with
    # type="unlock_document_flow" -- src/ui.py watches for that specific
    # type to reveal the Employee ID field, checklist, and uploader, which
    # are otherwise hidden until the conversation actually calls for them.
    actions: list[dict] = field(default_factory=list)


_UNLOCK_DOCUMENT_FLOW_ACTION = {
    "type": "unlock_document_flow",
    "label": "You can now enter your Employee ID and upload documents in the sidebar.",
    "status": "completed",
}


@dataclass
class _RunState:
    """Side effects collected across ReAct iterations for the final AgentResponse."""

    citations: list[Citation] = field(default_factory=list)
    web_citations: list[WebCitation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    insufficient_context: bool = False
    # Set when search_kb detects a faculty-class split it can't resolve — the
    # loop short-circuits and returns this question verbatim (Phase 2).
    clarification: str | None = None


def _check_input(message: str, client=None, session_id: str | None = None) -> GuardrailResult:
    """Pre-router input guardrails, before intent is known. The deterministic,
    no-LLM-call checks — prompt-injection regex and a plain toxicity wordlist —
    short-circuit blatant cases for free (Gemini quota is the #1 constraint).

    Phase 5: the LLM-as-judge is NOT run per turn by default — the router carries
    the same LLM safety signals (is_toxic/is_injection_attempt/is_jailbreak) in a
    call that happens anyway (see _check_semantic_guardrails). Set
    config.ENABLE_LLM_JUDGE to add it back as an extra defense-in-depth layer; it
    fails open, so a failed call never breaks the turn."""
    result = check_topic_and_injection(message)
    if not result.allowed:
        return result
    result = check_toxicity(message)
    if not result.allowed:
        return result
    if config.ENABLE_LLM_JUDGE:
        return check_input_llm(message, client=client, session_id=session_id)
    return GuardrailResult(allowed=True)


def _check_semantic_guardrails(classification: IntentClassification) -> GuardrailResult:
    """Post-router guardrail backstop, run immediately after classify_intent()
    returns — free, since that LLM call already happened and its response schema
    carries is_toxic/is_injection_attempt/is_jailbreak (ROUTER_PROMPT asks for
    all three). This is now the primary LLM safety pass (Phase 5), catching
    paraphrased abuse/injection/jailbreak the deterministic pre-layer misses."""
    result = check_injection_semantic(classification)
    if not result.allowed:
        return result
    result = check_jailbreak_semantic(classification)
    if not result.allowed:
        return result
    return check_toxicity_semantic(classification)


def _format_checklist_reply(status) -> str:
    """Renders a ChecklistStatus into a short human reply. PII-free by
    construction — ChecklistStatus/OnboardingDocument (src/schemas.py) never
    carry an extracted field value, only status/outcome per doc_type."""
    if not status.documents:
        return "I don't see any documents on file for you yet."
    lines = [f"- {doc.doc_type.value.replace('_', ' ')}: {doc.status.value.replace('_', ' ')}"
             for doc in status.documents]
    if status.missing:
        missing_labels = ", ".join(d.replace("_", " ") for d in status.missing)
        lines.append(f"\nStill needed: {missing_labels}.")
    else:
        lines.append("\nAll required documents are on file.")
    return "Here's your document checklist:\n" + "\n".join(lines)


def _handle_document_status(employee_id: str | None, steps: list[AgentStep]) -> str:
    """Read-only checklist lookup (Component 14, CV_INTEGRATION.md §1.5) —
    deliberately NOT a ReAct tool/action: this is a deterministic status
    lookup, not something that needs the model's reasoning loop, so it
    short-circuits in run_turn() before _react_loop() the same way
    Intent.OUT_OF_SCOPE already does."""
    if not employee_id:
        return NO_EMPLOYEE_ID_REPLY

    from src.memory import onboarding_status

    status = onboarding_status.get_status(employee_id)
    steps.append(
        AgentStep(
            thought="looked up onboarding document checklist",
            tool="get_onboarding_status",
            tool_args={"employee_id": employee_id},
            observation=status.model_dump_json(),
        )
    )
    return _format_checklist_reply(status)


def run_turn(
    session_id: str,
    message: str,
    history: list[dict[str, str]] | None = None,
    client=None,
    employee_id: str | None = None,
) -> AgentResponse:
    history = history or []
    steps: list[AgentStep] = []
    turn_started_at = datetime.now(timezone.utc)

    logger.info("session=%s turn_start", session_id)
    guardrail_result = _check_input(message, client=client, session_id=session_id)
    if not guardrail_result.allowed:
        return AgentResponse(reply=guardrail_result.reason, steps=steps)

    import httpx
    from google.genai.errors import APIError

    from src.agent.llm_client import LLMBackendError

    client = client or config.get_llm_client()

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

        semantic_result = _check_semantic_guardrails(classification)
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

        if classification.intent == Intent.DOCUMENT_UPLOAD:
            return AgentResponse(
                reply=DOCUMENT_UPLOAD_REPLY,
                steps=steps,
                token_usage=_turn_token_usage(turn_started_at, session_id),
                actions=[_UNLOCK_DOCUMENT_FLOW_ACTION],
            )

        if classification.intent == Intent.DOCUMENT_STATUS:
            return AgentResponse(
                reply=_handle_document_status(employee_id, steps),
                steps=steps,
                token_usage=_turn_token_usage(turn_started_at, session_id),
                actions=[_UNLOCK_DOCUMENT_FLOW_ACTION],
            )

        reply, run_state = _react_loop(
            message, classification, steps, client, session_id, history
        )
    except (APIError, LLMBackendError, httpx.TimeoutException) as exc:
        # httpx.TimeoutException added 2026-08-10: a real request timeout
        # raises httpx.ConnectTimeout/ReadTimeout, not APIError, and doesn't
        # carry a .code attribute -- getattr avoids crashing this handler
        # on exactly the case it exists to handle gracefully.
        logger.warning("session=%s llm_api_error status=%s", session_id, getattr(exc, "code", "timeout"))
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


def _format_scratchpad(scratchpad: list[dict]) -> str:
    """Render the running thought/action/observation trace for the next planning call."""
    if not scratchpad:
        return "(nothing yet — this is the first step)"
    lines = []
    for i, s in enumerate(scratchpad, 1):
        lines.append(
            f"{i}. thought: {s['thought']}\n"
            f"   action: {s['action']}(query={s['query']!r})\n"
            f"   observation: {s['observation']}"
        )
    return "\n".join(lines)


def _plan_next_step(
    message: str,
    scratchpad: list[dict],
    client,
    session_id: str,
    history: list[dict[str, str]] | None = None,
) -> ReActStep:
    """One planning call: the model reads the scratchpad and picks the next action.

    `history` lets the planner resolve a terse follow-up (a bare "Part-time"
    answering a clarifying question) into a complete standalone search query,
    the same conversation view the router already gets."""
    from google.genai import types

    response = client.models.generate_content(
        model=config.ACTIVE_CHAT_MODEL,
        contents=(
            prompts.REACT_SYSTEM_PROMPT
            + "\n\n"
            + prompts.REACT_STEP_PROMPT.format(
                question=message,
                scratchpad=_format_scratchpad(scratchpad),
                history=format_history(history or []),
            )
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReActStep,
            temperature=0.0,
            thinking_config=config.thinking_config(),
        ),
    )
    usage.record_usage(
        config.ACTIVE_CHAT_MODEL, usage.extract_usage(response), session_id=session_id
    )
    step = response.parsed
    if step is None:  # unparseable plan → stop gathering and answer from what we have
        return ReActStep(thought="unparseable planning step; finishing", action=ReActAction.FINISH)
    return step


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Union of chunks across hops, first occurrence wins (a multi-hop question can
    retrieve the same chunk from more than one sub-query)."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
    return out


def _react_loop(
    message: str,
    classification: IntentClassification,
    steps: list[AgentStep],
    client,
    session_id: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, _RunState]:
    """Model-driven ReAct loop (Module 7). Each iteration the model emits a thought
    and one action (search_kb / search_web / finish). It plans retrieval only — it
    can decompose a multi-part question into several search_kb queries and reformulate
    a query that came back insufficient — and the final answer is synthesized afterward
    over the union of gathered chunks (see _synthesize_final), so the grounding
    guardrail still verifies every citation. Bounded by MAX_REACT_ITERATIONS.

    Retrieval uses the model's (possibly reformulated) query, not a history-rewrite;
    the router still sees history for clarification.

    Each search_kb hop only *retrieves* (tools.retrieve_kb) — no per-hop grounded-answer
    call. The loop reasons over the retrieved chunks themselves (count, top similarity,
    a preview), and the single grounded answer is shaped once at the end over the union
    of chunks (_synthesize_final). A multi-hop turn therefore pays one grounded-answer
    call, not one per hop."""
    run_state = _RunState()
    scratchpad: list[dict] = []
    all_chunks: list[RetrievedChunk] = []
    kb_hops = 0
    web_answer = None

    for _ in range(config.MAX_REACT_ITERATIONS):
        step = _plan_next_step(message, scratchpad, client, session_id, history)

        if step.action == ReActAction.FINISH:
            steps.append(AgentStep(thought=step.thought, tool=None, tool_args={}, observation="finish"))
            break

        if step.action == ReActAction.SEARCH_WEB:
            if not config.ENABLE_WEB_FALLBACK:
                obs = "search_web is disabled in this deployment"
            else:
                web = tools.search_web(step.query or message, client=client, session_id=session_id)
                if not web.insufficient_context:
                    web_answer = web
                obs = json.dumps({
                    "source": web.source.value,
                    "insufficient_context": web.insufficient_context,
                    "answer_preview": web.answer[:200],
                })
            steps.append(AgentStep(thought=step.thought, tool="search_web",
                                   tool_args={"question": step.query}, observation=obs))
            scratchpad.append({"thought": step.thought, "action": "search_web",
                               "query": step.query, "observation": obs})
            continue

        # default: search_kb — retrieve only, no per-hop grounded-answer call.
        cat = step.category or classification.category
        query = step.query or message
        chunks, clarification = tools.retrieve_kb(query, category=cat)

        # Audience-segment split the retrieval couldn't resolve → ask, don't answer.
        if clarification is not None:
            run_state.chunks = []
            steps.append(AgentStep(thought=step.thought, tool="search_kb",
                                   tool_args={"question": query, "category": cat},
                                   observation="requires_clarification"))
            return clarification, run_state

        kb_hops += 1
        all_chunks.extend(chunks)
        top = chunks[0] if chunks else None
        obs = json.dumps({
            "chunks_found": len(chunks),
            "insufficient": len(chunks) == 0,
            "top_similarity": round(getattr(top, "similarity", 0.0), 3) if top else 0.0,
            "top_preview": getattr(top, "text", "")[:200] if top else "",
        })
        steps.append(AgentStep(thought=step.thought, tool="search_kb",
                               tool_args={"question": query, "category": cat}, observation=obs))
        scratchpad.append({"thought": step.thought, "action": "search_kb",
                           "query": query, "observation": obs})

    return _synthesize_final(
        message, all_chunks, kb_hops, web_answer, run_state, client, session_id, history
    )


def _synthesize_final(
    message: str,
    all_chunks: list[RetrievedChunk],
    kb_hops: int,
    web_answer,
    run_state: _RunState,
    client,
    session_id: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, _RunState]:
    """Turn the gathered evidence into one grounded reply once the loop finishes.

    Exactly one grounded-answer call, over the deduped union of every hop's chunks,
    so a multi-part answer is grounded in all of them. A web answer is used only when
    the KB produced nothing usable; otherwise we decline."""
    from src.rag.answerer import generate_grounded_answer, no_answer

    # The model only reaches for search_web when it judged the KB too weak to answer.
    # So a sufficient web answer is the model's decision that web — not the KB's
    # related-but-off text — is what actually answers this question; prefer it.
    if web_answer is not None and not web_answer.insufficient_context:
        run_state.citations = web_answer.citations
        run_state.web_citations = web_answer.web_citations
        run_state.chunks = []  # a web answer is grounded in web_citations, not KB chunks
        run_state.insufficient_context = False
        return web_answer.answer, run_state

    # Shape the one grounded answer over the union of chunks gathered across hops.
    chunks = _dedupe_chunks(all_chunks)
    if chunks:
        final = generate_grounded_answer(
            message, chunks, client=client, session_id=session_id, history=history
        )
        if not final.insufficient_context:
            run_state.citations = final.citations
            run_state.chunks = chunks
            run_state.insufficient_context = False
            return final.answer, run_state
        # Gathered evidence but it doesn't answer the question (and no usable web) →
        # return the model's grounded decline, withholding the rejected chunks (mirrors
        # answer_question, so the UI never shows "maybe relevant" excerpts beside "I don't know").
        run_state.citations = final.citations
        run_state.chunks = []
        run_state.insufficient_context = True
        return final.answer, run_state

    # Nothing retrieved across any hop (kb_hops may be >0 with all-empty retrievals).
    na = no_answer()
    run_state.insufficient_context = True
    return na.answer, run_state


def _source_dict_from_chunk(chunk: RetrievedChunk) -> dict:
    preview = " ".join(chunk.text.split())
    if len(preview) > 360:
        preview = preview[:357].rstrip() + "..."
    return {
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "section_path": chunk.section_path,
        "page": chunk.page_start,
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

    result = run_turn(session_id, message, history=history, client=client, employee_id=employee_id)

    if manage_memory:
        memory_session.append_turn(session_id, "user", message)
        memory_session.append_turn(session_id, "assistant", result.reply)
        memory_persistent.maybe_update_summary(session_id, employee_id, client=client)

    return {
        "session_id": session_id,
        "reply": result.reply,
        "citations": result.citations,
        "sources": [_source_dict_from_chunk(chunk) for chunk in result.chunks],
        "web_citations": result.web_citations,
        "actions": result.actions,
        "insufficient_context": result.insufficient_context,
        "token_usage": result.token_usage,
    }
