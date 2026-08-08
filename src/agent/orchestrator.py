"""ReAct agent loop (Module 7: ReAct Agent).

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
gets declined, neither reaches a tool. Everything else enters the ReAct loop:
Gemini picks a tool, tools.py executes it, the result is fed back as an
observation, and the model repeats until it answers in plain text or
MAX_REACT_ITERATIONS is hit.

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
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src import config
from src.agent import prompts, tools, usage
from src.agent.router import classify_intent, needs_clarification
from src.guardrails.grounding import check_grounding
from src.guardrails.input_checks import check_injection_semantic, check_topic_and_injection
from src.guardrails.llm_judge import check_input_llm
from src.guardrails.toxicity import check_toxicity, check_toxicity_semantic
from src.rag.retriever import RetrievedChunk
from src.schemas import (
    Citation,
    GuardrailResult,
    Intent,
    IntentClassification,
    TokenUsage,
    WebCitation,
)

OUT_OF_SCOPE_REPLY = (
    "I can only help with faculty onboarding, pre-employment requirements, and DLSU "
    "Faculty Manual questions. For anything else, please reach out to the right office "
    "directly."
)
FALLBACK_CLARIFYING_TEXT = "Could you clarify what you need help with?"
MAX_ITERATIONS_REPLY = (
    "I wasn't able to finish handling this in the usual number of steps. "
    "Please try rephrasing, or contact your college's HR office directly if this is urgent."
)
API_ERROR_REPLY = (
    "I'm having trouble reaching the assistant service right now. Please try again "
    "in a moment, or contact your college's HR office directly if this is urgent."
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
    """Pre-router input guardrails, before intent is known. First the
    deterministic, no-LLM-call checks — prompt-injection regex and a plain
    toxicity wordlist — which short-circuit blatant cases for free (Gemini
    quota is the #1 constraint, PLAN.md §2.1/§8). Anything that gets past
    those goes to the LLM-as-judge (src/guardrails/llm_judge.py), one
    structured call classifying toxicity/PII/injection/off-topic/jailbreak.
    The judge fails open, so a failed call never breaks the turn."""
    result = check_topic_and_injection(message)
    if not result.allowed:
        return result
    result = check_toxicity(message)
    if not result.allowed:
        return result
    return check_input_llm(message, client=client, session_id=session_id)


def _check_semantic_guardrails(classification: IntentClassification) -> GuardrailResult:
    """Post-router guardrail backstop, run immediately after classify_intent()
    returns — free, since that LLM call already happened and its response
    schema already carries is_toxic/is_injection_attempt (ROUTER_PROMPT asks
    for both). Catches paraphrased abuse/injection the pre-router
    deterministic layer misses."""
    result = check_injection_semantic(classification)
    if not result.allowed:
        return result
    return check_toxicity_semantic(classification)


def run_turn(
    session_id: str,
    message: str,
    history: list[dict[str, str]] | None = None,
    client=None,
) -> AgentResponse:
    history = history or []
    steps: list[AgentStep] = []
    turn_started_at = datetime.now(timezone.utc)

    logger.info("session=%s turn_start", session_id)
    guardrail_result = _check_input(message, client=client, session_id=session_id)
    if not guardrail_result.allowed:
        return AgentResponse(reply=guardrail_result.reason, steps=steps)

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

        # A faculty-class split the KB tool couldn't resolve is a stop condition:
        # return the clarifying question directly rather than letting the model
        # narrate around it or fall through to search_web.
        if run_state.clarification:
            return run_state.clarification, run_state

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
            description="Search the faculty onboarding & Faculty Manual knowledge base "
            "(DLSU Faculty Manual 2021 plus official onboarding companion documents: "
            "pre-employment requirements, hiring, academic & grading obligations, "
            "dress code, leaves).",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "question": types.Schema(
                        type="STRING", description="The faculty member's question"
                    ),
                    "category": types.Schema(
                        type="STRING",
                        enum=list(config.QUERY_CATEGORIES),
                        description="Topic category hint, if clearly inferable",
                    ),
                },
                required=["question"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_web",
            description="Search official Philippine government sources for national "
            "statutory pre-employment requirements (e.g. NBI, SSS, PhilHealth, Pag-IBIG, "
            "BIR) not covered by the internal knowledge base.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "question": types.Schema(
                        type="STRING", description="The statutory pre-employment question"
                    )
                },
                required=["question"],
            ),
        ),
    ]


def _execute_tool(name: str, args: dict, client, run_state: _RunState, session_id: str) -> dict:
    if name == "search_kb":
        answer, chunks = tools.search_kb(args["question"], category=args.get("category"))
        run_state.citations = answer.citations
        run_state.web_citations = answer.web_citations
        run_state.chunks = chunks
        run_state.insufficient_context = answer.insufficient_context
        if answer.requires_clarification:
            run_state.clarification = answer.clarifying_question or answer.answer
        return {
            "answer": answer.answer,
            "insufficient_context": answer.insufficient_context,
            "requires_clarification": answer.requires_clarification,
            "chunks_found": len(chunks),
        }

    if name == "search_web":
        answer = tools.search_web(args["question"], client=client, session_id=session_id)
        run_state.citations = answer.citations
        run_state.web_citations = answer.web_citations
        run_state.insufficient_context = answer.insufficient_context
        return {"answer": answer.answer, "insufficient_context": answer.insufficient_context}

    return {"error": f"unknown tool: {name}"}


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

    result = run_turn(session_id, message, history=history, client=client)

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
        "actions": [],
        "insufficient_context": result.insufficient_context,
        "token_usage": result.token_usage,
    }
