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

Guardrails (Module 6, src/guardrails/) run at multiple points in run_turn():
input checks (prompt-injection heuristics, toxicity) gate entry before the
router even runs; an output grounding check re-verifies citations before the
final AgentResponse is returned; and escalation for filed complaints is
decided by the deterministic rule engine in src/guardrails/escalation.py
(should_escalate) and src/guardrails/danger_scan.py -- never by the LLM. If
the loop exhausts MAX_REACT_ITERATIONS while the turn was already classified
as a complaint, _failsafe_escalate_incomplete_complaint() forces a Rule 7
escalation rather than letting the conversation end in silence.

Consent-gate / form-card intake (PLAN.md Sec 6.1, Steps A-B): the moment a
fresh Intent.COMPLAINT is detected (and the message isn't a ticket-status
lookup), run_turn() does not enter the ReAct loop at all -- it asks the
employee to choose (a) fill out a form or (b) escalate directly, tracked via
src/guardrails/escalation_state.py. Every turn spent in that state machine is
handled by _handle_escalation_flow_turn() below, which is 100% deterministic
parsing, never an LLM call, so the ReAct loop's own file_complaint tool call
is now only reached via the ticket-status-lookup bypass.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import ValidationError

from src import config
from src.agent import prompts, tools, usage
from src.agent.router import classify_intent, needs_clarification
from src.guardrails import escalation_state
from src.guardrails.escalation import fail_safe_decision, should_escalate
from src.guardrails.form_pii import to_escalation_event
from src.guardrails.grounding import check_grounding
from src.guardrails.input_checks import check_topic_and_injection
from src.guardrails.llm_judge import check_input_llm
from src.guardrails.toxicity import check_toxicity
from src.rag.retriever import RetrievedChunk
from src.schemas import (
    Citation,
    ComplaintCategory,
    ComplaintTicket,
    EscalationDecision,
    EscalationFlowState,
    EscalationFormSubmission,
    GuardrailResult,
    Intent,
    IntentClassification,
    Severity,
    TokenUsage,
    TriggerRule,
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

# --- Consent-gate / form-card intake (PLAN.md Sec 6.1, Steps A-B) -----------
# All deterministic, text-parsing only -- no LLM call decides any of this.

CONSENT_GATE_REPLY = (
    "It sounds like you'd like to raise a concern. Would you like to (a) fill out "
    "a complaint form with the details, or (b) skip the form and have this escalated "
    "to HR directly right now? Just reply with a or b."
)
FORM_INSTRUCTIONS_REPLY = (
    "Okay -- please fill out the complaint form that just appeared below with as "
    "much detail as you're comfortable sharing. You can also type 'cancel' at any "
    "point to back out."
)
CONSENT_CANCELLED_REPLY = (
    "No problem -- I won't file anything. Let me know if you'd like to raise this again."
)
FORM_CANCELLED_REPLY = "No problem -- I've cancelled that. Let me know if you'd like to start again."
FORM_REMINDER_REPLY = (
    "I'm still waiting on the complaint form above -- please fill that in with the "
    "details, or type 'cancel' if you'd rather not continue."
)

_TICKET_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_ESCALATE_CONSENT_PATTERN = re.compile(
    r"^\s*\(?b\)?\b|\boption b\b|\bescalate\b|\bstraight to hr\b|"
    r"\bskip (?:the )?form\b|\bjust (?:escalate|get me to hr)\b",
    re.IGNORECASE,
)
_FILE_CONSENT_PATTERN = re.compile(
    r"^\s*\(?a\)?\b|\boption a\b|\bfile (?:a |the )?(?:ticket|complaint)\b|"
    r"\bfill (?:out|in) the form\b|\bdescribe it\b|\buse the form\b",
    re.IGNORECASE,
)
_CANCEL_PATTERN = re.compile(r"\bcancel\b|\bnever ?mind\b|\bforget it\b", re.IGNORECASE)

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
    trigger_rule: str | None = None
    form_required: bool = False
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
    trigger_rule: str | None = None
    form_required: bool = False
    insufficient_context: bool = False


def _check_input(message: str, client=None, session_id: str | None = None) -> GuardrailResult:
    """Layered input guardrails. First the deterministic, no-LLM-call checks —
    prompt-injection heuristics and a small toxicity wordlist — which
    short-circuit blatant cases for free (Gemini quota is the #1 constraint,
    PLAN.md §2.1/§8). Anything that gets past those goes to the LLM-as-judge
    (src/guardrails/llm_judge.py), one structured call classifying toxicity /
    PII / injection / off-topic / jailbreak to catch the nuanced phrasing the
    wordlist/regex layer misses. The judge fails open, so a failed call never
    breaks the turn — the deterministic layer above is the backstop."""
    result = check_topic_and_injection(message)
    if not result.allowed:
        return result
    result = check_toxicity(message)
    if not result.allowed:
        return result
    return check_input_llm(message, client=client, session_id=session_id)


def run_turn(
    session_id: str,
    message: str,
    history: list[dict[str, str]] | None = None,
    client=None,
    escalation_form: EscalationFormSubmission | None = None,
) -> AgentResponse:
    history = history or []
    steps: list[AgentStep] = []
    turn_started_at = datetime.now(timezone.utc)

    logger.info("session=%s turn_start", session_id)
    guardrail_result = _check_input(message, client=client, session_id=session_id)
    if not guardrail_result.allowed:
        return AgentResponse(reply=guardrail_result.reason, steps=steps)

    flow_state = escalation_state.get_state(session_id)
    if flow_state != EscalationFlowState.NORMAL:
        # Deliberately checked before any LLM client is constructed: every
        # branch of the consent-gate/form flow is deterministic text parsing
        # (PLAN.md Sec 6.1, Steps A-B) and must never require a configured
        # LLM provider to function.
        flow_run_state = _RunState()
        reply = _handle_escalation_flow_turn(
            message, history, flow_state, session_id, flow_run_state, escalation_form
        )
        return AgentResponse(
            reply=reply,
            ticket_id=flow_run_state.ticket_id,
            escalated=flow_run_state.escalated,
            trigger_rule=flow_run_state.trigger_rule,
            form_required=flow_run_state.form_required,
            steps=steps,
            token_usage=_turn_token_usage(turn_started_at, session_id),
        )

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

        if classification.intent == Intent.COMPLAINT and not _TICKET_ID_PATTERN.search(message):
            # Fresh complaint, not a status lookup on an existing ticket --
            # gate through consent (Step A) instead of letting the ReAct loop
            # file anything. A message that references an existing ticket id
            # falls through to the loop below unchanged, so get_ticket_status
            # keeps working exactly as before.
            escalation_state.set_state(session_id, EscalationFlowState.AWAITING_CONSENT)
            return AgentResponse(
                reply=CONSENT_GATE_REPLY,
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
        trigger_rule=run_state.trigger_rule,
        form_required=run_state.form_required,
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
    # Guardrail scanning (danger_scan/should_escalate) needs the employee's
    # full, literal wording across the whole conversation, not just this
    # turn's fragment -- a multi-turn complaint often states the substantive
    # (and potentially dangerous/retaliatory) content in an earlier turn and
    # only dates/parties/desired-outcome in the turn that actually completes
    # the ComplaintTicket. Using only `message` here would silently miss
    # anything said before the final turn.
    guardrail_raw_text = _combine_user_text(message, history)

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
            observation = _execute_tool(call.name, args, client, run_state, session_id, guardrail_raw_text)
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

    if classification.intent == Intent.COMPLAINT and run_state.ticket_id is None:
        reply = _failsafe_escalate_incomplete_complaint(guardrail_raw_text, run_state, session_id)
        return reply, run_state

    return MAX_ITERATIONS_REPLY, run_state


def _combine_user_text(message: str, history: list[dict[str, str]]) -> str:
    """Concatenate every user-authored turn (prior history + this turn) for
    guardrail scanning. See the comment in _run_tool_loop for why this needs
    to span the whole conversation rather than just the current message."""
    prior_user_turns = [turn["content"] for turn in history if turn.get("role") == "user"]
    return "\n".join([*prior_user_turns, message])


def _file_and_escalate_best_effort(
    description_prefix: str,
    guardrail_raw_text: str,
    run_state: "_RunState",
    rationale: str,
) -> str:
    """Shared by every fail-safe path (Rule 7's iteration-cap case and the
    consent-gate's abandoned-form case): build a minimal ComplaintTicket from
    raw conversation text and force an escalation via
    guardrails.escalation.fail_safe_decision(), so no path can silently drop
    an employee's report. Returns the new ticket_id."""
    description = f"{description_prefix} {guardrail_raw_text}"
    ticket = ComplaintTicket(
        category=ComplaintCategory.OTHER,
        severity=Severity.HIGH,
        description=description[:2000],
    )
    ticket_id = tools.file_complaint(ticket)
    decision = fail_safe_decision(rationale)
    event = to_escalation_event(ticket, ticket_id, decision)
    tools.escalate_to_hr(event, ticket)

    run_state.ticket_id = ticket_id
    run_state.escalated = True
    run_state.trigger_rule = decision.trigger_rule.value if decision.trigger_rule else None
    return ticket_id


def _failsafe_escalate_incomplete_complaint(
    message: str, run_state: "_RunState", session_id: str
) -> str:
    """Rule 7 fail-safe (src/guardrails/escalation.py:fail_safe_decision).

    The router already classified this turn as a complaint, but the ReAct
    loop exhausted MAX_REACT_ITERATIONS without ever completing a valid
    file_complaint call -- e.g. the model kept re-asking for fields, or kept
    calling the wrong tool. Rather than return the generic MAX_ITERATIONS_REPLY
    (which would silently drop a possibly-sensitive report), file a best-effort
    ticket from the raw message and escalate it deterministically.
    """
    ticket_id = _file_and_escalate_best_effort(
        f"[Auto-filed after {config.MAX_REACT_ITERATIONS} ReAct iterations without a "
        "completed file_complaint call]",
        message,
        run_state,
        f"max_react_iterations_reached session={session_id} original_intent=complaint",
    )
    logger.warning("session=%s failsafe_escalation ticket=%s", session_id, ticket_id)
    return (
        "I wasn't able to finish gathering the details in the usual number of steps, "
        "so rather than risk losing your report I've escalated it to HR directly. "
        f"Reference ticket: {ticket_id}."
    )


def _parse_consent_reply(text: str) -> str | None:
    """Deterministic, no-LLM-call parse of the employee's (a)/(b) reply to
    CONSENT_GATE_REPLY. Returns "file_ticket", "escalate_now", or None if
    unparseable -- callers treat None the same as "escalate_now" (fail
    toward escalation, the same principle behind the Rule 7 fail-safe)."""
    if _ESCALATE_CONSENT_PATTERN.search(text):
        return "escalate_now"
    if _FILE_CONSENT_PATTERN.search(text):
        return "file_ticket"
    return None


def _is_cancel(text: str) -> bool:
    return bool(_CANCEL_PATTERN.search(text))


def _escalate_directly_from_consent(
    guardrail_raw_text: str, run_state: "_RunState", session_id: str
) -> str:
    """The employee chose (b) at the consent gate: skip the form, file a
    minimal ticket from the conversation so far, and escalate immediately.

    Still runs the real rule engine first -- a mandatory category or danger
    signal still determines the actual trigger_rule/severity -- and only
    overrides to TriggerRule.EMPLOYEE_REQUESTED if the automated rules alone
    wouldn't have escalated, since the employee's own request to reach HR
    directly is a legitimate trigger in its own right.
    """
    description = f"[Escalated directly at the employee's request via the consent gate] {guardrail_raw_text}"
    ticket = ComplaintTicket(
        category=ComplaintCategory.OTHER,
        # MEDIUM, not HIGH: a hardcoded HIGH would always satisfy Rule 2
        # (severity_escalation) on its own, making the EMPLOYEE_REQUESTED
        # override below unreachable -- MEDIUM lets danger_scan/retaliation
        # signals in the conversation still surface their real trigger_rule
        # first, and only falls through to EMPLOYEE_REQUESTED when nothing
        # else in the rule matrix actually applies.
        severity=Severity.MEDIUM,
        description=description[:2000],
    )
    ticket_id = tools.file_complaint(ticket)
    decision = should_escalate(ticket, raw_text=guardrail_raw_text)
    if not decision.should_escalate:
        decision = EscalationDecision(
            should_escalate=True,
            trigger_rule=TriggerRule.EMPLOYEE_REQUESTED,
            # HIGH regardless of the MEDIUM used for the rule-check above:
            # explicitly asking to skip the form and reach HR directly is
            # itself a signal this deserves prompt attention.
            effective_severity=Severity.HIGH,
            danger_flag=False,
            rationale="employee explicitly chose direct escalation via the consent gate",
        )
    event = to_escalation_event(ticket, ticket_id, decision)
    tools.escalate_to_hr(event, ticket)

    run_state.ticket_id = ticket_id
    run_state.escalated = True
    run_state.trigger_rule = decision.trigger_rule.value if decision.trigger_rule else None
    logger.warning("session=%s consent_gate_direct_escalation ticket=%s", session_id, ticket_id)
    return f"Understood -- I've escalated this directly to HR without further questions. Reference ticket: {ticket_id}."


def _escalate_from_abandoned_form(
    guardrail_raw_text: str, run_state: "_RunState", session_id: str
) -> str:
    """The employee typed in chat instead of submitting the rendered form
    while AWAITING_FORM. Reuses the same fail-safe machinery as Rule 7 so
    this path can't lose the report either -- the only difference from the
    iteration-cap case is the reply text and the log tag."""
    ticket_id = _file_and_escalate_best_effort(
        "[Auto-filed from chat text instead of the intake form]",
        guardrail_raw_text,
        run_state,
        f"employee_typed_instead_of_form session={session_id}",
    )
    logger.warning("session=%s form_abandoned_failsafe ticket=%s", session_id, ticket_id)
    return (
        "No problem -- I'll use what you've told me instead of the form. "
        f"I've filed and escalated this to HR directly. Reference ticket: {ticket_id}."
    )


def _file_from_form_submission(
    submission: EscalationFormSubmission, run_state: "_RunState", session_id: str
) -> str:
    """The employee actually submitted the rendered intake form (PLAN.md Sec
    6.1, Step B). Unlike the fail-safe paths, this builds a real,
    properly-categorized ComplaintTicket from their own field choices and
    runs it through the ordinary should_escalate() pipeline -- so a routine,
    low-severity complaint filed through the form is NOT force-escalated,
    exactly matching what the conversational file_complaint path already
    does today.
    """
    ticket = ComplaintTicket(
        category=submission.category,
        severity=submission.severity,
        description=submission.description,
        parties_involved=submission.parties_involved,
        incident_date=submission.incident_date,
        desired_outcome=submission.desired_outcome,
    )
    ticket_id = tools.file_complaint(ticket)
    decision = should_escalate(ticket, raw_text=submission.description)
    run_state.ticket_id = ticket_id
    run_state.escalated = decision.should_escalate
    run_state.trigger_rule = decision.trigger_rule.value if decision.trigger_rule else None
    if decision.should_escalate:
        event = to_escalation_event(ticket, ticket_id, decision)
        tools.escalate_to_hr(event, ticket)
        logger.warning("session=%s form_submitted_escalated ticket=%s", session_id, ticket_id)
        return (
            "Thanks -- I've filed your complaint and, given what you described, HR has "
            f"already been notified directly. Reference ticket: {ticket_id}."
        )
    logger.info("session=%s form_submitted_filed ticket=%s", session_id, ticket_id)
    return f"Thanks -- I've filed your complaint. Reference ticket: {ticket_id}."


def _handle_escalation_flow_turn(
    message: str,
    history: list[dict[str, str]],
    flow_state: EscalationFlowState,
    session_id: str,
    run_state: "_RunState",
    escalation_form: EscalationFormSubmission | None = None,
) -> str:
    """Handles a turn while the session is mid consent-gate or mid
    intake-form (PLAN.md Sec 6.1, Steps A-B). Never reaches the ReAct loop or
    the router -- every branch here is deterministic, no-LLM-call parsing,
    consistent with the "escalation control flow is never an LLM decision"
    principle already established for should_escalate()/danger_scan()."""
    guardrail_raw_text = _combine_user_text(message, history)

    if flow_state == EscalationFlowState.AWAITING_CONSENT:
        if _is_cancel(message):
            escalation_state.clear_state(session_id)
            return CONSENT_CANCELLED_REPLY

        choice = _parse_consent_reply(message)
        if choice == "file_ticket":
            escalation_state.set_state(session_id, EscalationFlowState.AWAITING_FORM)
            run_state.form_required = True
            return FORM_INSTRUCTIONS_REPLY

        # choice == "escalate_now" or unparseable -> fail toward escalation
        escalation_state.clear_state(session_id)
        return _escalate_directly_from_consent(guardrail_raw_text, run_state, session_id)

    if flow_state == EscalationFlowState.AWAITING_FORM:
        if _is_cancel(message):
            escalation_state.clear_state(session_id)
            return FORM_CANCELLED_REPLY

        if escalation_form is not None:
            # A real submission from the rendered form: file it as a proper,
            # correctly-categorized ComplaintTicket and run it through the
            # ordinary should_escalate() pipeline -- same rule engine as the
            # conversational path, so a low-severity, non-mandatory-category
            # submission can be filed WITHOUT escalating, exactly like today.
            escalation_state.clear_state(session_id)
            return _file_from_form_submission(escalation_form, run_state, session_id)

        # Typed in chat instead of using the rendered form. Give one
        # reminder rather than force-escalating on the very first stray
        # message (someone might just be asking "is the form still there?")
        # -- but never lose the report either, so a second miss fails safe.
        miss_count = escalation_state.record_form_miss(session_id)
        if miss_count >= 2:
            escalation_state.clear_state(session_id)
            return _escalate_from_abandoned_form(guardrail_raw_text, run_state, session_id)

        run_state.form_required = True
        return FORM_REMINDER_REPLY

    return FALLBACK_CLARIFYING_TEXT  # defensive default; unreachable given the two states above


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
    name: str, args: dict, client, run_state: _RunState, session_id: str, message: str
) -> dict:
    """`message` here is the combined multi-turn guardrail text built by
    _combine_user_text() in _run_tool_loop, not just the current turn --
    see should_escalate() below, which needs the employee's full wording
    across the conversation, not a single fragment of it."""
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
        decision = should_escalate(ticket, raw_text=message)
        run_state.ticket_id = ticket_id
        run_state.escalated = decision.should_escalate
        run_state.trigger_rule = decision.trigger_rule.value if decision.trigger_rule else None
        if decision.should_escalate:
            event = to_escalation_event(ticket, ticket_id, decision)
            tools.escalate_to_hr(event, ticket)
        return {
            "ticket_id": ticket_id,
            "escalated": decision.should_escalate,
            "danger_flag": decision.danger_flag,
        }

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
    escalation_form: EscalationFormSubmission | None = None,
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

    escalation_form is passed straight through to run_turn() -- it's only
    ever meaningful when the session is already in the AWAITING_FORM flow
    state (PLAN.md Sec 6.1, Step B); run_turn() ignores it otherwise.
    """
    from src.memory import persistent as memory_persistent
    from src.memory import session as memory_session

    session_id = session_id or str(uuid.uuid4())
    manage_memory = history is None
    if manage_memory:
        history = memory_persistent.get_context(session_id, employee_id)

    result = run_turn(
        session_id, message, history=history, client=client, escalation_form=escalation_form
    )

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
                "escalated": result.escalated,
                "trigger_rule": result.trigger_rule,
            }
        )
    if result.form_required:
        actions.append(
            {
                "type": "escalation_form_required",
                "label": "Please fill out the complaint intake form.",
                "status": "pending",
                "ticket_id": None,
                "escalated": False,
                "trigger_rule": None,
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
