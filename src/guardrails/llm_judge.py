"""LLM-as-Judge input guardrail (Module 6: Guardrails).

A second, nuance-aware layer behind the deterministic input guardrails
(src/guardrails/input_checks.py prompt-injection heuristics, toxicity.py
wordlist). One structured Gemini call classifies each incoming employee message
across five safety dimensions at once — toxicity, PII, prompt-injection,
off-topic, and jailbreak — catching adversarial phrasing the fixed
wordlist/regex layer can't (paraphrased injections, subtle abuse, novel
jailbreak framings).

Design notes:
- Detection vs. policy are kept separate. The model only *detects* (returns an
  LLMJudgeVerdict); the allow/block decision is deterministic code in
  to_guardrail_result() driven by config.LLM_JUDGE_BLOCKING_VIOLATIONS and
  config.LLM_JUDGE_CONFIDENCE_FLOOR — never the model's own judgment. This
  mirrors the rest of src/guardrails/ (escalation, grounding), where the LLM
  never gets to decide the gate itself.
- PII is detected but not blocking: employees legitimately share contact details
  in FAQ/complaint flows (see pii.py). It's surfaced for redaction/observability.
- Fail-open: if the judge call errors or returns nothing parseable, the message
  is allowed through. The deterministic layer already ran first and is the
  backstop, and breaking every turn because one extra LLM call failed is worse
  UX than leaning on that backstop. Errors are logged, not raised.
- Runs on both LLM backends: uses response_schema exactly like router.py, so the
  OllamaClient adapter (which fills response.parsed) works unchanged.
"""

import logging

from src import config
from src.agent import prompts, usage
from src.guardrails.input_checks import DECLINE_MESSAGE as INJECTION_DECLINE_MESSAGE
from src.guardrails.toxicity import DECLINE_MESSAGE as TOXICITY_DECLINE_MESSAGE
from src.schemas import GuardrailResult, LLMJudgeVerdict

logger = logging.getLogger(__name__)

OFF_TOPIC_DECLINE_MESSAGE = (
    "I can only help with company policy, DOLE labor law questions, and filing a "
    "complaint. For anything else, please reach out to the right team directly."
)

# User-facing decline copy per blocking violation. Jailbreak reuses the
# injection message — both are "trying to change how I operate" from the
# employee's point of view. Order here also sets which reason wins when a
# message trips several blocking violations at once (most manipulative first).
_DECLINE_MESSAGES = {
    "jailbreak": INJECTION_DECLINE_MESSAGE,
    "injection": INJECTION_DECLINE_MESSAGE,
    "toxicity": TOXICITY_DECLINE_MESSAGE,
    "off_topic": OFF_TOPIC_DECLINE_MESSAGE,
}


def judge_input(
    message: str, client=None, session_id: str | None = None
) -> LLMJudgeVerdict | None:
    """One structured Gemini call classifying `message` across the five
    guardrail dimensions. Returns None (fail-open — caller should allow) on any
    API/backend error or unparseable response; never raises."""
    from google.genai import types
    from google.genai.errors import APIError

    from src.agent.llm_client import LLMBackendError

    client = client or config.get_llm_client()
    try:
        response = client.models.generate_content(
            model=config.ACTIVE_CHAT_MODEL,
            contents=prompts.LLM_JUDGE_PROMPT.format(message=message),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LLMJudgeVerdict,
                temperature=0.0,
            ),
        )
    except (APIError, LLMBackendError) as exc:
        logger.warning("session=%s llm_judge_error status=%s", session_id, getattr(exc, "code", "?"))
        return None

    usage.record_usage(
        config.ACTIVE_CHAT_MODEL, usage.extract_usage(response), session_id=session_id
    )
    verdict: LLMJudgeVerdict | None = response.parsed
    return verdict


def to_guardrail_result(verdict: LLMJudgeVerdict | None) -> GuardrailResult:
    """Deterministic allow/block policy over a judge verdict. A None verdict
    (judge failed) fails open. A message is blocked only when a *blocking*
    violation (config.LLM_JUDGE_BLOCKING_VIOLATIONS) is set AND the judge's
    confidence clears config.LLM_JUDGE_CONFIDENCE_FLOOR — so low-confidence
    flags don't reject legitimate questions. PII alone never blocks."""
    if verdict is None:
        return GuardrailResult(allowed=True)
    if verdict.confidence < config.LLM_JUDGE_CONFIDENCE_FLOOR:
        return GuardrailResult(allowed=True)

    for violation in _DECLINE_MESSAGES:  # insertion order = priority
        if violation in config.LLM_JUDGE_BLOCKING_VIOLATIONS and getattr(verdict, violation):
            return GuardrailResult(allowed=False, reason=_DECLINE_MESSAGES[violation])
    return GuardrailResult(allowed=True)


def check_input_llm(message: str, client=None, session_id: str | None = None) -> GuardrailResult:
    """Convenience wrapper: judge the message and apply the block policy. Fails
    open (allowed=True) end-to-end if the judge call fails."""
    return to_guardrail_result(judge_input(message, client=client, session_id=session_id))
