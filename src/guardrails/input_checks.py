"""Input topic/injection guardrail (Module 6: Guardrails).

Topic restriction's authoritative enforcement is already the intent router
(src/agent/router.classify_intent): Intent.OUT_OF_SCOPE is a hard block in
orchestrator.run_turn() — it returns a decline and never enters the tool
loop. Re-implementing topic classification here with keyword matching would
be redundant with that and worse at nuance (an LLM router handles "what
counts as HR-related" far better than a wordlist).

What this module adds instead: a cheap, deterministic, pre-router check for
prompt-injection attempts (check_topic_and_injection). This is a well-defined
pattern-matching problem (unlike open-ended topic classification) and
specifically targets attempts to manipulate the router/model itself, so it
runs before that LLM call, not after — catching the known-pattern cases
without spending a request on them.

check_injection_semantic() is the backstop for phrasings the regex patterns
don't cover — it reads IntentClassification.is_injection_attempt, a field the
router's already-mandatory LLM call fills in, so it costs nothing beyond that
call. Unlike toxicity, injection detection has no "legitimate quoting" false-
positive risk (nobody has a real reason to say "ignore all previous
instructions" as part of an HR complaint), so the combined check is a
straightforward OR of both layers, no complaint-intent carve-out needed.
"""

import re

from src.schemas import GuardrailResult, IntentClassification

DECLINE_MESSAGE = (
    "I can't follow instructions that try to change how I operate. "
    "I can help with company policy questions, DOLE labor law questions, or filing a complaint."
)

_INJECTION_PATTERNS = (
    re.compile(r"ignore (all |any )?(previous|prior|above|earlier) instructions", re.IGNORECASE),
    re.compile(r"disregard (your |the )?(system )?prompt", re.IGNORECASE),
    re.compile(r"\byou are now\b", re.IGNORECASE),
    re.compile(r"new instructions\s*:", re.IGNORECASE),
    re.compile(r"(reveal|print|show|output) (your |the )?(system )?prompt", re.IGNORECASE),
)


def check_topic_and_injection(message: str) -> GuardrailResult:
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(message):
            return GuardrailResult(allowed=False, reason=DECLINE_MESSAGE)
    return GuardrailResult(allowed=True)


def check_injection_semantic(classification: IntentClassification) -> GuardrailResult:
    if classification.is_injection_attempt:
        return GuardrailResult(allowed=False, reason=DECLINE_MESSAGE)
    return GuardrailResult(allowed=True)
