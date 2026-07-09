"""Toxicity/abuse filter (Module 6: Guardrails).

Two layers:
1. check_toxicity() — a keyword heuristic, not an ML classifier. Free (no LLM
   call), catches known wordlist hits before the router even runs. This is an
   internal HR tool used by authenticated employees, not a public-facing
   surface, so the bar is blatant abuse directed at the bot/HR staff, not
   comprehensive content moderation (PLAN.md §8 flags scope creep as a risk;
   config.TOXIC_WORDLIST is intentionally small and documented, not exhaustive).
2. check_toxicity_semantic() — reads IntentClassification.is_toxic, a field
   the router's already-mandatory LLM call fills in. Catches paraphrased or
   creative-spelling abuse the wordlist misses, at no incremental LLM cost
   (the router call happens regardless).

check_toxicity_with_context() combines both, but NOT as a simple OR — for
COMPLAINT intent, the wordlist layer is skipped entirely and only the semantic
signal is trusted. This matters: a harassment complaint that quotes what was
said to the employee ("my coworker called me a bitch") legitimately contains
wordlist hits without being abusive language from the employee. Auto-blocking
on the wordlist alone would prevent exactly the complaints this system exists
to let through. The router prompt (src/agent/prompts.py) is explicitly
instructed to set is_toxic=false for quoted/reported abuse and only true for
the employee's own hostile language — see ROUTER_PROMPT.
"""

import re

from src import config
from src.schemas import GuardrailResult, Intent, IntentClassification

DECLINE_MESSAGE = (
    "I'm not able to continue this conversation given the language used. "
    "Please reach out to HR directly if you need help."
)

# Left word-boundary only (not \bword\b) so inflected forms match too -
# "bastards", "fucking" - a wordlist of exact singular forms would otherwise
# miss plurals/suffixes.
_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in config.TOXIC_WORDLIST) + r")",
    re.IGNORECASE,
)


def check_toxicity(message: str) -> GuardrailResult:
    """Deterministic wordlist check. Callers filing/describing a complaint
    should use check_toxicity_with_context() instead — this alone will
    false-positive on legitimate complaints that quote abusive language."""
    if _WORD_PATTERN.search(message):
        return GuardrailResult(allowed=False, reason=DECLINE_MESSAGE)
    return GuardrailResult(allowed=True)


def check_toxicity_semantic(classification: IntentClassification) -> GuardrailResult:
    """Reads the router's is_toxic judgment — the contextual backstop for
    what the wordlist can't distinguish (abuse directed at the assistant vs.
    abuse being reported)."""
    if classification.is_toxic:
        return GuardrailResult(allowed=False, reason=DECLINE_MESSAGE)
    return GuardrailResult(allowed=True)


def check_toxicity_with_context(
    message: str, classification: IntentClassification
) -> GuardrailResult:
    """The combined check orchestrator.run_turn() actually calls, after
    classification is available. Complaint intent trusts the semantic signal
    only; everything else is blocked on either layer."""
    if classification.intent == Intent.COMPLAINT:
        return check_toxicity_semantic(classification)

    wordlist_result = check_toxicity(message)
    if not wordlist_result.allowed:
        return wordlist_result
    return check_toxicity_semantic(classification)
