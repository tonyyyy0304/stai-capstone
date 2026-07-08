"""Toxicity/abuse filter (Module 6: Guardrails). Keyword heuristic, not an ML
classifier — this is an internal HR tool used by authenticated employees, not
a public-facing surface, so the bar is catching blatant abuse directed at the
bot/HR staff, not comprehensive content moderation. PLAN.md §8 explicitly
flags scope creep as a risk; the wordlist (config.TOXIC_WORDLIST) is
intentionally small and documented there rather than exhaustive.
"""

import re

from src import config
from src.schemas import GuardrailResult

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
    if _WORD_PATTERN.search(message):
        return GuardrailResult(allowed=False, reason=DECLINE_MESSAGE)
    return GuardrailResult(allowed=True)
