"""PII detection/redaction (Module 6: Guardrails). Regex-only, no LLM call.

Two uses, deliberately scoped differently:
(a) Redaction before logging — src/monitoring.py already never logs raw
    employee message/reply text (only char/word counts), so the CLAUDE.md
    rule ("redact PII before logging anything to MLflow") is satisfied by
    omission today. redact_pii() exists here so any future logging surface
    has a ready-made, tested redaction path instead of reinventing one.
(b) NOT used as an input-blocking gate. Employees legitimately mention
    emails/phone numbers in ordinary FAQ questions, and the complaint-intake
    flow (PLAN.md §6.1) intentionally collects PII (names, contact, parties)
    for the HR email — blocking on PII presence would break that flow and is
    poor UX for the FAQ path too. Detection feeds redaction, not rejection.
"""

import re

from src import config

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_PATTERN = re.compile(config.PHONE_PATTERN)
_EMPLOYEE_ID_PATTERN = re.compile(config.EMPLOYEE_ID_PATTERN, re.IGNORECASE)

_PATTERNS = (
    ("email", _EMAIL_PATTERN),
    ("phone", _PHONE_PATTERN),
    ("employee_id", _EMPLOYEE_ID_PATTERN),
)


def detect_pii(text: str) -> list[str]:
    """Returns the PII categories found (e.g. ["email", "phone"]), not the
    matched values themselves — callers that just need to know "should this
    be treated as containing PII" don't need the raw matches."""
    return [label for label, pattern in _PATTERNS if pattern.search(text)]


def redact_pii(text: str) -> str:
    """Replaces each detected PII match with a `[REDACTED_<TYPE>]` placeholder."""
    redacted = text
    for label, pattern in _PATTERNS:
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted
