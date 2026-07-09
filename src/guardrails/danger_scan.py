from __future__ import annotations

import re
from dataclasses import dataclass

_DANGER_PATTERNS = (
    r"\bweapon\b",
    r"\bgun\b",
    r"\bknife\b",
    r"\bhurt (?:myself|someone|him|her|them)\b",
    r"\bkill (?:myself|him|her|them|me)\b",
    r"\bsuicid\w*\b",
    r"\bself[- ]harm\w*\b",
    r"\bassault(?:ed|ing)?\b",
    r"\bthreaten(?:ed|ing)?\s+me\b",
    r"\bafraid for my (?:life|safety)\b",
    r"\bnot safe (?:here|at work|going back)\b",
)

_RETALIATION_PATTERNS = (
    r"\bfire(?:d)? (?:me )?for reporting\b",
    r"\bretaliat\w*\b",
    r"\bif i report\b",
    r"\bafraid (?:i'?ll|to) lose my job\b",
    r"\bpunish(?:ed|ing)? me\b",
)

_DANGER_RE = re.compile("|".join(_DANGER_PATTERNS), re.IGNORECASE)
_RETALIATION_RE = re.compile("|".join(_RETALIATION_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class DangerScanResult:
    is_dangerous: bool = False
    is_retaliation: bool = False
    matched_signal: str = ""  # category tag only, e.g. "danger_lexicon" -- never the raw match


def danger_scan(raw_text: str) -> DangerScanResult:
    """Pure, deterministic scan of the employee's raw message. No network
    call, no LLM call -- safe to run on every complaint-intent turn without
    touching the project's Gemini/Ollama request budget (PLAN.md Sec 2.1/8)."""
    if not raw_text:
        return DangerScanResult()

    if _DANGER_RE.search(raw_text):
        return DangerScanResult(is_dangerous=True, matched_signal="danger_lexicon")

    if _RETALIATION_RE.search(raw_text):
        return DangerScanResult(is_retaliation=True, matched_signal="retaliation_lexicon")

    return DangerScanResult()
