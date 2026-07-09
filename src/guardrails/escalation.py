from __future__ import annotations

from src.guardrails.danger_scan import danger_scan
from src.schemas import (
    ComplaintCategory,
    ComplaintTicket,
    EscalationDecision,
    Severity,
    TriggerRule,
)

MANDATORY_ESCALATION_CATEGORIES = frozenset(
    {
        ComplaintCategory.HARASSMENT,
        ComplaintCategory.DISCRIMINATION,
        ComplaintCategory.SAFETY,
        ComplaintCategory.LEGAL,
    }
)

_SEVERITY_ORDER = (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)


def _floor_severity(current: Severity, floor: Severity) -> Severity:
    """Raise `current` up to at least `floor`. Never lowers a severity the
    model/employee already reported as higher."""
    if _SEVERITY_ORDER.index(current) >= _SEVERITY_ORDER.index(floor):
        return current
    return floor


def should_escalate(ticket: ComplaintTicket, raw_text: str) -> EscalationDecision:
    """Evaluate the full rule matrix against a completed, schema-valid ticket."""
    danger = danger_scan(raw_text)

    if danger.is_dangerous:
        return EscalationDecision(
            should_escalate=True,
            trigger_rule=TriggerRule.DANGER_SCAN,
            effective_severity=Severity.CRITICAL,
            danger_flag=True,
            rationale="danger scan matched an immediate-risk signal in the employee's own words",
        )

    if ticket.category in MANDATORY_ESCALATION_CATEGORIES:
        return EscalationDecision(
            should_escalate=True,
            trigger_rule=TriggerRule.MANDATORY_CATEGORY,
            effective_severity=_floor_severity(ticket.severity, Severity.HIGH),
            danger_flag=False,
            rationale=f"category={ticket.category.value} is a mandatory-escalation category",
        )

    if danger.is_retaliation:
        return EscalationDecision(
            should_escalate=True,
            trigger_rule=TriggerRule.RETALIATION_FLOOR,
            effective_severity=_floor_severity(ticket.severity, Severity.HIGH),
            danger_flag=False,
            rationale="retaliation language detected; severity floored to high",
        )

    if ticket.severity in (Severity.HIGH, Severity.CRITICAL):
        return EscalationDecision(
            should_escalate=True,
            trigger_rule=TriggerRule.SEVERITY_ESCALATION,
            effective_severity=ticket.severity,
            danger_flag=False,
            rationale=f"severity={ticket.severity.value} meets the escalation threshold on its own",
        )

    return EscalationDecision(
        should_escalate=False,
        trigger_rule=None,
        effective_severity=ticket.severity,
        danger_flag=False,
        rationale="no escalation trigger matched; filed as a standard ticket",
    )


def fail_safe_decision(reason: str) -> EscalationDecision:
    """Rule 7: always escalates. Used when the ReAct loop cannot complete a
    structured ComplaintTicket but the turn was already classified as a
    complaint -- the conversation must never end in silence."""
    return EscalationDecision(
        should_escalate=True,
        trigger_rule=TriggerRule.PARSE_FAILURE,
        effective_severity=Severity.HIGH,
        danger_flag=False,
        rationale=reason,
    )
