"""PII / observability boundary (Module 6: Guardrails, Module 11: LLMOps
Monitoring).

to_escalation_event() is the only function allowed to construct an
EscalationEvent, and EscalationEvent is the only escalation-related object
allowed to reach logs, MLflow traces, or the HR notification channel. This
module never reads ComplaintTicket.description or parties_involved into
anything it returns -- the redacted summary is built from category, severity,
and trigger metadata only. Same boundary principle already applied to Memory
in src/memory/persistent.py, which "never reads complaint-form PII payloads".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.schemas import ComplaintTicket, EscalationDecision, EscalationEvent

SLA_HOURS = 24  # PLAN.md Sec 6.1, Rule 5: sla_deadline = created_at + 24h


def _redacted_summary(ticket: ComplaintTicket, decision: EscalationDecision) -> str:
    """Non-PII summary: category, severity, and trigger only -- never the
    employee's free-text description or named parties."""
    parts = [
        f"category={ticket.category.value}",
        f"severity={decision.effective_severity.value}",
    ]
    if decision.trigger_rule is not None:
        parts.append(f"trigger={decision.trigger_rule.value}")
    if decision.danger_flag:
        parts.append("danger_flag=true")
    return "; ".join(parts)


def to_escalation_event(
    ticket: ComplaintTicket, ticket_id: str, decision: EscalationDecision
) -> EscalationEvent:
    """Build the redacted, loggable event from a ticket + rule decision.

    Only call this once `decision.should_escalate` is True -- it raises
    otherwise, since an EscalationEvent with no trigger_rule is meaningless.
    """
    if not decision.should_escalate or decision.trigger_rule is None:
        raise ValueError(
            "to_escalation_event() requires an escalating EscalationDecision with a trigger_rule"
        )

    now = datetime.now(timezone.utc)
    return EscalationEvent(
        ticket_id=ticket_id,
        category=ticket.category,
        severity=decision.effective_severity,
        trigger_rule=decision.trigger_rule,
        sla_deadline=now + timedelta(hours=SLA_HOURS),
        redacted_summary=_redacted_summary(ticket, decision),
        created_at=now,
    )
