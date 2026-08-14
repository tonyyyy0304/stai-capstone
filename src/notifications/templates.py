"""Plain text + HTML rendering for the HR handoff packet, both from one
HrPacket so the two formats can't drift (Component 14 follow-on).

Structure and the esc() HTML-escaping helper are lifted in shape from the
deleted Midterm emailer's _build_email_bodies()
(git show 23417c9~1:src/agent/tools.py) — that function's own docstring
already establishes the house convention: one function, one source of truth,
HTML-escape everything that came from outside the system.
"""

from __future__ import annotations

import html

from src import config
from src.schemas import DocType, HrPacket

_DOC_TYPE_LABELS = {DocType.NBI_CLEARANCE: "NBI Clearance", DocType.GOVERNMENT_ID: "Government ID"}


def esc(text: str) -> str:
    return html.escape(text)


def subject_line(packet: HrPacket) -> str:
    return (
        f"[Onboarding] Requirements verified — {packet.employee_id} — "
        f"{packet.faculty_class_label} — NBI + Gov ID"
    )


def _verified_lines(packet: HrPacket) -> list[str]:
    lines = []
    for doc in packet.verified_docs:
        label = _DOC_TYPE_LABELS.get(doc.doc_type, doc.doc_type.value)
        lines.append(f"  - {label}: {doc.outcome.value} (validated {doc.validated_at or 'unknown date'})")
    return lines


def _outstanding_lines(packet: HrPacket) -> list[str]:
    lines = []
    for item in packet.outstanding:
        lines.append(f"  - {item.item} [{item.source_doc} {item.section}]")
    return lines


def build_email_bodies(packet: HrPacket) -> tuple[str, str]:
    """Returns (plain_text, html_body). The single authorized PII-egress
    point for name-discrepancy values (see hr_packet.py's module docstring
    for the precedent) — not currently populated by the gated send path,
    but any future caller that sets packet.discrepancy renders it here,
    nowhere else (never in RuleResult.detail, never in an MLflow tag)."""
    plain_lines = [
        "An employee's onboarding requirements have been verified by the Faculty Onboarding Concierge.",
        "",
        f"Employee ID:     {packet.employee_id}",
        f"Faculty class:   {packet.faculty_class_label}",
        f"Verified at:     {packet.verified_at or 'unknown'}",
        "",
    ]
    if packet.reduced_detail:
        plain_lines += [
            "NOTE: per-field detail from one document was unavailable at send time "
            "(the shared extraction cache had been cleared, e.g. by a restart between "
            "uploads). The checklist and outcomes below are still authoritative.",
            "",
        ]

    plain_lines += ["Verified by the Concierge", "--------------------------"]
    plain_lines += _verified_lines(packet)
    plain_lines.append("")

    plain_lines += ["Still outstanding (HRMO to collect)", "------------------------------------"]
    outstanding_lines = _outstanding_lines(packet)
    plain_lines += outstanding_lines or ["  (no faculty class on file — full checklist could not be scoped)"]
    plain_lines.append("")

    if packet.nbi_printed_valid_until or packet.nbi_resubmit_by:
        plain_lines += [
            "NBI Clearance re-submit-by",
            "---------------------------",
            f"  Printed validity (as issued): {packet.nbi_printed_valid_until or 'unknown'}",
            f"  Employer freshness deadline ({config.NBI_VALIDITY_MONTHS} months from print date): "
            f"{packet.nbi_resubmit_by or 'unknown'}",
            "",
        ]

    if packet.discrepancy is not None:
        plain_lines += [
            "Name discrepancy flagged for adjudication",
            "------------------------------------------",
            f"  As printed on NBI Clearance: {packet.discrepancy.nbi_name}",
            f"  As printed on Government ID: {packet.discrepancy.id_name}",
            "",
        ]

    plain_lines += [
        f"Next action: File under {packet.employee_id}; request the {len(packet.outstanding)} outstanding "
        "item(s) above.",
        "",
        "Confidential: this message contains information from a government-issued clearance and ID, "
        "which are sensitive personal information under the Philippine Data Privacy Act (RA 10173). "
        "Handle and retain per HRMO's data privacy policy.",
    ]
    plain_text = "\n".join(plain_lines)

    verified_html = "".join(
        f'<li style="padding:3px 0;">{esc(_DOC_TYPE_LABELS.get(d.doc_type, d.doc_type.value))}: '
        f'{esc(d.outcome.value)} (validated {esc(d.validated_at or "unknown date")})</li>'
        for d in packet.verified_docs
    )
    outstanding_html = "".join(
        f'<li style="padding:3px 0;">{esc(item.item)} '
        f'<span style="color:#9ca3af;font-size:12px;">[{esc(item.source_doc)} {esc(item.section)}]</span></li>'
        for item in packet.outstanding
    ) or '<li style="padding:3px 0;color:#9ca3af;">No faculty class on file — full checklist could not be scoped.</li>'

    reduced_banner = (
        '<tr><td style="padding:10px 20px;background:#fffbeb;color:#92400e;font-size:12px;">'
        "Per-field detail from one document was unavailable at send time (shared extraction cache "
        "cleared between uploads). The checklist and outcomes below are still authoritative.</td></tr>"
        if packet.reduced_detail
        else ""
    )
    resubmit_html = (
        f"""
        <tr><td style="padding:16px 20px 4px;font-size:13px;font-weight:600;color:#111827;">NBI Clearance re-submit-by</td></tr>
        <tr><td style="padding:0 20px 16px;font-size:13px;color:#374151;">
          Printed validity (as issued): {esc(packet.nbi_printed_valid_until or 'unknown')}<br>
          Employer freshness deadline ({config.NBI_VALIDITY_MONTHS} months from print date): {esc(packet.nbi_resubmit_by or 'unknown')}
        </td></tr>
        """
        if packet.nbi_printed_valid_until or packet.nbi_resubmit_by
        else ""
    )
    discrepancy_html = (
        f"""
        <tr><td style="padding:16px 20px 4px;font-size:13px;font-weight:600;color:#b91c1c;">Name discrepancy flagged for adjudication</td></tr>
        <tr><td style="padding:0 20px 16px;font-size:13px;color:#374151;">
          As printed on NBI Clearance: {esc(packet.discrepancy.nbi_name)}<br>
          As printed on Government ID: {esc(packet.discrepancy.id_name)}
        </td></tr>
        """
        if packet.discrepancy is not None
        else ""
    )

    html_body = f"""\
<html>
  <body style="margin:0;padding:0;background-color:#f3f4f6;font-family:Arial,Helvetica,sans-serif;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f3f4f6;padding:24px 0;">
      <tr>
        <td align="center">
          <table role="presentation" width="600" cellpadding="0" cellspacing="0"
                 style="background-color:#ffffff;border-radius:8px;overflow:hidden;border:1px solid #e5e7eb;">
            <tr>
              <td style="background-color:#0f766e;padding:18px 20px;">
                <span style="color:#ffffff;font-size:16px;font-weight:700;">Onboarding requirements verified</span><br>
                <span style="color:#ffffff;font-size:13px;opacity:0.9;">{esc(packet.employee_id)} &middot; {esc(packet.faculty_class_label)}</span>
              </td>
            </tr>
            {reduced_banner}
            <tr>
              <td style="padding:16px 20px 4px;font-size:13px;font-weight:600;color:#111827;">Verified by the Concierge ({len(packet.verified_docs)})</td>
            </tr>
            <tr><td style="padding:0 20px 8px;"><ul style="margin:0;padding-left:18px;font-size:13px;color:#374151;">{verified_html}</ul></td></tr>
            <tr>
              <td style="padding:16px 20px 4px;font-size:13px;font-weight:600;color:#111827;">Still outstanding (HRMO to collect)</td>
            </tr>
            <tr><td style="padding:0 20px 8px;"><ul style="margin:0;padding-left:18px;font-size:13px;color:#374151;">{outstanding_html}</ul></td></tr>
            {resubmit_html}
            {discrepancy_html}
            <tr>
              <td style="padding:14px 20px;background-color:#eff6ff;font-size:13px;color:#1e3a8a;">
                Next action: file under {esc(packet.employee_id)}; request the {len(packet.outstanding)} outstanding item(s) above.
              </td>
            </tr>
            <tr>
              <td style="padding:14px 20px;background-color:#f9fafb;border-top:1px solid #e5e7eb;font-size:11px;color:#9ca3af;">
                Confidential: contains information from a government-issued clearance and ID, sensitive personal
                information under the Philippine Data Privacy Act (RA 10173). Handle and retain per HRMO policy.
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""
    return plain_text, html_body
