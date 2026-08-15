"""Email transport for the HR handoff packet (Component 14 follow-on).

Two implementations, selected by config.EMAIL_DRY_RUN (checked first,
independent of EMAIL_PROVIDER — a fresh checkout must never send real mail
without an explicit opt-in):

- Dry-run (default): writes the fully-composed message to
  config.OUTBOX_DIR/*.eml and sends nothing. Revival of the deleted Midterm
  emailer's _write_mock_outbox() (git show 23417c9~1:src/agent/tools.py) —
  "stays inspectable during dev/demo without a real mail server."
- Mailtrap Sandbox (config.EMAIL_PROVIDER == "mailtrap"): the only real
  transport. This is a proof-of-concept project with no production mailbox
  — Mailtrap's Sandbox API is built for exactly that: every send lands in a
  private test inbox (never a real recipient), no domain verification, no
  risk of actually mailing a real person's clearance/ID data. Deliberately
  NOT smtplib (the deleted emailer's port 465/587 STARTTLS branching) — an
  HTTP API sidesteps Docker's port-587 egress nuisance entirely.

Never raises, same contract as the deleted emailer's
_send_email_notification(): "a misconfigured or unreachable mail server must
not fail the employee's [submission]" — here, must not fail /upload-doc's
200 response, since this always runs inside a FastAPI BackgroundTasks call
that starts after the response is already built.
"""

from __future__ import annotations

import logging
import time
from email.message import EmailMessage

from src import config
from src.notifications.templates import build_email_bodies, subject_line
from src.schemas import EmailSendResult, HrPacket, NotificationStatus

logger = logging.getLogger(__name__)

Attachment = tuple[str, bytes, str]  # (filename, content, mime_type)


def _build_message(subject: str, plain_text: str, html_body: str, attachments: list[Attachment] | None) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config.HR_EMAIL_FROM
    message["To"] = config.HR_EMAIL_TO
    message.set_content(plain_text)
    message.add_alternative(html_body, subtype="html")
    for filename, content, mime_type in attachments or []:
        maintype, _, subtype = mime_type.partition("/")
        message.add_attachment(content, maintype=maintype, subtype=subtype or "octet-stream", filename=filename)
    return message


def _send_dry_run(
    subject: str, plain_text: str, html_body: str, employee_id: str, content_hash: str,
    attachments: list[Attachment] | None,
) -> EmailSendResult:
    message = _build_message(subject, plain_text, html_body, attachments)
    config.OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    outbox_path = config.OUTBOX_DIR / f"{employee_id}_{content_hash[:12]}.eml"
    try:
        outbox_path.write_bytes(bytes(message))
    except OSError as exc:
        logger.warning("hr_email_dry_run_write_failed error=%s", exc)
        return EmailSendResult(status=NotificationStatus.FAILED, error="dry_run: failed to write outbox file")
    return EmailSendResult(status=NotificationStatus.SENT, provider_id=f"dry-run:{outbox_path.name}")


def _send_mailtrap(
    subject: str, plain_text: str, html_body: str, attachments: list[Attachment] | None,
) -> EmailSendResult:
    """POSTs to Mailtrap's Sandbox Sending API
    (https://sandbox.api.mailtrap.io/api/send/{inbox_id}) — every message is
    caught in a private test inbox, never delivered to a real address, which
    is what makes it safe to point at with real (even if mock) NBI/ID data
    on a proof-of-concept deployment that has no production mailbox."""
    import base64

    import httpx

    payload = {
        "from": {"email": config.HR_EMAIL_FROM, "name": "Faculty Onboarding Concierge"},
        "to": [{"email": config.HR_EMAIL_TO}],
        "subject": subject,
        "text": plain_text,
        "html": html_body,
    }
    if attachments:
        payload["attachments"] = [
            {
                "filename": filename,
                "content": base64.b64encode(content).decode("ascii"),
                "type": mime_type,
                "disposition": "attachment",
            }
            for filename, content, mime_type in attachments
        ]

    url = f"https://sandbox.api.mailtrap.io/api/send/{config.MAILTRAP_INBOX_ID}"
    last_error = "unknown error"
    for attempt in range(1, config.EMAIL_MAX_RETRIES + 1):
        try:
            response = httpx.post(
                url,
                headers={"Authorization": f"Bearer {config.get_mailtrap_api_token()}"},
                json=payload,
                timeout=30.0,
            )
        except httpx.TimeoutException as exc:
            last_error = "mailtrap request timed out"
            logger.warning("hr_email_send_timeout attempt=%s/%s error=%s", attempt, config.EMAIL_MAX_RETRIES, exc)
        except httpx.HTTPError as exc:
            last_error = "mailtrap request failed"
            logger.warning("hr_email_send_error attempt=%s/%s error=%s", attempt, config.EMAIL_MAX_RETRIES, exc)
        else:
            if response.status_code < 300:
                provider_id = ""
                try:
                    ids = response.json().get("message_ids") or []
                    provider_id = ids[0] if ids else ""
                except ValueError:
                    pass
                return EmailSendResult(status=NotificationStatus.SENT, provider_id=provider_id)
            if response.status_code < 500:
                # 4xx (bad request, auth, validation, unknown inbox) won't succeed on retry.
                logger.warning("hr_email_send_rejected status=%s", response.status_code)
                return EmailSendResult(
                    status=NotificationStatus.FAILED, error=f"mailtrap rejected request (status {response.status_code})"
                )
            last_error = f"mailtrap server error (status {response.status_code})"
            logger.warning("hr_email_send_5xx attempt=%s/%s status=%s", attempt, config.EMAIL_MAX_RETRIES, response.status_code)

        if attempt < config.EMAIL_MAX_RETRIES:
            time.sleep(_backoff_seconds(attempt))

    return EmailSendResult(status=NotificationStatus.FAILED, error=last_error)


def _backoff_seconds(attempt: int) -> float:
    """Small bounded exponential backoff (1s, 2s, 4s, ...) — a background
    task, so there's no request latency budget to protect, just no reason to
    hammer a struggling provider."""
    return float(2 ** (attempt - 1))


def send_email(
    subject: str,
    plain_text: str,
    html_body: str,
    employee_id: str,
    content_hash: str,
    attachments: list[Attachment] | None = None,
) -> EmailSendResult:
    """Sends (or dry-run-writes) any already-composed email to HR. Never
    raises — any unexpected failure degrades to a FAILED EmailSendResult with
    a PII-free error string, exactly like every other fail-safe path in this
    codebase (src/api.py's upload_doc, extractor.py's _call_gemini).

    Generic transport, not HrPacket-specific — employee_id/content_hash are
    only used to name the dry-run .eml file, same naming scheme as before
    (packet.employee_id/packet.packet_hash). This is what lets the
    correction-notice path (src/notifications/hr_packet.py's
    compose_correction_notice) reuse the exact same dry-run/Mailtrap/retry
    logic as the original handoff packet without a second copy of it."""
    try:
        if config.EMAIL_DRY_RUN:
            return _send_dry_run(subject, plain_text, html_body, employee_id, content_hash, attachments)
        return _send_mailtrap(subject, plain_text, html_body, attachments)
    except Exception as exc:  # last-resort fail-closed, matches upload_doc's own broad catch
        logger.exception("hr_email_send_unexpected_error error_type=%s", type(exc).__name__)
        return EmailSendResult(status=NotificationStatus.FAILED, error=f"unexpected error: {type(exc).__name__}")


def send_packet(packet: HrPacket, attachments: list[Attachment] | None = None) -> EmailSendResult:
    """Sends (or dry-run-writes) the HR handoff packet. Thin wrapper over
    send_email() — composes packet -> (subject, plain, html) via
    templates.py, same as before this function was generalized."""
    plain_text, html_body = build_email_bodies(packet)
    return send_email(subject_line(packet), plain_text, html_body, packet.employee_id, packet.packet_hash, attachments)
