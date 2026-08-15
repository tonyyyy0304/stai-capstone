"""Tests for the HR handoff email (Component 14 follow-on): hr_packet
composition, templates, the /upload-doc gate, idempotency, PII discipline,
attachment policy, failure isolation, and the corpus↔config citation
crossover. Mirrors tests/test_api.py's fixtures (_nbi_result/_id_result/
_png_bytes/_quality_pass) and its "mock extract_document, never a real
Gemini call" convention.
"""

import pdfplumber
from fastapi.testclient import TestClient

from src import api, config
from src.memory import hr_notifications
from src.notifications.hr_packet import compose_correction_notice, compose_packet, compose_review_alert
from src.notifications.templates import (
    build_correction_email_bodies,
    build_email_bodies,
    build_review_alert_email_bodies,
    correction_subject_line,
    review_alert_subject_line,
    subject_line,
)
from src.schemas import (
    ChecklistStatus,
    DocStatus,
    DocType,
    EmailSendResult,
    ExtractedField,
    IdExtractionResult,
    IdType,
    ImageQualityReport,
    NbiExtractionResult,
    NotificationStatus,
    OnboardingDocument,
    QualityVerdict,
    RuleResult,
    ValidationOutcome,
    ValidationResult,
)


def _use_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_email.db")


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"0" * 64


def _field(value, confidence=0.95):
    return ExtractedField(value=value, verbatim_text=value or "", model_confidence=confidence)


def _quality_pass():
    return ImageQualityReport(
        verdict=QualityVerdict.PASS, blur_score=1200.0, exposure_clip=0.01, skew_deg=1.0,
        min_dim_px=1200, quad_found=True, normalized_quality=0.95, reasons=[],
    )


def _nbi_result(**overrides):
    base = dict(
        doc_type=DocType.NBI_CLEARANCE, family_name=_field("REYES"), first_name=_field("MARIA"),
        middle_name=_field("SANTOS"), date_of_birth=_field("1990-01-01"),
        reference_no=_field("REYE900101-N00457821"), date_printed=_field("2026-02-14"),
        valid_until=_field("2027-02-14"), purpose=_field("Employment"), remarks=_field("NO DEROGATORY"),
        overall_confidence=0.95,
    )
    base.update(overrides)
    return NbiExtractionResult(**base)


def _id_result(id_type=IdType.NATIONAL_ID, **overrides):
    base = dict(
        doc_type=DocType.GOVERNMENT_ID, id_type=id_type, family_name=_field("REYES"), first_name=_field("MARIA"),
        middle_name=_field("SANTOS"), date_of_birth=_field("1990-01-01"), id_number=_field("1234-5678-9101-0001"),
        issue_date=_field("2019-06-14"), expiry_date=_field(None, 0.0), overall_confidence=0.95,
    )
    base.update(overrides)
    return IdExtractionResult(**base)


def _checklist(employee_id="EMP-04821", faculty_class="full_time_academic") -> ChecklistStatus:
    return ChecklistStatus(
        employee_id=employee_id,
        faculty_class=faculty_class,
        missing=[],
        documents=[
            OnboardingDocument(
                employee_id=employee_id, doc_type=DocType.NBI_CLEARANCE, status=DocStatus.VALIDATED,
                outcome=ValidationOutcome.ACCEPTED, validated_at="2026-08-14T00:00:00+00:00", source_hash="nbi-hash",
            ),
            OnboardingDocument(
                employee_id=employee_id, doc_type=DocType.GOVERNMENT_ID, status=DocStatus.VALIDATED,
                outcome=ValidationOutcome.ACCEPTED, validated_at="2026-08-14T00:01:00+00:00", source_hash="id-hash",
            ),
        ],
    )


# --- hr_packet.compose_packet (pure, no I/O) ----------------------------------

def test_compose_packet_full_detail():
    packet = compose_packet(_nbi_result(), _id_result(), _checklist(), "full_time_academic")
    assert packet.employee_id == "EMP-04821"
    assert packet.faculty_class_label == "Full-time Academic Faculty"
    assert len(packet.verified_docs) == 2
    assert packet.reduced_detail is False
    assert packet.nbi_printed_valid_until == "2027-02-14"
    assert packet.nbi_resubmit_by  # computed from date_printed + NBI_VALIDITY_MONTHS
    assert len(packet.outstanding) == len(config.PREEMPLOYMENT_CHECKLIST["full_time_academic"])


def test_compose_packet_reduced_detail_when_sibling_cache_missing():
    """Container restart between the two uploads -> sibling OCR cache entry
    gone -> packet still composes, flagged reduced_detail, never skipped."""
    packet = compose_packet(_nbi_result(), None, _checklist(), "full_time_academic")
    assert packet.reduced_detail is True
    assert len(packet.verified_docs) == 2  # checklist-derived, unaffected by the cache miss


def test_compose_packet_unknown_faculty_class_degrades_gracefully():
    packet = compose_packet(_nbi_result(), _id_result(), _checklist(), None)
    assert packet.outstanding == []
    assert packet.faculty_class_label == "Unspecified"


def test_compose_packet_hash_is_order_independent_and_stable():
    p1 = compose_packet(_nbi_result(), _id_result(), _checklist(), "full_time_academic")
    checklist_reordered = _checklist()
    checklist_reordered.documents = list(reversed(checklist_reordered.documents))
    p2 = compose_packet(_nbi_result(), _id_result(), checklist_reordered, "full_time_academic")
    assert p1.packet_hash == p2.packet_hash


def test_compose_packet_hash_changes_when_a_document_is_replaced():
    p1 = compose_packet(_nbi_result(), _id_result(), _checklist(), "full_time_academic")
    checklist2 = _checklist()
    checklist2.documents[0].source_hash = "new-nbi-hash"
    p2 = compose_packet(_nbi_result(), _id_result(), checklist2, "full_time_academic")
    assert p1.packet_hash != p2.packet_hash


# --- hr_packet.compose_correction_notice (pure, no I/O) -----------------------

def _rejected_validation(message="Reference number format looks incorrect.") -> ValidationResult:
    return ValidationResult(
        outcome=ValidationOutcome.REJECTED,
        rules=[RuleResult(rule="format", passed=False, detail="reference_no failed pattern check")],
        composite_confidence=0.2,
        message=message,
    )


def _regressed_checklist(employee_id="EMP-04821") -> ChecklistStatus:
    """NBI regressed from VALIDATED to REJECTED; Government ID is still
    VALIDATED and unaffected -- the shape compose_correction_notice sees
    right after api.py re-fetches the checklist post-record_result."""
    checklist = _checklist(employee_id=employee_id)
    checklist.documents[0].status = DocStatus.REJECTED
    checklist.documents[0].outcome = ValidationOutcome.REJECTED
    checklist.documents[0].source_hash = "new-nbi-hash"
    checklist.missing = [DocType.NBI_CLEARANCE.value]
    return checklist


def test_compose_correction_notice_shape():
    validation = _rejected_validation()
    notice = compose_correction_notice(DocType.NBI_CLEARANCE, validation, _regressed_checklist(), "full_time_academic")
    assert notice.employee_id == "EMP-04821"
    assert notice.faculty_class_label == "Full-time Academic Faculty"
    assert notice.doc_type == DocType.NBI_CLEARANCE
    assert notice.new_outcome == ValidationOutcome.REJECTED
    assert notice.new_status == DocStatus.REJECTED
    assert notice.message == validation.message
    # the Government ID is still VALIDATED and unaffected by the NBI regression
    assert len(notice.still_verified) == 1
    assert notice.still_verified[0].doc_type == DocType.GOVERNMENT_ID


def test_compose_correction_notice_hash_differs_from_original_send():
    """The whole idempotency story rests on this: a regression always
    changes the regressed document's source_hash, which is enough on its
    own to make notice_hash differ from the original packet_hash --
    already_sent() (src/memory/hr_notifications.py) needs no changes."""
    original_packet = compose_packet(_nbi_result(), _id_result(), _checklist(), "full_time_academic")
    notice = compose_correction_notice(
        DocType.NBI_CLEARANCE, _rejected_validation(), _regressed_checklist(), "full_time_academic"
    )
    assert notice.notice_hash != original_packet.packet_hash


def test_compose_correction_notice_unknown_faculty_class_degrades_gracefully():
    notice = compose_correction_notice(DocType.NBI_CLEARANCE, _rejected_validation(), _regressed_checklist(), None)
    assert notice.faculty_class_label == "Unspecified"


# --- hr_packet.compose_review_alert (pure, no I/O) -----------------------------

def _needs_review_validation(message="Extraction confidence too low for auto-decision.") -> ValidationResult:
    return ValidationResult(
        outcome=ValidationOutcome.NEEDS_REVIEW,
        rules=[RuleResult(rule="fail_safe", passed=False, detail="low confidence")],
        composite_confidence=0.3,
        message=message,
    )


def test_compose_review_alert_shape():
    validation = _needs_review_validation()
    alert = compose_review_alert(DocType.NBI_CLEARANCE, validation, _checklist(), "full_time_academic", "some-hash")
    assert alert.employee_id == "EMP-04821"
    assert alert.faculty_class_label == "Full-time Academic Faculty"
    assert alert.doc_type == DocType.NBI_CLEARANCE
    assert alert.message == validation.message


def test_compose_review_alert_hash_keys_off_doc_type_and_source_hash_only():
    """Deliberately NOT the whole checklist (unlike packet_hash/notice_hash)
    -- this alert is about one document's own content, so it must not
    change just because the sibling document changes."""
    alert1 = compose_review_alert(DocType.NBI_CLEARANCE, _needs_review_validation(), _checklist(), "full_time_academic", "hash-a")
    checklist2 = _checklist()
    checklist2.documents[1].source_hash = "a-totally-different-id-hash"  # sibling changes
    alert2 = compose_review_alert(DocType.NBI_CLEARANCE, _needs_review_validation(), checklist2, "full_time_academic", "hash-a")
    assert alert1.review_hash == alert2.review_hash

    alert3 = compose_review_alert(DocType.NBI_CLEARANCE, _needs_review_validation(), _checklist(), "full_time_academic", "hash-b")
    assert alert1.review_hash != alert3.review_hash

    alert4 = compose_review_alert(DocType.GOVERNMENT_ID, _needs_review_validation(), _checklist(), "full_time_academic", "hash-a")
    assert alert1.review_hash != alert4.review_hash  # same source_hash, different doc_type


def test_compose_review_alert_unknown_faculty_class_degrades_gracefully():
    alert = compose_review_alert(DocType.NBI_CLEARANCE, _needs_review_validation(), _checklist(), None, "some-hash")
    assert alert.faculty_class_label == "Unspecified"


# --- templates.build_review_alert_email_bodies / review_alert_subject_line ----

def test_review_alert_subject_line_is_triageable():
    alert = compose_review_alert(DocType.NBI_CLEARANCE, _needs_review_validation(), _checklist(), "full_time_academic", "some-hash")
    subject = review_alert_subject_line(alert)
    assert "EMP-04821" in subject
    assert "NBI Clearance" in subject
    assert "Review needed" in subject


def test_review_alert_email_bodies_contain_required_blocks():
    validation = _needs_review_validation("Extraction confidence too low for auto-decision.")
    alert = compose_review_alert(DocType.GOVERNMENT_ID, validation, _checklist(), "full_time_academic", "some-hash")
    plain_text, html_body = build_review_alert_email_bodies(alert)

    for body in (plain_text, html_body):
        assert "EMP-04821" in body
        assert "Government ID" in body
        assert "Extraction confidence too low for auto-decision." in body
        assert "Data Privacy Act" in body


def test_review_alert_email_bodies_escape_html_in_message():
    validation = _needs_review_validation("<script>alert(1)</script>")
    alert = compose_review_alert(DocType.NBI_CLEARANCE, validation, _checklist(), "full_time_academic", "some-hash")
    _, html_body = build_review_alert_email_bodies(alert)
    assert "<script>alert(1)</script>" not in html_body
    assert "&lt;script&gt;" in html_body


# --- templates.build_email_bodies / subject_line ------------------------------

def test_subject_line_is_triageable():
    packet = compose_packet(_nbi_result(), _id_result(), _checklist(), "full_time_academic")
    subject = subject_line(packet)
    assert "EMP-04821" in subject
    assert "Full-time Academic Faculty" in subject
    assert "NBI + Gov ID" in subject


def test_email_bodies_contain_required_blocks_and_stay_in_sync():
    packet = compose_packet(_nbi_result(), _id_result(), _checklist(), "part_time_academic")
    plain_text, html_body = build_email_bodies(packet)

    for body in (plain_text, html_body):
        assert "EMP-04821" in body
        assert "Part-time Academic Faculty" in body
        assert "NBI Clearance" in body
        assert "Government ID" in body
        assert "2027-02-14" in body  # printed validity
        assert packet.nbi_resubmit_by in body  # employer freshness deadline
        assert "Data Privacy Act" in body
    # every declared outstanding item for this class must actually render
    for item in config.PREEMPLOYMENT_CHECKLIST["part_time_academic"]:
        assert item["item"] in plain_text
        assert item["item"] in html_body
    assert "NBI Clearance" not in "\n".join(
        item["item"] for item in config.PREEMPLOYMENT_CHECKLIST["part_time_academic"]
    )  # sanity: the source list itself never re-lists what the Concierge already verified


def test_email_bodies_reduced_detail_banner():
    packet = compose_packet(_nbi_result(), None, _checklist(), "full_time_academic")
    plain_text, html_body = build_email_bodies(packet)
    assert "unavailable at send time" in plain_text
    assert "unavailable at send time" in html_body


def test_email_bodies_escape_html_in_faculty_class_label(monkeypatch):
    """esc() must actually run on interpolated values -- regression guard
    against a raw f-string slipping untouched into html_body."""
    packet = compose_packet(_nbi_result(), _id_result(), _checklist(), "full_time_academic")
    packet = packet.model_copy(update={"faculty_class_label": "<script>alert(1)</script>"})
    _, html_body = build_email_bodies(packet)
    assert "<script>alert(1)</script>" not in html_body
    assert "&lt;script&gt;" in html_body


# --- templates.build_correction_email_bodies / correction_subject_line --------

def test_correction_subject_line_is_triageable():
    notice = compose_correction_notice(DocType.NBI_CLEARANCE, _rejected_validation(), _regressed_checklist(), "full_time_academic")
    subject = correction_subject_line(notice)
    assert "EMP-04821" in subject
    assert "NBI Clearance" in subject
    assert "ACTION NEEDED" in subject


def test_correction_email_bodies_contain_required_blocks():
    validation = _rejected_validation("Reference number format looks incorrect.")
    notice = compose_correction_notice(DocType.NBI_CLEARANCE, validation, _regressed_checklist(), "full_time_academic")
    plain_text, html_body = build_correction_email_bodies(notice)

    for body in (plain_text, html_body):
        assert "EMP-04821" in body
        assert "NBI Clearance" in body
        assert "rejected" in body
        assert "Reference number format looks incorrect." in body
        assert "stale" in body.lower()
        assert "Government ID" in body  # still_verified section
        assert "Data Privacy Act" in body


def test_correction_email_bodies_no_still_verified_section_when_empty():
    """Both documents regressed (or the sibling was never validated) --
    still_verified is empty, and the section just doesn't render, no
    empty-list artifact."""
    checklist = _regressed_checklist()
    checklist.documents[1].status = DocStatus.MISSING
    notice = compose_correction_notice(DocType.NBI_CLEARANCE, _rejected_validation(), checklist, "full_time_academic")
    assert notice.still_verified == []
    plain_text, html_body = build_correction_email_bodies(notice)
    assert "Still verified" not in plain_text
    assert "Still verified" not in html_body


def test_correction_email_bodies_escape_html_in_message():
    validation = _rejected_validation("<script>alert(1)</script>")
    notice = compose_correction_notice(DocType.NBI_CLEARANCE, validation, _regressed_checklist(), "full_time_academic")
    _, html_body = build_correction_email_bodies(notice)
    assert "<script>alert(1)</script>" not in html_body
    assert "&lt;script&gt;" in html_body


# --- Corpus <-> config crossover: every declared citation must actually ------
# --- exist in the source PDF, so the email never asserts a fabricated item ---

def _raw_preemployment_text() -> str:
    path = config.RAW_DIR / f"{config.PREEMPLOYMENT_CHECKLIST_SOURCE}.pdf"
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def test_preemployment_checklist_sections_resolve_to_the_ingested_source():
    text = _raw_preemployment_text()
    for faculty_class, rows in config.PREEMPLOYMENT_CHECKLIST.items():
        for row in rows:
            assert row["section"] in text, (
                f"{faculty_class}: declared section {row['section']!r} not found in "
                f"{config.PREEMPLOYMENT_CHECKLIST_SOURCE}.pdf"
            )


# --- Gate: POST /upload-doc, end to end via TestClient ------------------------
# TestClient runs FastAPI BackgroundTasks synchronously before returning, so
# these assert directly on send_packet's call count / args.

def _monkeypatch_dual_extract(monkeypatch, nbi, id_doc):
    monkeypatch.setattr(
        api, "extract_document",
        lambda image_bytes, mime_type, expected_doc_type=None:
            (nbi, _quality_pass()) if expected_doc_type == DocType.NBI_CLEARANCE else (id_doc, _quality_pass()),
    )
    monkeypatch.setattr(
        api, "load_cached_result",
        lambda source_hash, doc_type, preprocess_flag=None:
            id_doc if doc_type == DocType.GOVERNMENT_ID else nbi,
    )


def _upload(client, employee_id, doc_type, filename, full_name="REYES, MARIA SANTOS", faculty_class="full_time_academic", file_bytes=None):
    return client.post(
        "/upload-doc",
        data={
            "employee_id": employee_id, "doc_type": doc_type,
            "full_name": full_name, "date_of_birth": "1990-01-01", "faculty_class": faculty_class,
        },
        files={"file": (filename, file_bytes or _png_bytes(), "image/png")},
    )


def test_gate_sends_only_once_both_documents_validated(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    nbi, id_doc = _nbi_result(), _id_result()
    _monkeypatch_dual_extract(monkeypatch, nbi, id_doc)
    calls = []
    monkeypatch.setattr(
        api, "send_packet",
        lambda packet, attachments=None: calls.append((packet, attachments)) or EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-1"),
    )
    client = TestClient(api.app)

    r1 = _upload(client, "EMP-10", "nbi_clearance", "nbi.png")
    assert r1.status_code == 200
    assert calls == []  # only one of two documents on file -- must not fire yet

    r2 = _upload(client, "EMP-10", "government_id", "id.png")
    assert r2.status_code == 200
    assert len(calls) == 1
    packet, attachments = calls[0]
    assert packet.employee_id == "EMP-10"
    assert attachments is None  # EMAIL_ATTACH_ORIGINALS is False by default


def test_gate_never_fires_on_needs_review(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    nbi = _nbi_result()
    low_confidence_id = _id_result(overall_confidence=0.1)  # trips fail_safe -> needs_review
    _monkeypatch_dual_extract(monkeypatch, nbi, low_confidence_id)
    calls = []
    monkeypatch.setattr(api, "send_packet", lambda packet, attachments=None: calls.append(1) or EmailSendResult(status=NotificationStatus.SENT))
    client = TestClient(api.app)

    _upload(client, "EMP-11", "nbi_clearance", "nbi.png")
    r2 = _upload(client, "EMP-11", "government_id", "id.png")

    assert r2.json()["validation"]["outcome"] == "needs_review"
    assert calls == []


def test_gate_never_fires_on_cross_document_mismatch(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    nbi = _nbi_result()
    mismatched_id = _id_result(family_name=_field("CRUZ"), first_name=_field("JUAN"), middle_name=_field(None, 0.0))
    _monkeypatch_dual_extract(monkeypatch, nbi, mismatched_id)
    calls = []
    monkeypatch.setattr(api, "send_packet", lambda packet, attachments=None: calls.append(1) or EmailSendResult(status=NotificationStatus.SENT))
    client = TestClient(api.app)

    _upload(client, "EMP-12", "nbi_clearance", "nbi.png")
    r2 = _upload(client, "EMP-12", "government_id", "id.png", full_name="CRUZ, JUAN")

    assert r2.json()["validation"]["outcome"] == "needs_review"
    assert calls == []


def test_gate_never_fires_on_rejected(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(api, "send_packet", lambda packet, attachments=None: calls.append(1) or EmailSendResult(status=NotificationStatus.SENT))
    client = TestClient(api.app)

    r = _upload(client, "EMP-13", "nbi_clearance", "fake.png", file_bytes=b"not a real image")

    assert r.json()["validation"]["outcome"] == "rejected"
    assert calls == []


def test_reuploading_identical_pair_sends_exactly_once(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    nbi, id_doc = _nbi_result(), _id_result()
    _monkeypatch_dual_extract(monkeypatch, nbi, id_doc)
    calls = []
    monkeypatch.setattr(
        api, "send_packet",
        lambda packet, attachments=None: calls.append(1) or EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-1"),
    )
    client = TestClient(api.app)

    _upload(client, "EMP-14", "nbi_clearance", "nbi.png")
    _upload(client, "EMP-14", "government_id", "id.png")
    assert len(calls) == 1

    # Re-upload the identical pair again (same bytes -> same source_hash ->
    # same packet_hash). Neither request may re-fire the send.
    _upload(client, "EMP-14", "nbi_clearance", "nbi.png")
    _upload(client, "EMP-14", "government_id", "id.png")
    assert len(calls) == 1


def test_replacing_a_document_after_sent_triggers_a_new_send(monkeypatch, tmp_path):
    """A genuinely new file for one doc_type changes its source_hash, which
    changes packet_hash -- this IS a legitimate resend, not a duplicate."""
    _use_tmp_db(monkeypatch, tmp_path)
    nbi, id_doc = _nbi_result(), _id_result()
    _monkeypatch_dual_extract(monkeypatch, nbi, id_doc)
    calls = []
    monkeypatch.setattr(
        api, "send_packet",
        lambda packet, attachments=None: calls.append(1) or EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-1"),
    )
    client = TestClient(api.app)

    _upload(client, "EMP-15", "nbi_clearance", "nbi.png")
    _upload(client, "EMP-15", "government_id", "id.png")
    assert len(calls) == 1

    # Different bytes for the government ID -> different source_hash.
    different_bytes = _png_bytes() + b"1"
    _upload(client, "EMP-15", "government_id", "id2.png", file_bytes=different_bytes)
    assert len(calls) == 2


# --- Upload-lock correction notice: a VALIDATED document that regresses -------

def _correction_calls(email_calls):
    return [c for c in email_calls if c[0].startswith("[Onboarding] ACTION NEEDED")]


def _review_calls(email_calls):
    return [c for c in email_calls if c[0].startswith("[Onboarding] Review needed")]


def test_regression_after_hr_already_notified_triggers_correction_and_review_emails(monkeypatch, tmp_path):
    """Both documents validate (original send fires), then the NBI Clearance
    is re-uploaded and regresses to needs_review -- two independently true
    facts fire on this one request: HR was already told this employee was
    fully verified (now stale -- correction notice), AND this upload's own
    outcome needs a human to look at it (review alert). Neither is a
    duplicate of the other; the original hr_notifications row is untouched
    (additive, not a rewrite)."""
    _use_tmp_db(monkeypatch, tmp_path)
    nbi, id_doc = _nbi_result(), _id_result()
    _monkeypatch_dual_extract(monkeypatch, nbi, id_doc)
    packet_calls, email_calls = [], []
    monkeypatch.setattr(
        api, "send_packet",
        lambda packet, attachments=None: packet_calls.append(packet) or EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-1"),
    )
    monkeypatch.setattr(
        api, "send_email",
        lambda subject, plain_text, html_body, employee_id, content_hash, attachments=None:
            email_calls.append((subject, employee_id, content_hash)) or EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-2"),
    )
    client = TestClient(api.app)

    _upload(client, "EMP-19", "nbi_clearance", "nbi.png")
    _upload(client, "EMP-19", "government_id", "id.png")
    assert len(packet_calls) == 1
    assert email_calls == []

    # NBI re-uploaded, now low-confidence enough to trip fail_safe -> needs_review.
    low_confidence_nbi = _nbi_result(overall_confidence=0.1)
    _monkeypatch_dual_extract(monkeypatch, low_confidence_nbi, id_doc)
    r3 = _upload(client, "EMP-19", "nbi_clearance", "nbi2.png", file_bytes=_png_bytes() + b"2")

    assert r3.json()["validation"]["outcome"] == "needs_review"
    assert len(packet_calls) == 1  # the original send does not re-fire
    assert len(_correction_calls(email_calls)) == 1
    assert len(_review_calls(email_calls)) == 1
    assert all(c[1] == "EMP-19" for c in email_calls)

    rows = hr_notifications.list_for_employee("EMP-19")
    assert len(rows) == 3  # original packet + correction notice + review alert, all distinct hashes
    assert all(row["status"] == "sent" for row in rows)


def test_regression_before_any_send_still_triggers_review_but_not_correction(monkeypatch, tmp_path):
    """The correction gate is has_ever_sent(), not "any regression" -- a
    document that regresses before the checklist was ever complete has
    nothing to correct, since HR was never told anything about this
    employee. The review-alert gate is independent of that: it only cares
    about THIS upload's own outcome, so it fires regardless."""
    _use_tmp_db(monkeypatch, tmp_path)
    nbi, id_doc = _nbi_result(), _id_result()
    _monkeypatch_dual_extract(monkeypatch, nbi, id_doc)
    packet_calls, email_calls = [], []
    monkeypatch.setattr(
        api, "send_packet",
        lambda packet, attachments=None: packet_calls.append(1) or EmailSendResult(status=NotificationStatus.SENT),
    )
    monkeypatch.setattr(
        api, "send_email",
        lambda subject, plain_text, html_body, employee_id, content_hash, attachments=None:
            email_calls.append((subject, employee_id, content_hash)) or EmailSendResult(status=NotificationStatus.SENT),
    )
    client = TestClient(api.app)

    _upload(client, "EMP-20", "nbi_clearance", "nbi.png")  # solo upload -- validates, but checklist incomplete
    assert packet_calls == []

    low_confidence_nbi = _nbi_result(overall_confidence=0.1)
    _monkeypatch_dual_extract(monkeypatch, low_confidence_nbi, id_doc)
    r2 = _upload(client, "EMP-20", "nbi_clearance", "nbi2.png", file_bytes=_png_bytes() + b"2")

    assert r2.json()["validation"]["outcome"] == "needs_review"
    assert _correction_calls(email_calls) == []  # has_ever_sent("EMP-20") is False -- nothing to correct
    assert len(_review_calls(email_calls)) == 1  # but a human still needs to look at this document


def test_review_alert_fires_on_first_upload_with_no_prior_state(monkeypatch, tmp_path):
    """The gap this feature closes: a document that lands on NEEDS_REVIEW on
    its very FIRST upload (previous_doc is None -- not a regression at all)
    used to be completely silent. Now it alerts."""
    _use_tmp_db(monkeypatch, tmp_path)
    low_confidence_nbi = _nbi_result(overall_confidence=0.1)
    _monkeypatch_dual_extract(monkeypatch, low_confidence_nbi, _id_result())
    review_calls = []
    monkeypatch.setattr(
        api, "send_email",
        lambda subject, plain_text, html_body, employee_id, content_hash, attachments=None:
            review_calls.append((subject, employee_id)) or EmailSendResult(status=NotificationStatus.SENT),
    )
    client = TestClient(api.app)

    r = _upload(client, "EMP-21", "nbi_clearance", "nbi.png")

    assert r.json()["validation"]["outcome"] == "needs_review"
    assert len(review_calls) == 1
    assert review_calls[0][1] == "EMP-21"


def test_review_alert_never_fires_on_rejected(monkeypatch, tmp_path):
    """REJECTED is the deterministic, self-service retry path -- not a
    human-judgment case, deliberately out of scope for this alert."""
    _use_tmp_db(monkeypatch, tmp_path)
    review_calls = []
    monkeypatch.setattr(
        api, "send_email",
        lambda subject, plain_text, html_body, employee_id, content_hash, attachments=None:
            review_calls.append(1) or EmailSendResult(status=NotificationStatus.SENT),
    )
    client = TestClient(api.app)

    r = _upload(client, "EMP-22", "nbi_clearance", "fake.png", file_bytes=b"not a real image")

    assert r.json()["validation"]["outcome"] == "rejected"
    assert review_calls == []


def test_review_alert_is_idempotent_on_same_source_hash(monkeypatch, tmp_path):
    """Re-fetching the checklist (e.g. the UI's polling GET) must not
    re-trigger the alert -- and neither should literally resubmitting the
    same bytes twice."""
    _use_tmp_db(monkeypatch, tmp_path)
    low_confidence_nbi = _nbi_result(overall_confidence=0.1)
    _monkeypatch_dual_extract(monkeypatch, low_confidence_nbi, _id_result())
    review_calls = []
    monkeypatch.setattr(
        api, "send_email",
        lambda subject, plain_text, html_body, employee_id, content_hash, attachments=None:
            review_calls.append(1) or EmailSendResult(status=NotificationStatus.SENT),
    )
    client = TestClient(api.app)

    _upload(client, "EMP-23", "nbi_clearance", "nbi.png")
    _upload(client, "EMP-23", "nbi_clearance", "nbi.png")  # identical bytes -> identical source_hash
    assert len(review_calls) == 1


# --- Failure isolation ---------------------------------------------------------

def test_provider_failure_never_breaks_the_upload_response(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    nbi, id_doc = _nbi_result(), _id_result()
    _monkeypatch_dual_extract(monkeypatch, nbi, id_doc)
    monkeypatch.setattr(
        api, "send_packet",
        lambda packet, attachments=None: EmailSendResult(status=NotificationStatus.FAILED, error="mailtrap rejected request (status 500)"),
    )
    client = TestClient(api.app)

    _upload(client, "EMP-16", "nbi_clearance", "nbi.png")
    r2 = _upload(client, "EMP-16", "government_id", "id.png")

    assert r2.status_code == 200
    assert r2.json()["validation"]["outcome"] == "accepted"  # upload path itself is untouched
    rows = hr_notifications.list_for_employee("EMP-16")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"


def test_provider_failure_error_is_pii_free(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    nbi, id_doc = _nbi_result(), _id_result()
    _monkeypatch_dual_extract(monkeypatch, nbi, id_doc)
    monkeypatch.setattr(
        api, "send_packet",
        lambda packet, attachments=None: EmailSendResult(status=NotificationStatus.FAILED, error="mailtrap request timed out"),
    )
    client = TestClient(api.app)

    _upload(client, "EMP-17", "nbi_clearance", "nbi.png")
    _upload(client, "EMP-17", "government_id", "id.png")

    conn = hr_notifications._get_connection()
    try:
        row = conn.execute(
            "SELECT last_error FROM hr_notifications WHERE employee_id = ?", ("EMP-17",)
        ).fetchone()
    finally:
        conn.close()
    assert "REYES" not in row["last_error"]
    assert "MARIA" not in row["last_error"]
    assert "SANTOS" not in row["last_error"]


# --- PII discipline --------------------------------------------------------

def test_hr_notifications_row_never_carries_a_field_value(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    nbi, id_doc = _nbi_result(), _id_result()
    _monkeypatch_dual_extract(monkeypatch, nbi, id_doc)
    monkeypatch.setattr(
        api, "send_packet",
        lambda packet, attachments=None: EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-1"),
    )
    client = TestClient(api.app)

    _upload(client, "EMP-18", "nbi_clearance", "nbi.png")
    _upload(client, "EMP-18", "government_id", "id.png")

    conn = hr_notifications._get_connection()
    try:
        row = conn.execute("SELECT * FROM hr_notifications WHERE employee_id = ?", ("EMP-18",)).fetchone()
    finally:
        conn.close()
    row_text = " ".join(str(v) for v in tuple(row))
    assert "REYES" not in row_text
    assert "MARIA" not in row_text
    assert "SANTOS" not in row_text
    assert "1234-5678-9101-0001" not in row_text  # id_number
