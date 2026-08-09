"""Tests for src/memory/onboarding_status.py (Component 14, CV_INTEGRATION.md
§2.8). tmp_path + monkeypatch.setattr(config, "SQLITE_PATH", ...), matching
tests/test_api.py:76's pattern — a fresh SQLite file per test, no shared state."""

from src import config
from src.memory import onboarding_status
from src.schemas import DocStatus, DocType, ValidationOutcome, ValidationResult


def _use_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_onboarding.db")


def _result(outcome: ValidationOutcome, composite: float = 0.9) -> ValidationResult:
    return ValidationResult(outcome=outcome, rules=[], composite_confidence=composite, message="")


def test_status_before_any_upload_is_all_missing(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    status = onboarding_status.get_status("EMP-04821")
    assert status.employee_id == "EMP-04821"
    assert set(status.missing) == set(config.REQUIRED_ONBOARDING_DOCS)
    assert all(d.status == DocStatus.MISSING for d in status.documents)
    assert len(status.documents) == len(config.REQUIRED_ONBOARDING_DOCS)


def test_record_result_then_get_status_round_trips_nbi(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    onboarding_status.record_result(
        "EMP-04821", DocType.NBI_CLEARANCE, _result(ValidationOutcome.ACCEPTED), source_hash="abc123"
    )
    status = onboarding_status.get_status("EMP-04821")
    nbi_doc = next(d for d in status.documents if d.doc_type == DocType.NBI_CLEARANCE)
    assert nbi_doc.status == DocStatus.VALIDATED
    assert nbi_doc.outcome == ValidationOutcome.ACCEPTED
    assert nbi_doc.source_hash == "abc123"
    assert "nbi_clearance" not in status.missing
    assert "government_id" in status.missing  # not uploaded yet


def test_record_result_round_trips_both_doc_types_on_same_employee(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    onboarding_status.record_result(
        "EMP-04821", DocType.NBI_CLEARANCE, _result(ValidationOutcome.ACCEPTED), source_hash="nbi-hash"
    )
    onboarding_status.record_result(
        "EMP-04821", DocType.GOVERNMENT_ID, _result(ValidationOutcome.ACCEPTED), source_hash="id-hash"
    )
    status = onboarding_status.get_status("EMP-04821")
    assert status.missing == []
    by_type = {d.doc_type: d for d in status.documents}
    assert by_type[DocType.NBI_CLEARANCE].source_hash == "nbi-hash"
    assert by_type[DocType.GOVERNMENT_ID].source_hash == "id-hash"


def test_needs_review_status_counts_as_missing(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    onboarding_status.record_result(
        "EMP-04821", DocType.NBI_CLEARANCE, _result(ValidationOutcome.NEEDS_REVIEW, composite=0.3), source_hash="h1"
    )
    status = onboarding_status.get_status("EMP-04821")
    assert "nbi_clearance" in status.missing
    nbi_doc = next(d for d in status.documents if d.doc_type == DocType.NBI_CLEARANCE)
    assert nbi_doc.status == DocStatus.NEEDS_REVIEW


def test_rejected_outcome_status_and_missing(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    onboarding_status.record_result(
        "EMP-04821", DocType.NBI_CLEARANCE, _result(ValidationOutcome.REJECTED, composite=0.0), source_hash="h1"
    )
    status = onboarding_status.get_status("EMP-04821")
    nbi_doc = next(d for d in status.documents if d.doc_type == DocType.NBI_CLEARANCE)
    assert nbi_doc.status == DocStatus.REJECTED
    assert "nbi_clearance" in status.missing


def test_record_result_upserts_on_resubmission(monkeypatch, tmp_path):
    """Re-uploading the same doc_type overwrites the row rather than
    creating a second one — one row per (employee_id, doc_type)."""
    _use_tmp_db(monkeypatch, tmp_path)
    onboarding_status.record_result(
        "EMP-04821", DocType.NBI_CLEARANCE, _result(ValidationOutcome.NEEDS_REVIEW, composite=0.3), source_hash="old-hash"
    )
    onboarding_status.record_result(
        "EMP-04821", DocType.NBI_CLEARANCE, _result(ValidationOutcome.ACCEPTED, composite=0.95), source_hash="new-hash"
    )
    status = onboarding_status.get_status("EMP-04821")
    nbi_docs = [d for d in status.documents if d.doc_type == DocType.NBI_CLEARANCE]
    assert len(nbi_docs) == 1
    assert nbi_docs[0].status == DocStatus.VALIDATED
    assert nbi_docs[0].source_hash == "new-hash"


def test_get_document_returns_none_when_not_yet_uploaded(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    assert onboarding_status.get_document("EMP-04821", DocType.GOVERNMENT_ID) is None


def test_get_document_sibling_lookup_for_cross_document_check(monkeypatch, tmp_path):
    """The mechanism validate_cross_document()'s caller (POST /upload-doc,
    Phase 6) uses to find the other document's source_hash."""
    _use_tmp_db(monkeypatch, tmp_path)
    onboarding_status.record_result(
        "EMP-04821", DocType.NBI_CLEARANCE, _result(ValidationOutcome.ACCEPTED), source_hash="nbi-hash"
    )
    sibling = onboarding_status.get_document("EMP-04821", DocType.NBI_CLEARANCE)
    assert sibling is not None
    assert sibling.source_hash == "nbi-hash"


def test_status_is_isolated_per_employee(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    onboarding_status.record_result(
        "EMP-04821", DocType.NBI_CLEARANCE, _result(ValidationOutcome.ACCEPTED), source_hash="h1"
    )
    other_status = onboarding_status.get_status("EMP-09999")
    assert set(other_status.missing) == set(config.REQUIRED_ONBOARDING_DOCS)


def test_faculty_class_round_trips(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    assert onboarding_status.get_faculty_class("EMP-04821") is None
    onboarding_status.set_faculty_class("EMP-04821", "part_time_academic")
    assert onboarding_status.get_faculty_class("EMP-04821") == "part_time_academic"
    status = onboarding_status.get_status("EMP-04821")
    assert status.faculty_class == "part_time_academic"


def test_faculty_class_upserts(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    onboarding_status.set_faculty_class("EMP-04821", "full_time_academic")
    onboarding_status.set_faculty_class("EMP-04821", "academic_service")
    assert onboarding_status.get_faculty_class("EMP-04821") == "academic_service"
