"""Tests for src/memory/hr_notifications.py — idempotent send-state for the
HR handoff email. Same tmp_path + monkeypatch(config.SQLITE_PATH) pattern as
tests/test_onboarding_status.py."""

from src import config
from src.memory import hr_notifications
from src.schemas import EmailSendResult, NotificationStatus


def _use_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_notifications.db")


def test_not_sent_before_any_attempt(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    assert hr_notifications.already_sent("EMP-04821", "hash-a") is False


def test_record_attempt_then_sent_marks_idempotent(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    attempts = hr_notifications.record_attempt("EMP-04821", "hash-a")
    assert attempts == 1
    hr_notifications.record_result(
        "EMP-04821", "hash-a", EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-1")
    )
    assert hr_notifications.already_sent("EMP-04821", "hash-a") is True


def test_failed_attempt_is_not_treated_as_sent(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    hr_notifications.record_attempt("EMP-04821", "hash-a")
    hr_notifications.record_result(
        "EMP-04821", "hash-a", EmailSendResult(status=NotificationStatus.FAILED, error="mailtrap rejected request")
    )
    assert hr_notifications.already_sent("EMP-04821", "hash-a") is False


def test_record_attempt_increments_on_retry(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    first = hr_notifications.record_attempt("EMP-04821", "hash-a")
    second = hr_notifications.record_attempt("EMP-04821", "hash-a")
    assert (first, second) == (1, 2)


def test_different_packet_hash_is_independent(monkeypatch, tmp_path):
    """A replaced document changes packet_hash — a new, legitimate send,
    not blocked by an earlier packet's sent status."""
    _use_tmp_db(monkeypatch, tmp_path)
    hr_notifications.record_attempt("EMP-04821", "hash-a")
    hr_notifications.record_result(
        "EMP-04821", "hash-a", EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-1")
    )
    assert hr_notifications.already_sent("EMP-04821", "hash-b") is False


def test_list_for_employee_reports_sent_rows(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    hr_notifications.record_attempt("EMP-04821", "hash-a")
    hr_notifications.record_result(
        "EMP-04821", "hash-a", EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-1")
    )
    rows = hr_notifications.list_for_employee("EMP-04821")
    assert len(rows) == 1
    assert rows[0]["status"] == "sent"
    assert rows[0]["sent_at"] is not None


def test_isolated_per_employee(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    hr_notifications.record_attempt("EMP-04821", "hash-a")
    hr_notifications.record_result(
        "EMP-04821", "hash-a", EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-1")
    )
    assert hr_notifications.already_sent("EMP-09999", "hash-a") is False


def test_has_ever_sent_false_before_any_send(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    assert hr_notifications.has_ever_sent("EMP-04821") is False


def test_has_ever_sent_false_when_only_attempted_or_failed(monkeypatch, tmp_path):
    _use_tmp_db(monkeypatch, tmp_path)
    hr_notifications.record_attempt("EMP-04821", "hash-a")
    hr_notifications.record_result(
        "EMP-04821", "hash-a", EmailSendResult(status=NotificationStatus.FAILED, error="mailtrap rejected request")
    )
    assert hr_notifications.has_ever_sent("EMP-04821") is False


def test_has_ever_sent_true_after_any_packet_hash_sent(monkeypatch, tmp_path):
    """The upload-lock correction-notice gate (src/api.py) only needs to know
    HR was told SOMETHING about this employee before -- any sent packet_hash
    qualifies, not specifically the original "verified" one."""
    _use_tmp_db(monkeypatch, tmp_path)
    hr_notifications.record_attempt("EMP-04821", "hash-a")
    hr_notifications.record_result(
        "EMP-04821", "hash-a", EmailSendResult(status=NotificationStatus.SENT, provider_id="msg-1")
    )
    assert hr_notifications.has_ever_sent("EMP-04821") is True
    assert hr_notifications.has_ever_sent("EMP-09999") is False
