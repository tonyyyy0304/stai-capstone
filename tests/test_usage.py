from datetime import datetime, timedelta, timezone

import pytest

from src import config
from src.agent import usage
from src.schemas import TokenUsage


@pytest.fixture(autouse=True)
def isolated_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")


class FakeUsageMetadata:
    def __init__(self, prompt=10, candidates=5, total=15):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


class FakeResponse:
    def __init__(self, usage_metadata=None):
        self.usage_metadata = usage_metadata


def test_extract_usage_from_response():
    response = FakeResponse(usage_metadata=FakeUsageMetadata(10, 5, 15))
    result = usage.extract_usage(response)
    assert result == TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_extract_usage_missing_metadata_returns_zeros():
    assert usage.extract_usage(FakeResponse(usage_metadata=None)) == TokenUsage()


def test_record_and_summarize_usage():
    usage.record_usage(
        "gemini-2.5-flash",
        TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        session_id="s1",
    )
    usage.record_usage(
        "gemini-2.5-flash",
        TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        session_id="s2",
    )

    summary = usage.get_usage_all_time()
    assert summary["request_count"] == 2
    assert summary["prompt_tokens"] == 30
    assert summary["completion_tokens"] == 13
    assert summary["total_tokens"] == 43


def test_record_usage_skips_all_zero_usage():
    usage.record_usage("gemini-2.5-flash", TokenUsage(), session_id="s1")
    assert usage.get_usage_all_time()["request_count"] == 0


def test_get_usage_summary_filters_by_session():
    usage.record_usage(
        "gemini-2.5-flash",
        TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        session_id="s1",
    )
    usage.record_usage(
        "gemini-2.5-flash",
        TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        session_id="s2",
    )

    summary = usage.get_usage_summary(session_id="s1")
    assert summary["request_count"] == 1
    assert summary["total_tokens"] == 15


def test_get_usage_summary_filters_by_since():
    usage.record_usage(
        "gemini-2.5-flash", TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )

    future = datetime.now(timezone.utc) + timedelta(hours=1)
    assert usage.get_usage_summary(since=future)["request_count"] == 0

    past = datetime.now(timezone.utc) - timedelta(hours=1)
    assert usage.get_usage_summary(since=past)["request_count"] == 1


def test_get_usage_today_includes_recent_usage():
    usage.record_usage(
        "gemini-2.5-flash", TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    assert usage.get_usage_today()["request_count"] == 1
