import pytest

from src import config
from src.agent import usage
from src.agent.router import classify_intent, format_history, needs_clarification
from src.schemas import Intent, IntentClassification, TokenUsage


class FakeUsageMetadata:
    def __init__(self, prompt=12, candidates=6, total=18):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


class FakeResponse:
    def __init__(self, parsed, usage_metadata=None):
        self.parsed = parsed
        self.usage_metadata = usage_metadata


class FakeModels:
    def __init__(self, parsed, usage_metadata=None):
        self._parsed = parsed
        self._usage_metadata = usage_metadata

    def generate_content(self, model, contents, config):
        return FakeResponse(self._parsed, usage_metadata=self._usage_metadata)


class FakeClient:
    def __init__(self, parsed, usage_metadata=None):
        self.models = FakeModels(parsed, usage_metadata=usage_metadata)


@pytest.fixture(autouse=True)
def isolated_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")


def test_classify_intent_passes_through_confident_result():
    parsed = IntentClassification(intent=Intent.FAQ, confidence=0.9, category="leave")
    result = classify_intent("how many vacation days do I get?", client=FakeClient(parsed))
    assert result.intent is Intent.FAQ
    assert result.category == "leave"


def test_classify_intent_forces_ambiguous_below_confidence_floor():
    parsed = IntentClassification(intent=Intent.FAQ, confidence=0.2)
    result = classify_intent("something vague", client=FakeClient(parsed))
    assert result.intent is Intent.AMBIGUOUS
    assert result.clarifying_question


def test_classify_intent_fails_closed_on_unparseable_response():
    result = classify_intent("garbled input", client=FakeClient(None))
    assert result.intent is Intent.AMBIGUOUS
    assert result.confidence == 0.0
    assert result.clarifying_question


def test_needs_clarification():
    assert needs_clarification(IntentClassification(intent=Intent.AMBIGUOUS, confidence=0.5))
    assert needs_clarification(IntentClassification(intent=Intent.FAQ, confidence=0.1))
    assert not needs_clarification(IntentClassification(intent=Intent.FAQ, confidence=0.9))


def test_format_history_empty_and_populated():
    assert format_history([]) == "(no prior turns)"
    formatted = format_history([{"role": "user", "content": "hi"}])
    assert formatted == "user: hi"


def test_classify_intent_records_token_usage():
    parsed = IntentClassification(intent=Intent.FAQ, confidence=0.9)
    client = FakeClient(parsed, usage_metadata=FakeUsageMetadata(prompt=12, candidates=6, total=18))

    classify_intent("how many vacation days do I get?", client=client, session_id="s1")

    summary = usage.get_usage_summary(session_id="s1")
    assert summary["request_count"] == 1
    assert summary["prompt_tokens"] == 12
    assert summary["completion_tokens"] == 6
    assert summary["total_tokens"] == 18
