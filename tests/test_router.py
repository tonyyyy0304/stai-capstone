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


def test_router_prompt_mentions_new_document_intents():
    """The Intent enum has had document_upload/document_status since the
    earlier merge, but ROUTER_PROMPT never mentioned them -- classify_intent
    would never actually produce them. Pins down that the prompt text
    actually offers these as options now (Phase 6)."""
    from src.agent import prompts

    assert "document_upload" in prompts.ROUTER_PROMPT
    assert "document_status" in prompts.ROUTER_PROMPT


def test_classify_intent_passes_through_document_upload():
    parsed = IntentClassification(intent=Intent.DOCUMENT_UPLOAD, confidence=0.9)
    result = classify_intent("how do I upload my NBI clearance?", client=FakeClient(parsed))
    assert result.intent is Intent.DOCUMENT_UPLOAD
    assert not needs_clarification(result)


def test_classify_intent_passes_through_document_status():
    parsed = IntentClassification(intent=Intent.DOCUMENT_STATUS, confidence=0.9)
    result = classify_intent("what's the status of my NBI clearance?", client=FakeClient(parsed))
    assert result.intent is Intent.DOCUMENT_STATUS
    assert not needs_clarification(result)


def test_classify_intent_records_token_usage():
    parsed = IntentClassification(intent=Intent.FAQ, confidence=0.9)
    client = FakeClient(parsed, usage_metadata=FakeUsageMetadata(prompt=12, candidates=6, total=18))

    classify_intent("how many vacation days do I get?", client=client, session_id="s1")

    summary = usage.get_usage_summary(session_id="s1")
    assert summary["request_count"] == 1
    assert summary["prompt_tokens"] == 12
    assert summary["completion_tokens"] == 6
    assert summary["total_tokens"] == 18
