from src.agent.router import classify_intent, format_history, needs_clarification
from src.schemas import Intent, IntentClassification


class FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed


class FakeModels:
    def __init__(self, parsed):
        self._parsed = parsed

    def generate_content(self, model, contents, config):
        return FakeResponse(self._parsed)


class FakeClient:
    def __init__(self, parsed):
        self.models = FakeModels(parsed)


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
