import pytest
from pydantic import ValidationError

from src.schemas import Citation, GroundedAnswer, Intent, IntentClassification


def test_intent_classification_valid():
    ic = IntentClassification(intent="faq", confidence=0.9, category="leave")
    assert ic.intent is Intent.FAQ
    assert ic.clarifying_question is None


def test_intent_confidence_bounds():
    with pytest.raises(ValidationError):
        IntentClassification(intent="faq", confidence=1.5)
    with pytest.raises(ValidationError):
        IntentClassification(intent="faq", confidence=-0.1)


def test_intent_rejects_unknown_value():
    with pytest.raises(ValidationError):
        IntentClassification(intent="chitchat", confidence=0.5)


def test_grounded_answer_roundtrip():
    ga = GroundedAnswer(
        answer="You get 15 days.",
        citations=[Citation(chunk_id="faculty-manual-2021#001", title="Faculty Manual 2021",
                            section_path="Full-time Academic Faculty > Benefits > Leaves (p.42)")],
    )
    parsed = GroundedAnswer.model_validate_json(ga.model_dump_json())
    assert parsed.citations[0].chunk_id == "faculty-manual-2021#001"
    assert parsed.insufficient_context is False
