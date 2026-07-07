import pytest

from src import config
from src.agent import tools
from src.schemas import (
    AnswerSource,
    ComplaintCategory,
    ComplaintTicket,
    GroundedAnswer,
    Severity,
)


@pytest.fixture(autouse=True)
def isolated_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")


def make_ticket(category=ComplaintCategory.HARASSMENT, severity=Severity.HIGH):
    return ComplaintTicket(
        category=category,
        severity=severity,
        description="My manager keeps yelling at me in front of the team.",
        parties_involved=["manager"],
    )


def test_file_complaint_and_get_ticket_status_roundtrip():
    ticket_id = tools.file_complaint(make_ticket())
    status = tools.get_ticket_status(ticket_id)
    assert status["category"] == "harassment"
    assert status["status"] == "open"
    assert status["escalated"] == 0


def test_get_ticket_status_missing_id_returns_none():
    assert tools.get_ticket_status("does-not-exist") is None


def test_escalate_to_hr_flags_ticket():
    ticket_id = tools.file_complaint(make_ticket())
    result = tools.escalate_to_hr(ticket_id, reason="category=harassment")
    assert result["escalated"] is True
    status = tools.get_ticket_status(ticket_id)
    assert status["status"] == "escalated"
    assert status["escalated"] == 1


@pytest.mark.parametrize(
    "category,expected",
    [
        (ComplaintCategory.HARASSMENT, True),
        (ComplaintCategory.DISCRIMINATION, True),
        (ComplaintCategory.SAFETY, True),
        (ComplaintCategory.LEGAL, True),
        (ComplaintCategory.PAYROLL, False),
        (ComplaintCategory.OTHER, False),
    ],
)
def test_should_escalate_is_deterministic(category, expected):
    assert tools.should_escalate(category) is expected


def test_search_kb_delegates_to_answerer(monkeypatch):
    expected = (GroundedAnswer(answer="15 days.", source=AnswerSource.INTERNAL_KB), [])
    captured = {}

    def fake_answer_question(question, category=None):
        captured["question"] = question
        captured["category"] = category
        return expected

    monkeypatch.setattr(tools, "answer_question", fake_answer_question)
    result = tools.search_kb("how many vacation days?", category="leave")
    assert result is expected
    assert captured == {"question": "how many vacation days?", "category": "leave"}


class FakeWeb:
    def __init__(self, uri, title):
        self.uri = uri
        self.title = title


class FakeGroundingChunk:
    def __init__(self, uri, title):
        self.web = FakeWeb(uri, title)


class FakeGroundingMetadata:
    def __init__(self, chunks):
        self.grounding_chunks = chunks


class FakeCandidate:
    def __init__(self, grounding_metadata):
        self.grounding_metadata = grounding_metadata


class FakeSearchResponse:
    def __init__(self, text, chunks):
        self.text = text
        self.candidates = [FakeCandidate(FakeGroundingMetadata(chunks))]


def test_extract_web_citations_filters_to_allowlisted_domains():
    response = FakeSearchResponse(
        "some grounded text",
        [
            FakeGroundingChunk("https://dole.gov.ph/some-advisory", "DOLE Advisory"),
            FakeGroundingChunk("https://random-blog.example/labor-law", "Unrelated blog"),
        ],
    )
    citations = tools._extract_web_citations(response)
    assert len(citations) == 1
    assert citations[0].url == "https://dole.gov.ph/some-advisory"


def test_no_web_answer_shape():
    answer = tools.no_web_answer()
    assert answer.source == AnswerSource.NONE
    assert answer.insufficient_context is True
