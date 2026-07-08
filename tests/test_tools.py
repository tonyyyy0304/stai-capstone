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


def test_no_web_answer_shape():
    answer = tools.no_web_answer()
    assert answer.source == AnswerSource.NONE
    assert answer.insufficient_context is True


class FakeTavilyClient:
    def __init__(self, results=None, raises=None):
        self._results = results if results is not None else []
        self._raises = raises
        self.captured_kwargs = None

    def search(self, query, **kwargs):
        self.captured_kwargs = {"query": query, **kwargs}
        if self._raises:
            raise self._raises
        return {"results": self._results}


TAVILY_RESULTS = [
    {
        "url": "https://dole.gov.ph/13th-month-pay-advisory",
        "title": "DOLE 13th Month Pay Advisory",
        "content": "13th month pay is mandated by Presidential Decree 851.",
    }
]


def test_tavily_search_restricts_to_allowed_domains():
    client = FakeTavilyClient(results=TAVILY_RESULTS)
    tools._tavily_search("is 13th month pay required?", tavily_client=client)
    assert client.captured_kwargs["include_domains"] == list(config.DOLE_ALLOWED_DOMAINS)


def test_tavily_search_fails_closed_on_api_error():
    from tavily.errors import InvalidAPIKeyError

    client = FakeTavilyClient(raises=InvalidAPIKeyError("bad key"))
    assert tools._tavily_search("is 13th month pay required?", tavily_client=client) == []


def test_format_tavily_results_includes_url_and_content():
    formatted = tools._format_tavily_results(TAVILY_RESULTS)
    assert "https://dole.gov.ph/13th-month-pay-advisory" in formatted
    assert "Presidential Decree 851" in formatted


class FakeShapeResponse:
    def __init__(self, parsed, usage_metadata=None):
        self.parsed = parsed
        self.usage_metadata = usage_metadata


class FakeShapeModels:
    def __init__(self, parsed):
        self._parsed = parsed

    def generate_content(self, model, contents, config):
        return FakeShapeResponse(self._parsed)


class FakeShapeClient:
    def __init__(self, parsed):
        self.models = FakeShapeModels(parsed)


def test_search_web_shapes_tavily_results_into_grounded_answer():
    tavily_client = FakeTavilyClient(results=TAVILY_RESULTS)
    parsed = GroundedAnswer(
        answer="13th month pay is mandated by PD 851.", source=AnswerSource.INTERNAL_KB
    )
    gemini_client = FakeShapeClient(parsed)

    answer = tools.search_web(
        "is 13th month pay required?", client=gemini_client, tavily_client=tavily_client
    )

    assert answer.source == AnswerSource.WEB
    assert answer.citations == []
    assert answer.web_citations[0].url == "https://dole.gov.ph/13th-month-pay-advisory"


def test_search_web_returns_no_web_answer_when_tavily_finds_nothing():
    tavily_client = FakeTavilyClient(results=[])
    answer = tools.search_web("obscure question", client=FakeShapeClient(None), tavily_client=tavily_client)
    assert answer.source == AnswerSource.NONE
    assert answer.insufficient_context is True


def test_search_web_fails_closed_when_shaped_answer_is_insufficient():
    tavily_client = FakeTavilyClient(results=TAVILY_RESULTS)
    parsed = GroundedAnswer(answer="", source=AnswerSource.INTERNAL_KB, insufficient_context=True)
    answer = tools.search_web(
        "is 13th month pay required?", client=FakeShapeClient(parsed), tavily_client=tavily_client
    )
    assert answer.source == AnswerSource.NONE
    assert answer.insufficient_context is True
