import pytest

from src import config
from src.agent import orchestrator, tools
from src.schemas import AnswerSource, ComplaintCategory, GroundedAnswer, Intent, IntentClassification


class FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class FakePart:
    def __init__(self, function_call=None):
        self.function_call = function_call


class FakeContent:
    def __init__(self, parts):
        self.parts = parts


class FakeCandidate:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    def __init__(self, parts, text=None):
        self.candidates = [FakeCandidate(FakeContent(parts))]
        self.text = text


class FakeModels:
    def __init__(self, responses):
        self._responses = iter(responses)

    def generate_content(self, model, contents, config):
        return next(self._responses)


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


def text_response(text):
    return FakeResponse([FakePart(function_call=None)], text=text)


def tool_call_response(name, args):
    return FakeResponse([FakePart(function_call=FakeFunctionCall(name, args))])


@pytest.fixture(autouse=True)
def isolated_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")


def mock_classification(monkeypatch, intent, confidence=0.9, category=None, clarifying_question=None):
    classification = IntentClassification(
        intent=intent, confidence=confidence, category=category, clarifying_question=clarifying_question
    )
    monkeypatch.setattr(orchestrator, "classify_intent", lambda message, history, client=None: classification)


def test_ambiguous_intent_short_circuits_without_tool_call(monkeypatch):
    mock_classification(monkeypatch, Intent.AMBIGUOUS, confidence=0.3, clarifying_question="Policy or complaint?")
    result = orchestrator.run_turn("s1", "something vague", client=FakeClient([]))
    assert result.reply == "Policy or complaint?"
    assert result.steps[-1].tool is None


def test_out_of_scope_declines(monkeypatch):
    mock_classification(monkeypatch, Intent.OUT_OF_SCOPE, confidence=0.9)
    result = orchestrator.run_turn("s2", "what's the weather today?", client=FakeClient([]))
    assert result.reply == orchestrator.OUT_OF_SCOPE_REPLY
    assert all(step.tool is None for step in result.steps)


def test_faq_uses_search_kb_then_answers(monkeypatch):
    mock_classification(monkeypatch, Intent.FAQ, category="leave")
    monkeypatch.setattr(
        tools,
        "search_kb",
        lambda question, category=None: (
            GroundedAnswer(answer="15 sick days a year.", source=AnswerSource.INTERNAL_KB),
            ["chunk1"],
        ),
    )
    client = FakeClient(
        [
            tool_call_response("search_kb", {"question": "sick leave days"}),
            text_response("You get 15 sick leave days per year."),
        ]
    )
    result = orchestrator.run_turn("s3", "how many sick leave days do I get?", client=client)
    assert result.reply == "You get 15 sick leave days per year."
    assert [s.tool for s in result.steps] == [None, "search_kb"]


def test_faq_falls_back_to_search_web_when_kb_insufficient(monkeypatch):
    mock_classification(monkeypatch, Intent.FAQ, category="labor_law")
    monkeypatch.setattr(
        tools,
        "search_kb",
        lambda question, category=None: (
            GroundedAnswer(answer="", source=AnswerSource.NONE, insufficient_context=True),
            [],
        ),
    )
    monkeypatch.setattr(
        tools,
        "search_web",
        lambda question, client=None: GroundedAnswer(
            answer="13th month pay is mandated by PD 851.", source=AnswerSource.WEB
        ),
    )
    client = FakeClient(
        [
            tool_call_response("search_kb", {"question": "13th month pay"}),
            tool_call_response("search_web", {"question": "13th month pay"}),
            text_response("13th month pay is mandated by PD 851."),
        ]
    )
    result = orchestrator.run_turn("s4", "is 13th month pay required by law?", client=client)
    assert result.reply == "13th month pay is mandated by PD 851."
    assert [s.tool for s in result.steps] == [None, "search_kb", "search_web"]


def test_complaint_files_ticket_and_escalates_deterministically(monkeypatch):
    mock_classification(monkeypatch, Intent.COMPLAINT)
    monkeypatch.setattr(tools, "should_escalate", lambda category: category == ComplaintCategory.HARASSMENT)
    client = FakeClient(
        [
            tool_call_response(
                "file_complaint",
                {
                    "category": "harassment",
                    "severity": "high",
                    "description": "My manager keeps yelling at me in front of the team.",
                },
            ),
            text_response("I've filed your complaint and HR has been notified directly."),
        ]
    )
    result = orchestrator.run_turn("s5", "my manager keeps yelling at me", client=client)
    assert result.ticket_id is not None
    assert result.escalated is True
    assert tools.get_ticket_status(result.ticket_id)["status"] == "escalated"


def test_complaint_missing_fields_asks_again_without_filing(monkeypatch):
    mock_classification(monkeypatch, Intent.COMPLAINT)
    filed = []
    monkeypatch.setattr(tools, "file_complaint", lambda ticket: filed.append(ticket) or "unused")
    client = FakeClient(
        [
            tool_call_response("file_complaint", {"category": "payroll"}),
            text_response("Could you tell me the severity and what happened?"),
        ]
    )
    result = orchestrator.run_turn("s6", "I have a payroll issue", client=client)
    assert result.reply == "Could you tell me the severity and what happened?"
    assert result.ticket_id is None
    assert filed == []


def test_loop_stops_at_max_iterations(monkeypatch):
    mock_classification(monkeypatch, Intent.FAQ)
    monkeypatch.setattr(
        tools,
        "search_kb",
        lambda question, category=None: (
            GroundedAnswer(answer="", source=AnswerSource.NONE, insufficient_context=True),
            [],
        ),
    )
    responses = [
        tool_call_response("search_kb", {"question": "x"}) for _ in range(config.MAX_REACT_ITERATIONS)
    ]
    result = orchestrator.run_turn("s7", "loop forever", client=FakeClient(responses))
    assert result.reply == orchestrator.MAX_ITERATIONS_REPLY
    assert len(result.steps) == 1 + config.MAX_REACT_ITERATIONS


def test_gemini_api_error_returns_graceful_fallback(monkeypatch):
    from google.genai.errors import ServerError

    def raise_unavailable(message, history, client=None):
        raise ServerError(503, {"error": {"message": "high demand"}})

    monkeypatch.setattr(orchestrator, "classify_intent", raise_unavailable)
    result = orchestrator.run_turn("s8", "how many vacation days do I get?", client=FakeClient([]))
    assert result.reply == orchestrator.API_ERROR_REPLY
