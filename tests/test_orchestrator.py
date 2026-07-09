import pytest

from src import config
from src.agent import orchestrator, tools
from src.schemas import AnswerSource, GroundedAnswer, Intent, IntentClassification


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


class FakeUsageMetadata:
    def __init__(self, prompt=10, candidates=5, total=15):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


class FakeResponse:
    def __init__(self, parts, text=None, usage_metadata=None):
        self.candidates = [FakeCandidate(FakeContent(parts))]
        self.text = text
        self.usage_metadata = usage_metadata


class FakeModels:
    def __init__(self, responses):
        self._responses = iter(responses)

    def generate_content(self, model, contents, config):
        return next(self._responses)


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


def text_response(text, usage_metadata=None):
    return FakeResponse([FakePart(function_call=None)], text=text, usage_metadata=usage_metadata)


def tool_call_response(name, args, usage_metadata=None):
    return FakeResponse(
        [FakePart(function_call=FakeFunctionCall(name, args))], usage_metadata=usage_metadata
    )


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("HR_ESCALATION_EMAIL_TO", raising=False)


def mock_classification(monkeypatch, intent, confidence=0.9, category=None, clarifying_question=None):
    classification = IntentClassification(
        intent=intent, confidence=confidence, category=category, clarifying_question=clarifying_question
    )
    monkeypatch.setattr(
        orchestrator,
        "classify_intent",
        lambda message, history, client=None, session_id=None: classification,
    )


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
        lambda question, client=None, session_id=None: GroundedAnswer(
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
    # No monkeypatch on should_escalate here: category="harassment" below hits
    # the real Rule 1 (mandatory category) in src/guardrails/escalation.py, so
    # this test exercises the actual deterministic rule, not a stand-in for
    # it. (The full rule matrix -- Rules 1/2/3/4/7 -- is covered in
    # tests/test_guardrails.py.)
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


def test_loop_exhausts_iterations_during_complaint_triggers_failsafe_escalation(monkeypatch):
    # Same shape as test_loop_stops_at_max_iterations above, but classified as
    # a COMPLAINT and never successfully completing a file_complaint call --
    # this must escalate (Rule 7), not just return MAX_ITERATIONS_REPLY.
    mock_classification(monkeypatch, Intent.COMPLAINT)
    responses = [
        tool_call_response("file_complaint", {"category": "payroll"})  # missing severity/description
        for _ in range(config.MAX_REACT_ITERATIONS)
    ]
    result = orchestrator.run_turn(
        "s10", "I have an ongoing issue I can't fully explain", client=FakeClient(responses)
    )
    assert result.reply != orchestrator.MAX_ITERATIONS_REPLY
    assert result.ticket_id is not None
    assert result.escalated is True
    assert result.trigger_rule == "parse_failure"
    status = tools.get_ticket_status(result.ticket_id)
    assert status["status"] == "escalated"
    assert status["category"] == "other"


def test_gemini_api_error_returns_graceful_fallback(monkeypatch):
    from google.genai.errors import ServerError

    def raise_unavailable(message, history, client=None, session_id=None):
        raise ServerError(503, {"error": {"message": "high demand"}})

    monkeypatch.setattr(orchestrator, "classify_intent", raise_unavailable)
    result = orchestrator.run_turn("s8", "how many vacation days do I get?", client=FakeClient([]))
    assert result.reply == orchestrator.API_ERROR_REPLY


def test_token_usage_aggregates_across_react_iterations(monkeypatch):
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
            tool_call_response(
                "search_kb",
                {"question": "sick leave days"},
                usage_metadata=FakeUsageMetadata(prompt=10, candidates=5, total=15),
            ),
            text_response(
                "You get 15 sick leave days per year.",
                usage_metadata=FakeUsageMetadata(prompt=20, candidates=8, total=28),
            ),
        ]
    )
    result = orchestrator.run_turn("s9", "how many sick leave days do I get?", client=client)
    assert result.token_usage.prompt_tokens == 30
    assert result.token_usage.completion_tokens == 13
    assert result.token_usage.total_tokens == 43
