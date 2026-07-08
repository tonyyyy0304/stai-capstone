from fastapi.testclient import TestClient

from src import api, config
from src.agent import orchestrator
from src.agent import usage as usage_tracker
from src.rag.retriever import RetrievedChunk
from src.schemas import GroundedAnswer, TokenUsage


def test_health_degraded_without_runtime_state(monkeypatch, tmp_path):
    monkeypatch.setattr(api.config, "CHROMA_DIR", tmp_path / "missing_chroma")
    monkeypatch.setattr(api.config, "MANIFEST_PATH", tmp_path / "missing_manifest.json")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_chat_falls_back_to_plain_rag_when_agent_unavailable(monkeypatch):
    """_try_agent_orchestrator returns None (e.g. import fails) -> plain RAG path."""
    monkeypatch.setattr(api, "_try_agent_orchestrator", lambda request, session_id: None)

    def fake_answer_question(question, category=None):
        return (
            GroundedAnswer(answer="Use the leave request form.", citations=[]),
            [],
        )

    monkeypatch.setattr(api, "answer_question", fake_answer_question)
    client = TestClient(api.app)

    response = client.post("/chat", json={"message": "How do I request vacation leave?"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Use the leave request form."
    assert body["session_id"]
    assert body["sources"] == []


def test_chat_routes_through_agent_when_available(monkeypatch):
    """src.agent.orchestrator.handle_message exists -> /chat uses the full ReAct agent."""
    scripted = orchestrator.AgentResponse(
        reply="You get 15 sick leave days per year.",
        chunks=[
            RetrievedChunk(
                chunk_id="leave-policy#003",
                text="Employees accrue 15 sick days per year.",
                similarity=0.82,
                doc_id="leave-policy",
                title="Leave Policy",
                section_path="Sick Leave",
                category="leave",
            )
        ],
    )
    monkeypatch.setattr(orchestrator, "run_turn", lambda *args, **kwargs: scripted)

    client = TestClient(api.app)
    response = client.post("/chat", json={"message": "How many sick leave days do I get?"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "You get 15 sick leave days per year."
    assert body["sources"][0]["chunk_id"] == "leave-policy#003"
    assert body["actions"] == []


def test_chat_threads_session_history_across_turns(monkeypatch, tmp_path):
    """History now comes from src/memory/ (SQLite), not an in-process dict in
    api.py — handle_message() loads/saves it internally since _try_agent_
    orchestrator no longer passes history= explicitly."""
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    seen_history = []

    def fake_run_turn(session_id, message, history=None, client=None):
        seen_history.append(history)
        return orchestrator.AgentResponse(reply=f"reply to: {message}")

    monkeypatch.setattr(orchestrator, "run_turn", fake_run_turn)
    client = TestClient(api.app)

    first = client.post("/chat", json={"session_id": "s1", "message": "first message"})
    session_id = first.json()["session_id"]
    client.post("/chat", json={"session_id": session_id, "message": "second message"})

    assert seen_history[0] == []
    assert seen_history[1] == [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "reply to: first message"},
    ]


def test_chat_reports_filed_and_escalated_complaint(monkeypatch):
    scripted = orchestrator.AgentResponse(
        reply="I've filed your complaint and HR has been notified directly.",
        ticket_id="ticket-abc",
        escalated=True,
    )
    monkeypatch.setattr(orchestrator, "run_turn", lambda *args, **kwargs: scripted)

    client = TestClient(api.app)
    response = client.post("/chat", json={"message": "my manager keeps yelling at me"})

    assert response.status_code == 200
    body = response.json()
    action = body["actions"][0]
    assert action["ticket_id"] == "ticket-abc"
    assert "escalated" in action["label"].lower()


def test_chat_includes_token_usage(monkeypatch):
    scripted = orchestrator.AgentResponse(
        reply="15 sick leave days.",
        token_usage=TokenUsage(prompt_tokens=30, completion_tokens=13, total_tokens=43),
    )
    monkeypatch.setattr(orchestrator, "run_turn", lambda *args, **kwargs: scripted)

    client = TestClient(api.app)
    response = client.post("/chat", json={"message": "how many sick leave days do I get?"})

    assert response.status_code == 200
    assert response.json()["token_usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 13,
        "total_tokens": 43,
    }


def test_usage_endpoint_reports_recorded_totals(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    usage_tracker.record_usage(
        "gemini-2.5-flash",
        TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        session_id="s1",
    )

    client = TestClient(api.app)
    response = client.get("/usage")

    assert response.status_code == 200
    body = response.json()
    assert body["today"]["request_count"] == 1
    assert body["today"]["total_tokens"] == 15
    assert body["all_time"]["total_tokens"] == 15

