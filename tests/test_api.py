from fastapi.testclient import TestClient

from src import api
from src.schemas import GroundedAnswer


def test_health_degraded_without_runtime_state(monkeypatch, tmp_path):
    monkeypatch.setattr(api.config, "CHROMA_DIR", tmp_path / "missing_chroma")
    monkeypatch.setattr(api.config, "MANIFEST_PATH", tmp_path / "missing_manifest.json")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_chat_returns_rag_response(monkeypatch):
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

