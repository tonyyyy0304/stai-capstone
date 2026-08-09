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
                chunk_id="faculty-manual-2021#003",
                text="Full-time academic faculty accrue 15 days of sick leave per year.",
                similarity=0.82,
                doc_id="faculty-manual-2021",
                title="Faculty Manual 2021",
                section_path="Full-time Academic Faculty > Benefits > Leaves > Sick Leave (p.44)",
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
    assert body["sources"][0]["chunk_id"] == "faculty-manual-2021#003"
    assert body["actions"] == []


def test_chat_threads_session_history_across_turns(monkeypatch, tmp_path):
    """History now comes from src/memory/ (SQLite), not an in-process dict in
    api.py — handle_message() loads/saves it internally since _try_agent_
    orchestrator no longer passes history= explicitly."""
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    seen_history = []

    def fake_run_turn(session_id, message, history=None, client=None, employee_id=None):
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


# --- POST /upload-doc / GET /onboarding-status (Component 14, Phase 6) -------
# extract_document is mocked throughout (never a real Gemini call) -- the
# extraction/validation logic itself is already covered by
# tests/test_extractor.py and tests/test_doc_validation.py. These tests are
# about the HTTP wiring: request parsing, checklist persistence, and the
# cross-document sibling lookup actually getting invoked.

from src.schemas import DocType, ExtractedField, IdExtractionResult, IdType, ImageQualityReport, NbiExtractionResult, QualityVerdict  # noqa: E402


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"0" * 64


def _field(value, confidence=0.95):
    return ExtractedField(value=value, verbatim_text=value or "", model_confidence=confidence)


def _quality_pass():
    return ImageQualityReport(
        verdict=QualityVerdict.PASS, blur_score=1200.0, exposure_clip=0.01, skew_deg=1.0,
        min_dim_px=1200, quad_found=True, normalized_quality=0.95, reasons=[],
    )


def _nbi_result(**overrides):
    base = dict(
        doc_type=DocType.NBI_CLEARANCE, family_name=_field("REYES"), first_name=_field("MARIA"),
        middle_name=_field("SANTOS"), date_of_birth=_field("1990-01-01"),
        reference_no=_field("REYE900101-N00457821"), date_printed=_field("2026-02-14"),
        valid_until=_field("2027-02-14"), purpose=_field("Employment"), remarks=_field("NO DEROGATORY"),
        overall_confidence=0.95,
    )
    base.update(overrides)
    return NbiExtractionResult(**base)


def _id_result(id_type=IdType.NATIONAL_ID, **overrides):
    base = dict(
        doc_type=DocType.GOVERNMENT_ID, id_type=id_type, family_name=_field("REYES"), first_name=_field("MARIA"),
        middle_name=_field("SANTOS"), date_of_birth=_field("1990-01-01"), id_number=_field("1234-5678-9101-0001"),
        issue_date=_field("2019-06-14"), expiry_date=_field(None, 0.0), overall_confidence=0.95,
    )
    base.update(overrides)
    return IdExtractionResult(**base)


def test_upload_doc_happy_path_nbi(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    monkeypatch.setattr(api, "extract_document", lambda *a, **k: (_nbi_result(), _quality_pass()))
    client = TestClient(api.app)

    response = client.post(
        "/upload-doc",
        data={
            "employee_id": "EMP-04821", "doc_type": "nbi_clearance",
            "full_name": "REYES, MARIA SANTOS", "date_of_birth": "1990-01-01",
        },
        files={"file": ("doc.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["validation"]["outcome"] == "accepted"
    assert body["checklist"]["missing"] == ["government_id"]


def test_upload_doc_oversized_rejected_without_calling_extractor(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    calls = []
    monkeypatch.setattr(api, "extract_document", lambda *a, **k: calls.append(1) or (_nbi_result(), _quality_pass()))
    client = TestClient(api.app)

    big = b"\x89PNG\r\n\x1a\n" + b"0" * (config.MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/upload-doc",
        data={"employee_id": "EMP-1", "doc_type": "nbi_clearance", "full_name": "X", "date_of_birth": "2000-01-01"},
        files={"file": ("big.png", big, "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["validation"]["outcome"] == "rejected"
    assert calls == []  # never reached the extractor


def test_upload_doc_bad_mime_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    client = TestClient(api.app)

    response = client.post(
        "/upload-doc",
        data={"employee_id": "EMP-2", "doc_type": "nbi_clearance", "full_name": "X", "date_of_birth": "2000-01-01"},
        files={"file": ("fake.png", b"not a real image", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["validation"]["outcome"] == "rejected"


def test_upload_doc_cross_document_check_fires_and_passes_on_second_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    nbi, id_doc = _nbi_result(), _id_result()

    monkeypatch.setattr(
        api, "extract_document",
        lambda image_bytes, mime_type, expected_doc_type=None:
            (nbi, _quality_pass()) if expected_doc_type == DocType.NBI_CLEARANCE else (id_doc, _quality_pass()),
    )
    monkeypatch.setattr(
        api, "load_cached_result",
        lambda source_hash, doc_type, preprocess_flag=None:
            id_doc if doc_type == DocType.GOVERNMENT_ID else nbi,
    )
    client = TestClient(api.app)

    r1 = client.post(
        "/upload-doc",
        data={
            "employee_id": "EMP-3", "doc_type": "nbi_clearance",
            "full_name": "REYES, MARIA SANTOS", "date_of_birth": "1990-01-01",
        },
        files={"file": ("nbi.png", _png_bytes(), "image/png")},
    )
    assert "cross_document_consistency" not in [r["rule"] for r in r1.json()["validation"]["rules"]]

    r2 = client.post(
        "/upload-doc",
        data={
            "employee_id": "EMP-3", "doc_type": "government_id",
            "full_name": "REYES, MARIA SANTOS", "date_of_birth": "1990-01-01",
        },
        files={"file": ("id.png", _png_bytes(), "image/png")},
    )
    body2 = r2.json()
    assert body2["validation"]["outcome"] == "accepted"
    cross_rule = next(r for r in body2["validation"]["rules"] if r["rule"] == "cross_document_consistency")
    assert cross_rule["passed"] is True
    assert body2["checklist"]["missing"] == []


def test_upload_doc_cross_document_mismatch_escalates_to_needs_review(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    nbi = _nbi_result()
    mismatched_id = _id_result(family_name=_field("CRUZ"), first_name=_field("JUAN"), middle_name=_field(None, 0.0))

    monkeypatch.setattr(
        api, "extract_document",
        lambda image_bytes, mime_type, expected_doc_type=None:
            (nbi, _quality_pass()) if expected_doc_type == DocType.NBI_CLEARANCE else (mismatched_id, _quality_pass()),
    )
    monkeypatch.setattr(
        api, "load_cached_result",
        lambda source_hash, doc_type, preprocess_flag=None:
            mismatched_id if doc_type == DocType.GOVERNMENT_ID else nbi,
    )
    client = TestClient(api.app)

    client.post(
        "/upload-doc",
        data={
            "employee_id": "EMP-4", "doc_type": "nbi_clearance",
            "full_name": "REYES, MARIA SANTOS", "date_of_birth": "1990-01-01",
        },
        files={"file": ("nbi.png", _png_bytes(), "image/png")},
    )
    r2 = client.post(
        "/upload-doc",
        data={
            "employee_id": "EMP-4", "doc_type": "government_id",
            "full_name": "CRUZ, JUAN", "date_of_birth": "1990-01-01",
        },
        files={"file": ("id.png", _png_bytes(), "image/png")},
    )

    assert r2.json()["validation"]["outcome"] == "needs_review"


def test_upload_doc_no_sibling_no_cross_document_rule(monkeypatch, tmp_path):
    """Only one document on file -- cross_document_consistency must be
    ABSENT from rules, never present with a synthesized passed value."""
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    monkeypatch.setattr(api, "extract_document", lambda *a, **k: (_nbi_result(), _quality_pass()))
    client = TestClient(api.app)

    response = client.post(
        "/upload-doc",
        data={
            "employee_id": "EMP-5", "doc_type": "nbi_clearance",
            "full_name": "REYES, MARIA SANTOS", "date_of_birth": "1990-01-01",
        },
        files={"file": ("nbi.png", _png_bytes(), "image/png")},
    )

    assert "cross_document_consistency" not in [r["rule"] for r in response.json()["validation"]["rules"]]


def test_onboarding_status_endpoint_all_missing_before_any_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    client = TestClient(api.app)

    response = client.get("/onboarding-status/EMP-NEW")

    assert response.status_code == 200
    assert set(response.json()["missing"]) == {"nbi_clearance", "government_id"}


def test_onboarding_status_endpoint_reflects_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test_hr_agent.db")
    monkeypatch.setattr(api, "extract_document", lambda *a, **k: (_nbi_result(), _quality_pass()))
    client = TestClient(api.app)

    client.post(
        "/upload-doc",
        data={
            "employee_id": "EMP-6", "doc_type": "nbi_clearance",
            "full_name": "REYES, MARIA SANTOS", "date_of_birth": "1990-01-01",
        },
        files={"file": ("nbi.png", _png_bytes(), "image/png")},
    )

    response = client.get("/onboarding-status/EMP-6")

    assert response.status_code == 200
    assert response.json()["missing"] == ["government_id"]

