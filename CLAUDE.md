# CLAUDE.md

## Project

**Onboarding Concierge — STAI100 Final Capstone** (pivoted from the Midterm's "HR FAQ & Complaint Chatbot"). An agentic system that answers onboarding FAQs via RAG (with citations) and verifies onboarding documents (NBI clearance, SSS, Pag-IBIG, BIR, PhilHealth, APE) via an OCR/CV tool, validated against deterministic checklist rules. Complaint intake/escalation was the Midterm's focus and has been **removed** from the codebase (not just descoped) — see PLAN.md §1.3.

- **Scope, plain-language:** [SCOPE.md](SCOPE.md) — what we're building and why, in/out of scope, a concrete demo walkthrough. Read this first if the idea is unclear.
- **Full plan:** [PLAN.md](PLAN.md) — architecture, data pipeline, component ownership, RRL, build order. Read it before implementing anything.
- **Governing spec:** `[Stratpoint x DLSU] Final Capstone - Project Specification.txt` (repo root) — do not edit. [specs.md](specs.md) is the superseded Midterm spec, kept for reference; also do not edit.

## Stack

- **LLM:** Google Gemini API via the `google-genai` Python SDK. Chat model is `GEMINI_CHAT_MODEL` in `src/config.py` (currently a lite tier, tuned for quota, not multimodal). Embeddings: `gemini-embedding-001` (768-dim, set `task_type` correctly: `RETRIEVAL_DOCUMENT` at ingest, `RETRIEVAL_QUERY` at query time).
- **CV/DS domain integration (mandatory component):** OCR/document field extraction via a separate `GEMINI_VISION_MODEL` (default `gemini-2.5-flash` — lite tiers aren't reliable at multimodal extraction), gated by a deterministic OpenCV quality check before any Gemini call — see PLAN.md §4.2, §5 for why this design was chosen over a standalone OCR engine.
- **Retrieval:** `RETRIEVER_MODE` selects dense-only or dense+BM25 hybrid (RRF fusion) — see PLAN.md §3.4.
- **Vector store:** ChromaDB (persistent, in `data/chroma/`, gitignored). **Structured data:** SQLite.
- **API:** FastAPI (`src/api.py`). **UI:** Streamlit (`src/ui.py`) — UI talks only to the FastAPI endpoint, never to Gemini directly.
- **Monitoring:** MLflow tracing. **Packaging:** Docker + docker-compose.

## Conventions

- Python 3.11+, dependencies pinned in `requirements.txt`.
- Secrets via `.env` (`GEMINI_API_KEY`); never commit keys. Keep `.env.example` current.
- All model/agent responses that feed downstream logic use Pydantic schemas in `src/schemas.py` with Gemini's `response_schema` — no free-text parsing.
- Model names, paths, top-k, similarity thresholds, and OCR validation thresholds live in `src/config.py`, not inline.
- The ingestion pipeline (`scripts/ingest.py`) is the only writer to `data/processed/` and the Chroma index; it must stay idempotent (re-running on unchanged docs is a no-op).
- Answers must be grounded: if retrieval returns nothing above the similarity floor, respond "I don't know" and offer HR routing — never answer HR policy questions from model memory.
- Document validation rules (type match, completeness, format validity, identity match, validity window, fail-safe) are deterministic code in `src/guardrails/doc_validation.py`, not LLM judgment — see PLAN.md §4.1. Fail toward "needs human review."
- OCR eval/demo documents must be synthetic mockups — never real government IDs or real employee data.
- Redact PII before logging anything to MLflow.
- Each component has a team-member owner (table in PLAN.md §4); coordinate before changing a component you don't own.

## Commands

```bash
pip install -r requirements.txt
python scripts/ingest.py            # rebuild knowledge base (data/raw → Chroma)
uvicorn src.api:app --reload        # API on :8000
streamlit run src/ui.py             # UI on :8501
python evals/run_retrieval_eval.py  # retrieval hit-rate on golden set
pytest tests/                       # unit tests
docker compose up --build           # full stack
```

(Directories above are the planned layout from PLAN.md §7; create them as the build progresses. OCR eval commands — `python evals/run_ocr_eval.py`, `run_validation_eval.py`, `run_answer_eval.py` — land once those harnesses exist.)
