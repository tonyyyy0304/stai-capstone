# CLAUDE.md

## Project

HR FAQ & Complaint Chatbot — STAI100 Midterm Capstone. An agentic system that answers HR policy questions via RAG (with citations) and handles complaint intake via a ReAct tool-calling loop with human escalation for sensitive cases.

- **Full plan:** [PLAN.md](PLAN.md) — architecture, data preprocessing pipeline, module ownership, build order. Read it before implementing anything.
- **Course spec:** [specs.md](specs.md) — grading requirements. Do not edit.

## Stack

- **LLM:** Google Gemini API via the `google-genai` Python SDK. Chat: `gemini-2.5-flash`. Embeddings: `gemini-embedding-001` (768-dim, set `task_type` correctly: `RETRIEVAL_DOCUMENT` at ingest, `RETRIEVAL_QUERY` at query time).
- **Vector store:** ChromaDB (persistent, in `data/chroma/`, gitignored). **Structured data:** SQLite.
- **API:** FastAPI (`src/api.py`). **UI:** Streamlit (`src/ui.py`) — UI talks only to the FastAPI endpoint, never to Gemini directly.
- **Monitoring:** MLflow tracing. **Packaging:** Docker + docker-compose.

## Conventions

- Python 3.11+, dependencies pinned in `requirements.txt`.
- Secrets via `.env` (`GEMINI_API_KEY`); never commit keys. Keep `.env.example` current.
- All model/agent responses that feed downstream logic use Pydantic schemas in `src/schemas.py` with Gemini's `response_schema` — no free-text parsing.
- Model names, paths, top-k, similarity thresholds live in `src/config.py`, not inline.
- The ingestion pipeline (`scripts/ingest.py`) is the only writer to `data/processed/` and the Chroma index; it must stay idempotent (re-running on unchanged docs is a no-op).
- Answers must be grounded: if retrieval returns nothing above the similarity floor, respond "I don't know" and offer HR routing — never answer HR policy questions from model memory.
- Escalation rules for harassment/safety/legal complaints are deterministic code in `src/guardrails/`, not LLM judgment.
- Redact PII before logging anything to MLflow.
- Each module has a team-member owner (table in PLAN.md §4); coordinate before changing a module you don't own.

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

(Directories above are the planned layout from PLAN.md §5; create them as the build progresses.)
