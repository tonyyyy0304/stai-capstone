# HR FAQ & Complaint Chatbot — STAI100 Midterm Capstone

An agentic system that answers HR policy questions via RAG (with verified citations) and handles complaint intake via a ReAct tool-calling loop with human escalation for sensitive cases. See [PLAN.md](PLAN.md) for the full architecture and [specs.md](specs.md) for the course requirements.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # then add your GEMINI_API_KEY
python scripts/ingest.py        # build the knowledge base (data/raw -> Chroma)
```

## Commands

```bash
python scripts/ingest.py            # rebuild knowledge base (idempotent; --force to redo all)
uvicorn src.api:app --reload        # API on :8000        (Member 4)
streamlit run src/ui.py             # UI on :8501         (Member 4)
python scripts/check_member4.py     # quick API/UI/MLflow readiness check
python evals/run_retrieval_eval.py  # retrieval hit-rate@k + MRR on the golden set
pytest tests/                       # unit tests
docker compose up --build           # full stack          (Member 4)
```

## Member 4: API, UI, MLflow, Docker

`src/api.py` exposes:

- `POST /chat` with `session_id` and `message`, returning `reply`, verified `citations`, retrieved `sources`, and workflow `actions`.
- `GET /tickets/{ticket_id}` for complaint-ticket lookup once Member 2's tool writes tickets to SQLite.
- `GET /health` for demo readiness checks: Chroma index, manifest, Gemini key, and MLflow URI.

The API uses `src.monitoring.chat_trace()` to log sanitized MLflow telemetry: latency, source/citation/action counts, route, and request size. It does not log raw employee messages or model answers, because complaint text can contain PII.

`src/ui.py` is a Streamlit chat client that talks only to the FastAPI endpoint. It renders citations and retrieved policy excerpts in expanders and carries a stable session ID across turns.

Docker Compose starts three runtime services plus a first-boot ingestion job:

- API: [http://localhost:8000](http://localhost:8000)
- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- MLflow: [http://localhost:5000](http://localhost:5000)

Before running Compose, copy `.env.example` to `.env` and set `GEMINI_API_KEY`.

To check the demo surfaces quickly after startup:

```bash
python scripts/check_member4.py
```

If you only want to check that the services are reachable without calling Gemini:

```bash
python scripts/check_member4.py --skip-chat
```

## Module Ownership

| Member | Modules | Code |
| --- | --- | --- |
| Member 1 | RAG + data pipeline, Structured Outputs | `scripts/ingest.py`, `src/rag/`, `src/schemas.py`, `data/raw/`, `evals/` |
| Member 2 | ReAct Agent, Tool Use, Disambiguation | `src/agent/` |
| Member 3 | Guardrails, Memory, Prompt Engineering | `src/guardrails/`, `src/memory/`, `src/agent/prompts.py` |
| Member 4 | Chat UI, API Endpoint, MLflow, Docker | `src/api.py`, `src/ui.py`, `Dockerfile` |

## Member 1: RAG + Data Pipeline, Structured Outputs

### Data pipeline (`scripts/ingest.py`)

`data/raw/` holds 10 synthetic-but-realistic HR policy documents (leave, benefits, payroll, onboarding, handbook, remote work, grievance procedure, anti-harassment, holidays as Markdown, plus a data-privacy policy as a **PDF** to exercise the conversion path). The pipeline:

1. **Parse & normalize** — Markdown passes through. PDFs are converted to structured Markdown by `src/rag/pdf_to_md.py` (pdfplumber): headings reconstructed from font-size/bold cues, ruled tables rendered as Markdown tables, repeated page headers/footers and page numbers stripped, wrapped paragraphs re-joined. PDF/DOCX metadata comes from a sibling `<name>.meta.yaml` (hashed together with the document, so editing either triggers re-ingestion). Then unicode/whitespace normalization and hyphenation cleanup. Output: `data/processed/<doc_id>.md` with YAML frontmatter.
2. **Chunk (structure-aware)** — split on headings first (a chunk never spans two policy sections), pack to ~400 tokens with ~50-token overlap, merge sections under 80 tokens into their parent, keep tables whole, and prepend a `"{title} > {section path}"` context header to every chunk (`src/rag/chunking.py`).
3. **Enrich metadata** — `doc_id`, `chunk_id`, `title`, `section_path`, `category`, `effective_date`, `version`, `token_count`.
4. **Embed & index** — `gemini-embedding-001` (`RETRIEVAL_DOCUMENT`, 768-dim, re-normalized) upserted into a persistent Chroma collection; `data/index_manifest.json` records file hashes so re-runs are idempotent (unchanged docs are skipped, deleted docs are removed from the index).

### Retrieval & grounded answers (`src/rag/`)

Queries are embedded with `RETRIEVAL_QUERY`, top-k=8 by cosine similarity, optionally filtered by `category`. If nothing clears the similarity floor (0.5, tuned in evals) the system answers "I don't know" and offers HR routing — it never answers from model memory. `src/rag/answerer.py` produces a `GroundedAnswer` via Gemini `response_schema` and drops any citation whose `chunk_id` was not actually retrieved (grounding guardrail). Member 2's `search_kb` tool calls `answer_question()`.

### Structured Outputs (`src/schemas.py`)

Pydantic models passed to Gemini as `response_schema` — no free-text parsing: `IntentClassification` (intent, confidence, clarifying question), `GroundedAnswer` (answer + verifiable citations + insufficient-context flag), `ComplaintTicket` (category, severity, parties, description, desired outcome). Escalation for harassment/safety/legal categories is deterministic code in `src/guardrails/`, never LLM judgment.

### Evaluation (`evals/`)

`golden_set.jsonl` holds 45 golden questions with expected source doc/section, authored alongside the corpus. `run_retrieval_eval.py` reports hit-rate@{1,3,5,8} and MRR, writes a per-question report to `evals/results/`, and can log runs to MLflow (`--mlflow`) — this backs the chunking/threshold ablations in PLAN.md §7.

All tunables (models, top-k, similarity floor, chunk sizes, paths) live in `src/config.py`.
