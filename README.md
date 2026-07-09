# HR FAQ & Complaint Chatbot — STAI100 Midterm Capstone

An agentic system that answers HR policy questions via RAG (with verified citations) and handles complaint intake via a ReAct tool-calling loop with human escalation for sensitive cases. See [PLAN.md](PLAN.md) for the full architecture (including the LLM/embedding provider abstraction, §2.1) and [specs.md](specs.md) for the course requirements.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              Streamlit Chat UI          │
                        └───────────────────┬─────────────────────┘
                                            │ HTTP
                        ┌───────────────────▼─────────────────────┐
                        │            FastAPI  (/chat, /health)     │
                        └───────────────────┬─────────────────────┘
                                            │
        ┌───────────────────────────────────▼───────────────────────────────────┐
        │                          Agent Orchestrator (ReAct loop)              │
        │  1. Input guardrails: deterministic + LLM-judge (5 dimensions)        │
        │  2. Intent router: FAQ | Complaint | Ambiguous → disambiguate         │
        │  3. Tool selection & execution                                        │
        │  4. Output guardrails (grounding check, PII redaction, tone)          │
        └──────┬───────────────┬────────────────┬──────────────┬────────────────┘
               │               │                │              │
        ┌──────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐ ┌────▼─────────┐
        │  RAG tool   │ │ Complaint    │ │  Escalation  │ │   Memory     │
        │  (Chroma +  │ │ filing tool  │ │  tool (email │ │  session +   │
        │  switchable │ │ (SQLite      │ │  /flag to    │ │  persistent  │
        │  embeddings)│ │  tickets)    │ │  HR human)   │ │  (SQLite)    │
        └─────────────┘ └──────────────┘ └──────────────┘ └──────────────┘

        Observability: MLflow tracing on every request (latency, tokens, tool calls, errors)
```

Chat/reasoning and embeddings each independently point at Gemini (default) or a self-hosted Ollama server via `LLM_PROVIDER`/`EMBEDDING_PROVIDER` — see [Setup](#setup) and [PLAN.md §2.1](PLAN.md).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # then add your GEMINI_API_KEY and TAVILY_API_KEY
python scripts/ingest.py        # build the knowledge base (data/raw -> Chroma)
```

`GEMINI_API_KEY` (chat + embeddings by default) and `TAVILY_API_KEY` (web search fallback for DOLE/labor-law questions) are both required. See `.env.example` for the full list.

### Optional: switch chat and/or embeddings to a self-hosted Ollama server

Both are independently switchable — set either or both:

```bash
# In .env
LLM_PROVIDER=ollama                          # default: gemini
OLLAMA_URL=http://<your-ollama-host>:11434
OLLAMA_CHAT_MODEL=gemma4:e4b                 # default

EMBEDDING_PROVIDER=ollama                    # default: gemini
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
```

Why: Gemini's free tier caps `gemini-2.5-flash` at 20 requests/day, and a single agent turn can cost several requests — Ollama removes that ceiling entirely if both vars are set. See [PLAN.md §2.1](PLAN.md) for the full design and known limitations (smaller local models are less reliable at strict tool-calling/schema conformance than Gemini).

**If you change `EMBEDDING_PROVIDER`, you must re-ingest against a cleared index** — Gemini and Ollama embeddings are different, incompatible vector spaces (both happen to be 768-dim, which hides the mismatch instead of erroring on it):

```bash
rm -rf data/chroma data/index_manifest.json
python scripts/ingest.py
```

**Don't mix local and Docker ingestion against the same index.** Running `scripts/ingest.py` on your host and `docker compose up` in containers against the same bind-mounted `data/chroma/` will crash with a chromadb Rust panic — the two platforms' chromadb builds aren't binary-compatible with the same on-disk index. Pick one environment per index; clear and re-ingest if you switch.

## Commands

```bash
python scripts/ingest.py            # rebuild knowledge base (idempotent; --force to redo all)
uvicorn src.api:app --reload        # API on :8000        (Member 4)
streamlit run src/ui.py             # UI on :8501         (Member 4)
python scripts/check_member4.py     # quick API/UI/MLflow readiness check
python evals/run_retrieval_eval.py  # retrieval hit-rate@k + MRR on the golden set
python evals/run_guardrail_eval.py  # guardrail red-team: injection/toxicity block rate, PII detection rate
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

Before running Compose, copy `.env.example` to `.env` and set `GEMINI_API_KEY` and `TAVILY_API_KEY` (`LLM_PROVIDER`/`OLLAMA_URL`/`EMBEDDING_PROVIDER` are optional — see [Setup](#setup)). Compose picks up `.env` via `env_file:` for every service.

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
| Member 2 | ReAct Agent, Tool Use, Disambiguation, LLM/embedding provider abstraction | `src/agent/orchestrator.py`, `router.py`, `tools.py`, `usage.py`, `llm_client.py` |
| Member 3 | Guardrails, Memory, Prompt Engineering | `src/guardrails/`, `src/memory/`, `src/agent/prompts.py` |
| Member 4 | Chat UI, API Endpoint, MLflow, Docker | `src/api.py`, `src/ui.py`, `Dockerfile` |

## Member 1: RAG + Data Pipeline, Structured Outputs

### Data pipeline (`scripts/ingest.py`)

`data/raw/` holds 10 synthetic-but-realistic HR policy documents (leave, benefits, payroll, onboarding, handbook, remote work, grievance procedure, anti-harassment, holidays as Markdown, plus a data-privacy policy as a **PDF** to exercise the conversion path). The pipeline:

1. **Parse & normalize** — Markdown passes through. PDFs are converted to structured Markdown by `src/rag/pdf_to_md.py` (pdfplumber): headings reconstructed from font-size/bold cues, ruled tables rendered as Markdown tables, repeated page headers/footers and page numbers stripped, wrapped paragraphs re-joined. PDF/DOCX metadata comes from a sibling `<name>.meta.yaml` (hashed together with the document, so editing either triggers re-ingestion). Then unicode/whitespace normalization and hyphenation cleanup. Output: `data/processed/<doc_id>.md` with YAML frontmatter.
2. **Chunk (structure-aware)** — split on headings first (a chunk never spans two policy sections), pack to ~400 tokens with ~50-token overlap, merge sections under 80 tokens into their parent, keep tables whole, and prepend a `"{title} > {section path}"` context header to every chunk (`src/rag/chunking.py`).
3. **Enrich metadata** — `doc_id`, `chunk_id`, `title`, `section_path`, `category`, `effective_date`, `version`, `token_count`.
4. **Embed & index** — via `config.get_embedder()`: Gemini's `gemini-embedding-001` (`RETRIEVAL_DOCUMENT`, 768-dim, re-normalized, default) or Ollama's `nomic-embed-text` depending on `EMBEDDING_PROVIDER`, upserted into a persistent Chroma collection; `data/index_manifest.json` records file hashes so re-runs are idempotent (unchanged docs are skipped, deleted docs are removed from the index). Switching `EMBEDDING_PROVIDER` requires clearing the index and re-ingesting — see [Setup](#setup).

### Retrieval & grounded answers (`src/rag/`)

Queries are embedded with `RETRIEVAL_QUERY`, top-k=8 by cosine similarity, optionally filtered by `category`. If nothing clears the similarity floor (0.5, tuned in evals) the system answers "I don't know" and offers HR routing — it never answers from model memory. `src/rag/answerer.py` produces a `GroundedAnswer` via `response_schema` (Gemini or Ollama, per `LLM_PROVIDER`) and drops any citation whose `chunk_id` was not actually retrieved (grounding guardrail); it also withholds the retrieved chunks from the response when the answer is insufficient, so the UI never shows "possibly relevant" excerpts next to an "I don't know" reply. Member 2's `search_kb` tool calls `answer_question()`.

### Structured Outputs (`src/schemas.py`)

Pydantic models passed as `response_schema` (Gemini or Ollama, per `LLM_PROVIDER`) — no free-text parsing: `IntentClassification` (intent, confidence, clarifying question), `GroundedAnswer` (answer + verifiable citations + insufficient-context flag), `ComplaintTicket` (category, severity, parties, description, desired outcome). Escalation for harassment/safety/legal categories is deterministic code in `src/guardrails/`, never LLM judgment.

### Evaluation (`evals/`)

`golden_set.jsonl` holds 45 golden questions with expected source doc/section, authored alongside the corpus. `run_retrieval_eval.py` reports hit-rate@{1,3,5,8} and MRR, writes a per-question report to `evals/results/`, and can log runs to MLflow (`--mlflow`) — this backs the chunking/threshold ablations in PLAN.md §7.

All tunables (models, top-k, similarity floor, chunk sizes, paths) live in `src/config.py`.

## Member 2: ReAct Agent, Tool Use, Disambiguation

### Disambiguation (`src/agent/router.py`)

One structured call (`response_schema=IntentClassification`) per turn classifies `faq | complaint | ambiguous | out_of_scope` with a confidence score. Below `ROUTER_CONFIDENCE_FLOOR` (0.6) or explicitly ambiguous, the router returns its own `clarifying_question` and the turn ends there — no tool ever runs on a guess. Out-of-scope input is declined the same way. An unparseable model response fails closed to `AMBIGUOUS` rather than crashing.

### ReAct Agent (`src/agent/orchestrator.py`)

A real Gemini/Ollama function-calling loop, not a hardcoded decision tree: the model picks a tool, `tools.py` executes it, the result is fed back as an observation, and the model repeats until it answers in plain text or hits `MAX_REACT_ITERATIONS` (5). `run_turn()` wraps the whole loop in a `try/except (APIError, LLMBackendError)` so a rate limit, outage, or Ollama connectivity failure degrades to a plain-language retry message instead of a 500. `handle_message()` adapts this to `src/api.py`'s `ChatResponse` contract.

### Tool Use (`src/agent/tools.py`)

Four model-callable tools: `search_kb` (internal RAG via Member 1's `answer_question()`), `search_web` (DOLE/labor-law fallback — Tavily search domain-restricted to `dole.gov.ph`/`officialgazette.gov.ph`/`lawphil.net`, then one `response_schema` call to shape results into a `GroundedAnswer`), `file_complaint`, and `get_ticket_status`. `escalate_to_hr` is deliberately **not** model-callable: `should_escalate(category)` is a plain deterministic function (harassment/discrimination/safety/legal → escalate), so escalation is never an LLM decision, only its result is fed back to the model so it can tell the employee HR was notified.

### LLM & embedding provider abstraction (`src/agent/llm_client.py`, `src/config.py`, `src/rag/embeddings.py`)

Both the chat/reasoning model and the embedding model are switchable independently via `LLM_PROVIDER` and `EMBEDDING_PROVIDER` (`gemini` default for both, or `ollama` for a self-hosted server — see [Setup](#setup)). `OllamaClient` mimics `google-genai`'s exact `client.models.generate_content(model, contents, config)` call signature and response shape, so the router, orchestrator, and tools needed almost no changes to support it. `src/agent/usage.py` logs every call's token usage to SQLite tagged by which model actually served it; `GET /usage` reports today's and all-time totals. Full design, verification notes, and known limitations in [PLAN.md §2.1](PLAN.md).

## Member 3: Escalation Subsystem

### Guardrails (`src/guardrails/escalation.py`, `danger_scan.py`, `form_pii.py`)

`should_escalate(ticket, raw_text)` is a deterministic function improved by an LLM judgement call, that decides whether a filed complaint reaches a human. It checks, in priority order: a danger-scan match on the employee's raw words (`danger_scan.py`, checked first so it overrides category/severity), a mandatory-escalation category (harassment/discrimination/safety/legal), a retaliation-language floor, then a plain severity threshold as well as with an LLM judgement call. If the ReAct loop exhausts `MAX_REACT_ITERATIONS` mid-complaint without completing a valid ticket, a fail-safe rule still files and escalates a best-effort ticket — a stuck loop never silently drops a report.

`form_pii.py` builds the only object allowed to leave this boundary: a redacted `EscalationEvent` (category/severity/trigger/SLA deadline, no free text). The full `ComplaintTicket` never reaches a log line or an MLflow tag.

### Structured Outputs (`src/schemas.py`)

`EscalationDecision`, `EscalationEvent`, `EscalationFormSubmission`, and `EscalationFlowState` are typed Pydantic models — the rule engine's output, the redacted record, the intake-form payload, and the per-session state are never free text the model could hallucinate or a dict with unchecked keys.

### ReAct Agent State Handling (`src/agent/orchestrator.py`)

Before a complaint reaches the model's tool-calling loop at all, `escalation_state.py` (SQLite-backed, mirrors the memory module's pattern) tracks whether a session is mid consent-gate or mid-form. While in either state, the ReAct loop is bypassed entirely — replies are generated by deterministic text parsing, not by handing the model another turn. This is what makes the consent question ("file a form, or escalate to HR now?") actually binding rather than something the model could talk its way around.

### Tool Use (`src/agent/tools.py`)

`escalate_to_hr()` sends an HTML email (plain-text fallback included) built from the full `ComplaintTicket` — description, parties involved, incident date, desired outcome — so HR gets everything needed to act. If `SMTP_HOST`/`HR_ESCALATION_EMAIL_TO` aren't set, it falls back automatically to `data/escalation_outbox.jsonl`, so the whole path runs with zero mail-server config during local testing.

### Evaluation & Setup

`python evals/run_escalation_eval.py` runs 24 adversarial prompts against the rule engine (paraphrased-danger false negatives, idiom false positives, category-ambiguity edge cases) and reports a real pass rate with every miss printed, not padded to look clean.

`pytest tests/test_guardrails.py tests/test_orchestrator.py` covers the full rule matrix and the consent/form state machine.

**Setup:** No extra keys required — SMTP is optional in `.env.example`; unset, escalation still works end-to-end via the mock outbox.
