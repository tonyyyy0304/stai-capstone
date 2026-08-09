# STATE.md — Current State of the Repo

> **Maintenance clause (READ THIS):** This file is the living snapshot of what is *actually built* vs. what is *still planned*. **Whenever you change what the code does — add, remove, or materially alter a feature, endpoint, module, config flag, or eval — update this file in the same change.** Keep the two lists honest: move an item from **Planned** to **Implemented** only when it exists and runs; delete or correct entries that no longer match the code. If a change makes a claim here stale, fixing STATE.md is part of that change, not a follow-up.
>
> Scope note: this file lists **implemented features and plans only** — no architecture rationale (that's [PLAN.md](PLAN.md)) and no problem framing (that's [SCOPE.md](SCOPE.md)).

_Last updated: 2026-08-09 · branch: `final-capstone`_

---

## What the project is (one line)

Agentic RAG that answers **DLSU Faculty Manual 2021** onboarding questions with page-level citations (primary), with a secondary **NBI Clearance** verification track. Retargetable to another org by config + data swap, not code.

---

## Implemented features

### RAG pipeline (primary)
- **Ingestion** (`scripts/ingest.py`) — idempotent build of the Chroma index from `data/faculty-manual-2021.pdf` + `data/raw/`. Only writer to `data/processed/` and the index.
- **PDF → Markdown** (`src/rag/pdf_to_md.py`) — font-size heuristic parse.
- **Chunking** (`src/rag/chunking.py`) — section-aware, carries `page_start`, `category`, and `audience_class` metadata; context header includes org > audience > section path.
- **Embeddings** (`src/rag/embeddings.py`) — Gemini `gemini-embedding-001` (768-dim, `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY`); Ollama embedder as a switchable fallback (`EMBEDDING_PROVIDER`).
- **Retrieval** (`src/rag/retriever.py`, `src/rag/hybrid.py`) — `RETRIEVER_MODE` selects **dense-only** (default) or **dense + BM25 (FTS5) with RRF fusion**. Soft category boost and audience-class boost re-rank ordering only; similarity floor sees true cosine.
- **Grounded answering** (`src/rag/answerer.py`) — single grounded-answer call over the deduped union of retrieved chunks; abstains (`insufficient_context`) when excerpts don't contain the fact; page-level citations; inline chunk citations removed from answer text.
- **Abstention** — similarity floor (`SIMILARITY_FLOOR=0.55`) + generation-layer faithfulness abstention; "I don't know" + routing fallback.

### Agent (ReAct)
- **Orchestrator** (`src/agent/orchestrator.py`) — `run_turn()` core; `handle_message()` adapts to the API contract and wires in memory.
- **Model-driven ReAct loop** — plans retrieval only (`search_kb` / `search_web` / `finish`), bounded by `MAX_REACT_ITERATIONS=5`; decomposes multi-part questions and reformulates weak queries; one grounded answer synthesized at the end.
- **Intent router** (`src/agent/router.py`) — classifies intent, gates ambiguous/low-confidence input to a clarifying question, declines out-of-scope.
- **Audience-class disambiguation** — asks which faculty class (full-time / part-time / ASF) when strong evidence spans >1 class (`DISAMBIG_TOP_N`).
- **Web search fallback** (`src/agent/tools.py`) — Tavily, domain-restricted to statutory `.gov.ph` sources, gated by `ENABLE_WEB_FALLBACK`.
- **Usage tracking** (`src/agent/usage.py`) — per-turn/per-model token accounting logged to SQLite.
- **Ollama backend** (`src/agent/llm_client.py`) — switchable chat backend via `LLM_PROVIDER`. Gemini internal thinking disabled by default (`GEMINI_THINKING_BUDGET=0`).

### Guardrails (`src/guardrails/`)
- **Pre-router**: deterministic prompt-injection regex + toxicity wordlist (short-circuit, no LLM cost).
- **Post-router semantic**: injection / jailbreak / toxicity re-check reusing the router's LLM call signals.
- **Output**: grounding check hard-verifies every citation maps to a chunk retrieved this turn.
- **PII** (`pii.py`), **LLM-as-judge** (`llm_judge.py`, off by default via `ENABLE_LLM_JUDGE`).

### Memory (`src/memory/`)
- **Session memory** (`session.py`) — full history persisted to SQLite.
- **Persistent memory** (`persistent.py`) — trimmed recent turns + rolling summary; integrated into the ReAct turn via `handle_message()`.

### API / UI / Monitoring
- **FastAPI** (`src/api.py`): `POST /chat`, `GET /usage`, `GET /health`.
- **Streamlit UI** (`src/ui.py`): talks only to the API, never Gemini directly.
- **MLflow tracing** (`src/monitoring.py`): each chat turn is a trace; **each ReAct step is logged**; sanitized telemetry only (no raw messages, answers, or field values).

### Config & deployment
- **Central config** (`src/config.py`): all models, paths, thresholds, and a **deployment profile** (org/corpus/audience specifics externalized) — retarget by config + data.
- **Docker**: `Dockerfile.api`, `Dockerfile.ui`, `Dockerfile.mlflow`, `docker-compose.yml`; Render start scripts under `scripts/`.

### Evals & tests
- **Golden set** (`evals/golden_set.jsonl`, 32 rows) — per-topic + per-tier.
- **Retrieval eval** (`evals/run_retrieval_eval.py`) — per-topic/per-tier hit-rate.
- **Guardrail eval** (`evals/run_guardrail_eval.py`) + red-team set (`evals/guardrail_redteam.jsonl`).
- **Unit tests** (`tests/`, 14 files) — api, chunking, guardrails, hybrid, ingest, llm_client, memory, orchestrator, pdf_to_md, retrieval, router, schemas, tools, usage.

---

## Planned / not yet implemented

### NBI Clearance verification (secondary track — CV/DS mandatory component)
- `src/ocr/` — OpenCV deterministic quality gate + one Gemini multimodal field-extraction call.
- `src/guardrails/doc_validation.py` — deterministic validation rules 1–6 (type match, completeness, format validity, identity match, validity window, fail-safe). Fails toward "needs human review".
- API: `POST /upload-doc` (NBI submission), `GET /onboarding-status/{employee_id}` (per-hire checklist state).
- Memory: onboarding-status / checklist state in SQLite.
- Evals: `evals/run_ocr_eval.py`, `run_validation_eval.py` — reported per subset (**real** vs **mock**, never pooled).

### End-to-end / answer evals
- `evals/run_answer_eval.py` — end-to-end answer accuracy harness.

### Removed from prior scope (do not reintroduce)
Complaint intake and escalation (`danger_scan`, `escalation`, `escalation_state`, `form_pii`) were **deleted** in the pivot from the Midterm chatbot — see PLAN.md §1.5.

---

## Pointers
- Plain-language scope: [SCOPE.md](SCOPE.md)
- Full architecture & build order: [PLAN.md](PLAN.md)
- Project instructions: [CLAUDE.md](CLAUDE.md)
- Component ownership table: [README.md](README.md) / PLAN.md §4
