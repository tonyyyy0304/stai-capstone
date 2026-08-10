# STATE.md — Current State of the Repo

> **Maintenance clause (READ THIS):** This file is the living snapshot of what is *actually built* vs. what is *still planned*. **Whenever you change what the code does — add, remove, or materially alter a feature, endpoint, module, config flag, or eval — update this file in the same change.** Keep the two lists honest: move an item from **Planned** to **Implemented** only when it exists and runs; delete or correct entries that no longer match the code. If a change makes a claim here stale, fixing STATE.md is part of that change, not a follow-up.
>
> Scope note: this file lists **implemented features and plans only** — no architecture rationale (that's [PLAN.md](PLAN.md)) and no problem framing (that's [SCOPE.md](SCOPE.md)).

_Last updated: 2026-08-10 · branch: `feat/final-capstone-cv`_

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

### CV/DS: NBI Clearance + Government ID verification (secondary track — Component 14)
Full design doc: [CV_INTEGRATION.md](CV_INTEGRATION.md). All 9 phases done (0–7 plus the optional 4a Ollama fallback) — backend/API/agent/evals/UI all built and **live-verified end to end** (real NBI clearance uploaded via the running Docker stack, correctly flagged/accepted, layout and styling confirmed working). RAG remains primary per CLAUDE.md's priority.
- **Config/schemas** (`src/config.py` CV/OCR block, `src/schemas.py`) — vision provider/model selection (Gemini primary, Ollama switchable fallback), quality thresholds, `DocType`/`IdType`/`ImageQualityReport`/`ExtractedField`/`NbiExtractionResult`/`IdExtractionResult`/`ValidationResult`/`ChecklistStatus`.
- **Mock dataset** (`data/references/mock/`) — 100 images, 2 document types (NBI Clearance + Government ID covering National ID/Driver's License/Passport), 5 degradation variants × 16 identities + 26 negatives (including cross-document mismatch fixtures); `data/references/real/` (gitignored, consented) and `data/references/samples/` (gitignored, layout reference) are separate, never-pooled subsets. Now 3 real specimens.
- **Layer 1 — quality gate** (`src/ocr/quality.py`) — OpenCV deterministic blur/skew/exposure/resolution check; a `reject` verdict short-circuits before any API call (the quota mechanism). `config.QUAD_NOT_FOUND_QUALITY_SCORE=0.75` (recalibrated 2026-08-10 from a hardcoded 0.5 — see below).
- **Layer 2 — extraction** (`src/ocr/extractor.py`) — one Gemini multimodal call per document, SHA-256-keyed disk cache (`evals/results/ocr_cache/`), one conditional retry on low-confidence fields (`OCR_ENABLE_RETRY`); `load_cached_result()` for a hash-only cache lookup (no image bytes needed) used by the cross-document sibling check. Detection only — never decides accept/reject.
- **Document registry** (`src/ocr/doctypes.py`) — declarative field/validator spec per doc type, shared by extraction prompts and validation.
- **Layer 3 — validation** (`src/guardrails/doc_validation.py`) — `validate_document()` (NBI) / `validate_id_document()` (Government ID), same six-rule shape (type match, completeness, format, identity, validity window, fail-safe); `validate_cross_document()` (NBI↔ID name/DOB consistency); `apply_cross_document_result()` folds a cross-check into an already-computed `ValidationResult`, escalating `accepted`→`needs_review` on mismatch, never downgrading further or upgrading a result that failed on its own merits. Rule 4 (identity) and Rule 6 (fail-safe) structurally can only ever escalate to `needs_review`, never auto-reject.
- **Checklist state** (`src/memory/onboarding_status.py`) — durable per-employee `(doc_type → status)` in SQLite; never stores an extracted field value, only status/outcome/source_hash.
- **Agent wiring** (`src/agent/orchestrator.py`, `src/agent/prompts.py`) — `Intent.DOCUMENT_UPLOAD`/`DOCUMENT_STATUS` short-circuit in `run_turn()` before `_react_loop()` (same pattern as `OUT_OF_SCOPE`), not new ReAct tools — the orchestrator turned out to be a closed-enum ReAct loop, not Gemini function-calling, so there was nothing to register a tool into. `ROUTER_PROMPT` now disambiguates "what documents do I need?" (stays `faq`) from "what's my document's status?" (`document_status`), per PLAN.md §4.4. Live-verified 8/8 against real Gemini.
- **API** (`src/api.py`) — `POST /upload-doc` (multipart: file + employee_id + doc_type + full_name + date_of_birth — no HR record source exists in this codebase, so identity is uploader-supplied, not looked up) runs quality→extract→validate→record→cross-check synchronously, outside the chat/ReAct path entirely; `GET /onboarding-status/{employee_id}`.
- **Monitoring** (`src/monitoring.py`) — `doc_trace()` sibling to `chat_trace()`; tag/metric allowlists extended (`doc_type`/`validation_outcome`/`quality_verdict` tags, `blur_score`/`skew_deg`/`extraction_confidence`/`fields_extracted`/`fields_missing`/`ocr_latency_ms` metrics), no field value ever added. `doc_trace()` deliberately does NOT take a session_id/employee_id (removed 2026-08-10) — an earlier version tagged uploads with the applicant's real employee_id, a stable per-person identifier unlike chat's random UUID; `onboarding_status.py`'s SQLite table is already the authoritative per-employee record, so MLflow didn't need to duplicate that correlation.
- **UI** (`src/ui.py`) — sidebar is checklist-only now (Employee ID field + Document Checklist card, color-coded status pills, refetched every render). The upload flow moved to the **main chat column** (not the sidebar) per user feedback, wrapped in `st.spinner()` during the `/upload-doc` call so a multi-second Gemini vision call doesn't look like the UI froze. Result banner still deferred to the next render (session_state + `st.rerun()`) so it survives that rerun, which is also what refreshes the checklist without a manual reload. Fixed a layout bug found via screenshot: `stMain` (flex sibling of `stSidebar`) was missing `min-width: 0`, so it wouldn't shrink to fit beside the sidebar and overflowed the viewport. **Live-verified in the running Docker stack** — layout, styled banners (accepted/needs_review colors), and checklist all confirmed working via real uploads.
- **Evals** (`evals/run_ocr_eval.py`, `evals/run_validation_eval.py`) — per-field exact-match + normalized CER (OCR) and precision/recall/F1 + false-auto-pass rate + cross-document trigger rate (validation), both reported per `doc_type`, never pooled; `run_validation_eval.py` runs entirely off cached extractions, genuinely zero API cost. Already found and fixed a real bug: `scripts/make_onboarding_docs.py`'s `wrong_person` negative fixture never actually swapped in the impostor's name (wrong dict key overridden), so it silently tested nothing — both fixtures now genuinely mismatch, re-verified at 0% false-auto-pass.
- **Ollama vision fallback** (`src/agent/llm_client.py`) — `OllamaClient` now handles `extract_document()`'s actual call shape (a flat `[Part(image), prompt]` list, not the `[Content(...), ...]` shape every other call site uses), base64-encoding the image onto Ollama's `images` field. Live-verified against a real local Ollama server (no vision-capable model pulled, so only the transport/fail-safe path was confirmed live, not full accuracy): a real HTTP 400 from a non-multimodal model correctly surfaced as `LLMBackendError` → `extract_document()` returned `(None, report)`.
- **`quad_found` quality penalty recalibrated** (`src/ocr/quality.py`, `config.QUAD_NOT_FOUND_QUALITY_SCORE`) — was a hardcoded `0.5` inline, never checked against real data. By the time a 3rd real specimen arrived, **all 3 real specimens tested had hit `quad_found=False`** (real photos never produce the clean rectangular contour synthetic mock renders trivially do), and since `normalized_quality = min(scores)`, that `0.5` unconditionally capped every real submission's composite confidence below `OCR_CONFIDENCE_FLOOR=0.70` regardless of extraction quality — a genuinely valid, 0.99-confidence real clearance (`real3.jpg`) landed on `needs_review` purely because of this one signal, having passed all 5 other rules cleanly. Softened to `0.75` (config constant, not inline) — re-verified: all 3 real specimens now read `normalized_quality=0.75`, and `real3.jpg` now reaches `accepted`.
- **Embedding rate-limit retry** (`src/rag/embeddings.py`) — `GeminiEmbedder` now retries on HTTP 429 with exponential backoff (10s→320s, 6 attempts) instead of crashing `ingest.py` mid-run with no progress saved. Found live: `embed_content`'s free-tier quota is per-minute, and a corpus needing many re-embedded batches in one ingest run could exceed it with zero retry logic previously in place.
- **Docker mlflow Host-header fix** (`docker-compose.yml`) — mlflow 3.x's DNS-rebinding protection rejects any request whose `Host` header isn't allowlisted; the `api` container talks to `mlflow` over the compose network as `http://mlflow:5000`, which wasn't allowlisted by default, breaking `configure_mlflow()` on every API startup. Fixed with `--allowed-hosts "mlflow:*,localhost:*,localhost"` — verified against a real running `mlflow server` (correct host passes, `evil.com` still correctly rejected).

### Evals & tests
- **Golden set** (`evals/golden_set.jsonl`, 32 rows) — per-topic + per-tier.
- **Retrieval eval** (`evals/run_retrieval_eval.py`) — per-topic/per-tier hit-rate.
- **Guardrail eval** (`evals/run_guardrail_eval.py`) + red-team set (`evals/guardrail_redteam.jsonl`).
- **Unit tests** (`tests/`, 20 files, 345 passed / 1 skipped) — api, chunking, doc_validation, doctypes, embeddings, extractor, guardrails, hybrid, ingest, llm_client, memory, onboarding_status, orchestrator, pdf_to_md, quality, retrieval, router, schemas, tools, usage.

---

## Planned / not yet implemented

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
