# CLAUDE.md

## Project

**Faculty Onboarding Concierge — STAI100 Final Capstone** (pivoted from the Midterm's "HR FAQ & Complaint Chatbot"). An agentic system that answers **faculty onboarding and Faculty Manual questions** via RAG with page-level citations over the real **DLSU Faculty Manual 2021** (`data/faculty-manual-2021.pdf`, **203 pages**), and — as a secondary track — verifies a submitted **NBI Clearance** via an OCR/VLM tool validated against deterministic rules.

Built for DLSU faculty, deliberately generalizable to other PH universities: swapping in another manual is a **data change, not a code change**. Complaint intake/escalation was the Midterm's focus and has been **removed** from the codebase (not just descoped) — see PLAN.md §1.5.

- **Scope, plain-language:** [SCOPE.md](SCOPE.md) — problem statement, the three topics, in/out of scope, demo walkthrough. Read this first.
- **Full plan:** [PLAN.md](PLAN.md) — architecture, corpus, evals, component ownership, RRL, build order. Read it before implementing anything.
- **Governing spec:** `[Stratpoint x DLSU] Final Capstone - Project Specification.txt` (repo root) — do not edit. [specs.md](specs.md) is the superseded Midterm spec, kept for reference; also do not edit.

## Priority: RAG is primary, CV is secondary

The headline metrics of this project are **retrieval/answer accuracy on the Faculty Manual**, broken out per topic and per difficulty tier. OCR/validation metrics are explicitly **secondary** and are reported as two separate numbers — **real** subset and **mock** subset, never pooled. Don't let CV work crowd out RAG work; if effort has to be cut, cut the CV stretch goals (PLAN.md §4.2).

## The three topics (RAG scope boundary)

- **T1 — Pre-employment & hiring requirements for faculty.** Manual: Hiring Procedure p.24, per-rank Criteria for Hiring pp.16–23. **Verified gap:** "NBI" appears zero times in the Manual, and SSS/PhilHealth/Pag-IBIG/BIR appear only in benefits/retirement contexts. Statutory requirements are listed as submission requirements in the supplementary `dlsu-faculty-preemployment-requirements` checklist (§2, National Statutory Documents) — never write a golden-set row expecting the Manual to answer a statutory question. That checklist lists *what to submit* and *how HRMO judges acceptability*; it deliberately does **not** describe what each document is or how to obtain one — those are web-search questions, and the golden set expects abstention on them.
- **T2 — Academic & grading obligations** (use this wording, not "DOs and DON'Ts"). Manual: General Functions §1.1 p.8; Appendix F Table of Offenses p.135. **Verified gap:** pop quizzes, 4-weeks-notice for major requirements, free cut, grading curves, bonus points, and grade appeals are **not in the Faculty Manual** — they're Student Handbook / academic-policy content. Either restrict T2 to Manual-backed items or ingest that second source; decide before authoring the golden set.
- **T3 — HR policies: dress code & leaves.** Manual: Appendix D p.131; Benefits §8 Leaves pp.42–48. Strongest alternative topic if swapped: **probation, renewal & permanency** (pp.29–33).

**Authoring rule:** every golden-set row must be traceable to text actually in the ingested corpus. No page, no row.

## Corpus hazard: three parallel faculty classes

Full-time Academic Faculty, Part-time Academic Faculty, and Academic Service Faculty each have their **own** hiring procedure, leaves, and benefits — near-duplicate text with different numbers. This is the biggest retrieval hazard in the corpus. Consequences:

- Chunk metadata must carry `faculty_class` and `page_start`, and the chunk context header must include the class (`"Faculty Manual 2021 > Part-time Academic Faculty > Benefits > Leaves"`).
- When a question is class-ambiguous, the correct behaviour is a **clarifying question, not an answer** — this is scored via the `disambiguation` golden tier.

## Stack

- **LLM:** Google Gemini API via the `google-genai` Python SDK. Chat model is `GEMINI_CHAT_MODEL` in `src/config.py` (currently a lite tier, tuned for quota, not multimodal). Embeddings: `gemini-embedding-001` (768-dim; `RETRIEVAL_DOCUMENT` at ingest, `RETRIEVAL_QUERY` at query time). Ollama is a switchable fallback for chat and embeddings independently (`LLM_PROVIDER` / `EMBEDDING_PROVIDER`) — but **not** for OCR.
- **CV/DS domain integration (mandatory component):** NBI Clearance field extraction via a separate `GEMINI_VISION_MODEL` (default `gemini-2.5-flash` — lite tiers aren't reliable at multimodal extraction), gated by a deterministic OpenCV quality check before any Gemini call — see PLAN.md §4.2, §5.
- **PDF parsing is on the critical path.** `src/rag/pdf_to_md.py` is a single-column font-size heuristic and the Manual is real typeset layout with deep numbering and large appendix tables. Spot-check the parse on T1/T2/T3 sections and Appendices D and F before trusting anything downstream.
- **Retrieval:** `RETRIEVER_MODE` selects dense-only or dense+BM25 hybrid (RRF fusion) — see PLAN.md §3.4.
- **Vector store:** ChromaDB (persistent, `data/chroma/`, gitignored). **Structured data:** SQLite.
- **API:** FastAPI (`src/api.py`). **UI:** Streamlit (`src/ui.py`) — UI talks only to the FastAPI endpoint, never to Gemini directly.
- **Monitoring:** MLflow tracing. **Packaging:** Docker + docker-compose.

## Conventions

- Python 3.11+, dependencies pinned in `requirements.txt`.
- Secrets via `.env` (`GEMINI_API_KEY`); never commit keys. Keep `.env.example` current.
- All model/agent responses that feed downstream logic use Pydantic schemas in `src/schemas.py` with Gemini's `response_schema` — no free-text parsing.
- Model names, paths, top-k, similarity thresholds, and OCR/validation thresholds live in `src/config.py`, not inline. This includes `NBI_VALIDITY_MONTHS` (default 6) — the six-month window is an *employer freshness policy*, not the clearance's printed one-year validity, and keeping it in config is what makes that distinction visible.
- The ingestion pipeline (`scripts/ingest.py`) is the only writer to `data/processed/` and the Chroma index; it must stay idempotent.
- Answers must be grounded: if retrieval returns nothing above the similarity floor, respond "I don't know" and offer routing — never answer Manual questions from model memory. A confident wrong answer about someone's employment terms is worse than no answer.
- Citations must include a **page number** — a citation into a 203-page manual isn't checkable without one.
- Document validation rules (type match, completeness, format validity, identity match, validity window, fail-safe) are deterministic code in `src/guardrails/doc_validation.py`, not LLM judgment — see PLAN.md §4.1. Fail toward "needs human review."
- Each component has a team-member owner (table in PLAN.md §4 / README.md); coordinate before changing a component you don't own.

## Handling document data

- **Two subsets, always separate:** `data/references/mock/` (synthetic, used in demos and slides) and `data/references/real/` (consented real NBI Clearances and government IDs). Metrics are reported per subset, never pooled. `data/references/samples/` is a third, distinct category — specimen/demo layout-reference images, also gitignored, never treated as real-subset data (CV_INTEGRATION.md Part 5).
- `data/references/real/` is **gitignored and never committed**. Real documents require recorded consent, hashed names/reference numbers anywhere they leave the machine, EXIF stripped on upload, and are **never** shown in slides, screenshots, or recordings.
- Redact PII before logging anything to MLflow — no extracted field value ever becomes a tag or metric.

## Commands

```bash
pip install -r requirements.txt
python scripts/ingest.py            # rebuild knowledge base (faculty-manual-2021.pdf + data/raw → Chroma)
uvicorn src.api:app --reload        # API on :8000
streamlit run src/ui.py             # UI on :8501
python evals/run_retrieval_eval.py  # per-topic + per-tier hit-rate on the golden set
pytest tests/                       # unit tests
docker compose up --build           # full stack
```

(Directories above are the planned layout from PLAN.md §7; create them as the build progresses. OCR eval commands — `python evals/run_ocr_eval.py`, `run_validation_eval.py`, `run_answer_eval.py` — land once those harnesses exist.)
