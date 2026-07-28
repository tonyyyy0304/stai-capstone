# Onboarding Concierge — Final Capstone Implementation Plan

**Course:** Introduction to Agentic AI (STAI100) — Final Capstone (Week 14)
**Use case type:** Vector DB RAG + CV/OCR Document Understanding (HR/Onboarding domain)
**LLM provider:** Google Gemini API (chat + embeddings), with a self-hosted Ollama server as a fully switchable fallback for both, independently, via `LLM_PROVIDER` / `EMBEDDING_PROVIDER` — see §2.1. OCR/document extraction (new, Final-only) is Gemini-multimodal specifically; see §5 RRL.
**Builds on:** the Midterm project ("HR FAQ & Complaint Chatbot"). Reused as-is: the RAG pipeline, the agent orchestrator (ReAct loop + tools), two-tier memory, guardrails (deterministic checks + LLM-judge), the LLM/embedding provider abstraction, and the API/UI/MLflow/Docker scaffolding — all of this is already built and tested (§2.1, §4). Complaint intake & escalation (the form-driven consent-gate design, `src/guardrails/escalation.py`, `danger_scan.py`, `form_pii.py`, `escalation_state.py`, and the `ComplaintTicket`/`Severity`/`TriggerRule`/`Escalation*` schemas) have been **removed from the codebase** — not just descoped — in favor of the onboarding-document-verification direction (see §1.3).
**Governing spec:** `[Stratpoint x DLSU] Final Capstone - Project Specification.txt` (repo root) — supersedes the Midterm spec ([specs.md](specs.md), kept for reference; do not edit either spec file).

---

## 1. Business Use Case

### 1.1 The narrowed problem

New hires don't just have policy questions — they also have to *produce paperwork* (NBI clearance, SSS, Pag-IBIG, BIR 2316, PhilHealth, APE/medical certificate) before HR can clear them to start. Today this is two disconnected manual loops: an HR generalist answers the same "what do I need for Day 1?" questions over chat/email, and separately eyeballs scanned documents to check they're the right type, legible, not expired, and belong to the right person.

**Final Capstone scope — "Onboarding Concierge":** a single conversational agent that (a) answers onboarding FAQs grounded in the company handbook, and (b) accepts a photo/scan of a required onboarding document, extracts its fields via OCR, and validates it against a deterministic checklist — telling the employee in the same conversation what's missing, invalid, or needs human review.

### 1.2 Sanity check (spec §5)

*Could ChatGPT/Claude/a Google search alone suffice?* No — the agent needs to (1) retrieve grounded, company-specific policy rather than generic advice, (2) run OCR + deterministic validation rules no general chatbot has (name-match against the employee record, PH-specific expiry rules, required-field checks per document type), and (3) hold per-employee state across turns ("2 of 5 docs received, missing Pag-IBIG MDF") — none of which a single stateless prompt can do.

### 1.3 What's explicitly out of scope for the Final's graded story

- **Complaint intake, harassment/safety escalation, human-in-the-loop ticket routing.** This was the Midterm's product — the form-driven consent-gate design (`src/guardrails/escalation.py`, `danger_scan.py`, `form_pii.py`, `escalation_state.py`), the `ComplaintTicket`/`Severity`/`TriggerRule`/`Escalation*` schemas, the consent-gate flow in `orchestrator.py`/`ui.py`, the `/tickets/{id}` endpoint, and `evals/run_escalation_eval.py` were built and tested at one point, but have since been **removed from the codebase entirely** (not just descoped from the demo). Rationale: the spec explicitly pushes toward a narrower PoC (§5), and one team maintaining two full products dilutes both the demo and the RRL/value-proposition story. The old design is preserved in git history if any of it needs to be revisited.
- Any HR topic outside onboarding (leave, payroll, benefits) stays in the corpus for grounding continuity but is **not** the eval/demo focus.
- Real employee PII or real government-ID images — all OCR training/eval documents are synthetic mockups (see §3.3).

The underlying infrastructure that *is* still reused (orchestrator skeleton, memory, guardrail patterns, eval harness conventions) means most of the Final's component work is **extending or repurposing working code**, not building from zero — see §4. The document-validation-specific rules (§4.1) are new code, though modeled on the same fail-toward-safety design philosophy the removed escalation rules used.

---

## 2. Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │      Streamlit Chat UI + Document Upload     │
                    └───────────────────────┬─────────────────────┘
                                            │ HTTP
                    ┌───────────────────────▼─────────────────────┐
                    │   FastAPI  (/chat, /upload-doc, /health)     │
                    └───────────────────────┬─────────────────────┘
                                            │
        ┌───────────────────────────────────▼───────────────────────────────────┐
        │                     Agent Orchestrator (ReAct loop)                    │
        │  1. Input guardrails: deterministic checks + LLM-judge (5 dimensions)  │
        │  2. Intent router: FAQ | Document upload | Ambiguous → disambiguate    │
        │  3. Tool selection & execution                                         │
        │  4. Output guardrails (grounding check, extraction-confidence check)   │
        └──────┬───────────────┬─────────────────┬──────────────┬────────────────┘
               │               │                 │              │
        ┌──────▼──────┐ ┌──────▼────────┐ ┌──────▼───────┐ ┌────▼─────────┐
        │  RAG tool   │ │  OCR extract  │ │  Checklist   │ │   Memory     │
        │  (Chroma +  │ │  tool ★ CV/DS │ │  validator   │ │  session +   │
        │  switchable │ │  domain       │ │  (determin-  │ │  persistent  │
        │  embeddings)│ │  integration) │ │  istic rules)│ │  (SQLite)    │
        └─────────────┘ └───────────────┘ └──────────────┘ └──────────────┘

        Observability: MLflow tracing on every request (latency, tokens, tool calls, OCR confidence, errors)
```

### Technology stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| LLM | `LLM_PROVIDER` switchable: Gemini 2.5 Flash (`gemini-2.5-flash`, default) via `google-genai` SDK, or a self-hosted Ollama model (default `gemma4:e4b`) via `src/agent/llm_client.py`'s adapter | Fast + cheap for chat; supports function calling and JSON schema output. Ollama option exists for quota-free testing — see §2.1. Multimodal (image input) covers both chat and OCR extraction on the Gemini path. |
| **CV/DS domain model (★ mandatory, Component 14)** | Gemini 2.5 Flash multimodal document extraction, `response_schema`-typed | See RRL, §5, for why this beats a standalone OCR engine for this task. Not currently exercised through the Ollama path — see §5. |
| Embeddings | `EMBEDDING_PROVIDER` switchable: `gemini-embedding-001` (768-dim via `output_dimensionality`, default) or Ollama's `nomic-embed-text` (native 768-dim) | Independent of the chat provider. Switching requires re-running `scripts/ingest.py` against a cleared index — the two embedding spaces aren't compatible despite both being 768-dim. |
| Vector store | ChromaDB (persistent, local) | Unchanged from Midterm; corpus reused as-is. |
| Structured data | SQLite | Memory tables stay; add employee onboarding-status records alongside them. |
| API | FastAPI | Add `POST /upload-doc` alongside existing `/chat`, `/usage`, `/health`. |
| UI | Streamlit | Add a file-upload widget to the existing chat UI. |
| Monitoring | MLflow | Reused; add OCR-specific trace fields (doc type, confidence, validation result). |
| Packaging | Docker | Reused; `docker compose up --build` already verified working end-to-end with both Gemini and Ollama backends. |

### 2.1 LLM Provider Abstraction — shipped (reused unchanged for the Final)

**Why:** Gemini's free tier caps `gemini-2.5-flash` at **20 `generate_content` requests/day per project**, not per-minute. A single agent turn already costs several requests (1 router classification + up to `MAX_REACT_ITERATIONS`=5 ReAct loop calls + 2 more if `search_web` fires), so the quota was exhausted mid-testing repeatedly. This risk is now directly relevant to the Final too: OCR extraction calls are additional Gemini requests on top of chat/routing, so quota pressure is if anything worse post-pivot.

**Mitigations shipped:**
- `run_turn()` catches `google.genai.errors.APIError` (and `LLMBackendError`) and degrades to a plain-language "try again" reply instead of a raw 500.
- `src/agent/usage.py` logs every agent LLM call's token usage to SQLite, tagged by which model actually served it; `GET /usage` reports today's + all-time request counts and token totals per model.
- `search_web` uses **Tavily** for search (domain-restricted via `include_domains`) plus one `response_schema` call to shape results — not dependent on Gemini's grounding tool.
- **`src/agent/llm_client.py`** — an `OllamaClient` adapter mimicking `google-genai`'s call signature and response shape, so `router.py`, `orchestrator.py`, and `tools.py` needed almost no changes. Selected via `LLM_PROVIDER=ollama` in `.env` (default stays `gemini`). Verified end-to-end against a real Ollama VM (`gemma4:e4b`).
- Embeddings are switchable too, independently: `EMBEDDING_PROVIDER` (`gemini` default or `ollama`) selects between `GeminiEmbedder` and `OllamaEmbedder` (`src/rag/embeddings.py`, `nomic-embed-text`). `config.get_embedder()` dispatches; `scripts/ingest.py` and `src/rag/retriever.py` both call it rather than hardcoding a class. `src/rag/answerer.py`'s grounded-answer generation also routes through `config.get_llm_client()`.
- **Not yet extended to OCR.** The new `extract_document` tool (§4, §5) is Gemini-multimodal only for the Final; Ollama vision-model coverage was out of scope for this pivot. If the Ollama path is used for the live demo, OCR calls still go to Gemini — factor that into quota planning.

**Switching `EMBEDDING_PROVIDER` requires re-running `scripts/ingest.py` against a cleared index** (`data/chroma/`, `data/index_manifest.json`).

**Known limitations:** small local models (`gemma4:e2b`/`e4b`) are less reliable at strict JSON-schema conformance and tool-calling than Gemini — expect more clarifying questions or `MAX_REACT_ITERATIONS` fallbacks on Ollama, never a crash (fail-closed paths already bound the blast radius).

**Operational note:** the verification Ollama VM was found reachable with no authentication — lock it down (firewall, reverse proxy with auth, or VPN-only) before relying on it for a live demo.

---

## 3. Knowledge Base & Data Pipeline

**Reused as-is from the Midterm** — no changes needed to parsing, chunking, embedding, or indexing.

### 3.1 Source corpus

9 synthetic-but-realistic HR documents in `data/raw/` (leave, benefits, payroll, onboarding, handbook, remote work, grievance procedure, anti-harassment, holidays as Markdown, plus a data-privacy policy as a PDF to exercise the conversion path). Kept as-is; never edited by the pipeline.

### 3.2 Pipeline stages (unchanged)

1. **Parse & normalize** — Markdown passes through; PDFs converted via `src/rag/pdf_to_md.py` (heading reconstruction from font-size/bold cues, table rendering, header/footer stripping). Output: `data/processed/<doc_id>.md` with YAML frontmatter.
2. **Chunk (structure-aware)** — split on headings first, pack to ~400 tokens with ~50-token overlap, merge sections under 80 tokens into their parent, keep tables whole, prepend a `"{title} > {section path}"` context header (`src/rag/chunking.py`).
3. **Metadata enrichment** — `doc_id`, `chunk_id`, `title`, `section_path`, `category`, `effective_date`, `version`, `token_count`.
4. **Embed & index** — via `config.get_embedder()` (§2.1); persistent Chroma collection keyed by `chunk_id`; `data/index_manifest.json` tracks hashes for idempotent re-ingestion.
5. **Retrieval (query time)** — top-k = 8 by cosine similarity, optional `category` filter; similarity floor (~0.5, tuned in evals) triggers "I don't know" + HR routing rather than a parametric-memory answer.
6. **Evaluation set** — golden Q&A pairs authored alongside the corpus, in `evals/golden_set.jsonl`.

### 3.3 New for the Final — OCR document set

- **Golden eval set additions**: onboarding-focused Q&A pairs added to `evals/golden_set.jsonl` as the primary Final eval slice (existing leave/payroll/benefits pairs stay for regression coverage, not the headline metric).
- **New data track — `data/onboarding_docs/`**: synthetic mockups of required onboarding documents (NBI clearance, SSS ID/UMID, Pag-IBIG MDF, BIR 2316, PhilHealth ID, APE/medical certificate), each with a ground-truth label (`*.expected.json`: doc type, name, ID number, issue/validity date). Include deliberate quality variance (skew, glare, partial crop, low-res phone-photo simulation) so the OCR eval reflects real submission conditions. **No real government IDs or real employee data.**

---

## 4. Component Breakdown & Ownership

Maps to the Final spec's 14-component checklist. Each member owns ≥2 components (6 total for a 3-person team, 8 for a 4-person team); **Component 14 is mandatory for the team**, not per-member. The RAG/agent/memory infrastructure is reused from the Midterm; the complaint-specific guardrail rules and tools were removed (§1.3) and document-validation is new code built on the same design philosophy.

| # | Component | Implementation for the Final | Status |
| --- | --- | --- | --- |
| 3 | **RAG** | Reused unchanged: ChromaDB + switchable Gemini/Ollama embeddings (§2.1), cited grounded answers (`src/rag/answerer.py` verifies citations against retrieved `chunk_id`s). Final eval slice narrows to onboarding-scoped queries. | ✅ built, reused |
| 2 | **Disambiguation** | Reused router (`src/agent/router.py`); `IntentClassification.intent` is currently `faq`/`ambiguous`/`out_of_scope` (complaint removed). Re-target to add `document_upload` for the Final. Same confidence-gated clarifying-question mechanism. | 🔄 repurpose |
| 4 | **Memory** | Reused two-tier memory (`src/memory/session.py` short-term, `persistent.py` rolling summary, cross-session recall by `employee_id`). Add a per-employee onboarding-status record (docs received/valid/missing) alongside the existing tables. | 🔄 extend |
| 5 | **Guardrails** | Input-safety layers reused unchanged (`input_checks.py`, `toxicity.py`, `pii.py`, `llm_judge.py` five-dimension check — see note below). NEW: `doc_validation.py`, a fail-toward-safety document-validation module (Rules 1–5, §4.1) built fresh — the escalation-specific rule engine it's modeled on was removed. | 🔄 new (input layers reused) |
| 6 | **Simple Chat UI** | Reused Streamlit app (`src/ui.py`); add a file-upload widget for document submission and a checklist-status display. | 🔄 extend |
| 7 | **API Endpoint Deployment** | Reused FastAPI app (`src/api.py`: `/chat`, `/usage`, `/health`); add `POST /upload-doc`. | 🔄 extend |
| 8 | **LLMOps (monitoring/tracing)** | Reused MLflow wiring (`src/monitoring.py`); add OCR-specific trace fields (doc type, confidence, validation outcome). Tool-call sequence/guardrail triggers still aren't in MLflow yet — carried over as a known gap. | 🔄 extend |
| 9 | **ReAct / Tool Use** | Reused orchestrator loop (`src/agent/orchestrator.py`, `MAX_REACT_ITERATIONS`=5, fail-closed on backend errors). Currently exposes `search_kb`/`search_web` only (complaint tools removed). Add `extract_document`, `validate_checklist`, `get_onboarding_status` for the onboarding persona. | 🔄 extend |
| **14** | **CV/DS Domain Integration ★ mandatory** | OCR/document-field-extraction tool wrapping Gemini multimodal input + `response_schema` (§5 RRL). | 🆕 new — the one ground-up build |
| 13 | **Evals** | Reused retrieval hit-rate@k harness and the guardrail red-team eval (`run_guardrail_eval.py`, input-safety only now — the complaint-exempt/escalation eval cases were removed with the feature). Add `run_ocr_eval.py` / `run_validation_eval.py` for document checks and an LLM-as-judge pass on FAQ answer quality. | 🔄 extend |

**Note on Guardrails (`llm_judge.py`):** an earlier version of this table (and of `main`, from an unresolved merge) described a simpler design where toxicity/injection signals were piggybacked onto the intent router's `IntentClassification` output. That was superseded — the shipped design is a separate `llm_judge.py` module (`response_schema=LLMJudgeVerdict`) classifying five dimensions (toxicity, PII, injection, off-topic, jailbreak) in one structured call, wired into `orchestrator._check_input()` after the deterministic checks. This is the version actually in `src/guardrails/llm_judge.py` — verified against the file, not assumed.

**Ownership table** (matches [README.md](README.md) — kept in sync):

| Member | Components |
| --- | --- |
| Baybayon | RAG, Evals |
| Del Rosario | ReAct/Tool Use, Disambiguation, LLM/embedding provider abstraction (§2.1) |
| Burayag | Memory, Guardrails |
| Tamondong | Chat UI, API Endpoint, LLMOps |
| **Team (shared)** | **CV/DS Domain Integration (Component 14, mandatory)** |

### 4.1 Document Validation Rules (Guardrails detail)

New module (`src/guardrails/doc_validation.py`, not yet built) applying the same **fail toward "needs human review," never silently accept** philosophy the Midterm's (now-removed) escalation rule engine used:

- **Rule 1 — Type match.** Extracted `doc_type` must match what the checklist is currently expecting; mismatches are rejected with a re-upload prompt, not guessed at.
- **Rule 2 — Identity match.** Extracted name must match the employee's on-file name (fuzzy match with a confirmed threshold); non-matches flag for human review, never auto-reject or auto-accept.
- **Rule 3 — Validity window.** Document-specific expiry rules (e.g., NBI clearance conventionally treated as stale beyond a fixed window) checked in code, not left to the LLM.
- **Rule 4 — Extraction confidence floor.** Below a configured OCR-confidence threshold, the result is never auto-validated — always routed to human review.
- **Rule 5 — Fail-safe.** Any schema-validation failure on the extracted fields escalates to human review by default.

New code: `src/guardrails/doc_validation.py` (`validate_document(extracted, employee_record) -> ValidationResult`), `OnboardingDocument` / `ValidationResult` schemas in `src/schemas.py`, thresholds/expiry windows in `src/config.py`.

---

## 5. Review of Related Literature (RRL) — OCR/Document Model Choice

Per spec §5, this isn't a separate deliverable but must be presented: state-of-the-art options considered, expected input/output, and why one was chosen.

| Option | Input → Output | Trade-offs |
| --- | --- | --- |
| **Gemini 2.5 Flash multimodal + `response_schema`** (chosen) | image/PDF → typed JSON fields directly | No new infra (same SDK/API key already in use); reuses the Structured Outputs pattern already in `src/schemas.py`; weaker on precise bounding-box/layout output than a dedicated OCR engine, but this project doesn't need layout, only field values. Consumes Gemini quota (§2.1) same as chat — not available on the Ollama path yet. |
| Google Cloud Vision OCR | image → raw text + bounding boxes | Mature, cheap per-call; needs a second Google Cloud service/credential and a separate field-parsing step on top of raw text. |
| Google Document AI (form/ID parser) | image → pre-structured key-value pairs | Purpose-built for ID/form parsing, likely highest raw accuracy; heavier setup (processor provisioning), overkill for a PoC-scoped eval set. |
| Tesseract / EasyOCR (open-source) | image → raw text | Free, fully local, no API dependency; weakest accuracy on skewed/low-quality phone photos, which is exactly the failure mode this project's eval set targets. |

**Decision:** Gemini 2.5 Flash multimodal extraction — lowest integration cost, reuses existing schema-validation infrastructure, and the PoC's accuracy bar is "flag for human review when uncertain" rather than "fully automate," which fits a general-purpose multimodal model better than a narrow OCR engine tuned for layout extraction. Revisit if the eval (§9) shows the confidence floor triggering human review too often to be useful, or if Gemini quota pressure (§2.1) makes a local OCR engine more attractive despite the accuracy trade-off.

---

## 6. Value Proposition & Unit of Measurement (UoM)

*(Placeholder — team to fill in with real or defensibly-estimated numbers before the presentation; spec §6.3 requires this to be qualified by time period and defensible.)*

- **Baseline cost estimate:** HR generalist time spent per new hire manually checking onboarding documents and answering "what do I need" questions — estimate `___ minutes/new-hire`, at `___ new hires/month`.
- **Value claim:** automating extraction + checklist validation saves an estimated `___ hours/month` ≈ `₱___/month` (state the hourly/monthly basis explicitly — do not leave a bare peso figure unqualified).
- **Intangible value (flag explicitly, don't monetize):** faster, more consistent Day-1 readiness; fewer back-and-forth emails for missing documents.

---

## 7. Repository Layout

```
stai-capstone/
├── CLAUDE.md
├── PLAN.md                      # this file
├── README.md
├── specs.md                                                    # Midterm spec (superseded, reference only)
├── [Stratpoint x DLSU] Final Capstone - Project Specification.txt  # governing spec
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/
│   ├── raw/                     # HR corpus (reused from Midterm)
│   ├── processed/
│   ├── chroma/                  # gitignored
│   └── onboarding_docs/         # NEW: synthetic document images + *.expected.json labels
├── scripts/
│   └── ingest.py                # reused unchanged
├── src/
│   ├── config.py                # + validation thresholds, expiry windows
│   ├── monitoring.py            # MLflow tracing helpers (built) + OCR trace fields
│   ├── agent/
│   │   ├── orchestrator.py      # reused ReAct loop + handle_message() adapter; retarget tool set
│   │   ├── router.py            # reused intent classification; retarget intent enum
│   │   ├── tools.py             # reused search_kb/search_web; ADD extract_document, validate_checklist, get_onboarding_status
│   │   ├── usage.py             # reused token usage tracking
│   │   ├── llm_client.py        # reused Ollama backend adapter (§2.1)
│   │   └── prompts.py           # versioned prompt variants
│   ├── rag/                     # reused unchanged (chunking.py, embeddings.py, retriever.py)
│   ├── ocr/                     # NEW
│   │   └── extractor.py         # Gemini multimodal extraction + response_schema
│   ├── guardrails/               # reused (input_checks.py, toxicity.py, pii.py, llm_judge.py, grounding.py)
│   │   └── doc_validation.py    # NEW: Rules 1–5, §4.1
│   ├── memory/                   # reused (session.py, persistent.py)
│   │   └── onboarding_status.py # NEW: per-employee doc checklist state (SQLite)
│   ├── schemas.py                # + OnboardingDocument, ValidationResult
│   ├── api.py                    # + POST /upload-doc
│   └── ui.py                     # + upload widget
├── evals/
│   ├── golden_set.jsonl          # reused + onboarding-focused additions
│   ├── run_retrieval_eval.py     # reused
│   ├── guardrail_redteam.jsonl   # reused unchanged (input-safety layer untouched by this pivot)
│   ├── run_guardrail_eval.py     # reused unchanged
│   ├── run_ocr_eval.py           # NEW: field-extraction accuracy vs data/onboarding_docs labels
│   ├── run_validation_eval.py    # NEW: doc_validation precision/recall
│   └── run_answer_eval.py        # NEW: LLM-as-judge on FAQ answers
└── tests/                        # existing suite (chunking, guardrails, schemas, agent, api) + new OCR/validation tests
```

---

## 8. Build Order

Most of the stack is already built (§2.1, §4); the Final's build order is short because it's mostly repurposing, not greenfield work.

1. **OCR extraction tool** — `src/ocr/extractor.py`, Gemini multimodal + `OnboardingDocument` schema; test against a handful of `data/onboarding_docs/` mockups. The one genuinely new component (14).
2. **Document validation guardrail** — `src/guardrails/doc_validation.py`, Rules 1–5 (§4.1); unit tests per rule.
3. **Extend the agent loop** — add `document_upload` to the router's intent enum and add the document tools to the orchestrator's exposed tool set (§4); the ReAct loop machinery itself needs no changes.
4. **Extend memory** — add the per-employee onboarding-status SQLite record alongside the existing session/summary tables.
5. **Extend interfaces** — `POST /upload-doc` on the existing FastAPI app; upload widget + status display in the existing Streamlit UI.
6. **Extend ops** — add OCR confidence/validation outcome fields to the existing MLflow tracing.
7. **Evals + RRL writeup** — OCR field-extraction accuracy, validation precision/recall, retrieval hit-rate@k (reused), LLM-as-judge on FAQ answers, trajectory-eval on ≥1 full agent decision chain (spec §6.1 requirement); finalize the RRL comparison table and value-proposition numbers for the deck.

---

## 9. Experiments to Report

Spec requires ≥3 quantitative eval metrics, at least one full reasoning-trace walkthrough, and (per §7) a more rigorous suite than the Midterm's sample-output grading.

| Experiment | Metric | Notes | Status |
| --- | --- | --- | --- |
| OCR field extraction | Per-field accuracy / character error rate vs. `data/onboarding_docs/*.expected.json` | Break out by document type and by image-quality variant (clean vs. skewed/low-res). | 🆕 new |
| Document validation | Precision/recall on "needs human review" vs. "auto-passes" | Deterministic-correctness harness, same shape as the eval the removed escalation rules used; false auto-passes are the costly failure mode. | 🆕 new |
| Onboarding FAQ retrieval | Hit-rate@5, MRR on onboarding-scoped golden-set slice | Reuses `run_retrieval_eval.py` unchanged. | 🔄 reused |
| Guardrail red-team (inherited) | Block rate (injection/toxicity), detection rate (PII) | Already measured: 100%/100% on 20 adversarial prompts (`run_guardrail_eval.py` + `guardrail_redteam.jsonl`). Reused unchanged — the input-safety layer isn't touched by this pivot. Off-topic block rate needs `--with-router` (costs LLM calls, off by default). | ✅ passing, reused |
| FAQ answer quality | LLM-as-judge faithfulness/citation accuracy | New `run_answer_eval.py`. | 🆕 new |
| Agent trajectory | Manual walkthrough of ≥1 full decision chain (ask → route → extract → validate → respond) | For the required reasoning-trace slide. | 🆕 new |
| Latency & cost | p50/p95 latency, tokens per request, from MLflow | Chat-only vs. chat+OCR request comparison; also Gemini vs. Ollama given §2.1's quota pressure. | 🔄 extend |

Document failure modes (e.g., low-quality photo submissions, name-match false negatives, router confusion between "asking about a document" vs. "submitting one") and mitigations for the Retrospective section.

---

## 10. Key Risks & Mitigations

- **OCR accuracy on poor-quality phone photos** → confidence floor + mandatory human-review fallback (Rule 4); measured explicitly in evals, not assumed.
- **Scope creep back toward the Midterm's complaint/escalation flow** → the feature is fully removed (§1.3), not just deprioritized; resist re-building it mid-Final unless the team explicitly decides to reverse that call (it's recoverable from git history).
- **Gemini rate limits/outage during demo** → **materialized during development**, not just theoretical: the free tier's 20 `generate_content`/day cap on `gemini-2.5-flash` was exhausted mid-testing repeatedly. Mitigations: fail-closed error handling instead of crashing; `GET /usage` for consumption visibility; `LLM_PROVIDER=ollama` + `EMBEDDING_PROVIDER=ollama` give real headroom for chat/RAG — but **not for OCR** (§2.1, §5), which stays on Gemini regardless, so quota planning must budget for it separately.
- **Ollama backend quality/availability for a live demo** → small local models are less reliable at strict tool-calling/schema conformance than Gemini, and the verification VM has no managed-service reliability guarantees or auth. Treat it as a development escape valve, not the default presentation path, unless dry-run tested beforehand — and lock down the VM's auth regardless.
- **Mixed local/Docker ingestion corrupts the index** → hit three times during development: running `scripts/ingest.py` on the host and `docker compose up` against the same bind-mounted `data/chroma/` causes a Rust panic on open (chromadb cross-platform incompatibility). Not data-destructive, just wastes time. Pick one environment per index.
- **Real PII/government-ID exposure** → all OCR eval/demo documents are synthetic mockups; never use real employee documents (§3.3).
- **Demo failure during presentation** → run fully local (Chroma + SQLite), record a fallback video, disclose upfront per spec.
