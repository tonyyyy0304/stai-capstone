# Onboarding Concierge — Final Capstone Implementation Plan

**Course:** Introduction to Agentic AI (STAI100) — Final Capstone (Week 14)
**Use case type:** Vector DB RAG + CV/OCR Document Understanding (HR/Onboarding domain)
**LLM provider:** Google Gemini API (chat + embeddings), with a self-hosted Ollama server as a fully switchable fallback for both, independently, via `LLM_PROVIDER` / `EMBEDDING_PROVIDER` — see §2.1. OCR/document extraction (new, Final-only) is Gemini-multimodal specifically; see §5 RRL.
**Builds on:** the Midterm project ("HR FAQ & Complaint Chatbot"). Reused as-is: the RAG pipeline, the agent orchestrator (ReAct loop + tools), two-tier memory, guardrails (deterministic checks + LLM-judge), the LLM/embedding provider abstraction, and the API/UI/MLflow/Docker scaffolding — all of this is already built and tested (§2.1, §4). Complaint intake & escalation (the form-driven consent-gate design, `src/guardrails/escalation.py`, `danger_scan.py`, `form_pii.py`, `escalation_state.py`, and the `ComplaintTicket`/`Severity`/`TriggerRule`/`Escalation*` schemas) have been **removed from the codebase** — not just descoped — in favor of the onboarding-document-verification direction (see §1.3).
**Governing spec:** `[Stratpoint x DLSU] Final Capstone - Project Specification.txt` (repo root) — supersedes the Midterm spec ([specs.md](specs.md), kept for reference; do not edit either spec file).
**Employment scope:** PH **regular** employment only. Contractual / project-based / agency setups are explicitly out of scope for both the handbook content and the onboarding checklist.

### Research questions driving this plan

Instructor feedback (post-Midterm review) framed the Final around two open questions the codebase could not yet answer:

- **(a) Can the agent reach ~100% eval accuracy answering FAQs from a handbook ≥100 pages** (comparable in scale to a DLSU faculty/student handbook)? The Midterm corpus is 10 synthetic docs, ~35 KB, 59 chunks — roughly 8 pages. Any accuracy claim from it doesn't transfer to the scale the question is actually asking about.
- **(b) How well can the agent verify that submitted onboarding documents are correct and complete?** — with an explicit counterpoint from the instructor: *"is this easy enough?"* A clean mockup fed to Gemini multimodal returning JSON is trivial and reads as decorative. The defensible version needs a real decision layer in front of and around the extraction call — see §4.1.

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
- Non-regular (contractual/project-based/agency) employment.

The underlying infrastructure that *is* still reused (orchestrator skeleton, memory, guardrail patterns, eval harness conventions) means most of the Final's component work is **extending or repurposing working code**, not building from zero — see §4. The document-validation-specific rules (§4.1) are new code, though modeled on the same fail-toward-safety design philosophy the removed escalation rules used.

---

## 2. Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │      Streamlit Chat UI + Document Upload     │
                    └───────────────────────┬─────────────────────┘
                                            │ HTTP
                    ┌───────────────────────▼─────────────────────┐
                    │   FastAPI  (/chat, /upload-doc,              │
                    │             /onboarding-status/{id}, /health)│
                    └───────────────────────┬─────────────────────┘
                                            │
        ┌───────────────────────────────────▼───────────────────────────────────┐
        │                     Agent Orchestrator (ReAct loop)                    │
        │  1. Input guardrails: deterministic checks + LLM-judge (5 dimensions)  │
        │  2. Intent router: FAQ | Document upload | Status | Ambiguous          │
        │  3. Tool selection & execution                                         │
        │  4. Output guardrails (grounding check, extraction-confidence check)   │
        └──────┬──────────────┬────────────────┬──────────────┬──────────────────┘
               │              │                │              │
        ┌──────▼──────┐┌──────▼─────────────┐┌──▼───────────┐┌▼─────────────┐
        │  RAG tool   ││  CV quality gate →  ││  Checklist   ││   Memory     │
        │  (Chroma +  ││  OCR extract tool   ││  validator   ││  session +   │
        │  dense +    ││  ★ CV/DS domain     ││  (determin-  ││  persistent +│
        │  BM25 hybrid││  integration        ││  istic rules)││  onboarding  │
        │  fusion)    ││  (Gemini multimodal)││              ││  status      │
        └─────────────┘└─────────────────────┘└──────────────┘└──────────────┘

        Observability: MLflow tracing on every request (latency, tokens, tool calls,
        image quality metrics, OCR confidence, validation outcome, errors)
```

The CV quality gate (OpenCV, deterministic, no LLM call) runs **before** the Gemini extraction call — it can reject or flag an image outright, which is both a real accuracy decision and the primary quota-conservation mechanism (§2.1).

### Technology stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| LLM | `LLM_PROVIDER` switchable: Gemini 2.5 Flash (`gemini-2.5-flash`, default) via `google-genai` SDK, or a self-hosted Ollama model (default `gemma4:e4b`) via `src/agent/llm_client.py`'s adapter | Fast + cheap for chat; supports function calling and JSON schema output. Ollama option exists for quota-free testing — see §2.1. |
| **CV/DS domain model (★ mandatory, Component 14)** | Gemini 2.5 Flash multimodal document extraction, `response_schema`-typed, gated by a deterministic OpenCV quality check | See RRL, §5, for why this beats a standalone OCR engine for this task. Not currently exercised through the Ollama path — see §5. |
| Embeddings | `EMBEDDING_PROVIDER` switchable: `gemini-embedding-001` (768-dim via `output_dimensionality`, default) or Ollama's `nomic-embed-text` (native 768-dim) | Independent of the chat provider. Switching requires re-running `scripts/ingest.py` against a cleared index — the two embedding spaces aren't compatible despite both being 768-dim. |
| Vector store | ChromaDB (persistent, local) + BM25 (SQLite FTS5) fused via RRF | Dense-only blurs exact identifiers ("Section 8.3.2", "Form 2316", "MDF") that a large manual is full of — see §3, Advanced RAG. |
| Structured data | SQLite | Memory tables stay; add employee onboarding-status records alongside them. |
| API | FastAPI | Add `POST /upload-doc`, `GET /onboarding-status/{employee_id}` alongside existing `/chat`, `/usage`, `/health`. |
| UI | Streamlit | Add a file-upload widget + checklist-status panel to the existing chat UI. |
| Monitoring | MLflow | Reused; add OCR/quality/validation trace fields. |
| Packaging | Docker | Reused; `docker compose up --build` already verified working end-to-end with both Gemini and Ollama backends. |

### 2.1 LLM Provider Abstraction — shipped (reused unchanged for the Final)

**Why:** Gemini's free tier caps `gemini-2.5-flash` at **20 `generate_content` requests/day per project**, not per-minute. A single agent turn already costs several requests (1 router classification + up to `MAX_REACT_ITERATIONS`=5 ReAct loop calls + 2 more if `search_web` fires), so the quota was exhausted mid-testing repeatedly. This is the binding constraint on the OCR eval too: a naive run over 60 images is 60 calls — three days of free quota for a single eval pass.

**Mitigations shipped:**
- `run_turn()` catches `google.genai.errors.APIError` (and `LLMBackendError`) and degrades to a plain-language "try again" reply instead of a raw 500.
- `src/agent/usage.py` logs every agent LLM call's token usage to SQLite, tagged by which model actually served it; `GET /usage` reports today's + all-time request counts and token totals per model.
- `search_web` uses **Tavily** for search (domain-restricted via `include_domains`) plus one `response_schema` call to shape results — not dependent on Gemini's grounding tool.
- **`src/agent/llm_client.py`** — an `OllamaClient` adapter mimicking `google-genai`'s call signature and response shape, so `router.py`, `orchestrator.py`, and `tools.py` needed almost no changes. Selected via `LLM_PROVIDER=ollama` in `.env` (default stays `gemini`). Verified end-to-end against a real Ollama VM (`gemma4:e4b`).
- Embeddings are switchable too, independently: `EMBEDDING_PROVIDER` (`gemini` default or `ollama`) selects between `GeminiEmbedder` and `OllamaEmbedder` (`src/rag/embeddings.py`, `nomic-embed-text`). `config.get_embedder()` dispatches; `scripts/ingest.py` and `src/rag/retriever.py` both call it rather than hardcoding a class. `src/rag/answerer.py`'s grounded-answer generation also routes through `config.get_llm_client()`.
- **Not yet extended to OCR.** The new `extract_document` tool (§4, §5) is Gemini-multimodal only for the Final; Ollama vision-model coverage was out of scope for this pivot. If the Ollama path is used for the live demo, OCR calls still go to Gemini — factor that into quota planning.
- **New for OCR — disk-cached extraction.** Every extraction result is cached to `evals/results/ocr_cache/<sha256 of the image bytes>.json`. Eval re-runs cost zero API calls once populated; a full 60-image eval can be spread across multiple days without re-spending quota. `run_ocr_eval.py` and `run_answer_eval.py` both read the cache by default; `--no-cache` forces live calls. A paid key should still be budgeted before demo week.

**Switching `EMBEDDING_PROVIDER` requires re-running `scripts/ingest.py` against a cleared index** (`data/chroma/`, `data/index_manifest.json`).

**Known limitations:** small local models (`gemma4:e2b`/`e4b`) are less reliable at strict JSON-schema conformance and tool-calling than Gemini — expect more clarifying questions or `MAX_REACT_ITERATIONS` fallbacks on Ollama, never a crash (fail-closed paths already bound the blast radius).

**Operational note:** the verification Ollama VM was found reachable with no authentication — lock it down (firewall, reverse proxy with auth, or VPN-only) before relying on it for a live demo.

---

## 3. Knowledge Base & Data Pipeline

Pipeline mechanics (parsing, chunking, embedding, indexing) are **reused as-is** from the Midterm. The corpus itself and the retrieval strategy are new for the Final — see below.

### 3.1 Source corpus

**Handbook (new, `data/raw/handbook/`)** — twelve synthetic-but-realistic PH employee-handbook chapters, ~4,500 words each, ≈54k words ≈120 printed pages ≈~180 chunks at the current 400-token target:

1. Employment status & regularization (probationary → regular, six-month rule)
2. Onboarding requirements & Day-1 checklist — ties directly into document verification
3. Code of conduct
4. Administrative due process (twin-notice rule, NTE, hearing, NOD) — the instructor's "due process / completeness check" angle
5. Working hours, attendance, overtime
6. Leaves (SIL, maternity/paternity, solo parent, special leave)
7. Compensation & 13th month
8. Statutory benefits (SSS, PhilHealth, Pag-IBIG)
9. Health, safety, APE
10. Data privacy & acceptable use
11. Grievance & disciplinary matrix
12. Separation & clearance

Each chapter keeps the existing YAML frontmatter contract (`doc_id`, `title`, `category`, `effective_date`, `version`), so `scripts/ingest.py` and `src/rag/chunking.py` need no code changes — just confirm the ingester globs `data/raw/` recursively (add recursion if it doesn't; that's the only pipeline change this requires). Chapters 2 and 4 are also rendered to PDF to keep `src/rag/pdf_to_md.py` exercised.

**Swap-in path:** the loader is corpus-agnostic by construction. Once available, the real **DLSU Faculty Manual (~273 pages)** replaces the synthetic handbook as a data change, not a code change — see the `pdf_to_md.py` risk in §10, since that parser is a single-column heuristic and the real manual is multi-column with deep numbering.

The original 10 synthetic HR docs (`data/raw/*.md`, leave/benefits/payroll/etc.) stay in place as a regression slice — not the Final's headline eval corpus, but still exercised by `run_retrieval_eval.py`.

### 3.2 Pipeline stages (unchanged)

1. **Parse & normalize** — Markdown passes through; PDFs converted via `src/rag/pdf_to_md.py` (heading reconstruction from font-size/bold cues, table rendering, header/footer stripping). Output: `data/processed/<doc_id>.md` with YAML frontmatter.
2. **Chunk (structure-aware)** — split on headings first, pack to ~400 tokens with ~50-token overlap, merge sections under 80 tokens into their parent, keep tables whole, prepend a `"{title} > {section path}"` context header (`src/rag/chunking.py`).
3. **Metadata enrichment** — `doc_id`, `chunk_id`, `title`, `section_path`, `category`, `effective_date`, `version`, `token_count`.
4. **Embed & index** — via `config.get_embedder()` (§2.1); persistent Chroma collection keyed by `chunk_id`; `data/index_manifest.json` tracks hashes for idempotent re-ingestion.
5. **Retrieval (query time)** — top-k = 8, dense cosine by default; optional hybrid mode (§3.4) fuses in BM25; optional `category` filter; similarity floor (~0.5, tuned in evals) triggers "I don't know" + HR routing rather than a parametric-memory answer.
6. **Evaluation set** — golden Q&A pairs in `evals/golden_set.jsonl` (§3.3).

### 3.3 Golden set — tiered for an honest "100%?" answer

The existing 45 rows stay as the regression slice. ~90 new rows target the handbook, tagged so metrics break out by tier:

| Tier | Count | What it measures |
| --- | --- | --- |
| `lookup` | 40 | Single-section factual retrieval |
| `multihop` | 25 | Requires synthesizing 2+ sections |
| `negative` | 15 | **Not answerable from the manual** — measures abstention, not recall |
| `near_miss` | 10 | Wording close to a real section but asking something the manual doesn't say |

Row schema extended with `difficulty` and `qtype`; legacy rows default to `lookup`/`legacy`. The `negative` and `near_miss` tiers exist specifically so "100%" can't be gamed by an agent that just over-answers — abstention on unanswerable questions is scored as correct, not as a miss.

### 3.4 Advanced RAG — hybrid retrieval (Component 12)

New module `src/rag/hybrid.py`: a BM25 index (SQLite FTS5 — SQLite is already a dependency) over the same chunks, fused with dense results via Reciprocal Rank Fusion:

```
score(chunk) = Σ_r 1 / (RRF_K + rank_r(chunk))
```

`RRF_K` lives in `src/config.py`. Wrapped behind the existing `Retriever` interface (`src/rag/retriever.py`), so `src/rag/answerer.py` and `src/agent/tools.py` need no changes. Selected via `RETRIEVER_MODE=dense|hybrid`.

**Why hybrid and not query-rewrite/rerank:** dense embeddings blur exact identifiers, which a 120-page manual is full of ("Section 8.3.2", "Form 2316", "MDF", "twin-notice"). Hybrid fusion fixes that at **zero extra LLM calls at query time**. Query rewriting and LLM-based reranking would each cost a Gemini call per query against the 20/day cap (§2.1) — rejected on quota grounds, not accuracy grounds, and documented as such for the RRL/design-tradeoffs discussion.

### 3.5 Onboarding document image set

New data track `data/onboarding_docs/` — synthetic mockups of the six required onboarding documents (NBI clearance, SSS, Pag-IBIG MDF, BIR 2316, PhilHealth, APE/medical certificate), generated programmatically (§4.2) with paired ground truth. **No real government IDs or real employee data**, per §1.3.

---

## 4. Component Breakdown & Ownership

Maps to the Final spec's 14-component checklist. Each member owns ≥2 components (6 total for a 3-person team, 8 for a 4-person team); **Component 14 is mandatory for the team**, not per-member. This plan claims 11 components against a floor of 8 (4 members × 2) — comfortable margin.

| # | Component | Implementation for the Final | Status |
| --- | --- | --- | --- |
| 3 | **RAG** | ChromaDB + switchable Gemini/Ollama embeddings (§2.1), cited grounded answers (`src/rag/answerer.py`). Corpus grows to the ~180-chunk handbook (§3.1); eval slice narrows to onboarding-scoped + tiered queries (§3.3). | 🔄 extend (corpus swap) |
| 2 | **Disambiguation** | Reused router (`src/agent/router.py`); `IntentClassification.intent` currently `faq`/`ambiguous`/`out_of_scope`. Add `document_upload` and `document_status`. Same confidence-gated clarifying-question mechanism. | 🔄 repurpose |
| 4 | **Memory** | Reused two-tier memory (`src/memory/session.py`, `persistent.py`). Add `src/memory/onboarding_status.py` — per-employee doc checklist state (§4.3). | 🔄 extend |
| 5 | **Guardrails** | Input-safety layers reused unchanged (`input_checks.py`, `toxicity.py`, `pii.py`, `llm_judge.py` — see note below). NEW: `doc_validation.py`, a fail-toward-safety document-validation module (Rules 1–6, §4.1). | 🔄 new (input layers reused) |
| 6 | **Simple Chat UI** | Reused Streamlit app (`src/ui.py`); add a file-upload widget and a checklist-status panel. | 🔄 extend |
| 7 | **API Endpoint Deployment** | Reused FastAPI app (`src/api.py`); add `POST /upload-doc`, `GET /onboarding-status/{employee_id}`. | 🔄 extend |
| 8 | **LLMOps (monitoring/tracing)** | Reused MLflow wiring (`src/monitoring.py`); extend the fail-closed allowlists with OCR/quality/validation fields (§4.4). | 🔄 extend |
| 9 | **ReAct / Tool Use** | Reused orchestrator loop (`src/agent/orchestrator.py`, `MAX_REACT_ITERATIONS`=5, fail-closed). Add `extract_document`, `validate_checklist`, `get_onboarding_status`. Loop machinery itself is unchanged. | 🔄 extend |
| **12** | **Advanced RAG** | Hybrid BM25 + dense retrieval with RRF fusion (§3.4). | 🆕 new |
| **14** | **CV/DS Domain Integration ★ mandatory** | OpenCV deterministic quality gate + Gemini multimodal extraction (`response_schema`) + deterministic validation rules (§4.1–§4.2, §5 RRL). | 🆕 new — the one ground-up build |
| 13 | **Evals** | Reused retrieval hit-rate@k harness and guardrail red-team eval. Add `run_ocr_eval.py`, `run_validation_eval.py`, `run_answer_eval.py` (§7). | 🔄 extend |

**Note on Guardrails (`llm_judge.py`):** the shipped design is a separate `llm_judge.py` module (`response_schema=LLMJudgeVerdict`) classifying five dimensions (toxicity, PII, injection, off-topic, jailbreak) in one structured call, wired into `orchestrator._check_input()` after the deterministic checks. Verified against the file, not assumed.

**Ownership table** (matches [README.md](README.md) — kept in sync):

| Member | Components |
| --- | --- |
| Baybayon | RAG, Advanced RAG (hybrid), Evals |
| Del Rosario | ReAct/Tool Use, Disambiguation, LLM/embedding provider abstraction (§2.1) |
| Burayag | Memory, Guardrails |
| Tamondong | Chat UI, API Endpoint, LLMOps |
| **Team (shared)** | **CV/DS Domain Integration (Component 14, mandatory)** — see per-module leads in §4.2–§4.3 |

### 4.1 Document Validation Rules (Guardrails detail)

New module `src/guardrails/doc_validation.py`, owned by Burayag, applying the same **fail toward "needs human review," never silently accept** philosophy the Midterm's (now-removed) escalation rule engine used:

- **Rule 1 — Type match.** Extracted `doc_type` must match what the checklist is currently expecting; mismatches prompt re-upload, never a guess.
- **Rule 2 — Completeness.** Every required field for that document type (per the `src/ocr/doctypes.py` registry) must be present and non-empty. This is the instructor's "completeness check."
- **Rule 3 — Format validity.** Each extracted field passes its registry regex/checksum (e.g. TIN `\d{3}-\d{3}-\d{3}(-\d{3})?`, SSS `\d{2}-\d{7}-\d`). A field that fails format is untrusted regardless of what confidence the model reported for it — a deterministic signal the LLM can't talk its way past.
- **Rule 4 — Identity match.** Extracted name matched against the employee's on-file record (`rapidfuzz.token_set_ratio`, threshold in config), handling PH name forms (middle initials, `Jr./III` suffixes, surname-first). Non-matches flag for human review, never auto-reject or auto-accept.
- **Rule 5 — Validity window.** Document-specific expiry rules (e.g. NBI clearance conventionally treated as stale beyond a fixed window) checked in code against an injectable `as_of` date, not left to the LLM — the injectable date keeps expiry tests deterministic and non-rotting.
- **Rule 6 — Fail-safe.** Any schema-validation failure, unparseable date, or a `warn`-level image-quality verdict (§4.2) paired with low model confidence escalates to human review by default.

Composite confidence = `min(normalized_image_quality, model_confidence)`, forced to zero by any format failure (Rule 3) — explainable and defensible on the deck. Outcome is one of `accepted | rejected | needs_review`; the module never silently accepts.

New code: `validate_document(extracted, employee_record, as_of) -> ValidationResult`; `OnboardingDocument` / `ValidationResult` schemas in `src/schemas.py`; thresholds/expiry windows in `src/config.py`.

### 4.2 CV/OCR pipeline (Component 14 detail)

New package `src/ocr/`, team-shared but needs per-module leads assigned or it stalls.

**`src/ocr/doctypes.py`** — declarative registry, one entry per document type: canonical name, aliases, required/optional fields, per-field format validator, validity window.

| Doc type | Key fields | Depth |
| --- | --- | --- |
| NBI Clearance | full name, date of issue, NBI ID no., purpose | ★ deep — full extraction + validation |
| BIR Form 2316 | employee name, TIN, taxable year, employer name/TIN, gross compensation | ★ deep — full extraction + validation |
| SSS (E-1 / UMID) | SS number, name, date of birth | classification only |
| Pag-IBIG MDF | MID number, name | classification only |
| PhilHealth ID | PIN, name | classification only |
| APE / Medical cert | name, exam date, clinic/physician, fitness statement | classification only |

All six are classified (doc-type identification, feeding the checklist and the confusion-matrix eval); NBI and BIR 2316 additionally get full field extraction and validation. This depth split is deliberate: it answers the completeness-check story across the whole checklist while keeping the extraction/validation engineering bounded to two document types.

**`src/ocr/quality.py`** — OpenCV, no LLM call. This module is the direct answer to "is this easy enough":
- `blur_score` (variance of Laplacian), `exposure_clip` (glare/underexposure via histogram clipping), `skew_deg` (`minAreaRect` over the largest text contour), `min_dim_px` (resolution floor), `quad_found` (document-boundary contour detection).
- Returns an `ImageQualityReport` with verdict `pass | warn | reject` and reasons. A `reject` verdict short-circuits **before** any Gemini call — a real quality decision that also directly relieves the §2.1 quota constraint.
- `preprocess()` — deskew, perspective-correct to the detected quad, CLAHE contrast correction, upscale to the resolution floor. Extraction accuracy with vs. without this step is a headline ablation (§7).

**`src/ocr/extractor.py`** — one Gemini call per document (quota discipline): `response_schema` returns `doc_type` plus a nullable superset of fields; `doctypes.py` then selects the required subset in code. Each field carries `value`, `verbatim_text` (what the model literally saw, for auditability), and `model_confidence`. Reuses the `response_schema` pattern already proven in `src/rag/answerer.py`, `src/agent/tools.py`, `src/agent/router.py`, `src/guardrails/llm_judge.py`, `src/memory/persistent.py`.

Vision calls use a new `GEMINI_VISION_MODEL` config value (default `gemini-2.5-flash`), kept separate from `GEMINI_CHAT_MODEL` — the current chat model (`gemini-3.1-flash-lite`) is a lite tier not well suited to multimodal field extraction.

**Dataset — `scripts/make_onboarding_docs.py`:**
- 8 synthetic identities (`data/onboarding_docs/identities.json`) — doubles as the employee record backing Rule 4.
- 6 Pillow-rendered layout templates, one per document type.
- 5 degradation variants per document: `clean`, `skew` (8–15° perspective warp), `blur` (gaussian k=7–11), `glare` (radial overlay + brightness clip), `lowres_jpeg` (0.35 downscale, q=35).
- 6 × 8 × 5 = 240 images, each variant sharing one `<identity_id>.expected.json` with its clean counterpart — which is what makes the preprocessing ablation free (same ground truth, different degradation). Eval runs sample a fixed 60-image slice given quota; the full 240 stays available.
- 10 negative cases: wrong doc type for the expected slot, a non-document photo, a blank page, a document belonging to a different person (Rule 4), an expired NBI clearance (Rule 5).
- **Known limitation, stated on the deck:** rendered-then-degraded images are easier than real phone photos; the sim-to-real gap is unmeasured unless a small photographed holdout is added later.

### 4.3 Checklist state (Memory detail)

`src/memory/onboarding_status.py`, owned by Burayag. SQLite table `onboarding_documents(employee_id, doc_type, status, outcome, validated_at, source_hash)`, reusing the `_get_connection()` pattern from `src/memory/session.py`. `get_status(employee_id) -> ChecklistStatus` powers replies like "2 of 6 received, missing Pag-IBIG MDF."

### 4.4 Interfaces & ops (UI/API/LLMOps detail)

Owned by Tamondong (UI/API/LLMOps) and Del Rosario (agent wiring):

- **Router** — add `DOCUMENT_UPLOAD`/`DOCUMENT_STATUS` to `Intent` in `src/schemas.py`; update `ROUTER_PROMPT`. Explicitly test the *asking about* vs. *submitting* a document confusion the instructor flagged as a real failure mode.
- **Tools** — register `extract_document`, `validate_checklist`, `get_onboarding_status` in `orchestrator._function_declarations()`.
- **`POST /upload-doc`** — FastAPI `UploadFile` (needs `python-multipart`). Enforce `MAX_UPLOAD_BYTES`; validate MIME by magic bytes, not file extension; **strip EXIF before any storage or logging** (geolocation is PII); don't persist the raw image past the request unless explicitly configured.
- **`GET /onboarding-status/{employee_id}`**.
- **UI** — `st.file_uploader` in the composer, checklist-status panel in the sidebar. UI still talks only to FastAPI, never to Gemini directly.
- **MLflow** — `src/monitoring.py` uses fail-closed allowlists; extend `_ALLOWED_TAG_KEYS` with `doc_type`, `validation_outcome`, `quality_verdict`, and `_ALLOWED_METRIC_KEYS` with `blur_score`, `skew_deg`, `extraction_confidence`, `fields_extracted`, `fields_missing`, `ocr_latency_ms`.

### New config values — `src/config.py`

`GEMINI_VISION_MODEL`, `OCR_CONFIDENCE_FLOOR`, `BLUR_VARIANCE_FLOOR`, `MIN_IMAGE_DIM_PX`, `MAX_SKEW_DEG`, `NAME_MATCH_THRESHOLD`, `DOC_VALIDITY_WINDOWS`, `REQUIRED_ONBOARDING_DOCS`, `MAX_UPLOAD_BYTES`, `ALLOWED_IMAGE_MIME`, `RETRIEVER_MODE`, `RRF_K`.

### New dependencies

`opencv-python-headless` (not `opencv-python` — the slim Docker base image lacks `libGL`), `Pillow`, `rapidfuzz`, `rank_bm25` (skip if using SQLite FTS5 instead), `python-multipart`. Pin in `requirements.txt` and `requirements-api.txt`.

---

## 5. Review of Related Literature (RRL) — OCR/Document Model Choice

Per spec §5, this isn't a separate deliverable but must be presented: state-of-the-art options considered, expected input/output, and why one was chosen.

| Option | Input → Output | Trade-offs |
| --- | --- | --- |
| **Gemini 2.5 Flash multimodal + `response_schema`** (chosen) | image/PDF → typed JSON fields directly | No new infra (same SDK/API key already in use); reuses the Structured Outputs pattern already in `src/schemas.py`; weaker on precise bounding-box/layout output than a dedicated OCR engine, but this project doesn't need layout, only field values. Consumes Gemini quota (§2.1) same as chat — not available on the Ollama path yet. |
| Google Cloud Vision OCR | image → raw text + bounding boxes | Mature, cheap per-call; needs a second Google Cloud service/credential and a separate field-parsing step on top of raw text. |
| Google Document AI (form/ID parser) | image → pre-structured key-value pairs | Purpose-built for ID/form parsing, likely highest raw accuracy; heavier setup (processor provisioning), overkill for a PoC-scoped eval set. |
| Tesseract / EasyOCR (open-source) | image → raw text | Free, fully local, no API dependency; weakest accuracy on skewed/low-quality phone photos, which is exactly the failure mode this project's eval set targets. |

**Decision:** Gemini 2.5 Flash multimodal extraction — lowest integration cost, reuses existing schema-validation infrastructure, and the PoC's accuracy bar is "flag for human review when uncertain" rather than "fully automate," which fits a general-purpose multimodal model better than a narrow OCR engine tuned for layout extraction. Revisit if the eval (§7) shows the confidence floor triggering human review too often to be useful, or if Gemini quota pressure (§2.1) makes a local OCR engine more attractive despite the accuracy trade-off.

**A second, deliberate design choice sits in front of this model call: the OpenCV quality gate (§4.2) is a separate, non-LLM, deterministic layer.** This is the direct answer to the instructor's "is this easy enough?" counterpoint — a bare multimodal-extraction call would be trivial and decorative; gating it on measurable image-quality signals, and backing it with deterministic validation rules (§4.1) that a general chatbot has no equivalent of, is what makes the CV/DS component load-bearing rather than cosmetic.

**Retrieval-side RRL (§3.4):** query rewriting and LLM-based reranking were considered and rejected specifically on Gemini quota grounds (each adds a call per query against the 20/day cap), in favor of BM25+dense RRF fusion, which is free at query time and targets the same failure mode (exact-identifier misses) that a 120-page manual surfaces.

---

## 6. Value Proposition & Unit of Measurement (UoM)

Derived, not asserted — every input stated and defensible per spec §6.3.

| Input | Value | Basis |
| --- | --- | --- |
| Staffing ratio | 1 HR : 40 employees | Instructor-given baseline (10 HR staff : 400 employees) |
| New hires/month | ≈5 | 400 employees × ~15% annual turnover ÷ 12 — **state the turnover-rate source used** when filling this in |
| HR minutes per hire (FAQ answering + document checking) | `___ min` | **Needs a real basis** — team estimate or a short informal survey; flag as an assumption, don't invent false precision |
| HR hourly cost | ≈₱170–215/hour | PH HR generalist ~₱30–38k/month ÷ (22 days × 8 h) — cite the salary-range source used |
| FAQ deflection rate | `___%` | **Taken directly from the retrieval eval's correct-and-confident rate (§3.3/§7), not guessed** |

Output as **hours/month saved** and **₱/month**, both explicitly qualified by period (spec §6.3 requirement). Intangibles — faster, more consistent Day-1 readiness; fewer back-and-forth emails for missing documents — listed separately and flagged as unmonetized, not folded into the peso figure.

*(Blanks above are placeholders for numbers the team fills in before the presentation; the derivation path, not the placeholder, is the point — deflection rate in particular should come from measured eval output rather than being asserted.)*

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
│   ├── raw/
│   │   ├── handbook/             # NEW: 12-chapter synthetic PH handbook (§3.1), swappable for DLSU manual later
│   │   └── *.md, *.pdf           # original 10 docs, kept as regression slice
│   ├── processed/
│   ├── chroma/                   # gitignored (dense vectors)
│   ├── bm25.sqlite                # NEW: FTS5 index for hybrid retrieval, gitignored
│   └── onboarding_docs/          # NEW: synthetic document images, identities.json, *.expected.json
├── scripts/
│   ├── ingest.py                 # reused, confirm/add recursive glob under data/raw/
│   └── make_onboarding_docs.py   # NEW: renders + degrades the OCR dataset
├── src/
│   ├── config.py                 # + vision model, OCR/quality/validation thresholds, RETRIEVER_MODE, RRF_K
│   ├── monitoring.py             # + OCR/quality/validation trace fields (allowlists, §4.4)
│   ├── agent/
│   │   ├── orchestrator.py       # + extract_document, validate_checklist, get_onboarding_status tools
│   │   ├── router.py             # + document_upload, document_status intents
│   │   ├── tools.py
│   │   ├── usage.py
│   │   ├── llm_client.py
│   │   └── prompts.py
│   ├── rag/
│   │   ├── chunking.py / embeddings.py / retriever.py   # reused unchanged
│   │   └── hybrid.py              # NEW: BM25 + dense RRF fusion (§3.4)
│   ├── ocr/                       # NEW (§4.2)
│   │   ├── doctypes.py            # document field registry + validators
│   │   ├── quality.py             # OpenCV quality gate + preprocessing
│   │   └── extractor.py           # Gemini multimodal extraction + response_schema
│   ├── guardrails/                # reused (input_checks.py, toxicity.py, pii.py, llm_judge.py, grounding.py)
│   │   └── doc_validation.py      # NEW: Rules 1–6, §4.1
│   ├── memory/                    # reused (session.py, persistent.py)
│   │   └── onboarding_status.py   # NEW: per-employee doc checklist state (§4.3)
│   ├── schemas.py                 # + OnboardingDocument, ExtractionResult, ValidationResult
│   ├── api.py                     # + POST /upload-doc, GET /onboarding-status/{id}
│   └── ui.py                      # + upload widget, checklist panel
├── evals/
│   ├── golden_set.jsonl           # reused + tiered handbook additions (§3.3)
│   ├── run_retrieval_eval.py      # extend: --retriever dense|hybrid, per-tier breakout
│   ├── guardrail_redteam.jsonl    # reused unchanged
│   ├── run_guardrail_eval.py      # reused unchanged
│   ├── run_ocr_eval.py            # NEW: classification + field-extraction accuracy, cached
│   ├── run_validation_eval.py     # NEW: doc_validation precision/recall
│   ├── run_answer_eval.py         # NEW: LLM-as-judge on FAQ answers, cached
│   └── results/
│       └── ocr_cache/             # NEW: disk-cached extraction results keyed by image SHA-256
└── tests/                         # existing suite + test_doc_validation.py, test_quality.py, test_hybrid.py
```

---

## 8. Build Order

Phase 0 is the critical path — everything OCR blocks on the dataset generator, everything RAG blocks on the corpus. Run the two Phase-0 tracks in parallel.

| Phase | Work | Blocks |
| --- | --- | --- |
| 0a | Handbook corpus (§3.1) + tiered golden set (§3.3) | all retrieval evals |
| 0b | Onboarding dataset generator (§4.2) | all OCR work |
| 1 | `doctypes.py`, `quality.py`, `extractor.py` (§4.2) | validation |
| 2 | `doc_validation.py` (§4.1), `onboarding_status.py` (§4.3) | agent wiring |
| 3 | Router intents, 3 new tools, `/upload-doc`, `/onboarding-status`, uploader widget, MLflow fields (§4.4) | demo |
| 4 | Hybrid retriever (§3.4) | retrieval ablation |
| 5 | All eval harnesses + trajectory-walkthrough capture (§7 experiments) | deck |
| 6 | RRL slide, architecture diagram update, UoM numbers (§6) | deck |

`README.md`'s architecture diagram and `docs/architecture.svg` currently omit the OCR path entirely — rubric criterion 2 (25%) requires the architecture to clearly show where the CV/DS model plugs in, so update both in Phase 6, along with the `CLAUDE.md` cross-reference that currently points at "§5" for the repo layout (it's §7 in this document).

---

## 9. Experiments to Report

Spec requires ≥3 quantitative eval metrics with interpretation, at least one full reasoning-trace walkthrough, and a more rigorous suite than the Midterm's sample-output grading.

| Experiment | Metric | Notes | Status |
| --- | --- | --- | --- |
| OCR classification | Doc-type accuracy + 6×6 confusion matrix (all 6 types) | Break out by degradation variant | 🆕 new |
| OCR field extraction | Per-field exact match + normalized CER (NBI, BIR 2316) | **Preprocessing on/off ablation** — same ground truth across clean/degraded pairs (§4.2) makes this a controlled comparison, not just a headline number | 🆕 new |
| Document validation | Precision/recall/F1 on `needs_review` vs. `accepted`; per-rule trigger counts | **Headline metric: false auto-pass rate** — accepting an invalid document is the costly failure mode | 🆕 new |
| Onboarding FAQ retrieval | Hit-rate@k, MRR, per difficulty tier (§3.3); abstention accuracy on the `negative` tier; dense vs. hybrid (§3.4) | Answers research question (a) as a **curve with a failure taxonomy**, not a single "100%" claim — expect `lookup` near-ceiling and `multihop`/`near_miss` to be where the real ceiling sits | 🔄 extend |
| Guardrail red-team (inherited) | Block rate (injection/toxicity), detection rate (PII) | Already measured: 100%/100% on 20 adversarial prompts. Reused unchanged. | ✅ passing, reused |
| FAQ answer quality | LLM-as-judge faithfulness/citation accuracy | New `run_answer_eval.py`, disk-cached (§2.1) | 🆕 new |
| Agent trajectory | Scripted walkthrough of one full decision chain: FAQ question → route → clarify → upload → quality gate → extract → validate → checklist update → grounded reply | For the required reasoning-trace slide | 🆕 new |
| Latency & cost | p50/p95 latency, tokens/request, from MLflow | Chat-only vs. chat+OCR; Gemini vs. Ollama | 🔄 extend |

Document failure modes for the Retrospective: low-quality photo submissions, name-match false negatives, router confusion between "asking about a document" vs. "submitting one," and the sim-to-real gap in the synthetic OCR dataset (§4.2).

---

## 10. Key Risks & Mitigations

- **Gemini 20/day quota** — the binding constraint on this whole plan; already exhausted repeatedly during Midterm development (§2.1), and an OCR eval multiplies the pressure. Mitigated by: disk-cached extractions keyed by image hash; the quality gate rejecting before spending a call; one extraction call per document; budgeting a paid key before demo week regardless.
- **OCR accuracy on poor-quality phone photos** → confidence floor + mandatory human-review fallback (Rule 6); measured explicitly via the preprocessing ablation, not assumed.
- **Synthetic OCR documents are easier than real photos** → state the sim-to-real gap as a known limitation on the deck; optionally add a small photographed holdout (~10 printed mockups) later for an honesty check.
- **`pdf_to_md.py` breaks on the real DLSU manual** → deferred by the synthetic-first corpus decision (§3.1), but budget parser work at swap time — 273 pages of multi-column academic layout will stress a single-column font-size heuristic.
- **Scope creep back toward the Midterm's complaint/escalation flow** → the feature is fully removed (§1.3), not just deprioritized; resist re-building it mid-Final unless the team explicitly decides to reverse that call (it's recoverable from git history).
- **Ollama backend quality/availability for a live demo** → small local models are less reliable at strict tool-calling/schema conformance than Gemini, and OCR stays on Gemini regardless of `LLM_PROVIDER` (§2.1). Treat Ollama as a development escape valve, not the default presentation path, unless dry-run tested beforehand — and lock down the VM's auth regardless.
- **Mixed local/Docker ingestion corrupts the index** → hit three times during Midterm development: running `scripts/ingest.py` on the host and `docker compose up` against the same bind-mounted `data/chroma/` causes a Rust panic on open (chromadb cross-platform incompatibility). Not data-destructive, just wastes time. Pick one environment per index.
- **`opencv-python` in a slim Docker image** → use `opencv-python-headless`; the slim base image lacks `libGL`.
- **Real PII/government-ID exposure** → all OCR eval/demo documents are synthetic mockups; never use real employee documents (§1.3, §4.2).
- **Demo failure during presentation** → run fully local (Chroma + SQLite), record a fallback video, disclose upfront per spec.
