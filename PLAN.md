# Faculty Onboarding Concierge — Final Capstone Implementation Plan

**Course:** Introduction to Agentic AI (STAI100) — Final Capstone (Week 14)
**Use case type:** Vector DB RAG (primary) + CV/OCR Document Verification (secondary) — **faculty onboarding in a Philippine university**
**Reference corpus:** `data/faculty-manual-2021.pdf` — the **De La Salle University Faculty Manual (2021), 203 pages** (verified page count; earlier drafts of this plan said 273 — that was an estimate, not a measurement). Specific to DLSU, deliberately generalizable to other PH universities (see §1.4).
**LLM provider:** Google Gemini API (chat + embeddings), with a self-hosted Ollama server as a fully switchable fallback for both, independently, via `LLM_PROVIDER` / `EMBEDDING_PROVIDER` — see §2.1. OCR/document extraction is Gemini-multimodal specifically; see §5 RRL.
**Builds on:** the Midterm project ("HR FAQ & Complaint Chatbot"). Reused as-is: the RAG pipeline, the agent orchestrator (ReAct loop + tools), two-tier memory, guardrails (deterministic checks + LLM-judge), the LLM/embedding provider abstraction, and the API/UI/MLflow/Docker scaffolding. Complaint intake & escalation have been **removed from the codebase** — not just descoped (see §1.5).
**Governing spec:** `[Stratpoint x DLSU] Final Capstone - Project Specification.txt` (repo root) — supersedes the Midterm spec ([specs.md](specs.md), kept for reference; do not edit either spec file).

---

### The one-sentence problem statement (say this on slide 2, verbatim)

> **A newly hired DLSU faculty member has to satisfy a 203-page Faculty Manual and a stack of pre-employment paperwork before they can teach — and today both are answered by a human, one email at a time. Can an agent answer their Manual questions accurately enough to be trusted, and check their submitted documents well enough to be useful?**

### Research questions driving this plan

- **(a) — PRIMARY, this is where the metrics live.** *Can the agent reliably answer **faculty onboarding & Faculty Manual** questions on **three specific topics** (§1.2)?* Proof point: **can we reach ~100% evaluation accuracy on a handbook of ≥100 pages?** The Manual is 203 pages, so the scale question is answered by construction; what remains is the accuracy question. RAG metrics (§3.3, §9) are the headline numbers of this project.
- **(b) — SECONDARY.** *How well can the agent verify that a submitted pre-employment document is correct and complete?* Scoped to **one document type: the NBI Clearance** (§4.2). Reported as **secondary metrics**, explicitly subordinate to (a) on the deck, and split into **two separate subsets — real documents and mock documents — never pooled into one number** (§3.5).

The instructor's counterpoint on (b) — *"is this easy enough?"* — still governs the design: a clean mockup fed to a multimodal model returning JSON is trivial and reads as decorative. The defensible version needs a real decision layer in front of and around the extraction call — see §4.1.

---

## 1. Business Use Case

### 1.1 The narrowed problem

A new DLSU faculty hire faces two loads at once:

1. **A 203-page Faculty Manual** governing their employment — ranks, working hours and load, probation and permanency, leaves, benefits, the dress code, the Table of Offenses. Almost none of it is read cover-to-cover; it is consulted by question ("how many consultation hours do I owe per 3 units?", "when does my syllabus have to be out?", "what leave can I take in my first year?"). Today those questions go to a Department Chair, an AFED representative, or HR — repeatedly, by the same new hires each term.
2. **Pre-employment paperwork.** Some is faculty-specific and named in the Manual (original transcript of records, biodata/CV, three references, teaching demonstration, clearance from the immediate past employer *and concerned government agency*, certification of physical fitness to teach). Some is the standard PH statutory set every employee files (NBI Clearance, SSS, PhilHealth, Pag-IBIG, BIR). Someone in HR eyeballs each scan to check it is the right type, legible, unexpired, and belongs to the right person.

**Final Capstone scope — "Faculty Onboarding Concierge":** one conversational agent that (a) answers faculty onboarding and Faculty Manual questions grounded in the real Manual with citations, and (b) accepts a photo/scan of an **NBI Clearance**, extracts its fields via OCR/VLM, and validates it against deterministic rules — telling the new hire in the same conversation what is missing, invalid, or needs human review.

### 1.2 The three topics (this is the scope boundary for RAG)

Question (a) is not "any question about the Manual." It is three named topics, chosen because each is (i) genuinely onboarding-relevant, (ii) densely covered by real Manual text, and (iii) the kind of question a new hire actually asks in month one.

| # | Topic | What it covers | Where it lives in the Manual |
| --- | --- | --- | --- |
| **T1** | **Pre-employment & hiring requirements for faculty** | Faculty-specific: original TOR & diplomas, biodata/CV, three references, teaching demonstration, clearance from immediate past employer and concerned government agency, certification of physical fitness to teach, minimum entry requirements per rank. Plus the general PH statutory set (NBI, SSS, PhilHealth, Pag-IBIG, BIR) — see the gap note below. | *Hiring Procedure* (p.24 ff), *Criteria for Hiring* under each rank (pp.16–23), Part-time §C *Hiring Procedure* (p.67), ASF §C (p.83) |
| **T2** | **Academic & grading obligations** *(the "DOs and DON'Ts")* | Faculty prerogative over grades and the Change of Grade form; deadlines for submission of grades; syllabus provision; exam-material handling; classroom conduct; the sanctions attached to each. | *General Functions §1.1 Teaching* (p.8); grade-deadline language in the renewal/promotion criteria (pp.29–31, and per-rank); **Appendix F, Table of Offenses and Sanctions** (p.135) — #8 unauthorized possession of final exam questions, #13 tampering with grading records, #14 changing a grade for remuneration, #19 dress-code violation, #20 mobile-device disruption, #23 non-provision of syllabus within the first two weeks, #24 non-compliance with residency/working-hours and load |
| **T3** | **HR policies — dress code & leaves** | Attire and grooming rules; the full leave catalogue (service, sabbatical, study/training, research, vacation, sick — short-term and prolonged, emergency, military, secondment, parental, technology commercialization) and the general considerations governing all of them. | **Appendix D, Dress Code / Attire and Grooming Policy** (p.131); *Benefits §8 Leaves* (pp.42–48) for full-time, §G for part-time (p.73), §F.6 for ASF (p.93) |

**"Better word than DOs & DON'Ts":** use **"Academic & grading obligations"** in the docs and on the deck, and **"obligations and prohibitions"** when you need both halves. It is the register the Manual itself uses ("responsibilities," "offenses") and it does not read like a listicle.

#### Two verified gaps to resolve before writing the golden set

Both were found by reading the actual PDF, not assumed. Neither blocks the build, but both change what the golden set may legitimately contain.

- **T1 gap — the statutory documents are not in the Manual.** "NBI" appears **zero times**. SSS / PhilHealth / Pag-IBIG / BIR appear only in the *benefits* and *retirement* sections (contribution sharing, CEAP plan, Appendix K), never as pre-employment submissions. The nearest hook is the hiring criterion *"clearance from immediate past employer and concerned government agency"* and *"certification of physical fitness to teach."*
  **Resolution:** add one short supplementary corpus document, `data/raw/supplementary/ph-statutory-preemployment.md` — a cited summary of the standard PH statutory pre-employment requirements (NBI Clearance, SSS number, PhilHealth, Pag-IBIG MID, BIR 1902/2316), sourced from the agencies' own public guidance, clearly labelled as a *supplementary* source with its own `doc_id` and `source_url`. Citations in answers will then show `faculty-manual-2021` vs `ph-statutory-preemployment`, which is honest and demonstrates multi-source grounding. **Do not** write golden-set rows that expect the Manual to answer statutory questions.
- **T2 gap — several of the listed "don'ts" are not Faculty Manual content.** Pop quizzes, "give a major requirement at least 4 weeks before its deadline," grading curves, bonus points, "free cut," and student grade appeals are **not in the Faculty Manual**. They live in the DLSU **Student Handbook / Academic Policies and Regulations**. What the Manual *does* give T2 is listed in the table above, and it is enough material for a topic.
  **Resolution — pick one before Phase 0a:** (i) restrict T2 to Manual-backed items only (safest, still substantive), or (ii) add the DLSU Student Handbook / academic-policy issuances as a second real corpus document, which makes the multi-hop tier genuinely interesting (*"the Manual says X about grade deadlines, the academic policy says Y about requirement notice"*) at the cost of one more ingestion source. **Recommendation: (ii)**, because the "4 weeks before deadline" style of question is exactly what makes T2 feel real to a faculty audience — but only if the document can be obtained; fall back to (i) otherwise.

#### T3 alternatives, as requested

Dress code + leaves is a fine topic — both are richly and unambiguously covered. If a stronger third topic is wanted, these are the Manual sections with comparable or better density, ranked:

1. **Probation, renewal & permanency** (pp.29–33) — definition and duration of probation, procedure and criteria for renewal, effectivity/non-renewal, composition of the Renewal and Permanency Boards. **Strongest alternative:** it is the single most consequential thing a new faculty member does not understand in year one, it is procedurally precise (good for exact-answer scoring), and it is unambiguously *onboarding*, which T3-as-HR-policies only partly is.
2. **Working hours & load** (pp.9–15) — the 40-hour week, 25 in-University hours, 12 teaching hours, 2.5 consultation hours per 3 units, the three-preparation limit. Very high question density, very exact numbers.
3. **Grievance procedure & Table of Offenses** (Appendix F/G, pp.135–145) — overlaps T2, so it competes rather than complements.

If you swap, swap T3 → **probation/renewal/permanency** and keep dress code + leaves as a stretch slice in the regression tier.

### 1.3 Sanity check (spec §5)

*Could ChatGPT or a Google search alone suffice?* No. The DLSU Faculty Manual 2021 is not reliably in any general model's parametric memory, and the failure mode when it isn't is a confident, plausible, wrong answer about someone's employment terms. The agent must (1) retrieve the actual Manual text and cite the section, (2) abstain when the Manual does not cover the question rather than generalize from other universities, (3) run deterministic validation rules on submitted documents that no general chatbot has (name match against the faculty record, NBI validity window, required-field checks), and (4) hold per-hire state across turns ("NBI received and validated; still missing TOR and physical-fitness certification").

### 1.4 Generalizability

The build targets DLSU specifically because we have the real 2021 Manual. Nothing in the pipeline is DLSU-specific: the corpus loader takes any PDF/Markdown, the three topics map onto any PH university's faculty manual (every one of them has hiring criteria, academic obligations, and a leave/dress policy), and the pre-employment document set is national law, not institutional policy. **Swapping to another PH university's manual is a data change, not a code change** — say this on the deck; it is the difference between a class project and a product story.

### 1.5 What's explicitly out of scope

- **Complaint intake, harassment/safety escalation, human-in-the-loop ticket routing.** The Midterm's product. The form-driven consent-gate design (`src/guardrails/escalation.py`, `danger_scan.py`, `form_pii.py`, `escalation_state.py`), the `ComplaintTicket`/`Severity`/`TriggerRule`/`Escalation*` schemas, the consent-gate flow, the `/tickets/{id}` endpoint, and `evals/run_escalation_eval.py` have been **removed from the codebase entirely**. Recoverable from git history if ever needed. Note the Manual's own grievance procedure (Appendix G) is *corpus content*, not a feature — the agent answers questions about it, it does not file grievances.
- **Manual topics outside T1–T3** — ranks, promotion grids, retirement, research incentives, AFED by-laws, CEAP plan. They stay in the index (they are part of the same PDF and removing them would be artificial) but they are **not** the eval target. They serve as realistic distractor content, which is exactly what makes hit-rate on 203 pages a meaningful number.
- **Document authenticity / forgery detection.** Completeness and correctness only.
- **Deep extraction on document types other than NBI Clearance.** See §4.2.
- **Non-faculty university staff** (co-academic personnel, APSP) — the Manual covers faculty; a different handbook covers them.

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
        │  Faculty    ││  OCR extract tool   ││  validator   ││  session +   │
        │  Manual 2021││  ★ CV/DS domain     ││  (determin-  ││  persistent +│
        │  (Chroma +  ││  integration        ││  istic rules)││  onboarding  │
        │  BM25 RRF)  ││  (Gemini multimodal)││  NBI rules   ││  status      │
        └─────────────┘└─────────────────────┘└──────────────┘└──────────────┘
             PRIMARY              SECONDARY
          (main metrics)      (secondary metrics)

        Observability: MLflow tracing on every request (latency, tokens, tool calls,
        image quality metrics, OCR confidence, validation outcome, errors)
```

The CV quality gate (OpenCV, deterministic, no LLM call) runs **before** the Gemini extraction call — it can reject or flag an image outright, which is both a real accuracy decision and the primary quota-conservation mechanism (§2.1).

### Technology stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| LLM | `LLM_PROVIDER` switchable: Gemini 2.5 Flash (default) via `google-genai`, or self-hosted Ollama (default `gemma4:e4b`) via `src/agent/llm_client.py` | Fast + cheap for chat; function calling and JSON schema output. Ollama for quota-free testing — §2.1. |
| **CV/DS domain model (★ mandatory, Component 14)** | Gemini 2.5 Flash multimodal document extraction, `response_schema`-typed, gated by a deterministic OpenCV quality check | See §5 RRL for why this beats a standalone OCR engine here. Not exercised through the Ollama path. |
| Embeddings | `EMBEDDING_PROVIDER` switchable: `gemini-embedding-001` (768-dim) or Ollama `nomic-embed-text` (native 768-dim) | Independent of chat provider. Switching requires re-ingest against a cleared index. |
| Vector store | ChromaDB (persistent, local) + BM25 (SQLite FTS5) fused via RRF | Dense-only blurs the exact identifiers a real manual is full of — "Appendix F", "Assistant Professor 3 to 7", "2 1/2 hours", "Change of Grade form". See §3.4. |
| PDF parsing | `src/rag/pdf_to_md.py` | **Now on the critical path** — the Manual is a real, typeset, appendix-heavy PDF, not a synthetic Markdown file. See §3.2 and the risk in §10. |
| Structured data | SQLite | Memory tables stay; add per-hire onboarding-status records. |
| API | FastAPI | Add `POST /upload-doc`, `GET /onboarding-status/{employee_id}`. |
| UI | Streamlit | Add file-upload widget + checklist-status panel. |
| Monitoring | MLflow | Reused; add OCR/quality/validation trace fields. |
| Packaging | Docker | Reused; `docker compose up --build` verified with both backends. |

### 2.1 LLM Provider Abstraction — shipped (reused unchanged)

**Why:** Gemini's free tier caps `gemini-2.5-flash` at **20 `generate_content` requests/day per project**, not per-minute. One agent turn already costs several requests (1 router classification + up to `MAX_REACT_ITERATIONS`=5 loop calls + 2 more if `search_web` fires), so quota was exhausted mid-testing repeatedly. This binds the OCR eval hardest: a naive run over 60 images is 60 calls.

**Mitigations shipped:**
- `run_turn()` catches `google.genai.errors.APIError` (and `LLMBackendError`) and degrades to a plain-language "try again" reply instead of a raw 500.
- `src/agent/usage.py` logs every agent LLM call's token usage to SQLite tagged by serving model; `GET /usage` reports today's + all-time counts per model.
- `search_web` uses **Tavily** (domain-restricted via `include_domains`) plus one `response_schema` call to shape results.
- **`src/agent/llm_client.py`** — an `OllamaClient` adapter mimicking `google-genai`'s call signature and response shape. `LLM_PROVIDER=ollama`; verified end-to-end against a real Ollama VM (`gemma4:e4b`).
- Embeddings switchable independently via `EMBEDDING_PROVIDER`; `config.get_embedder()` dispatches, and `scripts/ingest.py`, `src/rag/retriever.py`, `src/rag/answerer.py` all route through the config accessors rather than hardcoding a class.
- **Not extended to OCR.** `extract_document` is Gemini-multimodal only. If the Ollama path serves the live demo, OCR calls still hit Gemini — factor that into quota planning.
- **Disk-cached extraction.** Every extraction result caches to `evals/results/ocr_cache/<sha256 of image bytes>.json`. Eval re-runs cost zero API calls once populated; a full eval can be spread across days. `run_ocr_eval.py` and `run_answer_eval.py` read the cache by default; `--no-cache` forces live calls. Budget a paid key before demo week regardless.

**Switching `EMBEDDING_PROVIDER` requires re-running `scripts/ingest.py` against a cleared index** (`data/chroma/`, `data/index_manifest.json`).

**Known limitations:** small local models (`gemma4:e2b`/`e4b`) are less reliable at strict JSON-schema conformance and tool-calling than Gemini — expect more clarifying questions or `MAX_REACT_ITERATIONS` fallbacks on Ollama, never a crash.

**Operational note:** the verification Ollama VM was found reachable with no authentication — lock it down before relying on it for a live demo.

---

## 3. Knowledge Base & Data Pipeline

Pipeline mechanics (parsing, chunking, embedding, indexing) are **reused** from the Midterm. The corpus is now **real**, which is the substantive change.

### 3.1 Source corpus

**Primary — `data/faculty-manual-2021.pdf` (DLSU Faculty Manual 2021, 203 pages).** This replaces the previously planned 12-chapter synthetic PH handbook entirely. The synthetic-handbook plan existed only because a real ≥100pp manual wasn't in hand; it is now, so writing 54k words of fake handbook would be strictly worse — it would make the "100% on a real manual" claim unfalsifiable.

Structure that matters for chunking and citations:

- Front matter: Vision/Mission, Statement on Diversity and Inclusivity, Twelve Virtues, Code of Ethics, Responsibilities & Rights (pp.1–7)
- **Full-time Academic Faculty** (pp.8–52) — employment norms, working hours & load, ranks, **hiring procedure (T1)**, promotion, **probation & permanency**, grievance, severance, retirement, **benefits incl. leaves (T3)**
- **Part-time Academic Faculty** (pp.53–74) — parallel structure, own hiring procedure and benefits
- **Academic Service Faculty** (pp.75–102) — parallel again
- **Appendices A–N** (pp.104–203) — implementing guidelines, hiring/promotion grids, **Dress Code (Appendix D, p.131, T3)**, **Table of Offenses and Sanctions (Appendix F, p.135, T2)**, grievance procedures, AFED by-laws, CEAP plan, SSS, Safe Spaces policy

Note the three-way parallel structure (full-time / part-time / ASF each have their *own* hiring procedure, leaves, and benefits). **This is the single biggest retrieval hazard in the corpus** — a question like "how much service leave do I get?" has three different correct answers depending on faculty class. Two consequences, both load-bearing:
- `section_path` metadata must carry the faculty class, and the chunk context header must too (`"Faculty Manual 2021 > Part-time Academic Faculty > Benefits > Leaves"`), otherwise citations are ambiguous and retrieval mixes classes.
- The golden set must include a **disambiguation tier** (§3.3) of faculty-class-ambiguous questions where the *correct* behaviour is to ask which class the user is, not to answer.

**Supplementary — `data/raw/supplementary/ph-statutory-preemployment.md`** (new, small): the general PH statutory pre-employment requirements the Manual does not cover (§1.2 T1 gap). Own `doc_id`, own `source_url` frontmatter, clearly attributed in citations.

**Optional second real source — DLSU Student Handbook / Academic Policies**, if obtainable, to close the T2 gap (§1.2). Decide in Phase 0a.

The original 10 synthetic HR docs are **retired from `data/raw/`** (they are still in `data/processed/` from a prior ingest — clear that directory and the index at the start of Phase 0a). They were generic-corporate content that no longer matches the domain; keeping them would pollute retrieval with plausible-but-wrong non-DLSU policy. Their golden-set rows are dropped along with them.

### 3.2 Pipeline stages

1. **Parse & normalize** — `src/rag/pdf_to_md.py` converts the Manual (heading reconstruction from font-size/bold cues, table rendering, header/footer stripping). Output: `data/processed/faculty-manual-2021.md` with YAML frontmatter (`doc_id`, `title`, `category`, `effective_date: 2021`, `version`, `source_url`). **This step is now critical-path and must be manually spot-checked** — see §10.
2. **Chunk (structure-aware)** — split on headings first, pack to ~400 tokens with ~50-token overlap, merge sections under 80 tokens into their parent, keep tables whole (Appendix F's Table of Offenses and the Appendix B/C grids are tables — verify they survive), prepend a `"{title} > {section path}"` context header including faculty class (§3.1).
3. **Metadata enrichment** — `doc_id`, `chunk_id`, `title`, `section_path`, `faculty_class` (`full_time` | `part_time` | `asf` | `general`), `topic` (`T1` | `T2` | `T3` | `other`), `page_start`, `page_end`, `category`, `effective_date`, `version`, `token_count`. **`page_start` is new and required** — a citation into a 203-page manual must give a page number to be checkable by a human.
4. **Embed & index** — via `config.get_embedder()`; persistent Chroma collection keyed by `chunk_id`; `data/index_manifest.json` tracks hashes for idempotent re-ingestion.
5. **Retrieval (query time)** — top-k = 8, dense cosine by default; optional hybrid mode (§3.4); optional `category`/`faculty_class` filter; similarity floor (~0.5, tuned in evals) triggers "I don't know" + HR/Chair routing rather than a parametric-memory answer.
6. **Evaluation set** — golden Q&A pairs in `evals/golden_set.jsonl` (§3.3).

Expected scale: 203 pages of typeset text ≈ 60–75k words ≈ **250–330 chunks** at the 400-token target. Confirm the actual count after the first ingest and record it — it is a number the deck should state.

### 3.3 Golden set — tiered, per topic, for an honest "100%?" answer

**All legacy rows are dropped** with the synthetic corpus (§3.1). The new set is built from scratch against the real Manual, **~120 rows**, every row citing a real page. Rows are tagged `topic` (T1/T2/T3) *and* `difficulty`, so metrics break out both ways — the headline claim is per-topic, which is what research question (a) asks for.

| Tier | Count | What it measures |
| --- | --- | --- |
| `lookup` | 45 | Single-section factual retrieval (15 per topic) |
| `multihop` | 25 | Requires synthesizing 2+ sections (e.g. a T2 question spanning General Functions and Appendix F) |
| `disambiguation` | 15 | **Faculty-class-ambiguous** (§3.1) — correct behaviour is a clarifying question, not an answer |
| `negative` | 20 | **Not answerable from the corpus** — measures abstention, not recall (e.g. student-handbook grading policy if the Student Handbook is not ingested) |
| `near_miss` | 15 | Wording close to a real section but asking something the Manual doesn't say (e.g. a part-time benefit that only full-timers get) |

Row schema: `id`, `question`, `topic`, `difficulty`, `expected_doc_id`, `expected_section_path`, `expected_page`, `expected_answer_key_facts[]`, `expect_abstention` (bool), `expect_clarification` (bool).

The `negative`, `near_miss`, and `disambiguation` tiers exist specifically so "100%" cannot be gamed by an agent that over-answers. **Abstaining correctly scores as correct.** Report the headline as *per-topic accuracy with the tier breakdown visible* — a single pooled "100%" across a set you designed yourself is not a claim anyone should believe, and saying so is a strength on the deck, not a weakness.

**Authoring rule (non-negotiable):** every row must be traceable to text that is actually in the ingested corpus. If you cannot point at the page, the row does not go in the set. This is what the T1/T2 gap notes in §1.2 are protecting.

### 3.4 Advanced RAG — hybrid retrieval (Component 12)

New module `src/rag/hybrid.py`: a BM25 index (SQLite FTS5) over the same chunks, fused with dense results via Reciprocal Rank Fusion:

```
score(chunk) = Σ_r 1 / (RRF_K + rank_r(chunk))
```

`RRF_K` in `src/config.py`. Wrapped behind the existing `Retriever` interface, so `answerer.py` and `tools.py` need no changes. Selected via `RETRIEVER_MODE=dense|hybrid`.

**Why hybrid and not query-rewrite/rerank:** dense embeddings blur exact identifiers, and this Manual is dense with them — "Appendix F", "Assistant Professor 3 to 7", "ASF II-1 to II-9", "2 1/2 hours", "Change of Grade form", "twelve (12) hours". Hybrid fusion targets exactly that at **zero extra LLM calls at query time**. Query rewriting and LLM reranking each cost a Gemini call per query against the 20/day cap (§2.1) — rejected on quota grounds, documented as such for the RRL.

### 3.5 NBI Clearance image set — two separate subsets

New data track `data/references/`, containing **two subsets that are evaluated and reported separately, never pooled** (this split is the instructor's explicit ask):

**(a) `real/` — real NBI Clearances.** Small (target 8–15 documents), collected only from team members and consenting volunteers, **each with recorded informed consent**, used solely to measure sim-to-real gap. Handling rules, all mandatory:
- **Never committed to git.** `data/references/real/` is gitignored; the directory ships with a `README.md` and a `.gitkeep` only.
- Ground-truth JSON stores field values needed for scoring; the **NBI ID number and full name are stored hashed** in anything that leaves the machine, and raw values never reach MLflow (§4.4 allowlists are fail-closed already).
- No real document is used in a screenshot, slide, or recorded demo — the demo uses the mock subset.
- Contributors can withdraw; deletion means deleting the image, the ground truth, and the cached extraction keyed by its hash.

**(b) `mock/` — synthetic NBI Clearances,** generated programmatically (§4.2) with paired ground truth, covering the degradation variants and the negative cases. This is the subset that appears in demos, slides, and the repo.

Reporting: every OCR/validation metric in §9 is a **pair of numbers, (real, mock)**, with the real-subset n stated next to it. The gap between them *is* a finding — it is the sim-to-real honesty check the previous plan could only promise.

---

## 4. Component Breakdown & Ownership

Maps to the Final spec's 14-component checklist. Each member owns ≥2 components; **Component 14 is mandatory for the team**. This plan claims 11 against a floor of 8.

| # | Component | Implementation for the Final | Status |
| --- | --- | --- | --- |
| 3 | **RAG** | ChromaDB + switchable Gemini/Ollama embeddings, cited grounded answers (`src/rag/answerer.py`). Corpus is the real 203-page Faculty Manual (§3.1); citations now carry page numbers. | 🔄 extend (real-corpus swap) |
| 2 | **Disambiguation** | Reused router (`src/agent/router.py`). Add `document_upload`, `document_status` intents. **New load-bearing use: faculty-class disambiguation** (§3.1) — full-time vs part-time vs ASF changes the answer, so the clarifying-question mechanism is now scored (`disambiguation` tier, §3.3), not incidental. | 🔄 repurpose + extend |
| 4 | **Memory** | Reused two-tier memory. Add `src/memory/onboarding_status.py` — per-hire checklist state, including remembered faculty class so the disambiguation question is asked once, not every turn (§4.3). | 🔄 extend |
| 5 | **Guardrails** | Input-safety layers reused unchanged (`input_checks.py`, `toxicity.py`, `pii.py`, `llm_judge.py`). NEW: `doc_validation.py`, fail-toward-safety NBI validation (Rules 1–6, §4.1). | 🔄 new (input layers reused) |
| 6 | **Simple Chat UI** | Reused Streamlit app; add file-upload widget + checklist panel. | 🔄 extend |
| 7 | **API Endpoint Deployment** | Reused FastAPI app; add `POST /upload-doc`, `GET /onboarding-status/{employee_id}`. | 🔄 extend |
| 8 | **LLMOps (monitoring/tracing)** | Reused MLflow wiring; extend fail-closed allowlists with OCR/quality/validation fields (§4.4). | 🔄 extend |
| 9 | **ReAct / Tool Use** | Reused orchestrator loop (`MAX_REACT_ITERATIONS`=5, fail-closed). Add `extract_document`, `validate_checklist`, `get_onboarding_status`. | 🔄 extend |
| **12** | **Advanced RAG** | Hybrid BM25 + dense retrieval with RRF fusion (§3.4). | 🆕 new |
| **14** | **CV/DS Domain Integration ★ mandatory** | OpenCV quality gate + Gemini multimodal NBI extraction (`response_schema`) + deterministic validation (§4.1–§4.2, §5). | 🆕 new — the one ground-up build |
| 13 | **Evals** | Reused retrieval hit-rate@k harness and guardrail red-team. Add `run_ocr_eval.py`, `run_validation_eval.py`, `run_answer_eval.py` (§7). | 🔄 extend |

**Note on Guardrails (`llm_judge.py`):** the shipped design is a separate `llm_judge.py` (`response_schema=LLMJudgeVerdict`) classifying five dimensions (toxicity, PII, injection, off-topic, jailbreak) in one structured call, wired into `orchestrator._check_input()` after the deterministic checks. Verified against the file, not assumed.

**Ownership** (kept in sync with [README.md](README.md) and [SCOPE.md](SCOPE.md)):

| Member | Components |
| --- | --- |
| Baybayon | RAG, Guardrails, ReAct Tools (web search, calling the CV integration) |
| Del Rosario | CV Integration and its evaluation |
| Burayag | RRL, end-to-end evals |
| Tamondong | Evals dataset, Chat UI, API endpoint, LLMOps |

### 4.1 Document Validation Rules (Guardrails detail)

New module `src/guardrails/doc_validation.py`, applying **fail toward "needs human review," never silently accept**. Scoped to the NBI Clearance, with the registry structured so a second doc type is additive.

- **Rule 1 — Type match.** Extracted `doc_type` must be `nbi_clearance` and must match what the checklist is expecting; mismatches prompt re-upload, never a guess.
- **Rule 2 — Completeness (fields filled up).** Every required NBI field present and non-empty: **full name, date of issue, NBI ID / reference number, purpose**. This is the instructor's completeness check, and "fields filled up (name, date)" is exactly this rule.
- **Rule 3 — Format validity.** Each field passes its registry validator — name is non-numeric and has ≥2 tokens, date of issue parses to a real date not in the future, NBI reference matches its printed pattern. A field failing format is untrusted regardless of model confidence — a deterministic signal the LLM cannot talk past.
- **Rule 4 — Identity match.** Extracted name matched against the faculty member's on-file record (`rapidfuzz.token_set_ratio`, threshold in config), handling PH name forms (middle initials, `Jr./III` suffixes, surname-first, ñ). Non-matches flag for human review — never auto-reject, never auto-accept.
- **Rule 5 — Validity window (the 6-month rule).** `date_of_issue + NBI_VALIDITY_MONTHS` checked in code against an injectable `as_of` date. **`NBI_VALIDITY_MONTHS` defaults to 6** per the user requirement and common PH employer practice. **State this clearly on the deck:** the clearance itself is printed with a one-year validity; the six-month window is an *employer* freshness policy, not the document's own expiry. Encoding it as config (not a constant) is what makes that distinction visible and tunable per institution — and the injectable `as_of` keeps expiry tests deterministic and non-rotting.
- **Rule 6 — Fail-safe.** Any schema-validation failure, unparseable date, or a `warn`-level image-quality verdict (§4.2) paired with low model confidence escalates to human review by default.

Composite confidence = `min(normalized_image_quality, model_confidence)`, forced to zero by any Rule 3 failure — explainable and defensible on the deck. Outcome is one of `accepted | rejected | needs_review`; the module never silently accepts.

New code: `validate_document(extracted, faculty_record, as_of) -> ValidationResult`; `OnboardingDocument` / `ExtractionResult` / `ValidationResult` in `src/schemas.py`; thresholds and the validity window in `src/config.py`.

### 4.2 CV/OCR pipeline (Component 14 detail)

New package `src/ocr/`. **Scope: one document type, done deeply — the NBI Clearance.** The instructor's brief is "choose at least 1"; the previous plan's six-type classification matrix has been cut, because a shallow six-way confusion matrix bought less than a deep, honestly-evaluated single type — and because the CV track is explicitly *secondary* to RAG here. The registry is built so adding a second type (BIR 2316 is the obvious candidate) is a data change; treat that as a stretch goal only after the RAG metrics are done.

**`src/ocr/doctypes.py`** — declarative registry, one entry per document type: canonical name, aliases, required/optional fields, per-field format validator, validity window.

| Doc type | Key fields | Depth |
| --- | --- | --- |
| **NBI Clearance** | **full name, date of issue, NBI ID / reference no., purpose** | ★ deep — extraction + full Rules 1–6 validation |
| BIR 2316, SSS, PhilHealth, Pag-IBIG, physical-fitness certification | — | Stretch only; registry entries may be stubbed for the checklist's "still missing" message without any extraction |

**`src/ocr/quality.py`** — OpenCV, no LLM call. This module is the direct answer to "is this easy enough":
- `blur_score` (variance of Laplacian), `exposure_clip` (glare/underexposure via histogram clipping), `skew_deg` (`minAreaRect` over the largest text contour), `min_dim_px` (resolution floor), `quad_found` (document-boundary contour detection).
- Returns an `ImageQualityReport` with verdict `pass | warn | reject` and reasons. A `reject` short-circuits **before** any Gemini call — a real accuracy decision that also directly relieves the §2.1 quota constraint.
- `preprocess()` — deskew, perspective-correct to the detected quad, CLAHE contrast, upscale to the resolution floor. Accuracy with vs. without is a headline ablation (§9).

**`src/ocr/extractor.py`** — one Gemini call per document (quota discipline): `response_schema` returns `doc_type` plus the NBI field set. Each field carries `value`, `verbatim_text` (what the model literally saw, for auditability), and `model_confidence`. Reuses the `response_schema` pattern already proven in `answerer.py`, `tools.py`, `router.py`, `llm_judge.py`, `persistent.py`.

Vision calls use `GEMINI_VISION_MODEL` (default `gemini-2.5-flash`), separate from `GEMINI_CHAT_MODEL` — the lite chat tier is not reliable at multimodal field extraction.

**Mock dataset — `scripts/make_onboarding_docs.py`:**
- 8 synthetic identities (`data/references/mock/identities.json`) — doubles as the faculty record backing Rule 4.
- A Pillow-rendered NBI Clearance layout template.
- 5 degradation variants per document: `clean`, `skew` (8–15° perspective warp), `blur` (gaussian k=7–11), `glare` (radial overlay + brightness clip), `lowres_jpeg` (0.35 downscale, q=35).
- 8 identities × 5 variants = **40 mock images**, each variant sharing one `<identity_id>.expected.json` with its clean counterpart — which is what makes the preprocessing ablation a controlled comparison rather than a headline number.
- **10 negative cases:** a non-NBI document in the NBI slot, a non-document photo, a blank page, a clearance belonging to a different person (Rule 4), an expired clearance (Rule 5).
- **Known limitation, stated on the deck:** rendered-then-degraded images are easier than real phone photos. Unlike the previous plan, this is now *measured* rather than conceded — that is what the real subset (§3.5a) is for.

**Real subset handling:** see §3.5(a). Consent, gitignore, hashed identifiers, mock-only demos.

### 4.3 Checklist state (Memory detail)

`src/memory/onboarding_status.py`. SQLite table `onboarding_documents(employee_id, doc_type, status, outcome, validated_at, source_hash)`, reusing the `_get_connection()` pattern from `src/memory/session.py`. `get_status(employee_id) -> ChecklistStatus` powers replies like "NBI clearance validated; still missing TOR and physical-fitness certification."

Also stores the hire's **faculty class** (full-time / part-time / ASF) once resolved, so the disambiguation question (§3.1, §4.4) is asked once per session and then used as a retrieval filter.

### 4.4 Interfaces & ops (UI/API/LLMOps detail)

- **Router** — add `DOCUMENT_UPLOAD`/`DOCUMENT_STATUS` to `Intent` in `src/schemas.py`; update `ROUTER_PROMPT`. Explicitly test the *asking about* vs. *submitting* a document confusion. Add the faculty-class clarifying question to the disambiguation path.
- **Tools** — register `extract_document`, `validate_checklist`, `get_onboarding_status` in `orchestrator._function_declarations()`.
- **`POST /upload-doc`** — FastAPI `UploadFile` (needs `python-multipart`). Enforce `MAX_UPLOAD_BYTES`; validate MIME by magic bytes, not extension; **strip EXIF before any storage or logging** (geolocation is PII — and the real subset makes this a live concern, not a hypothetical); don't persist the raw image past the request unless explicitly configured.
- **`GET /onboarding-status/{employee_id}`**.
- **UI** — `st.file_uploader` in the composer, checklist panel in the sidebar, and a **page-number citation** rendered next to each Manual answer. UI talks only to FastAPI, never to Gemini directly.
- **MLflow** — `src/monitoring.py` uses fail-closed allowlists; extend `_ALLOWED_TAG_KEYS` with `doc_type`, `validation_outcome`, `quality_verdict`, `topic`, `faculty_class`, and `_ALLOWED_METRIC_KEYS` with `blur_score`, `skew_deg`, `extraction_confidence`, `fields_extracted`, `fields_missing`, `ocr_latency_ms`. **No extracted field value ever becomes a tag** — names and ID numbers are PII.

### New config values — `src/config.py`

`GEMINI_VISION_MODEL`, `OCR_CONFIDENCE_FLOOR`, `BLUR_VARIANCE_FLOOR`, `MIN_IMAGE_DIM_PX`, `MAX_SKEW_DEG`, `NAME_MATCH_THRESHOLD`, `NBI_VALIDITY_MONTHS` (default 6), `REQUIRED_ONBOARDING_DOCS`, `MAX_UPLOAD_BYTES`, `ALLOWED_IMAGE_MIME`, `RETRIEVER_MODE`, `RRF_K`.

### New dependencies

`opencv-python-headless` (not `opencv-python` — the slim Docker base lacks `libGL`), `Pillow`, `rapidfuzz`, `python-multipart`. Pin in `requirements.txt` and `requirements-api.txt`.

---

## 5. Review of Related Literature (RRL) — OCR/Document Model Choice

Per spec §5: state-of-the-art options considered, expected input/output, and why one was chosen.

| Option | Input → Output | Trade-offs |
| --- | --- | --- |
| **Gemini 2.5 Flash multimodal + `response_schema`** (chosen) | image/PDF → typed JSON fields directly | No new infra (same SDK/key already in use); reuses the Structured Outputs pattern already in `src/schemas.py`; weaker on precise bounding-box/layout output than a dedicated OCR engine, but this project needs field values, not layout. Consumes Gemini quota (§2.1); not available on the Ollama path. |
| Google Cloud Vision OCR | image → raw text + bounding boxes | Mature, cheap per call; needs a second Google Cloud service/credential and a separate field-parsing step on top of raw text. |
| Google Document AI (form/ID parser) | image → pre-structured key-value pairs | Purpose-built for ID/form parsing, likely highest raw accuracy; heavier setup (processor provisioning), overkill for a PoC-scoped, single-document-type eval. |
| Tesseract / EasyOCR (open-source) | image → raw text | Free, fully local, no API dependency; weakest accuracy on skewed/low-quality phone photos — exactly the failure mode the real subset (§3.5a) targets. |

**Decision:** Gemini 2.5 Flash multimodal extraction — lowest integration cost, reuses existing schema-validation infrastructure, and the PoC's bar is "flag for human review when uncertain" rather than "fully automate," which fits a general-purpose multimodal model better than a narrow layout-tuned OCR engine. Revisit if the eval (§9) shows the confidence floor firing too often to be useful, or if quota pressure makes a local engine more attractive despite the accuracy trade-off.

**A second, deliberate design choice sits in front of the model call: the OpenCV quality gate (§4.2) is a separate, non-LLM, deterministic layer.** This is the direct answer to "is this easy enough?" — a bare multimodal call would be decorative; gating it on measurable image-quality signals and backing it with deterministic rules (§4.1) is what makes the CV component load-bearing.

**Retrieval-side RRL (§3.4):** query rewriting and LLM reranking were considered and rejected on quota grounds (each adds a call per query against the 20/day cap), in favour of BM25+dense RRF fusion, which is free at query time and targets the same failure mode (exact-identifier misses) that a 203-page manual full of rank codes and appendix numbers surfaces.

---

## 6. Business Use Case, Competitor Analysis & Unit of Measurement

### 6.1 Who has this problem, quantitatively

State the pain point in numbers before proposing a solution — the spec and the instructor both ask for this.

| Input | Value | Basis (fill the source in before the deck; do not assert unsourced) |
| --- | --- | --- |
| Faculty headcount served (DLSU) | `___` | DLSU published faculty count — cite the year and page |
| New/renewing faculty per term | `___` | Ask HR or estimate from hiring-board frequency; state which |
| Manual-question volume per new hire, first term | `___ questions` | **Needs a real basis** — a 10-person informal survey of recent hires is enough and is honest; flag as an estimate |
| Minutes per question answered by Chair/HR/AFED | `___ min` | Same survey; include the asker's wait time separately from the answerer's time |
| Documents checked per hire | 5–8 | The Manual's hiring criteria + statutory set (§1.2 T1) |
| Minutes per document eyeballed | `___ min` | Team estimate; flag as such |
| Loaded hourly cost of the person answering | `≈₱___/hour` | For a Department Chair or HR generalist — cite the salary range source used |
| FAQ deflection rate | `___%` | **Taken directly from the measured per-topic correct-and-confident rate (§3.3, §9), not guessed** |

Output as **hours/month saved** and **₱/term**, both explicitly qualified by period (spec §6.3). Intangibles — faster time-to-teaching-ready, consistent answers across departments, fewer "ask three people, get three answers" incidents — listed separately and flagged as unmonetized.

*(The derivation path, not the placeholder, is the point. The deflection rate in particular must come from measured eval output.)*

### 6.2 Competitor / alternative analysis

What a new faculty member does *today*, and why each alternative falls short. This is the slide that stops "why not just use ChatGPT?" from being asked from the floor.

| Alternative | What it does well | Why it doesn't solve this |
| --- | --- | --- |
| **Ask the Department Chair / AFED rep / HR** | Authoritative, handles nuance | Doesn't scale, answers vary by who you ask, high-latency (days), and it is the cost this project is trying to reduce |
| **ChatGPT / Claude / Gemini, unaided** | Fluent, instant, free | The DLSU Faculty Manual 2021 is not reliably in parametric memory. The failure mode is a confident wrong answer about someone's employment terms — worse than no answer. No citations, no abstention, no per-hire state |
| **Ctrl-F the PDF** | Free, authoritative, always available | Requires knowing the Manual's vocabulary ("residency," "reclassification," "ASF") to find anything; three parallel faculty-class structures mean a naive search returns the wrong class's answer; no synthesis across sections |
| **University intranet / static onboarding FAQ page** | Curated, low-tech | Fixed question set; goes stale against Manual revisions; can't handle "does that apply to me as part-time?" |
| **HRIS onboarding modules** (Sprout, PayrollHero, Workday, SAP SF) | Checklist tracking, document storage, workflow | Document *collection*, not document *verification* — they store what you upload without checking the fields are filled, the name matches, or it's within the freshness window. And none of them answer manual questions. Complementary, not competing |
| **DocuSign / generic e-signature + manual review** | Compliant collection | Still a human eyeballing every scan |

**The gap this fills:** grounded, cited, abstention-capable Q&A over *this institution's own* manual, plus deterministic field-level document checks, in one stateful conversation. No single alternative above does two of those three.

---

## 7. Repository Layout

```
stai-capstone/
├── CLAUDE.md
├── PLAN.md                      # this file
├── README.md
├── SCOPE.md
├── specs.md                                                    # Midterm spec (superseded, reference only)
├── [Stratpoint x DLSU] Final Capstone - Project Specification.txt  # governing spec
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/
│   ├── faculty-manual-2021.pdf   # PRIMARY CORPUS — DLSU Faculty Manual 2021, 203pp
│   ├── raw/
│   │   └── supplementary/
│   │       └── ph-statutory-preemployment.md   # NEW: statutory reqs the Manual doesn't cover (§1.2)
│   ├── processed/                # faculty-manual-2021.md (+ supplementary) — clear old synthetic output first
│   ├── chroma/                   # gitignored (dense vectors)
│   ├── bm25.sqlite               # NEW: FTS5 index for hybrid retrieval, gitignored
│   └── references/               # renamed from onboarding_docs/ during the CV rework
│       ├── real/                 # NEW: consented real NBI + government ID docs — GITIGNORED, never committed (§3.5a)
│       ├── mock/                 # NEW: synthetic NBI + government ID images, identities.json, *.expected.json
│       └── samples/              # NEW: gitignored specimen/demo layout-reference images (CV_INTEGRATION.md Part 5)
├── scripts/
│   ├── ingest.py                 # reused; recursive glob under data/raw/ + the top-level manual PDF
│   └── make_onboarding_docs.py   # NEW: renders + degrades the mock NBI dataset
├── src/
│   ├── config.py                 # + vision model, OCR/quality thresholds, NBI_VALIDITY_MONTHS, RETRIEVER_MODE, RRF_K
│   ├── monitoring.py             # + OCR/quality/validation trace fields (allowlists, §4.4)
│   ├── agent/
│   │   ├── orchestrator.py       # + extract_document, validate_checklist, get_onboarding_status
│   │   ├── router.py             # + document_upload, document_status, faculty-class disambiguation
│   │   ├── tools.py / usage.py / llm_client.py / prompts.py
│   ├── rag/
│   │   ├── pdf_to_md.py          # NOW CRITICAL PATH — real typeset PDF (§3.2, §10)
│   │   ├── chunking.py / embeddings.py / retriever.py
│   │   └── hybrid.py             # NEW: BM25 + dense RRF fusion (§3.4)
│   ├── ocr/                      # NEW (§4.2)
│   │   ├── doctypes.py           # NBI field registry + validators
│   │   ├── quality.py            # OpenCV quality gate + preprocessing
│   │   └── extractor.py          # Gemini multimodal extraction + response_schema
│   ├── guardrails/               # reused (input_checks.py, toxicity.py, pii.py, llm_judge.py, grounding.py)
│   │   └── doc_validation.py     # NEW: Rules 1–6, §4.1
│   ├── memory/                   # reused (session.py, persistent.py)
│   │   └── onboarding_status.py  # NEW: per-hire checklist + faculty class (§4.3)
│   ├── schemas.py                # + OnboardingDocument, ExtractionResult, ValidationResult
│   ├── api.py                    # + POST /upload-doc, GET /onboarding-status/{id}
│   └── ui.py                     # + upload widget, checklist panel, page citations
├── evals/
│   ├── golden_set.jsonl          # REBUILT against the real Manual, tiered + per-topic (§3.3)
│   ├── run_retrieval_eval.py     # extend: --retriever dense|hybrid, per-topic + per-tier breakout
│   ├── guardrail_redteam.jsonl / run_guardrail_eval.py     # reused unchanged
│   ├── run_ocr_eval.py           # NEW: NBI field accuracy, real vs mock subsets reported separately
│   ├── run_validation_eval.py    # NEW: doc_validation precision/recall
│   ├── run_answer_eval.py        # NEW: LLM-as-judge on Manual answers, cached
│   └── results/
│       └── ocr_cache/            # NEW: disk-cached extractions keyed by image SHA-256
└── tests/                        # existing suite + test_doc_validation.py, test_quality.py, test_hybrid.py
```

---

## 8. Build Order

Phase 0 is the critical path. **Phase 0a now carries more risk than in the previous plan** because the corpus is a real PDF rather than Markdown we authored — parse quality is a discovery, not a given. Start it first and do not run the two Phase-0 tracks in parallel unless 0a's parse is already spot-checked.

| Phase | Work | Blocks |
| --- | --- | --- |
| **0a** | Ingest `faculty-manual-2021.pdf`; **manually spot-check the parse** on the T1/T2/T3 sections and Appendices D and F; add `page_start`/`faculty_class`/`topic` metadata; write the supplementary statutory doc; decide the T2 Student-Handbook question (§1.2) | everything |
| 0b | Tiered per-topic golden set against the *parsed* corpus (§3.3) — cannot start before 0a's parse is trusted | all retrieval evals |
| 0c | Mock NBI dataset generator (§4.2); begin real-subset consent collection in parallel (§3.5a) — collection has human latency, start it early | all OCR work |
| 1 | `doctypes.py`, `quality.py`, `extractor.py` (§4.2) | validation |
| 2 | `doc_validation.py` (§4.1), `onboarding_status.py` (§4.3) | agent wiring |
| 3 | Router intents + faculty-class disambiguation, 3 new tools, `/upload-doc`, `/onboarding-status`, uploader widget, MLflow fields (§4.4) | demo |
| 4 | Hybrid retriever (§3.4) | retrieval ablation |
| 5 | All eval harnesses + trajectory walkthrough (§9) | deck |
| 6 | Problem-statement slide, competitor analysis, RRL slide, architecture diagram update, UoM numbers (§6) | deck |

`README.md`'s architecture diagram and `docs/architecture.svg` must show where the CV/DS model plugs in (rubric criterion 2, 25%) — update both in Phase 6.

---

## 9. Experiments to Report

Spec requires ≥3 quantitative eval metrics with interpretation and at least one full reasoning-trace walkthrough. **Present the RAG block first and label it PRIMARY; present the OCR block second and label it SECONDARY.** That ordering is the instructor's instruction, not a stylistic choice.

### PRIMARY — RAG / Faculty Manual Q&A

| Experiment | Metric | Notes |
| --- | --- | --- |
| **Retrieval on the Manual** | Hit-rate@k and MRR, **broken out per topic (T1/T2/T3) and per tier** (§3.3) | The headline answer to research question (a). Report as a curve with a failure taxonomy, not a single "100%" — expect `lookup` near-ceiling and `multihop`/`near_miss` to be where the real ceiling sits |
| **Abstention** | Accuracy on the `negative` and `near_miss` tiers | Guards the 100% claim against over-answering |
| **Disambiguation** | Clarify-vs-answer accuracy on the `disambiguation` tier | Faculty-class ambiguity (§3.1); a wrong-class answer is a *worse* failure than a clarifying question |
| **Dense vs. hybrid ablation** | Same metrics under `RETRIEVER_MODE=dense` and `hybrid` (§3.4) | Advanced-RAG component's justification |
| **Answer quality** | LLM-as-judge faithfulness + citation accuracy, incl. **page-number correctness** | `run_answer_eval.py`, disk-cached |
| **Guardrail red-team (inherited)** | Block rate (injection/toxicity), detection rate (PII) | Already measured: 100%/100% on 20 adversarial prompts. Reused unchanged |

### SECONDARY — NBI Clearance verification

Every metric below is reported as **two separate numbers, real subset and mock subset**, with each n stated. Never pooled.

| Experiment | Metric | Notes |
| --- | --- | --- |
| NBI field extraction | Per-field exact match + normalized CER on name, date of issue, reference no., purpose | **Preprocessing on/off ablation** on the mock subset (same ground truth across degradation variants) |
| Document validation | Precision/recall/F1 on `needs_review` vs `accepted`; per-rule trigger counts | **Headline: false auto-pass rate** — accepting an invalid document is the costly failure |
| Real-vs-mock gap | Delta on every metric above | This *is* the sim-to-real honesty check; a large gap is a legitimate finding, not a failure |

### Cross-cutting

| Experiment | Metric | Notes |
| --- | --- | --- |
| Agent trajectory | Scripted walkthrough of one full decision chain: Manual question → route → faculty-class clarify → grounded cited answer → NBI upload → quality gate → extract → validate → checklist update | The required reasoning-trace slide |
| Latency & cost | p50/p95 latency, tokens/request, from MLflow | Chat-only vs chat+OCR; Gemini vs Ollama |

Failure modes to document for the Retrospective: PDF parse errors on the appendix tables, faculty-class cross-contamination in retrieval, low-quality photo submissions, name-match false negatives on PH name forms, router confusion between *asking about* and *submitting* a document, and the measured sim-to-real gap.

---

## 10. Key Risks & Mitigations

- **`pdf_to_md.py` on the real Manual — now the top risk.** The previous plan deferred this behind a synthetic corpus; that buffer is gone. 203 pages of typeset academic layout with deep numbering (`2.2.1.5`), three parallel faculty-class structures, and large tables (Appendix B/C grids, Appendix F offenses) will stress a single-column font-size heuristic. **Mitigation:** budget parser work in Phase 0a explicitly, spot-check the T1/T2/T3 sections and Appendices D and F by hand before writing a single golden-set row, and fall back to a per-page text dump with heading regexes if the heuristic fails on the appendices.
- **Faculty-class cross-contamination.** Full-time / part-time / ASF sections are near-duplicates with different numbers — the exact case dense retrieval handles worst. Mitigated by `faculty_class` metadata, class in the chunk context header, the `disambiguation` golden tier, and the hybrid retriever.
- **Golden set written against text that isn't in the corpus** (the §1.2 T1/T2 gaps). Mitigated by the §3.3 authoring rule: no page, no row.
- **Gemini 20/day quota** — the binding constraint; already exhausted repeatedly during Midterm development. Mitigated by disk-cached extractions keyed by image hash, the quality gate rejecting before spending a call, one extraction call per document, and budgeting a paid key before demo week.
- **Real NBI clearances are real PII.** Consent recorded, gitignored, hashed identifiers, EXIF stripped, never in slides or recordings, deletable on request (§3.5a). Getting this wrong is the one failure in this project with consequences outside the course.
- **Real-subset collection stalls** (people are reluctant, reasonably). Mitigated by starting collection in Phase 0c, accepting a small n (8–15) and stating it, and by the mock subset carrying the demo regardless.
- **Scope creep back toward the Midterm's complaint/escalation flow** → fully removed (§1.5), not deprioritized.
- **Mixed local/Docker ingestion corrupts the index** → hit three times during Midterm development: running `scripts/ingest.py` on the host and `docker compose up` against the same bind-mounted `data/chroma/` causes a Rust panic on open. Pick one environment per index.
- **`opencv-python` in a slim Docker image** → use `opencv-python-headless`; the slim base lacks `libGL`.
- **Copyright/attribution on the Manual** → it is DLSU's, reproducible for non-commercial use with acknowledgement per its own notice page. Acknowledge DLSU as source on the deck and in the README; do not redistribute the PDF outside the course context.
- **Demo failure during presentation** → run fully local (Chroma + SQLite), record a fallback video using the *mock* subset only, disclose upfront per spec.
