# HR FAQ & Complaint Chatbot — Implementation Plan

**Course:** Introduction to Agentic AI (STAI100) — Midterm Capstone (Week 9)
**Use case type:** Vector DB RAG (HR/Legal domain)
**LLM provider:** Google Gemini API

---

## 1. Business Use Case

Employees constantly ask HR the same questions (leave policy, benefits, payroll dates, onboarding steps) and file complaints through inconsistent channels (email, chat, walk-ins). This creates repetitive load on HR staff and inconsistent, untracked complaint handling.

**The agent solves two workflows in one conversational interface:**

1. **HR FAQ (RAG):** Answer policy questions grounded in the company's HR knowledge base, with citations to the exact policy section.
2. **Complaint intake (agentic):** Detect when an employee wants to file a complaint, disambiguate intent, collect required fields through conversation, validate against a structured schema, file a ticket via a tool call, and escalate sensitive cases (e.g., harassment) to a human.

**Why agentic (per Section 8 of specs):** multi-step reasoning (classify intent → retrieve → clarify → act), unstructured data (policy PDFs), conversational memory, a real measurable workflow (deflection rate, complaint completeness), and a human-in-the-loop for safety-critical escalations.

---

## 2. Architecture

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
        │  1. Input guardrails (topic filter, PII scan, injection check)        │
        │  2. Intent router: FAQ | Complaint | Ambiguous → disambiguate         │
        │  3. Tool selection & execution                                        │
        │  4. Output guardrails (grounding check, PII redaction, tone)          │
        └──────┬───────────────┬────────────────┬──────────────┬────────────────┘
               │               │                │              │
        ┌──────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐ ┌────▼─────────┐
        │  RAG tool   │ │ Complaint    │ │  Escalation  │ │   Memory     │
        │  (ChromaDB  │ │ filing tool  │ │  tool (email │ │  session +   │
        │  + Gemini   │ │ (SQLite      │ │  /flag to    │ │  persistent  │
        │  embeddings)│ │  tickets)    │ │  HR human)   │ │  (SQLite)    │
        └─────────────┘ └──────────────┘ └──────────────┘ └──────────────┘

        Observability: MLflow tracing on every request (latency, tokens, tool calls, errors)
```

### Technology stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| LLM | Gemini 2.5 Flash (`gemini-2.5-flash`) via `google-genai` SDK | Fast + cheap for chat; supports function calling and JSON schema output. Use `gemini-2.5-pro` only if Flash quality falls short in evals. |
| Embeddings | `gemini-embedding-001` (768-dim via `output_dimensionality`) | Same provider, one API key; 768 dims keeps Chroma small with negligible quality loss. |
| Vector store | ChromaDB (persistent, local) | Zero-ops, metadata filtering, runs inside the container. |
| Structured data | SQLite | Complaint tickets + long-term memory; no server needed. |
| API | FastAPI | REST endpoint requirement. |
| UI | Streamlit | Chat UI requirement; talks to FastAPI, not to the LLM directly. |
| Monitoring | MLflow (tracing + metrics) | Spec-suggested; autolog support for Gemini calls. |
| Packaging | Docker (single image, docker-compose optional) | Spec requirement. |

---

## 3. Knowledge Base & Data Preprocessing Pipeline

This is the foundation of the RAG module. The pipeline is a **standalone, re-runnable script** (`scripts/ingest.py`) — separate from the app — so the knowledge base can be rebuilt whenever documents change.

### 3.1 Source corpus

Collect/author ~8–15 HR documents (synthetic but realistic; PDF/DOCX/Markdown):

- Employee handbook (general policies, code of conduct)
- Leave policy (vacation, sick, parental, bereavement)
- Benefits guide (health insurance, allowances, retirement)
- Payroll & compensation FAQ
- Onboarding checklist / first-day guide
- Remote work / attendance policy
- Grievance & complaint procedure (defines the complaint workflow the agent follows)
- Anti-harassment & disciplinary policy (drives escalation rules)

Keep originals in `data/raw/`, never edited by the pipeline.

### 3.2 Pipeline stages

**Stage 1 — Parse & normalize** (`data/raw/` → `data/processed/*.md`)
- PDF → text via `pypdf` (or `docling` if layout is messy); DOCX via `python-docx`; Markdown passes through.
- Strip headers/footers, page numbers, watermarks; fix hyphenated line breaks; normalize whitespace and unicode.
- Convert to clean Markdown, **preserving heading hierarchy** (`#`, `##`, `###`) — headings drive chunking and citations.
- Each processed file gets a YAML frontmatter block: `doc_id`, `title`, `category`, `effective_date`, `version`, `source_file`.

**Stage 2 — Chunking (structure-aware)**
- Split on headings first so a chunk never spans two policy sections.
- Within a section, split to a target of **~400 tokens with ~50-token overlap**; merge tiny sections (<80 tokens) into their parent.
- Prepend a **context header** to every chunk before embedding: `"{doc title} > {section path}"` — this materially improves retrieval for short chunks ("Notice period" alone is ambiguous; "Leave Policy > Resignation > Notice period" is not).
- Tables (e.g., leave accrual rates) are kept whole as single chunks, rendered as Markdown tables.

**Stage 3 — Metadata enrichment**
Each chunk carries:

| Field | Example | Used for |
| --- | --- | --- |
| `doc_id`, `chunk_id` | `leave-policy`, `leave-policy#012` | citations, dedup |
| `title`, `section_path` | "Leave Policy", "Sick Leave > Documentation" | citations shown to user |
| `category` | `leave` \| `benefits` \| `payroll` \| `conduct` \| `complaints` \| `onboarding` | metadata-filtered retrieval after intent routing |
| `effective_date`, `version` | `2026-01-01`, `v3` | answer freshness disclaimers |
| `token_count` | 387 | context budget control |

**Stage 4 — Embed & index**
- Embed with `gemini-embedding-001`, `task_type="RETRIEVAL_DOCUMENT"`, `output_dimensionality=768`, in batches.
- Upsert into a persistent ChromaDB collection keyed by `chunk_id` (re-ingestion is idempotent; changed docs are re-embedded, deleted docs removed).
- Write an ingestion manifest (`data/index_manifest.json`): file hashes, chunk counts, embedding model + date — so we can detect stale indexes and report corpus stats in the write-up.

**Stage 5 — Retrieval (query time)**
- Embed the user query with `task_type="RETRIEVAL_QUERY"`.
- Top-k = 8 by cosine similarity, optionally filtered by `category` when the intent router is confident.
- Similarity floor (tuned in evals, start ~0.5): if no chunk passes, the agent says it doesn't know and offers to route to HR — **never answers from parametric knowledge**.
- Retrieved chunks go into the prompt with their `section_path`; the model must cite sections in its answer, and we verify cited IDs exist in the retrieved set (grounding guardrail).

**Stage 6 — Evaluation set (built alongside the corpus)**
- ~30–50 golden Q&A pairs (question, expected answer points, expected source chunk IDs) authored while writing the corpus.
- Used to measure retrieval hit-rate@k and answer faithfulness across chunking/prompt variants — this is the "Experiment Findings" section of the presentation.

---

## 4. Module Breakdown (maps to spec Section 4)

| # | Module | Implementation in this project |
| --- | --- | --- |
| 1 | **RAG** | Pipeline above; ChromaDB + Gemini embeddings; cited, grounded answers |
| 2 | **Prompt Engineering** | System prompt with role/scope/refusal rules; few-shot examples for intent routing; ablation across ≥3 prompt variants measured on the golden set |
| 3 | **Structured Outputs** | Pydantic schemas via Gemini `response_schema`: `IntentClassification`, `ComplaintTicket` (category, severity, parties, description, desired outcome), `GroundedAnswer` (answer + citations) |
| 4 | **Disambiguation** | Router emits `confidence`; low confidence → ask one clarifying question before retrieving or filing ("Do you want to know the policy, or file a complaint?") |
| 5 | **Memory** | Short-term: rolling session window with summarization past ~20 turns. Long-term: SQLite per-employee profile (name, department, open ticket IDs) recalled at session start |
| 6 | **Guardrails** | Input: topic filter (HR-only), prompt-injection heuristics, PII detection. Output: citation/grounding check, PII redaction in logs, mandatory escalation for harassment/safety/legal-risk complaints (human-in-the-loop — the bot never adjudicates) |
| 7 | **ReAct Agent** | Gemini function-calling loop: think → pick tool (`search_kb`, `file_complaint`, `escalate_to_hr`, `get_ticket_status`) → observe → repeat, max 5 iterations |
| 8 | **Tool Use** | The four tools above; `file_complaint` writes to SQLite and returns a ticket ID; `escalate_to_hr` sends email/webhook (mocked in dev) |
| 9 | **Chat UI** | Streamlit chat with session state, source-citation expanders, ticket confirmation cards |
| 10 | **API Endpoint** | FastAPI: `POST /chat` (session_id, message → reply, citations, actions), `GET /tickets/{id}`, `GET /health` |
| 11 | **LLMOps Monitoring** | MLflow tracing on every request: latency, input/output tokens, tool-call sequence, guardrail triggers, errors; eval runs logged as MLflow experiments |
| 12 | **Dockerization** | Single Dockerfile (FastAPI + Streamlit via supervisor or two-stage compose); `docker compose up` runs everything incl. ingestion on first boot |

**Ownership (team of 3–4, ≥2 modules each):** fill in names.

| Member | Modules |
| --- | --- |
| Member 1 | RAG + data pipeline, Structured Outputs |
| Member 2 | ReAct Agent, Tool Use, Disambiguation |
| Member 3 | Guardrails, Memory, Prompt Engineering |
| Member 4 (if any) | Chat UI, API Endpoint, MLflow, Docker |

---

## 5. Repository Layout

```
stai-capstone/
├── CLAUDE.md                  # working conventions for this repo
├── PLAN.md                    # this file
├── README.md                  # setup, architecture diagram, module ownership
├── specs.md                   # course spec (do not edit)
├── .env.example               # GEMINI_API_KEY=...
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/
│   ├── raw/                   # source HR documents (PDF/DOCX/MD)
│   ├── processed/             # cleaned markdown w/ frontmatter
│   └── chroma/                # persistent vector store (gitignored)
├── scripts/
│   └── ingest.py              # full preprocessing pipeline (Section 3)
├── src/
│   ├── config.py              # settings, model names, paths
│   ├── agent/
│   │   ├── orchestrator.py    # ReAct loop
│   │   ├── router.py          # intent classification + disambiguation
│   │   ├── tools.py           # search_kb, file_complaint, escalate, ticket_status
│   │   └── prompts.py         # versioned prompt variants
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   └── retriever.py
│   ├── guardrails/
│   │   ├── input_checks.py
│   │   └── output_checks.py
│   ├── memory/
│   │   ├── session.py
│   │   └── persistent.py
│   ├── schemas.py             # Pydantic models (tickets, intents, answers)
│   ├── api.py                 # FastAPI app
│   └── ui.py                  # Streamlit app
├── evals/
│   ├── golden_set.jsonl       # Q&A pairs w/ expected chunk IDs
│   ├── run_retrieval_eval.py  # hit-rate@k, MRR
│   └── run_answer_eval.py     # faithfulness / LLM-as-judge via Gemini
└── tests/                     # unit tests (chunking, guardrails, schemas)
```

---

## 6. Build Order (7 milestones)

1. **Corpus + ingestion** — author HR docs, build `ingest.py` end-to-end (parse → chunk → embed → Chroma), write golden eval set. *Everything else depends on this.*
2. **RAG core** — retriever + grounded-answer prompt + citations; retrieval eval running (hit-rate@k baseline).
3. **Agent loop** — intent router, ReAct orchestrator, four tools, complaint schema + SQLite tickets.
4. **Guardrails + memory** — input/output checks, escalation path, session + persistent memory.
5. **Interfaces** — FastAPI endpoint, then Streamlit UI on top of it.
6. **Ops** — MLflow tracing wired through the orchestrator; Dockerfile + compose; README.
7. **Experiments + deliverables** — prompt ablations, chunk-size ablation (e.g., 250 vs 400 vs 600 tokens), guardrail red-team tests; write-up (≥2,000 words), slides, demo dry-run + fallback recording.

---

## 7. Experiments to Report (grading: 20% eval weight)

| Experiment | Metric | Variants |
| --- | --- | --- |
| Chunking ablation | retrieval hit-rate@5, MRR on golden set | 250 / 400 / 600 tokens; with vs. without context headers |
| Prompt ablation | answer faithfulness (LLM-as-judge), citation accuracy | 3 system-prompt variants |
| Similarity threshold | false-answer rate vs. "I don't know" rate | sweep 0.3–0.7 |
| Guardrail red-team | block rate on off-topic / injection / PII probes | ~20 adversarial prompts |
| Escalation correctness | % of harassment/safety scenarios correctly escalated | scripted complaint scenarios |
| Latency & cost | p50/p95 latency, tokens per request (from MLflow) | Flash vs. Pro on a sample |

Document failure modes found (e.g., retrieval misses on table data, router confusion between "complain about policy" vs. "ask about complaint policy") and mitigations — this feeds the Retrospective section.

---

## 8. Key Risks & Mitigations

- **Demo failure during presentation** → run fully local (Chroma + SQLite), record a fallback video, disclose upfront per spec.
- **Gemini rate limits/outage during demo** → cache golden-path responses; keep the fallback video.
- **Hallucinated policy answers** → similarity floor + citation verification + "I don't know" path; measured in evals.
- **Sensitive complaints mishandled** → hard-coded escalation rules (not LLM-discretionary) for harassment/safety/legal categories.
- **Scope creep** → milestones 1–5 are the MVP; graph memory, reranking, multi-language are explicitly out of scope.
