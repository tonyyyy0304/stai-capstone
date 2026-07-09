# HR FAQ & Complaint Chatbot — Implementation Plan

**Course:** Introduction to Agentic AI (STAI100) — Midterm Capstone (Week 9)
**Use case type:** Vector DB RAG (HR/Legal domain)
**LLM provider:** fully switchable, chat *and* embeddings independently — Gemini (default for both) or a self-hosted Ollama server, via `LLM_PROVIDER` and `EMBEDDING_PROVIDER`. See §2.1.

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

### Technology stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| LLM | `LLM_PROVIDER` switchable: Gemini 2.5 Flash (`gemini-2.5-flash`, default) via `google-genai` SDK, or a self-hosted Ollama model (default `gemma4:e4b`) via `src/agent/llm_client.py`'s adapter | Fast + cheap for chat; supports function calling and JSON schema output. Ollama option exists for quota-free testing — see §2.1. |
| Embeddings | `EMBEDDING_PROVIDER` switchable: `gemini-embedding-001` (768-dim via `output_dimensionality`, default) or Ollama's `nomic-embed-text` (native 768-dim) | Independent of the chat provider — you can mix, e.g. Gemini chat + Ollama embeddings. Switching requires re-running `scripts/ingest.py`; the two embedding spaces aren't compatible. |
| Vector store | ChromaDB (persistent, local) | Zero-ops, metadata filtering, runs inside the container. |
| Structured data | SQLite | Complaint tickets + long-term memory; no server needed. |
| API | FastAPI | REST endpoint requirement. |
| UI | Streamlit | Chat UI requirement; talks to FastAPI, not to the LLM directly. |
| Monitoring | MLflow (tracing + metrics) | Spec-suggested; autolog support for Gemini calls. |
| Packaging | Docker (single image, docker-compose optional) | Spec requirement. |

### 2.1 LLM Provider Abstraction — shipped

**Why:** Gemini's free tier caps `gemini-2.5-flash` at **20 `generate_content` requests/day per project**, not per-minute. A single agent turn already costs several requests (1 router classification + up to `MAX_REACT_ITERATIONS`=5 ReAct loop calls + 2 more if `search_web` fires), so the quota was exhausted mid-testing repeatedly, well before the free allowance felt low. §8 already flags this risk ("Gemini rate limits/outage during demo") — it stopped being hypothetical.

**Mitigations shipped:**
- `run_turn()` catches `google.genai.errors.APIError` (and now `LLMBackendError`, see below) and degrades to a plain-language "try again" reply instead of a raw 500.
- `src/agent/usage.py` logs every agent LLM call's token usage to SQLite, tagged by which model actually served it; `GET /usage` reports today's + all-time request counts and token totals per model, so the team can see quota consumption before it's a surprise.
- `search_web` no longer depends on Gemini's `google_search` grounding tool — it uses **Tavily** for the actual search (domain-restricted via `include_domains`) and one `response_schema` call just to shape the results into a `GroundedAnswer`.
- **`src/agent/llm_client.py`** — an `OllamaClient` adapter that lets the whole agent (router classification, the ReAct tool-calling loop, and `search_web`'s shaping call) run against a self-hosted Ollama server instead of Gemini, selected via `LLM_PROVIDER=ollama` in `.env` (default stays `gemini`, fully backward compatible). The adapter mimics `google-genai`'s exact `client.models.generate_content(model, contents, config)` call signature and Gemini-shaped response attributes (`.text`, `.parsed`, `.candidates[0].content.parts[*].function_call`, `.usage_metadata`), so **router.py, orchestrator.py, and tools.py needed almost no changes** — the adapter conforms to what they already read. Verified live end-to-end against a real Ollama VM (`gemma4:e4b`): router classification, `search_kb` tool-calling, multi-field `file_complaint` tool-calling with correct deterministic escalation, and `search_web`'s Tavily+shaping path all work correctly.

**Embeddings are switchable too, independently of chat** — this expanded beyond the original scope of this section. `EMBEDDING_PROVIDER` (`gemini` default or `ollama`) selects between `GeminiEmbedder` and a new `OllamaEmbedder` (`src/rag/embeddings.py`, using Ollama's `nomic-embed-text`, native 768-dim, no truncation/re-normalization needed unlike Gemini's). `config.get_embedder()` dispatches between them; `scripts/ingest.py` and `src/rag/retriever.py` both call it rather than hardcoding a class. **`src/rag/answerer.py`'s grounded-answer generation also now routes through `config.get_llm_client()`** (previously hardcoded to Gemini) — so with `LLM_PROVIDER=ollama`, RAG answer synthesis runs on Ollama too, not just the agent's own reasoning. Net effect: with both `LLM_PROVIDER=ollama` and `EMBEDDING_PROVIDER=ollama` set, **the app makes zero Gemini calls** during normal operation.

Config additions backing this: `GEMINI_CHAT_MODEL`/`OLLAMA_CHAT_MODEL`/`ACTIVE_CHAT_MODEL` and `GEMINI_EMBEDDING_MODEL`/`OLLAMA_EMBEDDING_MODEL`/`ACTIVE_EMBEDDING_MODEL` (the old bare `CHAT_MODEL` constant no longer exists — it split into a real/resolved pair per concern, chat and embeddings separately).

**Switching `EMBEDDING_PROVIDER` requires re-running `scripts/ingest.py` against a cleared index.** Gemini and Ollama embeddings are different vector spaces (both happen to be 768-dim, which masks the incompatibility instead of erroring on it) — mixing them doesn't crash, it silently returns wrong/garbage similarity rankings. Clear `data/chroma/` and `data/index_manifest.json` and re-ingest whenever `EMBEDDING_PROVIDER` changes.

| Feature | Where used | Status on Ollama |
| --- | --- | --- |
| `response_schema` typed output | `router.py`, `orchestrator.py` (tool args), `tools.py`, `answerer.py` | Resolved — adapter translates to Ollama's `format=<json schema>` |
| Tool/function calling | `orchestrator.py` ReAct loop | Resolved — adapter translates `types.FunctionDeclaration` to Ollama's OpenAI-style `tools=`, and translates responses back into Gemini-shaped `function_call` parts |
| `google_search` grounding tool | `tools.search_web` | N/A — already resolved via Tavily, provider-independent |
| `gemini-embedding-001` | `src/rag/embeddings.py` | Resolved — `EMBEDDING_PROVIDER=ollama` uses `nomic-embed-text` via `OllamaEmbedder` |

**Known limitations:** small local models (`gemma4:e2b`/`e4b`) are less reliable at strict JSON-schema conformance and tool-calling than Gemini, especially on nested schemas like `GroundedAnswer`. This is a **quality** risk, not a correctness one — the existing fail-closed paths (`.parsed = None` → router asks a clarifying question, `search_web`/RAG return "I don't know") already bound the blast radius; expect more clarifying questions or `MAX_REACT_ITERATIONS` fallbacks on Ollama than on Gemini, never a crash. Retrieval quality with `nomic-embed-text` has been spot-checked (0.6–0.77 cosine similarity, correct top results on test queries) but not run through the golden-set hit-rate@k eval the way Gemini embeddings were.

**Operational note:** the Ollama VM used for verification (`OLLAMA_URL` in the user's local `.env`, not committed) was found reachable with **no authentication** — anyone who finds the port can use it, including its `:cloud`-suffixed models which proxy to billed cloud credits. Worth locking down (firewall to known IPs, reverse proxy with auth, or VPN-only binding) before this becomes a default path for a live demo.

`TokenUsage` (in `src/schemas.py`) is named provider-agnostically (`prompt_tokens`/`completion_tokens`/`total_tokens`, not Gemini's `*_token_count` field names), which is why `usage.py` needed zero changes to support the new backend.

**Bugs found and fixed while wiring this up** (worth knowing if similar patterns get copied elsewhere):
- `src/monitoring.py` referenced the old `config.CHAT_MODEL` name after it was split into `GEMINI_CHAT_MODEL`/`ACTIVE_CHAT_MODEL`. Since `chat_trace()` runs at the very start of every `/chat` request, this crashed **every single request** with `AttributeError` — surfaced as "monitoring not working," but the actual bug was upstream of monitoring, not in it.
- A duplicated-prefix typo, `config.ACTIVE_ACTIVE_CHAT_MODEL` (doesn't exist), was introduced in `router.py`, `orchestrator.py`, and `tools.py` — crashed usage tracking immediately after every successful LLM call.
- `src/rag/embeddings.py` had an unconditional top-level `from ollama import Client` — and unused, since `OllamaEmbedder` does its own lazy import already. This broke importing the embeddings module (and therefore most of the app) whenever the `ollama` pip package wasn't installed, *even on the pure-Gemini path*. Removed; the project convention is to lazy-import provider SDKs inside functions specifically to avoid this.
- Local (host) ingestion and containerized (Docker) ingestion sharing the same bind-mounted `data/chroma/` hit a recurring cross-platform chromadb incompatibility (Rust panic, `range start index out of range`) — happened three separate times this session. Not a code bug, an operational one: **don't run `python scripts/ingest.py` on the host and `docker compose up` against the same `data/chroma/` interchangeably** — pick one environment per index, or clear and re-ingest when switching.
- `answer_question()` (`src/rag/answerer.py`) was returning retrieved chunks to the caller even when the grounded-answer step decided it couldn't actually answer from them (`insufficient_context=True`) — the UI would show "Retrieved policy excerpts" right next to an "I don't know" reply, which read as contradictory. Fixed: chunks are withheld when the answer is insufficient.

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
- Embed via `config.get_embedder()` (§2.1): Gemini's `gemini-embedding-001` (`task_type="RETRIEVAL_DOCUMENT"`, `output_dimensionality=768`, default) or Ollama's `nomic-embed-text` (native 768-dim) depending on `EMBEDDING_PROVIDER`, in batches.
- Upsert into a persistent ChromaDB collection keyed by `chunk_id` (re-ingestion is idempotent; changed docs are re-embedded, deleted docs removed). Switching `EMBEDDING_PROVIDER` requires clearing the index and re-ingesting from scratch — the two embedding spaces aren't compatible even though both happen to be 768-dim.
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
| 1 | **RAG** | Pipeline above; ChromaDB + switchable embeddings (Gemini or Ollama, §2.1); cited, grounded answers |
| 2 | **Prompt Engineering** | System prompt with role/scope/refusal rules; few-shot examples for intent routing; ablation across ≥3 prompt variants measured on the golden set |
| 3 | **Structured Outputs** | Pydantic schemas via Gemini `response_schema`: `IntentClassification`, `ComplaintTicket` (category, severity, parties, description, desired outcome), `GroundedAnswer` (answer + citations) |
| 4 | **Disambiguation** | Router emits `confidence` + `intent` (`faq`\|`complaint`\|`ambiguous`\|`out_of_scope`); low confidence or ambiguous → ask one clarifying question before any tool runs; out-of-scope → declines without calling a tool. Implemented in `src/agent/router.py`. |
| 5 | **Memory** | Built, `src/memory/`. Short-term (`session.py`): SQLite-backed `session_turns`, replaced the in-process `_SESSION_HISTORY` dict api.py used to carry — survives a restart now. `get_history()` trims to the last `MEMORY_TRIM_TURNS` (20) turns for the in-context window; full history always persists. Long-term (`persistent.py`): a rolling per-session summary, extended via **one incremental LLM call** only once `MEMORY_SUMMARY_BATCH_SIZE` (5) turns have overflowed the trim window since the last summary — deliberately batched, not per-turn, given the Gemini quota constraint (§2.1). Cross-session recall: a brand-new session seeds from the same `employee_id`'s latest summary from any prior session. Wired into `orchestrator.handle_message()`, not `run_turn()` — keeps `run_turn()` a pure function over an explicit history list for existing tests. |
<<<<<<< HEAD
| 6 | **Guardrails** | Built, `src/guardrails/`. Two layers: (1) deterministic, no-LLM-call checks — `input_checks.py` (prompt-injection regex), `toxicity.py` (small documented wordlist, ~10 words), `pii.py` (regex email/phone/employee-ID detection + redaction — used for redaction, never as an input-blocking gate, since the complaint-intake flow legitimately collects PII and blocking ordinary contact-info mentions in FAQ chat would be bad UX), `grounding.py` (output-side, re-verifies every citation maps to a retrieved chunk_id, hard-verify defense-in-depth on top of `answerer.verify_citations()`). (2) a semantic backstop piggybacked on the intent router's already-mandatory LLM call — `IntentClassification` carries `is_toxic`/`is_injection_attempt`, filled in by the same `classify_intent()` call, catching paraphrased/creative-spelling cases the deterministic layer misses at zero incremental LLM cost. Toxicity is also complaint-aware: `check_toxicity_with_context()` skips the wordlist entirely for `Intent.COMPLAINT` and trusts only the semantic signal, because a harassment complaint quoting abuse said *to* the employee ("my coworker called me a bitch") legitimately contains wordlist hits without the employee being abusive — auto-blocking on the wordlist alone would have silently prevented exactly the complaints this system exists to let through (found and fixed while adding the semantic backstop). Wired into `orchestrator.run_turn()` in three stages: `_check_input()` (injection regex only) gates entry before the router runs; `_check_semantic_guardrails()` (injection + complaint-aware toxicity) runs immediately after `classify_intent()` returns, before disambiguation/out-of-scope handling; `check_grounding()` runs right before the final `AgentResponse` is returned. Topic restriction's actual enforcement is the intent router's `out_of_scope` classification, already a hard block in `orchestrator.py`; duplicating that with keyword topic-matching would be redundant and worse at nuance. Escalation itself is deterministic and already shipped: `tools.should_escalate(category)` fires for harassment/discrimination/safety/legal, is never LLM-callable, and its result is fed back to the model as a tool observation so the reply can state HR was notified. **Note:** this is a simpler, conversational version of what §6.1 below describes (no consent gate, no form UI, no danger scan yet) — see the flag at the top of §6.1. |
=======
| 6 | **Guardrails** | Built, `src/guardrails/`. Four deterministic, no-LLM-call checks plus one LLM-as-judge layer: `input_checks.py` (prompt-injection heuristics — topic restriction's actual enforcement is the intent router's `out_of_scope` classification, already a hard block in `orchestrator.py`; duplicating that with keyword topic-matching would be redundant and worse at nuance), `toxicity.py` (small documented wordlist, ~10 words), `pii.py` (regex email/phone/employee-ID detection + redaction — used for redaction, never as an input-blocking gate, since the complaint-intake flow legitimately collects PII and blocking ordinary contact-info mentions in FAQ chat would be bad UX), `grounding.py` (output-side, re-verifies every citation maps to a retrieved chunk_id, hard-verify defense-in-depth on top of `answerer.verify_citations()`). The **LLM-as-judge** (`llm_judge.py`, `response_schema=LLMJudgeVerdict`) is the fifth check and the only non-deterministic one: a single structured call classifies each message the deterministic layer let through across five dimensions at once — toxicity, PII, prompt-injection, off-topic (topic filter), and jailbreak — catching the paraphrased/novel adversarial phrasing a fixed wordlist can't. Detection stays separate from policy (per CLAUDE.md): the model only flags; the allow/block decision is deterministic code in `to_guardrail_result()`, gated by `LLM_JUDGE_BLOCKING_VIOLATIONS` + `LLM_JUDGE_CONFIDENCE_FLOOR`. PII is detected but never blocks (the complaint flow legitimately collects it), and the judge **fails open** — a judge API/backend error lets the turn proceed on the deterministic layer as backstop rather than 500-ing. It runs on Gemini or Ollama unchanged (uses `response_schema` exactly like the router). Wired into `orchestrator.run_turn()` at both ends: `_check_input()` runs the deterministic injection + toxicity checks first (free, short-circuits blatant cases) and then the LLM judge, gating entry before the router runs; `check_grounding()` runs right before the final `AgentResponse` is returned. Escalation itself is deterministic and already shipped: `tools.should_escalate(category)` fires for harassment/discrimination/safety/legal, is never LLM-callable, and its result is fed back to the model as a tool observation so the reply can state HR was notified. **Note:** this is a simpler, conversational version of what §6.1 below describes (no consent gate, no form UI, no danger scan yet) — see the flag at the top of §6.1. |
>>>>>>> b7c2dd7cf556602469856761c4fb585be98c7952
| 7 | **ReAct Agent** | Real function-calling loop in `src/agent/orchestrator.py`: model picks a tool → `tools.py` executes it → result fed back as an observation → repeat, capped at `MAX_REACT_ITERATIONS`=5. Falls back to a plain-language message on hitting the cap or on an LLM-backend error (`APIError`/`LLMBackendError` — rate limit, outage, or Ollama connectivity) instead of crashing. Runs on either Gemini or Ollama via `LLM_PROVIDER` (§2.1) — same loop code either way. |
| 8 | **Tool Use** | Model-callable tools: `search_kb` (internal RAG), `search_web` (DOLE/labor-law fallback: Tavily search domain-restricted to `dole.gov.ph`/`officialgazette.gov.ph`/`lawphil.net` + one `response_schema` call to shape results), `file_complaint`, `get_ticket_status`. `escalate_to_hr` is deliberately **not** model-callable — see Module 6. |
| 9 | **Chat UI** | Streamlit chat (`src/ui.py`) with session state, citation/source expanders, complaint action cards. Built; talks to the API, not to Gemini directly. |
| 10 | **API Endpoint** | FastAPI (`src/api.py`): `POST /chat` (session_id, message → reply, citations, sources, actions, token_usage), `GET /tickets/{id}`, `GET /usage` (today's + all-time agent token/request usage), `GET /health`. Built. |
| 11 | **LLMOps Monitoring** | MLflow tracing per request (`src/monitoring.py` + `src/api.py`): latency, citation/source/action counts, and now `prompt_tokens`/`completion_tokens`/`total_tokens` (this session — was previously untracked entirely). **Still missing:** tool-call sequence and guardrail triggers aren't in MLflow yet (the ReAct loop's per-step trace, `orchestrator.AgentStep`, exists but is discarded before it reaches the API response or MLflow). |
| 12 | **Dockerization** | Single Dockerfile (FastAPI + Streamlit via supervisor or two-stage compose); `docker compose up` runs everything incl. ingestion on first boot |

**Ownership (team of 3–4, ≥2 modules each):** fill in names.

| Member | Modules |
| --- | --- |
| Member 1 | RAG + data pipeline, Structured Outputs |
| Member 2 | (FAQ) ReAct Agent, Tool Use, Disambiguation, Guardrails, Memory, Prompt Engineering |
| Member 3 | (Escalation) ReAct Agent, Tool Use, Disambiguation, Guardrails, Memory, Prompt Engineering |
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
│   ├── monitoring.py          # MLflow tracing helpers (built)
│   ├── agent/
│   │   ├── orchestrator.py    # ReAct loop + handle_message() API adapter (built)
│   │   ├── router.py          # intent classification + disambiguation (built)
│   │   ├── tools.py           # search_kb, search_web, file_complaint, get_ticket_status, escalate_to_hr (built)
│   │   ├── usage.py           # token usage tracking, SQLite-backed (built)
│   │   ├── llm_client.py      # Ollama backend adapter (built, §2.1)
│   │   └── prompts.py         # versioned prompt variants
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── embeddings.py     # GeminiEmbedder + OllamaEmbedder, dispatched via config.get_embedder() (built, §2.1)
│   │   └── retriever.py
│   ├── guardrails/             # built (Module 6)
│   │   ├── input_checks.py    # prompt-injection heuristics (deterministic)
│   │   ├── toxicity.py        # small documented wordlist (deterministic)
│   │   ├── pii.py             # email/phone/employee-ID detect + redact
│   │   ├── llm_judge.py       # LLM-as-judge: toxicity/PII/injection/topic/jailbreak
│   │   └── grounding.py       # output-side citation re-verification
│   ├── memory/                 # built (Module 5)
│   │   ├── session.py         # SQLite short-term history, trim window
│   │   └── persistent.py      # rolling summary, batched incremental LLM call, cross-session recall
│   ├── schemas.py             # Pydantic models (tickets, intents, answers, token usage)
│   ├── api.py                 # FastAPI app (built)
│   └── ui.py                  # Streamlit app (built)
├── evals/
│   ├── golden_set.jsonl       # Q&A pairs w/ expected chunk IDs
│   ├── run_retrieval_eval.py  # hit-rate@k, MRR
│   └── run_answer_eval.py     # faithfulness / LLM-as-judge via Gemini
└── tests/                     # unit tests (chunking, guardrails, schemas, agent, api)
```

---

## 6. Build Order (7 milestones)

1. **Corpus + ingestion** — author HR docs, build `ingest.py` end-to-end (parse → chunk → embed → Chroma), write golden eval set. *Everything else depends on this.*
2. **RAG core** — retriever + grounded-answer prompt + citations; retrieval eval running (hit-rate@k baseline).
3. **Agent loop** — intent router, ReAct orchestrator, four tools, complaint schema + SQLite tickets. **Done.**
4. **Guardrails + memory** — input/output checks, escalation path, session + persistent memory. **Done for what's in scope:** deterministic category-based escalation, all four guardrail checks, and both memory tiers are built and tested (see Module 5/6 status above). The fuller form-driven escalation design in §6.1 remains unbuilt beyond Rule 1 — that's a separate, larger scope decision, not part of this milestone.
5. **Interfaces** — FastAPI endpoint, then Streamlit UI on top of it. **Done**, including wiring the agent orchestrator into `/chat` (was previously falling back to plain RAG only).
6. **Ops** — MLflow tracing wired through the orchestrator; Dockerfile + compose; README. **Done for the core loop, partial on completeness:** `docker compose up --build` verified working end-to-end (all four services: ingest, api, ui, mlflow) with both Gemini and Ollama backends; latency + token usage traced cleanly in MLflow. Still missing: tool-call sequence/guardrail triggers in MLflow, and the README hasn't caught up to the `LLM_PROVIDER`/`EMBEDDING_PROVIDER` env vars yet.
7. **Experiments + deliverables** — prompt ablations, chunk-size ablation (e.g., 250 vs 400 vs 600 tokens), guardrail red-team tests; write-up (≥2,000 words), slides, demo dry-run + fallback recording.

---

## 6.1 Escalation Rules (Guardrails detail)

> **Status flag:** this section describes the target design. What's actually implemented today (`src/agent/tools.py`, `orchestrator.py`) is simpler: conversational complaint intake through the ReAct loop (no consent gate, no form card), and `should_escalate()` only implements Rule 1 (mandatory category escalation for harassment/discrimination/safety/legal) — Rules 2–7 (severity floors, danger scan, consent gate, form PII handling, fail-safe escalation, `EscalationEvent`/24h SLA) are not built. Whoever owns Guardrails/Escalation should confirm whether to build toward this doc as written or reconcile the plan with the simpler shipped version before more work goes into either direction.

Escalation is a **consent-gated, form-driven flow** whose routing decisions are made by deterministic
functions in `src/guardrails/` — never by LLM judgment (per CLAUDE.md, §8 risk). Grounded in
`anti-harassment-policy.md` (§Reporting: 24h to a human HR officer; automated systems never triage,
adjudicate, or close) and `grievance-procedure.md` (§Scope). **Fail toward escalation:** any
ambiguity, parse error, or missing field on a potentially-sensitive category escalates.

### Escalation flow (user-facing)

- **Step A — Consent gate.** When the router detects complaint / escalation intent, the agent does
  not silently start filing. It asks the user which they want: **(a) write a complaint** (tracked
  ticket) or **(b) escalate the query to HR**. Nothing is sent until the user chooses. This reuses the
  Disambiguation module's clarifying-question mechanism (§4, Module 4).
- **Step B — Structured intake form.** On the user's choice, the agent presents a **Streamlit form
  card** (not free chat) with the `ComplaintTicket` fields (category, description, parties involved,
  incident date, desired outcome) plus the identity fields needed to write the HR email (employee
  name, department, contact). Form fields are submitted straight to the API, giving a clean PII
  boundary. If the danger scan (Rule 3) flags the case, the form additionally renders the
  building-security / local-911 emergency-contact banner at the top — the safety guidance is never
  dropped even though the flow always goes through the form.
- **Step C — Email generation & auto-send.** On submit, the validated form populates an HR email
  template; `escalate_to_hr` renders and sends it to the HR case queue (email/webhook, mocked in dev),
  while `file_complaint` writes the SQLite ticket. The user gets the ticket number and the 24h-review
  message.

### Form PII guardrail (distinct from the global one)

The intake form **intentionally collects PII** (names, contact, parties) because the HR email needs
it — so this guardrail's job is the inverse of the global "redact PII in logs" rule: it must
**preserve** PII on the path to the email/secure record while **excluding it from every observable
surface**. Concretely: the form payload bypasses normal MLflow request tracing; only a redacted
summary + the structured non-PII fields (category, severity, trigger_rule, ticket_id) are logged;
raw PII lives only in the email and the SQLite record, never in LLM context that gets traced.

### Deterministic routing rules

- **Rule 1 — Mandatory category escalation.** Route to escalation (option b behavior) regardless of
  severity/confidence/completeness when `ComplaintTicket.category` ∈
  `{harassment, discrimination, safety, legal}`.
- **Rule 2 — Severity escalation.** For all other categories (`payroll`, `benefits`,
  `workplace_conflict`, `policy_violation`, `other`): escalate only if `severity ∈ {high, critical}`;
  otherwise file a normal tracked ticket into the standard grievance queue.
- **Rule 3 — Danger scan.** A reviewed keyword/heuristic scan over the form text
  (weapon/assault/threat/self-harm/"right now" cues) flags the case `critical` and triggers the
  emergency-contact banner in Step B. Per team decision the flow still goes through the consent gate
  and form; the scan changes severity and surfaced guidance, it does not skip the form.
- **Rule 4 — Deterministic severity floors.** Rule 1 categories floor at `high` (Rule 3 hit →
  `critical`); any retaliation cue floors at `high`. The LLM does not freely set severity for
  sensitive categories.
- **Rule 5 — Escalation payload.** A firing case builds a structured `EscalationEvent`
  (`ticket_id`, `category`, `severity`, `trigger_rule`, `sla_deadline` = created_at + **24h**,
  `redacted_summary`) that both feeds the email template and is the only thing logged (Step C + PII
  guardrail).
- **Rule 6 — User-facing behavior.** After submit, tell the employee a human HR officer will review
  within 24h, give the ticket number for status inquiries, and (danger cases) reiterate the emergency
  contact. The bot never claims to resolve, judge, or close a sensitive report.
- **Rule 7 — Fail-safe.** Escalate with `trigger_rule = parse_failure` when the form fails schema
  validation, category is `other` but the danger scan is inconclusive, or complaint-intent confidence
  is below threshold. Never silently drop a sensitive-looking report.

New code: `src/guardrails/escalation.py` (`should_escalate(ticket, raw_text) -> EscalationDecision`,
Rules 1–4, 7), `src/guardrails/danger_scan.py` (Rule 3 + retaliation lexicons), and
`src/guardrails/form_pii.py` (Form PII guardrail); `EscalationForm` / `EscalationDecision` /
`EscalationEvent` schemas in `src/schemas.py`; an HR email template + `escalate_to_hr` sender (mocked
in dev); the form-card UI in `src/ui.py` and its submit endpoint in `src/api.py`; the 24h SLA constant
in `src/config.py`. Directly testable against the "% of harassment/safety scenarios correctly
escalated" eval (§7).

---

## 7. Experiments to Report (grading: 20% eval weight)

| Experiment | Metric | Variants |
| --- | --- | --- |
| Chunking ablation | retrieval hit-rate@5, MRR on golden set | 250 / 400 / 600 tokens; with vs. without context headers |
| Prompt ablation | answer faithfulness (LLM-as-judge), citation accuracy | 3 system-prompt variants |
| Similarity threshold | false-answer rate vs. "I don't know" rate | sweep 0.3–0.7 |
<<<<<<< HEAD
| Guardrail red-team | block rate (injection/toxicity), detection rate (PII), complaint-exempt carve-out correctness | `evals/run_guardrail_eval.py` + `evals/guardrail_redteam.jsonl`, 23 adversarial prompts — currently 100%/100%/100% on the deterministic checks + complaint-exempt combining logic (the latter runs against a synthetic `IntentClassification`, still zero LLM calls). Off-topic block rate needs `--with-router` (costs LLM calls, off by default) since that's the intent router's job, not a deterministic guardrail. |
=======
| Guardrail red-team | block rate (injection/toxicity), detection rate (PII) | `evals/run_guardrail_eval.py` + `evals/guardrail_redteam.jsonl`, 20 adversarial prompts — currently 100%/100% on the deterministic checks. Off-topic block rate needs `--with-router` (costs LLM calls, off by default) since that's the intent router's job, not a deterministic guardrail. The LLM-as-judge layer (`llm_judge.py`) adds nuance-aware coverage on top (toxicity/PII/injection/topic/jailbreak) but isn't in this automated eval yet — scoring it would cost one LLM call per probe. |
>>>>>>> b7c2dd7cf556602469856761c4fb585be98c7952
| Escalation correctness | % of harassment/safety scenarios correctly escalated | scripted complaint scenarios |
| Latency & cost | p50/p95 latency, tokens per request (from MLflow) | Flash vs. Pro on a sample |

Document failure modes found (e.g., retrieval misses on table data, router confusion between "complain about policy" vs. "ask about complaint policy") and mitigations — this feeds the Retrospective section.

---

## 8. Key Risks & Mitigations

- **Demo failure during presentation** → run fully local (Chroma + SQLite), record a fallback video, disclose upfront per spec.
- **Gemini rate limits/outage during demo** → **materialized during development**, not just a theoretical risk: the free tier's 20 `generate_content`/day cap on `gemini-2.5-flash` was exhausted mid-testing repeatedly (each agent turn costs several requests). Mitigations: `run_turn()` catches `APIError`/`LLMBackendError` and degrades gracefully instead of crashing; `GET /usage` gives visibility into consumption per model; `LLM_PROVIDER=ollama` + `EMBEDDING_PROVIDER=ollama` together give real headroom by moving chat/reasoning *and* embeddings off Gemini entirely (§2.1, verified working end-to-end — zero Gemini calls in that configuration). Still keep: cache golden-path responses, keep the fallback video, disclose upfront. New risk this introduces: the verified Ollama VM is currently reachable with no authentication — don't rely on it as the live-demo path without locking it down first (§2.1 operational note).
- **Ollama backend quality/availability for a live demo** → small local models are less reliable at strict tool-calling/schema conformance than Gemini (§2.1), and a remote self-hosted VM has none of Gemini's managed-service reliability guarantees. Treat `LLM_PROVIDER=ollama`/`EMBEDDING_PROVIDER=ollama` as a development/testing escape valve, not the default for the actual presentation, unless it's been dry-run tested end-to-end beforehand.
- **Mixed local/Docker ingestion corrupts the index** → hit three times this session: running `scripts/ingest.py` on the host (native Windows chromadb) and then `docker compose up` (Linux chromadb) against the same bind-mounted `data/chroma/` causes a Rust panic on open. Not data-destructive (the fix is just re-ingesting), but wastes real time if you don't recognize it immediately. Pick one environment per index; clear and re-ingest if you switch.
- **Hallucinated policy answers** → similarity floor + citation verification + "I don't know" path; measured in evals.
- **Sensitive complaints mishandled** → hard-coded escalation rules (not LLM-discretionary) for harassment/safety/legal categories. Currently the simple category-based version (§6.1 status flag) — confirm with whoever owns Guardrails whether the fuller form-driven design in §6.1 is still the target before the demo.
- **Scope creep** → milestones 1–5 are the MVP; graph memory, reranking, multi-language are explicitly out of scope.
