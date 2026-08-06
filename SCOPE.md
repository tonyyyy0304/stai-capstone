# Scope — What We're Building and Why

Plain-language companion to [PLAN.md](PLAN.md). PLAN.md is the implementation spec (file paths, schemas, build order). This doc answers the more basic question first: **what is the product, why does it look like this, and what are we explicitly not doing.**

---

## 1. What the instructor actually said

Direct quotes/paraphrase from the feedback session, kept verbatim where possible:

> "HR compliance chatbot: 1) query questions about code of conduct / employee manual — 10 HR people vs 400 employees, hours × salary (justification). 2) escalations — can de-scope, don't need to do."

> "Due process — completeness check. OCR — guide you if you have complete [documents] — could be useful. Counterpoint: 'is this easy enough?' Maybe as a stretch goal. CV — verification."

> "Can we reach 100% evaluation accuracy on a handbook with ≥100 pages (i.e. DLSU faculty manual, 273 pages / DLSU student handbook, ~300+ pages)? Focus: onboarding-related FAQs, stretch: general employee Qs."

(A second, differently-worded restatement of this same question — *"How well can an agent answer FAQs from a reference company/employee manual — can we push it to 100%?"* — was crossed out in the instructor's notes. Treat it as superseded by the line above, not as a separate ask.)

> "How well can an agent verify that your provided documents are correct and have the complete fields?"

> "1. Research PH onboarding reqs (NBI clearance, brgy certificate, tax records, Pag-IBIG number, SSS number, health records/APE). 2. Retrieve or mock up an employee manual — DLSU employee? 3. Build a dataset of queries vs correct answers, based on the employee manual (actual reference or mocked up). OCR: 1. Choose document to focus on. 2. Build dataset for evals — OCR metrics."

**The one line that matters most:** *"is this easy enough?"* — that's the instructor pre-empting the failure mode where a team bolts a single Gemini-vision call onto a chatbot and calls it "CV integration." Everything in §3 below about the quality gate and validation rules exists because of that sentence.

---

## 2. Two questions, one product

The instructor gave us two separate research questions. They sound like two projects. **They're one project** because both live inside the same onboarding conversation:

**Question A — FAQ answering.** *Can the agent answer questions from a large (≥100pp) employee manual accurately, using RAG?*
Scoped to **onboarding-related questions** (what do I need, when do I start, what's the code of conduct, what counts as due process). General HR topics (leave, payroll, benefits) are a stretch goal / regression corpus, not the graded target.

**Question B — Document verification.** *Can the agent check that a submitted onboarding document is the right type and has all required fields filled in?* The six documents in question — NBI Clearance, SSS, Pag-IBIG, BIR Form 2316, PhilHealth, APE — are standard PH **pre-employment requirements**: paperwork a new hire must submit before HR can officially clear them to start, not something they file after they've started.
This is a **completeness and correctness check**, not a legal/forensic authenticity check. We are not detecting forged documents. We are checking: is this the document type we asked for, are the required fields present and well-formed, does the name match the employee, is it still within its validity window.

**Why one product and not two:** a new hire's actual experience is one conversation — "what do I need for Day 1" (Question A) immediately followed by "here's my NBI clearance" (Question B). Building them as the same agent is also what makes the memory/checklist state meaningful (Component 4) and is what the instructor's own framing implies by putting both under "due process — completeness check."

---

## 3. The "is this easy enough" problem, answered

A naive version of Question B is: take a photo, send it to Gemini with `response_schema`, get JSON back. That's maybe 20 lines of code and would rightly get marked down as decorative — a thin wrapper around one API call, not something that needed an agent.

What makes it a real component instead of a decoration is a **decision layer around the model call**, not the model call itself:

1. **Before extraction** — a deterministic, non-LLM image-quality check (blur, glare, skew, crop, resolution) decides whether the photo is even usable. Bad photo → rejected before any Gemini call, with a specific reason ("too blurry, please retake").
2. **The extraction call itself** — one Gemini multimodal call per document, typed output.
3. **After extraction** — deterministic rules (not the LLM's opinion) decide accept / reject / needs-human-review: right document type, all required fields present, fields well-formed, name matches the employee, not expired.

None of steps 1 or 3 involve asking an LLM to judge — they're plain code, testable with plain unit tests, and they're what a general chatbot (ChatGPT, a Google search) fundamentally cannot do, because they require deterministic business rules plus per-employee state. That's the actual answer to "is this easy enough."

---

## 4. In scope

| Area | What we're doing |
|---|---|
| FAQ answering | RAG over a large (~120pp+) onboarding-focused employee handbook, grounded answers with citations, "I don't know" when not covered |
| Retrieval quality | Dense retrieval baseline + one hybrid (BM25) upgrade, measured, so the "can we hit 100%?" question gets an honest curve instead of one number |
| Document classification | Identify which of the 6 onboarding document types was submitted |
| Document field extraction | Full field extraction + validation on **2 of the 6** document types (NBI Clearance, BIR Form 2316) — chosen as the deep examples |
| Document completeness/validity check | Deterministic rules: right type, required fields present, correct format, name match, not expired |
| Checklist state | Per-employee record of which of the 6 documents have been received/validated |
| Evals | Retrieval hit-rate, OCR extraction accuracy, validation precision/recall, at least one full reasoning-trace walkthrough |
| Employment type | PH **regular** employment only |
| Everything already built (Midterm) | Chat UI, API, RAG pipeline, memory, guardrails, MLflow monitoring, Docker — reused, not rebuilt |

## 5. Explicitly out of scope

| Area | Why not |
|---|---|
| **Complaint intake / harassment escalation / human-in-the-loop tickets** | Was the Midterm's product. Instructor: "can de-scope/don't need to do." **Already removed from the codebase entirely**, not just hidden from the demo. Don't rebuild it. |
| **Document authenticity / forgery detection** | Out of reach for a PoC and not what "completeness check" means. We check the document is complete and well-formed, not that it's not photoshopped. |
| **All 6 documents getting full field extraction** | Would triple the labeling/validation work for marginal grading benefit. Classification covers all 6 (so the checklist story is complete); deep extraction is scoped to 2. |
| **General (non-onboarding) HR FAQs as the graded target** | Leave/payroll/benefits content stays in the corpus for grounding continuity, but it's explicitly called a stretch goal, not the headline eval — the instructor's question was specifically about onboarding FAQs. |
| **Non-regular employment (contractual, project-based, agency)** | Not part of the instructor's framing; adding it would widen scope right when the spec is asking us to narrow it. |
| **Real employee data / real government IDs** | Legal and safety issue, also explicitly instructed: all OCR training/eval documents are synthetic mockups. |
| **Query rewriting / LLM-based reranking for retrieval** | Each would cost a Gemini API call per question against a **20-requests/day free-tier cap** we already hit repeatedly during Midterm development. Rejected on quota grounds, documented as a considered-and-rejected tradeoff (a legitimate RRL talking point, not a gap). |
| **Ollama for OCR** | Local models don't reliably do multimodal extraction; OCR stays on Gemini regardless of which LLM backend is running chat. |

---

## 6. Why the scope is shaped this way (the load-bearing decisions)

- **Narrow, not broad.** Spec explicitly says: *"a good Final Capstone does one thing well rather than many things adequately."* Two documents done deeply, four classified, beats six documents done shallowly.
- **Deterministic rules, not LLM judgment, for anything that gates accept/reject.** This is what makes the CV component defensible against "is this easy enough" — and it mirrors the same fail-safe design philosophy the removed escalation rules used, so it's not a new pattern for the team.
- **Synthetic data, generated not hand-made.** We render document mockups from templates and then apply image degradations (blur, skew, glare, low-res) programmatically. That gives us exact ground truth for free and a controlled ablation (clean vs. degraded accuracy) — real photographed documents would take longer to collect and be harder to label precisely.
- **The 100%-accuracy question gets a curve, not a yes/no.** A large real manual will have questions the agent genuinely can't answer from the text — the eval set includes some of those on purpose (see PLAN.md §3.3) so that abstaining correctly counts as success, not failure. Otherwise "100%" is only reachable by over-answering, which is a worse product.
- **Corpus starts synthetic, swaps to real later.** We're building a ~120-page synthetic PH handbook now so the pipeline and evals are provable today; the real DLSU Faculty Manual (~273pp) is the intended eventual swap-in, and the corpus loader is built so that's a data change, not a code change.

---

## 7. What a finished demo looks like

One continuous conversation, to make the abstract concrete:

1. Employee asks: *"What documents do I need before my first day?"* → agent answers from the handbook, cites the onboarding chapter, lists the 6 required documents.
2. Employee asks something the handbook doesn't cover → agent says "I don't know" and offers HR routing, instead of guessing.
3. Employee uploads a photo of their NBI clearance → quality gate checks the photo is usable → Gemini extracts name, ID number, issue date → validation rules confirm it matches the employee's name and isn't expired → agent replies "NBI clearance received and validated. Still missing: SSS, Pag-IBIG, BIR 2316, PhilHealth, APE."
4. Employee uploads a blurry photo of their SSS ID → quality gate rejects it before any extraction call → agent asks for a retake, explains why.
5. Employee asks *"how many have I submitted so far?"* → agent reads the checklist state, answers "1 of 6."

Every arrow in that sequence maps to a component in PLAN.md §4. If a teammate can't point at which module handles a given step, that's the signal something's still unclear — ask before building around it.

---

## 8. One-line owner summary

| Who | Owns | In plain terms |
|---|---|---|
| Baybayon | RAG, Advanced RAG, Evals | The handbook corpus, retrieval quality, and all the eval scripts that produce our numbers |
| Del Rosario | ReAct/Tool Use, Disambiguation, LLM abstraction | The agent's decision loop — what tool to call, when to ask a clarifying question, which LLM backend serves a call |
| Burayag | Memory, Guardrails | Conversation/checklist state, and the deterministic document-validation rules |
| Tamondong | Chat UI, API, LLMOps | The Streamlit UI, the FastAPI endpoints, MLflow monitoring |
| Everyone | CV/DS Integration (mandatory) | The OCR quality gate + extraction pipeline — this one's shared because it's the component the grade hinges on most |

Full detail, file paths, and build order: [PLAN.md](PLAN.md).
