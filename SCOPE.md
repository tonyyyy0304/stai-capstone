# Scope — What We're Building and Why

Plain-language companion to [PLAN.md](PLAN.md). PLAN.md is the implementation spec (file paths, schemas, build order). This doc answers the more basic question first: **what is the product, why does it look like this, and what are we explicitly not doing.**

---

## 0. The problem statement, in one paragraph

A newly hired DLSU faculty member arrives with two obligations they cannot yet meet on their own. First, a **203-page Faculty Manual** governs their employment — hiring criteria, teaching load, grade deadlines, the dress code, what happens if their syllabus is late — and nobody reads it cover to cover, so every question becomes an email to a Department Chair, an AFED rep, or HR. Second, they owe a stack of **pre-employment documents**, some faculty-specific (original TOR, three references, physical-fitness certification, clearance from their previous employer) and some national (NBI Clearance, SSS, PhilHealth, Pag-IBIG, BIR) — each of which someone checks by eye.

**We are building one agent that does both:** answers Faculty Manual questions with real citations (and says "I don't know" when the Manual doesn't cover it), and checks a submitted NBI Clearance for completeness, correctness, and freshness. Built for DLSU because we have its real 2021 Manual; designed so another PH university is a data swap, not a rewrite.

---

## 1. What the instructor actually said

Kept close to verbatim, including the latest round of comments:

> "Can we reach 100% evaluation accuracy on a handbook with ≥100 pages?"

> **"Can an agent reliably answer FACULTY ONBOARDING & FACULTY MANUAL questions on 3 specific topics?"** — 3 topics, **focus on RAG metrics here (MAIN)**.

> Pivot: **pre-employment, and onboarding for the handbook of faculty / university-related employment. Specific: DLSU faculty, but generalizable to other PH universities.**

> Computer Vision integration: **choose at least 1 (e.g. NBI clearance)** — verification that fields are filled up (name, date — typically 6 months expiry), via OCR/VLM. These are **"SECONDARY METRICS" compared to the main RAG metrics**, and should use **(a) real and (b) mock data as separate subsets**.

> "How well can an agent verify that your provided documents are correct and have the complete fields?" … **Counterpoint: "is this easy enough?"**

> Review the business use case / competitor analysis / quantitative cost or pain point. Presentation note: **make sure the problem statement you're tackling is clear.**

**The two lines that shape everything below:** *"is this easy enough?"* — the instructor pre-empting a team bolting one Gemini-vision call onto a chatbot and calling it CV integration (answered in §4). And *"RAG metrics are MAIN, CV metrics are SECONDARY"* — which is why the CV track shrank from six document types to one (§5).

---

## 2. The three topics

The main research question is not "any Manual question." It is three named topics, each densely covered by real Manual text:

| # | Topic | Plain-language version | Manual sections |
|---|---|---|---|
| **T1** | **Pre-employment & hiring requirements** | "What do I have to submit before I can start teaching?" — faculty-specific (original TOR & diplomas, biodata/CV, three references, teaching demonstration, clearance from the previous employer and the concerned government agency, certification of physical fitness) plus the national statutory set | Hiring Procedure p.24; per-rank Criteria for Hiring pp.16–23; part-time p.67; ASF p.83 |
| **T2** | **Academic & grading obligations** | "What am I required to do — and forbidden from doing — around grades, syllabi, and exams?" — grade prerogative and the Change of Grade form, grade submission deadlines, syllabus within the first two weeks, exam-material handling, and the sanction attached to each | General Functions §1.1 p.8; **Appendix F, Table of Offenses and Sanctions, p.135** (#8, #13, #14, #19, #20, #23, #24) |
| **T3** | **HR policies — dress code & leaves** | "What can I wear, and what leave can I take?" | **Appendix D, Dress Code / Attire and Grooming, p.131**; Benefits §8 Leaves pp.42–48 (service, sabbatical, study, research, vacation, sick, emergency, military, secondment, parental, tech-commercialization) |

**On the wording of T2:** call it **"Academic & grading obligations"** rather than "DOs and DON'Ts." When you need both halves explicitly, say **"obligations and prohibitions."** That is the Manual's own register — it speaks of *responsibilities* and *offenses* — and it survives a slide better than a listicle title.

### Two gaps we found by actually reading the PDF

These are not speculation; both were verified against the text.

- **T1 — the statutory documents are not in the Faculty Manual.** "NBI" appears **zero times** in 203 pages. SSS, PhilHealth, Pag-IBIG and BIR appear only in the *benefits and retirement* sections, never as things a new hire submits. The closest the Manual gets is the hiring criterion *"clearance from immediate past employer and concerned government agency"* and *"certification of physical fitness to teach."*
  **What we're doing about it:** adding one short supplementary corpus document covering the national statutory requirements, with its own source attribution, so answers cite `ph-statutory-preemployment` rather than pretending the Manual said it. And **no golden-set question may expect the Manual to answer a statutory question.**
- **T2 — half the listed "don'ts" belong to a different document.** Pop quizzes, "give a major requirement at least 4 weeks before its deadline," grading curves, bonus points, "free cut," and student grade appeals are **not in the Faculty Manual**. They're DLSU Student Handbook / academic-policy content. What the Manual *does* give T2 (grade prerogative, submission deadlines, syllabus timing, exam security, the offense table) is still enough for a topic.
  **Decision needed in Phase 0a:** either restrict T2 to Manual-backed items (safe), or also ingest the Student Handbook / academic policies (better — it makes multi-hop questions genuinely interesting, e.g. *"the Manual sets the grade deadline, the academic policy sets the requirement-notice rule"*). **We recommend the second, if the document can be obtained.**

### The T3 "is there a better alternative?" question, answered

Dress code + leaves works — both are unambiguous and well-covered. But if a stronger third topic is wanted, ranked by how much real Manual text backs them:

1. **Probation, renewal & permanency** (pp.29–33) — **our recommendation if you swap.** It's procedurally exact (good for scoring), it's the single thing new faculty most misunderstand in year one, and it's unambiguously *onboarding*, which "HR policies" only half is.
2. **Working hours & load** (pp.9–15) — the 40-hour week, 12 teaching hours, 2½ consultation hours per 3 units, the three-preparation limit. Very high question density, very exact numbers.
3. **Grievance procedure & Table of Offenses** (Appendices F/G) — strong, but it overlaps T2, so it competes rather than complements.

---

## 3. Two questions, one product

**Question A — Faculty Manual Q&A. THIS IS THE MAIN EVENT.** *Can the agent reliably answer faculty onboarding and Faculty Manual questions on the three topics above, using RAG over a real 203-page manual?* All headline metrics come from here (PLAN.md §9, PRIMARY block).

**Question B — Document verification. SECONDARY.** *Can the agent check that a submitted **NBI Clearance** is the right document, has its required fields filled in (name, date of issue, reference number, purpose), matches the person, and is still fresh (6-month window)?* This is a **completeness and correctness check, not a forensic authenticity check.** We are not detecting forgeries.

**Why one product and not two:** a new hire's real experience is one conversation — "what do I need before I can start?" (A) immediately followed by "here's my NBI clearance" (B). Building them as one agent is also what makes the memory/checklist component meaningful, and it lets the agent remember something that changes every A-answer: **whether you're full-time, part-time, or Academic Service Faculty.**

---

## 4. The "is this easy enough" problem, answered

A naive version of Question B is: take a photo, send it to Gemini with `response_schema`, get JSON back. Twenty lines, and rightly marked down as decorative.

What makes it a real component is a **decision layer around the model call**, not the model call:

1. **Before extraction** — a deterministic, non-LLM image-quality check (blur, glare, skew, crop, resolution) decides whether the photo is usable at all. Bad photo → rejected *before* any API call, with a specific reason ("too blurry, please retake"). This is also our main defence against the 20-requests-per-day Gemini quota.
2. **The extraction call** — one Gemini multimodal call, typed output, each field carrying both the value and the verbatim text the model claims it saw (so a human can audit it).
3. **After extraction** — deterministic rules, not the LLM's opinion, decide accept / reject / needs-human-review: right document type, all four required fields present, each field well-formed, name matches the faculty record (fuzzy, PH name forms), issue date within the freshness window.

Steps 1 and 3 involve no LLM judgment. They're plain code with plain unit tests, and they're what a general chatbot fundamentally cannot do, because they need deterministic business rules plus per-hire state.

**One honest detail to say out loud:** the six-month window is an *employer freshness policy*, not the document's own expiry — an NBI Clearance prints a one-year validity. We encode it as `NBI_VALIDITY_MONTHS` in config precisely so that distinction stays visible and each institution can set its own.

---

## 5. In scope

| Area | What we're doing |
|---|---|
| **Manual Q&A (MAIN)** | RAG over the **real DLSU Faculty Manual 2021 (203pp)**, scoped to T1/T2/T3, grounded answers with **page-level citations**, "I don't know" when not covered |
| Faculty-class disambiguation | Full-time / part-time / ASF have *different* answers to the same question — the agent asks which you are rather than guessing, and this is scored |
| Retrieval quality | Dense baseline + hybrid (BM25 + RRF) upgrade, measured, so "can we hit 100%?" gets an honest curve per topic instead of one number |
| **NBI verification (SECONDARY)** | One document type, done properly: quality gate → extraction → deterministic validation |
| Two eval subsets | **Real** NBI clearances (consented, small n, never committed) and **mock** synthetic ones — reported as separate numbers, never pooled |
| Checklist state | Per-hire record of what's been received and validated, plus remembered faculty class |
| Evals | Per-topic retrieval hit-rate, abstention accuracy, disambiguation accuracy, OCR field accuracy, validation precision/recall, one full reasoning-trace walkthrough |
| Generalizability | Nothing DLSU-specific in the code; another PH university's manual is a data swap |
| Everything already built (Midterm) | Chat UI, API, RAG pipeline, memory, guardrails, MLflow, Docker — reused, not rebuilt |

## 6. Explicitly out of scope

| Area | Why not |
|---|---|
| **Complaint intake / harassment escalation / tickets** | Was the Midterm's product. Instructor: "can de-scope." **Already removed from the codebase**, not just hidden. (The Manual's own grievance procedure is *corpus content* — the agent answers questions about it; it does not file grievances.) |
| **Document authenticity / forgery detection** | Out of reach for a PoC and not what "completeness check" means. |
| **Deep extraction on document types beyond NBI Clearance** | The instructor said "choose at least 1." CV is the secondary track; one type done honestly beats six done shallowly. BIR 2316 is the obvious stretch if RAG lands early. |
| **Manual topics outside T1–T3** | Ranks, promotion grids, retirement, research incentives, AFED by-laws stay *in the index* (they're part of the same PDF, and removing them would be artificial — they're realistic distractors that make hit-rate meaningful) but they're not the eval target. |
| **Non-faculty university staff** | The Faculty Manual covers faculty; co-academic and administrative personnel have a different handbook. |
| **The old synthetic corpus** | The 10 generic PH-corporate HR docs and the planned 12-chapter synthetic handbook are **retired**. Writing a fake 120-page handbook when a real 203-page one is in hand would make the accuracy claim unfalsifiable. |
| **Query rewriting / LLM reranking** | Each costs a Gemini call per question against a **20-per-day free-tier cap** we hit repeatedly during the Midterm. Rejected on quota grounds — a legitimate RRL talking point, not a gap. |
| **Ollama for OCR** | Local models don't reliably do multimodal extraction; OCR stays on Gemini regardless of which backend serves chat. |

---

## 7. Why the scope is shaped this way

- **Real corpus, not synthetic.** The whole point of the ≥100-page question is scale *and* messiness. A real 203-page typeset PDF with three parallel faculty-class structures and fourteen appendices is a genuinely hard retrieval target; a handbook we wrote ourselves would not be. The cost is that PDF parsing is now on the critical path (PLAN.md §10) — we accept that.
- **The 100% question gets a curve, not a yes/no.** The eval set deliberately includes questions the Manual genuinely cannot answer, and questions where the right move is to ask "full-time or part-time?" Abstaining and clarifying correctly count as *success*. Otherwise "100%" is only reachable by over-answering, which is a worse product.
- **Deterministic rules, not LLM judgment, for anything that gates accept/reject.** This is what makes the CV component defensible against "is this easy enough."
- **Real *and* mock data, separately.** Mock data gives exact ground truth for free and a controlled clean-vs-degraded ablation. Real data is the only thing that measures the sim-to-real gap. Pooling them would hide exactly the number that matters, so we never do.
- **Narrow, not broad.** The spec says a good Final Capstone "does one thing well rather than many things adequately." One manual, three topics, one document type.

---

## 8. Handling real documents — the rules

Non-negotiable, because this is the one part of the project with consequences outside the course:

- Real NBI Clearances come only from team members and consenting volunteers, with consent recorded.
- `data/references/real/` is **gitignored**. Never committed, ever.
- Names and NBI reference numbers are **hashed** in anything that leaves the machine. No extracted field value is ever logged to MLflow.
- EXIF is stripped on upload (geolocation is PII).
- **No real document appears in a slide, screenshot, or recorded demo.** The demo uses mock documents only.
- Anyone can withdraw: that means deleting the image, its ground truth, and its cached extraction.

---

## 9. What a finished demo looks like

One continuous conversation:

1. *"What do I need to submit before I can start teaching?"* → agent answers from the Manual's Hiring Procedure, cites the page, and separately cites the supplementary statutory doc for NBI/SSS/PhilHealth/Pag-IBIG/BIR.
2. *"How much service leave do I get?"* → agent asks **"are you full-time, part-time, or academic service faculty?"** rather than guessing, then answers the right section and remembers the answer.
3. *"When does my syllabus have to be out?"* → cites Appendix F #23 (within the first two weeks of classes) **and** the sanction attached.
4. Asks something the corpus doesn't cover (e.g. a student-handbook grading rule) → **"I don't know"** plus a routing suggestion, instead of guessing.
5. Uploads a photo of an NBI Clearance → quality gate confirms it's usable → extraction pulls name, date of issue, reference number, purpose → rules confirm the name matches and the issue date is within six months → *"NBI Clearance received and validated. Still missing: TOR, physical-fitness certification."*
6. Uploads a blurry photo → quality gate rejects it **before any API call** → agent asks for a retake and says why.
7. *"What have I submitted so far?"* → reads checklist state.

Every arrow maps to a component in PLAN.md §4. If a teammate can't point at which module handles a step, ask before building around it.

---

## 10. One-line owner summary

| Who | Owns | In plain terms |
|---|---|---|
| Baybayon | RAG, Guardrails, ReAct tools | The Manual corpus, retrieval quality, the grounding/safety layers, and the tool the agent calls to reach the CV pipeline |
| Del Rosario | CV integration + its evaluation | The quality gate, the NBI extraction call, and the OCR/validation metrics |
| Burayag | RRL, end-to-end evals | The literature/design-tradeoff story and the full-system eval runs |
| Tamondong | Evals dataset, Chat UI, API, LLMOps | The golden set, the Streamlit UI, the FastAPI endpoints, MLflow |

Full detail, file paths, and build order: [PLAN.md](PLAN.md).
