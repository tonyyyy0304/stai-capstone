# CV/DS Integration (Component 14) — Conceptual & Technical Map

## Context

Owner: **Del Rosario — Component 14, CV/DS Domain Integration, end to end** (PLAN.md §4 ownership table: "Del Rosario | CV Integration and its evaluation"). `TASKS.md` (which split the OCR modules across three people) has been removed from the repo — **PLAN.md is now the sole source of truth for ownership and build order**; this doc maps PLAN.md §4.1/§4.2 into concrete files and sequencing, nothing more.

**Status: all 9 phases done (0–7 plus optional 4a) and live-verified end to end**, including in a real user test session against the running Docker stack with genuine real NBI clearances (not just mock fixtures). `POST /upload-doc` → quality → extraction → six-rule validation → cross-document check → checklist state, `GET /onboarding-status/{employee_id}`, `Intent.DOCUMENT_UPLOAD`/`DOCUMENT_STATUS` chat routing, and the styled UI (checklist card in the sidebar, upload flow in the main chat column with a spinner during the vision call) are all built, tested (345 passed, 1 skipped), and confirmed working live — not just against mocks. Three real specimens now collected; testing against them surfaced and fixed two genuine calibration bugs after the mock-only Phase 2 pass couldn't have caught either (see Phase 2's `quad_found` recalibration below) and confirmed Rule 5's validity-window math behaves correctly on a real document. UI structure changed once more after live feedback: the uploader moved from the sidebar into the main chat column (sidebar is checklist-only now), matching how the user actually wanted to use it once they saw it running.

The instructor's live objection to this track is *"is this easy enough?"* — a clean mockup fed to a multimodal model returning JSON is decorative. Everything below is organized around making that objection answerable.

**Scope note.** PLAN.md §4.2/§1.5 originally scoped Component 14 to **one document type, done deeply** — NBI Clearance — with everything else (BIR 2316, SSS, PhilHealth, Pag-IBIG, physical-fitness cert) stub-only, and CLAUDE.md's explicit priority is "if effort has to be cut, cut the CV stretch goals" ahead of RAG work. This doc now covers **two document types done deeply** — NBI Clearance plus a second, unified `government_id` type covering National ID / Driver's License / Passport under one classification — per explicit direction. Flagging this here because it's a real expansion beyond what PLAN.md currently says, not a hidden one; PLAN.md §4.2's table should be updated to match once this is reviewed.

**Companion doc:** `CV_PIPELINE_WALKTHROUGH.md` follows one applicant's NBI Clearance through every stage of this pipeline with concrete data at each hop, plus a debugging playbook and offline testing recipes. This doc explains structure; that one explains flow — read it when something's actually broken.

---

# Part 1 — Conceptual map

## 1.1 What the component actually decides

Not "read the text off an image." The decision is:

> **Given a photo a new hire just uploaded, should a human look at it?**

Three outcomes, never two: `accepted | rejected | needs_review`. The costly failure is **false auto-pass** — accepting an invalid document — not false review. Everything fails toward `needs_review`.

Two documents now go through this decision per applicant — NBI Clearance and a government ID (§1.4a) — each independently, plus one more check that only makes sense once both exist: do they agree on who this person is?

## 1.2 The three-layer trust ladder

This is the load-bearing idea and the answer to "is this easy enough?". The LLM is the **weakest-authority** layer, sandwiched between two deterministic ones:

```
  bytes in
     │
 ┌───▼──────────────────────────────────────────────┐
 │ LAYER 1 — OpenCV quality gate   (deterministic)  │  src/ocr/quality.py
 │ blur / exposure / skew / resolution / quad       │
 │ verdict: pass | warn | reject                    │
 └───┬──────────────────────────────────────────────┘
     │  reject → STOP. No Gemini call spent. Ask for re-upload.
     │  pass|warn ↓  (+ optional preprocess: deskew, warp, CLAHE, upscale)
 ┌───▼──────────────────────────────────────────────┐
 │ LAYER 2 — Gemini multimodal extraction  (LLM)    │  src/ocr/extractor.py
 │ ONE call, response_schema-typed                  │
 │ per field: value + verbatim_text + confidence    │
 │ DETECTION ONLY — proposes, never decides         │
 └───┬──────────────────────────────────────────────┘
     │
 ┌───▼──────────────────────────────────────────────┐
 │ LAYER 3 — Validation rules 1–6  (deterministic)  │  src/guardrails/doc_validation.py
 │ type · completeness · format · identity ·        │
 │ validity window · fail-safe                      │
 │ outcome: accepted | rejected | needs_review      │
 └───┬──────────────────────────────────────────────┘
     │
  checklist state (SQLite)  →  conversational reply
```

Three properties fall out of this shape, and each is a slide:

1. **The model never gets the last word.** Same detection-vs-policy split already shipped in `src/guardrails/llm_judge.py:88` (`to_guardrail_result()` owns the gate, the model owns only the verdict fields). Reusing that pattern is a consistency argument, not a coincidence.
2. **Layer 1 is a real accuracy decision *and* the quota mechanism.** A `reject` short-circuits before any API call. Gemini free tier is 20 requests/day (PLAN.md §2.1) — the gate is the only reason a 50-image eval is survivable.
3. **A deterministic signal beats model confidence.** A Rule 3 format failure forces composite confidence to zero regardless of how sure the model was. The LLM cannot talk past a regex.

## 1.3 Composite confidence

```
composite = min(quality.normalized_quality, extraction.overall_confidence)
composite = 0.0  if any Rule 3 (format) check failed
```

`min`, not a weighted average: a good extraction from an unreadable image is not trustworthy, and neither is a low-confidence read of a crisp image. `min` is the conservative operator and it's one line to defend on the deck.

## 1.4 The outcome table (this is a slide)

| Condition | Outcome | Why |
| --- | --- | --- |
| Quality verdict `reject` | `rejected` | Re-upload. Never spend a call, never guess. |
| Extraction returned `None` / unparseable, quality OK | `needs_review` | Fail-safe (Rule 6). |
| Rule 1 — `doc_type` mismatch | `rejected` | Wrong document in the slot. Prompt re-upload, never infer. |
| Rule 2 — required field missing | `needs_review` | The completeness check. Might be the photo, might be the doc. |
| Rule 3 — format invalid | `needs_review`, composite forced to 0 | Deterministic distrust. |
| Rule 4 — name doesn't match record | `needs_review` | **Never auto-reject** — PH name forms are messy (middle initials, `Jr./III`, surname-first, `ñ`). |
| Rule 5 — outside validity window | `rejected` | Expiry is an unambiguous computed fact. |
| Rule 6 — quality `warn` AND composite < `OCR_CONFIDENCE_FLOOR` | `needs_review` | The catch-all. |
| All pass, composite ≥ floor | `accepted` | Only path to auto-pass. |

Rule 5 is the one place this rejects outright rather than escalating, and it needs its caveat said out loud: **`NBI_VALIDITY_MONTHS = 6` is an employer freshness policy, layered on top of — not instead of — the clearance's own printed `VALID UNTIL` date.** A real sample (see §2.3) shows the document prints its own expiry directly; Rule 5 now takes the **earlier** of that and the employer's 6-month window from `date_printed`, so whichever is stricter governs. Living in config is what makes the employer-policy half of that distinction visible and tunable per institution.

**A field that wasn't in this table until a real sample was reviewed: `remarks`.** The document prints a clearance-status line ("NO DEROGATORY", in the sample) — this is the actual background-check result, arguably the single most load-bearing field on the whole document, and it was absent from the original 4-field design entirely. It's folded into **Rule 3 (format)**, not a new rule: an unrecognized remarks value forces `composite_confidence = 0` exactly like any other format failure, which routes to `needs_review` — never a silent `accepted` on a value the system doesn't recognize as clean. See §2.3/§2.5/§2.7.

## 1.4a The second document, and the three-way identity check

Reference samples reviewed for this section: `National_id.png`, `drivers_license.jpg`, `passport_ph.jpg` — all watermarked `SPECIMEN` or carrying an obvious template/marketplace watermark (`AUTODEAL` on the license). Same status as the NBI sample (§2.3): good for layout, **not** real-subset eligible, gitignored (Part 5 §9).

**One unified document type, three sub-layouts.** Per direction, National ID / Driver's License / Passport are **not** three separate `DocType` entries — they're one `DocType.GOVERNMENT_ID`, discriminated internally by an `id_type` field (`national_id | drivers_license | passport | other`). This mirrors the registry's existing generalizability claim (§2.5): the pipeline doesn't grow a new top-level branch per ID variant, one registry entry handles all three via a per-`id_type` field table.

**What's common across all three layouts** (and therefore what the unified extraction schema asks for): a name, a date of birth, an ID number, and — for two of the three — an expiry date:

| | National ID (PhilSys) | Driver's License (LTO) | Passport (DFA) |
| --- | --- | --- | --- |
| Name | Apelyido / Mga Pangalan / Gitnang Apelyido (family/given/middle, separate) | one line: `Last, First Middle` | Surname / Given Names (separate) |
| DOB | Petsa ng Kapanganakan | Date of Birth | Date of Birth |
| ID number | PSN/PCN, `1234-5678-9101-1213` | License No., `N03-12-123456` | Passport No., `P0000000A` |
| Expiry | **not printed on this sample** — PhilSys National ID commonly has no expiry for adult citizens | Expiration Date, `2022/10/04` | Date of Expiry, `26 JUN 2021` |
| Issue date | Araw ng pagkakaloob (back) | implied near photo, not clearly labeled | Date of Issue |
| Machine-readable zone | none | none | **MRZ**, two lines, fixed-width OCR-B font — the most reliable extraction source on the whole document (§2.6) |

**Why `date_of_birth` gets added to the NBI schema too, reopening a decision from §2.3.** The earlier revision deliberately excluded DOB from NBI extraction as PII-minimization — nothing consumed it. Something does now: **cross-document consistency** needs a field both documents actually carry, and name-only matching is weaker than name+DOB. This is a justified re-opening of that decision, not scope creep back toward "extract everything printed" — DOB is added because a concrete validation rule needs it, the same bar every other field in this doc had to clear.

**The three-way check** (this is the new headline mechanic):

```
                    Faculty/employee record (session)
                    (from onboarding_profile / HR data)
                         ╱                    ╲
              Rule 4 (existing)          ID-Rule 4 (new)
              name match, fuzzy          name match, fuzzy
                       ╱                          ╲
        NBI extraction ───── cross-document ───── ID extraction
                          (new) name + DOB match
```

Three independent pairwise checks, none of them new *in kind* — all reuse the same `rapidfuzz.token_set_ratio` + `config.NAME_MATCH_THRESHOLD` pattern already built for NBI's Rule 4 (§2.7):

1. **NBI ↔ employee record** — already built (existing Rule 4).
2. **ID ↔ employee record** — same rule, second document (§2.7, ID-Rule 4).
3. **NBI ↔ ID** — new, runs only once both documents exist for the employee (§2.7, cross-document check).

**Any single mismatch → `needs_review`, never auto-reject** — same philosophy as the original Rule 4 and for the same reason (PH name-form variance across two independently-transcribed documents is *expected* noise, not evidence of fraud). All three checks failing simultaneously is a stronger signal than any one alone, but this design doesn't escalate that distinction into a harsher outcome — `needs_review` is `needs_review`; a human sees the detail either way.

**Also required now:** "is the ID still valid" — an ID-side Rule 5 analog (§2.7), with the National-ID-has-no-expiry case treated as a pass-through, not a missing-field failure (it's correct behavior for that document type, not an extraction gap).

**Explicitly out of scope, stated so it isn't assumed later:** face/photo matching between the ID photo and the applicant, or between the ID photo and the NBI Clearance photo. Everything above is text-field comparison only. Biometric matching is a different, much heavier component (face embeddings, a similarity model, its own threshold-calibration problem) and nothing in this plan implies it.

## 1.5 Where it plugs into the agent

The upload is **not** a ReAct tool argument. Raw image bytes must never cross the function-calling boundary. The split:

- `POST /upload-doc` runs the whole pipeline **synchronously, outside the ReAct loop**, and writes the result to the checklist table.
- The agent's tools are **read-only over that result**: `get_onboarding_status`, `validate_checklist`, and `extract_document` taking a `source_hash` referencing an already-uploaded image (re-run against cache, never new bytes).

This is a deviation from a literal reading of PLAN.md §4.4 and worth stating on review — it's what keeps the tool schema small and the loop cheap.

## 1.6 How this connects to the existing ReAct agent — concretely

This is not a second agent. It's the same `run_turn()` loop (`src/agent/orchestrator.py:137`) that already answers Faculty Manual questions, with three additions layered onto infrastructure that's already faculty-onboarding-scoped — `OUT_OF_SCOPE_REPLY` (`orchestrator.py:60`) and the router's `DEFAULT_CLARIFYING_QUESTION` (`router.py:12`) already talk about faculty class and onboarding, not the Midterm's generic HR framing. No parallel pipeline, no separate entry point.

**What's shared, unchanged:**
- `_check_input()` (`orchestrator.py:108`) — the deterministic + LLM-judge guardrail stack runs on every message *before* intent is known, so a hostile or injection-laden message attached to a document-upload turn is caught exactly like a FAQ turn.
- `classify_intent()` (`router.py:22`) and the confidence-floor clarification path (`needs_clarification`, `router.py:60`) — reused verbatim for the new intents.
- `AgentResponse` / `_RunState` (`orchestrator.py:87-105`) — extended, not replaced (see below).

**What's added (PLAN.md §4.4):**

1. **`Intent` gets two new members** (`schemas.py:14`): `DOCUMENT_UPLOAD`, `DOCUMENT_STATUS`. `ROUTER_PROMPT` (`prompts.py`) is extended with examples that disambiguate *asking about* a document ("what documents do I need to submit?" → `FAQ`, routes to `search_kb` against the T1 Manual content) from *submitting/discussing* one ("I uploaded my NBI, what's next?" / "is my clearance still valid?" → `DOCUMENT_STATUS`) — this exact confusion is called out as a required test case in PLAN.md §4.4.
2. **Three new tools**, registered in `_function_declarations()` (`orchestrator.py:300`) next to `search_kb`/`search_web`, dispatched in `_execute_tool()` (`orchestrator.py:338`):
   - `get_onboarding_status(employee_id)` → reads `src/memory/onboarding_status.py`, returns a `ChecklistStatus` dict.
   - `validate_checklist(employee_id)` → same read path, phrased as "what's still missing."
   - `extract_document(source_hash)` → re-runs `src/ocr/extractor.py` against an **already-cached** image (uploaded via `/upload-doc`, never inline bytes) — lets the agent explain *why* something needs review ("the date of issue field was unreadable") without re-spending a Gemini call, since the cache in §2.6 already holds the result.
3. **`AgentResponse` gains `validation: ValidationResult | None`** (extends `orchestrator.py:87`) so `POST /upload-doc`'s pipeline result and a `DOCUMENT_STATUS`-routed chat turn both flow through the same response shape into `ChatResponse` (`api.py:58`) — the UI's checklist panel and its chat bubble read one contract either way.
4. **Memory carries `faculty_class` across both flows.** `src/memory/onboarding_status.py` stores it once it's resolved (via the RAG-side disambiguation tier, PLAN.md §3.1); `search_kb` and the CV pipeline's Rule 4 identity check both read it, so a faculty member is asked "full-time, part-time, or ASF?" **once per session**, not once per flow.

**One turn, traced end to end** (this is the PLAN.md §9 "trajectory walkthrough" slide):

```
1. "I uploaded my NBI clearance, is it good?"
2. _check_input()           → passes (no injection/toxicity)
3. classify_intent()        → DOCUMENT_STATUS, confidence 0.91
4. ReAct loop selects       → get_onboarding_status(employee_id)
5. tool reads SQLite        → { nbi_clearance: needs_review, reason: "name did not
                                 match faculty record (Rule 4)" }
6. model composes reply     → "Your NBI clearance needs a quick human check — the
                                 name on it didn't automatically match our record.
                                 HR will follow up. Everything else looks complete."
7. AgentResponse.validation → carried through to ChatResponse, UI checklist panel
                                 updates its NBI row to "needs review"
```

Note the extraction and the six rules already ran **before** this turn, inside `POST /upload-doc` — the chat turn above never touches Gemini vision at all, it's a pure SQLite read plus one chat-model call. That's the same quota discipline as Layer 1's reject short-circuit, applied at the conversation layer.

## 1.7 PII handling for document uploads — a separate mechanism from `src/guardrails/pii.py`

**`pii.py` does not apply to this track.** It's regex-only (`email`/`phone`/`employee_id` patterns, `pii.py:20-28`), scoped to chat-message text, used for redaction before logging — never as a blocking gate (`pii.py:9-13` says so explicitly, and it's the right call for FAQ turns where an employee legitimately types a phone number). An NBI Clearance's PII — full name, a government reference number, sometimes an address — never enters as message text, so this module never sees it. Document PII is protected by a different, deliberate set of controls, most already threaded through §1–2 above but not gathered in one place until now:

| Surface | Control | Where |
| --- | --- | --- |
| Raw image bytes | Never persisted past the request (`PERSIST_UPLOADS = False`); `cv2.imdecode` strips all EXIF, including geolocation, as a side effect of decoding to a raw array | §2.2, §2.4 |
| Extracted field values (name, reference no.) | **Never** written to the `onboarding_documents` table — that schema stores only `doc_type`/`status`/`outcome`/`source_hash`, no field values at all | §2.3 `OnboardingDocument`, §2.8 |
| MLflow tags/metrics | Fail-closed allowlist; extending it adds `doc_type`/`validation_outcome`/`quality_verdict`, never a field value | §2.8, `monitoring.py:22-45` |
| User-facing / logged messages | `RuleResult.detail` and `ValidationResult.message` are contractually PII-free by field description — say *"required field missing: valid_until"*, never the value | §2.3, §2.7 |
| Fields the real document prints but this system never extracts | NBI: address, place of birth, citizenship, civil status, gender, photo, signature, QR/fingerprint images, barcode. Government ID: address, blood type, marital status, place of birth, sex, height/weight/eye color (license), restrictions (license), signature, photo, QR/MRZ beyond the fields actually parsed. Deliberate ceiling, not an oversight — see §1.4a/§2.3/§2.5 for what *is* extracted and why | §2.3, §2.5 |
| Cross-document identity check (§1.4a, new) | Compares two already-in-memory extraction results at upload time; the comparison **outputs one `RuleResult`** (`passed: bool`, `detail` naming only *which* sub-check failed — `"name"`/`"date_of_birth"`), never the compared values themselves. The sibling document's values are read via a **cache lookup on its stored `source_hash`** (§2.7) — never re-collected, never persisted anywhere beyond the transient comparison | §1.4a, §2.7 |
| Real-subset documents | Name + reference/ID number hashed in anything leaving the machine; mock-only in demos/slides; `source_hash` is the deletion key for consent withdrawal | §1.7 below, PLAN.md §3.5a |
| Disk-cached extractions (`evals/results/ocr_cache/*.json`) | Hold the **full** extraction result, including `verbatim_text` — real PII at rest, for both document types now. Already covered by the blanket `evals/results/` rule in `.gitignore` (verified, not assumed), so this doesn't need a new gitignore line — but it does mean this directory must never be zipped into a submission or shared outside the team's machines. | verified against `.gitignore` |

**The gap this plan hadn't closed: session memory.** `src/memory/session.py` persists every turn's raw `content` indefinitely in `session_turns` (`session.py:39-56`), and `get_history()` feeds that straight back into the model's context window on the next turn (`session.py:70-81`). The ReAct loop's tool-call boundary (`_execute_tool()`, `orchestrator.py:338`) is a *dict*, not free text, so `pii.py`'s regex approach can't reach it anyway — meaning the only real control is discipline in what the tool observation dict contains.

**Consequence for the three new tools (§1.6):** `get_onboarding_status`, `validate_checklist`, and `extract_document` must return only status/outcome/confidence/PII-free reason strings — never `family_name`/`first_name`/`middle_name`, `reference_no`, `remarks`, or `verbatim_text` values. This falls out for free from the schema decisions already made (`ChecklistStatus`/`OnboardingDocument` carry no field values, `RuleResult.detail` is contractually PII-free), but it needs to be enforced as a rule at `_execute_tool()`'s three new branches, not assumed — a future edit that adds a "debug" field with `extracted.family_name.value` for convenience would silently defeat every control above by writing it straight into `session_turns`. Worth a one-line comment at each branch and a unit test asserting the returned dict has no field-value keys.

## 1.8 Two-subset discipline (non-negotiable)

`mock/` (synthetic, in-repo, carries every demo and slide) and `real/` (consented, gitignored, never in a screenshot). **Every metric is a pair of numbers with each `n` stated, never pooled.** The gap between them *is* the finding — it's the sim-to-real honesty check, and reporting it is a strength on the deck, not a concession. This now applies **per document type** — NBI and government ID each get their own real/mock split and their own reported numbers; a `government_id` accuracy figure is never pooled with an `nbi_clearance` one any more than real is pooled with mock.

`.gitignore` already ignores `data/references/real/`. That's the only part of this track currently in the repo.

---

# Part 2 — Technical map

## 2.1 Dependencies — `requirements.txt` + `requirements-api.txt`

```
opencv-python-headless==4.11.0.86   # headless: the slim Docker base lacks libGL
Pillow==11.2.1                      # mock dataset rendering
rapidfuzz==3.13.0                   # Rule 4 name matching
python-multipart==0.0.20            # FastAPI UploadFile
```

## 2.2 `src/config.py` — new block

Follow the existing idiom exactly: `os.environ.get(NAME, default)` for env-overridable strings (`config.py:29-32`), bare module constants with an explanatory comment for thresholds (`config.py:45-46`), tuples not lists for allowlists (`config.py:51`), `Path` off `REPO_ROOT` (`config.py:15-21`), client factories as functions so tests can mock (`config.py:140`).

```python
# --- CV/OCR document verification (Component 14) ---
# Gemini is the default and the evaluated path — every threshold, cache entry,
# and eval number in this doc assumes it. VISION_PROVIDER mirrors the existing
# LLM_PROVIDER/EMBEDDING_PROVIDER switch (config.py:29-65) so the same
# quota-relief pattern applies to vision: switchable, not blended, and NOT
# derived from LLM_PROVIDER — GEMINI_CHAT_MODEL is a lite tier tuned for
# quota, not multimodal extraction, so "vision follows chat" would be wrong
# even before quota enters the picture.
VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "gemini")
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")
ACTIVE_VISION_MODEL = (
    GEMINI_VISION_MODEL if VISION_PROVIDER == "gemini" else f"ollama:{OLLAMA_VISION_MODEL}"
)

# References folder (renamed from the original onboarding_docs/ during the
# CV rework — see the Phase 0/1 completeness note in Part 3): mock + real
# datasets live here, plus a samples/ subfolder for gitignored layout-
# reference specimen images (data/references/samples/, Part 5).
REFERENCES_DIR = DATA_DIR / "references"
MOCK_DOCS_DIR = REFERENCES_DIR / "mock"
REAL_DOCS_DIR = REFERENCES_DIR / "real"
OCR_CACHE_DIR = REPO_ROOT / "evals" / "results" / "ocr_cache"

# Image quality gate (src/ocr/quality.py). PROVISIONAL — calibrate against the
# mock dataset before Phase 5; see the calibration procedure in §2.4.
BLUR_VARIANCE_FLOOR = 100.0     # variance of Laplacian; below = reject
BLUR_VARIANCE_WARN = 250.0      # below = warn
MAX_SKEW_DEG = 12.0             # beyond = reject
SKEW_WARN_DEG = 5.0
MIN_IMAGE_DIM_PX = 640          # shorter side
EXPOSURE_CLIP_CEILING = 0.10    # fraction of pixels at 0 or 255 before warn
OCR_PREPROCESS = True           # ablated off via --no-preprocess in the eval

# Extraction + validation
OCR_CONFIDENCE_FLOOR = 0.70     # composite below this -> needs_review
NAME_MATCH_THRESHOLD = 85       # rapidfuzz token_set_ratio, 0-100
NBI_VALIDITY_MONTHS = 6         # EMPLOYER freshness policy, layered on top of
                                # (not instead of) the document's own printed
                                # valid_until — Rule 5 takes whichever is
                                # stricter. See PLAN.md §4.1 Rule 5, §2.7.
# Normalized clean-status strings for the `remarks` field (Rule 3). Deliberately
# incomplete and tunable — extend as real samples show more phrasing variants.
# Anything NOT in this tuple fails Rule 3 outright; never assumed clean by default.
NBI_CLEAN_REMARKS = ("NO DEROGATORY", "NO DEROGATORY RECORD", "NO RECORD")
# Government ID (§1.4a) — one required id_number pattern per sub-type, each
# from exactly one specimen sample, none confirmed. See doctypes.py §2.5.
ID_NUMBER_PATTERNS = {
    "national_id": r"^\d{4}-\d{4}-\d{4}-\d{4}$",
    "drivers_license": r"^[A-Z]\d{2}-\d{2}-\d{6}$",
    "passport": r"^[A-Z]\d{7}[A-Z]$",
}
REQUIRED_ONBOARDING_DOCS = ("nbi_clearance", "government_id")

# Upload handling
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_MIME = ("image/jpeg", "image/png")   # PDF is a stretch goal
PERSIST_UPLOADS = False         # don't keep raw images past the request
```

Plus one factory, same shape as `get_llm_client()` (`config.py:163-176`) but its own switch — vision and chat can be on different providers at once (e.g. `LLM_PROVIDER=ollama` for cheap chat testing, `VISION_PROVIDER=gemini` for reliable extraction):

```python
def get_vision_client():
    """Returns the active vision client per VISION_PROVIDER: a real
    google-genai Client (default) or OllamaClient. Both expose
    .models.generate_content(model, contents, config); callers don't need to
    know which one they got — same contract as get_llm_client()."""
    if VISION_PROVIDER == "gemini":
        return get_gemini_client()
    if VISION_PROVIDER == "ollama":
        from src.agent.llm_client import OllamaClient
        return OllamaClient(OLLAMA_URL, OLLAMA_VISION_MODEL)
    raise RuntimeError(f"Unknown VISION_PROVIDER={VISION_PROVIDER!r}; expected 'gemini' or 'ollama'.")
```

Keep `.env.example` in sync (`VISION_PROVIDER`, `GEMINI_VISION_MODEL`, `OLLAMA_VISION_MODEL`).

## 2.3 `src/schemas.py` — new section

Match the file's conventions: `(str, Enum)` with lowercase snake values, `Field(description=...)` on every field, `default_factory=list`, `Field(ge=0.0, le=1.0)` for bounded floats, and a class docstring that says *why the field exists* (see `LLMJudgeVerdict:122-162` as the template).

**Revised against a real sample.** The 4-field design (full name / date of issue / reference no. / purpose) below was written before anyone had looked at an actual NBI Clearance layout. A sample review (a demo-template image, `nbiclearance.org` branding — layout-accurate but **not** eligible as a real-subset document per §1.8/PLAN.md §3.5a) surfaced a real layout:

```
Republic of the Philippines / Department of Justice / National Bureau of Investigation
[control no., red, top-right: 08204583]           <- NOT extracted, see below

NBI ID NO: HGUR87H38D-U47204A873    VALID UNTIL: December 30, 2022
FAMILY NAME: DELA CRUZ              FIRST NAME: JUAN
MIDDLE NAME: SANTOS
ADDRESS: ...                        PLACE OF BIRTH: MANILA        <- not extracted
DATE OF BIRTH: MAY 31, 1984         CIVIL STATUS: MARRIED          <- not extracted
CITIZENSHIP: FILIPINO               GENDER: MALE                   <- not extracted
PURPOSE: MULTI-PURPOSE CLEARANCE
REMARKS: NO DEROGATORY

[photo] [signature] [QR + fingerprint] [barcode]   <- not extracted
Date Printed: Tuesday, January 3, 2021 04:20 PM
```

Three changes this forces, in order of how much they matter:

1. **`remarks` is a new required field**, and the most important one — it's the actual clearance result. See §1.4/§2.5/§2.7 for how it's checked (folded into Rule 3, not a new rule).
2. **There is no printed "date of issue."** The document gives `valid_until` (its own expiry) and `date_printed`. Rule 5's logic changes from "`date_of_issue + NBI_VALIDITY_MONTHS`" to "the earlier of `valid_until` and `date_printed + NBI_VALIDITY_MONTHS`" — see §2.7.
3. **Name is three fields**, not one string. `full_name` for Rule 4 matching is now *constructed* from the three parts, not extracted directly — and because `rapidfuzz.token_set_ratio` (already the chosen matcher, §2.7) is token-order-insensitive, the exact concatenation order doesn't affect the match quality.

**Deliberately still not extracted**, even though printed: address, place of birth, citizenship, civil status, gender, the control number, the photo/signature/QR/barcode images. None of these feed any validation rule — extracting them would be PII collection beyond what's actually checked, contradicting the minimization stance in §1.7.

**One field reopened since the first pass: `date_of_birth`.** It was on the original "deliberately not extracted" list — nothing consumed it. Something does now: §1.4a's cross-document consistency check (NBI ↔ government ID) needs a field both document types actually carry, and name-only matching is weaker than name+DOB. This is the bar every field here has to clear — added because a concrete rule needs it, not by default.

**Follow-up this forces, not done in this pass:** `scripts/make_onboarding_docs.py` (Phase 1, already built) renders the old 4-field layout and its `.expected.json` files use the old field names, and doesn't generate government-ID mocks at all yet. It needs a matching revision — new layout fields, renamed dates, `remarks`, `date_of_birth`, plus the three new ID sub-layouts (§1.4a) — **before** Phase 2's quality-gate calibration and Phase 3's registry cross-check run against it, or every downstream number is calibrated against a ground truth that no longer matches the schema.

```python
# --- CV/OCR document verification (Component 14) ---

class DocType(str, Enum):
    NBI_CLEARANCE = "nbi_clearance"
    GOVERNMENT_ID = "government_id"          # National ID / Driver's License / Passport — see IdType
    UNKNOWN_DOCUMENT = "unknown_document"    # a document, but not one we handle
    NOT_A_DOCUMENT = "not_a_document"        # blank page, random photo

class IdType(str, Enum):
    """Sub-classification within DocType.GOVERNMENT_ID (§1.4a) — one doc_type,
    three real-world layouts, discriminated here rather than as three
    separate DocType members."""
    NATIONAL_ID = "national_id"
    DRIVERS_LICENSE = "drivers_license"
    PASSPORT = "passport"
    OTHER = "other"                          # a government ID, but not one of the three above

class QualityVerdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    REJECT = "reject"

class ImageQualityReport(BaseModel):
    """NOT a Gemini response_schema — produced by OpenCV in src/ocr/quality.py.
    Raw signals and the derived verdict are kept separate for the same reason
    LLMJudgeVerdict separates detection from policy: the thresholds live in
    config and are testable without an image."""
    verdict: QualityVerdict
    blur_score: float
    exposure_clip: float
    skew_deg: float
    min_dim_px: int
    quad_found: bool
    normalized_quality: float = Field(ge=0.0, le=1.0)  # feeds composite confidence
    reasons: list[str] = Field(default_factory=list)

class ExtractedField(BaseModel):
    """verbatim_text is what the model literally saw, kept for auditability —
    it's how a human reviewer checks a normalization without re-opening the image."""
    value: str | None = Field(default=None)
    verbatim_text: str = Field(default="")
    model_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class NbiExtractionResult(BaseModel):
    """Gemini multimodal response_schema for DocType.NBI_CLEARANCE. DETECTION
    ONLY — carries no accept/reject decision; that lives in
    doc_validation.validate_document().

    Fields match the printed NBI Clearance layout (§2.3 revision note), not
    an assumed generic ID: separate name parts (PH forms print family/first/
    middle separately, and rapidfuzz's token_set_ratio in Rule 4 doesn't
    care about concatenation order), the document's own printed expiry
    (valid_until) plus its print date (date_printed) rather than a single
    invented "date_of_issue", remarks — the actual clearance result, checked
    under Rule 3 — and date_of_birth, added specifically to support the
    NBI<->government-ID cross-document check (§1.4a/§2.7), not by default."""
    doc_type: DocType
    family_name: ExtractedField
    first_name: ExtractedField
    middle_name: ExtractedField       # optional in practice; not every legal name has one
    date_of_birth: ExtractedField     # ISO-8601 in .value; added for cross-document matching (§1.4a)
    reference_no: ExtractedField      # "NBI ID NO" on the printed form — NOT the separate control number
    date_printed: ExtractedField      # ISO-8601 in .value, raw print in .verbatim_text
    valid_until: ExtractedField       # ISO-8601 in .value; the document's OWN printed expiry
    purpose: ExtractedField
    remarks: ExtractedField           # e.g. "NO DEROGATORY" — checked against an allowlist, Rule 3
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = Field(default="", description="One PII-free sentence on anything unusual")

    def full_name_display(self) -> str:
        """Constructs a single matchable string for Rule 4 / cross-document
        matching — not stored, computed on demand from the three extracted
        parts. Order is cosmetic: rapidfuzz.token_set_ratio ignores token
        order entirely."""
        if not self.family_name.value:
            return ""
        given = " ".join(p for p in (self.first_name.value, self.middle_name.value) if p)
        return f"{self.family_name.value}, {given}".rstrip(", ")


class IdExtractionResult(BaseModel):
    """Gemini multimodal response_schema for DocType.GOVERNMENT_ID (§1.4a).
    One schema covers all three layouts (National ID / Driver's License /
    Passport) — id_type records which, and fields that layout doesn't print
    are left null with 0 confidence rather than guessed (same rule as every
    other ExtractedField in this codebase). DETECTION ONLY, same as
    NbiExtractionResult — doc_validation.validate_id_document() decides."""
    doc_type: DocType                 # always GOVERNMENT_ID when this schema is used
    id_type: IdType
    family_name: ExtractedField
    first_name: ExtractedField
    middle_name: ExtractedField       # optional; passport commonly omits, license concatenates (§2.6)
    date_of_birth: ExtractedField     # ISO-8601 in .value — present on all three layouts
    id_number: ExtractedField         # PSN/PCN, License No., or Passport No., depending on id_type
    issue_date: ExtractedField        # optional/low-confidence on layouts that print it in small text
    expiry_date: ExtractedField       # optional — National ID commonly has none; see doc_validation.py
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = Field(default="", description="One PII-free sentence on anything unusual")

    def full_name_display(self) -> str:
        """Same construction as NbiExtractionResult.full_name_display() —
        kept as a duplicate method rather than a shared base class for now;
        revisit if a third document type makes the duplication annoying."""
        if not self.family_name.value:
            return ""
        given = " ".join(p for p in (self.first_name.value, self.middle_name.value) if p)
        return f"{self.family_name.value}, {given}".rstrip(", ")


class ValidationOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"

class RuleResult(BaseModel):
    rule: str          # type_match|completeness|format|identity|validity_window|fail_safe|cross_document_consistency
    passed: bool
    detail: str = Field(default="", description="PII-free; never echoes a field value")

class ValidationResult(BaseModel):
    outcome: ValidationOutcome
    rules: list[RuleResult] = Field(default_factory=list)
    composite_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = Field(default="", description="User-facing, PII-free")

class DocStatus(str, Enum):
    MISSING = "missing"
    SUBMITTED = "submitted"
    VALIDATED = "validated"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"

class OnboardingDocument(BaseModel):
    employee_id: str
    doc_type: DocType
    status: DocStatus
    outcome: ValidationOutcome | None = Field(default=None)
    validated_at: str | None = Field(default=None)
    source_hash: str = Field(default="", description="SHA-256 of image bytes; the deletion key")

class ChecklistStatus(BaseModel):
    employee_id: str
    documents: list[OnboardingDocument] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    faculty_class: str | None = Field(default=None)
```

**Shared-file coordination:** also add `DOCUMENT_UPLOAD` / `DOCUMENT_STATUS` to `Intent` (`schemas.py:14`). Keep that diff separate and small — `schemas.py` is touched by multiple owners.

## 2.4 `src/ocr/quality.py` — Layer 1

No LLM call. Split signals from policy:

```python
def load_image(data: bytes) -> np.ndarray
    """cv2.imdecode to BGR. This is also the EXIF strip point — decoding to a
    raw array discards all metadata, including geolocation (PII)."""

def _signals(image: np.ndarray) -> dict[str, float | bool]
    # blur_score   = cv2.Laplacian(gray, cv2.CV_64F).var()
    # exposure_clip= fraction of pixels at 0 or 255 via cv2.calcHist
    # skew_deg     = cv2.minAreaRect over the largest text contour
    # min_dim_px   = min(h, w)
    # quad_found   = 4-point contour approx over the largest external contour

def _verdict(signals) -> tuple[QualityVerdict, list[str], float]
    """Applies config thresholds. normalized_quality = min of per-signal
    0..1 scores — same conservative operator as composite confidence."""

def assess(image: np.ndarray) -> ImageQualityReport

def preprocess(image: np.ndarray) -> np.ndarray
    """Deskew, perspective-warp to the detected quad, CLAHE contrast,
    upscale to MIN_IMAGE_DIM_PX. On/off is the headline ablation (§2.9)."""
```

**Threshold calibration procedure** (this closes the TASKS.md open blocker):
run `assess()` over all 50 mock images, dump raw signals to CSV, and set each floor at the separation point between the `clean` variants and the `blur`/`lowres_jpeg` variants. Thresholds picked from data, not vibes — and the CSV is a defensible artifact at Q&A. **This requires the dataset to exist first**, which is why §3 puts the generator before everything else.

## 2.5 `src/ocr/doctypes.py` — the registry

Declarative, one entry per doc type. Frozen dataclasses (not Pydantic — this is static config, not a wire format).

```python
@dataclass(frozen=True)
class FieldSpec:
    name: str
    required: bool
    validator: Callable[[str], bool]
    prompt_hint: str          # goes into the extraction prompt

@dataclass(frozen=True)
class DocTypeSpec:
    doc_type: DocType
    canonical_name: str
    aliases: tuple[str, ...]
    fields: tuple[FieldSpec, ...]
    validity_months: int | None

REGISTRY: dict[DocType, DocTypeSpec] = {...}
def get_spec(doc_type: DocType) -> DocTypeSpec | None
def required_fields(doc_type: DocType) -> tuple[str, ...]
```

NBI field spec — 8 required + 1 optional, revised against the real layout and the cross-document requirement (§2.3):

| Field | Required | Validator (Rule 3) |
| --- | --- | --- |
| `family_name` | yes | ≥1 alpha token (allows multi-word compounds: `DELA CRUZ`, `DE LA CRUZ`, `SAN JUAN`), no digits, allows `ñ` |
| `first_name` | yes | same shape as `family_name` |
| `middle_name` | **no** | allowed empty — not every legal name records one; if present, same shape as above |
| `reference_no` | yes | the printed "NBI ID NO", e.g. `HGUR87H38D-U47204A873` — two alphanumeric groups joined by a dash. **From one sample only** — see the open decision in §5; do not present this as confirmed-authoritative until checked against more real clearances |
| `date_of_birth` | yes | parses to a real date, not in the future. Added specifically to support cross-document matching against the government ID (§1.4a/§2.7) |
| `date_printed` | yes | parses to a real date(time), not in the future |
| `valid_until` | yes | parses to a real date; sanity-checked against `date_printed` (should be later) as a Rule 3 structural check, not a model judgment |
| `purpose` | yes | non-empty; unrecognized values warn, never fail |
| `remarks` | yes | normalized value must match a config allowlist of clean-status strings (`NBI_CLEAN_REMARKS` — e.g. `"NO DEROGATORY"`, `"NO DEROGATORY RECORD"`, `"NO RECORD"`; deliberately incomplete and tunable). **Anything else fails Rule 3 outright** — an unrecognized remarks value is not assumed clean |

**Deliberately absent from this registry entirely, not even as optional fields:** the separate control number (top-right, red — a different identifier than "NBI ID NO", purpose unconfirmed), address, place of birth, citizenship, civil status, gender. See the PII table in §1.7 for why.

### Government ID field spec (§1.4a, new) — one `DocTypeSpec`, per-`id_type` field table

`GOVERNMENT_ID`'s `DocTypeSpec.fields` covers all three layouts at once; `required` and the validator both branch on `IdType` internally rather than the registry growing three separate entries (that would defeat the point of unifying them under one `DocType`, §1.4a):

| Field | Required | Validator (Rule 3) |
| --- | --- | --- |
| `id_type` | yes | must be a valid `IdType` member; the model's own classification, not independently verified |
| `family_name` | yes | same shape as NBI's — alpha tokens, compounds allowed, `ñ` allowed |
| `first_name` | yes | same shape |
| `middle_name` | **no** | passport commonly omits it entirely; allowed empty |
| `date_of_birth` | yes | parses to a real date, not in the future |
| `id_number` | yes | **pattern depends on `id_type`**, each from exactly one sample, none confirmed (Part 5): National ID `^\d{4}-\d{4}-\d{4}-\d{4}$` (`1234-5678-9101-1213`); Driver's License `^[A-Z]\d{2}-\d{2}-\d{6}$` (`N03-12-123456`); Passport `^[A-Z]\d{7}[A-Z]$` (`P0000000A`) — or, better, cross-check against the passport's MRZ line directly if extracted (§2.6), since MRZ passport numbers are positionally fixed and more reliable than the visual header |
| `issue_date` | **no** | if present, real date, not in the future |
| `expiry_date` | **required for `drivers_license`/`passport`; not required for `national_id`** | if present, real date; for `drivers_license`/`passport`, sanity-checked after `issue_date` when both are present |

**Deliberately absent, even though printed on one or more of the three layouts:** address, blood type, marital status, place of birth, sex, height/weight/eye color (license), restrictions (license), the biometric QR/fingerprint image, the passport's non-MRZ visual fields once the MRZ has been read. Same minimization stance as NBI's list above.

Adding BIR 2316 (or a fourth ID sub-type) later is then a registry entry, not code — which is the "generalizable" claim, made concrete, twice now.

## 2.6 `src/ocr/extractor.py` — Layer 2

The canonical structured-Gemini pattern, identical in shape to `src/guardrails/llm_judge.py:66-85`, `src/rag/answerer.py:63-74`, `src/agent/router.py:29-47`.

```python
def extract_document(
    image_bytes: bytes,
    mime_type: str,
    expected_doc_type: DocType = DocType.NBI_CLEARANCE,
    client=None,
    session_id: str | None = None,
    use_cache: bool = True,
    preprocess: bool | None = None,
) -> tuple[NbiExtractionResult | IdExtractionResult | None, ImageQualityReport]:
```

`expected_doc_type` now **dispatches which prompt and which `response_schema` get used** (§1.4a) — this is the one real branch this function grows for the second document type:

```python
_SCHEMA_BY_DOC_TYPE = {
    DocType.NBI_CLEARANCE: (prompts.NBI_EXTRACTION_PROMPT, NbiExtractionResult),
    DocType.GOVERNMENT_ID: (prompts.ID_EXTRACTION_PROMPT, IdExtractionResult),
}
```

Flow (unchanged in shape from the NBI-only version — this is the same function, generalized):
1. `quality.load_image` → `quality.assess`
2. **`verdict is REJECT` → return `(None, report)` immediately.** No API call. This is the quota mechanism, and it applies identically regardless of which vision provider or document type is active.
3. `quality.preprocess()` if enabled
4. Cache key = `sha256(original bytes) + preprocess_flag + config.ACTIVE_VISION_MODEL`, JSON at `config.OCR_CACHE_DIR/<key>.json`. Hit → deserialize, zero calls. **Provider is part of the key** — a Gemini-produced cache entry never masquerades as an Ollama result or vice versa, which matters for the eval (`run_ocr_eval.py` can report accuracy per provider without cache cross-contamination). `expected_doc_type` doesn't need to be in the key — the same image bytes are always the same document, so there's nothing to disambiguate.
5. One call, provider-agnostic at the call site, prompt/schema selected by `expected_doc_type`:
   ```python
   prompt, schema = _SCHEMA_BY_DOC_TYPE[expected_doc_type]
   client = client or config.get_vision_client()
   response = client.models.generate_content(
       model=config.ACTIVE_VISION_MODEL,
       contents=[
           types.Part.from_bytes(data=processed_bytes, mime_type=mime_type),
           prompt.format(fields=...),
       ],
       config=types.GenerateContentConfig(
           response_mime_type="application/json",
           response_schema=schema,
           temperature=0.0,
       ),
   )
   ```
6. `usage.record_usage(config.ACTIVE_VISION_MODEL, usage.extract_usage(response), session_id=...)` — recorded **under the active vision model name**, so `GET /usage` breaks vision spend out from chat spend, and out from *which vision provider* if both get exercised during eval work.
7. `except (APIError, LLMBackendError)` → `logger.warning(...)`, return `(None, report)`. **Both exceptions now**, matching the pattern already used everywhere `get_llm_client()`'s output is called (`orchestrator.py:195`, `llm_judge.py:77`) — vision can now come from either backend, so the catch clause must too.
8. `response.parsed is None` → return `(None, report)`. Fails toward `needs_review` downstream, never toward accept — this matters more on the Ollama path, where a small local model is more likely to miss strict schema conformance.

**`NBI_EXTRACTION_PROMPT`** (`src/agent/prompts.py`, UPPER_SNAKE triple-quoted with `{placeholders}`, matching `ROUTER_PROMPT:10`). Must instruct: return `not_a_document`/`unknown_document` honestly rather than guessing; put ISO-8601 in `value` and the literal print in `verbatim_text`; never fabricate an unreadable field — leave `value` null with low confidence; extract exactly the 9 fields in §2.3/§2.5 (`family_name`/`first_name`/`middle_name`/`date_of_birth`/`reference_no`/`date_printed`/`valid_until`/`purpose`/`remarks`) and nothing else — not address/citizenship/civil status/gender even though they're on the document, and not the separate control number.

Two NBI-specific prompt notes worth stating explicitly rather than discovering during calibration: **`date_printed` is small print** near the QR/barcode block on a real layout, not a headline field like `valid_until` — expect lower `model_confidence` on it than on the other fields, and don't be surprised if Rule 6's fail-safe fires on this field specifically more than others. **`remarks` must be transcribed verbatim**, not paraphrased or summarized by the model — the format check in §2.5 needs the literal printed string to match against `NBI_CLEAN_REMARKS`, so the prompt should say "quote the remarks field exactly as printed."

**`ID_EXTRACTION_PROMPT`** (new, same file/style). Must instruct: first classify `id_type` (national_id/drivers_license/passport/other) from the layout, then extract only the 8 fields in §1.4a/§2.5 for whichever layout it is, leaving fields that layout simply doesn't print as `null` rather than guessed (National ID's `expiry_date`, most commonly). Two notes specific to this prompt:
- **If the document is a passport, prioritize the MRZ** (the two fixed-width lines at the bottom, `P<PHL...`) over the visual header fields for `family_name`/`first_name`/`id_number`/`date_of_birth`/`expiry_date` — MRZ uses a standardized OCR-B font specifically designed for machine reading, and is measurably more reliable than the stylized header text on the same document. The prompt should say this explicitly, not assume the model already prioritizes it.
- **Driver's License prints name as one concatenated line** (`Last, First Middle`), unlike National ID/Passport which label the three parts separately — the prompt must instruct the model to still *split* it into `family_name`/`first_name`/`middle_name`, not return the concatenated string in `family_name` alone.

## 2.6a Gemini main, Ollama fallback — and other free options considered

**Gemini stays the default and the evaluated path** — `VISION_PROVIDER=gemini` out of the box, and every threshold, cache entry, and headline eval number in this doc assumes it. Gemini 2.5 Flash is already free at this project's volume (50 mock images ≈ ₱10 in tokens; disk-cached re-runs are free), so the real constraint is the **20-requests/day rate cap**, not money.

**Ollama is a real, switchable fallback, not a hypothetical** — set `VISION_PROVIDER=ollama` and every call in `extractor.py` routes through the same `OllamaClient` adapter already serving chat (`src/agent/llm_client.py`). One piece of new work this requires, called out explicitly rather than assumed away:

- `OllamaClient._contents_to_messages()` (`llm_client.py:146-188`) currently only handles text/function parts. It needs an image branch: when a `types.Content`'s parts include a `Part.from_bytes(...)`, base64-encode the data and attach it as `message["images"] = [b64_string]` — Ollama's `/api/chat` wire format takes a per-message `images` list of base64 strings for multimodal models (`llama3.2-vision`, `qwen2.5vl`, `minicpm-v`, `moondream` all support this). Everything else in the adapter (`response_schema` → `format`, `.parsed`, usage extraction) already works unchanged, because `extractor.py` calls the same `generate_content(model, contents, config)` contract as every other structured call in this codebase.
- Add this as its own build-order step (Part 3, step 4a below) rather than folding it into `extractor.py`'s first pass — it's infra work on a shared file (`llm_client.py`), not CV-specific logic, and should land as a reviewable diff on its own.
- **Expect it to be the weak link, not a broken one.** Small local vision models are less reliable at strict `response_schema` conformance than Gemini (same limitation PLAN.md §2.1 already documents for chat) — more `response.parsed is None` fallouts, which the pipeline already fails toward `needs_review` for (§1.4). It should never crash; it will produce noisier verdicts. That's a legitimate thing to report on the deck as a provider ablation, not just a fallback: run `run_ocr_eval.py` once per provider and show the accuracy delta, the same honesty move as the real-vs-mock and preprocess-on/off comparisons already in the plan.

**Other free/non-Gemini options considered and set aside**, for completeness on the RRL slide:

| Option | Cost | Setup | Fit for this project |
| --- | --- | --- | --- |
| **Tesseract / `pytesseract`** | Free, unlimited, local | `pip install pytesseract` + a system Tesseract binary | Already evaluated and rejected in PLAN.md §5's RRL table: raw text only, no structure, weakest on skewed/low-quality phone photos — exactly the failure mode the mock `skew`/`blur`/`glare` variants and the real subset target. Needs a second field-parsing layer (regex over raw text) on top, undoing the `response_schema` pattern used everywhere else in this codebase. |
| **EasyOCR** (PyTorch) | Free, unlimited, local | `pip install easyocr`; first run downloads model weights (~100MB+) | Better than Tesseract on skew/low-res, still raw text, still needs a field-parsing layer. Slower on CPU-only machines. |
| **PaddleOCR / PP-Structure** | Free, unlimited, local | Heavier install (PaddlePaddle framework); PP-StructureV2 does key-value extraction, closer to what's needed | Most capable of the local-OCR options for structured ID-style documents, but the heaviest to stand up for a component that's explicitly secondary (CLAUDE.md: "if effort has to be cut, cut the CV stretch goals"). |
| **Google Cloud Vision OCR** | Free up to 1,000 units/month | New GCP project + service-account credential | Already evaluated and rejected in PLAN.md §5: a second Google credential, and still needs a field-parsing step since it returns raw text + bounding boxes, not typed fields. |
| **Azure AI Document Intelligence / AWS Textract** | Free tiers (500 pages/mo Azure; 1,000 pages for 3 months AWS) | New cloud account + credential each | Same objection as Cloud Vision, plus a third cloud provider in a project that otherwise touches only Google + Tavily. |

None of these reuse existing infrastructure or the `response_schema` pattern the way the Ollama path does — that's why Ollama is the one implemented fallback rather than just documented as an option.

## 2.7 `src/guardrails/doc_validation.py` — Layer 3

Two single-document validators, same six-rule shape, plus one cross-document check that only makes sense once both exist. All three are public `validate_*` functions returning a Pydantic model, per `src/guardrails/` conventions; user-facing copy as module constants; empty `__init__.py`.

### `validate_document()` — NBI Clearance

```python
def validate_document(
    extracted: NbiExtractionResult | None,
    quality: ImageQualityReport,
    faculty_record: dict,          # {"full_name": ..., "date_of_birth": ..., "employee_id": ...}
    expected_doc_type: DocType = DocType.NBI_CLEARANCE,
    as_of: date | None = None,     # injectable so expiry tests don't rot
) -> ValidationResult
```

Six private `_rule_*(...) -> RuleResult` functions, then a deterministic outcome table implementing §1.4.

**Rule 3 (format)** also checks `remarks` against `config.NBI_CLEAN_REMARKS` (§2.5) — a normalized-string membership test, not a model judgment. This is the check that makes a document with an actual derogatory-record hit fail format outright (composite forced to 0, per §1.3) rather than sail through on a good-looking name and a valid date. It also sanity-checks `valid_until` falls after `date_printed`.

**Rule 4** matches against `extracted.full_name_display()` using `rapidfuzz.fuzz.token_set_ratio` against `config.NAME_MATCH_THRESHOLD`, after normalizing: case-fold, strip punctuation, drop `Jr./Sr./II/III`, NFKD-fold `ñ`, sort tokens so surname-first vs given-name-first both match. `token_set_ratio` is order-insensitive, so exact concatenation order doesn't change match quality.

**Rule 5 (validity window)**:
```python
effective_deadline = min(
    parse(extracted.valid_until.value),                 # the document's own printed expiry
    parse(extracted.date_printed.value) + relativedelta(months=config.NBI_VALIDITY_MONTHS),
)
outside_window = as_of > effective_deadline
```
`min()`, not just the employer's 6-month math alone — same conservative operator as composite confidence (§1.3): whichever deadline is stricter governs. `as_of` injection is what keeps this test deterministic.

### `validate_id_document()` — Government ID (§1.4a, new)

Same six-rule shape, deliberately, as a consistency argument rather than inventing a new pattern:

```python
def validate_id_document(
    extracted: IdExtractionResult | None,
    quality: ImageQualityReport,
    faculty_record: dict,
    expected_doc_type: DocType = DocType.GOVERNMENT_ID,
    as_of: date | None = None,
) -> ValidationResult
```

- **Rule 1 (type match)** — `doc_type == GOVERNMENT_ID`; `id_type` itself is never wrong/right, any of the four values is a valid government ID submission.
- **Rule 2 (completeness)** — required fields per `id_type` (§2.5's field table) — `expiry_date` is required for `drivers_license`/`passport`, explicitly **not required** for `national_id`.
- **Rule 3 (format)** — `id_number` validated against the pattern for its `id_type` (§2.5); dates parse and aren't nonsensical.
- **Rule 4 (identity vs. employee record)** — same `rapidfuzz`/`full_name_display()` pattern as NBI's Rule 4, against the same `faculty_record`. Never auto-reject, same reasoning as §1.4a.
- **Rule 5 (validity window)** — `as_of > extracted.expiry_date.value` → `rejected`, **when `expiry_date` is present**. When absent (National ID, by design) this rule is **skipped, not failed** — a `RuleResult(rule="validity_window", passed=True, detail="no printed expiry for this id_type")`, not a missing-field error. Absence of an expiry field on this specific `id_type` is correct, not a gap.
- **Rule 6 (fail-safe)** — same as NBI: `extracted is None` or a `warn` quality verdict with low composite → `needs_review`.

### `validate_cross_document()` — NBI ↔ ID consistency (§1.4a, new)

```python
def validate_cross_document(
    nbi: NbiExtractionResult,
    id_doc: IdExtractionResult,
) -> RuleResult
```

Compares `full_name_display()` (`rapidfuzz.token_set_ratio` ≥ `config.NAME_MATCH_THRESHOLD`, same as every other identity check in this file) and `date_of_birth.value` (exact string equality — DOB isn't a fuzzy field, two documents either agree on a birthdate or they don't) between the two already-in-memory extraction results. Returns one `RuleResult(rule="cross_document_consistency", ...)`, `detail` stating only which sub-check failed (`"name"`, `"date_of_birth"`, or both) — never the values themselves.

**Where the two extraction results come from at call time.** This function takes two already-extracted objects — it doesn't do any lookup itself. The caller (inside `POST /upload-doc`, §2.8) is responsible for getting there: when document B is uploaded for an employee who already has document A on file, look up A's `source_hash` from the `onboarding_documents` row, and call `extract_document()` again with A's cached bytes — which is a **cache hit** (§2.6 step 4), so this costs zero new API calls and never touches new image bytes. This reuses exactly the mechanism already built for the `extract_document(source_hash)` agent tool (§1.6) — cross-document validation and "explain why this needs review" are the same underlying operation, called from two different places.

**When there's nothing to compare yet** (only one of the two documents has been uploaded so far), this rule is simply **omitted** from the `rules` list entirely — not included as a `passed=True` placeholder, not failed. It runs (and can flip an already-`accepted` single-document outcome to `needs_review`) automatically the moment the second document arrives, since every upload re-checks for a sibling.

**Every `RuleResult.detail` across all three functions is PII-free**: say `"required field missing: valid_until"` or `"cross-document name mismatch"`, never the value.

## 2.8 State, interfaces, ops

**`src/memory/onboarding_status.py`** — reuse the `_get_connection()` pattern from `src/memory/session.py:18-36` verbatim (`sqlite3.Row`, inline `CREATE TABLE IF NOT EXISTS` on every connect, no migrations, `try/finally: conn.close()` with explicit `commit()`).

```sql
CREATE TABLE IF NOT EXISTS onboarding_documents(
  employee_id TEXT NOT NULL, doc_type TEXT NOT NULL, status TEXT NOT NULL,
  outcome TEXT, validated_at TEXT, source_hash TEXT,
  PRIMARY KEY (employee_id, doc_type));
CREATE TABLE IF NOT EXISTS onboarding_profile(
  employee_id TEXT PRIMARY KEY, faculty_class TEXT);
```

`record_result()`, `get_status() -> ChecklistStatus`, `set_faculty_class()`, `get_faculty_class()`. `source_hash` is the deletion key — withdrawing consent means deleting image, ground truth, and the cache entry, all findable by that one hash.

**`src/agent/orchestrator.py`** — three additions, matching existing shape:
- `_function_declarations()` (`orchestrator.py:300`): add `extract_document(source_hash)`, `validate_checklist(employee_id)`, `get_onboarding_status(employee_id)` as `types.FunctionDeclaration`s.
- `_execute_tool()` (`orchestrator.py:338`): three more branches in the if-chain, each returning a plain JSON-able dict; unknown names already fall through to `{"error": ...}`.
- `_RunState` (`orchestrator.py:98`) + `AgentResponse` (`:87`): a `validation: ValidationResult | None` field so the outcome reaches the API layer.

**`src/api.py`** — two endpoints, matching the file's conventions (request/response models local to `api.py`, explicit `response_model=`, `Literal[...]` for closed enums, no `HTTPException` — degrade gracefully instead, per `_try_agent_orchestrator:121-145`):
- `POST /upload-doc` — enforce `MAX_UPLOAD_BYTES`; **validate MIME by magic bytes, not extension**; EXIF stripped by the `cv2.imdecode` in `quality.load_image`; don't persist raw bytes unless `PERSIST_UPLOADS`; run quality → extract (`validate_document` or `validate_id_document`, dispatched by `doc_type`) → record. **Then, new:** check `onboarding_documents` for a sibling row (the other of `nbi_clearance`/`government_id`) belonging to the same `employee_id`; if found, load it via cache-hit `extract_document(sibling_source_hash)` and run `validate_cross_document()` (§2.7), appending its `RuleResult` into whichever document's `ValidationResult` is being returned this request and re-evaluating the overall outcome (a prior `accepted` on either document can flip to `needs_review` here). Return `ValidationResult` + `ChecklistStatus`.
- `GET /onboarding-status/{employee_id}`.

**`faculty_record`** (passed into both `validate_document()` and `validate_id_document()`) now needs `date_of_birth` alongside `full_name` — wherever this dict is sourced from (HR data, `identities.json` in the mock dataset) must carry it, or Rule 4's identity check silently has no DOB to fall back on for the cross-document leg.

**`src/monitoring.py`** — extend both fail-closed frozensets (`monitoring.py:22-45`) and add a `doc_trace(session_id, doc_type)` contextmanager sibling to `chat_trace` (the existing one takes a `message` and is chat-shaped).
- tags: `doc_type`, `validation_outcome`, `quality_verdict`, `faculty_class`, `topic`
- metrics: `blur_score`, `skew_deg`, `extraction_confidence`, `fields_extracted`, `fields_missing`, `ocr_latency_ms`

**No extracted field value is ever a tag.** The allowlist is fail-closed by construction, so the risk is only in what you add to it — none of the keys above can carry a name or ID number.

**`src/ui.py`** — `st.file_uploader` in the composer, checklist panel in the sidebar, a new key on the message dict (`ui.py:457-467`) handled in `_render_pills_and_panels`. UI talks only to FastAPI (`ui.py:472-509` is the only network path) — never Gemini.

## 2.9 Dataset + evals

**`scripts/make_onboarding_docs.py`** — Pillow-rendered layouts for **both** document types now, matching §2.3/§1.4a's field sets:

- **NBI** (family/first/middle name, DOB, reference no., date printed, valid until, purpose, remarks). 8 identities × 5 degradation variants (`clean`, `skew` 8–15° warp, `blur` gaussian k=7–11, `glare` radial overlay + brightness clip, `lowres_jpeg` 0.35 downscale q=35) = 40, plus 10 negatives (non-NBI doc, non-document photo, blank page, wrong person, expired).
- **Government ID** (new) — same 8 identities, each rendered as **one** of the three `id_type` layouts (not all three per identity — keeps the dataset size sane; distribute roughly evenly, e.g. identities 1–3 as `national_id`, 4–6 as `drivers_license`, 7–8 as `passport`), same 5 degradation variants = 40 more images, plus negatives covering the ID-specific failure modes: **cross-document name mismatch** (an ID whose name doesn't match its paired NBI — the fixture that exercises `validate_cross_document()`), **cross-document DOB mismatch**, expired ID (license/passport only, National ID has no expiry to violate), wrong person vs. employee record.
- Both document types' `.expected.json` files for one identity share the same `employee_id` and the same underlying name/DOB, so a "clean pair" identity round-trips through Rule 4 (both documents) *and* the cross-document check without manufactured disagreement — that agreement is itself a fixture worth asserting on, not just the mismatch cases.

`identities.json` gains `date_of_birth` (needed now by `faculty_record`, §2.8) and doubles as ground truth for both document types. Seed with `random.Random(SEED)` so regeneration is reproducible, same as before. **Filename convention, now that two document types share one output directory:** `nbi_<identity_id>_<variant>.png` / `id_<identity_id>_<variant>.png` (e.g. `nbi_id01_clean.png`, `id_id01_clean.png`), replacing the original bare `<identity_id>_<variant>.png` — the doc-type prefix is what keeps `ls data/references/mock/` scannable and keeps eval scripts' `--subset`/glob logic from needing to open every file to know which validator to run.

**`evals/run_ocr_eval.py`** — per-field exact match + normalized CER, **reported separately per `doc_type`** (never NBI and government-ID accuracy pooled into one number, same discipline as real-vs-mock, §1.8) and per-field within each, so `remarks`/`reference_no`/`id_number` — the fields with the least real-sample confidence (Part 5) — don't hide inside an average. **`--subset real|mock`**, still never pooled. `--no-preprocess` for the ablation. Cache-first by default, `--no-cache` forces live calls. Follow `evals/run_guardrail_eval.py` structure: `sys.path` bootstrap, module path constants, `argparse` with quota-costing flags off by default, rates printed as `{x:.1%}`, results JSON stamped `%Y%m%dT%H%M%SZ`, optional MLflow block.

**`evals/run_validation_eval.py`** — precision/recall/F1 on `needs_review` vs `accepted`, per-rule trigger counts, **headline metric: false auto-pass rate**, all **per `doc_type`**, plus a dedicated section for `cross_document_consistency` (trigger rate, and specifically whether it ever fires on the "clean pair" fixtures — it shouldn't). Runs entirely off cached extractions → zero API cost, re-runnable freely.

**`tests/`** — `test_quality.py` (signals + verdict boundaries on generated variants), `test_doc_validation.py` (one test per NBI rule, one per ID rule, plus `test_validate_cross_document` covering match/name-mismatch/DOB-mismatch/not-yet-applicable), `test_extractor.py` (mock the client, assert the reject short-circuit spends no call, assert cache hit spends no call, for both `expected_doc_type` values), `test_onboarding_status.py` (`tmp_path` + `monkeypatch.setattr(config, "SQLITE_PATH", ...)`, per `tests/test_api.py:76`). Plain pytest functions, no classes.

---

# Part 3 — Build order, phased with acceptance criteria

Dataset first. Every threshold, every test fixture, and every eval depends on it, and building it first is what turns the "thresholds not chosen" open question (PLAN.md §4.2/§4.3) into a calibration step.

Each phase below states what "done" means as checkable criteria — a count, a command, an exact expected value — not a vibe. Don't move to the next phase until the current one's criteria all pass. Real-subset consent collection has human latency — **start it during Phase 0**, in parallel, not at Phase 7.

### Phase 0 — Foundations ✅ DONE (verified 2026-08-08)

**Goal:** every shared file has the CV additions in place, nothing else changed.
**Files:** `requirements.txt`, `requirements-api.txt`, `src/config.py`, `src/schemas.py`, `.env.example`, `src/ocr/__init__.py` (new, empty).
**Depends on:** nothing.

**Completeness-check finding, since fixed:** Phase 0 was originally executed against an *earlier* version of this spec, before the NBI real-sample revision (remarks/DOB/date_printed+valid_until) and before §1.4a's government-ID expansion. `src/schemas.py` still had the single old `ExtractionResult` (4 fields), and `src/config.py` was missing `NBI_CLEAN_REMARKS`/`ID_NUMBER_PATTERNS` and only listed one required document — real code had silently drifted from the plan doc across two revisions. Re-synced: `schemas.py` now has `NbiExtractionResult`/`IdExtractionResult`/`IdType`/`DocType.GOVERNMENT_ID` matching §2.3 exactly; `config.py` has `NBI_CLEAN_REMARKS`, `ID_NUMBER_PATTERNS`, `REQUIRED_ONBOARDING_DOCS = ("nbi_clearance", "government_id")`, and the renamed `REFERENCES_DIR`/`MOCK_DOCS_DIR`/`REAL_DOCS_DIR` (data/references/, this session's folder refactor — old `ONBOARDING_DOCS_DIR = data/onboarding_docs` is gone from both the plan and the code).

**Acceptance criteria:**
- [x] `pip install -r requirements.txt` completes clean, including `opencv-python-headless`, `Pillow`, `rapidfuzz`, `python-multipart`.
- [x] `python -c "from src import config; print(config.ACTIVE_VISION_MODEL)"` prints `gemini-2.5-flash`.
- [x] `python -c "from src import schemas; schemas.NbiExtractionResult; schemas.IdExtractionResult; schemas.IdType; schemas.ValidationResult; schemas.ChecklistStatus"` imports without error.
- [x] `.env.example` lists `VISION_PROVIDER`, `GEMINI_VISION_MODEL`, `OLLAMA_VISION_MODEL`.
- [x] `pytest tests/` is still fully green — **146 passed, 1 pre-existing failure** (`test_entire_raw_corpus_chunks_cleanly`, a `pdfminer` "Unexpected EOF" on a raw PDF; confirmed via `git stash` to fail identically with every CV change removed — unrelated, not this component's scope, flagged separately to the RAG owner).

**Verify:**
```bash
pip install -r requirements.txt
python -c "from src import config; assert config.ACTIVE_VISION_MODEL == 'gemini-2.5-flash'"
pytest tests/
```

---

### Phase 1 — Mock dataset ✅ DONE (verified 2026-08-08)

**Goal:** the dataset every later phase calibrates and tests against exists and is reproducible.
**Files:** `scripts/make_onboarding_docs.py`, `data/references/mock/`.
**Depends on:** Phase 0.

Fully reworked (not patched) to the current §2.3/§1.4a field sets — both document types, all three `id_type` layouts, cross-document negative fixtures. Output relocated to `data/references/mock/` as part of this session's folder refactor (specimen/demo layout-reference images now live separately, in `data/references/samples/`, gitignored — Part 5).

**Acceptance criteria:**
- [x] Exactly 50 NBI images (8 identities × 5 degradation variants + 10 negatives).
- [x] Exactly 50 government-ID images (8 identities × 5 degradation variants — distributed 3×`national_id`/3×`drivers_license`/2×`passport` — + 10 negatives: 2 each of `non_id_document`, `blank_page`, `cross_name_mismatch`, `cross_dob_mismatch`, `expired_id`).
- [x] Each identity has one NBI `.expected.json` and one ID `.expected.json` (16 total), sharing `employee_id`/name/DOB — verified programmatically (`all('date_of_birth' in i for i in identities)`).
- [x] All 20 negatives individually labeled — verified 9 distinct category strings across `negatives.json` (5 NBI + 5 ID, `blank_page` shared by name). `cross_name_mismatch`/`cross_dob_mismatch` fixtures carry `pairs_with: "nbi_<identity_id>"` so eval/test code knows which NBI fixture to cross-check against.
- [x] Re-running the script twice produces byte-identical output — confirmed via `diff -rq`.
- [x] `identities.json` includes `date_of_birth` and `id_type` for all 8 identities.

**Verify:**
```bash
python scripts/make_onboarding_docs.py
ls data/references/mock/*.png data/references/mock/*.jpg | wc -l   # 100
ls data/references/mock/*.expected.json | wc -l                     # 16
python scripts/make_onboarding_docs.py --out /tmp/rerun && diff -rq data/references/mock /tmp/rerun && rm -rf /tmp/rerun
```

---

### Phase 2 — Quality gate (Layer 1) ✅ DONE (verified 2026-08-08)

**Goal:** deterministic, calibrated, zero-network image quality assessment.
**Files:** `src/ocr/quality.py`, `tests/test_quality.py`, threshold values in `src/config.py`.
**Depends on:** Phase 1 (calibration needs the dataset).

**Calibration result, against the real dataset (`evals/results/quality_signals_20260808.csv`, re-run after the mock rework below):** `clean` blur_score range 900.9–2558.7, `blur` range 0.3–0.7 — a huge gap. The provisional `BLUR_VARIANCE_FLOOR=100`/`BLUR_VARIANCE_WARN=250` from Phase 0 sit safely inside it; **no threshold change was needed**, but this is now a measured fact, not a guess. `lowres_jpeg` variants correctly reject on `MIN_IMAGE_DIM_PX`. Skew variants (rendered at 8–15°, §2.9) split 11 warn / 5 reject against `MAX_SKEW_DEG=12` — expected, the generator's range straddles that threshold on purpose.

**Two bugs found and fixed during calibration, both in `_detect_document()`/`preprocess()`'s angle handling** — both are the same class of mistake (mishandling OpenCV's `minAreaRect` width/height/angle convention), caught at two different times:
1. **`preprocess()`'s warp** originally mapped output width/height from `minAreaRect` directly instead of the ordered corners' actual pixel distances, producing `skew_deg=90.0` after "fixing" a skewed image. Fixed with the standard four-point transform (`_warp_to_quad()`).
2. **`_detect_document()`'s angle normalization** branched on `rect_w < rect_h` to decide whether to add 90° — this double-corrected OpenCV's own w/h-swap convention specifically when the detected rectangle was already axis-aligned (`angle=90.0` with `w<h`, a real OpenCV return value for a perfectly clean rectangle, not a mistake in the image). Surfaced only once the mock layout below got dense enough for the frame contour's `(w, h)` ordering to flip between clean and skewed cases — **all 16 clean variants started reporting `skew_deg=90.0` and rejecting**, caught immediately by the parametrized test suite. Fixed by reducing the angle modulo 90 into `(-45, 45]` instead of branching — a rectangle's angle is only meaningful mod 90 given its four-fold rotational symmetry, so there's nothing to branch on. Re-verified against both a clean fixture (90.0° raw → 0.0° fixed) and a skewed one (77.5° raw → 12.48° fixed, matching the pre-rework measurement exactly) before landing.

**Dataset limitation found, not fixed (Layer 1 is working correctly; the mock generator's degradation is too gentle):** all 16 `glare` variants pass with `exposure_clip≈0.0001` — the radial-gradient composite in `apply_glare()` brightens a region without ever pushing pixels to pure 255, so it doesn't trip the exposure signal the way a real flash reflection would. Don't read "100% pass on glare" in Phase 7 as the quality gate being lenient — it's the synthetic degradation being too soft to test that path.

**Mock layout reworked after a real-sample review surfaced two problems (not part of Phase 2 originally, done here since it directly affects calibration validity):**
1. **Field completeness.** The original NBI mock rendered only the 9 fields `ExtractionResult` reads, on an otherwise-empty page. A real clearance is dense — full name block, address, place of birth, citizenship, civil status, gender, photo, signature, thumbprint, QR, barcode, and an Agency/CASID/O.R. No/O.R. Date/DATID/BIOID/RECID/DST PAID/INTID/PRTID metadata grid (confirmed against a real specimen to be genuine printed fields, not generator artifacts — corrected after initially misreading them as template tells). A sparse mock measures an easier extraction task than the real one, which would have confounded the real-vs-mock gap this project exists to report honestly. `render_nbi_clean()` now renders the full field set; `ExtractionResult`/ground truth are unchanged — the extra fields are visual/distractor density only, never asserted against.
2. **Orientation.** NBI Clearance, National ID, and Driver's License are all landscape in reality; the original renderer used one portrait canvas for all four layouts. Passport (already portrait) was already correct. Fixed via per-doc-type canvas sizes (`NBI_SIZE`, `NATIONAL_ID_SIZE`, `DRIVERS_LICENSE_SIZE`, `PASSPORT_SIZE`).

Both changes are rendering-only — `NbiExtractionResult`/`IdExtractionResult` and every validation rule are unaffected. Dataset regenerated: same 100 images, 16 `.expected.json`, 20 negatives, byte-identical on re-run (reproducibility re-verified).

**Resolution floor recalibrated 2026-08-09 against real evidence, after Phase 4 verification pushed on it.** `MIN_IMAGE_DIM_PX=640` was a Phase 0 placeholder, never actually checked against a real document — only against the mock dataset's synthetic gap (clean ~1000px+ vs. `lowres_jpeg` ~300-500px), which says nothing about where real legibility actually breaks down. `data/references/real/nbi-clearance-real.WEBP` (768×518, shorter side 518) rejected on this floor. Rather than accept that as correct because "640 was calibrated," bypassed the floor and ran the real extraction: **Gemini returned all 9 fields at 0.98–0.99 confidence.** The floor was wrong, not the image. Recalibrated to a real two-tier split, same shape as blur/skew: `MIN_IMAGE_DIM_PX=400` (reject floor, well below the one confirmed-working sample — margin, not a fit to n=1) / `MIN_IMAGE_DIM_WARN=640` (the old value, demoted to a soft signal). The real sample now reads `warn` (`quad_found=False` also still fires, unrelated finding from Part 5) and proceeds to extraction instead of being blocked. Mock dataset's verdict distribution unchanged after the change (all `lowres_jpeg` variants are 300-350px shorter side, still comfortably below the new 400 floor) — confirmed via a full CSV re-dump, not assumed.

**`quad_found=False` penalty recalibrated 2026-08-10, after a third real specimen surfaced a much bigger problem than the resolution floor did.** The penalty for no rectangular contour found was a hardcoded `0.5` inline in `quality.py` (`scores.append(0.5)  # not a hard fail alone, but a real confidence hit`) — like `MIN_IMAGE_DIM_PX`, never checked against real data. By the time a third real specimen (`real3.jpg`, a user-supplied genuinely-valid NBI clearance) came in, **all three real specimens collected so far had hit `quad_found=False`** (real photos never produce the clean rectangular edge the mock dataset's synthetic renders trivially do). Because `normalized_quality = min(scores)`, that hardcoded `0.5` became the dominant score every time, capping `composite_confidence` at `0.5` — below `OCR_CONFIDENCE_FLOOR=0.70` **unconditionally**, regardless of extraction quality. Concretely: `real3.jpg` extracted at 0.99 confidence, passed all five other rules (type match, completeness, format, identity, validity window) cleanly, and still landed on `needs_review` purely because of this one signal. Every real submission was structurally incapable of reaching `accepted` — not a hypothetical, an observed 3-for-3. Moved to `config.QUAD_NOT_FOUND_QUALITY_SCORE = 0.75` (named constant, not inline) — softened, not removed; still a real confidence hit, just no longer alone enough to guarantee a sub-floor composite when every other signal is clean. Re-verified: all three real specimens now read `normalized_quality=0.75`, and `real3.jpg` re-run end to end now reaches `accepted` (`composite_confidence=0.75`).

**Acceptance criteria:**
- [x] All 16 `clean` variants (8 NBI + 8 ID) → verdict `pass` — verified, 16/16.
- [x] All 16 `blur` variants → verdict `reject` — verified, 16/16.
- [x] The calibration CSV shows **no overlap** between `clean` and `blur` variants' `blur_score` ranges — verified, huge gap (900.9–2558.7 vs 0.3–0.7).
- [x] `preprocess()` applied to a `skew` variant drops its re-measured `skew_deg` below `SKEW_WARN_DEG` — verified (and the bug that broke this the first time is documented above).
- [x] `assess()` runs with `GEMINI_API_KEY` unset — verified, both manually and as `test_assess_has_no_network_dependency`.
- [x] `pytest tests/test_quality.py` green, including boundary cases — **49 passed** (parametrized clean/blur checks across all 16 identities × both doc types, boundary tests for every floor/warn pair, plus a test pinning `config.QUAD_NOT_FOUND_QUALITY_SCORE` to its recalibrated value, added alongside the recalibration above).

**Verify:**
```bash
python -m src.ocr.quality --dump-csv data/references/mock > evals/results/quality_signals.csv
# confirm clean/blur blur_score columns don't overlap
pytest tests/test_quality.py -v   # 48 passed
pytest tests/ -q                   # 246 passed, 1 pre-existing unrelated failure (Phase 0 note), 1 skipped (Phase 4's opt-in live test)
```

---

### Phase 3 — Document registry ✅ DONE (verified 2026-08-08)

**Goal:** declarative field/validator registry, decoupled from extraction and validation logic.
**Files:** `src/ocr/doctypes.py`, `tests/test_doctypes.py`.
**Depends on:** Phase 0.

**Design note not in the original spec text:** `id_number`'s format and `expiry_date`'s requiredness both depend on `id_type`, which `FieldSpec.validator`'s single-string signature can't carry. Resolved with two small dispatch helpers rather than forcing it into the dataclass: `id_number_validator_for(id_type)` (the registry's static `id_number` `FieldSpec` only checks non-emptiness; `doc_validation.py`'s Rule 3, Phase 5, calls this dispatcher directly once it has the extracted `id_type`) and `required_fields(doc_type, id_type=None)` (the optional second argument gates `expiry_date`).

**The cross-check test caught a real bug on first run — this is exactly the failure mode it exists for.** Every mock NBI identity's `reference_no` (format `NBI-YYYY-NNNNNNNN`, invented in Phase 1 before any real specimen existed) failed `is_valid_nbi_reference_no()` (format grounded in two real specimens: one dash, two 6–12-char alnum groups) — two dashes vs. one. The regex is the one grounded in evidence; the mock data was stale. Fixed by regenerating `reference_no` values in the mock generator's shape (`FAMILY[:4]+YYMMDD-N+8digits`, e.g. `REYE900101-N00457821`) — documented inline in `scripts/make_onboarding_docs.py` next to `IDENTITIES`. `CV_PIPELINE_WALKTHROUGH.md`'s illustrative example (id01) updated to match. Dataset regenerated, reproducibility re-verified.

**Acceptance criteria:**
- [x] `required_fields(DocType.NBI_CLEARANCE)` returns exactly the 8 required fields — verified.
- [x] `required_fields(DocType.GOVERNMENT_ID, id_type=...)` conditional `expiry_date` behavior verified for all three `id_type` values.
- [x] Each field validator has passing + failing test cases, including `remarks` against `config.NBI_CLEAN_REMARKS` and `id_number` against all three `config.ID_NUMBER_PATTERNS` entries (both accept-matching-shape and reject-wrong-shape-for-type parametrized).
- [x] `full_name_display()` order-insensitivity verified for both `NbiExtractionResult` and `IdExtractionResult` via `rapidfuzz.fuzz.token_set_ratio`.
- [x] **Cross-check against Phase 1's dataset** — verified for both document types, 16/16 identities, after the `reference_no` fix above.
- [x] `pytest tests/test_doctypes.py` green — **39 passed**.

**Verify:**
```bash
pytest tests/test_doctypes.py -v   # 39 passed
pytest tests/ -q                    # 231 passed, 1 pre-existing unrelated failure
```

---

### Phase 4 — Extractor, Gemini path (Layer 2) ✅ DONE (verified 2026-08-08, incl. 3 live Gemini calls)

**Goal:** one Gemini multimodal call producing typed extraction, cached, quota-safe, for **both** document types.
**Files:** `src/ocr/extractor.py`, `NBI_EXTRACTION_PROMPT` + `ID_EXTRACTION_PROMPT` in `src/agent/prompts.py`, `tests/test_extractor.py`.
**Depends on:** Phase 2 (quality gate is upstream of every call), Phase 3 (registry feeds prompt field hints).

**Bug found via live verification, fixed in `doctypes.py` (Phase 3's module):** the real extraction returned `date_printed = "2026-02-14 09:00:00"` — a full timestamp, not a bare date. Both real specimens print a time alongside "Date Printed" (`§2.3`), and the `NBI_FIELDS` prompt hint already said "date/time" for this field — Gemini was faithfully correct, `is_valid_past_or_present_date()` (date-only `date.fromisoformat`) was too strict. Fixed by making `_parse_date()` fall back to `datetime.fromisoformat(...).date()` — benefits `date_of_birth`/`issue_date` too, harmlessly, since neither should ever actually carry a time component in practice. Right fix was loosening the validator, not truncating real information out of the prompt.

**Live verification (3 real Gemini calls, ~2000-2100 tokens each, well under the 20/day free-tier cap):**
1. **NBI, clean image** — every one of the 9 fields matched ground truth exactly (`overall_confidence=0.99`), including the corrected `reference_no` shape from Phase 3.
2. **Government ID, `national_id` layout** — all fields matched ground truth; `expiry_date` correctly came back `None` (National ID's no-expiry carve-out held on a real call, not just in the mock's ground truth).
3. **Wrong-document-type negative** (a Barangay Clearance rendered into the NBI slot) — Gemini honestly returned `doc_type=unknown_document` rather than force-matching NBI fields onto the wrong document. Quality gate passed it through (it's a legible image, just the wrong type) — the correct layer, Layer 2/Rule 1, is what catches it, exactly as designed (§1.4's outcome table).
4. **Cache verified against the real extraction, not just a fake client:** re-running the same NBI image spent **0 additional calls**, one JSON file present in `evals/results/ocr_cache/`.

**Retry-on-uncertainty added 2026-08-09, after a second real specimen (`data/references/real/real2.jpg`) exposed run-to-run nondeterminism.** Live call on this document returned `middle_name=null` at `model_confidence=0.1` — field is legible on the document (`RARELA`). Investigated before accepting "model limitation": `preprocess=True` didn't fix it (rules out resolution), a third independent call on the same bytes returned `RARELA` at `confidence=0.99`. `temperature=0.0` reduces but does not guarantee determinism on Gemini's hosted API. Added one conditional retry, gated by `config.OCR_ENABLE_RETRY` (default on) and `config.OCR_RETRY_CONFIDENCE_FLOOR=0.5`:
- `_should_retry()` fires when either `overall_confidence` is below the floor, or any required field is null with `model_confidence` below the floor — **except** fields that are structurally absent by design (National ID's `expiry_date`), which must not waste a retry.
- `_merge_results()` picks whichever of the two attempts has the higher `overall_confidence` as the base, then fills only that base's null fields from the other attempt — never discards a good field from the higher-confidence attempt to prefer a worse one.
- At most one retry is spent, ever — no loop. A failed retry (`APIError`/unparseable) falls back to the first attempt rather than losing it.
- `retry: bool | None` param on `extract_document()` lets callers force it on/off per call, independent of the config default (mirrors the existing `preprocess` param's shape).

**Test coverage added (`tests/test_extractor.py`, `_SequentialFakeClient`/`_SequentialFakeModels` — returns a different canned result per successive `generate_content()` call):**
- retry triggers and recovers a low-confidence null field
- merge preserves the base attempt's already-good fields rather than overwriting them
- retry does NOT trigger for National ID's structurally-absent `expiry_date`
- retry DOES trigger for the same null+low-confidence shape on a Driver's License, where `expiry_date` is required
- retry triggers on low `overall_confidence` alone, even with every field present
- exactly one retry spent even when the retry is itself uncertain — no loop
- `retry=False` param and `OCR_ENABLE_RETRY=False` config both suppress it even when otherwise warranted
- a confident first attempt never retries (pins down the existing canned fixtures don't accidentally trigger it)
- a failed retry call keeps the first attempt's good fields rather than returning nothing

**Acceptance criteria:**
- [x] A clean NBI mock image returns all 8 required fields non-null — verified live, exact match to ground truth.
- [x] A clean ID mock image returns `doc_type=government_id` with correct `id_type` and required fields non-null (National ID's `expiry_date` null) — verified live for `national_id`; `drivers_license`/`passport` covered by the mocked-client unit tests (doc-type dispatch is provider-agnostic, so this isn't three separate live risks).
- [x] Re-running the same image spends **0 API calls** — verified against the real cache, not just mocked.
- [x] A `reject`-quality image returns `(None, report)` with **0 API calls** — verified (mocked client, `call_count == 0`).
- [x] A non-document/wrong-type negative doesn't get a guessed read — verified live (`unknown_document`, not a guessed NBI match).
- [x] Both `APIError` and `LLMBackendError` return `(None, report)` rather than raising — verified (mocked, both exception types).
- [x] Cache keys include the preprocess flag — verified (`preprocess=True` vs `False` on the same image produces 2 distinct cache files, 2 calls).
- [x] `pytest tests/test_extractor.py` green — **23 passed, 1 skipped** (13 original + 10 retry-specific; the skip is the opt-in live test, gated behind `RUN_LIVE_OCR_TESTS=1` so routine `pytest tests/` never spends quota on its own — the 3 live calls above were run directly, not via that suite).

**Verify:**
```bash
pytest tests/test_extractor.py -v          # 23 passed, 1 skipped (live test, opt-in)
pytest tests/ -q                            # 266 passed, 1 skipped

# one real extraction + cache proof (spends 1 API call, then 0 on re-run)
python -c "
from src.agent import usage
from src.ocr.extractor import extract_document
from src.schemas import DocType
before = usage.get_usage_today()
img = open('data/references/mock/nbi_id01_clean.png', 'rb').read()
extract_document(img, 'image/png', DocType.NBI_CLEARANCE)
extract_document(img, 'image/png', DocType.NBI_CLEARANCE)  # should hit cache
after = usage.get_usage_today()
print('calls spent:', after['request_count'] - before['request_count'])  # expect 1, not 2
"

# opt in to the live pytest test specifically
RUN_LIVE_OCR_TESTS=1 pytest tests/test_extractor.py::test_live_extraction_against_real_gemini -v
```

---

### Phase 4a — Ollama fallback ✅ DONE (verified 2026-08-09)

**Goal:** `VISION_PROVIDER=ollama` works end to end through the same `extract_document()` call site, no `extractor.py` changes required.
**Files:** `src/agent/llm_client.py` (`OllamaClient._contents_to_messages()` image branch), `tests/test_llm_client.py`.
**Depends on:** Phase 4.

`config.get_vision_client()`'s Ollama branch already existed (built during the earlier merge) — the actual gap was `_contents_to_messages()`: `extract_document()`'s `_call_gemini()` calls `generate_content(contents=[types.Part.from_bytes(...), prompt], ...)`, a **flat list mixing a raw `Part` and a plain string** — not the `[Content(...), ...]` shape every other call site (router/ReAct/search_web) uses. The old code assumed every item had `.parts`/`.role` and would have raised `AttributeError` on this exact shape; it also had no image-encoding path at all. Added `_is_flat_part_list()` to detect the shape and `_flat_parts_to_message()` to base64-encode the image onto Ollama's `images` field (`/api/chat`'s actual multimodal contract) while leaving the existing `Content`-list handling untouched.

**Live-verified against a real local Ollama server, not just mocked** — no vision-capable model was pulled (`llama3.2-vision` isn't present, only text models), so full OCR-quality accuracy couldn't be checked, but the transport/request-format and fail-safe paths were verified for real:
1. A real request against `llama3.2:latest` (non-multimodal) with an embedded image → Ollama returned a real HTTP 400 with `"Multimodal data provided, but model does not support multimodal requests"` — a *semantic* rejection, not a malformed-request error, which is exactly the confirmation that the request format (base64 image on `images`, JSON body shape) was well-formed enough for the real server to parse and understand.
2. That 400 correctly surfaced as `LLMBackendError` and `extract_document(..., use_cache=False)` correctly returned `(None, report)` — verified live, not simulated.
3. Cache-key model slugs confirmed distinct: `gemini-2.5-flash` vs. `ollama-llama3.2-vision`.

**Acceptance criteria:**
- [x] `VISION_PROVIDER=ollama` round-trips through `extract_document()` to a clean `(None, report)` on a real backend failure — never a crash. Full accuracy round-trip (a real answer, not just the fail-safe path) needs a vision-capable model pulled (`ollama pull llama3.2-vision` or similar), not available this session.
- [x] Cache keys differ between `gemini` and `ollama` runs — verified.
- [x] `LLMBackendError` → `(None, report)` — verified live.
- [x] Existing chat tests (`tests/test_llm_client.py`) still pass, plus 3 new tests for the flat-part-list image path (including a guard that the normal `Content`-list shape still works unaffected) — **14 passed** (11 existing + 3 new).

**Verify:**
```bash
pytest tests/test_llm_client.py -v   # 14 passed
# needs a vision-capable Ollama model pulled for a real accuracy round-trip:
# ollama pull llama3.2-vision
VISION_PROVIDER=ollama OLLAMA_VISION_MODEL=llama3.2-vision python -c "
from src.ocr.extractor import extract_document
img = open('data/references/mock/nbi_id01_clean.png', 'rb').read()
result, report = extract_document(img, 'image/png')
print(result)
"
```

---

### Phase 5 — Validation + state (Layer 3) ✅ DONE (verified 2026-08-09)

**Goal:** deterministic accept/reject/review policy for both document types, the cross-document check, and durable per-employee checklist state.
**Files:** `src/guardrails/doc_validation.py`, `src/memory/onboarding_status.py`, `tests/test_doc_validation.py`, `tests/test_onboarding_status.py`.
**Depends on:** Phase 4 (needs real extraction result shapes to validate against).

**Design note not in the original spec text:** the six rules are always computed unconditionally (`_rule_type_match` through `_rule6_fail_safe`), then reduced to an outcome by one shared `_resolve_outcome()` — a pure function over the 6 `RuleResult`s, reused by both `validate_document()` and `validate_id_document()` since the rule-name strings are identical across both. This is also what makes "Rule 4/Rule 6 never auto-reject" a **structural** guarantee rather than a per-call-site convention: `_resolve_outcome()` only ever routes an identity or fail-safe failure to `NEEDS_REVIEW`, never `REJECTED`, so there's no call site that could get this wrong.

**Real bug found via the cross-check, fixed in `config.py` (not `doc_validation.py`) before Rule 3 could be trusted:** `config.NBI_CLEAN_REMARKS` didn't include `"NO RECORD ON FILE"` — the exact, verbatim clean-status phrasing printed on `data/references/real/real2.jpg` (confirmed via its cached extraction, `model_confidence` 0.99–1.0 across two live calls, Phase 4's retry investigation). Rule 3 would have failed this genuinely-clean real document and forced `composite_confidence = 0.0` on it. Added to the allowlist with a comment citing the specimen — same "real evidence overrides an untested placeholder" pattern as the `MIN_IMAGE_DIM_PX` recalibration.

**`python-dateutil` pinned explicitly** in `requirements.txt`/`requirements-api.txt` (`relativedelta` powers Rule 5) — it was only ever present as a transitive dependency (via another package's own requirement), never declared, so a clean install could silently have dropped it.

**Cross-checked against Phase 1's full mock dataset, not just hand-built fixtures** (same discipline as Phase 3's registry cross-check): all 8 NBI `.expected.json` ground-truth identities → `accepted`, 0 rule failures. All 8 ID `.expected.json` ground-truth identities (3 `national_id` + 3 `drivers_license` + 2 `passport`) → `accepted`, 0 rule failures. All 8 clean-pair identities → `validate_cross_document()` → `passed=True`, 0 false-positive mismatches. Confirms the validator and the mock generator agree on what "clean" means, independently of the unit tests below.

**Acceptance criteria — `validate_document()` (NBI):**
- [x] One passing test per rule (1 through 6), including Rule 3's `remarks` check — verified, including the real-remarks-phrasing fix above.
- [x] Rule 5 takes the earlier of `valid_until` and `date_printed + NBI_VALIDITY_MONTHS` — verified (`test_rule5_stricter_printed_valid_until_governs_over_employer_window`).
- [x] Expired document with a pinned `as_of` → `rejected` — verified.
- [x] Wrong-person document → `needs_review`, not `rejected` — verified, and structurally guaranteed (see design note above).
- [x] All rules passing → `accepted` — verified, plus the full 8/8 mock cross-check above.
- [x] A Rule 3 (format) failure forces `composite_confidence == 0.0` regardless of extraction confidence — verified with `overall_confidence=0.99`.
- [x] `extracted=None` (quality fine) → `needs_review` (Rule 6 fail-safe) — verified.

**Acceptance criteria — `validate_id_document()` (§1.4a):**
- [x] One passing test per rule for each `id_type` where behavior differs (Rule 2/5) — verified for all three.
- [x] `national_id` with no `expiry_date` → Rule 5 skipped with `passed=True`; `drivers_license`/`passport` with no `expiry_date` → Rule 2 fails — verified both directions, plus `passport` explicitly.
- [x] Expired license/passport with a pinned `as_of` → `rejected` — verified both.
- [x] Wrong-person ID → `needs_review`, not `rejected` — verified.
- [x] All three `id_type` "clean" fixtures individually reach `accepted` — verified, plus the full 8/8 mock cross-check above.

**Acceptance criteria — `validate_cross_document()` (§1.4a):**
- [x] Matching name + matching DOB across a paired NBI/ID fixture → `passed=True` — verified, plus all 8 mock clean pairs.
- [x] Name mismatch → `passed=False`, `detail` mentions `"name"`, not the actual values — verified.
- [x] DOB mismatch → `passed=False`, `detail` mentions `"date_of_birth"`, not the actual values — verified.
- [ ] Only one document on file → rule absent from `rules`, not present with any `passed` value.
- [ ] Sibling lookup is a cache hit — `GET /usage` doesn't move on the second upload.

The last two are marked open, not skipped: they describe behavior of the **caller** (`POST /upload-doc`'s sibling-lookup wiring, §2.7/§2.8), not of `validate_cross_document()` itself — the function's signature takes two already-extracted results, so there's no "only one document" case to construct at this layer. `src/memory/onboarding_status.py`'s `get_document()` (the lookup half of that wiring) is built and tested now (`test_get_document_returns_none_when_not_yet_uploaded`, `test_get_document_sibling_lookup_for_cross_document_check`); the "rule absent from `rules`" and "cache hit" assertions belong in Phase 6's `test_api.py` once `POST /upload-doc` exists to call it from.

**Acceptance criteria — shared/state:**
- [x] `record_result()` then `get_status()` round-trips the same outcome, for both `doc_type` values on the same `employee_id` — verified, plus upsert-on-resubmission, per-employee isolation, and `NEEDS_REVIEW`/`REJECTED` both correctly still counting as `missing`.
- [x] **PII assertion:** no fixture's name, DOB, reference/ID number appears as a substring in any `RuleResult.detail` or `ValidationResult.message`, across `validate_document`, `validate_id_document`, and `validate_cross_document` — verified (`test_no_pii_leaks_into_any_rule_detail_or_message`).
- [x] `pytest tests/test_doc_validation.py tests/test_onboarding_status.py` green — **42 passed** (31 + 11).

**Verify:**
```bash
pytest tests/test_doc_validation.py tests/test_onboarding_status.py -v   # 42 passed
pytest tests/ -q                                                          # 308 passed, 1 skipped
```

---

### Phase 6 — Agent + interface wiring ✅ DONE (verified 2026-08-09; UI styled 2026-08-09, live-verified in a real Docker test session 2026-08-10 — layout, styling, and a real NBI clearance upload all confirmed working)

**Goal:** the pipeline is reachable through chat, the API, and the UI, using the existing ReAct/API/UI patterns unchanged.
**Files:** `src/agent/orchestrator.py`, `src/agent/prompts.py` (`ROUTER_PROMPT`), `src/schemas.py` (`Intent`), `src/api.py`, `src/ui.py`, `src/monitoring.py`.
**Depends on:** Phase 5.

**Two design decisions made against the actual current codebase, not the spec text above (both audited before writing any code — see chat history 2026-08-09):**
1. **No `_function_declarations()`/`_execute_tool()` exist anywhere** — the real `orchestrator.py` is a closed-enum ReAct loop (`ReActAction = SEARCH_KB | SEARCH_WEB | FINISH`), not Gemini function-calling. `Intent.DOCUMENT_UPLOAD`/`DOCUMENT_STATUS` are handled as short-circuit branches in `run_turn()`, the same pattern already used for `Intent.OUT_OF_SCOPE` — no `validate_checklist`/`extract_document(source_hash)` ReAct tools were built, since there's no tool-calling mechanism for them to register into. `_react_loop()`/`ReActAction`/the ReAct prompts are untouched.
2. **No HR/employee record source exists anywhere in the codebase** (confirmed by grep — `faculty_record` has never had an upstream producer). `POST /upload-doc` takes `full_name`/`date_of_birth` as request fields alongside `employee_id`, supplied by the uploader, rather than an assumed HR lookup.

**UI scope, explicit:** backend + API + agent fully built and tested. `src/ui.py` got a minimal, deliberately **unstyled** `st.file_uploader` block (sidebar, no custom CSS) proving the pipe is connected end to end — not integrated into the existing pixel-tuned design system, since that's a genuinely bespoke, teammate-owned visual design (recreates a specific design handoff) with no existing mockup for this feature. Styling it is an explicit follow-up.

**Acceptance criteria:**
- [x] `POST /upload-doc` returns 200 with both `validation` and `checklist` in the body for a clean mock image, for both `doc_type` values — verified live (real cache-hit extractions against `nbi_id01_clean.png`/`id_id01_clean.png`, zero new API calls) plus unit tests.
- [x] Uploading NBI then ID for the same "clean pair" employee → the second upload's response includes a `cross_document_consistency` `RuleResult` with `passed=True` — **verified live**, not just mocked (see above); `GET /usage` wasn't separately re-checked live but the sibling lookup path (`load_cached_result`) is proven to return the cached result rather than calling the vision client, which is what makes the zero-extra-calls property hold.
- [x] Uploading a mismatched pair → the second upload's outcome reflects `needs_review` even if that document's own single-document rules all passed — verified via a mocked test (`test_upload_doc_cross_document_mismatch_escalates_to_needs_review`); not live-verified, since the mismatched mock pairing wasn't already in the OCR cache and a fresh live call wasn't worth spending quota on when `apply_cross_document_result()`'s escalation logic is already directly unit-tested.
- [x] `ChecklistStatus.missing` correctly lists both `nbi_clearance` and `government_id` when neither has been uploaded yet, and drops each as it's submitted — verified.
- [x] An oversized file is rejected before any image processing — verified (`extract_document` call count asserted at 0).
- [x] A malformed/wrong-content file is rejected by the magic-byte check (PNG/JPEG signatures), not the filename — verified.
- [x] `GET /onboarding-status/{employee_id}` returns both rows once both documents are uploaded — verified.
- [x] *"Is my NBI clearance okay?"* routes to `DOCUMENT_STATUS`; *"what documents do I need to submit?"* routes to `FAQ` → `search_kb` — **live-verified against real Gemini 2026-08-09, 8/8 correct** (`document_upload`/`document_status`/`faq` phrasings from PLAN.md §4.4 and this doc's own acceptance-criteria wording, plus an unrelated-topic control case), confidence 0.90–0.95 across all eight, 8 API calls spent (well under the daily free-tier cap).
- [x] The one tool-observation this design actually produces (`get_onboarding_status`'s `AgentStep.observation`, inside `run_turn()`'s `DOCUMENT_STATUS` short-circuit — see design decision 1 above, there is no separate `validate_checklist`/`extract_document(source_hash)` tool) contains no field-value substring for either document type — asserted directly in a test (`test_document_status_tool_observation_is_pii_free`), not just trusted from the schema docstring.
- [ ] UI checklist panel shows both required documents and updates independently as each is uploaded, without a manual page refresh. **Deferred** — the stub uploader shows the raw JSON response, not a styled independent-updating panel; that's the explicit UI-scope deferral above.
- [ ] An MLflow run for an upload carries the new tags/metrics — `doc_trace()` exists, is wired into `POST /upload-doc`, and runs (without crashing) on every `test_upload_doc_*` test via a real `TestClient` call. **Not manually inspected in the MLflow UI** — the local dev `data/mlflow.db` has an unrelated schema-version mismatch (Docker's `mlflow==3.15.1` vs. the host's `mlflow==2.22.0`) blocking that inspection right now; verified with a throwaway tracking URI instead, which proves the code path runs but not what actually lands in a real trace.

**Verify:**
```bash
pytest tests/test_router.py tests/test_orchestrator.py tests/test_doc_validation.py \
       tests/test_extractor.py tests/test_api.py -v
pytest tests/ -q   # 334 passed, 1 skipped

uvicorn src.api:app --reload &
curl -F "file=@data/references/mock/nbi_id01_clean.png" -F "employee_id=EMP-04821" \
     -F "full_name=REYES, MARIA SANTOS" -F "date_of_birth=1990-01-01" -F "doc_type=nbi_clearance" \
     http://localhost:8000/upload-doc
curl -F "file=@data/references/mock/id_id01_clean.png" -F "employee_id=EMP-04821" \
     -F "full_name=REYES, MARIA SANTOS" -F "date_of_birth=1990-01-01" -F "doc_type=government_id" \
     http://localhost:8000/upload-doc   # second upload: watch for cross_document_consistency in the response
curl http://localhost:8000/onboarding-status/EMP-04821
streamlit run src/ui.py   # manual: sidebar has an Employee ID field + "Upload a document (stub)" expander
```

---

### Phase 7 — Evals ✅ DONE (verified 2026-08-09)

**Goal:** the headline secondary metrics (PLAN.md §9) exist and are reproducible from cache.
**Files:** `evals/run_ocr_eval.py`, `evals/run_validation_eval.py`.
**Depends on:** Phase 6 (needs the full pipeline to generate cache entries against), though it can start against Phase 4's cache alone for the OCR-only numbers.

**A real bug found and fixed via `run_validation_eval.py`'s first real run — this is exactly the failure mode it exists for.** `nbi_neg_wrong_person_2.png` scored a **false auto-pass**: expected `needs_review`, got `accepted`. Root cause in `scripts/make_onboarding_docs.py`'s `render_nbi_negative()`'s `wrong_person` branch — it built the impostor fixture as `dict(base, full_name=impostor["full_name"])`, but `render_nbi_clean()` reads `identity["family_name"]`/`["first_name"]`/`["middle_name"]` directly, never `full_name`. The override targeted a key nothing reads, so the rendered image showed the **victim's own real name** — the negative never tested Rule 4 at all, for either `wrong_person` fixture (both share the buggy branch). `render_id_negative()`'s `cross_name_mismatch` branch already had the correct pattern (overrides all three name-part fields); fixed `wrong_person` to match it, regenerated the mock dataset, live-verified the two fixed images now render genuinely different names, re-ran the eval: **0% false auto-pass rate, 100% precision/recall/F1 on `needs_review`.**

**A second, smaller bug caught during my own verification of `run_ocr_eval.py`, in the eval script itself, not the extractor:** `date_printed` scored 0% exact-match / high CER on a fixture that was actually extracted correctly. `date_printed` legitimately carries a time component on real documents (`"2026-02-14 09:00:00"`, the same finding that drove `doctypes._parse_date()`'s fallback in Phase 4) — the mock's `.expected.json` ground truth is a bare date with no time, so a naive string comparison unfairly penalized a correct extraction. Fixed by normalizing both sides to date-only before comparing (`_normalize_for_comparison()`), mirroring `doctypes._parse_date()`'s reasoning rather than duplicating its private implementation.

**Real subset has no ground-truth labels yet.** `data/references/real/` has no `*.expected.json` files — nobody has hand-transcribed the two real specimens. `--subset real` runs cleanly and reports zero fixtures with a clear message, rather than fabricating labels from a prior model extraction (which would grade the model against its own output — circular, proves nothing).

**Quota discipline during verification:** `run_validation_eval.py` runs entirely off `extractor.load_cached_result()` (hash-only lookup, never calls the vision model) plus a purely local `quality.assess()` re-run — genuinely zero API cost, not just documented as such. `run_ocr_eval.py` does call the model on a cache miss, so verification here was deliberately scoped with `--limit 1` (one identity per doc type) rather than running the full 16-identity mock set, to avoid spending quota without asking first — a full run (`--subset mock`, no `--limit`) is available whenever real numbers across the whole dataset are wanted.

**Acceptance criteria:**
- [x] Per-field exact-match + normalized CER printed to console and written to `evals/results/`, broken out per `doc_type` — verified.
- [x] `--no-preprocess` mechanically produces a **distinct cache entry** from the default run (confirmed: two separate `..._0_...json`/`..._1_...json` cache files for the same image hash) — on the one clean, easy fixture tested, both preprocess settings happened to score 100% exact-match, so the *numbers* tied even though the underlying calls were genuinely separate live calls, not a cache-reuse no-op. A full run across harder/degraded fixtures would be needed to see the ablation move a headline number, not just prove the plumbing is real.
- [x] `run_validation_eval.py` reports precision/recall/F1 and the false auto-pass rate per `doc_type`, plus per-rule trigger counts, plus a dedicated `cross_document_consistency` section (overall trigger rate + the clean-pair false-trigger rate specifically) — verified, and it's what caught the real bug above.
- [x] `--subset real`/`--subset mock` never pooled — verified (separate report files, separate JSON sections).
- [x] A second run against the same cache spends 0 API calls — verified by construction for `run_validation_eval.py` (never calls the vision model at all, cache-hit or miss); verified for `run_ocr_eval.py` via the cache-file check above (re-running the same `--subset mock --limit 1` after the first call hits the existing cache entry, same mechanism already proven in Phase 4/6).

**Verify:**
```bash
python evals/run_ocr_eval.py --subset mock --limit 1     # quota-cheap smoke run
python evals/run_ocr_eval.py --subset mock --limit 1 --no-preprocess
python evals/run_validation_eval.py                        # zero API cost, safe to run in full anytime
# python evals/run_ocr_eval.py --subset mock                # full 16-identity run -- spends real quota, ask first
```

---

# Part 4 — Verification

Per-phase acceptance criteria now live in Part 3 — check there first when validating a specific stage. This section is the **end-to-end smoke test**, run once everything is wired, plus the negative-path checks that are the whole point of the component. For a narrated walk through one document with concrete data at every hop (useful when actually debugging, not just confirming), see `CV_PIPELINE_WALKTHROUGH.md`.

```bash
pip install -r requirements.txt

# 1. dataset — naming convention: nbi_<id>_<variant>.png and id_<id>_<variant>.png
python scripts/make_onboarding_docs.py
ls data/references/mock/ | wc -l  # expect 100 images + identities.json + 16 *.expected.json

# 2. unit tests — all offline, no API key needed
pytest tests/test_quality.py tests/test_doc_validation.py \
       tests/test_extractor.py tests/test_onboarding_status.py -v

# 3. quality gate calibration (no API calls)
python -m src.ocr.quality --dump-csv data/references/mock > quality_signals.csv
#    confirm clean variants separate from blur/lowres before locking thresholds

# 4. one live extraction per document type, then confirm the cache makes it free
python -c "from src.ocr.extractor import extract_document; ..."   # spends 1 call, NBI
python -c "from src.ocr.extractor import extract_document; ..."   # spends 1 call, government ID
#    re-run either: 0 calls. Check evals/results/ocr_cache/ for the sha256 JSONs.

# 5. end to end — both documents for one employee, in order, watch the second response
uvicorn src.api:app --reload
curl -F "file=@data/references/mock/nbi_id01_clean.png" \
     -F "employee_id=EMP-00123" http://localhost:8000/upload-doc
curl -F "file=@data/references/mock/id_id01_clean.png" \
     -F "employee_id=EMP-00123" http://localhost:8000/upload-doc   # expect cross_document_consistency in the response
curl http://localhost:8000/onboarding-status/EMP-00123             # expect both doc_type rows
curl http://localhost:8000/usage      # vision spend broken out by model name

# 6. UI
streamlit run src/ui.py               # upload both mock images, see both checklist rows update

# 7. evals
python evals/run_ocr_eval.py --subset mock                     # both doc_types, reported separately
python evals/run_ocr_eval.py --subset mock --no-preprocess     # the ablation
python evals/run_validation_eval.py                            # zero API cost, incl. cross-document rate

# 7a. provider fallback check (once step 4a lands) — same image, both providers
VISION_PROVIDER=gemini python -c "from src.ocr.extractor import extract_document; ..."
VISION_PROVIDER=ollama python -c "from src.ocr.extractor import extract_document; ..."
#     confirm both return an extraction result (possibly with lower confidence
#     from Ollama) and that GET /usage attributes spend to the right model name
```

**Negative paths to verify by hand, because they're the whole point:**
- Upload the blurriest mock variant → `rejected`, and **`/usage` does not increment**. That single check proves the gate is load-bearing.
- Upload the wrong-person negative → `needs_review`, not `rejected`.
- Upload the expired negative with a pinned `as_of` → `rejected` on Rule 5.
- Upload a non-document photo → `not_a_document`, `rejected`, no crash.
- Inspect an MLflow run → confirm **no name or reference number appears in any tag**.

---

# Part 5 — Open decisions

Flag these; none block starting at step 0.

1. **NBI reference-number format — now grounded in one real-ish sample, still not confirmed.** `HGUR87H38D-U47204A873` is two alphanumeric groups joined by a dash (~10+9 chars). Update the Rule 3 pattern to `^[A-Z0-9]{6,12}-[A-Z0-9]{6,12}$` — structurally informed rather than arbitrary, but **n=1**, and that one sample is a demo-generator template (see item 6), not a confirmed-authentic document. Tighten further once the real subset (§1.8) exists.
2. **`purpose` field values.** Warn-only on unrecognized values until the real subset shows the actual vocabulary.
3. **`remarks` allowlist (`NBI_CLEAN_REMARKS`).** Currently three plausible phrasings, from zero confirmed real samples. This is the field with the highest cost if wrong in either direction — too narrow and legitimate clean clearances get bounced to `needs_review` unnecessarily; too permissive and it stops meaning anything. Needs the real subset before it's trustworthy, more urgently than the other thresholds.
4. **The two identifiers problem.** The sample shows a labeled "NBI ID NO" *and* a separate unlabeled red control number top-right. Only the former is extracted (§2.3/§2.5). Confirm with a second real sample whether the control number is ever needed (e.g. for duplicate-submission detection) before assuming it's safe to ignore permanently.
5. **Paid Gemini key vs. local fallback.** 20 requests/day makes step 4 slow even with caching. The cache means a full eval can be spread across days; budget a key before demo week, or fall back to an Ollama vision model per §2.6a if that's not possible in time.
6. **`NBI-Clearance.jpg`'s provenance.** Branded `nbiclearance.org`, cartoon placeholder photo — those two are still real tells of a demo-mockup source. **Correction: the Agency/CASID/DATAID/BIOID/RECID/O.R. No/O.R. Date/DST PAID/INTID/PRTID block is a genuine printed field on real NBI Clearances** (usually mostly blank, a few filled — confirmed directly, not a generator artifact as first assumed here). Not eligible as a real-subset document either way (PLAN.md §3.5a needs actual consented clearances from real people). Gitignored (item 9).
6a. **A second sample, `data/references/real/nbi-clearance-real.WEBP`, has the same open provenance question** — placed in `real/` but showing the same structural hallmarks (identical field layout/labels to item 6's sample, tiled background watermark). Not yet resolved whether this is an actual consented document or another specimen; if the latter, it belongs in `samples/`, not `real/`, and can't count toward PLAN.md §3.5a's real-subset requirement. **This sample is what drove the `MIN_IMAGE_DIM_PX` recalibration in Phase 2** — it originally hard-`reject`ed on the old 640px floor despite being genuinely legible (confirmed by bypassing the floor and running a real extraction: all 9 fields at 0.98–0.99 confidence). Floor lowered to 400/warn-at-640; the sample now reads `warn`, not `reject`, and proceeds to extraction. **`quad_found=False` still fires and is unresolved** — `_detect_document()`'s contour heuristic has only ever been exercised against our own mocks, which all draw an explicit full-frame border rectangle; a real document photo without that artificial frame may never trip `quad_found=True` at all. Doesn't currently block anything by itself (soft signal only), but it's the first empirical sign of a real mock-vs-real generalization gap in Layer 1 — not yet enough samples to say if it's systematic. Revisit once 2–3 more real samples exist.
7. **PDF uploads** are out of `ALLOWED_IMAGE_MIME` for now. Add only if a real submission arrives as PDF.
8. **National ID's missing expiry — confirmed pattern or one-sample artifact?** The sample shows no printed expiry at all for an adult citizen's PhilSys card, which matches general public knowledge of the PhilSys ID's design, but this design treats it as confirmed real-world behavior off a single specimen. If a future real sample shows an expiry field on some National ID variant, `validate_id_document()`'s Rule 2/5 carve-out (§2.7) needs revisiting — right now it hard-codes "never required for `national_id`," not "required if present."
9. **All four reference images (`NBI-Clearance.jpg`, `National_id.png`, `drivers_license.jpg`, `passport_ph.jpg`) are specimen/demo templates**, not real documents — `SPECIMEN` watermarks on three, an `AUTODEAL` marketplace watermark on the fourth. Same treatment as item 6: good for layout, not real-subset eligible, gitignored. **Every `id_number` pattern in `config.ID_NUMBER_PATTERNS` (§2.2/§2.5) is n=1 per `id_type` and unconfirmed** — same caveat as item 1's NBI pattern, at three times the exposure (three unconfirmed patterns instead of one). Tighten all three once the real subset exists.
10. **MRZ-first extraction for passports (§2.6) is an assumption, not yet validated against this codebase's actual accuracy.** Standard OCR literature supports MRZ being more machine-reliable than visual header text, but that's a general claim, not one measured against Gemini's multimodal reading specifically. Worth a targeted check in Phase 7: does prompting "prefer the MRZ" actually improve `passport` accuracy over the visual fields, or was Gemini already effectively doing this? If it's already implicit, the prompt instruction is free insurance; if it hurts (e.g. an MRZ misread outweighing a clear header), un-do it.
11. **Face/photo matching is explicitly not implemented, stated so it isn't assumed present later** (§1.4a). If a future requirement wants biometric matching between an applicant's ID photo, their NBI photo, and/or a live capture, that's a materially different and heavier component — a face-embedding model, its own similarity threshold, its own accuracy eval — not an extension of the current text-field pipeline.
12. **Unrelated but adjacent:** `config.CATEGORIES` and `COLLECTION_NAME = "hr_policies"` are still the Midterm HR taxonomy, and `Citation` has no `page` field despite CLAUDE.md requiring page-level citations. Not this component's scope — raise with the RAG owner rather than fixing in a CV diff.
