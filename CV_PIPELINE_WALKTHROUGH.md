# CV Pipeline Walkthrough — One NBI Clearance, Start to Finish

This is the companion to `CV_INTEGRATION.md`. That doc explains *what each layer is and why it exists*. This doc follows *one document through the whole system*, showing what the data actually looks like at each hop, and what to check when a stage misbehaves.

Read this when you're debugging. Read `CV_INTEGRATION.md` when you're deciding how something should work.

**Scenario, carried through every stage below:**

> A part-time faculty applicant, in the pre-employment stage, uploads a phone photo of their NBI Clearance in the chat UI and asks the agent to check it.

- Applicant: **Reyes, Maria Santos** (fictional, mock-subset identity — never a real name; PLAN.md §3.5a)
- `employee_id`: `EMP-04821`
- Photo: `nbi_clearance_photo.jpg`, 1920×2560, 2.1MB, taken with a phone, slight tilt, decent lighting
- Today's date (for the walkthrough): 2026-08-08

This scenario ends in a clean **`accepted`** outcome — the point is to see the whole pipeline working. Every stage's **"When it breaks"** section then covers the ways this exact scenario could have gone differently.

---

## The three-layer idea, in one sentence each

Before the stages — the shape to hold in your head:

1. **A cheap deterministic check decides whether the photo is even worth reading** (`src/ocr/quality.py`).
2. **One Gemini call reads the photo and proposes field values** — it never decides anything (`src/ocr/extractor.py`).
3. **Deterministic rules decide accept/reject/review from those proposed values** (`src/guardrails/doc_validation.py`).

The model sits in the middle, and it's the weakest authority of the three. Full rationale: `CV_INTEGRATION.md` §1.2.

---

## Stage 0 — Before the upload

**What happens:** The applicant is already mid-conversation with the agent. A `session_id` exists, `employee_id` is known (however the app authenticates that — out of scope here), and if the RAG side already asked "full-time, part-time, or ASF?" earlier in the session, `faculty_class` is already sitting in `onboarding_profile`. Nothing CV-specific has happened yet.

**The code:** N/A — this is just prior state.

**The data right now:**
```json
{"session_id": "sess_8f2c1a", "employee_id": "EMP-04821", "faculty_class": "part_time"}
```

**When it breaks:** If `faculty_class` isn't resolved yet, Rule 4 (identity match) still runs — it just can't apply a class-specific expected format if one exists. Not a blocker for the NBI Clearance (identity rules don't depend on faculty class), but it will matter later for the T3 leave/benefits RAG answers in the same session.

---

## Stage 1 — UI → API

**What happens:** The applicant picks the photo in the chat composer's file uploader and hits send. The browser POSTs the raw bytes to the FastAPI backend. **The UI never talks to Gemini directly** — this is a hard rule (`CLAUDE.md`, `CV_INTEGRATION.md` §2.8) and the only reason it's safe to keep API keys server-side only.

**The code:** `st.file_uploader(...)` in `src/ui.py`'s composer → `requests.post(f"{api_url}/upload-doc", files=..., data={"employee_id": ...})`, the same request pattern as `_fetch_pending_response()` (`ui.py:472-509`) already uses for `/chat`.

**The data right now:** an HTTP multipart request, ~2.1MB body, `employee_id=EMP-04821`.

**When it breaks:**
- UI shows "could not reach the API" → same failure path as a chat message failing (`ui.py`'s `except requests.RequestException`) — check the API is actually running (`uvicorn src.api:app --reload`), not a CV-specific bug.
- Upload silently does nothing → check the file uploader actually fired a POST (browser devtools → Network tab); a Streamlit rerun can eat a click if state isn't handled right.

---

## Stage 2 — Door checks (`POST /upload-doc`, before any image processing)

**What happens:** Three cheap checks run before a single pixel is touched. Any failure here returns immediately — no OpenCV, no Gemini.

**The code:** `src/api.py`'s `upload_doc()` handler:
1. `len(file_bytes) <= config.MAX_UPLOAD_BYTES` (8MB) — reject oversized uploads before reading them into memory.
2. **MIME check by magic bytes, not the filename extension** — a `.txt` renamed to `.png` must fail here. (`python-multipart` gives you the raw bytes; sniff the first few bytes against `image/jpeg`/`image/png` signatures, don't trust `file.content_type`.)
3. `cv2.imdecode(...)` — this is also the **EXIF strip point**. Decoding to a raw pixel array discards all metadata, including phone geolocation, as a side effect. No separate "strip EXIF" step exists or is needed.

**The data right now:** `bytes` (2.1MB) → `np.ndarray`, shape `(2560, 1920, 3)`, dtype `uint8`, no metadata attached.

**When it breaks:**
- A legitimate large photo gets rejected → check `MAX_UPLOAD_BYTES`; modern phone cameras can produce 8MB+ JPEGs, so this ceiling may need raising.
- A real image gets MIME-rejected → your magic-byte sniff is checking the wrong offsets, or the phone saved a HEIC file (`image/heic` isn't in `ALLOWED_IMAGE_MIME`) — a real failure mode worth testing for.

---

## Stage 3 — Layer 1: the OpenCV quality gate

**What happens:** Five deterministic signals get computed from the pixel array — no model, no network call. They combine into one verdict: `pass`, `warn`, or `reject`. **A `reject` stops the pipeline here.** This is both a real accuracy decision (an unreadable photo can't be extracted from reliably) and the reason a 50-image eval doesn't burn the whole day's Gemini quota.

**The code:** `src/ocr/quality.py`, `assess(image) -> ImageQualityReport`.

**The data right now** — for this photo, decent lighting, slight tilt:
```json
{
  "blur_score": 340.5,
  "exposure_clip": 0.02,
  "skew_deg": 3.1,
  "min_dim_px": 1920,
  "quad_found": true,
  "verdict": "pass",
  "normalized_quality": 0.93,
  "reasons": []
}
```
Checked against config: `blur_score` 340.5 > `BLUR_VARIANCE_WARN` 250 (comfortably sharp); `skew_deg` 3.1 < `SKEW_WARN_DEG` 5 (barely tilted); `min_dim_px` 1920 > `MIN_IMAGE_DIM_PX` 640. Nothing trips a warning, so verdict is a clean `pass`.

**When it breaks:**
- **Everything gets `reject`ed, even decent photos** → thresholds miscalibrated. Re-run the calibration CSV (`CV_INTEGRATION.md` §2.4) — dump `assess()` output over the whole mock dataset and check where `clean` vs `blur` variants actually separate in `blur_score`. Don't hand-tune floors against vibes.
- **A garbage photo passes** (e.g. a photo of a wall) → `quad_found` is probably the missing check; a document-boundary contour that never gets detected should itself be a `reject` reason, not just a lower `normalized_quality`.
- **`skew_deg` reads nonsense** (e.g. 90°) → the largest contour `minAreaRect` picked up wasn't the document — usually happens when the background has stronger edges than the paper itself (a patterned desk, a hand holding the clearance).
- **Quick isolation test:** `assess()` takes only a `np.ndarray`. No API key needed, no other module imported. If quality output looks wrong, this is the cheapest possible thing to test in isolation — see Testing Recipes below.

---

## Stage 4 — Preprocess (optional, config-gated)

**What happens:** If `config.OCR_PREPROCESS` is on and the verdict was `pass` or `warn`, the image gets deskewed, perspective-warped flat against the detected document boundary, contrast-enhanced (CLAHE), and upscaled toward `MIN_IMAGE_DIM_PX`. This step exists to help Gemini read it, and its on/off difference is the project's preprocessing ablation.

**The code:** `src/ocr/quality.py`, `preprocess(image) -> np.ndarray`.

**The data right now:** same shape, but `skew_deg` after preprocess should read near 0° if re-assessed — the image is now "flat."

**When it breaks:**
- Preprocessing makes things *worse* (rare but real) → perspective-warp based on a bad `quad_found` detection can crop into the text. If accuracy drops with preprocess on, check whether `quad_found` was reliable for that image before trusting the warp.
- This is genuinely optional per-call (`extract_document(..., preprocess=None)` defaults to config, but the eval harness overrides it with `--no-preprocess`) — if you're debugging and unsure whether preprocessing is the culprit, run both and diff.

---

## Stage 5 — Cache lookup

**What happens:** Before spending an API call, check whether this exact image (same bytes, same preprocess flag, same vision provider) has already been extracted. If yes, load the saved JSON and skip straight to Stage 7. This is what makes re-running the OCR eval free after the first pass.

**The code:** `src/ocr/extractor.py`, cache key = `sha256(original_bytes) + preprocess_flag + config.ACTIVE_VISION_MODEL`, stored at `config.OCR_CACHE_DIR/<key>.json`.

**The data right now:** cache miss (first time seeing this photo) → key `a3f9...e21c.json` doesn't exist yet. Proceed to Stage 6.

**When it breaks:**
- **`/usage` climbs on every run even though you're testing the "same" image** → the cache key changed underneath you. Usual cause: the preprocess step isn't deterministic (e.g. a random element snuck into `preprocess()`), or you're testing against slightly different crops/re-saves of "the same" photo that hash differently. Byte-identical input is required for a cache hit — resaving a JPEG at a different quality changes every byte.
- **Stale results after fixing the extraction prompt** → the cache key doesn't include a prompt version. If you change `NBI_EXTRACTION_PROMPT` and don't see new behavior, you're reading a stale cache entry from before the change — clear `evals/results/ocr_cache/` or add a prompt-version component to the key.

---

## Stage 6 — Layer 2: Gemini reads the photo

**What happens:** This is the one LLM call in the whole pipeline. **There is no separate "OCR to text, then parse" step** — the image goes to Gemini directly, and Gemini returns typed JSON in one shot. This is the detail that most differs from a classic OCR pipeline (Tesseract → raw text → regex), and it's worth sitting with: the model is not asked to read the image and then separately asked to structure what it read. One call does both, constrained by `response_schema` so the output is always parseable JSON, never free text.

**The code:** `src/ocr/extractor.py`:
```python
response = client.models.generate_content(
    model=config.ACTIVE_VISION_MODEL,
    contents=[
        types.Part.from_bytes(data=processed_bytes, mime_type="image/jpeg"),
        prompts.NBI_EXTRACTION_PROMPT.format(fields=...),
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ExtractionResult,
        temperature=0.0,
    ),
)
```

**The data right now** — what Gemini hands back:
```json
{
  "doc_type": "nbi_clearance",
  "full_name": {"value": "REYES, MARIA SANTOS", "verbatim_text": "REYES, MARIA SANTOS", "model_confidence": 0.97},
  "date_of_issue": {"value": "2026-02-14", "verbatim_text": "02/14/2026", "model_confidence": 0.95},
  "reference_no": {"value": "REYE900101-N00457821", "verbatim_text": "REYE900101-N00457821", "model_confidence": 0.93},
  "purpose": {"value": "Employment", "verbatim_text": "EMPLOYMENT", "model_confidence": 0.98},
  "overall_confidence": 0.94,
  "notes": ""
}
```
Notice `date_of_issue.value` is normalized to ISO-8601 while `verbatim_text` keeps the literal print — that split exists so a human reviewer can check a normalization without re-opening the photo.

**This result is a proposal, not a decision.** Nothing in `ExtractionResult` says `accepted` or `rejected` — that word doesn't exist in this schema on purpose (`CV_INTEGRATION.md` §1.2, §2.3).

**When it breaks:**
- **`response.parsed is None`** → the model didn't conform to the schema (rare on Gemini, much more common on the Ollama fallback with a small local model). The function returns `(None, report)`, which downstream always resolves to `needs_review`, never a crash.
- **A field comes back with `value: null` and low confidence** → the model is telling you honestly it couldn't read that field. This should happen sometimes on blurry-but-not-quite-`reject` photos — if it *never* happens, the model may be guessing instead of abstaining, worth spot-checking against `verbatim_text`.
- **`APIError` / `LLMBackendError`** → logged as a warning, function returns `(None, report)`. Check `GEMINI_API_KEY` is set, or that the 20-requests/day quota isn't exhausted (`GET /usage` shows today's count).
- **Isolation test:** feed `extract_document()` a known clean mock image directly (no API upload flow, no orchestrator) — see Testing Recipes.

---

## Stage 7 — Layer 3: the six validation rules

**What happens:** Six independent, deterministic checks run against the `ExtractionResult` from Stage 6. None of them involve the LLM again. Each produces a `RuleResult` (`{rule, passed, detail}`), and `detail` is contractually PII-free — it explains *what* failed, never *what value* was there.

**The code:** `src/guardrails/doc_validation.py`, `validate_document(extracted, quality, faculty_record, as_of=date(2026, 8, 8))`.

**The data right now** — walking each rule against Stage 6's output:

| Rule | Check | Result |
| --- | --- | --- |
| 1. Type match | `doc_type == nbi_clearance`? | ✅ pass |
| 2. Completeness | all 4 required fields non-empty? | ✅ pass |
| 3. Format | name ≥2 tokens non-numeric; date parses, not future; reference matches permissive pattern | ✅ pass |
| 4. Identity match | `rapidfuzz.token_set_ratio("REYES, MARIA SANTOS", faculty_record["full_name"])` ≥ `NAME_MATCH_THRESHOLD` (85)? | ✅ pass — 100 (exact match here) |
| 5. Validity window | `2026-02-14 + 6 months = 2026-08-14`; is `as_of` (2026-08-08) before that? | ✅ pass — **6 days from expiring**, worth noticing |
| 6. Fail-safe | quality `warn` and low composite? | N/A — quality was `pass` |

**The data right now — the decision:**
```json
{
  "outcome": "accepted",
  "composite_confidence": 0.93,
  "rules": [
    {"rule": "type_match", "passed": true, "detail": ""},
    {"rule": "completeness", "passed": true, "detail": ""},
    {"rule": "format", "passed": true, "detail": ""},
    {"rule": "identity", "passed": true, "detail": ""},
    {"rule": "validity_window", "passed": true, "detail": ""},
    {"rule": "fail_safe", "passed": true, "detail": ""}
  ],
  "message": "Your NBI Clearance has been verified and accepted."
}
```
`composite_confidence = min(0.93 quality, 0.94 extraction) = 0.93`, which clears `OCR_CONFIDENCE_FLOOR` (0.70) → `accepted`.

**When it breaks — this is the part worth understanding well, since it's the whole point of the component:**
- **Rule 4 fails** (name doesn't match, e.g. OCR read "REYES, MARIA S." and the faculty record says "REYES, MARIA SANTOS DELA CRUZ") → **`needs_review`, never `rejected`.** PH names are messy — middle initials, suffixes, surname-first ordering — a mismatch might be the OCR, might be the record, might be legitimately a different person. This rule is designed to escalate, not decide.
- **Rule 5 fails** (say `date_of_issue` were `2026-01-01` instead) → `2026-01-01 + 6 months = 2026-07-01`, which is *before* `as_of` 2026-08-08 → `rejected`. This is the one rule that rejects outright, because expiry is an unambiguous computed fact, not a judgment call. Remember: **6 months is an employer freshness policy** (`NBI_VALIDITY_MONTHS`), not the clearance's own printed one-year validity — don't confuse the two when explaining a rejection to an applicant.
- **Rule 3 fails on any field** → `composite_confidence` is forced to exactly `0.0`, regardless of how confident Gemini was. A deterministic format failure overrides model confidence entirely — this is the one line that answers "what stops the model from talking its way past a bad read."
- **Extraction was `None`** (Stage 6 failed) → Rule 6 fires, `needs_review`, generic "please allow time for manual review" message.

---

## Stage 8 — Checklist write

**What happens:** The outcome gets written to the applicant's onboarding record. **Only status, not values.**

**The code:** `src/memory/onboarding_status.py`, `record_result(employee_id, doc_type, outcome, source_hash)`.

**The data right now** — the SQLite row:
```
employee_id | doc_type      | status    | outcome  | validated_at         | source_hash
EMP-04821   | nbi_clearance | validated | accepted | 2026-08-08T09:14:02Z | a3f9...e21c
```
Notice what's **not** in this row: no name, no reference number, no date of issue. `source_hash` is the SHA-256 from Stage 5's cache key — it's the one thread connecting this row back to the actual image and cached extraction, and it's also the deletion key if consent is ever withdrawn.

**When it breaks:**
- Row never appears → check `record_result()` actually got called after Stage 7 (not skipped by an early return), and that `config.SQLITE_PATH` points where you think it does.
- Two rows for the same document → `PRIMARY KEY (employee_id, doc_type)` should make this a REPLACE, not an insert; check the SQL uses `INSERT OR REPLACE` / `ON CONFLICT`.

---

## Stage 9 — Response → UI

**What happens:** `POST /upload-doc` returns the `ValidationResult` plus a fresh `ChecklistStatus`. The UI updates the sidebar checklist panel and shows a chat-style confirmation bubble.

**The code:** `src/api.py`'s `upload_doc()` return value → `src/ui.py`'s checklist panel render, keyed off the same message-dict shape the chat flow already uses (`ui.py:457-467`).

**The data right now:**
```json
{
  "validation": {"outcome": "accepted", "composite_confidence": 0.93, "message": "Your NBI Clearance has been verified and accepted."},
  "checklist": {
    "employee_id": "EMP-04821",
    "documents": [{"doc_type": "nbi_clearance", "status": "validated", "outcome": "accepted"}],
    "missing": [],
    "faculty_class": "part_time"
  }
}
```

**When it breaks:** checklist panel doesn't update → confirm the UI is actually re-reading `GET /onboarding-status/{employee_id}` (or the inline response) after upload, not just showing stale session state from before the request.

---

## Stage 10 — A later chat turn asks about it

**What happens:** In a later message, the applicant asks *"is my NBI clearance okay?"* This is a completely separate turn through the normal ReAct agent loop — same guardrails, same router, same loop that answers Faculty Manual questions. **No vision call happens on this turn.** Everything needed is already sitting in SQLite from Stage 8.

**The code:** `src/agent/orchestrator.py`'s `run_turn()`:
```
1. _check_input()      → passes
2. classify_intent()   → DOCUMENT_STATUS, confidence 0.91
3. ReAct loop calls     get_onboarding_status(employee_id)
4. tool reads SQLite  → {"nbi_clearance": {"status": "validated", "outcome": "accepted"}}
5. model replies      → "Yes — your NBI Clearance has been verified and accepted.
                          Nothing further needed for that document."
```

**The data right now:** a plain-text SQLite read, then one chat-model call to phrase the reply. Zero calls to `GEMINI_VISION_MODEL`.

**When it breaks:**
- Router sends this to `FAQ`/`search_kb` instead of `DOCUMENT_STATUS` → this is the exact confusion PLAN.md §4.4 calls out ("asking about" vs. "submitting/discussing" a document); check `ROUTER_PROMPT`'s examples cover phrasings like "is my X okay/valid/good".
- Tool observation accidentally includes `full_name` or `reference_no` → this is the PII boundary described in `CV_INTEGRATION.md` §1.7. The tool's returned dict must never carry field values, because it flows straight into `src/memory/session.py`'s persisted transcript.

---

## Stage 11 — Observability

**What happens:** An MLflow run records the shape of what happened — never the content.

**The code:** a `doc_trace()` contextmanager sibling to `chat_trace()` (`src/monitoring.py`).

**The data right now:**
```
tags:    doc_type=nbi_clearance, validation_outcome=accepted, quality_verdict=pass, faculty_class=part_time
metrics: blur_score=340.5, skew_deg=3.1, extraction_confidence=0.94, fields_extracted=4, fields_missing=0, ocr_latency_ms=1840
```

**Deliberately absent:** `full_name`, `reference_no`, `date_of_issue`, `verbatim_text` — none of these are allow-listed keys, and the allowlist is fail-closed by construction (`monitoring.py:22-45`), so accidentally passing one in just gets silently dropped rather than logged.

**When it breaks:** if a PII value ever *does* show up in an MLflow tag, the bug is upstream of `monitoring.py` — someone added a key to the allowlist that shouldn't be there, or is bypassing `doc_trace()`'s dict entirely and calling `mlflow.set_tag()` directly somewhere. Grep for direct `mlflow.` calls outside `monitoring.py` if this ever happens.

---

## Follow the data

One row per stage — what the data *is* at that point, so you can eyeball where a shape went wrong.

| Stage | Data | Type/shape |
| --- | --- | --- |
| 1 | raw upload | HTTP multipart body, ~2.1MB |
| 2 | decoded pixels | `np.ndarray` (2560, 1920, 3) uint8, no metadata |
| 3 | quality report | `ImageQualityReport` — 5 floats/bools + verdict |
| 4 | preprocessed pixels | `np.ndarray`, same shape, deskewed/upscaled |
| 5 | cache key | `str`, sha256 hex + flags |
| 6 | extraction | `ExtractionResult` — 4 fields, each `{value, verbatim_text, confidence}` |
| 7 | validation | `ValidationResult` — outcome + 6 `RuleResult`s + composite float |
| 8 | checklist row | SQLite row — status/outcome/hash only, **no field values** |
| 9 | API response | `{validation, checklist}` JSON |
| 10 | chat reply | plain text, sourced from the Stage 8 row, not Stage 6 |
| 11 | MLflow run | tags/metrics dict, allow-listed keys only |

---

## Debugging playbook

| Symptom | Likely cause | Where to look |
| --- | --- | --- |
| Everything comes back `needs_review` | `OCR_CONFIDENCE_FLOOR` too high, or quality thresholds too tight | `config.py` thresholds; re-run the calibration CSV (Stage 3) |
| `/usage` climbing faster than expected | Cache key changing between "identical" runs | Confirm byte-identical input; check `preprocess()` for non-determinism |
| A genuinely good photo gets `reject`ed | `BLUR_VARIANCE_FLOOR` / `MAX_SKEW_DEG` mis-calibrated | Stage 3; dump signals for that specific image and compare to the calibration CSV |
| Name never matches, even for the right person | `NAME_MATCH_THRESHOLD` too strict, or PH-name normalization missing a case (suffixes, `ñ`, surname-first) | `doc_validation.py` Rule 4 normalization step |
| `extraction returns None` frequently | Schema conformance issue — much more likely on `VISION_PROVIDER=ollama` with a small local model | Stage 6; try the same image on `VISION_PROVIDER=gemini` to isolate provider vs. image |
| Quality gate passes an obviously bad photo | `quad_found` not actually gating anything, or its weight in `normalized_quality` is too small | Stage 3; check the `_verdict()` combination logic |
| A field value shows up somewhere it shouldn't (logs, MLflow, chat history) | A PII boundary was bypassed — likely a tool observation or a debug print carrying `.value` instead of just status | `CV_INTEGRATION.md` §1.7; grep the offending surface for `full_name.value` / `reference_no.value` |

---

## Testing recipes — cheapest first

Most of this pipeline is testable **without spending an API call**. Only Stage 6 costs quota.

1. **Quality gate alone** — no API key needed at all:
   ```python
   from src.ocr.quality import load_image, assess
   img = load_image(open("data/references/mock/id01_clean.png", "rb").read())
   print(assess(img))
   ```

2. **Validation rules alone** — hand-write an `ExtractionResult`, no image, no network:
   ```python
   from src.schemas import ExtractionResult, ExtractedField, DocType, ImageQualityReport, QualityVerdict
   from src.guardrails.doc_validation import validate_document
   from datetime import date

   extracted = ExtractionResult(
       doc_type=DocType.NBI_CLEARANCE,
       full_name=ExtractedField(value="REYES, MARIA SANTOS", model_confidence=0.97),
       date_of_issue=ExtractedField(value="2026-02-14", model_confidence=0.95),
       reference_no=ExtractedField(value="REYE900101-N00457821", model_confidence=0.93),
       purpose=ExtractedField(value="Employment", model_confidence=0.98),
       overall_confidence=0.94,
   )
   quality = ImageQualityReport(verdict=QualityVerdict.PASS, blur_score=340.5, exposure_clip=0.02,
                                 skew_deg=3.1, min_dim_px=1920, quad_found=True, normalized_quality=0.93)
   result = validate_document(extracted, quality, {"full_name": "REYES, MARIA SANTOS"}, as_of=date(2026, 8, 8))
   print(result.outcome)  # accepted
   ```
   Flip one field (wrong name, past-window date, empty reference) and watch the outcome change — this is the fastest way to understand the six rules without touching a real image.

3. **Extractor against the cache** — costs one real call the first time, then free:
   ```python
   from src.ocr.extractor import extract_document
   result, report = extract_document(open("...clean.png", "rb").read(), "image/png")
   ```
   Run it twice; the second run should hit the disk cache (`evals/results/ocr_cache/`) and log no API activity.

4. **Full HTTP round-trip** — the only test that exercises everything at once, so debug the earlier three first:
   ```bash
   uvicorn src.api:app --reload
   curl -F "file=@data/references/mock/id01_clean.png" -F "employee_id=EMP-04821" \
        http://localhost:8000/upload-doc
   ```
