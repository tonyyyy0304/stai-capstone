"""Central configuration: model names, paths, retrieval and chunking parameters.

Every tunable lives here (per CLAUDE.md) so evals can sweep them without
touching module code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
# The Faculty Manual is the primary corpus, but it's ingested as an *explicit*
# source rather than from data/raw/ so the fast raw-corpus unit test doesn't
# re-parse a 203-page PDF on every run. scripts/ingest.py globs data/raw/ AND
# appends any of these that exist. Each still needs a sibling <name>.meta.yaml.
MANUAL_SOURCES = (DATA_DIR / "faculty-manual-2021.pdf",)
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
MANIFEST_PATH = DATA_DIR / "index_manifest.json"
SQLITE_PATH = DATA_DIR / "hr_agent.db"

# --- Deployment profile (GENERALIZATION) ---
# Everything org/corpus-specific is centralized here so retargeting to another
# university or a company is a config + data change, not a code change. Prompts,
# decline messages, and the UI/API branding are all built from these.
ORG_NAME = os.environ.get("ORG_NAME", "De La Salle University")
ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "Faculty Onboarding Concierge")
CORPUS_TITLE = os.environ.get("CORPUS_TITLE", "DLSU Faculty Manual 2021")
# Who to route to when the assistant can't help (decline/abstain/error copy).
HELP_CONTACT = os.environ.get("HELP_CONTACT", "your college's HR office")
# One noun phrase describing the assistant's scope, used in scope/decline copy.
SCOPE_PHRASE = os.environ.get(
    "SCOPE_PHRASE",
    "faculty onboarding, pre-employment requirements, and the DLSU Faculty Manual",
)
# The reader the assistant serves ("faculty member", "employee", "new hire").
READER_NOUN = os.environ.get("READER_NOUN", "faculty member")

# --- API / UI ---
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
API_URL = os.environ.get("API_URL", f"http://localhost:{API_PORT}")

# --- Models ---
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "gemini")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
ACTIVE_EMBEDDING_MODEL = GEMINI_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "gemini" else f"ollama:{OLLAMA_EMBEDDING_MODEL}"
EMBEDDING_DIM = 768  # via output_dimensionality; vectors are re-normalized after truncation
EMBED_BATCH_SIZE = 64

# --- Vector store ---
COLLECTION_NAME = "hr_policies"

# --- Chunking (Stage 2 of the ingestion pipeline) ---
CHUNK_TARGET_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
CHUNK_MIN_TOKENS = 80  # sections smaller than this are merged into their parent

# --- Retrieval (Stage 5) ---
TOP_K = 8
# Below this the agent must say "I don't know" instead of answering. Tuned via a
# floor sweep over the golden set (evals/run_retrieval_eval.py): the lowest
# genuinely-answerable question retrieves at ~0.63, so 0.55 stays safely below
# every known-good answer (never suppresses recall) while still rejecting clearly
# irrelevant sub-0.55 matches. NOTE: the floor is NOT the abstention lever for
# near-topic negatives (e.g. "NBI fee in pesos", "HRMO office hours") — those
# retrieve topically-adjacent chunks at 0.68–0.70, above real answers, so no safe
# floor catches them. Abstaining on those is the generation layer's job (the model
# sets insufficient_context when the excerpts don't contain the fact); a
# deterministic faithfulness check is the Phase 3 hardening for it.
SIMILARITY_FLOOR = 0.55

# --- Advanced RAG: hybrid retrieval (PLAN.md §3.4) ---
# "dense" = cosine-only over Chroma (default, unchanged behavior).
# "hybrid" = dense + BM25 (SQLite FTS5) fused with Reciprocal Rank Fusion. Dense
# embeddings blur the exact identifiers this corpus is full of ("BIR Form 1902",
# "Assistant Professor", "p.24"); BM25 catches those at zero extra LLM cost.
RETRIEVER_MODE = os.environ.get("RETRIEVER_MODE", "dense")
# RRF constant: score(chunk) = Σ_r 1 / (RRF_K + rank_r(chunk)). Larger K flattens
# the contribution of top ranks; 60 is the value from the original RRF paper.
RRF_K = int(os.environ.get("RRF_K", "60"))
# How many candidates each retriever contributes to the fusion pool before the
# top-k cut. Wider than TOP_K so a chunk ranked well by one retriever but missed
# by the other still enters the fusion.
RRF_CANDIDATE_POOL = int(os.environ.get("RRF_CANDIDATE_POOL", "20"))
BM25_SQLITE_PATH = DATA_DIR / "bm25.sqlite"

# Valid *document-level* categories (validated at ingest). Category is a soft
# retrieval signal (see CATEGORY_BOOST), not a hard filter, and it's now the weak
# lever — audience_class carries the real disambiguation weight (Phase 2). The
# Midterm-era values (payroll, complaints, labor_law) are removed. "faculty_manual"
# is the Manual's own doc label; it's never a query-side category (a user question
# isn't "faculty_manual"), so the Manual competes on pure similarity + audience_class.
CATEGORIES = (
    "onboarding",
    "conduct",
    "leave",
    "benefits",
    "faculty_manual",
)
# The subset the router/search_kb may predict from a user question. Excludes
# faculty_manual (a doc label, not a user-facing topic) — a query about leaves in
# the Manual is category "leave", not "faculty_manual".
QUERY_CATEGORIES = ("onboarding", "conduct", "leave", "benefits")
# Soft category re-rank: a retrieved chunk whose category matches the query's
# category gets this added to its *ordering* score (not its stored similarity, so
# the similarity floor still sees true cosine). Small, so it only breaks ties /
# nudges near-equal chunks — it never excludes a relevant off-category chunk the
# way the old hard $eq filter did.
CATEGORY_BOOST = 0.05

# --- Audience segmentation (the corpus's biggest retrieval hazard) ---
# GENERALIZATION: a corpus often carries near-duplicate policy text for different
# sub-populations ("audience classes") with the SAME structure but DIFFERENT rules
# — for DLSU, the three faculty classes (full-time 8.x leaves vs ASF 6.x leaves);
# for a company, employment types (regular / probationary / contractor). Handing a
# reader another segment's rules is a confident wrong answer, so chunks are tagged
# with an audience_class slug and the agent asks which segment the reader is when
# the evidence spans more than one. This taxonomy is the ONLY place the segments
# are defined — swap it (and the manual's markers) to retarget, no code change.
#   slug:     stable id stored in chunk metadata ("" = applies to all segments)
#   label:    display name (chunk context header + clarifying question)
#   markers:  UPPERCASE heading substrings that begin this segment's region in the
#             manual (positional detection; see rag/chunking.detect_section_audience)
#   keywords: lowercase phrases in a user message that signal they stated this segment
AUDIENCE_CLASSES = (
    {
        "slug": "full_time_academic",
        "label": "Full-time Academic Faculty",
        "markers": ("FULL-TIME ACADEMIC FACULTY",),
        "keywords": ("full-time", "full time", "fulltime"),
    },
    {
        "slug": "part_time_academic",
        "label": "Part-time Academic Faculty",
        "markers": ("PART-TIME ACADEMIC FACULTY",),
        "keywords": ("part-time", "part time", "parttime"),
    },
    {
        "slug": "academic_service",
        "label": "Academic Service Faculty",
        "markers": ("ACADEMIC SERVICE FACULTY",),
        "keywords": ("academic service", "asf"),
    },
)
# Heading substrings that END audience-region tracking (content applies to all
# segments thereafter): appendices, shared annexes, etc.
AUDIENCE_RESET_MARKERS = ("APPENDIX",)
# What to call the segmentation in the clarifying question ("faculty class",
# "employment type", "membership tier", …).
AUDIENCE_NOUN = os.environ.get("AUDIENCE_NOUN", "faculty class")
# Derived lookups (do not edit — computed from AUDIENCE_CLASSES).
AUDIENCE_LABELS = {c["slug"]: c["label"] for c in AUDIENCE_CLASSES}
AUDIENCE_ORDER = tuple(c["slug"] for c in AUDIENCE_CLASSES)
# Soft re-rank weight when the reader's stated segment matches a chunk's segment —
# same mechanism/rationale as CATEGORY_BOOST (ordering only, floor sees true cosine).
AUDIENCE_CLASS_BOOST = 0.05
# Class disambiguation only fires when >1 class appears among the top-N retrieved
# chunks (the *strong* evidence), not anywhere in top-k. A genuine class-split
# question puts both classes at the very top (leave, permanency); an unanswerable
# question that merely brushes a class-specific section by lexical overlap (e.g.
# "HRMO office hours" grazing "Working Hours") has the second class ranked lower —
# so it correctly falls through to a normal "I don't know" instead of a bogus
# "which class are you?".
DISAMBIG_TOP_N = 3

# --- Agent (Module 7: ReAct Agent) ---
MAX_REACT_ITERATIONS = 5
ROUTER_CONFIDENCE_FLOOR = 0.6  # below this, treat as ambiguous and ask a clarifying question

# --- LLM backend selection ---
# "gemini" (default, unchanged behavior) or "ollama" (self-hosted, for testing
# without Gemini's free-tier daily request cap). Chat/reasoning only — RAG
# embeddings and grounded-answer generation stay on Gemini regardless.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma4:e4b")
GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.1-flash-lite")
ACTIVE_CHAT_MODEL = GEMINI_CHAT_MODEL if LLM_PROVIDER == "gemini" else f"ollama:{OLLAMA_CHAT_MODEL}"

# Gemini's model-internal "thinking" tokens are billed as output and add latency
# to every call, and they surface a `thought_signature` non-text part that trips
# the SDK's `.text` accessor. This project's ReAct loop lives at the orchestration
# layer (the explicit `thought` field in ReActStep), so the model-internal trace
# buys us little on the structured planning/routing/judging calls. Default off (0).
# Set to -1 for dynamic (the model decides per-question) or a positive int to cap
# thinking tokens. Only meaningful on Gemini; the Ollama client ignores it.
GEMINI_THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "0"))

# --- Memory (Module 5) ---
# Short-term in-context window; full history still persists in SQLite
# (src/memory/session.py) regardless of this trim. PLAN.md §4 originally
# scoped "summarization past ~20 turns" — kept that number.
MEMORY_TRIM_TURNS = 20
# Don't re-summarize on every single turn once the window overflows (that
# would be an LLM call per turn) - batch evictions and only summarize once
# this many turns have fallen out of the window since the last summary.
MEMORY_SUMMARY_BATCH_SIZE = 5

# --- Guardrails (Module 6) ---
# Small, documented wordlist - this is an internal HR tool (authenticated
# employees), not public-facing, so the bar is catching blatant abuse aimed
# at the bot/HR staff, not comprehensive content moderation (PLAN.md §8
# explicitly flags scope creep as a risk).
TOXIC_WORDLIST = (
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "asshole",
    "bastard",
    "cunt",
    "whore",
    "retard",
    "retarded",
)
# --- LLM-as-Judge input guardrail (Module 6) ---
# Second, defense-in-depth input guardrail that always runs after the cheap
# deterministic checks: one structured Gemini call classifies each incoming
# message for toxicity / PII / prompt-injection / off-topic / jailbreak at once,
# catching nuanced adversarial phrasing the deterministic wordlist+regex layer
# (toxicity.py, input_checks.py) misses. The deterministic checks run first and
# short-circuit blatant cases for free (PLAN.md §2.1/§8 — Gemini quota is the #1
# constraint), so the judge only spends a call on messages that got past them.
# Below this confidence, the judge's flags are treated as too weak to block
# (fail-open toward the employee rather than blocking a legitimate question).
LLM_JUDGE_CONFIDENCE_FLOOR = 0.6
# Phase 5 (quota): the separate LLM-as-judge input call is folded into the router
# by default — the router already returns is_toxic/is_injection_attempt/is_jailbreak
# in the same call, so a second LLM safety pass per turn is redundant. Set true to
# re-enable the judge as an extra defense-in-depth layer (one more call/turn).
ENABLE_LLM_JUDGE = os.environ.get("ENABLE_LLM_JUDGE", "false").lower() == "true"
# Which detected violations actually block entry. PII is intentionally excluded —
# it's detected for redaction/observability, not rejection (see pii.py); off_topic
# stays authoritative at the router (input_checks.py), the judge just catches it
# one call earlier.
LLM_JUDGE_BLOCKING_VIOLATIONS = ("toxicity", "injection", "off_topic", "jailbreak")

# Employee-ID format assumed for the PII guardrail — no convention exists
# elsewhere in this repo's data/schemas, so this is a documented invention:
# "EMP-" followed by 4-6 digits (e.g. EMP-00123).
EMPLOYEE_ID_PATTERN = r"\bEMP-\d{4,6}\b"
# PH mobile format (+639XXXXXXXXX or 09XXXXXXXXX), matching the PH-flavored
# HR content throughout data/raw/.
PHONE_PATTERN = r"(?:\+63|0)9\d{2}[-.\s]?\d{3}[-.\s]?\d{4}"

# --- Web search fallback (Module 8: Tool Use) ---
# GENERALIZATION: the statutory web fallback is PH/university-specific (national
# agencies below). A deployment that doesn't need it — most companies — sets this
# false and the search_web tool is not offered to the agent at all.
ENABLE_WEB_FALLBACK = os.environ.get("ENABLE_WEB_FALLBACK", "true").lower() == "true"
# Ground web results in a specific country (Tavily `country` param boosts sources
# from it). Kept in the deployment profile so a non-PH deployment retargets it.
SEARCH_COUNTRY = os.environ.get("SEARCH_COUNTRY", "philippines")
# search_web is restricted to these domains so it can't become a general-purpose
# search engine (would defeat the on-topic guardrail). Enforced via Tavily's
# include_domains param at search time, not post-hoc filtering. These are the
# official sources for the national statutory pre-employment agencies the fallback
# actually covers (NBI/SSS/PhilHealth/Pag-IBIG/BIR) plus DOLE/gazette/lawphil for
# labor-law text. (Previously only the three DOLE-side domains were allowed, so
# the fallback could never reach the agency sites its own prompt names.)
STATUTORY_GOV_DOMAINS = (
    "nbi.gov.ph",
    "sss.gov.ph",
    "philhealth.gov.ph",
    "pagibigfund.gov.ph",
    "bir.gov.ph",
    "dole.gov.ph",
    "officialgazette.gov.ph",
    "lawphil.net",
)
TAVILY_MAX_RESULTS = 5

# --- Monitoring ---
# SQLite backend (MLflow 3.x): a single file, no server needed, and unlike the
# legacy file store it supports the trace-metrics aggregates the Traces UI draws.
# The old file store lives at data/mlruns (kept as a backup, migrated via
# `mlflow migrate-filestore`).
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(DATA_DIR / 'mlflow.db').as_posix()}"
)
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "hr-agent")

# --- CV/OCR document verification (Component 14) ---
# Gemini is the default and the evaluated path — every threshold, cache entry,
# and eval number assumes it. VISION_PROVIDER mirrors the LLM_PROVIDER/
# EMBEDDING_PROVIDER switch above so the same quota-relief pattern applies to
# vision: switchable, not blended, and NOT derived from LLM_PROVIDER —
# GEMINI_CHAT_MODEL is a lite tier tuned for quota, not multimodal
# extraction, so "vision follows chat" would be wrong even before quota
# enters the picture. See CV_INTEGRATION.md §2.2/§2.6a.
VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "gemini")
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")
ACTIVE_VISION_MODEL = (
    GEMINI_VISION_MODEL if VISION_PROVIDER == "gemini" else f"ollama:{OLLAMA_VISION_MODEL}"
)

# References folder (renamed from the original onboarding_docs/ during the
# CV rework): mock + real datasets live here, plus a samples/ subfolder for
# gitignored layout-reference specimen images (data/references/samples/).
REFERENCES_DIR = DATA_DIR / "references"
MOCK_DOCS_DIR = REFERENCES_DIR / "mock"
REAL_DOCS_DIR = REFERENCES_DIR / "real"
OCR_CACHE_DIR = REPO_ROOT / "evals" / "results" / "ocr_cache"

# Image quality gate (src/ocr/quality.py). PROVISIONAL — calibrate against the
# mock dataset (data/references/mock/) before trusting these in an eval;
# see the calibration procedure in CV_INTEGRATION.md §2.4 / Phase 2.
BLUR_VARIANCE_FLOOR = 100.0     # variance of Laplacian; below = reject
BLUR_VARIANCE_WARN = 250.0      # below = warn
MAX_SKEW_DEG = 12.0             # beyond = reject
SKEW_WARN_DEG = 5.0
# Recalibrated 2026-08-09 against a real 768x518 specimen (shorter side 518)
# that Gemini extracted at 0.98-0.99 confidence on every field once the old
# MIN_IMAGE_DIM_PX=640 floor was bypassed to test it -- 640 was a Phase 0
# placeholder never actually checked against a real document, only against
# the mock dataset's synthetic clean-vs-lowres_jpeg gap (~1000px vs
# ~300-500px), which said nothing about where real legibility breaks down.
# Floor dropped well below the one confirmed-working sample (margin, not a
# fit to n=1); old value demoted to a warn-only threshold, mirroring the
# blur/skew floor+warn pattern. Revisit as more real samples arrive.
MIN_IMAGE_DIM_PX = 400          # shorter side, px; below = reject
MIN_IMAGE_DIM_WARN = 640        # below = warn
EXPOSURE_CLIP_CEILING = 0.10    # fraction of pixels at 0 or 255 before warn
# Recalibrated 2026-08-10: was a hardcoded 0.5 in quality.py, "not a hard
# fail alone, but a real confidence hit" -- never checked against real data.
# 3/3 real specimens tested (nbi-clearance-real.WEBP, real2.jpg, real3.jpg)
# hit quad_found=False, and since normalized_quality = min(scores), 0.5
# became the dominant score every time -- capping composite_confidence at
# 0.5 regardless of extraction quality, which is below OCR_CONFIDENCE_FLOOR
# unconditionally. Real photos never produced the clean rectangular contour
# the mock dataset's synthetic renders trivially do, so this one signal was
# silently forcing every real submission to needs_review no matter how
# confident the extraction was. Softened, not removed -- still a real
# confidence hit, just no longer alone enough to guarantee sub-floor.
QUAD_NOT_FOUND_QUALITY_SCORE = 0.75
OCR_PREPROCESS = True           # ablated off via --no-preprocess in the eval

# One retry when the model itself signals genuine uncertainty (low
# overall_confidence, or a specific field null at low confidence) -- NOT a
# blanket retry on every missing optional field, since some are legitimately
# absent by design (e.g. National ID's expiry_date -- retrying that would
# just waste a call on a field that's never going to be there). Added
# 2026-08-09 after a real document's middle_name came back null at 0.1
# confidence, then correct at 0.99 confidence on an immediate identical
# re-call of the same image/prompt -- Gemini's temperature=0.0 reduces but
# does not guarantee run-to-run determinism. Costs at most one extra API
# call per extraction, and only on an already-live-call path (a cache hit
# never reaches this code at all). See CV_INTEGRATION.md Phase 4 note.
OCR_ENABLE_RETRY = os.environ.get("OCR_ENABLE_RETRY", "true").lower() == "true"
OCR_RETRY_CONFIDENCE_FLOOR = 0.5

# Extraction + validation (src/ocr/extractor.py, src/guardrails/doc_validation.py)
OCR_CONFIDENCE_FLOOR = 0.70     # composite below this -> needs_review
NAME_MATCH_THRESHOLD = 85       # rapidfuzz token_set_ratio, 0-100
NBI_VALIDITY_MONTHS = 6         # EMPLOYER freshness policy, layered on top of
                                # (not instead of) the document's own printed
                                # valid_until — Rule 5 takes whichever is
                                # stricter. See PLAN.md §4.1 Rule 5, CV_INTEGRATION.md §2.7.
# Normalized clean-status strings for the `remarks` field (Rule 3). Deliberately
# incomplete and tunable — extend as real samples show more phrasing variants.
# Anything NOT in this tuple fails Rule 3 outright; never assumed clean by default.
# "NO RECORD ON FILE" added 2026-08-09: a second real specimen
# (data/references/real/real2.jpg) prints exactly this phrasing, confirmed via
# its cached extraction (model_confidence 0.99-1.0 across two live calls) —
# would otherwise fail Rule 3 on a genuinely clean document.
NBI_CLEAN_REMARKS = ("NO DEROGATORY", "NO DEROGATORY RECORD", "NO RECORD", "NO RECORD ON FILE")
# Government ID (CV_INTEGRATION.md §1.4a) — one required id_number pattern per
# sub-type, each from exactly one specimen sample, none confirmed. See doctypes.py.
ID_NUMBER_PATTERNS = {
    "national_id": r"^\d{4}-\d{4}-\d{4}-\d{4}$",
    "drivers_license": r"^[A-Z]\d{2}-\d{2}-\d{6}$",
    "passport": r"^[A-Z]\d{7}[A-Z]$",
}
REQUIRED_ONBOARDING_DOCS = ("nbi_clearance", "government_id")

# Upload handling (POST /upload-doc)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_MIME = ("image/jpeg", "image/png")   # PDF is a stretch goal
PERSIST_UPLOADS = False         # don't keep raw images past the request


def get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def get_gemini_client():
    """Create a google-genai client. Kept as a function so tests can mock it."""
    from google import genai

    return genai.Client(api_key=get_gemini_api_key())


def get_tavily_api_key() -> str:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def get_tavily_client():
    """Create a Tavily client. Kept as a function so tests can mock it."""
    from tavily import TavilyClient

    return TavilyClient(api_key=get_tavily_api_key())


def get_llm_client():
    """Returns the active chat/reasoning client per LLM_PROVIDER: either a
    real google-genai Client (default) or an OllamaClient adapter exposing
    the same .models.generate_content(model, contents, config) interface.
    Callers don't need to know which one they got. The client is wrapped so
    each generate_content call is traced as a child span when a chat turn is
    active (src/monitoring.py); the wrapper is transparent otherwise."""
    from src.monitoring import trace_llm_client  # lazy: monitoring imports config

    if LLM_PROVIDER == "gemini":
        return trace_llm_client(get_gemini_client())
    if LLM_PROVIDER == "ollama":
        from src.agent.llm_client import OllamaClient

        return trace_llm_client(OllamaClient(OLLAMA_URL, OLLAMA_CHAT_MODEL))
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}; expected 'gemini' or 'ollama'."
    )


def get_vision_client():
    """Returns the active vision client per VISION_PROVIDER: a real
    google-genai Client (default) or an OllamaClient. Both expose
    .models.generate_content(model, contents, config); callers don't need to
    know which one they got — same contract as get_llm_client(). Deliberately
    its own switch, not derived from LLM_PROVIDER: vision and chat can be on
    different providers at once (e.g. LLM_PROVIDER=ollama for cheap chat
    testing, VISION_PROVIDER=gemini for reliable extraction).

    NOTE: unlike get_llm_client(), this does not go through
    trace_llm_client() — vision calls aren't yet wrapped in the MLflow
    child-span tracing get_llm_client() added. Worth revisiting so OCR calls
    show up in the same traces (flagged during the final-capstone merge,
    2026-08-09), not done here since it's outside the scope of the merge."""
    if VISION_PROVIDER == "gemini":
        return get_gemini_client()
    if VISION_PROVIDER == "ollama":
        from src.agent.llm_client import OllamaClient

        return OllamaClient(OLLAMA_URL, OLLAMA_VISION_MODEL)
    raise RuntimeError(
        f"Unknown VISION_PROVIDER={VISION_PROVIDER!r}; expected 'gemini' or 'ollama'."
    )


def thinking_config():
    """A google-genai ThinkingConfig carrying GEMINI_THINKING_BUDGET, to pass as
    GenerateContentConfig(thinking_config=...). Centralizes the budget so it's set
    in one place. The Ollama client reads config via getattr and ignores it."""
    from google.genai import types

    return types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET)


def get_embedder():
    if EMBEDDING_PROVIDER == "gemini":
        from src.rag.embeddings import GeminiEmbedder
        return GeminiEmbedder()
    if EMBEDDING_PROVIDER == "ollama":
        from src.rag.embeddings import OllamaEmbedder
        return OllamaEmbedder()
    raise RuntimeError(f"Unknown EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}")