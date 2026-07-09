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
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
MANIFEST_PATH = DATA_DIR / "index_manifest.json"
SQLITE_PATH = DATA_DIR / "hr_agent.db"

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
SIMILARITY_FLOOR = 0.5  # below this the agent must say "I don't know" (tuned in evals)

# Valid document categories; used for metadata-filtered retrieval after intent routing.
# "labor_law" has no internal chunks (no DOLE docs in data/raw/) — it's the signal
# the router uses to route straight to the search_web fallback instead of search_kb.
CATEGORIES = ("leave", "benefits", "payroll", "conduct", "complaints", "onboarding", "labor_law")

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
# Employee-ID format assumed for the PII guardrail — no convention exists
# elsewhere in this repo's data/schemas, so this is a documented invention:
# "EMP-" followed by 4-6 digits (e.g. EMP-00123).
EMPLOYEE_ID_PATTERN = r"\bEMP-\d{4,6}\b"
# PH mobile format (+639XXXXXXXXX or 09XXXXXXXXX), matching the PH-flavored
# HR content throughout data/raw/.
PHONE_PATTERN = r"(?:\+63|0)9\d{2}[-.\s]?\d{3}[-.\s]?\d{4}"

# --- Web search fallback (Module 8: Tool Use) ---
# search_web is restricted to these domains so it can't become a general-purpose
# search engine (would defeat the HR-only topic-filter guardrail). Enforced via
# Tavily's include_domains param at search time, not post-hoc filtering.
DOLE_ALLOWED_DOMAINS = ("dole.gov.ph", "officialgazette.gov.ph", "lawphil.net")
TAVILY_MAX_RESULTS = 5

# --- Monitoring ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", (DATA_DIR / "mlruns").as_uri())
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "hr-agent")


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
    Callers don't need to know which one they got."""
    if LLM_PROVIDER == "gemini":
        return get_gemini_client()
    if LLM_PROVIDER == "ollama":
        from src.agent.llm_client import OllamaClient

        return OllamaClient(OLLAMA_URL, OLLAMA_CHAT_MODEL)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}; expected 'gemini' or 'ollama'."
    )


def get_embedder():
    if EMBEDDING_PROVIDER == "gemini":
        from src.rag.embeddings import GeminiEmbedder
        return GeminiEmbedder()
    if EMBEDDING_PROVIDER == "ollama":
        from src.rag.embeddings import OllamaEmbedder
        return OllamaEmbedder()
    raise RuntimeError(f"Unknown EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}")