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
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
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
