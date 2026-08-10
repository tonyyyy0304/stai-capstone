"""Gemini embeddings (Stage 4 ingest-side, Stage 5 query-side).

task_type matters: RETRIEVAL_DOCUMENT at ingest, RETRIEVAL_QUERY at query time.
gemini-embedding-001 vectors truncated below 3072 dims are not unit-length, so
we re-normalize before storing/querying (cosine space in Chroma).
"""

import logging
import math
import time
from typing import Protocol

from src import config

logger = logging.getLogger(__name__)

# embed_content's free tier is rate-limited per MINUTE, not per day (the
# error is literally "EmbedContentRequestsPerMinutePerUserPerProjectPerModel-
# FreeTier") -- a corpus needing more than ~100 batches re-embedded in one
# ingest run trips it, and without a retry, ingest just dies mid-run with no
# progress saved (the manifest is only written at the very end). Backoff
# turns that into "ingest pauses and finishes" instead of "crash, and the
# retry hits the exact same wall again."
_EMBED_MAX_RETRIES = 6
_EMBED_BASE_DELAY_S = 10.0


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class Embedder(Protocol):
    """Structural type both embedders satisfy — lets Retriever/ingest.py accept
    either without importing a concrete class."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class GeminiEmbedder:
    def __init__(self, client=None):
        self._client = client  # created lazily so tests can inject a fake

    @property
    def client(self):
        if self._client is None:
            self._client = config.get_gemini_client()
        return self._client

    def _embed_batch_with_retry(self, batch: list[str], task_type: str):
        from google.genai import types
        from google.genai.errors import APIError

        for attempt in range(_EMBED_MAX_RETRIES):
            try:
                return self.client.models.embed_content(
                    model=config.GEMINI_EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=config.EMBEDDING_DIM,
                    ),
                )
            except APIError as exc:
                is_last_attempt = attempt == _EMBED_MAX_RETRIES - 1
                if getattr(exc, "code", None) != 429 or is_last_attempt:
                    raise
                delay = _EMBED_BASE_DELAY_S * (2**attempt)
                logger.warning(
                    "embed_content rate-limited, retrying in %.0fs (attempt %d/%d)",
                    delay, attempt + 1, _EMBED_MAX_RETRIES,
                )
                print(f"  embed_content rate-limited, retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{_EMBED_MAX_RETRIES})...")
                time.sleep(delay)

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
            batch = texts[start : start + config.EMBED_BATCH_SIZE]
            response = self._embed_batch_with_retry(batch, task_type)
            vectors.extend(_normalize(e.values) for e in response.embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]



class OllamaEmbedder:
    def __init__(self, client=None):
        self._base_url = config.OLLAMA_URL
        self._model = config.OLLAMA_EMBEDDING_MODEL
        self._client = client  # created lazily so tests can inject a fake

    @property
    def client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self._base_url)
        return self._client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        from ollama import ResponseError

        vectors: list[list[float]] = []
        for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
            batch = texts[start : start + config.EMBED_BATCH_SIZE]
            try:
                response = self.client.embed(model=self._model, input=batch)
            except ResponseError as exc:
                from src.agent.llm_client import LLMBackendError

                raise LLMBackendError(
                    f"Ollama embed failed: {exc}", code=getattr(exc, "status_code", 503)
                ) from exc
            vectors.extend(response.embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]