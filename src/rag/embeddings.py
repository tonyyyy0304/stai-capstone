"""Gemini embeddings (Stage 4 ingest-side, Stage 5 query-side).

task_type matters: RETRIEVAL_DOCUMENT at ingest, RETRIEVAL_QUERY at query time.
gemini-embedding-001 vectors truncated below 3072 dims are not unit-length, so
we re-normalize before storing/querying (cosine space in Chroma).
"""

import math
from typing import Protocol

from src import config


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

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
            batch = texts[start : start + config.EMBED_BATCH_SIZE]
            response = self.client.models.embed_content(
                model=config.GEMINI_EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=config.EMBEDDING_DIM,
                ),
            )
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