"""Gemini embeddings (Stage 4 ingest-side, Stage 5 query-side).

task_type matters: RETRIEVAL_DOCUMENT at ingest, RETRIEVAL_QUERY at query time.
gemini-embedding-001 vectors truncated below 3072 dims are not unit-length, so
we re-normalize before storing/querying (cosine space in Chroma).
"""

import math

from src import config


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


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
                model=config.EMBEDDING_MODEL,
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
