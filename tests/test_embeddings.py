"""Tests for src/rag/embeddings.py's rate-limit retry (GeminiEmbedder), added
after embed_content's free-tier per-minute quota kept killing ingest.py
mid-run with no progress saved. time.sleep is monkeypatched throughout so
these run instantly, not for real seconds."""

from types import SimpleNamespace

import httpx
import pytest
from google.genai.errors import APIError

from src import config
from src.rag import embeddings as embeddings_mod
from src.rag.embeddings import GeminiEmbedder


def _rate_limit_error() -> APIError:
    return APIError(429, {"error": {"message": "RESOURCE_EXHAUSTED", "status": "RESOURCE_EXHAUSTED"}})


def _server_error() -> APIError:
    return APIError(500, {"error": {"message": "internal error"}})


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedContentResponse:
    def __init__(self, n: int, dim: int = 4):
        self.embeddings = [_FakeEmbedding([1.0] * dim) for _ in range(n)]


class _FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def embed_content(self, model, contents, config):
        self.call_count += 1
        result = self._responses[min(self.call_count, len(self._responses)) - 1]
        if isinstance(result, Exception):
            raise result
        return result


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(embeddings_mod.time, "sleep", lambda s: slept.append(s))
    return slept


@pytest.fixture(autouse=True)
def small_batch(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH_SIZE", 2)


def test_retries_on_429_then_succeeds(no_real_sleep):
    client = _FakeClient([_rate_limit_error(), _rate_limit_error(), _FakeEmbedContentResponse(2)])
    embedder = GeminiEmbedder(client=client)

    vectors = embedder.embed_documents(["a", "b"])

    assert len(vectors) == 2
    assert client.models.call_count == 3  # 2 failed attempts + 1 success
    assert len(no_real_sleep) == 2  # slept before each retry, not after the final success


def test_backoff_delay_increases_between_retries(no_real_sleep):
    client = _FakeClient([_rate_limit_error(), _rate_limit_error(), _FakeEmbedContentResponse(2)])
    GeminiEmbedder(client=client).embed_documents(["a", "b"])
    assert no_real_sleep[1] > no_real_sleep[0]  # exponential, not fixed


def test_gives_up_after_max_retries(no_real_sleep):
    client = _FakeClient([_rate_limit_error()] * 10)  # more than _EMBED_MAX_RETRIES
    embedder = GeminiEmbedder(client=client)

    with pytest.raises(APIError) as exc_info:
        embedder.embed_documents(["a", "b"])

    assert exc_info.value.code == 429
    assert client.models.call_count == embeddings_mod._EMBED_MAX_RETRIES


def test_non_429_error_raises_immediately_without_retry(no_real_sleep):
    client = _FakeClient([_server_error(), _FakeEmbedContentResponse(2)])
    embedder = GeminiEmbedder(client=client)

    with pytest.raises(APIError) as exc_info:
        embedder.embed_documents(["a", "b"])

    assert exc_info.value.code == 500
    assert client.models.call_count == 1  # no retry spent on a non-rate-limit error
    assert no_real_sleep == []


def test_multiple_batches_each_get_their_own_retry_budget(no_real_sleep):
    """Batch size 2, 4 texts -> 2 batches. First batch rate-limited once,
    second batch succeeds clean -- confirms retry state doesn't leak across
    batches and a transient limit on one batch doesn't poison the rest."""
    client = _FakeClient([
        _rate_limit_error(), _FakeEmbedContentResponse(2),  # batch 1: retry then succeed
        _FakeEmbedContentResponse(2),                        # batch 2: clean
    ])
    embedder = GeminiEmbedder(client=client)

    vectors = embedder.embed_documents(["a", "b", "c", "d"])

    assert len(vectors) == 4
    assert client.models.call_count == 3


def test_retries_on_timeout_then_succeeds(no_real_sleep):
    """Found 2026-08-10: a real embed_content call intermittently hung
    indefinitely with no client-side timeout configured (config.
    GEMINI_REQUEST_TIMEOUT_MS now sets one). A timeout is exactly as
    transient/retryable as a 429 here -- same backoff loop handles both."""
    client = _FakeClient([httpx.ReadTimeout("timed out"), _FakeEmbedContentResponse(2)])
    embedder = GeminiEmbedder(client=client)

    vectors = embedder.embed_documents(["a", "b"])

    assert len(vectors) == 2
    assert client.models.call_count == 2


def test_timeout_gives_up_after_max_retries(no_real_sleep):
    client = _FakeClient([httpx.ReadTimeout("timed out")] * 10)
    embedder = GeminiEmbedder(client=client)

    with pytest.raises(httpx.TimeoutException):
        embedder.embed_documents(["a", "b"])

    assert client.models.call_count == embeddings_mod._EMBED_MAX_RETRIES


def test_embed_query_also_retries(no_real_sleep):
    client = _FakeClient([_rate_limit_error(), _FakeEmbedContentResponse(1)])
    embedder = GeminiEmbedder(client=client)

    vector = embedder.embed_query("how many leave days?")

    assert len(vector) == 4
    assert client.models.call_count == 2
