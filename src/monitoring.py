"""MLflow tracing for the API layer (MLflow 3.x GenAI tracing).

Each chat turn is recorded as a single MLflow **trace** rooted at a `chat_turn`
span, replacing the per-turn nested-run approach used before the 3.x upgrade.
The span's own timing is the turn latency / agent response time; token usage and
sanitized counts are attached as span attributes, and searchable labels (route,
status, models, session) are set as trace tags.

PII invariant (unchanged): raw employee messages and model answers are **never**
logged. Only message *shape* (char/word counts), allowlisted numeric metrics, and
allowlisted string tags reach MLflow. The allowlists below are fail-closed —
anything a caller stuffs into ``trace_state["metrics"]`` / ``["tags"]`` that is
not named here is silently dropped rather than logged.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

from src import config

# Fail-closed allowlists (Guardrails + LLMOps). Only these keys ever leave the
# process as trace data.
#
# Numeric span attributes — the headline metrics: token usage, latency, and the
# retrieval/answer shape counts.
_ALLOWED_METRIC_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_ms",
        "agent_response_ms",
        "citation_count",
        "source_count",
        "web_citation_count",
        "action_count",
        # Component 14 (CV/OCR, Phase 6) — never a field value, only shape/score.
        "blur_score",
        "skew_deg",
        "extraction_confidence",
        "fields_extracted",
        "fields_missing",
        "ocr_latency_ms",
    }
)
# String trace tags — searchable/filterable labels in the MLflow Traces UI.
_ALLOWED_TAG_KEYS = frozenset(
    {
        "route",
        "insufficient_context",
        "status",
        "error_type",
        "session_id",
        "chat_model",
        "embedding_model",
        # Component 14 (CV/OCR, Phase 6) — classification labels only.
        "doc_type",
        "validation_outcome",
        "quality_verdict",
    }
)


def _safe_import_mlflow():
    try:
        import mlflow
    except Exception:
        return None
    return mlflow


_configured = False


def configure_mlflow() -> None:
    """Point MLflow at the configured store/experiment; keep the app usable if
    MLflow is unavailable. Idempotent — safe to call at startup and per turn."""
    global _configured
    if _configured:
        return
    mlflow = _safe_import_mlflow()
    if mlflow is None:
        return

    # A local file:// store is in maintenance mode under MLflow 3.x and raises
    # unless this opt-in is set; default it on so a fresh checkout traces
    # instead of 500-ing. An explicit env value (or a DB backend) still wins.
    if config.MLFLOW_TRACKING_URI.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    _configured = True


@contextmanager
def chat_trace(session_id: str, message: str) -> Iterator[dict[str, Any]]:
    """Record one chat turn as an MLflow trace.

    Yields a mutable ``trace_state`` dict; callers populate
    ``trace_state["metrics"]`` (numeric) and ``trace_state["tags"]`` (string)
    before the block exits, and both are filtered through the allowlists above.
    When MLflow is unavailable the same dict is yielded and nothing is logged,
    so callers never need to branch on monitoring being present.
    """
    trace_state: dict[str, Any] = {
        "metrics": {},
        "tags": {},
        "attributes": {
            "session_id": session_id,
            "message_chars": len(message),
            "message_words": len(message.split()),
        },
    }

    mlflow = _safe_import_mlflow()
    if mlflow is None:
        yield trace_state
        return

    configure_mlflow()
    started = perf_counter()
    status = "ok"
    with mlflow.start_span(name="chat_turn", span_type="AGENT") as span:
        # Sanitized inputs only — message shape, never its text.
        span.set_attributes(
            {
                **trace_state["attributes"],
                "chat_model": config.ACTIVE_CHAT_MODEL,
                "embedding_model": config.ACTIVE_EMBEDDING_MODEL,
                "top_k": config.TOP_K,
                "similarity_floor": config.SIMILARITY_FLOOR,
            }
        )
        try:
            yield trace_state
        except Exception as exc:
            status = "error"
            trace_state["tags"]["error_type"] = type(exc).__name__
            raise
        finally:
            # Span duration is the turn latency / agent response time; record it
            # explicitly too so it is queryable as a plain metric attribute.
            latency_ms = (perf_counter() - started) * 1000.0
            metrics = {
                key: value
                for key, value in trace_state.get("metrics", {}).items()
                if key in _ALLOWED_METRIC_KEYS and isinstance(value, (int, float))
            }
            metrics.setdefault("latency_ms", latency_ms)
            metrics.setdefault("agent_response_ms", latency_ms)
            span.set_attributes(metrics)

            tags = {
                key: str(value)
                for key, value in trace_state.get("tags", {}).items()
                if key in _ALLOWED_TAG_KEYS
            }
            tags["status"] = status
            tags["session_id"] = session_id
            mlflow.update_current_trace(tags=tags)


@contextmanager
def doc_trace(session_id: str, doc_type: str) -> Iterator[dict[str, Any]]:
    """Record one document-upload pipeline run (Component 14, Phase 6) as an
    MLflow trace. Same shape/contract as chat_trace — a mutable trace_state
    dict for metrics/tags, filtered through the same fail-closed allowlists,
    a no-op yield when MLflow is unavailable. Callers populate trace_state
    from ImageQualityReport/ValidationResult fields, never from an
    ExtractedField's .value."""
    trace_state: dict[str, Any] = {
        "metrics": {},
        "tags": {},
        "attributes": {"session_id": session_id, "doc_type": doc_type},
    }

    mlflow = _safe_import_mlflow()
    if mlflow is None:
        yield trace_state
        return

    configure_mlflow()
    started = perf_counter()
    status = "ok"
    with mlflow.start_span(name="doc_upload", span_type="AGENT") as span:
        span.set_attributes({**trace_state["attributes"], "vision_model": config.ACTIVE_VISION_MODEL})
        try:
            yield trace_state
        except Exception as exc:
            status = "error"
            trace_state["tags"]["error_type"] = type(exc).__name__
            raise
        finally:
            latency_ms = (perf_counter() - started) * 1000.0
            metrics = {
                key: value
                for key, value in trace_state.get("metrics", {}).items()
                if key in _ALLOWED_METRIC_KEYS and isinstance(value, (int, float))
            }
            metrics.setdefault("latency_ms", latency_ms)
            metrics.setdefault("ocr_latency_ms", latency_ms)
            span.set_attributes(metrics)

            tags = {
                key: str(value)
                for key, value in trace_state.get("tags", {}).items()
                if key in _ALLOWED_TAG_KEYS
            }
            tags["status"] = status
            tags["session_id"] = session_id
            tags["doc_type"] = doc_type
            mlflow.update_current_trace(tags=tags)


# --- Per-call LLM tracing -------------------------------------------------
#
# Wrapping the client factory (config.get_llm_client) makes every
# ``client.models.generate_content(...)`` call — router, each ReAct step,
# search_web, the answerer — a child span nested under the active ``chat_turn``
# trace, for *both* the Gemini and Ollama backends, with no changes at the call
# sites. Only sanitized attributes are recorded (provider, model, token usage,
# tool/schema flags) — never the prompt contents or the completion text, so the
# PII invariant holds here exactly as it does for the chat trace itself.


class _TracedModels:
    """Wraps a client's ``.models`` so ``generate_content`` opens a child span.

    The span is created only when a parent trace is already active (i.e. inside
    a chat turn); standalone LLM calls from ingest or the eval scripts pass
    straight through and never spawn a stray top-level trace.
    """

    def __init__(self, models, mlflow):
        self._models = models
        self._mlflow = mlflow

    def __getattr__(self, name):  # delegate everything else unchanged
        return getattr(self._models, name)

    def generate_content(self, *args, **kwargs):
        mlflow = self._mlflow
        if mlflow.get_current_active_span() is None:
            return self._models.generate_content(*args, **kwargs)

        cfg = kwargs.get("config", args[2] if len(args) >= 3 else None)
        model = kwargs.get("model", args[0] if args else None)
        schema = getattr(cfg, "response_schema", None)
        # The response schema's class name reveals the step for free
        # (IntentClassification=router, ReActStep=react planning, ...).
        step = getattr(schema, "__name__", None) or "generate_content"

        with mlflow.start_span(name=f"llm.{step}", span_type="LLM") as span:
            span.set_attributes(
                {
                    "provider": config.LLM_PROVIDER,
                    "model": str(model),
                    "has_tools": bool(getattr(cfg, "tools", None)),
                    "has_schema": schema is not None,
                }
            )
            response = self._models.generate_content(*args, **kwargs)
            meta = getattr(response, "usage_metadata", None)
            if meta is not None:
                span.set_attributes(
                    {
                        "prompt_tokens": getattr(meta, "prompt_token_count", 0) or 0,
                        "completion_tokens": getattr(meta, "candidates_token_count", 0) or 0,
                        "total_tokens": getattr(meta, "total_token_count", 0) or 0,
                    }
                )
            return response


class _TracedLLMClient:
    """Thin proxy over an LLM client; only ``.models`` is instrumented."""

    def __init__(self, client, mlflow):
        self._client = client
        self.models = _TracedModels(client.models, mlflow)

    def __getattr__(self, name):
        return getattr(self._client, name)


def trace_llm_client(client):
    """Wrap an LLM client so its ``generate_content`` calls become child spans.

    Returns the client unchanged when MLflow is unavailable, so callers never
    need to branch on monitoring being present.
    """
    mlflow = _safe_import_mlflow()
    if mlflow is None or client is None:
        return client
    return _TracedLLMClient(client, mlflow)
