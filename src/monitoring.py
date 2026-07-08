"""MLflow instrumentation helpers for the API layer.

The monitoring module deliberately avoids logging raw employee messages or model
answers. Complaint text can contain PII, so traces store request shape, latency,
tool/action names, source counts, and error metadata only.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

from src import config


def _safe_import_mlflow():
    try:
        import mlflow
    except Exception:
        return None
    return mlflow


def configure_mlflow() -> None:
    """Configure MLflow if available; keep the app usable when it is not."""
    mlflow = _safe_import_mlflow()
    if mlflow is None:
        return
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


@contextmanager
def chat_trace(session_id: str, message: str) -> Iterator[dict[str, Any]]:
    """Record sanitized request telemetry around one chat turn."""
    mlflow = _safe_import_mlflow()
    trace_state: dict[str, Any] = {
        "started_at": perf_counter(),
        "attributes": {
            "session_id": session_id,
            "message_chars": len(message),
            "message_words": len(message.split()),
        },
    }
    if mlflow is None:
        yield trace_state
        return

    configure_mlflow()
    with mlflow.start_run(run_name="chat_turn", nested=True):
        mlflow.set_tags(
            {
                "component": "api",
                "session_id": session_id,
                "chat_model": config.ACTIVE_CHAT_MODEL,
                "embedding_model": config.ACTIVE_EMBEDDING_MODEL,
            }
        )
        mlflow.log_params(
            {
                "message_chars": trace_state["attributes"]["message_chars"],
                "message_words": trace_state["attributes"]["message_words"],
                "top_k": config.TOP_K,
                "similarity_floor": config.SIMILARITY_FLOOR,
            }
        )
        try:
            yield trace_state
        except Exception as exc:
            mlflow.set_tag("status", "error")
            mlflow.log_param("error_type", type(exc).__name__)
            raise
        finally:
            latency_ms = (perf_counter() - trace_state["started_at"]) * 1000
            mlflow.log_metric("latency_ms", latency_ms)
            for key, value in trace_state.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            for key, value in trace_state.get("tags", {}).items():
                mlflow.set_tag(key, str(value))

