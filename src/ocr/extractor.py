"""Gemini multimodal document extraction (Layer 2, Component 14,
CV_INTEGRATION.md §2.6).

One Gemini call per document, gated by the deterministic quality check in
src/ocr/quality.py — a `reject` verdict short-circuits before any API call
(the quota mechanism, PLAN.md §2.1) — and a SHA-256-keyed disk cache, so
re-running the same image spends zero calls. DETECTION ONLY: this module
proposes field values, never decides accept/reject; that decision lives
entirely in src/guardrails/doc_validation.py (Layer 3).

extract_document()'s prompt/response_schema are selected by expected_doc_type
(NBI_CLEARANCE -> NbiExtractionResult, GOVERNMENT_ID -> IdExtractionResult) —
same call site, same generate_content() contract as every other structured
call in this codebase (src/guardrails/llm_judge.py, src/rag/answerer.py,
src/agent/router.py).
"""

from __future__ import annotations

import hashlib
import json
import logging

import cv2

from src import config
from src.agent import prompts, usage
from src.ocr import doctypes, quality
from src.schemas import DocType, IdExtractionResult, ImageQualityReport, NbiExtractionResult, QualityVerdict

logger = logging.getLogger(__name__)

_SCHEMA_BY_DOC_TYPE = {
    DocType.NBI_CLEARANCE: (prompts.NBI_EXTRACTION_PROMPT, NbiExtractionResult),
    DocType.GOVERNMENT_ID: (prompts.ID_EXTRACTION_PROMPT, IdExtractionResult),
}


def _field_hint_block(doc_type: DocType) -> str:
    """Builds the prompt's {fields} list from the doctypes registry, so the
    field set requested and the response_schema actually returned can never
    drift apart (CV_INTEGRATION.md §2.5/§2.6)."""
    spec = doctypes.get_spec(doc_type)
    if spec is None:
        return ""
    lines = [
        f"- {field.name} ({'required' if field.required else 'optional'}): {field.prompt_hint}"
        for field in spec.fields
    ]
    return "\n".join(lines)


def _cache_key(original_bytes: bytes, preprocess_flag: bool) -> str:
    digest = hashlib.sha256(original_bytes).hexdigest()
    model_slug = config.ACTIVE_VISION_MODEL.replace("/", "-").replace(":", "-")
    return f"{digest}_{int(preprocess_flag)}_{model_slug}"


def _cache_path(cache_key: str):
    return config.OCR_CACHE_DIR / f"{cache_key}.json"


def _load_from_cache(cache_path, schema):
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        return schema.model_validate(data)
    except Exception:  # corrupt cache entry or schema drift — treat as a miss, never crash
        logger.warning("ocr_cache_read_failed path=%s", cache_path)
        return None


def _save_to_cache(cache_path, result) -> None:
    try:
        config.OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(result.model_dump_json(indent=2))
    except OSError:
        logger.warning("ocr_cache_write_failed path=%s", cache_path)


def extract_document(
    image_bytes: bytes,
    mime_type: str,
    expected_doc_type: DocType = DocType.NBI_CLEARANCE,
    client=None,
    session_id: str | None = None,
    use_cache: bool = True,
    preprocess: bool | None = None,
) -> tuple[NbiExtractionResult | IdExtractionResult | None, ImageQualityReport]:
    if expected_doc_type not in _SCHEMA_BY_DOC_TYPE:
        raise ValueError(f"extract_document: unsupported expected_doc_type={expected_doc_type!r}")

    image = quality.load_image(image_bytes)
    report = quality.assess(image)

    # Reject short-circuits before any cache lookup or API call — this is
    # the quota mechanism, and it applies identically regardless of which
    # vision provider or document type is active.
    if report.verdict == QualityVerdict.REJECT:
        logger.info("session=%s ocr_reject reasons=%s", session_id, report.reasons)
        return None, report

    do_preprocess = config.OCR_PREPROCESS if preprocess is None else preprocess
    if do_preprocess:
        processed_array = quality.preprocess(image)
        ok, encoded = cv2.imencode(".png", processed_array)
        if not ok:
            logger.warning("session=%s ocr_encode_failed", session_id)
            return None, report
        processed_bytes: bytes = encoded.tobytes()
        send_mime_type = "image/png"
    else:
        processed_bytes = image_bytes
        send_mime_type = mime_type

    # Cache key is derived from the ORIGINAL bytes (not the preprocessed
    # array) plus the preprocess flag and active model — deterministic,
    # and doesn't require re-deriving anything to look up a hit.
    cache_key = _cache_key(image_bytes, do_preprocess)
    cache_path = _cache_path(cache_key)
    prompt_template, schema = _SCHEMA_BY_DOC_TYPE[expected_doc_type]

    if use_cache:
        cached = _load_from_cache(cache_path, schema)
        if cached is not None:
            return cached, report

    from google.genai import types
    from google.genai.errors import APIError

    from src.agent.llm_client import LLMBackendError

    vision_client = client or config.get_vision_client()
    try:
        response = vision_client.models.generate_content(
            model=config.ACTIVE_VISION_MODEL,
            contents=[
                types.Part.from_bytes(data=processed_bytes, mime_type=send_mime_type),
                prompt_template.format(fields=_field_hint_block(expected_doc_type)),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )
    except (APIError, LLMBackendError) as exc:
        # Both exceptions, matching the pattern used everywhere else
        # get_llm_client()'s output is called — vision can come from either
        # backend (VISION_PROVIDER), so the catch clause must too.
        logger.warning("session=%s ocr_api_error status=%s", session_id, getattr(exc, "code", "?"))
        return None, report

    usage.record_usage(config.ACTIVE_VISION_MODEL, usage.extract_usage(response), session_id=session_id)

    result = response.parsed
    if result is None:
        # Unparseable response — fails toward needs_review downstream, never
        # toward accept. More likely on the Ollama path (small local models
        # are less reliable at strict response_schema conformance).
        logger.warning("session=%s ocr_unparseable", session_id)
        return None, report

    if use_cache:
        _save_to_cache(cache_path, result)

    return result, report
