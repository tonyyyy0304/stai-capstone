"""Tests for src/ocr/extractor.py.

Unit tests use a fake vision client (_FakeModels/_FakeClient below) — zero
API calls, zero quota, runs with no GEMINI_API_KEY. They verify the parts
that matter most: the reject short-circuit and the cache both spend 0 calls,
both exception types are caught, both doc types dispatch to the right
schema.

One test (test_live_extraction_against_real_gemini) actually calls Gemini.
Skipped by default — set RUN_LIVE_OCR_TESTS=1 (and a real GEMINI_API_KEY) to
opt in. This mirrors the "quota-costing flags off by default" pattern
already used in evals/run_guardrail_eval.py; routine `pytest tests/` must
never spend live quota on its own.
"""

import hashlib
import os
from types import SimpleNamespace

import pytest

from google.genai.errors import APIError

from src import config
from src.agent.llm_client import LLMBackendError
from src.ocr import extractor
from src.schemas import DocType, ExtractedField, IdExtractionResult, IdType, NbiExtractionResult

MOCK_DIR = config.MOCK_DOCS_DIR


def _requires_mock_dataset():
    if not MOCK_DIR.exists() or not any(MOCK_DIR.glob("*.png")):
        pytest.skip("mock dataset not generated — run scripts/make_onboarding_docs.py first")


def _bytes(name: str) -> bytes:
    return (MOCK_DIR / name).read_bytes()


# --- Fake vision client — mimics client.models.generate_content(...) --------

class _FakeModels:
    def __init__(self, parsed_result=None, raise_exc: Exception | None = None):
        self.parsed_result = parsed_result
        self.raise_exc = raise_exc
        self.call_count = 0

    def generate_content(self, model, contents, config):
        self.call_count += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return SimpleNamespace(parsed=self.parsed_result, usage_metadata=None)


class _FakeClient:
    def __init__(self, **kwargs):
        self.models = _FakeModels(**kwargs)


class _SequentialFakeModels:
    """Returns a different canned result on each successive call — for
    testing the retry path, where the first and second attempts genuinely
    differ (this is what actually happened live: a real document's
    middle_name came back null on one call, correct on the next)."""

    def __init__(self, results: list):
        self.results = list(results)
        self.call_count = 0

    def generate_content(self, model, contents, config):
        self.call_count += 1
        result = self.results[min(self.call_count, len(self.results)) - 1]
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(parsed=result, usage_metadata=None)


class _SequentialFakeClient:
    def __init__(self, results: list):
        self.models = _SequentialFakeModels(results)


def _canned_nbi_result() -> NbiExtractionResult:
    return NbiExtractionResult(
        doc_type=DocType.NBI_CLEARANCE,
        family_name=ExtractedField(value="REYES", model_confidence=0.95),
        first_name=ExtractedField(value="MARIA", model_confidence=0.95),
        middle_name=ExtractedField(value="SANTOS", model_confidence=0.9),
        date_of_birth=ExtractedField(value="1990-01-01", model_confidence=0.9),
        reference_no=ExtractedField(value="REYE900101-N00457821", model_confidence=0.9),
        date_printed=ExtractedField(value="2026-02-14", model_confidence=0.8),
        valid_until=ExtractedField(value="2027-02-14", model_confidence=0.9),
        purpose=ExtractedField(value="Employment", model_confidence=0.95),
        remarks=ExtractedField(value="NO DEROGATORY", model_confidence=0.95),
        overall_confidence=0.9,
    )


def _canned_id_result(id_type: IdType = IdType.NATIONAL_ID) -> IdExtractionResult:
    return IdExtractionResult(
        doc_type=DocType.GOVERNMENT_ID,
        id_type=id_type,
        family_name=ExtractedField(value="REYES", model_confidence=0.9),
        first_name=ExtractedField(value="MARIA", model_confidence=0.9),
        middle_name=ExtractedField(value="SANTOS", model_confidence=0.9),
        date_of_birth=ExtractedField(value="1990-01-01", model_confidence=0.9),
        id_number=ExtractedField(value="1234-5678-9101-0001", model_confidence=0.9),
        issue_date=ExtractedField(value="2019-06-14", model_confidence=0.8),
        expiry_date=ExtractedField(value=None, model_confidence=0.0),
        overall_confidence=0.85,
    )


# --- Reject short-circuit: zero API calls ------------------------------------

def test_reject_quality_returns_none_and_spends_zero_calls(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(parsed_result=_canned_nbi_result())

    result, report = extractor.extract_document(
        _bytes("nbi_id01_blur.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )

    assert result is None
    assert report.verdict.value == "reject"
    assert fake_client.models.call_count == 0


def test_reject_quality_short_circuits_before_cache_lookup_too(tmp_path, monkeypatch):
    """Even with use_cache=True, a reject never touches the cache or the client."""
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(parsed_result=_canned_nbi_result())

    result, _ = extractor.extract_document(
        _bytes("nbi_id01_blur.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=True,
    )

    assert result is None
    assert fake_client.models.call_count == 0
    assert list(tmp_path.glob("*.json")) == []


# --- Cache: second call on the same image spends zero API calls -------------

def test_cache_hit_spends_zero_calls_on_second_call(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    canned = _canned_nbi_result()
    fake_client = _FakeClient(parsed_result=canned)
    image_bytes = _bytes("nbi_id01_clean.png")

    first, _ = extractor.extract_document(
        image_bytes, "image/png", DocType.NBI_CLEARANCE, client=fake_client, use_cache=True,
    )
    second, _ = extractor.extract_document(
        image_bytes, "image/png", DocType.NBI_CLEARANCE, client=fake_client, use_cache=True,
    )

    assert fake_client.models.call_count == 1  # only the first call actually hit the client
    assert first is not None and second is not None
    assert first.family_name.value == second.family_name.value == "REYES"
    assert list(tmp_path.glob("*.json")) != []  # one cache file was written


def test_cache_key_differs_by_preprocess_flag(tmp_path, monkeypatch):
    """Cache key includes the preprocess flag, so preprocess=True and
    preprocess=False runs on the same image are two independent cache
    entries — required for run_ocr_eval.py's ablation to mean anything."""
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(parsed_result=_canned_nbi_result())
    image_bytes = _bytes("nbi_id01_clean.png")

    extractor.extract_document(image_bytes, "image/png", DocType.NBI_CLEARANCE,
                                client=fake_client, use_cache=True, preprocess=True)
    extractor.extract_document(image_bytes, "image/png", DocType.NBI_CLEARANCE,
                                client=fake_client, use_cache=True, preprocess=False)

    assert fake_client.models.call_count == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_use_cache_false_always_calls_the_client(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(parsed_result=_canned_nbi_result())
    image_bytes = _bytes("nbi_id01_clean.png")

    extractor.extract_document(image_bytes, "image/png", DocType.NBI_CLEARANCE,
                                client=fake_client, use_cache=False)
    extractor.extract_document(image_bytes, "image/png", DocType.NBI_CLEARANCE,
                                client=fake_client, use_cache=False)

    assert fake_client.models.call_count == 2


# --- Error handling: both exception types return (None, report), never raise ---

def test_api_error_returns_none_report_not_raise(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(raise_exc=APIError(503, {"error": {"message": "unavailable"}}))

    result, report = extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )

    assert result is None
    assert report.verdict.value in ("pass", "warn")  # quality was fine; the API call is what failed


def test_llm_backend_error_returns_none_report_not_raise(tmp_path, monkeypatch):
    """Ollama-path failures (VISION_PROVIDER=ollama) must be caught the same
    way as Gemini's APIError — vision can come from either backend."""
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(raise_exc=LLMBackendError("connection refused", code=503))

    result, report = extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )

    assert result is None
    assert report.verdict.value in ("pass", "warn")


def test_unparseable_response_returns_none_not_raise(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(parsed_result=None)  # response.parsed is None — schema conformance failure

    result, report = extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )

    assert result is None
    assert report.verdict.value in ("pass", "warn")


# --- Doc-type dispatch: right schema/prompt for each ------------------------

def test_nbi_doc_type_dispatches_to_nbi_schema(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(parsed_result=_canned_nbi_result())

    result, _ = extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )
    assert isinstance(result, NbiExtractionResult)


def test_government_id_doc_type_dispatches_to_id_schema(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(parsed_result=_canned_id_result())

    result, _ = extractor.extract_document(
        _bytes("id_id01_clean.png"), "image/png", DocType.GOVERNMENT_ID,
        client=fake_client, use_cache=False,
    )
    assert isinstance(result, IdExtractionResult)


def test_unsupported_doc_type_raises_value_error():
    with pytest.raises(ValueError):
        extractor.extract_document(b"irrelevant", "image/png", DocType.UNKNOWN_DOCUMENT)


# --- Field-hint block reflects the registry, not a hardcoded duplicate ------

def test_field_hint_block_lists_all_nbi_required_fields():
    block = extractor._field_hint_block(DocType.NBI_CLEARANCE)
    for name in ("family_name", "first_name", "date_of_birth", "reference_no",
                 "date_printed", "valid_until", "purpose", "remarks"):
        assert name in block


def test_field_hint_block_lists_all_id_fields():
    block = extractor._field_hint_block(DocType.GOVERNMENT_ID)
    for name in ("id_type", "family_name", "first_name", "date_of_birth", "id_number"):
        assert name in block


# --- Retry on low confidence -------------------------------------------------
# Added after a live run where a real document's middle_name came back null
# at confidence 0.1, then correct at confidence 0.99 on an immediate
# identical re-call — Gemini's temperature=0.0 reduces but does not
# guarantee run-to-run determinism (CV_INTEGRATION.md Phase 4/5 note).

def _low_confidence_nbi_result() -> NbiExtractionResult:
    """Mirrors the actual live miss: every field confident except
    middle_name, null at 0.1 — exactly the shape that should trigger a retry."""
    good = _canned_nbi_result()
    return good.model_copy(update={"middle_name": ExtractedField(value=None, model_confidence=0.1)})


def test_retry_triggers_on_null_field_with_low_confidence(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _SequentialFakeClient([_low_confidence_nbi_result(), _canned_nbi_result()])

    result, _ = extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )

    assert fake_client.models.call_count == 2  # first attempt + one retry
    assert result.middle_name.value == "SANTOS"  # recovered from the retry


def test_retry_prefers_higher_confidence_attempt_as_base(tmp_path, monkeypatch):
    """The merge picks the more-confident attempt as the base and only fills
    ITS gaps — it doesn't just blindly prefer the retry."""
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    # Retry comes back with an even better middle_name reading than the
    # base — but the base's OTHER fields (all 0.9-0.95) should be preserved,
    # not overwritten, since the merge only fills nulls.
    fake_client = _SequentialFakeClient([_low_confidence_nbi_result(), _canned_nbi_result()])

    result, _ = extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )

    assert result.family_name.value == "REYES"
    assert result.family_name.model_confidence == 0.95  # untouched by the merge


def test_retry_does_not_trigger_for_structurally_absent_national_id_expiry(tmp_path, monkeypatch):
    """expiry_date=None at confidence 0.0 is CORRECT for a National ID, not
    a miss — must not waste a retry on it."""
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _SequentialFakeClient([_canned_id_result(id_type=IdType.NATIONAL_ID)])

    result, _ = extractor.extract_document(
        _bytes("id_id01_clean.png"), "image/png", DocType.GOVERNMENT_ID,
        client=fake_client, use_cache=False,
    )

    assert fake_client.models.call_count == 1  # no retry spent
    assert result.expiry_date.value is None


def test_retry_does_trigger_for_missing_expiry_on_drivers_license(tmp_path, monkeypatch):
    """The same null+low-confidence expiry_date IS retry-worthy for a
    license/passport, where expiry_date is required and always printed."""
    _requires_mock_dataset()
    license_missing_expiry = _canned_id_result(id_type=IdType.DRIVERS_LICENSE).model_copy(
        update={"expiry_date": ExtractedField(value=None, model_confidence=0.1)}
    )
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _SequentialFakeClient([license_missing_expiry, _canned_id_result(id_type=IdType.DRIVERS_LICENSE)])

    result, _ = extractor.extract_document(
        _bytes("id_id04_clean.png"), "image/png", DocType.GOVERNMENT_ID,
        client=fake_client, use_cache=False,
    )

    assert fake_client.models.call_count == 2


def test_retry_triggers_on_low_overall_confidence_even_if_all_fields_present(tmp_path, monkeypatch):
    low_overall = _canned_nbi_result().model_copy(update={"overall_confidence": 0.3})
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _SequentialFakeClient([low_overall, _canned_nbi_result()])

    extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )

    assert fake_client.models.call_count == 2


def test_retry_spends_at_most_one_extra_call_even_if_retry_also_uncertain(tmp_path, monkeypatch):
    """No retry loop — one attempt, one retry, done, whatever the outcome."""
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _SequentialFakeClient([_low_confidence_nbi_result(), _low_confidence_nbi_result()])

    result, _ = extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )

    assert fake_client.models.call_count == 2
    assert result is not None  # still returns best-effort, never None just because retry didn't help either


def test_retry_disabled_via_param_skips_it_even_when_warranted(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _SequentialFakeClient([_low_confidence_nbi_result(), _canned_nbi_result()])

    result, _ = extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False, retry=False,
    )

    assert fake_client.models.call_count == 1
    assert result.middle_name.value is None  # not recovered — retry was explicitly off


def test_retry_disabled_via_config_default(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "OCR_ENABLE_RETRY", False)
    fake_client = _SequentialFakeClient([_low_confidence_nbi_result(), _canned_nbi_result()])

    extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,  # retry=None -> falls back to config, now False
    )

    assert fake_client.models.call_count == 1


def test_confident_result_never_retries(tmp_path, monkeypatch):
    """The existing canned fixtures (used throughout the rest of this file)
    must not accidentally trigger a retry — pins that down explicitly."""
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _SequentialFakeClient([_canned_nbi_result()])

    extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )

    assert fake_client.models.call_count == 1


def test_retry_failed_api_call_keeps_first_attempt(tmp_path, monkeypatch):
    """If the retry call itself errors out, don't lose the first attempt's
    (partial but real) result."""
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _SequentialFakeClient([
        _low_confidence_nbi_result(),
        APIError(503, {"error": {"message": "unavailable"}}),
    ])

    result, _ = extractor.extract_document(
        _bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=False,
    )

    assert result is not None
    assert result.family_name.value == "REYES"  # first attempt's good fields survive
    assert result.middle_name.value is None      # the field that prompted the retry stays null — honest, not fabricated


# --- load_cached_result() (Phase 6, cross-document sibling lookup) -----------
# Hash-only lookup — no image bytes required, since POST /upload-doc can't
# assume the sibling document's original bytes are still around
# (config.PERSIST_UPLOADS is False by default).

def test_load_cached_result_hits_by_hash_alone(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(parsed_result=_canned_nbi_result())
    image_bytes = _bytes("nbi_id01_clean.png")
    source_hash = hashlib.sha256(image_bytes).hexdigest()

    extractor.extract_document(
        image_bytes, "image/png", DocType.NBI_CLEARANCE, client=fake_client, use_cache=True,
    )

    result = extractor.load_cached_result(source_hash, DocType.NBI_CLEARANCE)

    assert result is not None
    assert result.family_name.value == "REYES"


def test_load_cached_result_miss_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    assert extractor.load_cached_result("0" * 64, DocType.NBI_CLEARANCE) is None


def test_load_cached_result_respects_preprocess_flag(tmp_path, monkeypatch):
    """The preprocess flag is part of the cache key -- a hash cached under
    preprocess=True must not spuriously hit a preprocess=False lookup."""
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(parsed_result=_canned_nbi_result())
    image_bytes = _bytes("nbi_id01_clean.png")
    source_hash = hashlib.sha256(image_bytes).hexdigest()

    extractor.extract_document(
        image_bytes, "image/png", DocType.NBI_CLEARANCE,
        client=fake_client, use_cache=True, preprocess=True,
    )

    assert extractor.load_cached_result(source_hash, DocType.NBI_CLEARANCE, preprocess_flag=True) is not None
    assert extractor.load_cached_result(source_hash, DocType.NBI_CLEARANCE, preprocess_flag=False) is None


def test_load_cached_result_unsupported_doc_type_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    assert extractor.load_cached_result("0" * 64, DocType.UNKNOWN_DOCUMENT) is None


def test_load_cached_result_works_for_government_id_too(tmp_path, monkeypatch):
    _requires_mock_dataset()
    monkeypatch.setattr(config, "OCR_CACHE_DIR", tmp_path)
    fake_client = _FakeClient(parsed_result=_canned_id_result(id_type=IdType.NATIONAL_ID))
    image_bytes = _bytes("id_id01_clean.png")
    source_hash = hashlib.sha256(image_bytes).hexdigest()

    extractor.extract_document(
        image_bytes, "image/png", DocType.GOVERNMENT_ID, client=fake_client, use_cache=True,
    )

    result = extractor.load_cached_result(source_hash, DocType.GOVERNMENT_ID)
    assert result is not None
    assert result.id_type == IdType.NATIONAL_ID


# --- Optional live test — real Gemini call, skipped by default --------------

_LIVE_OPT_IN = os.environ.get("RUN_LIVE_OCR_TESTS") == "1" and bool(os.environ.get("GEMINI_API_KEY"))


@pytest.mark.skipif(not _LIVE_OPT_IN, reason="set RUN_LIVE_OCR_TESTS=1 and GEMINI_API_KEY to run (spends real quota)")
def test_live_extraction_against_real_gemini():
    _requires_mock_dataset()
    result, report = extractor.extract_document(_bytes("nbi_id01_clean.png"), "image/png", DocType.NBI_CLEARANCE)
    assert report.verdict.value == "pass"
    assert result is not None
    assert result.doc_type == DocType.NBI_CLEARANCE
    for field_name in ("family_name", "first_name", "date_of_birth", "reference_no",
                        "date_printed", "valid_until", "purpose", "remarks"):
        assert getattr(result, field_name).value, f"{field_name} came back empty on a clean image"
