"""Deterministic accept/reject/needs_review policy — Layer 3, Component 14
(CV_INTEGRATION.md §1.4/§1.4a/§2.7).

The model (src/ocr/extractor.py) only detects field values; this module owns
every accept/reject decision, matching the same detection-vs-policy split
already used in src/guardrails/llm_judge.py's to_guardrail_result(). Two
single-document validators share one six-rule shape (type match,
completeness, format, identity, validity window, fail-safe), plus one
cross-document check that only makes sense once both documents exist.

composite_confidence = min(quality.normalized_quality, extraction.overall_confidence),
forced to 0.0 by any Rule 3 (format) failure — a deterministic signal beats
model confidence; the LLM cannot talk past a failed regex.

Rule 4 (identity) and Rule 6 (fail-safe) never reject outright, only
escalate to needs_review — PH name-form variance (middle initials, Jr./III,
surname-first) is expected noise, not evidence of fraud, and a human should
see anything the deterministic checks aren't fully sure about. This is
enforced structurally by _resolve_outcome() below, not by convention.

Every RuleResult.detail and ValidationResult.message is PII-free by
construction — field names and rule names only, never a value.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from enum import Enum
from typing import Callable

from dateutil.relativedelta import relativedelta
from rapidfuzz import fuzz

from src import config
from src.ocr import doctypes
from src.schemas import (
    DocType,
    ExtractedField,
    IdExtractionResult,
    IdType,
    ImageQualityReport,
    NbiExtractionResult,
    QualityVerdict,
    RuleResult,
    ValidationOutcome,
    ValidationResult,
)

# --- User-facing copy, PII-free ----------------------------------------------

_MSG_QUALITY_REJECTED = "The uploaded image could not be processed — please re-upload a clearer photo or scan."
_MSG_EXTRACTION_FAILED = "We couldn't automatically read this document. It has been sent for manual review."
_MSG_WRONG_TYPE = "This doesn't look like the expected document type. Please re-upload the correct document."
_MSG_EXPIRED = "This document is outside its validity window and cannot be accepted as-is."
_MSG_INCOMPLETE = "Some required information could not be read from this document. It has been sent for manual review."
_MSG_FORMAT = "Some fields on this document could not be verified. It has been sent for manual review."
_MSG_IDENTITY = "The name on this document does not match our records. It has been sent for manual review."
_MSG_NEEDS_REVIEW = "This document needs manual review before it can be accepted."
_MSG_ACCEPTED = "Document verified successfully."

_SUFFIXES = {"JR", "SR", "II", "III", "IV"}
_NON_ALNUM = re.compile(r"[^A-Z0-9 ]")


def _normalize_name(value: str | None) -> str:
    """case-fold, strip punctuation, drop Jr./Sr./II/III, NFKD-fold ñ. Token
    order is left alone — rapidfuzz.fuzz.token_set_ratio is already
    order-insensitive, so surname-first vs. given-name-first both match
    without sorting here."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").upper()
    text = _NON_ALNUM.sub(" ", text)
    tokens = [t for t in text.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def _parse_flexible_date(value: str | None) -> date | None:
    """Bare ISO date or ISO datetime, same shape as doctypes._parse_date but
    kept local — this module reasons about dates as a validation concern,
    not a format-registry concern, and the two shouldn't reach into each
    other's private helpers."""
    if not value:
        return None
    text = value.strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        from datetime import datetime

        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _field_raw_value(extracted, name: str):
    """Unwraps an ExtractedField's .value, or an Enum's .value (id_type) —
    everything else (doc_type is also an Enum) falls through the same path,
    since IdType/DocType are both str Enums and already carry .value."""
    obj = getattr(extracted, name, None)
    if isinstance(obj, ExtractedField):
        return obj.value
    if isinstance(obj, Enum):
        return obj.value
    return obj


# --- Shared rule builders, reused by both validators -------------------------


def _rule_type_match(extracted, expected_doc_type: DocType) -> RuleResult:
    passed = extracted.doc_type == expected_doc_type
    detail = (
        "document type matches"
        if passed
        else f"expected {expected_doc_type.value}, got {extracted.doc_type.value}"
    )
    return RuleResult(rule="type_match", passed=passed, detail=detail)


def _rule_completeness(extracted, doc_type: DocType, id_type: IdType | None = None) -> RuleResult:
    required = doctypes.required_fields(doc_type, id_type=id_type)
    missing = [name for name in required if not _field_raw_value(extracted, name)]
    passed = not missing
    detail = "all required fields present" if passed else f"required field missing: {', '.join(missing)}"
    return RuleResult(rule="completeness", passed=passed, detail=detail)


def _rule_format(extracted, doc_type: DocType, overrides: dict[str, Callable] | None = None) -> RuleResult:
    spec = doctypes.get_spec(doc_type)
    overrides = overrides or {}
    bad = []
    for field in spec.fields:
        validator = overrides.get(field.name, field.validator)
        if not validator(_field_raw_value(extracted, field.name)):
            bad.append(field.name)
    passed = not bad
    detail = "all fields well-formed" if passed else f"invalid format: {', '.join(bad)}"
    return RuleResult(rule="format", passed=passed, detail=detail)


def _rule_identity(extracted, faculty_record: dict) -> RuleResult:
    score = fuzz.token_set_ratio(
        _normalize_name(extracted.full_name_display()), _normalize_name(faculty_record.get("full_name"))
    )
    passed = score >= config.NAME_MATCH_THRESHOLD
    detail = "name matches employee record" if passed else "name on document does not match employee record"
    return RuleResult(rule="identity", passed=passed, detail=detail)


def _rule6_fail_safe(composite: float) -> RuleResult:
    passed = composite >= config.OCR_CONFIDENCE_FLOOR
    detail = "composite confidence sufficient" if passed else "composite confidence below floor"
    return RuleResult(rule="fail_safe", passed=passed, detail=detail)


def _resolve_outcome(rules: list[RuleResult]) -> tuple[ValidationOutcome, str]:
    """Deterministic outcome table (CV_INTEGRATION.md §1.4), evaluated in
    priority order. Rule 4 (identity) and Rule 6 (fail-safe) can only ever
    route to NEEDS_REVIEW here — never REJECTED — which is what guarantees
    the "never auto-reject on identity" property structurally rather than by
    convention at each call site."""
    by_name = {r.rule: r for r in rules}

    if not by_name["type_match"].passed:
        return ValidationOutcome.REJECTED, _MSG_WRONG_TYPE
    if not by_name["validity_window"].passed:
        return ValidationOutcome.REJECTED, _MSG_EXPIRED
    if not by_name["completeness"].passed:
        return ValidationOutcome.NEEDS_REVIEW, _MSG_INCOMPLETE
    if not by_name["format"].passed:
        return ValidationOutcome.NEEDS_REVIEW, _MSG_FORMAT
    if not by_name["identity"].passed:
        return ValidationOutcome.NEEDS_REVIEW, _MSG_IDENTITY
    if not by_name["fail_safe"].passed:
        return ValidationOutcome.NEEDS_REVIEW, _MSG_NEEDS_REVIEW
    return ValidationOutcome.ACCEPTED, _MSG_ACCEPTED


def _pre_extraction_reject() -> ValidationResult:
    rule = RuleResult(rule="fail_safe", passed=False, detail="image quality below acceptable threshold")
    return ValidationResult(
        outcome=ValidationOutcome.REJECTED, rules=[rule], composite_confidence=0.0, message=_MSG_QUALITY_REJECTED
    )


def _extraction_failed() -> ValidationResult:
    rule = RuleResult(rule="fail_safe", passed=False, detail="extraction returned no parseable result")
    return ValidationResult(
        outcome=ValidationOutcome.NEEDS_REVIEW, rules=[rule], composite_confidence=0.0, message=_MSG_EXTRACTION_FAILED
    )


# --- NBI Clearance ------------------------------------------------------------


def _rule_format_nbi(extracted: NbiExtractionResult) -> RuleResult:
    base = _rule_format(extracted, DocType.NBI_CLEARANCE)
    if not base.passed:
        return base
    # Sanity check beyond per-field parsing: the document's own printed
    # expiry must not predate its print date (CV_INTEGRATION.md §2.7).
    vu = _parse_flexible_date(extracted.valid_until.value)
    dp = _parse_flexible_date(extracted.date_printed.value)
    if vu is not None and dp is not None and vu < dp:
        return RuleResult(rule="format", passed=False, detail="invalid format: valid_until precedes date_printed")
    return base


def _rule_validity_window_nbi(extracted: NbiExtractionResult, as_of: date) -> RuleResult:
    dp = _parse_flexible_date(extracted.date_printed.value)
    vu = _parse_flexible_date(extracted.valid_until.value)
    if dp is None or vu is None:
        return RuleResult(rule="validity_window", passed=True, detail="skipped: date fields did not parse")
    effective_deadline = min(vu, dp + relativedelta(months=config.NBI_VALIDITY_MONTHS))
    passed = as_of <= effective_deadline
    detail = "within validity window" if passed else "outside validity window"
    return RuleResult(rule="validity_window", passed=passed, detail=detail)


def validate_document(
    extracted: NbiExtractionResult | None,
    quality: ImageQualityReport,
    faculty_record: dict,
    expected_doc_type: DocType = DocType.NBI_CLEARANCE,
    as_of: date | None = None,
) -> ValidationResult:
    as_of = as_of or date.today()

    if quality.verdict == QualityVerdict.REJECT:
        return _pre_extraction_reject()
    if extracted is None:
        return _extraction_failed()

    rule1 = _rule_type_match(extracted, expected_doc_type)
    rule2 = _rule_completeness(extracted, expected_doc_type)
    rule3 = _rule_format_nbi(extracted)
    rule4 = _rule_identity(extracted, faculty_record)
    rule5 = _rule_validity_window_nbi(extracted, as_of)

    composite = min(quality.normalized_quality, extracted.overall_confidence)
    if not rule3.passed:
        composite = 0.0
    rule6 = _rule6_fail_safe(composite)

    rules = [rule1, rule2, rule3, rule4, rule5, rule6]
    outcome, message = _resolve_outcome(rules)
    return ValidationResult(outcome=outcome, rules=rules, composite_confidence=composite, message=message)


# --- Government ID (§1.4a) ----------------------------------------------------


def _rule_validity_window_id(extracted: IdExtractionResult, as_of: date) -> RuleResult:
    expiry_raw = extracted.expiry_date.value
    if not expiry_raw:
        # Absence is correct behavior for National ID, not a gap. For
        # drivers_license/passport a missing expiry is already flagged by
        # Rule 2 (completeness) — this rule isn't the place to double-count it.
        detail = (
            "no printed expiry for this id_type"
            if extracted.id_type == IdType.NATIONAL_ID
            else "expiry_date missing; see completeness rule"
        )
        return RuleResult(rule="validity_window", passed=True, detail=detail)
    expiry = _parse_flexible_date(expiry_raw)
    if expiry is None:
        return RuleResult(rule="validity_window", passed=True, detail="skipped: expiry_date did not parse")
    passed = as_of <= expiry
    detail = "within validity window" if passed else "outside validity window"
    return RuleResult(rule="validity_window", passed=passed, detail=detail)


def validate_id_document(
    extracted: IdExtractionResult | None,
    quality: ImageQualityReport,
    faculty_record: dict,
    expected_doc_type: DocType = DocType.GOVERNMENT_ID,
    as_of: date | None = None,
) -> ValidationResult:
    as_of = as_of or date.today()

    if quality.verdict == QualityVerdict.REJECT:
        return _pre_extraction_reject()
    if extracted is None:
        return _extraction_failed()

    rule1 = _rule_type_match(extracted, expected_doc_type)
    rule2 = _rule_completeness(extracted, expected_doc_type, id_type=extracted.id_type)
    rule3 = _rule_format(extracted, DocType.GOVERNMENT_ID, overrides={
        "id_number": doctypes.id_number_validator_for(extracted.id_type),
    })
    rule4 = _rule_identity(extracted, faculty_record)
    rule5 = _rule_validity_window_id(extracted, as_of)

    composite = min(quality.normalized_quality, extracted.overall_confidence)
    if not rule3.passed:
        composite = 0.0
    rule6 = _rule6_fail_safe(composite)

    rules = [rule1, rule2, rule3, rule4, rule5, rule6]
    outcome, message = _resolve_outcome(rules)
    return ValidationResult(outcome=outcome, rules=rules, composite_confidence=composite, message=message)


# --- Cross-document (§1.4a, new) ----------------------------------------------


def validate_cross_document(nbi: NbiExtractionResult, id_doc: IdExtractionResult) -> RuleResult:
    """Compares two already-in-memory extraction results — no lookup, no new
    image bytes. The caller (POST /upload-doc) is responsible for getting a
    sibling document's extraction via a cache-hit extract_document() call
    keyed by its stored source_hash (CV_INTEGRATION.md §2.7). `detail` names
    only which sub-check failed, never the compared values."""
    mismatches = []

    name_score = fuzz.token_set_ratio(_normalize_name(nbi.full_name_display()), _normalize_name(id_doc.full_name_display()))
    if name_score < config.NAME_MATCH_THRESHOLD:
        mismatches.append("name")

    nbi_dob = nbi.date_of_birth.value
    id_dob = id_doc.date_of_birth.value
    if not nbi_dob or not id_dob or nbi_dob != id_dob:
        mismatches.append("date_of_birth")

    passed = not mismatches
    detail = (
        "identity and date of birth agree across both documents"
        if passed
        else f"cross-document mismatch: {', '.join(mismatches)}"
    )
    return RuleResult(rule="cross_document_consistency", passed=passed, detail=detail)
