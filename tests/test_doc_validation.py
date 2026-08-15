"""Tests for src/guardrails/doc_validation.py (Layer 3, Component 14,
CV_INTEGRATION.md Phase 5). Pure-function tests — no images, no Gemini, no
quota — fixtures build NbiExtractionResult/IdExtractionResult directly.

Fixture identity mirrors data/references/mock/identities.json's id01
(REYES, MARIA SANTOS, EMP-04821) so anyone cross-checking a failure against
the mock dataset finds matching values.
"""

from datetime import date

from src import config
from src.guardrails import doc_validation
from src.schemas import (
    DocType,
    ExtractedField,
    IdExtractionResult,
    IdType,
    ImageQualityReport,
    NbiExtractionResult,
    QualityVerdict,
    ValidationOutcome,
)


def _quality(verdict=QualityVerdict.PASS, normalized_quality=0.95) -> ImageQualityReport:
    return ImageQualityReport(
        verdict=verdict,
        blur_score=1200.0,
        exposure_clip=0.01,
        skew_deg=1.0,
        min_dim_px=1200,
        quad_found=True,
        normalized_quality=normalized_quality,
        reasons=[],
    )


def _field(value, confidence: float = 0.95) -> ExtractedField:
    return ExtractedField(value=value, verbatim_text=value or "", model_confidence=confidence)


def _faculty_record(**overrides) -> dict:
    base = {"employee_id": "EMP-04821", "full_name": "REYES, MARIA SANTOS", "date_of_birth": "1990-01-01"}
    base.update(overrides)
    return base


def _clean_nbi(**overrides) -> NbiExtractionResult:
    base = dict(
        doc_type=DocType.NBI_CLEARANCE,
        family_name=_field("REYES"),
        first_name=_field("MARIA"),
        middle_name=_field("SANTOS"),
        date_of_birth=_field("1990-01-01"),
        reference_no=_field("REYE900101-N00457821"),
        date_printed=_field("2026-02-14"),
        valid_until=_field("2027-02-14"),
        purpose=_field("Employment"),
        remarks=_field("NO DEROGATORY"),
        overall_confidence=0.95,
    )
    base.update(overrides)
    return NbiExtractionResult(**base)


_ID_NUMBER_BY_TYPE = {
    IdType.NATIONAL_ID: "1234-5678-9101-0001",
    IdType.DRIVERS_LICENSE: "N03-12-123456",
    IdType.PASSPORT: "P1234567A",
}


def _clean_id(id_type: IdType = IdType.NATIONAL_ID, **overrides) -> IdExtractionResult:
    base = dict(
        doc_type=DocType.GOVERNMENT_ID,
        id_type=id_type,
        family_name=_field("REYES"),
        first_name=_field("MARIA"),
        middle_name=_field("SANTOS"),
        date_of_birth=_field("1990-01-01"),
        id_number=_field(_ID_NUMBER_BY_TYPE[id_type]),
        issue_date=_field("2019-06-14"),
        expiry_date=_field(None, confidence=0.0) if id_type == IdType.NATIONAL_ID else _field("2028-06-14"),
        overall_confidence=0.95,
    )
    base.update(overrides)
    return IdExtractionResult(**base)


def _by_rule(result, rule_name: str):
    return next(r for r in result.rules if r.rule == rule_name)


# --- validate_document() (NBI) ------------------------------------------------


def test_all_rules_passing_accepts():
    result = doc_validation.validate_document(_clean_nbi(), _quality(), _faculty_record())
    assert result.outcome == ValidationOutcome.ACCEPTED
    assert all(r.passed for r in result.rules)


def test_rule1_type_mismatch_rejects():
    extracted = _clean_nbi(doc_type=DocType.UNKNOWN_DOCUMENT)
    result = doc_validation.validate_document(extracted, _quality(), _faculty_record())
    assert result.outcome == ValidationOutcome.REJECTED
    assert not _by_rule(result, "type_match").passed


def test_rule2_missing_required_field_needs_review():
    extracted = _clean_nbi(purpose=_field(None, confidence=0.0))
    result = doc_validation.validate_document(extracted, _quality(), _faculty_record())
    assert result.outcome == ValidationOutcome.NEEDS_REVIEW
    rule = _by_rule(result, "completeness")
    assert not rule.passed
    assert "purpose" in rule.detail


def test_rule3_clean_remarks_passes():
    extracted = _clean_nbi(remarks=_field("NO DEROGATORY"))
    result = doc_validation.validate_document(extracted, _quality(), _faculty_record())
    assert _by_rule(result, "format").passed


def test_rule3_derogatory_remarks_fails_format_and_forces_composite_zero():
    """A non-allowlisted remarks value simulates an actual derogatory-record
    hit — must fail Rule 3 outright (needs_review) and force
    composite_confidence to 0.0 regardless of how confident the extraction was."""
    extracted = _clean_nbi(remarks=_field("WITH DEROGATORY RECORD"), overall_confidence=0.99)
    result = doc_validation.validate_document(extracted, _quality(normalized_quality=0.99), _faculty_record())
    assert result.outcome == ValidationOutcome.NEEDS_REVIEW
    assert not _by_rule(result, "format").passed
    assert result.composite_confidence == 0.0


def test_rule3_format_failure_forces_composite_zero_even_with_high_confidence():
    extracted = _clean_nbi(reference_no=_field("not-a-valid-shape!!"), overall_confidence=0.99)
    result = doc_validation.validate_document(extracted, _quality(normalized_quality=0.99), _faculty_record())
    assert result.composite_confidence == 0.0


def test_rule4_wrong_person_needs_review_not_rejected():
    """Never auto-reject on identity mismatch — PH name-form variance is
    expected noise, not evidence of fraud."""
    result = doc_validation.validate_document(
        _clean_nbi(), _quality(), _faculty_record(full_name="SANTOS, JUAN DELA CRUZ")
    )
    assert result.outcome == ValidationOutcome.NEEDS_REVIEW
    assert not _by_rule(result, "identity").passed


def test_rule5_expired_with_pinned_as_of_rejects():
    result = doc_validation.validate_document(
        _clean_nbi(), _quality(), _faculty_record(), as_of=date(2028, 1, 1)
    )
    assert result.outcome == ValidationOutcome.REJECTED
    assert not _by_rule(result, "validity_window").passed


def test_rule5_stricter_employer_window_governs_over_printed_validity(monkeypatch):
    """A realistic printed pair (date_printed 2026-02-14, valid_until
    2027-02-14 -- NBI's normal fixed one-year term, so it clears the
    plausibility gate below) but the employer's freshness policy is set
    stricter than that -- Rule 5 must reject on the EMPLOYER's earlier
    deadline, not wait for the document's own later printed expiry.
    (Previously this test used an implausible 2-week printed validity to
    make the same point -- replaced because real NBI clearances are always
    printed with a fixed one-year term (see the real specimen that prompted
    config.NBI_PRINTED_VALIDITY_MONTHS), so that scenario doesn't occur on
    a genuine document and is exactly what the plausibility gate now flags.)"""
    monkeypatch.setattr(config, "NBI_VALIDITY_MONTHS", 6)
    extracted = _clean_nbi(date_printed=_field("2026-02-14"), valid_until=_field("2027-02-14"))
    result = doc_validation.validate_document(
        extracted, _quality(), _faculty_record(), as_of=date(2026, 9, 1)  # > dp+6mo, still < vu
    )
    assert result.outcome == ValidationOutcome.REJECTED
    assert not _by_rule(result, "validity_window").passed


def test_rule5_within_both_windows_passes():
    extracted = _clean_nbi(date_printed=_field("2026-02-14"), valid_until=_field("2027-02-14"))
    result = doc_validation.validate_document(
        extracted, _quality(), _faculty_record(), as_of=date(2026, 4, 1)
    )
    assert _by_rule(result, "validity_window").passed


# --- Plausibility gate: a valid_until inconsistent with NBI's fixed one- ------
# --- year printed term is treated as a likely OCR misread, not a real expiry --

def test_rule3_implausible_valid_until_fails_format():
    """valid_until a full year short of date_printed + one-year term --
    exactly the shape of a single-digit year misread (e.g. 2026 read as
    2025), not a real document variant."""
    extracted = _clean_nbi(date_printed=_field("2025-11-04"), valid_until=_field("2025-11-04"))
    result = doc_validation.validate_document(extracted, _quality(), _faculty_record())
    rule = _by_rule(result, "format")
    assert not rule.passed
    assert "one-year printed validity" in rule.detail


def test_rule5_defers_to_format_on_implausible_valid_until_instead_of_rejecting():
    """The actual bug this closes: date_printed correct (2025-11-04),
    valid_until misread a year short (should be 2026-11-04) -- as_of is
    within the TRUE window but past the misread one. Without the
    plausibility gate, Rule 5 independently reject on the same bad value
    and win _resolve_outcome's priority race before Rule 3's needs_review
    verdict is ever consulted. With it: NEEDS_REVIEW, not a false REJECT."""
    extracted = _clean_nbi(date_printed=_field("2025-11-04"), valid_until=_field("2025-11-04"))
    result = doc_validation.validate_document(
        extracted, _quality(), _faculty_record(), as_of=date(2026, 8, 15)
    )
    assert result.outcome == ValidationOutcome.NEEDS_REVIEW
    assert _by_rule(result, "validity_window").passed  # deferred, not independently failed
    assert not _by_rule(result, "format").passed


def test_rule5_plausible_valid_until_within_tolerance_still_enforced():
    """valid_until 10 days off the expected one-year mark -- inside
    config.NBI_PRINTED_VALIDITY_TOLERANCE_DAYS (14), so the plausibility
    gate does NOT fire, and a genuinely expired document past that date
    still correctly REJECTs."""
    extracted = _clean_nbi(date_printed=_field("2025-01-01"), valid_until=_field("2026-01-11"))  # +10 days
    result = doc_validation.validate_document(
        extracted, _quality(), _faculty_record(), as_of=date(2026, 6, 1)
    )
    assert result.outcome == ValidationOutcome.REJECTED
    assert not _by_rule(result, "validity_window").passed


def test_rule6_low_composite_needs_review_even_when_quality_pass():
    extracted = _clean_nbi(overall_confidence=0.2)
    result = doc_validation.validate_document(extracted, _quality(normalized_quality=0.9), _faculty_record())
    assert result.outcome == ValidationOutcome.NEEDS_REVIEW
    assert not _by_rule(result, "fail_safe").passed


def test_extracted_none_with_quality_ok_needs_review():
    result = doc_validation.validate_document(None, _quality(), _faculty_record())
    assert result.outcome == ValidationOutcome.NEEDS_REVIEW
    assert result.composite_confidence == 0.0


def test_quality_reject_short_circuits_to_rejected():
    result = doc_validation.validate_document(_clean_nbi(), _quality(verdict=QualityVerdict.REJECT), _faculty_record())
    assert result.outcome == ValidationOutcome.REJECTED
    assert result.composite_confidence == 0.0


# --- validate_id_document() (Government ID, §1.4a) ----------------------------


def test_national_id_all_pass_accepts():
    result = doc_validation.validate_id_document(_clean_id(IdType.NATIONAL_ID), _quality(), _faculty_record())
    assert result.outcome == ValidationOutcome.ACCEPTED


def test_drivers_license_all_pass_accepts():
    result = doc_validation.validate_id_document(_clean_id(IdType.DRIVERS_LICENSE), _quality(), _faculty_record())
    assert result.outcome == ValidationOutcome.ACCEPTED


def test_passport_all_pass_accepts():
    result = doc_validation.validate_id_document(_clean_id(IdType.PASSPORT), _quality(), _faculty_record())
    assert result.outcome == ValidationOutcome.ACCEPTED


def test_national_id_missing_expiry_skips_rule5_as_passed_not_missing_field():
    extracted = _clean_id(IdType.NATIONAL_ID)
    result = doc_validation.validate_id_document(extracted, _quality(), _faculty_record())
    rule5 = _by_rule(result, "validity_window")
    assert rule5.passed
    assert "no printed expiry" in rule5.detail
    # And completeness must NOT complain about the missing expiry either.
    assert _by_rule(result, "completeness").passed


def test_drivers_license_missing_expiry_fails_completeness():
    extracted = _clean_id(IdType.DRIVERS_LICENSE, expiry_date=_field(None, confidence=0.0))
    result = doc_validation.validate_id_document(extracted, _quality(), _faculty_record())
    rule2 = _by_rule(result, "completeness")
    assert not rule2.passed
    assert "expiry_date" in rule2.detail
    assert result.outcome == ValidationOutcome.NEEDS_REVIEW


def test_passport_missing_expiry_fails_completeness():
    extracted = _clean_id(IdType.PASSPORT, expiry_date=_field(None, confidence=0.0))
    result = doc_validation.validate_id_document(extracted, _quality(), _faculty_record())
    assert not _by_rule(result, "completeness").passed


def test_expired_drivers_license_with_pinned_as_of_rejects():
    extracted = _clean_id(IdType.DRIVERS_LICENSE, expiry_date=_field("2020-01-01"))
    result = doc_validation.validate_id_document(
        extracted, _quality(), _faculty_record(), as_of=date(2026, 1, 1)
    )
    assert result.outcome == ValidationOutcome.REJECTED
    assert not _by_rule(result, "validity_window").passed


def test_expired_passport_with_pinned_as_of_rejects():
    extracted = _clean_id(IdType.PASSPORT, expiry_date=_field("2020-01-01"))
    result = doc_validation.validate_id_document(
        extracted, _quality(), _faculty_record(), as_of=date(2026, 1, 1)
    )
    assert result.outcome == ValidationOutcome.REJECTED


def test_id_number_wrong_shape_for_type_fails_format():
    """A national_id id_number that doesn't match the national_id pattern —
    dispatched via id_number_validator_for(id_type), not the registry's
    static non-emptiness-only validator."""
    extracted = _clean_id(IdType.NATIONAL_ID, id_number=_field("N03-12-123456"))  # drivers_license shape
    result = doc_validation.validate_id_document(extracted, _quality(), _faculty_record())
    rule3 = _by_rule(result, "format")
    assert not rule3.passed
    assert result.composite_confidence == 0.0


def test_id_wrong_person_needs_review_not_rejected():
    result = doc_validation.validate_id_document(
        _clean_id(), _quality(), _faculty_record(full_name="SANTOS, JUAN DELA CRUZ")
    )
    assert result.outcome == ValidationOutcome.NEEDS_REVIEW
    assert not _by_rule(result, "identity").passed


def test_id_type_itself_is_never_wrong():
    """Any of the four IdType values is a valid government ID submission —
    the type-match rule checks doc_type, not id_type."""
    for id_type in IdType:
        if id_type == IdType.OTHER:
            continue  # OTHER has no id_number pattern; format rule would legitimately fail, not the point here
        result = doc_validation.validate_id_document(_clean_id(id_type), _quality(), _faculty_record())
        assert _by_rule(result, "type_match").passed


def test_id_quality_reject_short_circuits_to_rejected():
    result = doc_validation.validate_id_document(
        _clean_id(), _quality(verdict=QualityVerdict.REJECT), _faculty_record()
    )
    assert result.outcome == ValidationOutcome.REJECTED


def test_id_extracted_none_needs_review():
    result = doc_validation.validate_id_document(None, _quality(), _faculty_record())
    assert result.outcome == ValidationOutcome.NEEDS_REVIEW


# --- validate_cross_document() (§1.4a, new) ------------------------------------


def test_cross_document_matching_pair_passes():
    result = doc_validation.validate_cross_document(_clean_nbi(), _clean_id())
    assert result.passed
    assert result.rule == "cross_document_consistency"


def test_cross_document_name_mismatch_flagged_without_leaking_values():
    id_doc = _clean_id(family_name=_field("CRUZ"), first_name=_field("JUAN"), middle_name=_field(None, 0.0))
    result = doc_validation.validate_cross_document(_clean_nbi(), id_doc)
    assert not result.passed
    assert "name" in result.detail
    assert "CRUZ" not in result.detail
    assert "REYES" not in result.detail


def test_cross_document_dob_mismatch_flagged_without_leaking_values():
    id_doc = _clean_id(date_of_birth=_field("1985-05-05"))
    result = doc_validation.validate_cross_document(_clean_nbi(), id_doc)
    assert not result.passed
    assert "date_of_birth" in result.detail
    assert "1990-01-01" not in result.detail
    assert "1985-05-05" not in result.detail


def test_cross_document_both_mismatch_lists_both():
    id_doc = _clean_id(
        family_name=_field("CRUZ"), first_name=_field("JUAN"), middle_name=_field(None, 0.0),
        date_of_birth=_field("1985-05-05"),
    )
    result = doc_validation.validate_cross_document(_clean_nbi(), id_doc)
    assert not result.passed
    assert "name" in result.detail
    assert "date_of_birth" in result.detail


# --- apply_cross_document_result() (Phase 6, POST /upload-doc's sibling check) -

def test_apply_cross_document_escalates_accepted_to_needs_review_on_mismatch():
    accepted = doc_validation.validate_document(_clean_nbi(), _quality(), _faculty_record())
    assert accepted.outcome == ValidationOutcome.ACCEPTED
    cross_rule = doc_validation.validate_cross_document(
        _clean_nbi(), _clean_id(family_name=_field("CRUZ"), first_name=_field("JUAN"), middle_name=_field(None, 0.0))
    )

    result = doc_validation.apply_cross_document_result(accepted, cross_rule)

    assert result.outcome == ValidationOutcome.NEEDS_REVIEW
    assert "cross_document_consistency" in [r.rule for r in result.rules]
    assert not next(r for r in result.rules if r.rule == "cross_document_consistency").passed


def test_apply_cross_document_passing_check_leaves_accepted_untouched():
    accepted = doc_validation.validate_document(_clean_nbi(), _quality(), _faculty_record())
    cross_rule = doc_validation.validate_cross_document(_clean_nbi(), _clean_id())

    result = doc_validation.apply_cross_document_result(accepted, cross_rule)

    assert result.outcome == ValidationOutcome.ACCEPTED
    assert result.message == accepted.message


def test_apply_cross_document_never_downgrades_below_needs_review():
    """A document already rejected/needs_review on its own merits stays
    exactly that -- a failing cross-check doesn't make it "more rejected"."""
    already_needs_review = doc_validation.validate_document(
        _clean_nbi(purpose=_field(None, 0.0)), _quality(), _faculty_record()
    )
    assert already_needs_review.outcome == ValidationOutcome.NEEDS_REVIEW
    mismatched_cross_rule = doc_validation.validate_cross_document(
        _clean_nbi(), _clean_id(family_name=_field("CRUZ"), first_name=_field("JUAN"), middle_name=_field(None, 0.0))
    )

    result = doc_validation.apply_cross_document_result(already_needs_review, mismatched_cross_rule)

    assert result.outcome == ValidationOutcome.NEEDS_REVIEW
    assert "cross_document_consistency" in [r.rule for r in result.rules]


def test_apply_cross_document_never_upgrades_a_rejected_outcome():
    """A passing cross-check doesn't excuse an expired document -- agreement
    between two documents isn't proof the first one is still valid."""
    expired = doc_validation.validate_document(
        _clean_nbi(), _quality(), _faculty_record(), as_of=date(2028, 1, 1)
    )
    assert expired.outcome == ValidationOutcome.REJECTED
    passing_cross_rule = doc_validation.validate_cross_document(_clean_nbi(), _clean_id())

    result = doc_validation.apply_cross_document_result(expired, passing_cross_rule)

    assert result.outcome == ValidationOutcome.REJECTED


def test_apply_cross_document_appends_rule_without_dropping_existing_ones():
    accepted = doc_validation.validate_document(_clean_nbi(), _quality(), _faculty_record())
    cross_rule = doc_validation.validate_cross_document(_clean_nbi(), _clean_id())

    result = doc_validation.apply_cross_document_result(accepted, cross_rule)

    original_rule_names = {r.rule for r in accepted.rules}
    result_rule_names = {r.rule for r in result.rules}
    assert original_rule_names <= result_rule_names
    assert len(result.rules) == len(accepted.rules) + 1


# --- PII assertion, across a representative sweep -----------------------------


def test_no_pii_leaks_into_any_rule_detail_or_message():
    pii_needles = ["REYES", "MARIA", "SANTOS", "1990-01-01", "REYE900101-N00457821", "CRUZ", "JUAN"]
    cases = [
        doc_validation.validate_document(_clean_nbi(), _quality(), _faculty_record()),
        doc_validation.validate_document(
            _clean_nbi(purpose=_field(None, 0.0)), _quality(), _faculty_record()
        ),
        doc_validation.validate_document(
            _clean_nbi(), _quality(), _faculty_record(full_name="SANTOS, JUAN DELA CRUZ")
        ),
        doc_validation.validate_id_document(_clean_id(), _quality(), _faculty_record()),
        doc_validation.validate_id_document(
            _clean_id(), _quality(), _faculty_record(full_name="SANTOS, JUAN DELA CRUZ")
        ),
    ]
    for result in cases:
        assert not any(needle in result.message for needle in pii_needles)
        for rule in result.rules:
            assert not any(needle in rule.detail for needle in pii_needles)

    cross = doc_validation.validate_cross_document(
        _clean_nbi(), _clean_id(family_name=_field("CRUZ"), first_name=_field("JUAN"), middle_name=_field(None, 0.0))
    )
    assert not any(needle in cross.detail for needle in pii_needles)
