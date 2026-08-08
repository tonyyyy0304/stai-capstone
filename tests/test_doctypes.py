import json

import pytest
from rapidfuzz import fuzz

from src import config
from src.ocr import doctypes
from src.schemas import DocType, ExtractedField, IdExtractionResult, IdType, NbiExtractionResult

MOCK_DIR = config.MOCK_DOCS_DIR


def _requires_mock_dataset():
    if not MOCK_DIR.exists() or not any(MOCK_DIR.glob("*.expected.json")):
        pytest.skip("mock dataset not generated — run scripts/make_onboarding_docs.py first")


# --- required_fields() -------------------------------------------------------

def test_nbi_required_fields_exact_set():
    fields = set(doctypes.required_fields(DocType.NBI_CLEARANCE))
    assert fields == {
        "family_name", "first_name", "date_of_birth", "reference_no",
        "date_printed", "valid_until", "purpose", "remarks",
    }
    assert "middle_name" not in fields


def test_id_required_fields_always_include_core_set():
    for id_type in (IdType.NATIONAL_ID, IdType.DRIVERS_LICENSE, IdType.PASSPORT):
        fields = set(doctypes.required_fields(DocType.GOVERNMENT_ID, id_type=id_type))
        assert {"id_type", "family_name", "first_name", "date_of_birth", "id_number"} <= fields
        assert "middle_name" not in fields


def test_expiry_date_required_for_license_and_passport_only():
    assert "expiry_date" in doctypes.required_fields(DocType.GOVERNMENT_ID, id_type=IdType.DRIVERS_LICENSE)
    assert "expiry_date" in doctypes.required_fields(DocType.GOVERNMENT_ID, id_type=IdType.PASSPORT)
    assert "expiry_date" not in doctypes.required_fields(DocType.GOVERNMENT_ID, id_type=IdType.NATIONAL_ID)


def test_required_fields_unknown_doc_type_is_empty():
    assert doctypes.required_fields(DocType.UNKNOWN_DOCUMENT) == ()


def test_get_spec_returns_none_for_unregistered_doc_type():
    assert doctypes.get_spec(DocType.NOT_A_DOCUMENT) is None


# --- Field validators: one passing + one failing case each -------------------

@pytest.mark.parametrize("value", ["DELA CRUZ", "DE LA CRUZ", "SAN JUAN", "REYES", "NIÑO"])
def test_is_valid_name_accepts_compounds_and_enye(value):
    assert doctypes.is_valid_name(value)


@pytest.mark.parametrize("value", [None, "", "   ", "REYES3", "123", "reyes!"])
def test_is_valid_name_rejects_empty_digits_and_symbols(value):
    assert not doctypes.is_valid_name(value)


def test_is_valid_optional_name_accepts_empty():
    assert doctypes.is_valid_optional_name(None)
    assert doctypes.is_valid_optional_name("")
    assert doctypes.is_valid_optional_name("SANTOS")


def test_is_valid_optional_name_still_rejects_malformed_when_present():
    assert not doctypes.is_valid_optional_name("SANTOS3")


def test_past_or_present_date_accepts_today_and_past():
    assert doctypes.is_valid_past_or_present_date("1990-01-01")


def test_past_or_present_date_rejects_future_and_garbage():
    assert not doctypes.is_valid_past_or_present_date("2099-01-01")
    assert not doctypes.is_valid_past_or_present_date("not-a-date")
    assert not doctypes.is_valid_past_or_present_date(None)


def test_valid_date_accepts_future_dates():
    """valid_until/expiry_date are legitimately in the future — unlike
    date_of_birth/date_printed, this validator must not reject them."""
    assert doctypes.is_valid_date("2099-01-01")


def test_valid_date_rejects_garbage():
    assert not doctypes.is_valid_date("not-a-date")
    assert not doctypes.is_valid_date(None)


def test_is_valid_nbi_reference_no_accepts_observed_shape():
    assert doctypes.is_valid_nbi_reference_no("REYE900101-N00457821")  # mock generator's shape
    assert doctypes.is_valid_nbi_reference_no("HGUR87H38D-U47204A873")  # specimen sample's shape


def test_is_valid_nbi_reference_no_rejects_malformed():
    assert not doctypes.is_valid_nbi_reference_no("no-dash-here-at-all-too-long-xxxxx")
    assert not doctypes.is_valid_nbi_reference_no(None)
    assert not doctypes.is_valid_nbi_reference_no("")
    # Two dashes — the original invented mock format, before any real
    # specimen existed; kept as a regression test for the exact mismatch
    # Phase 3's cross-check caught (CV_INTEGRATION.md, script header comment).
    assert not doctypes.is_valid_nbi_reference_no("NBI-2026-00457821")


def test_is_valid_remarks_accepts_allowlisted_clean_status():
    for value in config.NBI_CLEAN_REMARKS:
        assert doctypes.is_valid_remarks(value)
    assert doctypes.is_valid_remarks("no derogatory")  # case-insensitive


def test_is_valid_remarks_rejects_unrecognized_value():
    """This is the check that catches an actual derogatory-record hit —
    anything not on the allowlist must fail, never be assumed clean."""
    assert not doctypes.is_valid_remarks("WITH DEROGATORY RECORD")
    assert not doctypes.is_valid_remarks("PENDING CASE")
    assert not doctypes.is_valid_remarks(None)


@pytest.mark.parametrize(("id_type", "value"), [
    ("national_id", "1234-5678-9101-0001"),
    ("drivers_license", "N04-12-000004"),
    ("passport", "P0000007A"),
])
def test_id_number_validator_accepts_matching_shape(id_type, value):
    assert doctypes.id_number_validator_for(id_type)(value)


@pytest.mark.parametrize(("id_type", "value"), [
    ("national_id", "N04-12-000004"),        # license-shaped, not national_id-shaped
    ("drivers_license", "1234-5678-9101-0001"),  # national_id-shaped, not license-shaped
    ("passport", "1234-5678-9101-0001"),     # national_id-shaped, not passport-shaped
])
def test_id_number_validator_rejects_wrong_shape_for_type(id_type, value):
    assert not doctypes.id_number_validator_for(id_type)(value)


def test_id_number_validator_for_other_falls_back_to_non_empty():
    validator = doctypes.id_number_validator_for(IdType.OTHER)
    assert validator("anything-nonempty")
    assert not validator("")
    assert not validator(None)


def test_is_valid_id_type_accepts_registered_values_only():
    assert doctypes.is_valid_id_type("passport")
    assert not doctypes.is_valid_id_type("student_id")
    assert not doctypes.is_valid_id_type(None)


# --- full_name_display() order-insensitivity (Rule 4 / cross-document) ------
# Full constructors (not model_construct, which skips nested-model coercion
# and would leave family_name etc. as plain dicts) with an empty
# ExtractedField() for every field the test doesn't care about — every
# ExtractedField subfield already defaults, so ExtractedField() alone is valid.

def _nbi_stub(**overrides) -> NbiExtractionResult:
    base = dict(
        doc_type=DocType.NBI_CLEARANCE,
        family_name=ExtractedField(), first_name=ExtractedField(), middle_name=ExtractedField(),
        date_of_birth=ExtractedField(), reference_no=ExtractedField(), date_printed=ExtractedField(),
        valid_until=ExtractedField(), purpose=ExtractedField(), remarks=ExtractedField(),
    )
    base.update(overrides)
    return NbiExtractionResult(**base)


def _id_stub(**overrides) -> IdExtractionResult:
    base = dict(
        doc_type=DocType.GOVERNMENT_ID, id_type=IdType.NATIONAL_ID,
        family_name=ExtractedField(), first_name=ExtractedField(), middle_name=ExtractedField(),
        date_of_birth=ExtractedField(), id_number=ExtractedField(), issue_date=ExtractedField(),
        expiry_date=ExtractedField(),
    )
    base.update(overrides)
    return IdExtractionResult(**base)


def test_nbi_full_name_display_is_order_insensitive_for_matching():
    result = _nbi_stub(
        family_name=ExtractedField(value="DELA CRUZ"),
        first_name=ExtractedField(value="JUAN"),
        middle_name=ExtractedField(value="P"),
    )
    family_first = result.full_name_display()
    assert family_first == "DELA CRUZ, JUAN P"
    given_first = "JUAN P DELA CRUZ"
    assert fuzz.token_set_ratio(family_first, given_first) == fuzz.token_set_ratio(given_first, family_first)
    assert fuzz.token_set_ratio(family_first, given_first) >= config.NAME_MATCH_THRESHOLD


def test_id_full_name_display_is_order_insensitive_for_matching():
    result = _id_stub(
        family_name=ExtractedField(value="REYES"),
        first_name=ExtractedField(value="MARIA"),
        middle_name=ExtractedField(value="SANTOS"),
    )
    family_first = result.full_name_display()
    given_first = "MARIA SANTOS REYES"
    assert fuzz.token_set_ratio(family_first, given_first) >= config.NAME_MATCH_THRESHOLD


def test_full_name_display_empty_without_family_name():
    result = _nbi_stub()  # all ExtractedField() defaults — family_name.value is None
    assert result.full_name_display() == ""


# --- Cross-check against Phase 1's mock dataset ------------------------------

def test_every_mock_nbi_identity_passes_its_own_registry_validators():
    """Catches 'the dataset generator and the registry silently disagree on
    format' before it shows up as phantom Rule-3 failures in a full eval
    run (CV_INTEGRATION.md Phase 3)."""
    _requires_mock_dataset()
    spec = doctypes.get_spec(DocType.NBI_CLEARANCE)
    checked = 0
    for path in sorted(MOCK_DIR.glob("nbi_id*.expected.json")):
        data = json.loads(path.read_text())
        for field in spec.fields:
            assert field.validator(data.get(field.name)), (
                f"{path.name} fails registry validator for field '{field.name}' "
                f"(value={data.get(field.name)!r})"
            )
        checked += 1
    assert checked == 8  # 8 identities


def test_every_mock_id_identity_passes_its_own_registry_validators():
    _requires_mock_dataset()
    spec = doctypes.get_spec(DocType.GOVERNMENT_ID)
    checked = 0
    for path in sorted(MOCK_DIR.glob("id_id*.expected.json")):
        data = json.loads(path.read_text())
        for field in spec.fields:
            if field.name == "id_number":
                continue  # per-id_type dispatch checked separately below
            assert field.validator(data.get(field.name)), (
                f"{path.name} fails registry validator for field '{field.name}' "
                f"(value={data.get(field.name)!r})"
            )
        id_validator = doctypes.id_number_validator_for(data["id_type"])
        assert id_validator(data["id_number"]), (
            f"{path.name} (id_type={data['id_type']}) fails its own id_number pattern: "
            f"{data['id_number']!r}"
        )
        checked += 1
    assert checked == 8
