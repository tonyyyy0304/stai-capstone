"""Declarative document-type registry (Component 14, CV_INTEGRATION.md §2.5).

One entry per document type: required/optional fields, per-field format
validators (feeding Rule 3 in src/guardrails/doc_validation.py), and the
extraction-prompt hints baked into NBI_EXTRACTION_PROMPT / ID_EXTRACTION_PROMPT
(src/agent/prompts.py). Static config, not a wire format — frozen dataclasses,
not Pydantic.

Adding a new document type (e.g. BIR 2316) is a new REGISTRY entry, not new
code — that's the "generalizable" claim this project makes, held to.

Government ID (CV_INTEGRATION.md §1.4a) is ONE DocTypeSpec covering three
real layouts (National ID / Driver's License / Passport), not three separate
registry entries — but two of its fields genuinely differ per id_type:
  - `id_number`'s format (config.ID_NUMBER_PATTERNS) — the static FieldSpec
    validator below only checks non-emptiness; id_number_validator_for(id_type)
    is the real dispatcher, called directly by doc_validation.py's Rule 3
    once it has the extracted id_type. FieldSpec.validator's single-string
    signature has nowhere to carry that context, so it isn't forced there.
  - `expiry_date`'s requiredness (required for drivers_license/passport,
    not for national_id) — required_fields() takes an optional id_type
    parameter for exactly this branch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

from src import config
from src.schemas import DocType, IdType


@dataclass(frozen=True)
class FieldSpec:
    name: str
    required: bool
    validator: Callable[[str | None], bool]
    prompt_hint: str  # goes into the extraction prompt


@dataclass(frozen=True)
class DocTypeSpec:
    doc_type: DocType
    canonical_name: str
    aliases: tuple[str, ...]
    fields: tuple[FieldSpec, ...]
    validity_months: int | None


# --- Shared validators ---------------------------------------------------
# All accept the model's raw extracted string (or None) and return bool.
# "Optional" variants exist for fields where FieldSpec.required=False (e.g.
# middle_name) — empty/None should pass there, not be indistinguishable
# from a malformed value.

_NAME_PATTERN = re.compile(r"^[A-ZÑ][A-ZÑ .'-]*$")


def is_valid_name(value: str | None) -> bool:
    """>=1 alpha token, allows multi-word compounds (DELA CRUZ, DE LA CRUZ,
    SAN JUAN), no digits, allows ñ and hyphens/apostrophes. Empty/None
    fails — for family_name/first_name (required). middle_name's
    optionality is handled by FieldSpec.required + is_valid_optional_name,
    not by this validator accepting empty."""
    if not value or not value.strip():
        return False
    v = value.strip().upper()
    if any(ch.isdigit() for ch in v):
        return False
    return bool(_NAME_PATTERN.match(v))


def is_valid_optional_name(value: str | None) -> bool:
    """Same shape as is_valid_name, but empty/None passes — not every legal
    name has a middle name (CV_INTEGRATION.md §2.5)."""
    if value is None or not value.strip():
        return True
    return is_valid_name(value)


def _parse_date(value: str | None) -> date | None:
    """Accepts a bare ISO date OR an ISO datetime. `date_printed` genuinely
    carries a timestamp on real documents (confirmed against two specimens,
    e.g. "31 July 2024 01:26 PM"), and Gemini extracts it as such when the
    prompt hint says "date/time" — found live during Phase 4 verification,
    where a real extraction's date_printed came back "2026-02-14 09:00:00"
    and failed a date-only parse. Fixed here rather than by telling the
    model to discard real information."""
    if not value:
        return None
    text = value.strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def is_valid_past_or_present_date(value: str | None) -> bool:
    """Parses to a real date, not in the future. For date_of_birth,
    date_printed, issue_date — dates that describe something that already
    happened."""
    parsed = _parse_date(value)
    return parsed is not None and parsed <= date.today()


def is_valid_optional_past_or_present_date(value: str | None) -> bool:
    if value is None or not value.strip():
        return True
    return is_valid_past_or_present_date(value)


def is_valid_date(value: str | None) -> bool:
    """Parses to a real date — no future/past constraint. For valid_until
    and expiry_date, which are legitimately in the future."""
    return _parse_date(value) is not None


def is_valid_optional_date(value: str | None) -> bool:
    if value is None or not value.strip():
        return True
    return is_valid_date(value)


def is_non_empty(value: str | None) -> bool:
    return bool(value and value.strip())


def is_valid_nbi_reference_no(value: str | None) -> bool:
    """Two alphanumeric groups joined by a dash, ~6-12 chars each — shape
    confirmed against two specimen samples (CV_INTEGRATION.md Part 5 item 1),
    neither confirmed-authentic. Deliberately permissive until the real
    subset exists; do not tighten off n=2."""
    if not value:
        return False
    return bool(re.match(r"^[A-Z0-9]{6,12}-[A-Z0-9]{6,12}$", value.strip().upper()))


def is_valid_remarks(value: str | None) -> bool:
    """Normalized value must be in config.NBI_CLEAN_REMARKS. This is the
    check that catches an actual derogatory-record hit — an unrecognized
    value is never assumed clean (CV_INTEGRATION.md §1.4/§2.7)."""
    if not value:
        return False
    return value.strip().upper() in config.NBI_CLEAN_REMARKS


def id_number_validator_for(id_type: IdType | str) -> Callable[[str | None], bool]:
    """The real per-id_type id_number format check — see module docstring
    for why this isn't a static FieldSpec.validator. doc_validation.py's
    Rule 3 calls this directly with the extracted id_type."""
    value = id_type.value if isinstance(id_type, IdType) else id_type
    pattern_str = config.ID_NUMBER_PATTERNS.get(value)
    if pattern_str is None:
        return is_non_empty  # IdType.OTHER or unrecognized — fail toward completeness-only, not a crash
    pattern = re.compile(pattern_str)

    def _validate(v: str | None) -> bool:
        if not v:
            return False
        return bool(pattern.match(v.strip().upper()))

    return _validate


def is_valid_id_type(value: str | None) -> bool:
    return value in {t.value for t in IdType}


# --- Registry ------------------------------------------------------------

NBI_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("family_name", True, is_valid_name, "The family/last name, printed under FAMILY NAME."),
    FieldSpec("first_name", True, is_valid_name, "The first/given name, printed under FIRST NAME."),
    FieldSpec("middle_name", False, is_valid_optional_name, "The middle name, printed under MIDDLE NAME, if present."),
    FieldSpec("date_of_birth", True, is_valid_past_or_present_date, "ISO-8601 date printed under DATE OF BIRTH."),
    FieldSpec("reference_no", True, is_valid_nbi_reference_no, "The printed NBI ID NO, verbatim."),
    FieldSpec("date_printed", True, is_valid_past_or_present_date, "ISO-8601 date/time printed under Date Printed."),
    FieldSpec("valid_until", True, is_valid_date, "ISO-8601 date printed under VALID UNTIL — the document's own expiry."),
    FieldSpec("purpose", True, is_non_empty, "The stated purpose, printed under PURPOSE."),
    FieldSpec("remarks", True, is_valid_remarks, "The clearance result, printed under REMARKS — quote verbatim."),
)

ID_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("id_type", True, is_valid_id_type, "Classify: national_id, drivers_license, passport, or other."),
    FieldSpec("family_name", True, is_valid_name, "The family/last name."),
    FieldSpec("first_name", True, is_valid_name, "The first/given name."),
    FieldSpec("middle_name", False, is_valid_optional_name, "The middle name, if present."),
    FieldSpec("date_of_birth", True, is_valid_past_or_present_date, "ISO-8601 date of birth."),
    FieldSpec("id_number", True, is_non_empty,
              "The printed ID number (PSN/PCN, License No., or Passport No.). "
              "Format regex dispatch is per-id_type — see id_number_validator_for()."),
    FieldSpec("issue_date", False, is_valid_optional_past_or_present_date, "ISO-8601 date of issue, if printed."),
    FieldSpec("expiry_date", False, is_valid_optional_date,
              "ISO-8601 expiry date. Required for drivers_license/passport, "
              "not for national_id — see required_fields()."),
)

REGISTRY: dict[DocType, DocTypeSpec] = {
    DocType.NBI_CLEARANCE: DocTypeSpec(
        doc_type=DocType.NBI_CLEARANCE,
        canonical_name="NBI Clearance",
        aliases=("National Bureau of Investigation Clearance", "NBI Certificate"),
        fields=NBI_FIELDS,
        validity_months=config.NBI_VALIDITY_MONTHS,
    ),
    DocType.GOVERNMENT_ID: DocTypeSpec(
        doc_type=DocType.GOVERNMENT_ID,
        canonical_name="Government-Issued ID",
        aliases=("National ID", "PhilSys ID", "Driver's License", "Passport"),
        fields=ID_FIELDS,
        # Varies by id_type and comes from the document's own printed expiry,
        # not a fixed employer policy window the way NBI_VALIDITY_MONTHS is.
        validity_months=None,
    ),
}


def get_spec(doc_type: DocType) -> DocTypeSpec | None:
    return REGISTRY.get(doc_type)


def required_fields(doc_type: DocType, id_type: IdType | None = None) -> tuple[str, ...]:
    """Required field names for `doc_type`. For DocType.GOVERNMENT_ID,
    `id_type` additionally gates expiry_date (required for
    drivers_license/passport, never for national_id)."""
    spec = get_spec(doc_type)
    if spec is None:
        return ()
    names = [f.name for f in spec.fields if f.required]
    if doc_type == DocType.GOVERNMENT_ID and id_type in (IdType.DRIVERS_LICENSE, IdType.PASSPORT):
        names.append("expiry_date")
    return tuple(names)
