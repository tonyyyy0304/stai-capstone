"""Generate the mock onboarding-document dataset (Component 14, PLAN.md §4.2,
CV_INTEGRATION.md §2.9/§1.4a).

Renders 8 synthetic identities as **two** document types each — an NBI
Clearance and a government ID (National ID / Driver's License / Passport,
distributed across identities) — every document in 5 degradation variants,
plus 10 labeled negative cases per document type (20 total). Every variant
of one identity+doc_type shares a single ground-truth file, so accuracy
differences between variants are attributable to image quality alone —
that's what makes the preprocessing ablation (quality.preprocess() on/off)
a controlled comparison rather than a headline number pulled from noise.

None of these are real people. `identities.json` doubles as the mock
"faculty record" backing Rule 4 (identity match) in doc_validation.py, and
carries date_of_birth for the NBI<->ID cross-document check.

NBI layout matches the full real field set (reviewed against two specimen
samples, data/references/samples/ and data/references/real/), not just the
9 fields src/ocr/extractor.py actually reads: NBI ID NO, VALID UNTIL, FAMILY/
FIRST/MIDDLE NAME, ADDRESS, DATE OF BIRTH, PLACE OF BIRTH, CITIZENSHIP,
CIVIL STATUS, GENDER, PURPOSE, REMARKS, plus photo/signature/thumbprint/QR/
barcode placeholders and the Agency/CASID/O.R. No/O.R. Date/DATID/BIOID/
RECID/DST PAID/INTID/PRTID metadata grid (real fields on the actual
document, usually mostly blank with a few filled — matched here, not a
generator artifact). This matters for calibration validity, not just
extraction correctness: a mock that's visually sparser than a real document
makes "mock accuracy" measure an easier task than the real one, which
would confound the real-vs-mock gap this project is supposed to measure
honestly. Fields beyond the 9 that ExtractionResult reads are rendered for
visual/distractor density only — never asserted against in *.expected.json,
since ground truth there tracks what's actually extracted.

NBI Clearance, National ID, and Driver's License are all landscape in
reality — the original renderer used one portrait canvas for everything,
which was wrong for three of the four layouts. Fixed here (passport stays
portrait, correctly).

Degradations are deliberately simple (rotation as a skew proxy, Gaussian
blur, a radial brightness overlay for glare, JPEG-quality downscaling) —
they exercise the quality gate's five signals, not photorealism. The real
subset (data/references/real/, gitignored) is what measures the gap between
this and an actual phone photo (PLAN.md §3.5). Layout references for both
document types live in data/references/samples/ (specimen/demo images,
also gitignored — see CV_INTEGRATION.md Part 5).

Filename convention: nbi_<identity_id>_<variant>.png / id_<identity_id>_<variant>.png,
plus nbi_neg_<kind>_<n>.png / id_neg_<kind>_<n>.png for negatives.

Usage:
    python scripts/make_onboarding_docs.py
    python scripts/make_onboarding_docs.py --out /tmp/rerun --seed 20260808
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "references" / "mock"
DEFAULT_SEED = 20260808  # today's date at plan time — arbitrary but fixed, not "random"

# Per-doc-type canvas sizes. NBI/National ID/Driver's License are landscape
# in reality (the original single portrait IMAGE_SIZE was wrong for three
# of the four layouts); passport bio pages are genuinely portrait.
NBI_SIZE = (1400, 950)
NATIONAL_ID_SIZE = (1350, 850)
DRIVERS_LICENSE_SIZE = (1400, 880)
PASSPORT_SIZE = (1000, 1400)
ID_SIZES = {
    "national_id": NATIONAL_ID_SIZE,
    "drivers_license": DRIVERS_LICENSE_SIZE,
    "passport": PASSPORT_SIZE,
}

BG_COLOR = (250, 248, 245)
INK_COLOR = (25, 25, 25)
RULE_COLOR = (140, 140, 140)

VARIANTS = ("clean", "skew", "blur", "glare", "lowres_jpeg")
NBI_NEGATIVE_TYPES = (
    "non_nbi_document",
    "non_document_photo",
    "blank_page",
    "wrong_person",
    "expired",
)
ID_NEGATIVE_TYPES = (
    "non_id_document",
    "blank_page",
    "cross_name_mismatch",
    "cross_dob_mismatch",
    "expired_id",
)

# 8 fictional identities, each with both an NBI Clearance and one of the
# three government-ID layouts. id01 intentionally matches the scenario
# walked through in CV_PIPELINE_WALKTHROUGH.md, so the docs read as one
# story. date_printed values are deliberately recent relative to "today"
# (2026-08-08, the date this plan was written against) so NBI_VALIDITY_MONTHS
# (6) doesn't accidentally expire a "clean" fixture — except id01, which is
# intentionally near its 6-month boundary, matching the walkthrough.
#
# nbi_render_extra: fields the real document prints but ExtractionResult
# deliberately never reads (address, place of birth, citizenship, civil
# status, gender — see CV_INTEGRATION.md §1.7's minimization stance).
# Rendered for visual/distractor density only, never written to
# *.expected.json, never part of any validation rule.
IDENTITIES = [
    {
        "identity_id": "id01",
        "employee_id": "EMP-04821",
        "full_name": "REYES, MARIA SANTOS",
        "family_name": "REYES",
        "first_name": "MARIA",
        "middle_name": "SANTOS",
        "date_of_birth": "1990-01-01",
        "nbi": {
            "reference_no": "NBI-2026-00457821",
            "date_printed": "2026-02-14",
            "valid_until": "2027-02-14",
            "purpose": "Employment",
            "remarks": "NO DEROGATORY",
        },
        "nbi_render_extra": {
            "address": "45 Mabini Street, Brgy. Poblacion, Quezon City",
            "place_of_birth": "Manila",
            "citizenship": "Filipino",
            "civil_status": "Single",
            "gender": "Female",
        },
        "id_type": "national_id",
        "id": {"id_number": "1234-5678-9101-0001", "issue_date": "2019-06-14", "expiry_date": None},
    },
    {
        "identity_id": "id02",
        "employee_id": "EMP-05122",
        "full_name": "DELA CRUZ, JUAN P",
        "family_name": "DELA CRUZ",
        "first_name": "JUAN",
        "middle_name": "P",
        "date_of_birth": "1988-05-12",
        "nbi": {
            "reference_no": "NBI-2026-01123344",
            "date_printed": "2026-05-01",
            "valid_until": "2027-05-01",
            "purpose": "Employment",
            "remarks": "NO DEROGATORY",
        },
        "nbi_render_extra": {
            "address": "12 Rizal Avenue, Brgy. San Antonio, Makati City",
            "place_of_birth": "Cebu City",
            "citizenship": "Filipino",
            "civil_status": "Married",
            "gender": "Male",
        },
        "id_type": "national_id",
        "id": {"id_number": "1234-5678-9102-0002", "issue_date": "2019-08-20", "expiry_date": None},
    },
    {
        "identity_id": "id03",
        "employee_id": "EMP-05310",
        "full_name": "SANTOS, ANA LIZA M",
        "family_name": "SANTOS",
        "first_name": "ANA LIZA",
        "middle_name": "M",
        "date_of_birth": "1995-11-20",
        "nbi": {
            "reference_no": "NBI-2025-00987651",
            "date_printed": "2026-06-05",
            "valid_until": "2027-06-05",
            "purpose": "Local Employment",
            "remarks": "NO DEROGATORY",
        },
        "nbi_render_extra": {
            "address": "78 Bonifacio Street, Brgy. Bagong Silang, Caloocan City",
            "place_of_birth": "Davao City",
            "citizenship": "Filipino",
            "civil_status": "Single",
            "gender": "Female",
        },
        "id_type": "national_id",
        "id": {"id_number": "1234-5678-9103-0003", "issue_date": "2020-01-10", "expiry_date": None},
    },
    {
        "identity_id": "id04",
        "employee_id": "EMP-05477",
        "full_name": "GARCIA, PEDRO JR",
        "family_name": "GARCIA",
        "first_name": "PEDRO",
        "middle_name": "JR",
        "date_of_birth": "1987-10-04",
        "nbi": {
            "reference_no": "NBI-2026-01345567",
            "date_printed": "2026-06-10",
            "valid_until": "2027-06-10",
            "purpose": "Employment",
            "remarks": "NO DEROGATORY",
        },
        "nbi_render_extra": {
            "address": "23 Luna Street, Brgy. Maligaya, Pasig City",
            "place_of_birth": "Baguio City",
            "citizenship": "Filipino",
            "civil_status": "Married",
            "gender": "Male",
        },
        "id_type": "drivers_license",
        "id": {"id_number": "N04-12-000004", "issue_date": "2017-11-24", "expiry_date": "2028-10-04"},
    },
    {
        "identity_id": "id05",
        "employee_id": "EMP-05602",
        "full_name": "MENDOZA, CARLA B",
        "family_name": "MENDOZA",
        "first_name": "CARLA",
        "middle_name": "B",
        "date_of_birth": "1992-03-15",
        "nbi": {
            "reference_no": "NBI-2026-00223311",
            "date_printed": "2026-05-20",
            "valid_until": "2027-05-20",
            "purpose": "Employment",
            "remarks": "NO DEROGATORY",
        },
        "nbi_render_extra": {
            "address": "56 Aguinaldo Highway, Brgy. San Roque, Cavite City",
            "place_of_birth": "Iloilo City",
            "citizenship": "Filipino",
            "civil_status": "Single",
            "gender": "Female",
        },
        "id_type": "drivers_license",
        "id": {"id_number": "N05-12-000005", "issue_date": "2019-06-01", "expiry_date": "2029-06-01"},
    },
    {
        "identity_id": "id06",
        "employee_id": "EMP-05789",
        "full_name": "TORRES, RICARDO III",
        "family_name": "TORRES",
        "first_name": "RICARDO",
        "middle_name": "III",
        "date_of_birth": "1984-07-22",
        "nbi": {
            "reference_no": "NBI-2026-00778899",
            "date_printed": "2026-04-15",
            "valid_until": "2027-04-15",
            "purpose": "Local Employment",
            "remarks": "NO DEROGATORY",
        },
        "nbi_render_extra": {
            "address": "9 Quezon Avenue, Brgy. Sta. Cruz, Manila",
            "place_of_birth": "Zamboanga City",
            "citizenship": "Filipino",
            "civil_status": "Married",
            "gender": "Male",
        },
        "id_type": "drivers_license",
        "id": {"id_number": "N06-12-000006", "issue_date": "2016-02-10", "expiry_date": "2028-02-10"},
    },
    {
        "identity_id": "id07",
        "employee_id": "EMP-05890",
        "full_name": "FLORES, KATRINA D",
        "family_name": "FLORES",
        "first_name": "KATRINA",
        "middle_name": "D",
        "date_of_birth": "1980-03-16",
        "nbi": {
            "reference_no": "NBI-2026-01556677",
            "date_printed": "2026-07-01",
            "valid_until": "2027-07-01",
            "purpose": "Employment",
            "remarks": "NO DEROGATORY",
        },
        "nbi_render_extra": {
            "address": "34 Kalayaan Street, Brgy. Central, Quezon City",
            "place_of_birth": "Bacolod City",
            "citizenship": "Filipino",
            "civil_status": "Single",
            "gender": "Female",
        },
        "id_type": "passport",
        "id": {"id_number": "P0000007A", "issue_date": "2021-06-27", "expiry_date": "2031-06-26"},
    },
    {
        "identity_id": "id08",
        "employee_id": "EMP-05991",
        "full_name": "RAMOS, ELMER S",
        "family_name": "RAMOS",
        "first_name": "ELMER",
        "middle_name": "S",
        "date_of_birth": "1975-09-15",
        "nbi": {
            "reference_no": "NBI-2025-00445566",
            "date_printed": "2026-03-10",
            "valid_until": "2027-03-10",
            "purpose": "Employment",
            "remarks": "NO DEROGATORY",
        },
        "nbi_render_extra": {
            "address": "67 EDSA, Brgy. Guadalupe, Mandaluyong City",
            "place_of_birth": "Cagayan de Oro",
            "citizenship": "Filipino",
            "civil_status": "Married",
            "gender": "Male",
        },
        "id_type": "passport",
        "id": {"id_number": "P0000008A", "issue_date": "2020-01-15", "expiry_date": "2030-01-14"},
    },
]


# --- Fonts -------------------------------------------------------------------
# No TTF asset is shipped with the repo — Pillow's scalable default bitmap
# font (load_default(size=...), available since Pillow 10.1) keeps this
# script dependency-free and identical across machines/OSes.

def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _display_date(iso: str | None) -> str:
    if not iso:
        return ""
    y, m, d = iso.split("-")
    return f"{m}/{d}/{y}"


def _frame(img: Image.Image) -> ImageDraw.ImageDraw:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, w - 20, h - 20], outline=RULE_COLOR, width=3)
    return draw


# --- Visual placeholders (photo/signature/QR/fingerprint/barcode/metadata) ---
# Deliberately simple, not scannable/checksum-valid — these exist to give
# the mock the same visual density and distractor-field count as a real
# document, since ExtractionResult never reads any of them. See the module
# docstring for why sparse mocks would bias the real-vs-mock accuracy gap.

def _control_number(identity: dict) -> str:
    seed = int(identity["employee_id"].split("-")[1])
    return str(10_000_000 + seed % 90_000_000).zfill(8)


def _metadata_block(identity: dict) -> dict[str, str]:
    """Agency/CASID/O.R. No/O.R. Date/DATID/BIOID/RECID/DST PAID/INTID/PRTID
    — real fields on the printed document (confirmed against a real
    specimen, not a generator artifact), usually mostly blank with a few
    filled in. Two of the six optional slots are filled per identity,
    deterministically, so the pattern is reproducible but varies."""
    idx = int(identity["identity_id"][2:]) - 1
    optional_slots = ("CASID", "DATAID", "BIOID", "RECID", "INTID", "PRTID")
    filled = {idx % 6, (idx + 3) % 6}
    values = {
        slot: (f"{identity['first_name'][:4].lower()}{idx + 1}" if i in filled else "")
        for i, slot in enumerate(optional_slots)
    }
    return {
        "Agency": f"L{idx + 1:02d}",
        "O.R. No": f"{identity['family_name'][:3].upper()}{identity['employee_id'][-4:]}",
        "O.R. Date": _display_date(identity["nbi"]["date_printed"]),
        "DST PAID": "30.00",
        **values,
    }


def _draw_qr_placeholder(draw: ImageDraw.ImageDraw, rng: random.Random, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=RULE_COLOR, width=1)
    cell = 6
    for yy in range(y0 + 4, y1 - 4, cell):
        for xx in range(x0 + 4, x1 - 4, cell):
            if rng.random() > 0.5:
                draw.rectangle([xx, yy, xx + cell - 1, yy + cell - 1], fill=INK_COLOR)


def _draw_fingerprint_placeholder(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=RULE_COLOR, width=1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    max_r = min(x1 - x0, y1 - y0) / 2 - 4
    r = 4
    while r < max_r:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=INK_COLOR, width=1)
        r += 5


def _draw_barcode(draw: ImageDraw.ImageDraw, rng: random.Random, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    x = x0
    while x < x1:
        bar_w = rng.choice([2, 3, 5])
        if rng.random() > 0.5:
            draw.rectangle([x, y0, x + bar_w, y1], fill=INK_COLOR)
        x += bar_w + rng.choice([1, 2])


# --- Rendering: NBI Clearance ------------------------------------------------

def render_nbi_clean(identity: dict) -> Image.Image:
    """NBI-Clearance-style layout matching the full field set confirmed
    against two real specimens (CV_INTEGRATION.md §2.3, Part 5): every
    printed field, not just the 9 ExtractionResult reads. Landscape, like
    the real document — see the module docstring."""
    img = Image.new("RGB", NBI_SIZE, BG_COLOR)
    draw = _frame(img)
    w, h = NBI_SIZE
    nbi = identity["nbi"]
    extra = identity["nbi_render_extra"]
    local_rng = random.Random(identity["identity_id"] + "-nbi")

    draw.text((w / 2, 45), "Republic of the Philippines", font=_font(22), fill=INK_COLOR, anchor="mm")
    draw.text((w / 2, 72), "Department of Justice", font=_font(20), fill=INK_COLOR, anchor="mm")
    draw.text((w / 2, 105), "National Bureau of Investigation", font=_font(26), fill=INK_COLOR, anchor="mm")
    draw.text((w - 60, 55), _control_number(identity), font=_font(20), fill=(180, 40, 40), anchor="rm")
    draw.text((60, 130), "This is to certify that the person whose name, picture, signature and thumbprint",
               font=_font(13), fill=RULE_COLOR)
    draw.text((60, 148), "appearing below applied for NBI Clearance and the results is as follows:",
               font=_font(13), fill=RULE_COLOR)
    draw.line([60, 168, w - 60, 168], fill=RULE_COLOR, width=2)

    left_fields = [
        ("NBI ID NO", nbi["reference_no"]),
        ("FAMILY NAME", identity["family_name"]),
        ("MIDDLE NAME", identity["middle_name"]),
        ("ADDRESS", extra["address"]),
        ("DATE OF BIRTH", _display_date(identity["date_of_birth"])),
        ("CITIZENSHIP", extra["citizenship"]),
        ("PURPOSE", nbi["purpose"]),
        ("REMARKS", nbi["remarks"]),
    ]
    y = 190
    for label, value in left_fields:
        draw.text((60, y), label, font=_font(15), fill=RULE_COLOR)
        draw.text((60, y + 20), value, font=_font(22), fill=INK_COLOR)
        y += 62

    mid_fields = [
        ("VALID UNTIL", _display_date(nbi["valid_until"])),
        ("FIRST NAME", identity["first_name"]),
        ("PLACE OF BIRTH", extra["place_of_birth"]),
        ("CIVIL STATUS", extra["civil_status"]),
    ]
    y = 190
    for label, value in mid_fields:
        draw.text((560, y), label, font=_font(15), fill=RULE_COLOR)
        draw.text((560, y + 20), value, font=_font(22), fill=INK_COLOR)
        y += 62

    draw.rectangle([1010, 190, 1190, 400], outline=RULE_COLOR, width=2)
    draw.text((1100, 295), "PHOTO", font=_font(20), fill=RULE_COLOR, anchor="mm")
    draw.text((1210, 190), "GENDER", font=_font(15), fill=RULE_COLOR)
    draw.text((1210, 210), extra["gender"], font=_font(20), fill=INK_COLOR)

    draw.rectangle([1010, 410, 1190, 460], outline=RULE_COLOR, width=2)
    draw.text((1100, 435), "SIGNATURE", font=_font(14), fill=RULE_COLOR, anchor="mm")

    _draw_qr_placeholder(draw, local_rng, (1010, 480, 1090, 560))
    _draw_fingerprint_placeholder(draw, (1110, 480, 1190, 560))

    draw.text((1010, 580), "Date Printed:", font=_font(13), fill=RULE_COLOR)
    draw.text((1010, 598), f"{_display_date(nbi['date_printed'])} 09:00 AM", font=_font(15), fill=INK_COLOR)

    my = 630
    for label, value in _metadata_block(identity).items():
        draw.text((1010, my), f"{label}:", font=_font(12), fill=RULE_COLOR)
        draw.text((1120, my), value, font=_font(12), fill=INK_COLOR)
        my += 20

    _draw_barcode(draw, local_rng, (60, h - 140, 500, h - 100))
    draw.text((60, h - 95), nbi["reference_no"], font=_font(14), fill=INK_COLOR)

    draw.line([560, h - 120, 900, h - 120], fill=RULE_COLOR, width=1)
    draw.text((730, h - 100), "Director", font=_font(14), fill=RULE_COLOR, anchor="mm")

    return img


def render_nbi_negative(kind: str, index: int, rng: random.Random) -> tuple[Image.Image, dict]:
    """Returns (image, ground_truth) for one of the 5 NBI negative categories."""
    w, h = NBI_SIZE

    if kind == "non_nbi_document":
        img = Image.new("RGB", NBI_SIZE, BG_COLOR)
        draw = _frame(img)
        draw.text((w / 2, 100), "BARANGAY CLEARANCE", font=_font(34), fill=INK_COLOR, anchor="mm")
        draw.text((w / 2, 160), "OFFICE OF THE BARANGAY CAPTAIN", font=_font(22), fill=INK_COLOR, anchor="mm")
        draw.text((320, 300), f"Name: {rng.choice(IDENTITIES)['full_name']}", font=_font(26), fill=INK_COLOR)
        expected_doc_type, expected_outcome = "unknown_document", "rejected"
        reason = "wrong document type in the NBI Clearance slot"

    elif kind == "non_document_photo":
        img = Image.new("RGB", NBI_SIZE, (0, 0, 0))
        for y in range(h):
            shade = int(255 * (y / h))
            for x0 in range(0, w, 4):
                img.putpixel((x0, y), (shade, int(shade * 0.8), int(shade * 0.6)))
        expected_doc_type, expected_outcome = "not_a_document", "rejected"
        reason = "not a document at all"

    elif kind == "blank_page":
        img = Image.new("RGB", NBI_SIZE, (255, 255, 255))
        expected_doc_type, expected_outcome = "not_a_document", "rejected"
        reason = "blank page, nothing to extract"

    elif kind == "wrong_person":
        base = IDENTITIES[index % len(IDENTITIES)]
        impostor = IDENTITIES[(index + 1) % len(IDENTITIES)]
        fake = dict(base, full_name=impostor["full_name"])
        img = render_nbi_clean(fake)
        expected_doc_type, expected_outcome = "nbi_clearance", "needs_review"
        reason = f"name on document does not match faculty record for {base['employee_id']} (Rule 4)"

    elif kind == "expired":
        base = IDENTITIES[index % len(IDENTITIES)]
        old_date = date.today() - timedelta(days=365 * 3)
        expired_nbi = dict(base["nbi"], date_printed=old_date.isoformat(),
                            valid_until=(old_date + timedelta(days=365)).isoformat())
        fake = dict(base, nbi=expired_nbi)
        img = render_nbi_clean(fake)
        expected_doc_type, expected_outcome = "nbi_clearance", "rejected"
        reason = "valid_until / date_printed+NBI_VALIDITY_MONTHS both outside the validity window (Rule 5)"

    else:
        raise ValueError(f"unknown NBI negative kind: {kind}")

    return img, {
        "category": kind, "expected_doc_type": expected_doc_type,
        "expected_outcome": expected_outcome, "reason": reason,
    }


# --- Rendering: Government ID (§1.4a) ----------------------------------------

def render_national_id_clean(identity: dict) -> Image.Image:
    """PhilSys National ID layout: PCN, Apelyido/Last Name, Mga Pangalan/
    Given Names, Gitnang Apelyido/Middle Name, DOB. No expiry field —
    deliberate, matches the real specimen (CV_INTEGRATION.md §1.4a).
    Landscape — matches the real card, unlike the original portrait render."""
    size = NATIONAL_ID_SIZE
    img = Image.new("RGB", size, (235, 244, 250))
    draw = _frame(img)
    w = size[0]
    id_data = identity["id"]

    draw.text((w / 2, 60), "REPUBLIKA NG PILIPINAS", font=_font(24), fill=INK_COLOR, anchor="mm")
    draw.text((w / 2, 90), "Republic of the Philippines", font=_font(18), fill=RULE_COLOR, anchor="mm")
    draw.text((w / 2, 130), "PAMBANSANG PAGKAKAKILANLAN", font=_font(26), fill=INK_COLOR, anchor="mm")
    draw.line([80, 160, w - 80, 160], fill=RULE_COLOR, width=2)
    draw.text((80, 190), id_data["id_number"], font=_font(28), fill=INK_COLOR)

    draw.rectangle([80, 260, 280, 500], outline=RULE_COLOR, width=2)
    draw.text((180, 380), "PHOTO", font=_font(22), fill=RULE_COLOR, anchor="mm")

    fields = [
        ("Apelyido / Last Name", identity["family_name"]),
        ("Mga Pangalan / Given Names", identity["first_name"]),
        ("Gitnang Apelyido / Middle Name", identity["middle_name"]),
        ("Petsa ng Kapanganakan / Date of Birth", _display_date(identity["date_of_birth"])),
    ]
    y = 290
    for label, value in fields:
        draw.text((320, y), label, font=_font(18), fill=RULE_COLOR)
        draw.text((320, y + 26), value, font=_font(26), fill=INK_COLOR)
        y += 72

    draw.text((80, y + 20), "Araw ng pagkakaloob / Date of Issue", font=_font(16), fill=RULE_COLOR)
    draw.text((80, y + 44), _display_date(id_data["issue_date"]), font=_font(22), fill=INK_COLOR)

    return img


def render_drivers_license_clean(identity: dict) -> Image.Image:
    """LTO Non-Professional Driver's License layout. Name is one
    concatenated line ("Last, First Middle"), unlike National ID/Passport
    which label the three parts separately — deliberate, matches the real
    specimen (CV_INTEGRATION.md §1.4a/§2.6: the extraction prompt has to
    split this back into three fields). Landscape, matches the real card."""
    size = DRIVERS_LICENSE_SIZE
    img = Image.new("RGB", size, (232, 245, 250))
    draw = _frame(img)
    w, h = size
    id_data = identity["id"]
    given = " ".join(p for p in (identity["first_name"], identity["middle_name"]) if p)

    draw.text((w / 2, 60), "REPUBLIC OF THE PHILIPPINES", font=_font(22), fill=INK_COLOR, anchor="mm")
    draw.text((w / 2, 90), "DEPARTMENT OF TRANSPORTATION", font=_font(20), fill=INK_COLOR, anchor="mm")
    draw.text((w / 2, 118), "LAND TRANSPORTATION OFFICE", font=_font(20), fill=INK_COLOR, anchor="mm")
    draw.text((w / 2, 155), "NON-PROFESSIONAL DRIVER'S LICENSE", font=_font(26), fill=INK_COLOR, anchor="mm")
    draw.line([80, 185, w - 80, 185], fill=RULE_COLOR, width=2)

    draw.rectangle([80, 220, 280, 460], outline=RULE_COLOR, width=2)
    draw.text((180, 340), "PHOTO", font=_font(22), fill=RULE_COLOR, anchor="mm")

    draw.text((320, 230), "Last Name, First Name, Middle Name", font=_font(18), fill=RULE_COLOR)
    draw.text((320, 256), f"{identity['family_name']}, {given}", font=_font(26), fill=INK_COLOR)

    fields = [
        ("Date of Birth", _display_date(identity["date_of_birth"])),
        ("License No.", id_data["id_number"]),
        ("Expiration Date", _display_date(id_data["expiry_date"])),
    ]
    y = 320
    for label, value in fields:
        draw.text((320, y), label, font=_font(18), fill=RULE_COLOR)
        draw.text((320, y + 26), value, font=_font(26), fill=INK_COLOR)
        y += 72

    draw.text((80, h - 200), "Date Issued", font=_font(14), fill=RULE_COLOR)
    draw.text((80, h - 178), _display_date(id_data["issue_date"]), font=_font(18), fill=INK_COLOR)
    draw.text((w - 280, h - 178), "Signature of Licensee", font=_font(14), fill=RULE_COLOR)

    return img


def render_passport_clean(identity: dict) -> Image.Image:
    """DFA Philippine Passport bio page, incl. a simplified two-line MRZ —
    illustrative, NOT checksum-valid, but enough to exercise "prefer the
    MRZ" prompt behavior mentioned in CV_INTEGRATION.md §2.6. Portrait —
    already correct, passport bio pages genuinely are."""
    size = PASSPORT_SIZE
    img = Image.new("RGB", size, (245, 240, 250))
    draw = _frame(img)
    w, h = size
    id_data = identity["id"]

    draw.text((w / 2, 60), "REPUBLIKA NG PILIPINAS", font=_font(20), fill=INK_COLOR, anchor="mm")
    draw.text((w / 2, 86), "REPUBLIC OF THE PHILIPPINES", font=_font(20), fill=INK_COLOR, anchor="mm")
    draw.text((w / 2, 130), "PASAPORTE / PASSPORT", font=_font(28), fill=INK_COLOR, anchor="mm")
    draw.line([80, 160, w - 80, 160], fill=RULE_COLOR, width=2)

    draw.rectangle([80, 190, 280, 430], outline=RULE_COLOR, width=2)
    draw.text((180, 310), "PHOTO", font=_font(22), fill=RULE_COLOR, anchor="mm")

    fields = [
        ("Surname", identity["family_name"]),
        ("Given Names", " ".join(p for p in (identity["first_name"], identity["middle_name"]) if p)),
        ("Nationality", "FILIPINO"),
        ("Date of Birth", _display_date(identity["date_of_birth"])),
        ("Passport No.", id_data["id_number"]),
        ("Date of Issue", _display_date(id_data["issue_date"])),
        ("Date of Expiry", _display_date(id_data["expiry_date"])),
    ]
    y = 210
    for label, value in fields:
        draw.text((320, y), label, font=_font(16), fill=RULE_COLOR)
        draw.text((320, y + 22), value, font=_font(24), fill=INK_COLOR)
        y += 64

    # Simplified MRZ (illustrative only — not checksum-valid).
    mrz1 = f"P<PHL{identity['family_name'].replace(' ', '<')}<<{identity['first_name'].replace(' ', '<')}"
    mrz1 = (mrz1 + "<" * 44)[:44]
    dob_compact = identity["date_of_birth"][2:4] + identity["date_of_birth"][5:7] + identity["date_of_birth"][8:10]
    exp = id_data["expiry_date"] or "00000000"
    exp_compact = exp[2:4] + exp[5:7] + exp[8:10]
    mrz2 = f"{id_data['id_number']}PHL{dob_compact}F{exp_compact}"
    mrz2 = (mrz2 + "<" * 44)[:44]
    draw.rectangle([60, h - 190, w - 60, h - 90], outline=RULE_COLOR, width=2)
    draw.text((80, h - 165), mrz1, font=_font(16), fill=INK_COLOR)
    draw.text((80, h - 130), mrz2, font=_font(16), fill=INK_COLOR)

    return img


_ID_RENDERERS = {
    "national_id": render_national_id_clean,
    "drivers_license": render_drivers_license_clean,
    "passport": render_passport_clean,
}


def render_id_clean(identity: dict) -> Image.Image:
    return _ID_RENDERERS[identity["id_type"]](identity)


def render_id_negative(kind: str, index: int, rng: random.Random) -> tuple[Image.Image, dict]:
    """Returns (image, ground_truth) for one of the 5 government-ID negative
    categories. cross_name_mismatch/cross_dob_mismatch carry `pairs_with`
    naming the NBI fixture they should be validated against — that pairing
    is what exercises validate_cross_document() (CV_INTEGRATION.md §2.7).
    A cross_name_mismatch fixture also happens to mismatch the employee
    record (since the substituted name belongs to a different identity),
    so it doubles as ID-Rule-4 coverage rather than needing a 6th category.
    non_id_document/blank_page use NATIONAL_ID_SIZE as a generic landscape
    canvas — they don't correspond to a specific id_type."""
    w = NATIONAL_ID_SIZE[0]

    if kind == "non_id_document":
        img = Image.new("RGB", NATIONAL_ID_SIZE, BG_COLOR)
        draw = _frame(img)
        draw.text((w / 2, 100), "COMPANY LIBRARY CARD", font=_font(30), fill=INK_COLOR, anchor="mm")
        draw.text((320, 300), f"Name: {rng.choice(IDENTITIES)['full_name']}", font=_font(24), fill=INK_COLOR)
        expected_doc_type, expected_outcome = "unknown_document", "rejected"
        reason = "wrong document type in the government ID slot"
        pairs_with = None

    elif kind == "blank_page":
        img = Image.new("RGB", NATIONAL_ID_SIZE, (255, 255, 255))
        expected_doc_type, expected_outcome = "not_a_document", "rejected"
        reason = "blank page, nothing to extract"
        pairs_with = None

    elif kind == "cross_name_mismatch":
        base = IDENTITIES[index % len(IDENTITIES)]
        impostor = IDENTITIES[(index + 1) % len(IDENTITIES)]
        fake = dict(base, family_name=impostor["family_name"], first_name=impostor["first_name"],
                    middle_name=impostor["middle_name"], full_name=impostor["full_name"])
        img = render_id_clean(fake)
        expected_doc_type, expected_outcome = "government_id", "needs_review"
        reason = "cross-document name mismatch vs. paired NBI Clearance (and vs. employee record)"
        pairs_with = f"nbi_{base['identity_id']}"

    elif kind == "cross_dob_mismatch":
        base = IDENTITIES[index % len(IDENTITIES)]
        shifted_dob = (date.fromisoformat(base["date_of_birth"]) - timedelta(days=365 * 5)).isoformat()
        fake = dict(base, date_of_birth=shifted_dob)
        img = render_id_clean(fake)
        expected_doc_type, expected_outcome = "government_id", "needs_review"
        reason = "cross-document date_of_birth mismatch vs. paired NBI Clearance"
        pairs_with = f"nbi_{base['identity_id']}"

    elif kind == "expired_id":
        # Only meaningful for drivers_license/passport — national_id has no
        # expiry to violate (CV_INTEGRATION.md §2.7 Rule 5 carve-out).
        candidates = [i for i in IDENTITIES if i["id_type"] != "national_id"]
        base = candidates[index % len(candidates)]
        old_expiry = (date.today() - timedelta(days=365)).isoformat()
        fake = dict(base, id=dict(base["id"], expiry_date=old_expiry))
        img = render_id_clean(fake)
        expected_doc_type, expected_outcome = "government_id", "rejected"
        reason = "expiry_date outside the validity window (ID Rule 5)"
        pairs_with = None

    else:
        raise ValueError(f"unknown ID negative kind: {kind}")

    ground_truth = {
        "category": kind, "expected_doc_type": expected_doc_type,
        "expected_outcome": expected_outcome, "reason": reason,
    }
    if pairs_with:
        ground_truth["pairs_with"] = pairs_with
    return img, ground_truth


# --- Degradations (doc-type-agnostic — operate on a PIL Image regardless of content) ---

def apply_skew(img: Image.Image, rng: random.Random) -> Image.Image:
    """Rotation as a skew proxy (8-15 degrees). quality.py's skew_deg signal
    is itself computed from a minAreaRect angle, so a pure rotation gives a
    clean, predictable ground truth for calibrating MAX_SKEW_DEG against."""
    angle = rng.uniform(8, 15) * rng.choice((1, -1))
    return img.rotate(angle, expand=True, fillcolor=BG_COLOR)


def apply_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    """Gaussian blur; radius ~3.5-5.5 approximates the k=7-11 kernel size
    named in CV_INTEGRATION.md §2.9 (kernel ~= 2*radius + 1)."""
    radius = rng.uniform(3.5, 5.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_glare(img: Image.Image, rng: random.Random) -> Image.Image:
    """Radial white overlay + brightness clip, simulating a flash reflection
    off the document surface — targets quality.py's exposure_clip signal."""
    w, h = img.size
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.2, 0.5) * h
    overlay = Image.new("L", (w, h), 0)
    odraw = ImageDraw.Draw(overlay)
    max_r = int(min(w, h) * 0.35)
    for r in range(max_r, 0, -4):
        alpha = int(255 * (1 - r / max_r) ** 2)
        odraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=alpha)
    white = Image.new("RGB", (w, h), (255, 255, 255))
    return Image.composite(white, img, overlay)


def apply_lowres_jpeg(img: Image.Image, path: Path, rng: random.Random) -> None:
    """Downscales to 0.35x (the degradation itself — left small, not
    upscaled back) and saves as JPEG at quality=35. Writes directly since
    the output format differs from every other variant."""
    w, h = img.size
    scaled = img.resize((max(1, int(w * 0.35)), max(1, int(h * 0.35))), Image.BILINEAR)
    scaled.save(path, format="JPEG", quality=35)


def _write_variants(clean: Image.Image, out_dir: Path, stem: str, rng: random.Random) -> None:
    clean.save(out_dir / f"{stem}_clean.png")
    apply_skew(clean, rng).convert("RGB").save(out_dir / f"{stem}_skew.png")
    apply_blur(clean, rng).save(out_dir / f"{stem}_blur.png")
    apply_glare(clean, rng).save(out_dir / f"{stem}_glare.png")
    apply_lowres_jpeg(clean, out_dir / f"{stem}_lowres_jpeg.jpg", rng)


# --- Main --------------------------------------------------------------------

def generate(out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for identity in IDENTITIES:
        iid = identity["identity_id"]

        _write_variants(render_nbi_clean(identity), out_dir, f"nbi_{iid}", rng)
        nbi_expected = {
            "identity_id": iid, "employee_id": identity["employee_id"], "doc_type": "nbi_clearance",
            "full_name": identity["full_name"], "family_name": identity["family_name"],
            "first_name": identity["first_name"], "middle_name": identity["middle_name"],
            "date_of_birth": identity["date_of_birth"], **identity["nbi"],
        }
        (out_dir / f"nbi_{iid}.expected.json").write_text(json.dumps(nbi_expected, indent=2))

        _write_variants(render_id_clean(identity), out_dir, f"id_{iid}", rng)
        id_expected = {
            "identity_id": iid, "employee_id": identity["employee_id"], "doc_type": "government_id",
            "id_type": identity["id_type"], "full_name": identity["full_name"],
            "family_name": identity["family_name"], "first_name": identity["first_name"],
            "middle_name": identity["middle_name"], "date_of_birth": identity["date_of_birth"],
            **identity["id"],
        }
        (out_dir / f"id_{iid}.expected.json").write_text(json.dumps(id_expected, indent=2))

    negatives_manifest = []
    neg_index = 0
    for kind in NBI_NEGATIVE_TYPES:
        for copy_n in range(2):
            neg_index += 1
            filename = f"nbi_neg_{kind}_{copy_n + 1}.png"
            img, ground_truth = render_nbi_negative(kind, neg_index, rng)
            img.convert("RGB").save(out_dir / filename)
            ground_truth["filename"] = filename
            negatives_manifest.append(ground_truth)

    for kind in ID_NEGATIVE_TYPES:
        for copy_n in range(2):
            neg_index += 1
            filename = f"id_neg_{kind}_{copy_n + 1}.png"
            img, ground_truth = render_id_negative(kind, neg_index, rng)
            img.convert("RGB").save(out_dir / filename)
            ground_truth["filename"] = filename
            negatives_manifest.append(ground_truth)

    (out_dir / "negatives.json").write_text(json.dumps(negatives_manifest, indent=2))
    (out_dir / "identities.json").write_text(json.dumps(IDENTITIES, indent=2))

    image_count = len(list(out_dir.glob("*.png"))) + len(list(out_dir.glob("*.jpg")))
    expected_count = len(list(out_dir.glob("*.expected.json")))
    print(f"Wrote {image_count} images to {out_dir}")
    print(f"  {len(IDENTITIES)} identities x 2 doc_types x {len(VARIANTS)} variants "
          f"= {len(IDENTITIES) * 2 * len(VARIANTS)}")
    print(f"  + {len(negatives_manifest)} negatives "
          f"({len(NBI_NEGATIVE_TYPES) * 2} NBI + {len(ID_NEGATIVE_TYPES) * 2} ID)")
    print(f"  identities.json, negatives.json, {expected_count} *.expected.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for reproducible regeneration")
    args = parser.parse_args()
    generate(args.out, args.seed)


if __name__ == "__main__":
    main()
