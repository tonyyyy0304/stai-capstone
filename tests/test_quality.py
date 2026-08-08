import numpy as np
import pytest

from src import config
from src.ocr import quality
from src.schemas import QualityVerdict

MOCK_DIR = config.MOCK_DOCS_DIR


def _requires_mock_dataset():
    if not MOCK_DIR.exists() or not any(MOCK_DIR.glob("*.png")):
        pytest.skip("mock dataset not generated — run scripts/make_onboarding_docs.py first")


def _load(name: str) -> np.ndarray:
    return quality.load_image((MOCK_DIR / name).read_bytes())


# --- assess() against real generated variants --------------------------------

@pytest.mark.parametrize("stem", [f"nbi_id{i:02d}" for i in range(1, 9)] + [f"id_id{i:02d}" for i in range(1, 9)])
def test_clean_variant_passes(stem):
    _requires_mock_dataset()
    report = quality.assess(_load(f"{stem}_clean.png"))
    assert report.verdict == QualityVerdict.PASS
    assert report.reasons == []


@pytest.mark.parametrize("stem", [f"nbi_id{i:02d}" for i in range(1, 9)] + [f"id_id{i:02d}" for i in range(1, 9)])
def test_blur_variant_rejects(stem):
    _requires_mock_dataset()
    report = quality.assess(_load(f"{stem}_blur.png"))
    assert report.verdict == QualityVerdict.REJECT
    assert report.blur_score < config.BLUR_VARIANCE_FLOOR


def test_clean_and_blur_blur_scores_do_not_overlap():
    """The calibration invariant Phase 2 requires: no clean image should
    ever score lower than the worst blur image."""
    _requires_mock_dataset()
    clean_scores = [quality.assess(_load(f"nbi_id{i:02d}_clean.png")).blur_score for i in range(1, 9)]
    blur_scores = [quality.assess(_load(f"nbi_id{i:02d}_blur.png")).blur_score for i in range(1, 9)]
    assert min(clean_scores) > max(blur_scores)


def test_lowres_jpeg_rejects_on_resolution_floor():
    _requires_mock_dataset()
    report = quality.assess(_load("nbi_id01_lowres_jpeg.jpg"))
    assert report.verdict == QualityVerdict.REJECT
    assert report.min_dim_px < config.MIN_IMAGE_DIM_PX


def test_blank_page_negative_rejects():
    """A blank page has zero edges (blur_score ~0) and no document-shaped
    contour — the quality gate should catch it without ever reaching Gemini."""
    _requires_mock_dataset()
    report = quality.assess(_load("nbi_neg_blank_page_1.png"))
    assert report.verdict == QualityVerdict.REJECT
    assert report.quad_found is False


# --- Verdict boundary cases (no image needed — pure signal->verdict logic) ---

def _signals(**overrides):
    base = {
        "blur_score": config.BLUR_VARIANCE_WARN + 1,
        "exposure_clip": 0.0,
        "skew_deg": 0.0,
        "min_dim_px": config.MIN_IMAGE_DIM_WARN + 1,  # above the warn tier too, not just the reject floor
        "quad_found": True,
    }
    base.update(overrides)
    return base


def test_blur_score_exactly_at_floor_does_not_reject():
    """`< floor` rejects, so a score exactly at the floor should not — the
    boundary is exclusive on the low side, and this is the test that pins
    that down rather than leaving it to be discovered later."""
    verdict, reasons, _ = quality._verdict(_signals(blur_score=config.BLUR_VARIANCE_FLOOR))
    assert verdict != QualityVerdict.REJECT


def test_blur_score_just_below_floor_rejects():
    verdict, reasons, score = quality._verdict(_signals(blur_score=config.BLUR_VARIANCE_FLOOR - 0.01))
    assert verdict == QualityVerdict.REJECT
    assert score == 0.0


def test_blur_score_between_floor_and_warn_is_warn_not_reject():
    midpoint = (config.BLUR_VARIANCE_FLOOR + config.BLUR_VARIANCE_WARN) / 2
    verdict, reasons, _ = quality._verdict(_signals(blur_score=midpoint))
    assert verdict == QualityVerdict.WARN


def test_skew_exactly_at_max_does_not_reject():
    verdict, _, _ = quality._verdict(_signals(skew_deg=config.MAX_SKEW_DEG))
    assert verdict != QualityVerdict.REJECT


def test_skew_just_beyond_max_rejects():
    verdict, _, score = quality._verdict(_signals(skew_deg=config.MAX_SKEW_DEG + 0.01))
    assert verdict == QualityVerdict.REJECT
    assert score == 0.0


def test_min_dim_below_floor_rejects():
    verdict, _, score = quality._verdict(_signals(min_dim_px=config.MIN_IMAGE_DIM_PX - 1))
    assert verdict == QualityVerdict.REJECT
    assert score == 0.0


def test_min_dim_between_floor_and_warn_is_warn_not_reject():
    """The floor/warn split added 2026-08-09 after a real 518px specimen
    (between the new floor=400 and the old floor, now warn=640) extracted
    at 0.98-0.99 confidence on every field — see config.py's comment."""
    midpoint = (config.MIN_IMAGE_DIM_PX + config.MIN_IMAGE_DIM_WARN) // 2
    verdict, reasons, _ = quality._verdict(_signals(min_dim_px=midpoint))
    assert verdict == QualityVerdict.WARN
    assert any("min_dim_px" in r for r in reasons)


def test_min_dim_exactly_at_warn_does_not_warn():
    verdict, reasons, score = quality._verdict(_signals(min_dim_px=config.MIN_IMAGE_DIM_WARN))
    assert verdict == QualityVerdict.PASS
    assert reasons == []
    assert score == 1.0


def test_no_quad_found_is_not_a_hard_reject_alone():
    """Missing a document-shaped contour lowers confidence but isn't, by
    itself, a floor breach the way blur/skew/resolution are."""
    verdict, reasons, score = quality._verdict(_signals(quad_found=False))
    assert verdict != QualityVerdict.REJECT
    assert score < 1.0


def test_all_signals_clean_yields_pass_with_no_reasons():
    verdict, reasons, score = quality._verdict(_signals())
    assert verdict == QualityVerdict.PASS
    assert reasons == []
    assert score == 1.0


# --- preprocess() ----------------------------------------------------------

def test_preprocess_reduces_skew_below_warn_threshold():
    _requires_mock_dataset()
    image = _load("nbi_id01_skew.png")
    before = quality.assess(image)
    assert before.skew_deg > config.SKEW_WARN_DEG  # sanity: fixture actually is skewed

    fixed = quality.preprocess(image)
    after = quality.assess(fixed)
    assert after.skew_deg < config.SKEW_WARN_DEG


# --- No network dependency --------------------------------------------------

def test_assess_has_no_network_dependency(monkeypatch):
    """quality.py must never need GEMINI_API_KEY — it's OpenCV-only, and
    that's the whole reason it's the quota-relief mechanism (CV_INTEGRATION.md
    §1.2)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _requires_mock_dataset()
    report = quality.assess(_load("nbi_id01_clean.png"))
    assert report.verdict == QualityVerdict.PASS


# --- load_image --------------------------------------------------------------

def test_load_image_rejects_garbage_bytes():
    with pytest.raises(ValueError):
        quality.load_image(b"not an image")
