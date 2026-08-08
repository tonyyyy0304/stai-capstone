"""Image quality gate (Layer 1, Component 14). No LLM call, no network
dependency at all — every signal here is computed by OpenCV against the raw
pixel array. This is both a real accuracy decision (an unreadable photo
can't be extracted from reliably) and the quota mechanism: a `reject`
verdict short-circuits src/ocr/extractor.py before it spends a Gemini call
(CV_INTEGRATION.md §1.2/§2.4).

Five signals feed one verdict:
    blur_score    variance of the Laplacian (sharpness)
    exposure_clip fraction of pixels blown out at 0 or 255 (glare/underexposure)
    skew_deg      rotation of the detected document boundary
    min_dim_px    shorter-side resolution
    quad_found    whether a document-shaped contour was found at all

Signals and the derived verdict are kept separate on purpose (assess()
composes them): thresholds live in src/config.py and are testable without
touching an image, same reasoning as LLMJudgeVerdict's detection/policy
split (src/guardrails/llm_judge.py).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.schemas import ImageQualityReport, QualityVerdict


def load_image(data: bytes) -> np.ndarray:
    """Decodes raw image bytes to a BGR pixel array. This is also the EXIF
    strip point — decoding to a raw array discards all metadata, including
    phone geolocation, as a side effect (CV_INTEGRATION.md §1.7)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image data — not a valid JPEG/PNG")
    return image


# --- Document contour / skew detection ---------------------------------------

def _detect_document(gray: np.ndarray) -> tuple[float, bool, np.ndarray | None]:
    """Returns (signed_angle_deg, quad_found, contour). The largest
    thresholded contour above a minimum-area floor is treated as the
    document boundary; too small or absent means quad_found=False and the
    image has nothing document-shaped to anchor a skew measurement on."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, False, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.05 * gray.size:
        return 0.0, False, None

    rect = cv2.minAreaRect(largest)
    (_, _), (_, _), angle = rect
    # A rectangle's angle is only meaningful mod 90 (four-fold rotational
    # symmetry) — reduce into (-45, 45] rather than branching on rect_w vs
    # rect_h. The branching version double-corrected OpenCV's own w/h-swap
    # convention on an axis-aligned rectangle (angle=90.0 with w<h) and
    # reported skew_deg=90 on perfectly clean images once the mock layout
    # got dense enough for the frame contour's w/h ordering to flip —
    # caught during the NBI mock rework, verified against both a clean and
    # a skewed fixture before landing.
    angle = angle % 90
    if angle > 45:
        angle -= 90
    return float(angle), True, largest


# --- Signals -------------------------------------------------------------------

def _signals(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    exposure_clip = float((hist[0] + hist[255]) / gray.size)

    signed_angle, quad_found, _ = _detect_document(gray)
    skew_deg = abs(signed_angle)

    h, w = gray.shape[:2]
    min_dim_px = int(min(h, w))

    return {
        "blur_score": blur_score,
        "exposure_clip": exposure_clip,
        "skew_deg": skew_deg,
        "min_dim_px": min_dim_px,
        "quad_found": quad_found,
    }


def _verdict(signals: dict) -> tuple[QualityVerdict, list[str], float]:
    """Applies config thresholds. normalized_quality = min() of per-signal
    0..1 scores — same conservative operator as composite confidence
    (CV_INTEGRATION.md §1.3): one bad signal caps the whole score."""
    reasons: list[str] = []
    scores: list[float] = []
    hard_fail = False

    blur = signals["blur_score"]
    if blur < config.BLUR_VARIANCE_FLOOR:
        reasons.append(f"blur_score {blur:.1f} below reject floor {config.BLUR_VARIANCE_FLOOR}")
        hard_fail = True
        scores.append(0.0)
    else:
        if blur < config.BLUR_VARIANCE_WARN:
            reasons.append(f"blur_score {blur:.1f} below warn threshold {config.BLUR_VARIANCE_WARN}")
        scores.append(min(1.0, blur / config.BLUR_VARIANCE_WARN))

    skew = signals["skew_deg"]
    if skew > config.MAX_SKEW_DEG:
        reasons.append(f"skew_deg {skew:.1f} beyond reject ceiling {config.MAX_SKEW_DEG}")
        hard_fail = True
        scores.append(0.0)
    else:
        if skew > config.SKEW_WARN_DEG:
            reasons.append(f"skew_deg {skew:.1f} beyond warn threshold {config.SKEW_WARN_DEG}")
        scores.append(max(0.0, 1.0 - skew / config.MAX_SKEW_DEG))

    min_dim = signals["min_dim_px"]
    if min_dim < config.MIN_IMAGE_DIM_PX:
        reasons.append(f"min_dim_px {min_dim} below floor {config.MIN_IMAGE_DIM_PX}")
        hard_fail = True
        scores.append(0.0)
    else:
        scores.append(1.0)

    exposure = signals["exposure_clip"]
    if exposure > config.EXPOSURE_CLIP_CEILING:
        reasons.append(f"exposure_clip {exposure:.3f} beyond warn ceiling {config.EXPOSURE_CLIP_CEILING}")
    # Exposure only has a warn tier in config (no independent reject floor) —
    # glare alone shouldn't hard-block, but it should visibly cap the score.
    scores.append(max(0.0, 1.0 - exposure / (config.EXPOSURE_CLIP_CEILING * 3)))

    if not signals["quad_found"]:
        reasons.append("no document-shaped contour found (quad_found=False)")
        scores.append(0.5)  # not a hard fail alone, but a real confidence hit
    else:
        scores.append(1.0)

    normalized_quality = min(scores)

    if hard_fail:
        verdict = QualityVerdict.REJECT
    elif reasons:
        verdict = QualityVerdict.WARN
    else:
        verdict = QualityVerdict.PASS

    return verdict, reasons, normalized_quality


def assess(image: np.ndarray) -> ImageQualityReport:
    signals = _signals(image)
    verdict, reasons, normalized_quality = _verdict(signals)
    return ImageQualityReport(
        verdict=verdict,
        blur_score=signals["blur_score"],
        exposure_clip=signals["exposure_clip"],
        skew_deg=signals["skew_deg"],
        min_dim_px=signals["min_dim_px"],
        quad_found=signals["quad_found"],
        normalized_quality=normalized_quality,
        reasons=reasons,
    )


# --- Preprocessing -------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 box points as top-left, top-right, bottom-right, bottom-left
    — the standard sum/diff trick for perspective-warp source points."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _warp_to_quad(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Standard four-point perspective transform: output width/height are
    measured directly from the ordered corner distances, not taken from
    minAreaRect's (w, h) — those don't reliably line up with the tl/tr/br/bl
    order _order_points() produces once the box is rotated, and mixing the
    two was warping portrait content into a landscape-shaped, effectively
    90-degrees-off result (caught by re-running assess() after preprocess()
    during Phase 2 calibration — skew_deg came back as 90.0 every time)."""
    tl, tr, br, bl = _order_points(box)
    width_a, width_b = np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b), 1)
    height_a, height_b = np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b), 1)
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(_order_points(box), dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def preprocess(image: np.ndarray) -> np.ndarray:
    """Deskew (perspective-warp to the detected document quad if found,
    else a plain rotation correction), CLAHE contrast, upscale to
    MIN_IMAGE_DIM_PX. On/off is the preprocessing ablation
    (CV_INTEGRATION.md §2.9/Phase 7)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    signed_angle, quad_found, contour = _detect_document(gray)

    if quad_found and contour is not None:
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        working = _warp_to_quad(image, box)
    else:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, signed_angle, 1.0)
        working = cv2.warpAffine(image, matrix, (w, h), borderValue=(250, 248, 245))

    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    working = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    h, w = working.shape[:2]
    min_dim = min(h, w)
    if min_dim < config.MIN_IMAGE_DIM_PX and min_dim > 0:
        scale = config.MIN_IMAGE_DIM_PX / min_dim
        working = cv2.resize(working, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    return working


# --- CLI: threshold calibration ------------------------------------------------

def _dump_csv(directory: Path) -> None:
    """Assesses every image in `directory` and writes a CSV to stdout —
    the calibration artifact CV_INTEGRATION.md §2.4 describes: dump raw
    signals, find where clean/blur variants actually separate, set floors
    from that instead of guessing."""
    writer = csv.writer(sys.stdout)
    writer.writerow([
        "filename", "blur_score", "exposure_clip", "skew_deg",
        "min_dim_px", "quad_found", "verdict", "normalized_quality", "reasons",
    ])
    paths = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))
    for path in paths:
        image = load_image(path.read_bytes())
        report = assess(image)
        writer.writerow([
            path.name, f"{report.blur_score:.2f}", f"{report.exposure_clip:.4f}",
            f"{report.skew_deg:.2f}", report.min_dim_px, report.quad_found,
            report.verdict.value, f"{report.normalized_quality:.3f}",
            ";".join(report.reasons),
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-csv", type=Path, metavar="DIR",
                         help="Assess every image in DIR, write a CSV to stdout")
    args = parser.parse_args()
    if args.dump_csv:
        _dump_csv(args.dump_csv)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
