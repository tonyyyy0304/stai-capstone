"""Deterministic validation-policy eval (Component 14, Phase 7, CV_INTEGRATION.md §2.9).

Precision/recall/F1 on needs_review vs. accepted (positive class:
needs_review), per-rule trigger counts, and the headline safety metric --
FALSE AUTO-PASS RATE: among fixtures that should NOT have been auto-accepted
(expected_outcome is needs_review or rejected), how many were anyway. All
reported per doc_type, never pooled (same discipline as everywhere else in
this component). A dedicated section covers cross_document_consistency:
overall trigger rate, and specifically whether it ever fires on the "clean
pair" fixtures -- it shouldn't.

Runs ENTIRELY off already-cached extractions (src/ocr/extractor.py's
load_cached_result -- a hash-only lookup that never calls the vision model,
live or otherwise) plus a purely local, zero-cost quality re-assessment
(src/ocr/quality.py is pure OpenCV, no network). A fixture whose extraction
hasn't been cached yet (run run_ocr_eval.py, or POST /upload-doc, against it
first) is skipped and reported as such -- never silently spends a live call
to fill the gap. That's what makes "zero API cost, re-runnable freely"
literally true rather than just an intention.

Usage:
    python evals/run_validation_eval.py
    python evals/run_validation_eval.py --mlflow
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guardrails.doc_validation import validate_cross_document, validate_document, validate_id_document
from src.ocr import quality as quality_mod
from src.ocr.extractor import load_cached_result
from src.schemas import DocType

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = REPO_ROOT / "data" / "references" / "mock"
RESULTS_DIR = Path(__file__).parent / "results"

_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
_EMP_ID_PATTERN = re.compile(r"EMP-\d+")


def _load_json(path: Path):
    return json.loads(path.read_text())


def _identities_by_employee_id() -> dict[str, dict]:
    identities = _load_json(MOCK_DIR / "identities.json")
    return {identity["employee_id"]: identity for identity in identities}


def _identities_by_identity_id() -> dict[str, dict]:
    identities = _load_json(MOCK_DIR / "identities.json")
    return {identity["identity_id"]: identity for identity in identities}


def _cached_extraction(image_path: Path, doc_type: DocType):
    """Returns (result, quality_report) or (None, quality_report) on a cache
    miss -- never calls the vision model. Quality is always computed locally
    (pure OpenCV) regardless of cache state."""
    import hashlib

    image_bytes = image_path.read_bytes()
    quality_report = quality_mod.assess(quality_mod.load_image(image_bytes))
    source_hash = hashlib.sha256(image_bytes).hexdigest()
    result = load_cached_result(source_hash, doc_type)
    return result, quality_report


def _positive_fixtures(doc_type: DocType) -> list[dict]:
    """One fixture per mock identity's clean image -- expected_outcome is
    "accepted" by construction (Phase 5's cross-check already confirmed all
    16 clean identities validate as accepted)."""
    prefix = "nbi" if doc_type == DocType.NBI_CLEARANCE else "id"
    identities_by_id = _identities_by_identity_id()
    fixtures = []
    for expected_path in sorted(MOCK_DIR.glob(f"{prefix}_*.expected.json")):
        bare = expected_path.name[: -len(".expected.json")]
        image_matches = list(MOCK_DIR.glob(f"{bare}_clean.*"))
        if not image_matches:
            continue
        expected = _load_json(expected_path)
        identity = identities_by_id.get(expected["identity_id"], {})
        fixtures.append({
            "kind": "positive",
            "identity_id": expected["identity_id"],
            "image_path": image_matches[0],
            "expected_outcome": "accepted",
            "faculty_record": {
                "employee_id": identity.get("employee_id", expected.get("employee_id")),
                "full_name": identity.get("full_name", expected.get("full_name")),
                "date_of_birth": identity.get("date_of_birth", expected.get("date_of_birth")),
            },
        })
    return fixtures


def _negative_fixtures(doc_type: DocType) -> list[dict]:
    """Negatives are keyed by filename prefix (nbi_/id_) for which SLOT they
    were uploaded into -- that's independent of negatives.json's
    expected_doc_type, which is what the MODEL should classify them as
    (unknown_document/not_a_document for the wrong-type/blank-page cases)."""
    prefix = "nbi_neg_" if doc_type == DocType.NBI_CLEARANCE else "id_neg_"
    identities_by_emp = _identities_by_employee_id()
    negatives = _load_json(MOCK_DIR / "negatives.json")
    fixtures = []
    for neg in negatives:
        if not neg["filename"].startswith(prefix):
            continue
        image_path = MOCK_DIR / neg["filename"]
        if not image_path.exists():
            continue

        faculty_record = {"employee_id": "UNKNOWN", "full_name": "UNKNOWN", "date_of_birth": "2000-01-01"}
        emp_match = _EMP_ID_PATTERN.search(neg.get("reason", ""))
        if emp_match and emp_match.group(0) in identities_by_emp:
            identity = identities_by_emp[emp_match.group(0)]
            faculty_record = {
                "employee_id": identity["employee_id"],
                "full_name": identity["full_name"],
                "date_of_birth": identity["date_of_birth"],
            }

        fixtures.append({
            "kind": "negative",
            "category": neg["category"],
            "image_path": image_path,
            "expected_outcome": neg["expected_outcome"],
            "faculty_record": faculty_record,
            "pairs_with": neg.get("pairs_with"),
        })
    return fixtures


def evaluate_doc_type(doc_type: DocType, fixtures: list[dict]) -> dict:
    validator = validate_document if doc_type == DocType.NBI_CLEARANCE else validate_id_document
    rule_trigger_counts: dict[str, int] = {}
    rule_eval_counts: dict[str, int] = {}
    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}  # positive class: needs_review, restricted to accepted/needs_review expectations
    should_not_pass, false_auto_pass = 0, 0
    skipped = []
    evaluated = []

    for fixture in fixtures:
        mime_type = _MIME_BY_SUFFIX.get(fixture["image_path"].suffix.lower(), "image/png")
        result, quality_report = _cached_extraction(fixture["image_path"], doc_type)
        if result is None and quality_report.verdict.value != "reject":
            # Not a quality-gate rejection (which validate_document handles
            # deterministically without needing an extraction) -- this is an
            # actual cache miss. Skip rather than spend a live call.
            skipped.append(fixture["image_path"].name)
            continue

        validation = validator(result, quality_report, fixture["faculty_record"])
        actual_outcome = validation.outcome.value
        expected_outcome = fixture["expected_outcome"]

        for rule in validation.rules:
            rule_eval_counts[rule.rule] = rule_eval_counts.get(rule.rule, 0) + 1
            if not rule.passed:
                rule_trigger_counts[rule.rule] = rule_trigger_counts.get(rule.rule, 0) + 1

        if expected_outcome != "accepted":
            should_not_pass += 1
            if actual_outcome == "accepted":
                false_auto_pass += 1

        if expected_outcome in ("accepted", "needs_review"):
            expected_positive = expected_outcome == "needs_review"
            actual_positive = actual_outcome == "needs_review"
            if expected_positive and actual_positive:
                confusion["tp"] += 1
            elif not expected_positive and actual_positive:
                confusion["fp"] += 1
            elif expected_positive and not actual_positive:
                confusion["fn"] += 1
            elif not expected_positive and not actual_positive:
                confusion["tn"] += 1

        evaluated.append({
            "file": fixture["image_path"].name,
            "kind": fixture["kind"],
            "expected_outcome": expected_outcome,
            "actual_outcome": actual_outcome,
            "correct": actual_outcome == expected_outcome,
        })

    tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "doc_type": doc_type.value,
        "evaluated_count": len(evaluated),
        "skipped_not_cached": skipped,
        "precision_needs_review": round(precision, 4),
        "recall_needs_review": round(recall, 4),
        "f1_needs_review": round(f1, 4),
        "confusion": confusion,
        "false_auto_pass_rate": round(false_auto_pass / should_not_pass, 4) if should_not_pass else 0.0,
        "should_not_pass_count": should_not_pass,
        "rule_trigger_counts": rule_trigger_counts,
        "rule_eval_counts": rule_eval_counts,
        "fixtures": evaluated,
    }


def evaluate_cross_document() -> dict:
    """Pairs every mock GOVERNMENT_ID negative that has a pairs_with (and,
    separately, every clean identity pair) against its NBI sibling and runs
    validate_cross_document -- entirely off cached extractions."""
    identities_by_id = _identities_by_identity_id()
    negatives = _load_json(MOCK_DIR / "negatives.json")

    pairs = []
    for identity_id in identities_by_id:
        pairs.append({
            "label": f"clean_pair_{identity_id}",
            "nbi_image": MOCK_DIR / f"nbi_{identity_id}_clean.png",
            "id_glob": f"id_{identity_id}_clean.*",
            "expect_mismatch": False,
        })
    for neg in negatives:
        if not neg.get("pairs_with"):
            continue
        pairs.append({
            "label": neg["category"],
            "nbi_image": MOCK_DIR / f"{neg['pairs_with']}_clean.png",
            "id_glob": neg["filename"],
            "expect_mismatch": True,
        })

    results, skipped = [], []
    for pair in pairs:
        id_matches = list(MOCK_DIR.glob(pair["id_glob"]))
        if not pair["nbi_image"].exists() or not id_matches:
            continue
        nbi_result, _ = _cached_extraction(pair["nbi_image"], DocType.NBI_CLEARANCE)
        id_result, _ = _cached_extraction(id_matches[0], DocType.GOVERNMENT_ID)
        if nbi_result is None or id_result is None:
            skipped.append(pair["label"])
            continue
        cross_rule = validate_cross_document(nbi_result, id_result)
        results.append({
            "label": pair["label"],
            "expect_mismatch": pair["expect_mismatch"],
            "actual_passed": cross_rule.passed,
            "correct": cross_rule.passed != pair["expect_mismatch"],
        })

    clean_pair_results = [r for r in results if not r["expect_mismatch"]]
    clean_pair_false_trigger_rate = (
        sum(1 for r in clean_pair_results if not r["actual_passed"]) / len(clean_pair_results)
        if clean_pair_results else 0.0
    )
    overall_trigger_rate = (
        sum(1 for r in results if not r["actual_passed"]) / len(results) if results else 0.0
    )
    return {
        "evaluated_count": len(results),
        "skipped_not_cached": skipped,
        "overall_trigger_rate": round(overall_trigger_rate, 4),
        "clean_pair_false_trigger_rate": round(clean_pair_false_trigger_rate, 4),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic validation-policy eval")
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()

    results = {}
    for doc_type in (DocType.NBI_CLEARANCE, DocType.GOVERNMENT_ID):
        fixtures = _positive_fixtures(doc_type) + _negative_fixtures(doc_type)
        print(f"\n=== {doc_type.value} ({len(fixtures)} fixtures) ===")
        summary = evaluate_doc_type(doc_type, fixtures)
        results[doc_type.value] = summary

        print(f"  evaluated: {summary['evaluated_count']}  skipped (not cached): {len(summary['skipped_not_cached'])}")
        if summary["skipped_not_cached"]:
            print(f"    (run run_ocr_eval.py or POST /upload-doc against these first: {summary['skipped_not_cached']})")
        print(f"  precision/recall/F1 (needs_review): {summary['precision_needs_review']:.1%} / "
              f"{summary['recall_needs_review']:.1%} / {summary['f1_needs_review']:.3f}")
        print(f"  FALSE AUTO-PASS RATE (headline): {summary['false_auto_pass_rate']:.1%} "
              f"({summary['should_not_pass_count']} fixtures should not have passed)")
        for rule, triggered in summary["rule_trigger_counts"].items():
            evaluated_n = summary["rule_eval_counts"].get(rule, 0)
            print(f"    rule={rule:<24} triggered {triggered}/{evaluated_n}")
        for f in summary["fixtures"]:
            if not f["correct"]:
                print(f"    MISS [{f['file']}] expected={f['expected_outcome']} got={f['actual_outcome']}")

    print("\n=== cross_document_consistency ===")
    cross = evaluate_cross_document()
    results["cross_document_consistency"] = cross
    print(f"  evaluated: {cross['evaluated_count']}  skipped (not cached): {len(cross['skipped_not_cached'])}")
    print(f"  overall trigger rate (mismatch detected): {cross['overall_trigger_rate']:.1%}")
    print(f"  false-trigger rate on CLEAN pairs (should be 0%): {cross['clean_pair_false_trigger_rate']:.1%}")
    for r in cross["results"]:
        if not r["correct"]:
            print(f"    MISS [{r['label']}] expect_mismatch={r['expect_mismatch']} actual_passed={r['actual_passed']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"validation_eval_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("validation-policy-eval")
        with mlflow.start_run():
            for doc_type_value in (DocType.NBI_CLEARANCE.value, DocType.GOVERNMENT_ID.value):
                summary = results[doc_type_value]
                mlflow.log_metric(f"{doc_type_value}_false_auto_pass_rate", summary["false_auto_pass_rate"])
                mlflow.log_metric(f"{doc_type_value}_f1_needs_review", summary["f1_needs_review"])
            mlflow.log_metric("cross_document_clean_pair_false_trigger_rate", cross["clean_pair_false_trigger_rate"])
            mlflow.log_artifact(str(out_path))
        print("Logged to MLflow experiment 'validation-policy-eval'.")


if __name__ == "__main__":
    main()
