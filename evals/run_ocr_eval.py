"""OCR extraction accuracy eval (Component 14, Phase 7, CV_INTEGRATION.md §2.9).

Per-field exact match + normalized CER (rapidfuzz Levenshtein), reported
SEPARATELY per doc_type (NBI Clearance vs. Government ID) and per subset
(mock vs. real) -- never pooled, same discipline as the real-vs-mock rule
(CV_INTEGRATION.md §1.8) and the type-vs-type separation every other part of
this component follows.

Cache-first by default (src/ocr/extractor.py's SHA-256 disk cache) -- a full
mock-subset run after the cache is warm costs 0 API calls. --no-cache forces
live calls. --no-preprocess re-runs without the deskew/contrast preprocessing
step as an ablation (a genuinely separate cache entry, since the cache key
includes the preprocess flag -- this is what makes the ablation flag actually
do something rather than silently reusing the default run's cache).

Accuracy is measured against each identity's un-degraded `clean` image only --
the blur/skew/glare/lowres_jpeg variants are src/ocr/quality.py's concern
(reject/warn verdicts), not extraction accuracy's.

The REAL subset currently has NO ground-truth labels (data/references/real/
has no *.expected.json files) -- a human hasn't transcribed the actual
documents by hand. This script does NOT fabricate labels from a prior model
extraction (that would grade the model against its own output, which proves
nothing). --subset real runs cleanly and reports zero fixtures rather than
crashing; annotate data/references/real/<name>.expected.json by hand
(mirroring the mock shape) before this subset can report real numbers.

Usage:
    python evals/run_ocr_eval.py --subset mock
    python evals/run_ocr_eval.py --subset mock --no-preprocess
    python evals/run_ocr_eval.py --subset mock --no-cache      # spends real quota
    python evals/run_ocr_eval.py --subset mock --limit 3       # first N identities per doc_type, for a quota-cheap smoke run
    python evals/run_ocr_eval.py --subset real --mlflow
"""

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidfuzz.distance import Levenshtein

from src.ocr import doctypes
from src.ocr.extractor import extract_document
from src.schemas import DocType

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = REPO_ROOT / "data" / "references" / "mock"
REAL_DIR = REPO_ROOT / "data" / "references" / "real"
RESULTS_DIR = Path(__file__).parent / "results"

_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _field_names(doc_type: DocType) -> list[str]:
    """All ExtractedField-typed field names for doc_type -- reuses the
    doctypes registry so this can never drift from the actual schema, same
    reasoning as extractor.py's _retryable_field_names()."""
    spec = doctypes.get_spec(doc_type)
    if spec is None:
        return []
    return [f.name for f in spec.fields if f.name != "id_type"]


def _mock_fixtures(doc_type: DocType) -> list[tuple[Path, dict]]:
    prefix = "nbi" if doc_type == DocType.NBI_CLEARANCE else "id"
    fixtures = []
    for expected_path in sorted(MOCK_DIR.glob(f"{prefix}_*.expected.json")):
        bare = expected_path.name[: -len(".expected.json")]
        image_matches = list(MOCK_DIR.glob(f"{bare}_clean.*"))
        if not image_matches:
            continue
        fixtures.append((image_matches[0], json.loads(expected_path.read_text())))
    return fixtures


def _real_fixtures(doc_type: DocType) -> list[tuple[Path, dict]]:
    fixtures = []
    for expected_path in sorted(REAL_DIR.glob("*.expected.json")):
        expected = json.loads(expected_path.read_text())
        if expected.get("doc_type") != doc_type.value:
            continue
        bare = expected_path.name[: -len(".expected.json")]
        image_matches = [p for p in REAL_DIR.glob(f"{bare}.*") if p.suffix.lower() != ".json"]
        if not image_matches:
            continue
        fixtures.append((image_matches[0], expected))
    return fixtures


def _normalize_for_comparison(value: str | None) -> str | None:
    """Dates/datetimes compare by date only, ignoring any printed time
    component -- date_printed genuinely carries a time on real documents
    (found live during Phase 4; src/ocr/doctypes.py's _parse_date() handles
    the same case for validation), but the mock ground truth doesn't render
    one. A naive string compare would score a correct extraction as 0%
    exact-match purely because of a time suffix the model was CORRECTLY
    asked to include. Falls through unchanged for non-date fields (names,
    reference_no/id_number, purpose, remarks)."""
    if not value:
        return value
    text = value.strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return value


def _compare_field(actual_value, expected_value) -> tuple[bool, float]:
    """Returns (exact_match, normalized_cer). Both empty -> exact, cer=0.0.
    One empty and the other not -> not exact, cer=1.0 (maximally wrong,
    not skipped -- a missed field should hurt the score, not vanish from it)."""
    a = (_normalize_for_comparison(actual_value) or "").strip()
    e = (_normalize_for_comparison(expected_value) or "").strip()
    if not a and not e:
        return True, 0.0
    if not a or not e:
        return False, 1.0
    return a.upper() == e.upper(), Levenshtein.normalized_distance(a.upper(), e.upper())


def evaluate_doc_type(
    doc_type: DocType,
    fixtures: list[tuple[Path, dict]],
    preprocess: bool,
    use_cache: bool,
) -> dict:
    field_names = _field_names(doc_type)
    per_field = {name: {"exact": 0, "cer_sum": 0.0, "n": 0} for name in field_names}
    id_type_correct, id_type_n = 0, 0
    per_fixture = []

    for image_path, expected in fixtures:
        image_bytes = image_path.read_bytes()
        mime_type = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/png")
        result, quality_report = extract_document(
            image_bytes, mime_type, expected_doc_type=doc_type,
            use_cache=use_cache, preprocess=preprocess,
        )
        fixture_result: dict = {"file": image_path.name, "quality_verdict": quality_report.verdict.value}
        if result is None:
            fixture_result["extracted"] = False
            per_fixture.append(fixture_result)
            continue

        fixture_result["extracted"] = True
        field_scores = {}
        for name in field_names:
            actual_field = getattr(result, name, None)
            actual_value = actual_field.value if actual_field is not None else None
            exact, cer = _compare_field(actual_value, expected.get(name))
            per_field[name]["n"] += 1
            per_field[name]["exact"] += int(exact)
            per_field[name]["cer_sum"] += cer
            field_scores[name] = {"exact": exact, "cer": round(cer, 4)}
        fixture_result["fields"] = field_scores

        if doc_type == DocType.GOVERNMENT_ID and "id_type" in expected:
            id_type_n += 1
            id_type_correct += int(getattr(result, "id_type", None) and result.id_type.value == expected["id_type"])

        per_fixture.append(fixture_result)

    field_summary = {
        name: {
            "exact_match_rate": (stats["exact"] / stats["n"]) if stats["n"] else 0.0,
            "avg_cer": round((stats["cer_sum"] / stats["n"]), 4) if stats["n"] else 0.0,
            "n": stats["n"],
        }
        for name, stats in per_field.items()
    }
    overall_exact = sum(s["exact"] for s in per_field.values())
    overall_n = sum(s["n"] for s in per_field.values())

    summary = {
        "doc_type": doc_type.value,
        "fixture_count": len(fixtures),
        "field_summary": field_summary,
        "overall_exact_match_rate": (overall_exact / overall_n) if overall_n else 0.0,
        "fixtures": per_fixture,
    }
    if doc_type == DocType.GOVERNMENT_ID:
        summary["id_type_classification_accuracy"] = (id_type_correct / id_type_n) if id_type_n else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR extraction accuracy eval")
    parser.add_argument("--subset", choices=["mock", "real"], default="mock")
    parser.add_argument("--no-preprocess", action="store_true", help="ablation: skip deskew/contrast preprocessing")
    parser.add_argument("--no-cache", action="store_true", help="force live calls even on a cache hit (spends quota)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N identities per doc_type (quota-cheap smoke run)")
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()

    preprocess = not args.no_preprocess
    use_cache = not args.no_cache
    fixture_loader = _mock_fixtures if args.subset == "mock" else _real_fixtures

    results = {}
    for doc_type in (DocType.NBI_CLEARANCE, DocType.GOVERNMENT_ID):
        fixtures = fixture_loader(doc_type)
        if args.limit is not None:
            fixtures = fixtures[: args.limit]
        print(f"\n=== {doc_type.value} ({args.subset} subset, {len(fixtures)} fixtures, preprocess={preprocess}) ===")
        if not fixtures:
            print("  no fixtures with ground truth found" + (
                " -- data/references/real/ has no *.expected.json labels yet (hand-annotate before this subset reports numbers)"
                if args.subset == "real" else ""
            ))
            results[doc_type.value] = {"doc_type": doc_type.value, "fixture_count": 0, "field_summary": {}, "overall_exact_match_rate": 0.0, "fixtures": []}
            continue

        summary = evaluate_doc_type(doc_type, fixtures, preprocess=preprocess, use_cache=use_cache)
        results[doc_type.value] = summary

        print(f"  overall exact-match rate: {summary['overall_exact_match_rate']:.1%}")
        for name, stats in summary["field_summary"].items():
            print(f"    {name:<16} exact={stats['exact_match_rate']:.1%}  avg_cer={stats['avg_cer']:.3f}  n={stats['n']}")
        if "id_type_classification_accuracy" in summary:
            print(f"  id_type classification accuracy: {summary['id_type_classification_accuracy']:.1%}")

    report = {
        "subset": args.subset,
        "preprocess": preprocess,
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"ocr_eval_{args.subset}_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("ocr-extraction-eval")
        with mlflow.start_run():
            mlflow.log_params({"subset": args.subset, "preprocess": preprocess})
            for doc_type_value, summary in results.items():
                mlflow.log_metric(f"{doc_type_value}_overall_exact_match_rate", summary["overall_exact_match_rate"])
            mlflow.log_artifact(str(out_path))
        print("Logged to MLflow experiment 'ocr-extraction-eval'.")


if __name__ == "__main__":
    main()
