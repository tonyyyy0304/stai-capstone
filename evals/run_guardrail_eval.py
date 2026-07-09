"""Guardrail red-team eval (PLAN.md §7): block rate on off-topic / injection /
PII probes, ~20 adversarial prompts (evals/guardrail_redteam.jsonl).

Deliberately does NOT call the LLM router by default — injection and toxicity
are deterministic checks (src/guardrails/input_checks.py, toxicity.py) and PII
detection is regex-only (src/guardrails/pii.py), so this eval runs for free
and can be re-run as often as needed without touching Gemini's quota (the
project's #1 documented constraint, PLAN.md §2.1/§8).

Off-topic probes are *not* evaluated here for that reason: blocking them is
the intent router's job (Intent.OUT_OF_SCOPE), which requires an LLM call.
Pass --with-router to also evaluate those (costs one LLM call per off-topic
probe) — off by default.

complaint_exempt probes cover the toxicity/complaint carve-out
(check_toxicity_with_context, src/guardrails/toxicity.py): a harassment
complaint quoting abuse said TO the employee must not be blocked by the
wordlist. These run against a *synthetic* IntentClassification (is_toxic
fixed to False, matching what a correctly-behaving router would emit) rather
than the real router, so they still cost zero LLM calls — they test the
combining logic in toxicity.py, not the router's judgment itself.

Usage:
    python evals/run_guardrail_eval.py
    python evals/run_guardrail_eval.py --with-router --mlflow
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guardrails.input_checks import check_topic_and_injection
from src.guardrails.pii import detect_pii
from src.guardrails.toxicity import check_toxicity, check_toxicity_with_context
from src.schemas import Intent, IntentClassification

REDTEAM_SET = Path(__file__).parent / "guardrail_redteam.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_redteam_set(path: Path = REDTEAM_SET) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_injection_and_toxicity(items: list[dict]) -> list[dict]:
    results = []
    for item in items:
        if item["category"] == "injection":
            result = check_topic_and_injection(item["prompt"])
        elif item["category"] == "toxicity":
            result = check_toxicity(item["prompt"])
        else:
            continue
        blocked = not result.allowed
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "expected_blocked": item["expected_blocked"],
                "actual_blocked": blocked,
                "correct": blocked == item["expected_blocked"],
            }
        )
    return results


def evaluate_pii(items: list[dict]) -> list[dict]:
    results = []
    for item in items:
        if item["category"] != "pii":
            continue
        detected = set(detect_pii(item["prompt"]))
        expected = set(item["expected_pii"])
        results.append(
            {
                "id": item["id"],
                "category": "pii",
                "expected_pii": sorted(expected),
                "actual_pii": sorted(detected),
                "correct": detected == expected,
            }
        )
    return results


def evaluate_complaint_exempt(items: list[dict]) -> list[dict]:
    results = []
    for item in items:
        if item["category"] != "complaint_exempt":
            continue
        classification = IntentClassification(
            intent=Intent(item["intent"]), confidence=0.9, is_toxic=False
        )
        result = check_toxicity_with_context(item["prompt"], classification)
        blocked = not result.allowed
        results.append(
            {
                "id": item["id"],
                "category": "complaint_exempt",
                "intent": item["intent"],
                "expected_blocked": item["expected_blocked"],
                "actual_blocked": blocked,
                "correct": blocked == item["expected_blocked"],
            }
        )
    return results


def evaluate_off_topic_with_router(items: list[dict]) -> list[dict]:
    from src.agent.router import classify_intent
    from src.schemas import Intent

    results = []
    for item in items:
        if item["category"] != "off_topic":
            continue
        classification = classify_intent(item["prompt"])
        blocked = classification.intent == Intent.OUT_OF_SCOPE
        results.append(
            {
                "id": item["id"],
                "category": "off_topic",
                "classified_intent": classification.intent.value,
                "correctly_flagged_out_of_scope": blocked,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrail red-team eval")
    parser.add_argument(
        "--with-router",
        action="store_true",
        help="also evaluate off-topic probes via the LLM intent router (costs one call each)",
    )
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()

    items = load_redteam_set()
    print(f"Evaluating {len(items)} adversarial prompts...")

    injection_toxicity = evaluate_injection_and_toxicity(items)
    pii_results = evaluate_pii(items)
    complaint_exempt_results = evaluate_complaint_exempt(items)
    off_topic_results = evaluate_off_topic_with_router(items) if args.with_router else []

    block_rate = (
        sum(r["correct"] for r in injection_toxicity) / len(injection_toxicity)
        if injection_toxicity
        else 0.0
    )
    pii_detection_rate = (
        sum(r["correct"] for r in pii_results) / len(pii_results) if pii_results else 0.0
    )
    complaint_exempt_rate = (
        sum(r["correct"] for r in complaint_exempt_results) / len(complaint_exempt_results)
        if complaint_exempt_results
        else 0.0
    )

    print(f"\nInjection/toxicity block-rate (deterministic): {block_rate:.1%}")
    for r in injection_toxicity:
        if not r["correct"]:
            print(f"  MISS [{r['id']}] expected_blocked={r['expected_blocked']} got={r['actual_blocked']}")

    print(f"\nPII detection rate: {pii_detection_rate:.1%}")
    for r in pii_results:
        if not r["correct"]:
            print(f"  MISS [{r['id']}] expected={r['expected_pii']} got={r['actual_pii']}")

    print(f"\nComplaint-exempt toxicity carve-out correctness: {complaint_exempt_rate:.1%}")
    for r in complaint_exempt_results:
        if not r["correct"]:
            print(f"  MISS [{r['id']}] expected_blocked={r['expected_blocked']} got={r['actual_blocked']}")

    if off_topic_results:
        off_topic_rate = sum(r["correctly_flagged_out_of_scope"] for r in off_topic_results) / len(
            off_topic_results
        )
        print(f"\nOff-topic block-rate (via router): {off_topic_rate:.1%}")
    else:
        print("\nOff-topic probes not evaluated (pass --with-router to include; costs LLM calls).")

    report = {
        "injection_toxicity": injection_toxicity,
        "pii": pii_results,
        "complaint_exempt": complaint_exempt_results,
        "off_topic": off_topic_results,
        "metrics": {
            "injection_toxicity_block_rate": block_rate,
            "pii_detection_rate": pii_detection_rate,
            "complaint_exempt_correctness": complaint_exempt_rate,
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"guardrail_eval_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("guardrail-redteam-eval")
        with mlflow.start_run():
            mlflow.log_params({"prompt_count": len(items), "with_router": args.with_router})
            mlflow.log_metrics(report["metrics"])
            mlflow.log_artifact(str(out_path))
        print("Logged to MLflow experiment 'guardrail-redteam-eval'.")


if __name__ == "__main__":
    main()
