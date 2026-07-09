"""Escalation red-team eval (PLAN.md §7): correctness of the deterministic
escalation rule engine (src/guardrails/escalation.py, danger_scan.py) against
~24 adversarial prompts (evals/escalation_redteam.jsonl).

Unlike unit tests (tests/test_guardrails.py), which confirm the rules do what
we already wrote them to do, this set is designed to surface cases we did NOT
anticipate: paraphrased danger language the lexicon might miss (false
negatives), benign phrases that happen to contain a trigger word (false
positives), and softly-worded serious complaints that should still escalate
purely on category (manipulation-resistance).

Deliberately does NOT call the LLM — should_escalate() and danger_scan() are
pure, deterministic functions, so this eval runs for free and can be re-run
as often as needed without touching Gemini's quota (PLAN.md §2.1/§8).

Usage:
    python evals/run_escalation_eval.py
    python evals/run_escalation_eval.py --mlflow
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guardrails.escalation import should_escalate
from src.schemas import ComplaintCategory, ComplaintTicket, Severity

REDTEAM_SET = Path(__file__).parent / "escalation_redteam.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_redteam_set(path: Path = REDTEAM_SET) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(items: list[dict]) -> list[dict]:
    results = []
    for item in items:
        ticket = ComplaintTicket(
            category=ComplaintCategory(item["ticket_category"]),
            severity=Severity(item["ticket_severity"]),
            description=item["raw_text"],
        )
        decision = should_escalate(ticket, raw_text=item["raw_text"])
        actual_trigger = decision.trigger_rule.value if decision.trigger_rule else None
        correct = (
            decision.should_escalate == item["expected_should_escalate"]
            and actual_trigger == item["expected_trigger_rule"]
        )
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "raw_text": item["raw_text"],
                "expected_should_escalate": item["expected_should_escalate"],
                "actual_should_escalate": decision.should_escalate,
                "expected_trigger_rule": item["expected_trigger_rule"],
                "actual_trigger_rule": actual_trigger,
                "correct": correct,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Escalation rule-engine red-team eval")
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()

    items = load_redteam_set()
    print(f"Evaluating {len(items)} adversarial prompts against the escalation rule engine...")

    results = evaluate(items)
    overall_rate = sum(r["correct"] for r in results) / len(results) if results else 0.0

    print(f"\nOverall correctness rate: {overall_rate:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")

    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    for category, cat_results in sorted(by_category.items()):
        cat_rate = sum(r["correct"] for r in cat_results) / len(cat_results)
        print(f"\n[{category}] {cat_rate:.1%} ({sum(r['correct'] for r in cat_results)}/{len(cat_results)})")
        for r in cat_results:
            if not r["correct"]:
                print(
                    f"  MISS [{r['id']}] expected_escalate={r['expected_should_escalate']} "
                    f"(trigger={r['expected_trigger_rule']}) -> "
                    f"got_escalate={r['actual_should_escalate']} (trigger={r['actual_trigger_rule']})"
                )
                print(f"        text: \"{r['raw_text']}\"")

    report = {
        "results": results,
        "metrics": {
            "overall_correctness_rate": overall_rate,
            "total_prompts": len(results),
            "total_correct": sum(r["correct"] for r in results),
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"escalation_eval_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("escalation-redteam-eval")
        with mlflow.start_run():
            mlflow.log_params({"prompt_count": len(items)})
            mlflow.log_metrics(report["metrics"])
            mlflow.log_artifact(str(out_path))
        print("Logged to MLflow experiment 'escalation-redteam-eval'.")


if __name__ == "__main__":
    main()
