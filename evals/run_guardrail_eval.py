"""Guardrail red-team eval (PLAN.md §7): block rate on off-topic / injection /
jailbreak / toxicity / PII probes, ~24 adversarial prompts
(evals/guardrail_redteam.jsonl), matching the three-stage guardrail
architecture in src/agent/orchestrator.py (module docstring, "Guardrails"):

1. Deterministic, no-LLM-call layer: prompt-injection regex and a toxicity
   wordlist (src/guardrails/input_checks.py, toxicity.py) plus regex-only PII
   detection (src/guardrails/pii.py). Runs for free by default and can be
   re-run as often as needed without touching Gemini's quota (the project's
   #1 documented constraint, PLAN.md §2.1/§8).
2. Router semantic backstop: classify_intent()'s is_injection_attempt/
   is_jailbreak/is_toxic signals (src/schemas.IntentClassification), read by
   check_injection_semantic/check_jailbreak_semantic/check_toxicity_semantic.
   Also where off_topic probes are actually caught (Intent.OUT_OF_SCOPE) —
   there's no deterministic off-topic or jailbreak check, so this is the only
   layer that can block them. Costs one LLM call per probe; pass --with-router.
3. LLM-as-judge (src/guardrails/llm_judge.py): a single structured call
   classifying toxicity/pii/injection/off_topic/jailbreak at once. Runs in
   production only when config.ENABLE_LLM_JUDGE is set, but this eval can
   exercise it directly regardless of that flag via --with-llm-judge (costs
   one LLM call per injection/toxicity/off_topic/jailbreak probe).

Usage:
    python evals/run_guardrail_eval.py
    python evals/run_guardrail_eval.py --with-router --with-llm-judge --mlflow
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.guardrails.input_checks import check_topic_and_injection
from src.guardrails.pii import detect_pii
from src.guardrails.toxicity import check_toxicity

REDTEAM_SET = Path(__file__).parent / "guardrail_redteam.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

# Categories with no deterministic check at all (src/guardrails/ has neither
# a wordlist nor a regex for these) — off_topic and jailbreak are only ever
# caught by the router or the LLM judge, both LLM calls.
_LLM_ONLY_CATEGORIES = {"off_topic", "jailbreak"}


def load_redteam_set(path: Path = REDTEAM_SET) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_injection_and_toxicity(items: list[dict]) -> list[dict]:
    """Deterministic layer only. jailbreak/off_topic are excluded on purpose —
    src/guardrails/ has no wordlist/regex for either, so scoring them here
    would just measure that a deterministic check doesn't exist, not whether
    the guardrail works."""
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
                "correct": blocked == item["expected_blocked"],
            }
        )
    return results


def evaluate_jailbreak_with_router(items: list[dict]) -> list[dict]:
    """jailbreak has no deterministic check (src/guardrails/input_checks.py
    only regexes for classic injection phrasing) — check_jailbreak_semantic()
    reads classify_intent()'s is_jailbreak signal, so this is the only layer
    besides the LLM judge that can catch it at all."""
    from src.agent.router import classify_intent
    from src.guardrails.input_checks import check_jailbreak_semantic

    results = []
    for item in items:
        if item["category"] != "jailbreak":
            continue
        classification = classify_intent(item["prompt"])
        result = check_jailbreak_semantic(classification)
        blocked = not result.allowed
        results.append(
            {
                "id": item["id"],
                "category": "jailbreak",
                "expected_blocked": item["expected_blocked"],
                "actual_blocked": blocked,
                "correct": blocked == item["expected_blocked"],
            }
        )
    return results


def evaluate_with_llm_judge(items: list[dict]) -> list[dict]:
    """Exercises src/guardrails/llm_judge.py end to end (judge_input +
    to_guardrail_result) directly, independent of config.ENABLE_LLM_JUDGE —
    that flag only controls whether production calls it per turn; this eval
    wants to know whether the judge itself is accurate. Skips pii: PII is
    detection-only there (never blocking), so there's no "blocked" outcome
    to score against expected_blocked."""
    from src.guardrails.llm_judge import check_input_llm

    results = []
    for item in items:
        if item["category"] not in ("injection", "toxicity", "off_topic", "jailbreak"):
            continue
        result = check_input_llm(item["prompt"])
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


def _print_misses(label: str, rate: float, results: list[dict]) -> None:
    print(f"\n{label}: {rate:.1%}")
    for r in results:
        if not r["correct"]:
            print(f"  MISS [{r['id']}] expected_blocked={r['expected_blocked']} got={r['actual_blocked']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrail red-team eval")
    parser.add_argument(
        "--with-router",
        action="store_true",
        help="also evaluate off-topic and jailbreak probes via the LLM intent router "
        "(costs one call per probe)",
    )
    parser.add_argument(
        "--with-llm-judge",
        action="store_true",
        help="also evaluate injection/toxicity/off-topic/jailbreak probes via the "
        "LLM-as-judge layer (src/guardrails/llm_judge.py), regardless of "
        "config.ENABLE_LLM_JUDGE (costs one call per probe)",
    )
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()

    items = load_redteam_set()
    print(f"Evaluating {len(items)} adversarial prompts...")

    injection_toxicity = evaluate_injection_and_toxicity(items)
    pii_results = evaluate_pii(items)
    off_topic_results = evaluate_off_topic_with_router(items) if args.with_router else []
    jailbreak_results = evaluate_jailbreak_with_router(items) if args.with_router else []
    llm_judge_results = evaluate_with_llm_judge(items) if args.with_llm_judge else []

    block_rate = (
        sum(r["correct"] for r in injection_toxicity) / len(injection_toxicity)
        if injection_toxicity
        else 0.0
    )
    pii_detection_rate = (
        sum(r["correct"] for r in pii_results) / len(pii_results) if pii_results else 0.0
    )

    print(f"\nInjection/toxicity block-rate (deterministic): {block_rate:.1%}")
    for r in injection_toxicity:
        if not r["correct"]:
            print(f"  MISS [{r['id']}] expected_blocked={r['expected_blocked']} got={r['actual_blocked']}")

    print(f"\nPII detection rate: {pii_detection_rate:.1%}")
    for r in pii_results:
        if not r["correct"]:
            print(f"  MISS [{r['id']}] expected={r['expected_pii']} got={r['actual_pii']}")

    off_topic_rate = None
    if off_topic_results:
        off_topic_rate = sum(r["correctly_flagged_out_of_scope"] for r in off_topic_results) / len(
            off_topic_results
        )
        print(f"\nOff-topic block-rate (via router): {off_topic_rate:.1%}")
    else:
        print("\nOff-topic probes not evaluated (pass --with-router to include; costs LLM calls).")

    jailbreak_rate = None
    if jailbreak_results:
        jailbreak_rate = sum(r["correct"] for r in jailbreak_results) / len(jailbreak_results)
        _print_misses("Jailbreak block-rate (via router semantic check)", jailbreak_rate, jailbreak_results)
    else:
        print("\nJailbreak probes not evaluated (pass --with-router to include; costs LLM calls).")

    llm_judge_rate = None
    if llm_judge_results:
        llm_judge_rate = sum(r["correct"] for r in llm_judge_results) / len(llm_judge_results)
        _print_misses("LLM-judge block-rate (all categories)", llm_judge_rate, llm_judge_results)
    else:
        print("\nLLM-judge layer not evaluated (pass --with-llm-judge to include; costs LLM calls).")

    report = {
        "injection_toxicity": injection_toxicity,
        "pii": pii_results,
        "off_topic": off_topic_results,
        "jailbreak": jailbreak_results,
        "llm_judge": llm_judge_results,
        "metrics": {
            "injection_toxicity_block_rate": block_rate,
            "pii_detection_rate": pii_detection_rate,
            **({"off_topic_block_rate": off_topic_rate} if off_topic_rate is not None else {}),
            **({"jailbreak_block_rate": jailbreak_rate} if jailbreak_rate is not None else {}),
            **({"llm_judge_block_rate": llm_judge_rate} if llm_judge_rate is not None else {}),
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
            mlflow.log_params(
                {
                    "prompt_count": len(items),
                    "with_router": args.with_router,
                    "with_llm_judge": args.with_llm_judge,
                }
            )
            mlflow.log_metrics(report["metrics"])
            mlflow.log_artifact(str(out_path))
        print("Logged to MLflow experiment 'guardrail-redteam-eval'.")


if __name__ == "__main__":
    main()
