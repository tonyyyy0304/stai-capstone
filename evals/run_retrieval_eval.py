"""Retrieval evaluation on the golden set: hit-rate@k and MRR (PLAN.md §3.2, §9).

The headline RAG number. A retrieved chunk is a "hit" when its doc_id matches
the expected doc and the expected section keyword appears in its section_path or
text (robust to chunk index changes across re-ingestions).

Metrics are broken out **per topic (T1/T2/T3) and per difficulty tier**, which is
what research question (a) asks for — a single pooled number would hide exactly
where the ceiling is. Rows are handled by tier:

  - lookup / multihop / near_miss / disambiguation → scored on hit-rate@k + MRR
    (they have an expected chunk in the corpus).
  - negative (expected_doc_id is null, expect_abstention=true) → scored on the
    **answer-level abstention signal** (default): the answerer is run end-to-end and
    the row counts as correct when it declines — i.e. GroundedAnswer.insufficient_context
    is true, or the reply is the templated "I don't know". The retrieval similarity floor
    is reported alongside as a *diagnostic only*: for in-vocabulary-but-unanswerable
    questions the top chunk sits above SIMILARITY_FLOOR, so the floor cannot gate them and
    must not be read as the abstention metric. Pass --abstention floor to score on the
    floor alone (legacy behaviour).

Each run writes two files to evals/results/, stamped with a UTC timestamp:
    retrieval_eval_<stamp>.log   — human-readable summary of the important metrics
    retrieval_eval_<stamp>.json  — full machine-readable report (per-question detail)

Usage:
    python evals/run_retrieval_eval.py                       # config RETRIEVER_MODE
    python evals/run_retrieval_eval.py --retriever hybrid
    python evals/run_retrieval_eval.py --retriever both      # dense vs hybrid ablation
    python evals/run_retrieval_eval.py --top-k 8 --mlflow
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.rag.retriever import RetrievedChunk, Retriever
from src.rag.hybrid import HybridRetriever
from src.rag.answerer import answer_question, IDK_ANSWER

GOLDEN_SET = Path(__file__).parent / "golden_set.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
HIT_KS = (1, 3, 5, 8)


def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_hit(chunk: RetrievedChunk, expected_doc_id: str, expected_section: str) -> bool:
    if chunk.doc_id != expected_doc_id:
        return False
    if not expected_section:
        return True
    needle = expected_section.lower()
    return needle in chunk.section_path.lower() or needle in chunk.text.lower()


def first_relevant_rank(chunks: list[RetrievedChunk], item: dict) -> int | None:
    """1-based rank of the first relevant chunk, or None if absent."""
    for rank, chunk in enumerate(chunks, start=1):
        if is_hit(chunk, item["expected_doc_id"], item.get("expected_section", "")):
            return rank
    return None


def _hit_metrics(rows: list[dict], top_k: int) -> dict:
    """hit_rate@k + MRR over rows that have an expected chunk."""
    n = len(rows)
    if n == 0:
        return {}
    ranks = [r["first_relevant_rank"] for r in rows]
    metrics = {
        f"hit_rate_at_{k}": sum(1 for r in ranks if r is not None and r <= k) / n
        for k in HIT_KS
        if k <= top_k
    }
    metrics["mrr"] = sum(1.0 / r for r in ranks if r is not None) / n
    metrics["n"] = n
    return metrics


def _make_retriever(mode: str):
    if mode == "hybrid":
        return HybridRetriever()
    return Retriever()


def _answer_abstained(question: str) -> tuple[bool, dict]:
    """Run the answerer end-to-end and decide whether it correctly declined.

    A negative is "correctly abstained" when the answerer signals it cannot answer
    the question from the corpus — either GroundedAnswer.insufficient_context is true,
    or the reply is the templated IDK response (the floor/no-context path). This is the
    layer where abstention is actually decided; the retrieval floor cannot see it.
    """
    ans, _ = answer_question(question)
    abstained = bool(ans.insufficient_context) or ans.answer.strip() == IDK_ANSWER.strip()
    detail = {
        "insufficient_context": bool(ans.insufficient_context),
        "n_citations": len(ans.citations),
        "answer_preview": ans.answer[:240],
    }
    return abstained, detail


def evaluate(mode: str, golden: list[dict], top_k: int, abstention_mode: str = "answer") -> dict:
    retriever = _make_retriever(mode)
    scored = []          # rows with an expected chunk (retrieval-scored tiers)
    negatives = []       # expect_abstention rows (abstention-scored)

    for item in golden:
        chunks = retriever.retrieve(item["question"], top_k=top_k)
        top_sim = chunks[0].similarity if chunks else 0.0
        record = {
            "id": item["id"],
            "topic": item.get("topic", "?"),
            "difficulty": item.get("difficulty", "?"),
            "question": item["question"],
            "top_similarity": round(top_sim, 4),
            "retrieved": [
                {"chunk_id": c.chunk_id, "section_path": c.section_path,
                 "similarity": round(c.similarity, 4)}
                for c in chunks
            ],
        }
        if item.get("expect_abstention"):
            # Retrieval-floor gate is a diagnostic only — it cannot separate
            # in-vocabulary-but-unanswerable questions (their top chunk is above
            # the floor). Abstention is judged at the answer layer by default.
            record["floor_gate_below"] = top_sim < config.SIMILARITY_FLOOR
            if abstention_mode == "answer":
                abstained, detail = _answer_abstained(item["question"])
                record.update(detail)
                record["abstained_correctly"] = abstained
            else:
                record["abstained_correctly"] = record["floor_gate_below"]
            negatives.append(record)
        else:
            record["expected_doc_id"] = item["expected_doc_id"]
            record["expected_section"] = item.get("expected_section", "")
            record["first_relevant_rank"] = first_relevant_rank(chunks, item)
            scored.append(record)

    overall = _hit_metrics(scored, top_k)

    per_topic, per_tier = {}, {}
    by_topic, by_tier = defaultdict(list), defaultdict(list)
    for r in scored:
        by_topic[r["topic"]].append(r)
        by_tier[r["difficulty"]].append(r)
    for topic, rows in sorted(by_topic.items()):
        per_topic[topic] = _hit_metrics(rows, top_k)
    for tier, rows in sorted(by_tier.items()):
        per_tier[tier] = _hit_metrics(rows, top_k)

    abstention = {}
    if negatives:
        abstention = {
            "mode": abstention_mode,
            "abstention_rate": sum(1 for r in negatives if r["abstained_correctly"]) / len(negatives),
            "n": len(negatives),
            # diagnostic: how many the retrieval floor alone would have caught
            "floor_gate_rate": sum(1 for r in negatives if r.get("floor_gate_below")) / len(negatives),
        }

    return {
        "retriever_mode": mode,
        "top_k": top_k,
        "similarity_floor": config.SIMILARITY_FLOOR,
        "overall": overall,
        "per_topic": per_topic,
        "per_tier": per_tier,
        "abstention": abstention,
        "scored": scored,
        "negatives": negatives,
    }


def _report_lines(report: dict) -> list[str]:
    """Human-readable summary of one retriever's run as a list of lines.

    Shared by the console output and the .log file so both stay identical.
    """
    lines: list[str] = []
    mode = report["retriever_mode"]
    lines.append(f"\n===== retriever={mode}  top_k={report['top_k']} =====")
    lines.append("Overall (lookup+multihop+near_miss+disambiguation):")
    for name, value in report["overall"].items():
        lines.append(f"  {name:>14}: {value:.3f}" if isinstance(value, float)
                     else f"  {name:>14}: {value}")

    lines.append("\nPer topic:")
    for topic, m in report["per_topic"].items():
        lines.append(f"  {topic}: hit@1={m.get('hit_rate_at_1', 0):.2f} "
                     f"hit@3={m.get('hit_rate_at_3', 0):.2f} mrr={m.get('mrr', 0):.2f} "
                     f"(n={m.get('n', 0)})")

    lines.append("Per tier:")
    for tier, m in report["per_tier"].items():
        lines.append(f"  {tier:>14}: hit@1={m.get('hit_rate_at_1', 0):.2f} "
                     f"hit@3={m.get('hit_rate_at_3', 0):.2f} mrr={m.get('mrr', 0):.2f} "
                     f"(n={m.get('n', 0)})")

    if report["abstention"]:
        a = report["abstention"]
        if a.get("mode") == "answer":
            lines.append(f"\nAbstention (negative tier, answer-level): {a['abstention_rate']:.3f} "
                         f"correctly declined (n={a['n']})")
            lines.append(f"  [diagnostic] retrieval-floor gate <{report['similarity_floor']}: "
                         f"{a.get('floor_gate_rate', 0):.3f} — non-discriminative for "
                         f"in-vocabulary questions, not the abstention metric")
        else:
            lines.append(f"\nAbstention (negative tier, floor gate <{report['similarity_floor']}): "
                         f"{a['abstention_rate']:.3f} (n={a['n']})")

    misses = [r for r in report["scored"] if r["first_relevant_rank"] is None]
    if misses:
        lines.append(f"\nMisses ({len(misses)}):")
        for r in misses:
            lines.append(f"  [{r['id']}/{r['difficulty']}] {r['question']}  "
                         f"(expected {r['expected_doc_id']} / {r['expected_section']})")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval hit-rate eval on the golden set")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--retriever", choices=("dense", "hybrid", "both"),
                        default=config.RETRIEVER_MODE, help="which retriever(s) to evaluate")
    parser.add_argument("--mlflow", action="store_true", help="log the run(s) to MLflow")
    parser.add_argument("--abstention", choices=("answer", "floor"), default="answer",
                        help="how to score the negative tier: 'answer' (default) runs the "
                             "answerer end-to-end and checks the decline signal; 'floor' uses "
                             "only the retrieval similarity floor (legacy diagnostic)")
    args = parser.parse_args()

    golden = load_golden_set()
    modes = ("dense", "hybrid") if args.retriever == "both" else (args.retriever,)
    n_scored = sum(1 for g in golden if not g.get("expect_abstention"))
    n_neg = sum(1 for g in golden if g.get("expect_abstention"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports = [evaluate(mode, golden, top_k=args.top_k, abstention_mode=args.abstention)
               for mode in modes]

    # Build a single human-readable summary shared by the console and the .log file.
    log_lines = [
        "Retrieval accuracy eval — golden set",
        f"timestamp:       {datetime.now(timezone.utc).isoformat()}",
        f"retriever(s):    {', '.join(modes)}",
        f"top_k:           {args.top_k}",
        f"embedding_model: {config.ACTIVE_EMBEDDING_MODEL}",
        f"rrf_k:           {config.RRF_K}",
        f"golden set:      {len(golden)} questions ({n_scored} scored, {n_neg} negatives)",
        f"abstention:      {args.abstention}-level scoring",
    ]
    for report in reports:
        log_lines.extend(_report_lines(report))

    summary = "\n".join(log_lines)
    print(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"retrieval_eval_{stamp}.json"
    out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    log_path = RESULTS_DIR / f"retrieval_eval_{stamp}.log"
    log_path.write_text(summary + "\n", encoding="utf-8")
    print(f"\nSummary log written to {log_path}")
    print(f"Full JSON report written to {out_path}")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("retrieval-eval")
        for report in reports:
            with mlflow.start_run(run_name=f"retrieval-{report['retriever_mode']}"):
                mlflow.log_params({
                    "retriever_mode": report["retriever_mode"],
                    "top_k": args.top_k,
                    "embedding_model": config.ACTIVE_EMBEDDING_MODEL,
                    "rrf_k": config.RRF_K,
                    "golden_set_size": len(golden),
                })
                flat = {f"overall_{k}": v for k, v in report["overall"].items()}
                for topic, m in report["per_topic"].items():
                    flat[f"{topic}_hit_rate_at_1"] = m.get("hit_rate_at_1", 0)
                if report["abstention"]:
                    flat["abstention_rate"] = report["abstention"]["abstention_rate"]
                    flat["abstention_floor_gate_rate"] = report["abstention"].get("floor_gate_rate", 0)
                mlflow.log_metrics(flat)
                mlflow.log_artifact(str(out_path))
        print("Logged to MLflow experiment 'retrieval-eval'.")


if __name__ == "__main__":
    main()
