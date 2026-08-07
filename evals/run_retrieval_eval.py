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
    **abstention signal**: the top similarity must fall below SIMILARITY_FLOOR so
    the agent would answer "I don't know" instead of retrieving a false positive.

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


def evaluate(mode: str, golden: list[dict], top_k: int) -> dict:
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
            record["abstained_correctly"] = top_sim < config.SIMILARITY_FLOOR
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
            "abstention_rate": sum(1 for r in negatives if r["abstained_correctly"]) / len(negatives),
            "n": len(negatives),
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
        lines.append(f"\nAbstention (negative tier): {a['abstention_rate']:.3f} "
                     f"correctly below floor {report['similarity_floor']} (n={a['n']})")

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
    args = parser.parse_args()

    golden = load_golden_set()
    modes = ("dense", "hybrid") if args.retriever == "both" else (args.retriever,)
    n_scored = sum(1 for g in golden if not g.get("expect_abstention"))
    n_neg = sum(1 for g in golden if g.get("expect_abstention"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports = [evaluate(mode, golden, top_k=args.top_k) for mode in modes]

    # Build a single human-readable summary shared by the console and the .log file.
    log_lines = [
        "Retrieval accuracy eval — golden set",
        f"timestamp:       {datetime.now(timezone.utc).isoformat()}",
        f"retriever(s):    {', '.join(modes)}",
        f"top_k:           {args.top_k}",
        f"embedding_model: {config.ACTIVE_EMBEDDING_MODEL}",
        f"rrf_k:           {config.RRF_K}",
        f"golden set:      {len(golden)} questions ({n_scored} scored, {n_neg} negatives)",
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
                mlflow.log_metrics(flat)
                mlflow.log_artifact(str(out_path))
        print("Logged to MLflow experiment 'retrieval-eval'.")


if __name__ == "__main__":
    main()
