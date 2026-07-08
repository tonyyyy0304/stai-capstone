"""Retrieval evaluation on the golden set: hit-rate@k and MRR (PLAN.md §3.2 Stage 6).

A retrieved chunk is a "hit" when its doc_id matches the expected doc and the
expected section keyword appears in its section_path or text (robust to chunk
index changes across re-ingestions).

Usage:
    python evals/run_retrieval_eval.py                 # uses config TOP_K
    python evals/run_retrieval_eval.py --top-k 8 --mlflow
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.rag.retriever import RetrievedChunk, Retriever

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


def evaluate(retriever: Retriever, golden: list[dict], top_k: int) -> dict:
    per_question = []
    for item in golden:
        chunks = retriever.retrieve(item["question"], top_k=top_k)
        rank = first_relevant_rank(chunks, item)
        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_doc_id": item["expected_doc_id"],
                "expected_section": item.get("expected_section", ""),
                "first_relevant_rank": rank,
                "top_similarity": chunks[0].similarity if chunks else 0.0,
                "retrieved": [
                    {"chunk_id": c.chunk_id, "section_path": c.section_path,
                     "similarity": round(c.similarity, 4)}
                    for c in chunks
                ],
            }
        )

    n = len(per_question)
    ranks = [q["first_relevant_rank"] for q in per_question]
    metrics = {
        f"hit_rate_at_{k}": sum(1 for r in ranks if r is not None and r <= k) / n
        for k in HIT_KS
        if k <= top_k
    }
    metrics["mrr"] = sum(1.0 / r for r in ranks if r is not None) / n
    return {"metrics": metrics, "per_question": per_question}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval hit-rate eval on the golden set")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()

    golden = load_golden_set()
    print(f"Evaluating retrieval on {len(golden)} golden questions (top_k={args.top_k})...")
    report = evaluate(Retriever(), golden, top_k=args.top_k)

    print("\nMetrics:")
    for name, value in report["metrics"].items():
        print(f"  {name:>14}: {value:.3f}")

    misses = [q for q in report["per_question"] if q["first_relevant_rank"] is None]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for q in misses:
            print(f"  [{q['id']}] {q['question']}  (expected {q['expected_doc_id']} / "
                  f"{q['expected_section']})")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"retrieval_eval_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("retrieval-eval")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "top_k": args.top_k,
                    "embedding_model": config.ACTIVE_EMBEDDING_MODEL,
                    "embedding_dim": config.EMBEDDING_DIM,
                    "chunk_target_tokens": config.CHUNK_TARGET_TOKENS,
                    "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
                    "golden_set_size": len(golden),
                }
            )
            mlflow.log_metrics(report["metrics"])
            mlflow.log_artifact(str(out_path))
        print("Logged to MLflow experiment 'retrieval-eval'.")


if __name__ == "__main__":
    main()
