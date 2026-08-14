"""End-to-end answer evaluation on the golden set.

Design notes:

- **Runs the real agent, not just the RAG layer.** Every row goes through
  `orchestrator.run_turn()`, so the router, guardrails, ReAct loop, web fallback,
  and grounded answerer are all in the measured path. `answerer.answer_question()`
  would have been cheaper, but it bypasses the agent entirely -- that would make
  this a second generation eval wearing an end-to-end name.

- **Task success is tier-dependent.** The golden set's five tiers do not share a
  success condition, so pooling them into one "accuracy" number would be
  meaningless. A `negative` row succeeds by *declining*; a `disambiguation` row
  succeeds by *asking*; the answerable tiers succeed by answering correctly, with
  a valid citation, faithfully. Each is scored on its own terms and reported
  separately, then combined into an overall task-success rate.

- **Detection vs. policy, as everywhere else in this repo.** The LLM judge only
  *scores* (faithfulness 1-5, key-fact coverage); the pass/fail thresholds are
  deterministic constants below, never the model's own verdict. Mirrors
  src/guardrails/llm_judge.py.

- **Judge calls are SHA-256 disk-cached** (same pattern as src/ocr/extractor.py)
  because the free-tier quota is per-day and this harness is re-run often. A
  cached row costs zero API calls. Change RUBRIC_VERSION to invalidate.

- **Citation scoring is doc+section, not page.** The golden set carries
  `expected_doc_id` / `expected_section` but no `expected_page`, so page-number
  *correctness* is not scoreable here and is deliberately not claimed. Page
  *population* is reported instead as a coverage diagnostic (are citations
  carrying page numbers at all), which is what the data supports.

Each run writes two files to evals/results/, stamped with a UTC timestamp:
    answer_eval_<stamp>.log   -- human-readable summary
    answer_eval_<stamp>.json  -- full machine-readable report (per-question detail)

Usage:
    python evals/run_answer_eval.py                      # full golden set
    python evals/run_answer_eval.py --limit 5            # smoke test, 5 rows
    python evals/run_answer_eval.py --tier lookup        # one tier only
    python evals/run_answer_eval.py --no-judge           # deterministic metrics only, zero judge calls
    python evals/run_answer_eval.py --judge-model gemini-2.0-flash   # cross-model judge (bias check)
    python evals/run_answer_eval.py --trace L01          # dump one full reasoning trace, no scoring
    python evals/run_answer_eval.py --mlflow
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from src import config
from src.agent.orchestrator import API_ERROR_REPLY, OUT_OF_SCOPE_REPLY, run_turn
from src.agent.router import needs_clarification
from src.rag.answerer import IDK_ANSWER
from src.schemas import IntentClassification

GOLDEN_SET = Path(__file__).parent / "golden_set.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
JUDGE_CACHE_DIR = RESULTS_DIR / "judge_cache"

# Bump when the rubric or prompt changes -- old cached verdicts were scored
# against a different rubric and must not be reused.
RUBRIC_VERSION = "v1"

# Deterministic pass thresholds. The judge scores; these decide.
FAITHFULNESS_PASS = 4      # 1-5 scale, >=4 means core claims are grounded
KEY_FACT_PASS = 0.67       # >=2/3 of the row's answer_key_facts must be covered

ANSWERABLE_TIERS = ("lookup", "multihop", "near_miss")


# --- Judge schema (local to the eval; src/schemas.py is app-side) -----------

class AnswerVerdict(BaseModel):
    """One structured judge call scoring an answer on two axes at once."""

    reasoning: str = Field(description="Step-by-step justification, written before the scores")
    faithfulness: int = Field(
        ge=1, le=5,
        description=(
            "Is every claim in the answer supported by the provided excerpts? "
            "5=every claim directly supported; 4=minor unsupported detail, core grounded; "
            "3=some claims lack support; 2=major claims unsupported; 1=contradicts or ignores excerpts"
        ),
    )
    key_facts_covered: list[str] = Field(
        default_factory=list,
        description="Verbatim copies of the expected key facts that the answer actually states",
    )


JUDGE_PROMPT = """You are a strict evaluator for a university faculty onboarding assistant.

You are given a question, the source excerpts the assistant retrieved, the expected \
key facts a correct answer must contain, and the assistant's actual answer.

Score two things:

1. FAITHFULNESS (1-5) -- is every claim in the answer supported by the excerpts?
   5 = every claim is directly supported by the excerpts
   4 = minor unsupported detail, but the core answer is grounded
   3 = some claims lack support in the excerpts
   2 = major claims are unsupported
   1 = the answer contradicts or ignores the excerpts
   Judge only against the excerpts provided. An answer can be factually true in the \
real world and still score low here if the excerpts do not support it.

2. KEY FACTS COVERED -- for each expected key fact, decide whether the answer \
actually states it. Copy the matching expected key facts verbatim into \
key_facts_covered. A fact counts as covered if the answer states it in any \
wording; it does not count if the answer merely gestures at the topic.

Do not reward length, confidence, or polish. A short answer that states the key \
facts and stays grounded scores higher than a long one that does not.

Question: {question}

Retrieved excerpts:
{context}

Expected key facts:
{key_facts}

Assistant's answer:
{answer}

Think step by step in `reasoning`, then output the scores."""


# --- Judge with disk cache --------------------------------------------------

def _cache_key(question: str, answer: str, context: str, key_facts: list[str], model: str) -> str:
    payload = json.dumps(
        {"v": RUBRIC_VERSION, "q": question, "a": answer,
         "c": context, "k": key_facts, "m": model},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_verdict(key: str) -> AnswerVerdict | None:
    path = JUDGE_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return AnswerVerdict.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # corrupt entry or schema drift -- treat as a miss, never crash
        return None


def _save_verdict(key: str, verdict: AnswerVerdict) -> None:
    try:
        JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (JUDGE_CACHE_DIR / f"{key}.json").write_text(
            verdict.model_dump_json(indent=2), encoding="utf-8"
        )
    except Exception:
        pass  # a cache write failure must never fail the eval run


def judge_answer(
    question: str, answer: str, context: str, key_facts: list[str],
    model: str, client=None,
) -> tuple[AnswerVerdict | None, bool]:
    """Returns (verdict, was_cached). Verdict is None on any backend error --
    fail-open, same as src/guardrails/llm_judge.py: one flaky call must not
    take down a 30-row eval run."""
    key = _cache_key(question, answer, context, key_facts, model)
    cached = _load_cached_verdict(key)
    if cached is not None:
        return cached, True

    import httpx
    from google.genai import types
    from google.genai.errors import APIError

    from src.agent.llm_client import LLMBackendError

    client = client or config.get_llm_client()
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context or "(no excerpts retrieved)",
        key_facts="\n".join(f"- {f}" for f in key_facts) or "(none specified)",
        answer=answer,
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnswerVerdict,
                temperature=0.0,
                thinking_config=config.thinking_config(),
            ),
        )
        verdict = response.parsed
    except (APIError, LLMBackendError, httpx.TimeoutException) as exc:
        print(f"    [judge error] {type(exc).__name__}: {exc}")
        return None, False

    if verdict is None:
        return None, False
    _save_verdict(key, verdict)
    return verdict, False


# --- Behaviour detection off the real AgentResponse -------------------------

def _did_clarify(response) -> bool:
    """True when the agent asked a clarifying question instead of answering.

    Two separate paths reach that outcome and neither sets a flag on
    AgentResponse, so both are reconstructed from `steps` here rather than by
    string-matching the reply:

    1. Router-level (orchestrator.py, `needs_clarification`): the classification
       step's observation is the full IntentClassification JSON, so the same
       predicate the orchestrator used can simply be re-applied to it.
    2. Audience-class level (orchestrator.py `_react_loop`): leaves an explicit
       AgentStep with observation == "requires_clarification".
    """
    for step in response.steps:
        if step.observation == "requires_clarification":
            return True
    for step in response.steps:
        if step.tool is None and step.thought.startswith("classified intent="):
            try:
                classification = IntentClassification.model_validate_json(step.observation)
            except Exception:
                continue
            if needs_clarification(classification):
                return True
    return False


def _is_infrastructure_error(response) -> bool:
    """True when the turn never produced an agent answer at all -- an LLM
    timeout or backend failure surfaced as API_ERROR_REPLY.

    These rows must not be scored. Counting a 120s timeout as "the agent
    answered but cited badly" silently converts an infrastructure problem into
    an accuracy problem and drags the headline number down for the wrong
    reason. They are excluded from every rate and reported separately.
    """
    return response.reply.strip() == API_ERROR_REPLY.strip()


def _did_abstain(response) -> bool:
    """The agent declined to answer from the corpus.

    Three distinct replies mean "I won't answer this", and all three are
    correct behaviour on the negative tier: the answerer's templated IDK, the
    structured insufficient_context flag, and the router's out-of-scope decline
    (a negative row like "how long does NBI take to release a clearance" can
    legitimately be caught at the router before retrieval ever runs).
    """
    return (
        bool(response.insufficient_context)
        or response.reply.strip() == IDK_ANSWER.strip()
        or response.reply.strip() == OUT_OF_SCOPE_REPLY.strip()
    )


def _citation_ok(response, item: dict) -> bool:
    """A citation is valid when it points at the expected document and, when the
    row names a section, that section.

    Deliberately mirrors run_retrieval_eval.py's is_hit(): the section needle is
    matched against the chunk's section_path **or its text**, because
    `expected_section` in the golden set is sometimes a content phrase rather
    than a heading (L03's is "competency interview"). Matching section_path
    alone fails correct answers.

    Citation carries no text, so the cited chunk is looked up by chunk_id in the
    chunks the turn actually retrieved -- which is also a second, implicit check
    that the citation refers to something real.

    Page numbers are not scored: the golden set has no expected_page.
    """
    expected_doc = item.get("expected_doc_id")
    if not expected_doc:
        return False
    section = (item.get("expected_section") or "").lower()
    by_id = {c.chunk_id: c for c in response.chunks}

    for citation in response.citations:
        chunk = by_id.get(citation.chunk_id)
        doc_id = chunk.doc_id if chunk else ""
        # Fall back to the chunk_id prefix when the chunk isn't in this turn's
        # set: chunking.py mints ids as f"{doc_id}#{nnn}".
        if not doc_id:
            doc_id = (citation.chunk_id or "").split("#")[0]
        if doc_id != expected_doc:
            continue
        if not section:
            return True
        haystack = (citation.section_path or "").lower()
        if chunk is not None:
            haystack += " " + chunk.text.lower()
        if section in haystack:
            return True
    return False


def _context_text(response, max_chars: int = 6000) -> str:
    parts = [f"[{c.section_path}] {c.text}" for c in response.chunks]
    return "\n\n".join(parts)[:max_chars]


# --- Per-row scoring --------------------------------------------------------

def score_row(item: dict, response, latency_ms: float, use_judge: bool,
              judge_model: str, client=None) -> dict:
    tier = item.get("difficulty", "?")
    abstained = _did_abstain(response)
    clarified = _did_clarify(response)
    citation_ok = _citation_ok(response, item)

    record = {
        "id": item["id"],
        "topic": item.get("topic", "?"),
        "difficulty": tier,
        "question": item["question"],
        "reply_preview": response.reply[:300],
        "abstained": abstained,
        "clarified": clarified,
        "citation_ok": citation_ok,
        "n_citations": len(response.citations),
        "n_citations_with_page": sum(1 for c in response.citations if c.page),
        "latency_ms": round(latency_ms, 1),
        "total_tokens": response.token_usage.total_tokens,
        "n_steps": len(response.steps),
        "tool_sequence": [s.tool for s in response.steps if s.tool],
        "judged": False,
        "error": _is_infrastructure_error(response),
    }

    # The turn never produced an answer (timeout / backend failure). Not scored
    # either way -- excluded from all rates, surfaced separately in the report.
    if record["error"]:
        record["success"] = None
        record["partial"] = None
        return record

    if tier == "negative":
        record["success"] = abstained
        record["partial"] = 1.0 if abstained else 0.0
        return record

    if item.get("expect_clarification"):
        record["success"] = clarified
        record["partial"] = 1.0 if clarified else 0.0
        return record

    # Answerable tiers: must not abstain, must cite correctly, and (when the
    # judge is on) must be faithful and cover the expected key facts.
    key_facts = item.get("answer_key_facts", [])
    components = {"answered": not abstained and not clarified, "citation_ok": citation_ok}

    if use_judge and components["answered"]:
        verdict, was_cached = judge_answer(
            item["question"], response.reply, _context_text(response),
            key_facts, judge_model, client=client,
        )
        if verdict is not None:
            coverage = (len(verdict.key_facts_covered) / len(key_facts)) if key_facts else 1.0
            record.update({
                "judged": True,
                "judge_cached": was_cached,
                "faithfulness": verdict.faithfulness,
                "key_fact_coverage": round(coverage, 3),
                "judge_reasoning": verdict.reasoning[:400],
            })
            components["faithful"] = verdict.faithfulness >= FAITHFULNESS_PASS
            components["key_facts"] = coverage >= KEY_FACT_PASS

    record["components"] = components
    record["success"] = all(components.values())
    record["partial"] = round(sum(1 for v in components.values() if v) / len(components), 3)
    return record


# --- Aggregation ------------------------------------------------------------

def _rate(rows: list[dict], field: str = "success") -> float:
    return (sum(1 for r in rows if r.get(field)) / len(rows)) if rows else 0.0


def _summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "task_success_rate": round(_rate(rows), 3),
        "partial_credit": round(sum(r.get("partial", 0.0) for r in rows) / len(rows), 3),
    }


def aggregate(all_rows: list[dict]) -> dict:
    # Infrastructure failures are excluded from every rate below, including
    # latency: a 120s timeout is not a latency measurement of a working agent.
    errored = [r for r in all_rows if r.get("error")]
    rows = [r for r in all_rows if not r.get("error")]
    if not rows:
        return {"overall": {}, "per_tier": {}, "per_topic": {},
                "n_errors": len(errored), "errored_ids": [r["id"] for r in errored]}

    by_tier, by_topic = defaultdict(list), defaultdict(list)
    for r in rows:
        by_tier[r["difficulty"]].append(r)
        by_topic[r["topic"]].append(r)

    answerable = [r for r in rows if r["difficulty"] in ANSWERABLE_TIERS
                  and not r.get("clarified")]
    judged = [r for r in rows if r.get("judged")]
    negatives = [r for r in rows if r["difficulty"] == "negative"]
    clarify_rows = [r for r in rows if r["difficulty"] == "disambiguation"]
    latencies = sorted(r["latency_ms"] for r in rows)

    def _pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(int(round(p * (len(latencies) - 1))), len(latencies) - 1)
        return round(latencies[idx], 1)

    return {
        "overall": _summarise(rows),
        "per_tier": {t: _summarise(rs) for t, rs in sorted(by_tier.items())},
        "per_topic": {t: _summarise(rs) for t, rs in sorted(by_topic.items())},
        "n_errors": len(errored),
        "errored_ids": [r["id"] for r in errored],
        "citation_validity_rate": round(_rate(answerable, "citation_ok"), 3) if answerable else 0.0,
        "citation_page_population_rate": round(
            sum(r["n_citations_with_page"] for r in answerable)
            / max(sum(r["n_citations"] for r in answerable), 1), 3),
        "abstention_accuracy": round(_rate(negatives), 3) if negatives else None,
        "clarification_accuracy": round(_rate(clarify_rows), 3) if clarify_rows else None,
        "mean_faithfulness": round(
            statistics.mean(r["faithfulness"] for r in judged), 2) if judged else None,
        "mean_key_fact_coverage": round(
            statistics.mean(r["key_fact_coverage"] for r in judged), 3) if judged else None,
        "n_judged": len(judged),
        "latency_p50_ms": _pct(0.50),
        "latency_p95_ms": _pct(0.95),
        "mean_tokens_per_turn": round(
            statistics.mean(r["total_tokens"] for r in rows), 1) if rows else 0.0,
    }


# --- Reporting --------------------------------------------------------------

def _report_lines(summary: dict, rows: list[dict]) -> list[str]:
    lines = ["\n===== end-to-end answer eval ====="]
    if not summary.get("overall"):
        lines.append(f"No scoreable rows -- all {summary.get('n_errors', 0)} turns failed "
                     f"with a backend/timeout error: {summary.get('errored_ids')}")
        return lines
    o = summary["overall"]
    lines.append(f"Task success (all tiers): {o['task_success_rate']:.3f}  "
                 f"partial-credit {o['partial_credit']:.3f}  (n={o['n']})")
    if summary.get("n_errors"):
        lines.append(f"  [excluded] {summary['n_errors']} turn(s) failed with a backend/timeout "
                     f"error and are not scored: {', '.join(summary['errored_ids'])}")

    lines.append("\nPer tier:")
    for tier, m in summary["per_tier"].items():
        lines.append(f"  {tier:>15}: success={m['task_success_rate']:.2f} "
                     f"partial={m['partial_credit']:.2f} (n={m['n']})")

    lines.append("Per topic:")
    for topic, m in summary["per_topic"].items():
        flag = "  <- n too small to interpret" if m["n"] < 5 else ""
        lines.append(f"  {topic:>15}: success={m['task_success_rate']:.2f} "
                     f"(n={m['n']}){flag}")

    lines.append("\nBehaviour metrics:")
    if summary["abstention_accuracy"] is not None:
        lines.append(f"  abstention accuracy (negative tier):     {summary['abstention_accuracy']:.3f}")
    if summary["clarification_accuracy"] is not None:
        lines.append(f"  clarification accuracy (disambig tier):  {summary['clarification_accuracy']:.3f}")
    lines.append(f"  citation validity (answerable rows):     {summary['citation_validity_rate']:.3f}")
    lines.append(f"  citations carrying a page number:        {summary['citation_page_population_rate']:.3f} "
                 f"(coverage diagnostic; page correctness is not scored -- no expected_page in the golden set)")

    if summary["n_judged"]:
        lines.append(f"\nLLM-as-judge (n={summary['n_judged']}):")
        lines.append(f"  mean faithfulness (1-5):    {summary['mean_faithfulness']:.2f} "
                     f"(pass bar >={FAITHFULNESS_PASS})")
        lines.append(f"  mean key-fact coverage:     {summary['mean_key_fact_coverage']:.3f} "
                     f"(pass bar >={KEY_FACT_PASS})")

    lines.append("\nCost / efficiency:")
    lines.append(f"  latency p50: {summary['latency_p50_ms']:.0f} ms   "
                 f"p95: {summary['latency_p95_ms']:.0f} ms")
    lines.append(f"  mean tokens/turn: {summary['mean_tokens_per_turn']:.0f}")

    failures = [r for r in rows if not r.get("error") and not r.get("success")]
    if failures:
        lines.append(f"\nFailures ({len(failures)}):")
        for r in failures:
            detail = r.get("components") or {}
            failed = [k for k, v in detail.items() if not v] or ["behaviour"]
            lines.append(f"  [{r['id']}/{r['difficulty']}] {r['question'][:80]}")
            lines.append(f"      failed on: {', '.join(failed)}  |  {r['reply_preview'][:100]}")
    return lines


def _print_trace(item: dict, response) -> None:
    """One full agent decision chain, for the reasoning-trace slide (spec §6.1)."""
    print(f"\n===== reasoning trace: {item['id']} ({item.get('topic')}/{item.get('difficulty')}) =====")
    print(f"Question: {item['question']}\n")
    for i, step in enumerate(response.steps, start=1):
        print(f"  Step {i}")
        print(f"    Think:  {step.thought}")
        if step.tool:
            print(f"    Tool:   {step.tool}({json.dumps(step.tool_args, ensure_ascii=False)})")
        obs = step.observation or ""
        print(f"    Observe: {obs[:400]}{'...' if len(obs) > 400 else ''}")
    print(f"\n  Final answer: {response.reply}")
    print(f"  Citations ({len(response.citations)}):")
    for c in response.citations:
        print(f"    - {c.title} > {c.section_path} (p.{c.page or '?'}) [{c.chunk_id}]")
    print(f"  Tokens: {response.token_usage.total_tokens}   Steps: {len(response.steps)}")


# --- Entry point ------------------------------------------------------------

def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end answer eval on the golden set")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N rows")
    parser.add_argument("--tier", default=None,
                        help="only run one tier (lookup|multihop|disambiguation|negative|near_miss)")
    parser.add_argument("--topic", default=None, help="only run one topic (T1|T2|T3)")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip all LLM-judge calls (deterministic metrics only, zero API cost)")
    parser.add_argument("--judge-model", default=config.ACTIVE_CHAT_MODEL,
                        help="model to judge with; use a different one from the answering model "
                             "to mitigate self-preference bias")
    parser.add_argument("--trace", default=None, metavar="ROW_ID",
                        help="print one full reasoning trace for this row id and exit")
    parser.add_argument("--sleep", type=float, default=0.0, metavar="SECONDS",
                        help="wait this long between rows. The free-tier Gemini key is "
                             "rate-limited per minute and a full 32-row run makes ~150 calls; "
                             "4-6s spreads them out and avoids a lockout window")
    parser.add_argument("--retry", type=int, default=0, metavar="N",
                        help="retry a row up to N times when the turn fails with a backend/"
                             "timeout error, backing off 15s, 30s, 60s... between attempts")
    parser.add_argument("--resume", default=None, metavar="PRIOR_JSON",
                        help="path to a previous answer_eval_*.json; rows that already scored "
                             "successfully are reused as-is and only errored/missing rows are "
                             "re-run. Lets a full run be accumulated across several quota windows")
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()

    golden = load_golden_set()

    if args.trace:
        matches = [g for g in golden if g["id"] == args.trace]
        if not matches:
            print(f"No golden-set row with id={args.trace}")
            sys.exit(1)
        item = matches[0]
        response = run_turn(session_id=f"trace-{uuid.uuid4().hex[:8]}", message=item["question"])
        _print_trace(item, response)
        return

    if args.tier:
        golden = [g for g in golden if g.get("difficulty") == args.tier]
    if args.topic:
        golden = [g for g in golden if g.get("topic") == args.topic]
    if args.limit:
        golden = golden[:args.limit]
    if not golden:
        print("No rows matched the filters.")
        sys.exit(1)

    use_judge = not args.no_judge
    client = config.get_llm_client() if use_judge else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # --resume: reuse rows that already scored cleanly in a prior run, so a
    # rate-limited run can be completed across several quota windows instead of
    # being restarted from scratch each time.
    reused: dict[str, dict] = {}
    if args.resume:
        prior = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        reused = {r["id"]: r for r in prior.get("rows", []) if not r.get("error")}
        print(f"Resuming from {args.resume}: {len(reused)} row(s) already scored, "
              f"{len(golden) - len(reused)} to run.\n")

    rows = []
    for i, item in enumerate(golden, start=1):
        if item["id"] in reused:
            rows.append(reused[item["id"]])
            print(f"[{i}/{len(golden)}] {item['id']} -- reused from prior run")
            continue

        print(f"[{i}/{len(golden)}] {item['id']} ({item.get('difficulty')}) {item['question'][:70]}")
        record = None
        for attempt in range(args.retry + 1):
            started = time.perf_counter()
            response = run_turn(session_id=f"eval-{uuid.uuid4().hex[:8]}", message=item["question"])
            latency_ms = (time.perf_counter() - started) * 1000
            record = score_row(item, response, latency_ms, use_judge, args.judge_model, client=client)
            if not record["error"] or attempt == args.retry:
                break
            backoff = 15 * (2 ** attempt)
            print(f"      backend error -- retrying in {backoff}s "
                  f"(attempt {attempt + 2}/{args.retry + 1})")
            time.sleep(backoff)

        rows.append(record)
        status = "ERROR (excluded)" if record["error"] else f"success={record['success']}"
        print(f"      -> {status}  {record['latency_ms']:.0f}ms  {record['total_tokens']} tok"
              + (f"  faithfulness={record['faithfulness']}" if record.get("judged") else ""))

        if args.sleep and i < len(golden):
            time.sleep(args.sleep)

    summary = aggregate(rows)

    header = [
        "End-to-end answer eval -- golden set",
        f"timestamp:       {datetime.now(timezone.utc).isoformat()}",
        f"rows evaluated:  {len(rows)} of {len(load_golden_set())} in the golden set",
        f"answer model:    {config.ACTIVE_CHAT_MODEL}",
        f"judge model:     {args.judge_model if use_judge else '(judge disabled)'}",
        f"retriever mode:  {config.RETRIEVER_MODE}   top_k={config.TOP_K}",
        f"self-preference: {'SAME model judges its own output -- see write-up caveat' if use_judge and args.judge_model == config.ACTIVE_CHAT_MODEL else 'cross-model judge'}",
    ]
    output = "\n".join(header + _report_lines(summary, rows))
    print(output)

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / f"answer_eval_{stamp}.json"
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    log_path = RESULTS_DIR / f"answer_eval_{stamp}.log"
    log_path.write_text(output + "\n", encoding="utf-8")
    print(f"\nSummary log written to {log_path}")
    print(f"Full JSON report written to {json_path}")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("answer-eval")
        with mlflow.start_run(run_name=f"answer-eval-{stamp}"):
            mlflow.log_params({
                "answer_model": config.ACTIVE_CHAT_MODEL,
                "judge_model": args.judge_model if use_judge else "disabled",
                "retriever_mode": config.RETRIEVER_MODE,
                "top_k": config.TOP_K,
                "rows_evaluated": len(rows),
                "rubric_version": RUBRIC_VERSION,
            })
            flat = {
                "task_success_rate": summary["overall"]["task_success_rate"],
                "partial_credit": summary["overall"]["partial_credit"],
                "citation_validity_rate": summary["citation_validity_rate"],
                "latency_p50_ms": summary["latency_p50_ms"],
                "latency_p95_ms": summary["latency_p95_ms"],
                "mean_tokens_per_turn": summary["mean_tokens_per_turn"],
            }
            for key in ("abstention_accuracy", "clarification_accuracy",
                        "mean_faithfulness", "mean_key_fact_coverage"):
                if summary.get(key) is not None:
                    flat[key] = summary[key]
            for tier, m in summary["per_tier"].items():
                flat[f"tier_{tier}_success"] = m["task_success_rate"]
            mlflow.log_metrics(flat)
            mlflow.log_artifact(str(json_path))
        print("Logged to MLflow experiment 'answer-eval'.")


if __name__ == "__main__":
    main()