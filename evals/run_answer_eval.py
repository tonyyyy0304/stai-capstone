"""End-to-end answer evaluation on the golden set.

Design notes:

- **Runs the real agent, not just the RAG layer.** Every row goes through
  `orchestrator.run_turn()`, so the router, guardrails, ReAct loop, web fallback,
  and grounded answerer are all in the measured path. `answerer.answer_question()`
  would have been cheaper, but it bypasses the agent entirely -- that would make
  this a second generation eval wearing an end-to-end name.

- **Task success is tier-dependent.** The golden set's tiers do not share a
  success condition, so pooling them into one "accuracy" number would be
  meaningless. A `negative` row succeeds by *declining*; every other tier
  (lookup/multihop/near_miss) succeeds by answering correctly, with a valid
  citation, faithfully. Each is scored on its own terms and reported separately,
  then combined into an overall task-success rate. There is no separate
  `disambiguation` tier -- see below.

- **No row is scored on "did it ask a clarifying question" -- ever.** Any row,
  regardless of tier, may hit the agent's audience-segment clarification (the
  corpus's near-duplicate per-class sections). When that happens
  (`_did_clarify`), the harness itself generates a plausible follow-up reply
  (`_auto_generate_clarify_reply`) -- no golden-set field scripts this -- sends
  it as a second turn with the real conversation history threaded through
  (exactly like a live follow-up), and scores THAT final answer on
  faithfulness/correctness like any other row. If the turn answers directly
  without asking (the fact turned out to be class-invariant), that first
  answer is scored as-is. Both paths go through the identical judge and
  component checks -- asking is never itself rewarded or penalized, only the
  resulting answer is. `_auto_generate_clarify_reply` first tries a
  deterministic parse of the audience-segment question template (picks the
  first listed class -- reproducible, zero extra API cost) and falls back to
  one cached LLM call for any clarifying question that doesn't match that
  shape (e.g. a router-level "could you be more specific?").

- **Detection vs. policy, as everywhere else in this repo.** The LLM judge only
  *scores* (faithfulness 1-5, answer correctness 1-5); the
  pass/fail thresholds are deterministic constants below, never the model's own
  verdict. Mirrors src/guardrails/llm_judge.py.

- **Answer correctness is scored against a reference answer, not the excerpts.**
  Reads from `evals/golden_set_with_ground_truth.jsonl` -- the golden set plus a
  `ground_truth` field per answerable row (negative rows are correct by
  abstaining, so those alone have no single-answer reference). Rows that used
  to be the `disambiguation` tier keep a single-class-scoped `ground_truth`
  (see `_auto_generate_clarify_reply`'s "first listed class" determinism, which
  is what keeps that reference reproducible run to run). This is a second axis
  from faithfulness: faithfulness checks the answer against what was
  *retrieved*, correctness checks it against what was *expected*, so an answer
  can be faithful to bad excerpts yet still be marked wrong here.

- **Judge calls are SHA-256 disk-cached** (same pattern as src/ocr/extractor.py)
  because the free-tier quota is per-day and this harness is re-run often. A
  cached row costs zero API calls. Change RUBRIC_VERSION to invalidate.

- **Citation scoring is doc+section, not page.** The golden set carries
  `expected_doc_id` / `expected_section` but no `expected_page`, so page-number
  *correctness* is not scoreable here and is deliberately not claimed. Page
  *population* is reported instead as a coverage diagnostic (are citations
  carrying page numbers at all), which is what the data supports.

- **Context precision and context recall are retrieval-only diagnostics, not
  success criteria.** Both are scored by the same judge call as faithfulness
  (one extra pair of fields, not a second API call): context precision judges
  which of the *retrieved* KB chunks were actually relevant to the question
  (rank-weighted, RAGAS-style average precision over the numbered excerpts);
  context recall judges whether the expected key facts are *supported by the
  retrieved excerpts themselves*, independent of whether the assistant's
  answer stated them -- it checks what retrieval handed the answerer, not
  what the answer did with it. Both are skipped for web-sourced rows
  (`expect_web_source`), since there's no ranked KB retrieval list to score.

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
    python evals/run_answer_eval.py --no-turn-cache  # force fresh agent turns, ignore run_turn() cache

Every run_turn() call is disk-cached to evals/results/turn_cache/ (same
SHA-256 pattern as the judge cache, keyed on question + answer model +
retriever mode + top_k). Re-scoring the same rows -- a different judge model,
a rubric tweak, --no-judge -- reuses the cached agent turn and costs zero
agent API calls; only the judge call (if any) re-runs. Bump RUN_CACHE_VERSION
when AgentResponse's shape changes or an upstream change (prompts, retriever,
guardrails) should invalidate previously cached turns.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from src import config
from src.agent.orchestrator import (
    API_ERROR_REPLY,
    OUT_OF_SCOPE_REPLY,
    AgentResponse,
    AgentStep,
    run_turn,
)
from src.agent.router import needs_clarification
from src.rag.answerer import IDK_ANSWER
from src.rag.retriever import RetrievedChunk
from src.schemas import Citation, IntentClassification, TokenUsage, WebCitation

GOLDEN_SET = Path(__file__).parent / "golden_set_with_ground_truth.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
JUDGE_CACHE_DIR = RESULTS_DIR / "judge_cache"
TURN_CACHE_DIR = RESULTS_DIR / "turn_cache"
CLARIFY_CACHE_DIR = RESULTS_DIR / "clarify_cache"

# Bump when the rubric or prompt changes -- old cached verdicts were scored
# against a different rubric and must not be reused.
RUBRIC_VERSION = "v4"

# Bump when AgentResponse's shape changes or when a change upstream (prompts,
# retriever, guardrails) should invalidate previously cached turns.
RUN_CACHE_VERSION = "v1"

# Deterministic pass thresholds. The judge scores; these decide.
FAITHFULNESS_PASS = 4      # 1-5 scale, >=4 means core claims are grounded
ANSWER_CORRECTNESS_PASS = 4  # 1-5 scale, >=4 means the answer matches the ground truth

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
    answer_correctness: int = Field(
        ge=1, le=5,
        description=(
            "Does the assistant's answer match the reference (ground-truth) answer in "
            "substance? 5=matches in substance -- every material fact (numbers, entities, "
            "conditions) is correct; differences limited to phrasing, formatting, or an "
            "immaterial extra/omitted detail that wouldn't change what the reader does or "
            "believes still count as a 5; 4=matches but a fact the reader would actually "
            "need is missing or off; 3=partially matches, a notable gap or imprecision; "
            "2=largely different from the reference; 1=contradicts the reference or "
            "answers a different question"
        ),
    )
    relevant_chunk_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "The [N] numbers of retrieved excerpts (see the numbered list above) that "
            "are actually relevant to answering the question -- judge this independent "
            "of whether the assistant's answer used them. An excerpt about the wrong "
            "faculty class/rank/section, or an unrelated topic, is NOT relevant even if "
            "it superficially shares wording with the question."
        ),
    )
    key_facts_in_context: list[str] = Field(
        default_factory=list,
        description=(
            "Verbatim copies of the expected key facts that are stated or directly "
            "supported by the retrieved excerpts THEMSELVES -- not by the assistant's "
            "answer. This measures whether retrieval surfaced what was needed."
        ),
    )


JUDGE_PROMPT = """You are a strict evaluator for a university faculty onboarding assistant.

You are given a question, the source excerpts the assistant retrieved, the expected \
key facts a correct answer must contain, and the assistant's actual answer.

The retrieved excerpts below are numbered [1], [2], .... Those numbers are what \
relevant_chunk_numbers refers to.

Score four things:

1. FAITHFULNESS (1-5) -- is every claim in the answer supported by the excerpts?
   5 = every claim is directly supported by the excerpts
   4 = minor unsupported detail, but the core answer is grounded
   3 = some claims lack support in the excerpts
   2 = major claims are unsupported
   1 = the answer contradicts or ignores the excerpts
   Judge only against the excerpts provided. An answer can be factually true in the \
real world and still score low here if the excerpts do not support it.

2. ANSWER CORRECTNESS (1-5) -- does the assistant's answer match the reference \
answer below in substance (numbers, entities, conditions), regardless of phrasing \
or length?
   5 = matches in substance -- every material fact (numbers, entities, conditions) is \
correct; differences limited to phrasing, formatting, or an immaterial extra/omitted \
detail that wouldn't change what the reader does or believes still count as a 5 -- \
do not withhold a 5 just because the wording differs from the reference
   4 = matches, but a fact the reader would actually need is missing or off
   3 = partially matches, a notable gap or imprecision
   2 = largely different from the reference
   1 = contradicts the reference or answers a different question
   Judge this against the reference answer, not the excerpts -- a fluent answer \
that drifts from the reference on a MATERIAL number or condition scores low here even \
if it was faithful to the excerpts.

3. CONTEXT RELEVANCE -- of the numbered excerpts, which are actually relevant to \
answering the question? List their numbers in relevant_chunk_numbers. Judge \
relevance on its own terms, not on whether the assistant happened to use the \
excerpt.

4. CONTEXT RECALL -- for each expected key fact, decide whether the retrieved \
excerpts themselves (not the assistant's answer) state it or directly support it. \
Copy the matching expected key facts verbatim into key_facts_in_context. A fact \
counts only if an excerpt actually contains it -- not if the assistant's answer \
happens to state it from outside knowledge.

Do not reward length, confidence, or polish. A short answer that states the key \
facts and stays grounded scores higher than a long one that does not.

Question: {question}

Retrieved excerpts:
{context}

Expected key facts:
{key_facts}

Reference (ground-truth) answer:
{ground_truth}

Assistant's answer:
{answer}

Think step by step in `reasoning`, then output the scores."""


# --- Judge with disk cache --------------------------------------------------

def _cache_key(question: str, answer: str, context: str, key_facts: list[str],
                ground_truth: str, model: str) -> str:
    payload = json.dumps(
        {"v": RUBRIC_VERSION, "q": question, "a": answer,
         "c": context, "k": key_facts, "g": ground_truth, "m": model},
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
    ground_truth: str, model: str, client=None,
) -> tuple[AnswerVerdict | None, bool]:
    """Returns (verdict, was_cached). Verdict is None on any backend error --
    fail-open, same as src/guardrails/llm_judge.py: one flaky call must not
    take down a 30-row eval run."""
    key = _cache_key(question, answer, context, key_facts, ground_truth, model)
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
        ground_truth=ground_truth or "(no reference answer provided)",
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


# --- Simulated user: auto-generated clarifying reply -------------------------
#
# Any row, regardless of tier, can hit the agent's audience-segment
# clarification (see src/rag/answerer.py::retrieve_kb). The golden set does not
# script a reply for this -- the harness generates one itself, the same way a
# human tester would just answer the question, so every row can be scored on
# its final answer rather than on whether it happened to trip the clarifier.

_CLARIFY_OPTIONS_RE = re.compile(r"which applies to you:\s*(.+?)\??\s*$", re.IGNORECASE)


def _parse_first_clarify_option(clarifying_question: str) -> str | None:
    """Deterministic (no LLM) parse of the audience-segment clarification
    template (src/rag/answerer.py::_clarification_question): "The answer
    depends on your {noun}. Which applies to you: {options}?" -- returns the
    first listed option as the simulated reply. This is the common case (the
    only clarification the fixed router/retrieve_kb should still raise), and
    picking the first option is reproducible run to run -- config.AUDIENCE_ORDER
    always lists "Full-time Academic Faculty" first when it's among the
    options, which is what the golden set's ground_truth for these rows
    assumes. Returns None when the question doesn't match this shape (e.g. a
    router-level ambiguity with no explicit option list) -- caller falls back
    to an LLM-generated reply."""
    m = _CLARIFY_OPTIONS_RE.search(clarifying_question.strip())
    if not m:
        return None
    options_text = re.sub(r"\bor\b", ",", m.group(1), flags=re.IGNORECASE)
    options = [o.strip(" .") for o in options_text.split(",") if o.strip(" .")]
    return options[0] if options else None


def _clarify_cache_key(clarifying_question: str, question: str) -> str:
    payload = json.dumps(
        {"v": RUN_CACHE_VERSION, "cq": clarifying_question, "q": question},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generate_clarify_reply_llm(clarifying_question: str, question: str, client=None) -> str:
    """LLM fallback for a clarifying question that isn't the audience-segment
    template (see _parse_first_clarify_option) -- e.g. a router-level "could
    you be more specific?". One short, cached, deterministic (temperature=0)
    call that role-plays a real user answering the assistant's own question,
    so the harness never needs a golden-set-authored reply."""
    key = _clarify_cache_key(clarifying_question, question)
    path = CLARIFY_CACHE_DIR / f"{key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()

    from google.genai import types

    client = client or config.get_llm_client()
    prompt = (
        "You are role-playing a university faculty member using an onboarding chatbot. "
        f"You originally asked: \"{question}\"\n"
        f"The assistant replied with this clarifying question: \"{clarifying_question}\"\n"
        "Write a short, natural, one-sentence reply that answers the assistant's question "
        "and lets it proceed -- pick whichever specific detail (faculty class, document, "
        "timeframe, etc.) the assistant is asking for. Output ONLY the reply text, "
        "nothing else."
    )
    try:
        response = client.models.generate_content(
            model=config.ACTIVE_CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, thinking_config=config.thinking_config()),
        )
        reply = (response.text or "").strip()
    except Exception as exc:  # fail open -- a flaky call must not take down the row
        print(f"    [clarify-reply error] {type(exc).__name__}: {exc}")
        return "Full-time"

    if not reply:
        reply = "Full-time"
    try:
        CLARIFY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(reply, encoding="utf-8")
    except Exception:
        pass  # a cache write failure must never fail the eval run
    return reply


def _auto_generate_clarify_reply(clarifying_question: str, question: str, client=None) -> str:
    """Simulated-user reply to whatever clarifying question the agent asked.
    Deterministic parse first (free, reproducible), LLM fallback second."""
    parsed = _parse_first_clarify_option(clarifying_question)
    return parsed if parsed is not None else _generate_clarify_reply_llm(
        clarifying_question, question, client=client
    )


# --- run_turn() with disk cache ---------------------------------------------
#
# The agent turn (ReAct loop + guardrails + grounded answer) is the expensive
# part of this harness -- an order of magnitude more calls than the judge.
# Re-running score_row() with a different judge model, rubric tweak, or
# --no-judge should not re-run the agent, so the turn itself is cached
# separately from the judge verdict. Same disk-cache pattern as judge_answer():
# SHA-256 key, fail-open, a version const to invalidate.

def _turn_cache_key(message: str, history: list[dict[str, str]] | None = None) -> str:
    payload = json.dumps(
        {
            "v": RUN_CACHE_VERSION,
            "q": message,
            "history": history or [],
            "model": config.ACTIVE_CHAT_MODEL,
            "retriever_mode": config.RETRIEVER_MODE,
            "top_k": config.TOP_K,
        },
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _response_to_dict(response: AgentResponse) -> dict:
    return {
        "reply": response.reply,
        "citations": [c.model_dump() for c in response.citations],
        "web_citations": [c.model_dump() for c in response.web_citations],
        "chunks": [asdict(c) for c in response.chunks],
        "insufficient_context": response.insufficient_context,
        "token_usage": response.token_usage.model_dump(),
        "steps": [asdict(s) for s in response.steps],
        "actions": response.actions,
    }


def _response_from_dict(d: dict) -> AgentResponse:
    return AgentResponse(
        reply=d["reply"],
        citations=[Citation(**c) for c in d["citations"]],
        web_citations=[WebCitation(**c) for c in d["web_citations"]],
        chunks=[RetrievedChunk(**c) for c in d["chunks"]],
        insufficient_context=d["insufficient_context"],
        token_usage=TokenUsage(**d["token_usage"]),
        steps=[AgentStep(**s) for s in d["steps"]],
        actions=d.get("actions", []),
    )


def _load_cached_turn(key: str) -> tuple[AgentResponse, float] | None:
    path = TURN_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _response_from_dict(data["response"]), data["latency_ms"]
    except Exception:  # corrupt entry or schema drift -- treat as a miss, never crash
        return None


def _save_turn(key: str, response: AgentResponse, latency_ms: float) -> None:
    try:
        TURN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"response": _response_to_dict(response), "latency_ms": latency_ms}
        (TURN_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass  # a cache write failure must never fail the eval run


def run_turn_cached(
    message: str, history: list[dict[str, str]] | None, use_cache: bool
) -> tuple[AgentResponse, float, bool]:
    """Returns (response, latency_ms, was_cached). Infrastructure errors are
    never cached -- a stored timeout would silently "succeed" on every
    subsequent run instead of being retried. `history` is part of the cache
    key so a plain first turn and an auto-clarify follow-up (same question
    text, prior turns attached) never collide."""
    key = _turn_cache_key(message, history)
    if use_cache:
        cached = _load_cached_turn(key)
        if cached is not None:
            return cached[0], cached[1], True

    started = time.perf_counter()
    response = run_turn(session_id=f"eval-{uuid.uuid4().hex[:8]}", message=message, history=history)
    latency_ms = (time.perf_counter() - started) * 1000
    if use_cache and not _is_infrastructure_error(response):
        _save_turn(key, response, latency_ms)
    return response, latency_ms, False


def run_turn_with_auto_clarify(
    item: dict, use_cache: bool, client=None
) -> tuple[AgentResponse, float, bool, bool]:
    """Runs the row's question, and if the agent asks a clarifying question --
    ANY row, any tier, no golden-set scripting required -- generates a
    plausible reply (_auto_generate_clarify_reply), sends it as a second turn
    (full history threaded through, exactly like a real follow-up), and
    returns THAT response instead. See the module docstring's "No row is
    scored on 'did it ask a clarifying question'" note.

    Returns (response, total_latency_ms, turn_cached, clarified_then_answered).
    turn_cached is True only when every call involved was served from cache."""
    response, latency_ms, cached = run_turn_cached(item["question"], None, use_cache)
    if not _did_clarify(response):
        return response, latency_ms, cached, False

    auto_reply = _auto_generate_clarify_reply(response.reply, item["question"], client=client)
    history = [
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": response.reply},
    ]
    followup, followup_latency, followup_cached = run_turn_cached(
        auto_reply, history, use_cache
    )
    return followup, latency_ms + followup_latency, cached and followup_cached, True


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

    Rows flagged expect_web_source (evals/golden_set.jsonl) are answered by
    tools.search_web, not search_kb: the KB chunks/expected_doc_id path doesn't
    apply -- search_web always returns citations=[] and puts its evidence in
    web_citations (src/agent/tools.py::search_web). A citation is "ok" there
    simply if the turn actually carries a web citation, i.e. it didn't fall
    back to the no-web-answer decline.
    """
    if item.get("expect_web_source"):
        return bool(response.web_citations)

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
    """KB chunks are the evidence for a normal answer; a web-sourced answer
    (expect_web_source rows) has no chunks -- run_state.chunks is cleared in
    _synthesize_final since it's grounded in web_citations instead (see
    src/agent/orchestrator.py). Include both so the judge always sees whatever
    evidence the turn actually used.

    KB chunks are numbered [1], [2], ... in retrieval order (the order
    _react_loop accumulates them in -- first occurrence across hops, itself
    score-ordered within a hop). That numbering is what the judge's
    relevant_chunk_numbers refers to and what _context_precision() below scores
    against, so context precision is implicitly rank-aware. Web citations are
    appended unnumbered -- context precision/recall are retriever-only metrics
    and don't apply to a web-sourced answer (see score_row)."""
    parts = [f"[{i}] ({c.section_path}) {c.text}" for i, c in enumerate(response.chunks, start=1)]
    parts += [f"[web: {c.title}] {c.snippet}" for c in response.web_citations]
    return "\n\n".join(parts)[:max_chars]


def _context_precision(relevant_numbers: list[int], n_chunks: int) -> float | None:
    """RAGAS-style context precision: average of precision@k taken at each rank
    the judge marked relevant, over the retrieved KB chunks (numbered 1..n_chunks
    in retrieval order -- see _context_text). Rewards relevant chunks surfacing
    early, not just being present somewhere in the retrieved set (unlike a plain
    hit-rate). None when there were no KB chunks to score (web-sourced row)."""
    if n_chunks <= 0:
        return None
    relevant = {n for n in relevant_numbers if 1 <= n <= n_chunks}
    if not relevant:
        return 0.0
    precisions_at_k = [
        sum(1 for r in relevant if r <= k) / k
        for k in sorted(relevant)
    ]
    return round(sum(precisions_at_k) / len(relevant), 3)


# --- Per-row scoring --------------------------------------------------------

def score_row(item: dict, response, latency_ms: float, use_judge: bool,
              judge_model: str, client=None, clarified_then_answered: bool = False) -> dict:
    tier = item.get("difficulty", "?")
    abstained = _did_abstain(response)
    clarified = _did_clarify(response)
    citation_ok = _citation_ok(response, item)

    record = {
        "id": item["id"],
        "topic": item.get("topic", "?"),
        "difficulty": tier,
        "question": item["question"],
        "reply": response.reply,
        "reply_preview": response.reply[:300],
        "citations_text": "; ".join(
            f"{c.title} > {c.section_path} (p.{c.page or '?'})" for c in response.citations
        ),
        "web_citations_text": "; ".join(
            f"{c.title} ({c.url})" for c in response.web_citations
        ),
        "abstained": abstained,
        # Whether THIS (possibly post-auto-clarify) response still asks a
        # question -- e.g. a double clarification, which is a real failure.
        "clarified": clarified,
        # Whether the row went through the auto-clarify follow-up at all --
        # a diagnostic, not gated into success (see run_turn_with_auto_clarify).
        "clarified_then_answered": clarified_then_answered,
        "citation_ok": citation_ok,
        "n_citations": len(response.citations),
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

    # Answerable tiers, scored on the final (post-auto-clarify, if any) answer:
    # must not abstain, must not still be asking (a double clarification is a
    # failure, not a pass), must cite correctly, and (when the judge is on)
    # must be faithful and match the reference answer.
    key_facts = item.get("answer_key_facts", [])
    record["expected_key_facts"] = "; ".join(key_facts)
    record["ground_truth"] = item.get("ground_truth", "")
    components = {"answered": not abstained and not clarified, "citation_ok": citation_ok}

    if use_judge and components["answered"]:
        verdict, was_cached = judge_answer(
            item["question"], response.reply, _context_text(response),
            key_facts, item.get("ground_truth", ""), judge_model, client=client,
        )
        if verdict is not None:
            record.update({
                "judged": True,
                "judge_cached": was_cached,
                "faithfulness": verdict.faithfulness,
                "answer_correctness": verdict.answer_correctness,
                "judge_reasoning": verdict.reasoning[:400],
            })
            components["faithful"] = verdict.faithfulness >= FAITHFULNESS_PASS
            if item.get("ground_truth"):
                components["correct"] = verdict.answer_correctness >= ANSWER_CORRECTNESS_PASS

            # Retrieval-only diagnostics -- not gated into success/components.
            # Skipped for web-sourced rows (no ranked KB retrieval to score).
            if response.chunks:
                record["context_precision"] = _context_precision(
                    verdict.relevant_chunk_numbers, len(response.chunks)
                )
                if key_facts:
                    record["context_recall"] = round(
                        len(verdict.key_facts_in_context) / len(key_facts), 3
                    )

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
    correctness_scored = [r for r in judged if "answer_correctness" in r]
    precision_scored = [r for r in judged if "context_precision" in r]
    recall_scored = [r for r in judged if "context_recall" in r]
    negatives = [r for r in rows if r["difficulty"] == "negative"]
    latencies = sorted(r["latency_ms"] for r in rows)
    _mean_correctness = (
        statistics.mean(r["answer_correctness"] for r in correctness_scored)
        if correctness_scored else None
    )

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
        "abstention_accuracy": round(_rate(negatives), 3) if negatives else None,
        # Diagnostic only, not gated into success: how often ANY row's first
        # turn hit the clarifier and had to be auto-answered (see
        # run_turn_with_auto_clarify) vs. answering directly.
        "auto_clarify_rate": round(_rate(rows, "clarified_then_answered"), 3) if rows else None,
        "mean_faithfulness": round(
            statistics.mean(r["faithfulness"] for r in judged), 2) if judged else None,
        "mean_answer_correctness": round(_mean_correctness, 2) if _mean_correctness is not None else None,
        "mean_context_precision": round(
            statistics.mean(r["context_precision"] for r in precision_scored), 3
        ) if precision_scored else None,
        "mean_context_recall": round(
            statistics.mean(r["context_recall"] for r in recall_scored), 3
        ) if recall_scored else None,
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
    if summary["auto_clarify_rate"] is not None:
        lines.append(f"  rows that hit the clarifier and were auto-answered: {summary['auto_clarify_rate']:.3f} "
                     f"(diagnostic only -- both paths are scored identically on the final answer, "
                     f"see module docstring)")
    lines.append(f"  citation validity (answerable rows):     {summary['citation_validity_rate']:.3f}")

    if summary["n_judged"]:
        lines.append(f"\nLLM-as-judge (n={summary['n_judged']}):")
        lines.append(f"  mean faithfulness (1-5):    {summary['mean_faithfulness']:.2f} "
                     f"(pass bar >={FAITHFULNESS_PASS})")
        if summary["mean_answer_correctness"] is not None:
            lines.append(f"  mean answer correctness (1-5): {summary['mean_answer_correctness']:.2f} "
                         f"(pass bar >={ANSWER_CORRECTNESS_PASS}, vs. golden-set ground_truth)")
        if summary["mean_context_precision"] is not None:
            lines.append(f"  mean context precision:    {summary['mean_context_precision']:.3f} "
                         f"(retrieval-only diagnostic, rank-weighted, not gated into success)")
        if summary["mean_context_recall"] is not None:
            lines.append(f"  mean context recall:       {summary['mean_context_recall']:.3f} "
                         f"(retrieval-only diagnostic: key facts supported by retrieved context, not gated into success)")

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


# --- Debug table -------------------------------------------------------------
#
# One row per golden-set item, every scored field a human needs to diagnose a
# failure without cross-referencing the JSON: the full agent answer, the
# reference answer, citations, per-component pass/fail, and the judge's own
# reasoning. Opens directly in Excel/Sheets for sorting and filtering by tier,
# topic, or which component failed -- the JSON report is complete but not
# skimmable, and the .log's failures section truncates the reply to 100 chars.

TABLE_COLUMNS = [
    "id", "topic", "difficulty", "success", "partial", "failed_on",
    "question", "reply", "ground_truth", "expected_key_facts",
    "citations_text", "web_citations_text",
    "abstained", "clarified", "clarified_then_answered", "citation_ok",
    "faithfulness", "answer_correctness",
    "context_precision", "context_recall",
    "judge_reasoning",
    "latency_ms", "total_tokens", "n_steps", "tool_sequence",
    "turn_cached", "judge_cached", "error",
]


def _table_row(r: dict) -> dict:
    components = r.get("components") or {}
    failed_on = ", ".join(k for k, v in components.items() if not v)
    out = {col: r.get(col, "") for col in TABLE_COLUMNS}
    out["failed_on"] = failed_on
    out["tool_sequence"] = " -> ".join(r.get("tool_sequence") or [])
    return out


def write_table(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(_table_row(r))


# --- Entry point ------------------------------------------------------------

def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end answer eval on the golden set")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N rows")
    parser.add_argument("--tier", default=None,
                        help="only run one tier (lookup|multihop|near_miss|negative)")
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
    parser.add_argument("--no-turn-cache", action="store_true",
                        help="skip the on-disk run_turn() cache: always re-run the agent "
                             "even for a question seen in a prior run (still writes fresh "
                             "cache entries unless combined with a bumped RUN_CACHE_VERSION)")
    parser.add_argument("--threads", type=int, default=1, metavar="N",
                        help="run N rows concurrently in a thread pool. Cuts wall-clock time "
                             "since each row is network-bound (waiting on Gemini), but "
                             "multiplies the request rate against the free-tier per-minute "
                             "quota -- pair with --sleep and/or --retry if you raise this")
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

    print_lock = threading.Lock()

    def _log(msg: str) -> None:
        with print_lock:
            print(msg)

    for i, item in enumerate(golden, start=1):
        if item["id"] in reused:
            print(f"[{i}/{len(golden)}] {item['id']} -- reused from prior run")

    def _process_row(i: int, item: dict, is_last: bool) -> dict:
        _log(f"[{i}/{len(golden)}] {item['id']} ({item.get('difficulty')}) {item['question'][:70]}")
        record = None
        for attempt in range(args.retry + 1):
            response, latency_ms, turn_cached, clarified_then_answered = run_turn_with_auto_clarify(
                item, use_cache=not args.no_turn_cache, client=client
            )
            record = score_row(item, response, latency_ms, use_judge, args.judge_model,
                               client=client, clarified_then_answered=clarified_then_answered)
            record["turn_cached"] = turn_cached
            # A cached turn already succeeded once (errors are never cached), so
            # retrying it would just re-read the same file -- stop immediately.
            if not record["error"] or turn_cached or attempt == args.retry:
                break
            backoff = 15 * (2 ** attempt)
            _log(f"      [{item['id']}] backend error -- retrying in {backoff}s "
                 f"(attempt {attempt + 2}/{args.retry + 1})")
            time.sleep(backoff)

        status = "ERROR (excluded)" if record["error"] else f"success={record['success']}"
        cache_flag = "  [turn cached]" if record.get("turn_cached") else ""
        _log(f"      -> [{item['id']}] {status}  {record['latency_ms']:.0f}ms  {record['total_tokens']} tok"
             + (f"  faithfulness={record['faithfulness']}" if record.get("judged") else "")
             + cache_flag)

        if args.sleep and not is_last and not record.get("turn_cached"):
            time.sleep(args.sleep)
        return record

    # (i, item) pairs still needing a real run, in golden-set order; --resume
    # rows are already satisfied above and skip this entirely.
    pending = [(i, item) for i, item in enumerate(golden, start=1) if item["id"] not in reused]

    results_by_id: dict[str, dict] = {}
    if args.threads > 1 and pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {
                pool.submit(_process_row, i, item, idx == len(pending) - 1): item["id"]
                for idx, (i, item) in enumerate(pending)
            }
            for future in concurrent.futures.as_completed(futures):
                results_by_id[futures[future]] = future.result()
    else:
        for idx, (i, item) in enumerate(pending):
            results_by_id[item["id"]] = _process_row(i, item, idx == len(pending) - 1)

    rows = [reused[item["id"]] if item["id"] in reused else results_by_id[item["id"]] for item in golden]

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
    table_path = RESULTS_DIR / f"answer_eval_{stamp}_table.csv"
    write_table(rows, table_path)
    print(f"\nSummary log written to {log_path}")
    print(f"Full JSON report written to {json_path}")
    print(f"Debug table written to {table_path}")

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
            for key in ("abstention_accuracy", "auto_clarify_rate",
                        "mean_faithfulness",
                        "mean_answer_correctness", "mean_context_precision",
                        "mean_context_recall"):
                if summary.get(key) is not None:
                    flat[key] = summary[key]
            for tier, m in summary["per_tier"].items():
                flat[f"tier_{tier}_success"] = m["task_success_rate"]
            mlflow.log_metrics(flat)
            mlflow.log_artifact(str(json_path))
        print("Logged to MLflow experiment 'answer-eval'.")


if __name__ == "__main__":
    main()