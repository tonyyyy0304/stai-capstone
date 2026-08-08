"""Grounded answer generation with verified citations (Module 1: RAG).

Retrieved chunks go into the prompt with their section_path; Gemini returns a
GroundedAnswer via response_schema; cited chunk_ids are verified against the
retrieved set (grounding guardrail). If nothing passes the similarity floor,
we return the "I don't know" path without calling the model.

Member 2's `search_kb` tool should call answer_question().
"""

import re

from src import config
from src.rag.retriever import (
    RetrievedChunk,
    Retriever,
    apply_floor,
    get_retriever,
    rerank_by_audience,
)
from src.schemas import Citation, GroundedAnswer


def detect_stated_audience(text: str) -> str | None:
    """Extract the reader's audience segment from their message, or None if
    unstated or contradictory (two segments mentioned → None). Config-driven
    (config.AUDIENCE_CLASSES keywords); query-side counterpart to the positional
    detection at ingest."""
    t = text.lower()
    hits = set()
    for segment in config.AUDIENCE_CLASSES:
        if any(re.search(rf"\b{re.escape(kw)}\b", t) for kw in segment["keywords"]):
            hits.add(segment["slug"])
    return hits.pop() if len(hits) == 1 else None


def _clarification_answer(present: set[str]) -> GroundedAnswer:
    """Build the segment-disambiguation response for evidence that spans >1 segment."""
    # Order by the configured order first, then any leftover slugs (defensive:
    # a slug in the index but not in current config still degrades gracefully).
    ordered = [s for s in config.AUDIENCE_ORDER if s in present]
    ordered += [s for s in present if s not in config.AUDIENCE_ORDER]
    labels = [config.AUDIENCE_LABELS.get(s, s) for s in ordered]
    options = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 1 else labels[0]
    question = (
        f"The answer depends on your {config.AUDIENCE_NOUN}. "
        f"Which applies to you: {options}?"
    )
    return GroundedAnswer(
        answer=question,
        citations=[],
        insufficient_context=False,
        requires_clarification=True,
        clarifying_question=question,
    )


IDK_ANSWER = (
    f"I couldn't find this in the {config.CORPUS_TITLE} or the companion documents I have, "
    f"so I don't want to guess. You may want to consult the {config.CORPUS_TITLE} directly "
    f"or {config.HELP_CONTACT} for this one."
)

# The segment-disambiguation rule is only included when the deployment defines
# audience segments (config.AUDIENCE_CLASSES); a corpus with no segmentation skips it.
_SEGMENT_RULE = (
    "\n- Requirements often differ by "
    f"{config.AUDIENCE_NOUN} ({', '.join(config.AUDIENCE_LABELS[s] for s in config.AUDIENCE_ORDER)}). "
    "If the excerpts give segment-specific answers and the question does not say which "
    "segment the reader is, do NOT guess: set insufficient_context to true and ask which "
    f"{config.AUDIENCE_NOUN} they belong to."
    if config.AUDIENCE_CLASSES
    else ""
)

ANSWER_PROMPT = f"""You are the {config.ASSISTANT_NAME}. You answer a {config.READER_NOUN}'s \
questions about {config.SCOPE_PHRASE} using ONLY the excerpts below.

Rules:
- Base every claim on the excerpts; never use outside knowledge or another organization's \
policy. A confident wrong answer about someone's employment terms is worse than no answer.
- Cite every excerpt you used by its exact chunk_id, title, and section_path. Each excerpt \
header includes a page number — state it in your answer (e.g. "p.131") so the reader can \
check the source. When a section_path names an appendix (e.g. "Appendix F"), keep that too.
- Quote specific numbers, durations, deadlines, form names, and codes exactly as written \
(e.g. "15 working days", "BIR Form 1902", "Assistant Professor").{_SEGMENT_RULE}
- Answer only the specific question asked. If the excerpts do not answer THAT question, set \
insufficient_context to true and say you don't know — even when the excerpts contain related or \
adjacent information. A partial, nearby, or "the documents only say X instead" fact is NOT an \
answer: in that case set insufficient_context to true (you may briefly note what the excerpts do \
cover). Never fill the gap from memory.

Excerpts:
{{context}}

{config.READER_NOUN.capitalize()}'s question: {{question}}"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        page = f" | page: {c.page_start}" if c.page_start else ""
        parts.append(
            f"[chunk_id: {c.chunk_id} | title: {c.title} | section_path: {c.section_path}{page}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def verify_citations(answer: GroundedAnswer, chunks: list[RetrievedChunk]) -> GroundedAnswer:
    """Grounding guardrail: drop any citation whose chunk_id was not retrieved,
    and stamp each surviving citation's page from the chunk metadata (deterministic
    — the page never comes from the model)."""
    by_id = {c.chunk_id: c for c in chunks}
    verified = [
        c.model_copy(update={"page": by_id[c.chunk_id].page_start})
        for c in answer.citations
        if c.chunk_id in by_id
    ]
    return answer.model_copy(update={"citations": verified})


def no_answer() -> GroundedAnswer:
    return GroundedAnswer(answer=IDK_ANSWER, citations=[], insufficient_context=True)


def generate_grounded_answer(
    question: str, chunks: list[RetrievedChunk], client=None, session_id: str | None = None
) -> GroundedAnswer:
    """One Gemini call with response_schema=GroundedAnswer over the given chunks."""
    from google.genai import types

    from src.agent import usage

    if not chunks:
        return no_answer()
    client = client or config.get_llm_client()
    response = client.models.generate_content(
        model=config.ACTIVE_CHAT_MODEL,
        contents=ANSWER_PROMPT.format(context=_format_context(chunks), question=question),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GroundedAnswer,
            temperature=0.2,
            thinking_config=config.thinking_config(),
        ),
    )
    # Account this call under the turn's session (previously unlogged, so per-turn
    # token usage undercounted the answer call).
    usage.record_usage(config.ACTIVE_CHAT_MODEL, usage.extract_usage(response), session_id=session_id)
    answer: GroundedAnswer = response.parsed
    if answer is None:  # model returned unparseable output — fail closed
        return no_answer()
    answer = verify_citations(answer, chunks)
    # requires_clarification is a code-set signal (the disambiguation branch in
    # answer_question), never the model's to raise — reset whatever it returned.
    answer = answer.model_copy(
        update={"requires_clarification": False, "clarifying_question": ""}
    )
    if answer.insufficient_context and not answer.citations:
        return no_answer()
    return answer


def answer_question(
    question: str,
    category: str | None = None,
    retriever: Retriever | None = None,
    client=None,
    session_id: str | None = None,
) -> tuple[GroundedAnswer, list[RetrievedChunk]]:
    """End-to-end RAG: retrieve → floor check → grounded answer.

    Returns the answer plus the chunks that passed the floor (for UI expanders
    and MLflow traces). Chunks are withheld when the answer is insufficient —
    otherwise the UI would show "possibly relevant" excerpts right next to an
    "I don't know" reply, which reads as contradictory even though those
    excerpts were exactly what the model just checked and rejected.
    """
    retriever = retriever or get_retriever()
    chunks = apply_floor(retriever.retrieve(question, category=category))
    if not chunks:
        return no_answer(), []

    # Audience-segment disambiguation (the corpus's biggest hazard): if the
    # evidence spans more than one segment and the reader hasn't said which they
    # are, ask instead of answering — a merged answer across segments is a
    # confident wrong answer about someone's terms. If they did state a segment,
    # softly re-rank toward it. Entirely gated on config.AUDIENCE_CLASSES, so a
    # deployment with no segmentation skips this cleanly (no clarifying question).
    if config.AUDIENCE_CLASSES:
        stated = detect_stated_audience(question)
        # Only the top-N chunks (the strong evidence) count toward the segment
        # split, so a lower-ranked lexical brush with a segment-specific section
        # doesn't trigger a bogus clarification on an unanswerable question.
        present = {
            c.audience_class for c in chunks[: config.DISAMBIG_TOP_N] if c.audience_class
        }
        if stated is None and len(present) >= 2:
            return _clarification_answer(present), []
        if stated:
            chunks = rerank_by_audience(chunks, stated)

    answer = generate_grounded_answer(question, chunks, client=client, session_id=session_id)
    if answer.insufficient_context:
        return answer, []
    return answer, chunks
