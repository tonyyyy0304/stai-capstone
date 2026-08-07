"""Grounded answer generation with verified citations (Module 1: RAG).

Retrieved chunks go into the prompt with their section_path; Gemini returns a
GroundedAnswer via response_schema; cited chunk_ids are verified against the
retrieved set (grounding guardrail). If nothing passes the similarity floor,
we return the "I don't know" path without calling the model.

Member 2's `search_kb` tool should call answer_question().
"""

from src import config
from src.rag.retriever import RetrievedChunk, Retriever, apply_floor, get_retriever
from src.schemas import Citation, GroundedAnswer

IDK_ANSWER = (
    "I couldn't find this in the DLSU Faculty Manual or the onboarding documents I "
    "have, so I don't want to guess. You may want to consult the DLSU Faculty Manual "
    "directly or your college's HR resources for this one."
)

ANSWER_PROMPT = """You are the DLSU Faculty Onboarding Concierge. You answer a faculty \
member's onboarding and Faculty Manual questions using ONLY the excerpts below, which come \
from the DLSU Faculty Manual 2021 and its official onboarding companion documents.

Rules:
- Base every claim on the excerpts; never use outside knowledge or another university's \
policy. A confident wrong answer about someone's employment terms is worse than no answer.
- Cite every excerpt you used by its exact chunk_id, title, and section_path. When a \
section_path names a page or appendix (e.g. "p.24", "Appendix F"), keep it in your answer \
so the reader can check the source.
- Quote specific numbers, durations, deadlines, form names, and rank codes exactly as \
written (e.g. "15 working days", "BIR Form 1902", "Assistant Professor").
- Requirements often differ by faculty class — Full-time Academic Faculty, Part-time \
Academic Faculty, and Academic Service Faculty (ASF). If the excerpts give class-specific \
answers and the question does not say which class the reader is, do NOT guess: set \
insufficient_context to true and ask which faculty class they belong to.
- If the excerpts do not contain the answer, set insufficient_context to true and say you \
don't know rather than filling the gap from memory.

Excerpts:
{context}

Faculty member's question: {question}"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[chunk_id: {c.chunk_id} | title: {c.title} | section_path: {c.section_path}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def verify_citations(answer: GroundedAnswer, chunks: list[RetrievedChunk]) -> GroundedAnswer:
    """Grounding guardrail: drop any citation whose chunk_id was not retrieved."""
    retrieved_ids = {c.chunk_id for c in chunks}
    verified = [c for c in answer.citations if c.chunk_id in retrieved_ids]
    return answer.model_copy(update={"citations": verified})


def no_answer() -> GroundedAnswer:
    return GroundedAnswer(answer=IDK_ANSWER, citations=[], insufficient_context=True)


def generate_grounded_answer(
    question: str, chunks: list[RetrievedChunk], client=None
) -> GroundedAnswer:
    """One Gemini call with response_schema=GroundedAnswer over the given chunks."""
    from google.genai import types

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
        ),
    )
    answer: GroundedAnswer = response.parsed
    if answer is None:  # model returned unparseable output — fail closed
        return no_answer()
    answer = verify_citations(answer, chunks)
    if answer.insufficient_context and not answer.citations:
        return no_answer()
    return answer


def answer_question(
    question: str,
    category: str | None = None,
    retriever: Retriever | None = None,
    client=None,
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
    answer = generate_grounded_answer(question, chunks, client=client)
    if answer.insufficient_context:
        return answer, []
    return answer, chunks
