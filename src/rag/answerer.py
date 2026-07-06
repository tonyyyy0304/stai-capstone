"""Grounded answer generation with verified citations (Module 1: RAG).

Retrieved chunks go into the prompt with their section_path; Gemini returns a
GroundedAnswer via response_schema; cited chunk_ids are verified against the
retrieved set (grounding guardrail). If nothing passes the similarity floor,
we return the "I don't know" path without calling the model.

Member 2's `search_kb` tool should call answer_question().
"""

from src import config
from src.rag.retriever import RetrievedChunk, Retriever, apply_floor
from src.schemas import Citation, GroundedAnswer

IDK_ANSWER = (
    "I couldn't find this in our HR policy documents, so I don't want to guess. "
    "I can route your question to the HR team instead — would you like that?"
)

ANSWER_PROMPT = """You are an HR policy assistant. Answer the employee's question using ONLY the policy excerpts below.

Rules:
- Base every claim on the excerpts; never use outside knowledge.
- Cite every excerpt you used by its exact chunk_id, title, and section_path.
- If the excerpts do not contain the answer, set insufficient_context to true and say you don't know.
- Be concise and direct; quote specific numbers, durations, and amounts exactly as written.

Policy excerpts:
{context}

Employee question: {question}"""


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
    client = client or config.get_gemini_client()
    response = client.models.generate_content(
        model=config.CHAT_MODEL,
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
    and MLflow traces).
    """
    retriever = retriever or Retriever()
    chunks = apply_floor(retriever.retrieve(question, category=category))
    if not chunks:
        return no_answer(), []
    return generate_grounded_answer(question, chunks, client=client), chunks
