"""Output grounding guardrail (Module 6: Guardrails). No LLM call — a pure
set-membership check, run at the AgentResponse boundary.

Most grounding work already happens upstream: SIMILARITY_FLOOR (config.py)
at retrieval time, and src.rag.answerer.verify_citations() right after the
model's structured GroundedAnswer response for the search_kb path. This
module is the last-line, hard-verify defense-in-depth check described in the
capstone brief — it doesn't trust that upstream code correctly, it re-checks
at the point the final AgentResponse is assembled, so a future change to
search_kb/search_web that forgets to verify citations still can't leak an
unverifiable claim past this boundary.
"""

from src.rag.retriever import RetrievedChunk
from src.schemas import Citation


def verify_response_citations(
    citations: list[Citation], chunks: list[RetrievedChunk]
) -> list[Citation]:
    """Strips any citation whose chunk_id wasn't actually retrieved this turn."""
    if not citations:
        return []
    retrieved_ids = {c.chunk_id for c in chunks}
    return [c for c in citations if c.chunk_id in retrieved_ids]


def check_grounding(
    citations: list[Citation],
    chunks: list[RetrievedChunk],
    insufficient_context: bool,
) -> tuple[list[Citation], bool]:
    """Returns (verified citations, corrected insufficient_context).

    Structural consistency, not just chunk-id verification: citations
    shouldn't exist alongside an insufficient-context answer, and if every
    citation turns out unverifiable the answer isn't actually grounded
    either way — both cases fall back to insufficient_context=True with no
    citations, matching the "strip the claim or fall back to I don't know"
    behavior the guardrail is meant to enforce.
    """
    if insufficient_context:
        return [], True

    verified = verify_response_citations(citations, chunks)
    if citations and not verified:
        return [], True
    return verified, insufficient_context
