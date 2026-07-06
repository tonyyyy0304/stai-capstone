from pathlib import Path

from src import config
from src.rag.answerer import no_answer, verify_citations
from src.rag.chunking import chunk_document
from src.rag.retriever import RetrievedChunk, apply_floor
from src.schemas import Citation, GroundedAnswer


def make_chunk(chunk_id="leave-policy#001", similarity=0.8):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="Leave Policy > Vacation Leave\n\nEmployees accrue 15 days.",
        similarity=similarity,
        doc_id="leave-policy",
        title="Leave Policy",
        section_path="Vacation Leave",
        category="leave",
    )


def test_apply_floor_filters_below_threshold():
    chunks = [make_chunk(similarity=0.9), make_chunk("x#000", similarity=0.2)]
    kept = apply_floor(chunks, floor=0.5)
    assert [c.chunk_id for c in kept] == ["leave-policy#001"]


def test_verify_citations_drops_hallucinated_ids():
    answer = GroundedAnswer(
        answer="15 days.",
        citations=[
            Citation(chunk_id="leave-policy#001", title="Leave Policy", section_path="Vacation Leave"),
            Citation(chunk_id="made-up#999", title="Fake", section_path="Nowhere"),
        ],
    )
    verified = verify_citations(answer, [make_chunk()])
    assert [c.chunk_id for c in verified.citations] == ["leave-policy#001"]
    assert verified.answer == "15 days."


def test_no_answer_offers_hr_routing():
    answer = no_answer()
    assert answer.insufficient_context is True
    assert answer.citations == []
    assert "HR" in answer.answer


def test_entire_raw_corpus_chunks_cleanly():
    """Every authored source doc parses, chunks, and carries valid metadata."""
    raw_files = sorted(config.RAW_DIR.glob("*.md"))
    assert len(raw_files) >= 8, "PLAN.md §3.1 calls for ~8-15 source documents"
    for path in raw_files:
        chunks = chunk_document(Path(path).read_text(encoding="utf-8"))
        assert chunks, f"{path.name} produced no chunks"
        for chunk in chunks:
            assert chunk.category in config.CATEGORIES
            assert chunk.section_path
            assert chunk.text.startswith(chunk.title)
            assert chunk.token_count > 0
