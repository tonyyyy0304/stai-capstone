from pathlib import Path

from src import config
from src.rag.answerer import answer_question, no_answer, verify_citations
from src.rag.chunking import chunk_document
from src.rag.retriever import RetrievedChunk, apply_floor, rerank_by_category
from src.schemas import AnswerSource, Citation, GroundedAnswer


def make_chunk(chunk_id="faculty-manual-2021#001", similarity=0.8):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=(
            "Faculty Manual 2021 > Full-time Academic Faculty > Benefits > Leaves (p.42)\n\n"
            "Full-time academic faculty accrue 15 working days of vacation leave per year."
        ),
        similarity=similarity,
        doc_id="faculty-manual-2021",
        title="Faculty Manual 2021",
        section_path="Full-time Academic Faculty > Benefits > Leaves (p.42)",
        category="leave",
    )


def test_apply_floor_filters_below_threshold():
    chunks = [make_chunk(similarity=0.9), make_chunk("x#000", similarity=0.2)]
    kept = apply_floor(chunks, floor=0.5)
    assert [c.chunk_id for c in kept] == ["faculty-manual-2021#001"]


def _cat_chunk(chunk_id, similarity, category):
    return RetrievedChunk(
        chunk_id=chunk_id, text="t", similarity=similarity, doc_id="d",
        title="t", section_path="s", category=category,
    )


def test_rerank_by_category_lifts_match_without_dropping():
    # off-category chunk is slightly more similar, but the category match wins the
    # ordering — and neither chunk is dropped (soft signal, not a filter).
    chunks = [
        _cat_chunk("a", 0.80, "benefits"),
        _cat_chunk("b", 0.78, "leave"),
    ]
    ranked = rerank_by_category(chunks, "leave", boost=0.05)
    assert [c.chunk_id for c in ranked] == ["b", "a"]
    assert {c.chunk_id for c in ranked} == {"a", "b"}


def test_rerank_by_category_preserves_similarity_and_noop_without_category():
    chunks = [_cat_chunk("a", 0.80, "benefits"), _cat_chunk("b", 0.78, "leave")]
    # No category => untouched order; similarity values are never mutated.
    assert rerank_by_category(chunks, None) == chunks
    ranked = rerank_by_category(chunks, "leave", boost=0.05)
    assert next(c for c in ranked if c.chunk_id == "b").similarity == 0.78


def test_verify_citations_drops_hallucinated_ids():
    answer = GroundedAnswer(
        answer="15 days.",
        citations=[
            Citation(
                chunk_id="faculty-manual-2021#001",
                title="Faculty Manual 2021",
                section_path="Full-time Academic Faculty > Benefits > Leaves (p.42)",
            ),
            Citation(chunk_id="made-up#999", title="Fake", section_path="Nowhere"),
        ],
    )
    verified = verify_citations(answer, [make_chunk()])
    assert [c.chunk_id for c in verified.citations] == ["faculty-manual-2021#001"]
    assert verified.answer == "15 days."


def test_no_answer_offers_hr_routing():
    answer = no_answer()
    assert answer.insufficient_context is True
    assert answer.citations == []
    assert "HR" in answer.answer


class FakeShapeResponse:
    def __init__(self, parsed):
        self.parsed = parsed


class FakeShapeModels:
    def __init__(self, parsed):
        self._parsed = parsed

    def generate_content(self, model, contents, config):
        return FakeShapeResponse(self._parsed)


class FakeShapeClient:
    def __init__(self, parsed):
        self.models = FakeShapeModels(parsed)


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, question, top_k=None, category=None):
        return self._chunks


def test_answer_question_withholds_chunks_when_insufficient_context():
    """Chunks passed the similarity floor, but the model still couldn't answer
    from them — the UI shouldn't show them as if they were the evidence."""
    retriever = FakeRetriever([make_chunk(similarity=0.6)])
    insufficient = GroundedAnswer(
        answer="I don't know", source=AnswerSource.INTERNAL_KB, insufficient_context=True
    )
    client = FakeShapeClient(insufficient)

    answer, chunks = answer_question("obscure question", retriever=retriever, client=client)

    assert answer.insufficient_context is True
    assert chunks == []


def test_answer_question_returns_chunks_when_answer_is_grounded():
    retriever = FakeRetriever([make_chunk(similarity=0.8)])
    grounded = GroundedAnswer(answer="15 days.", source=AnswerSource.INTERNAL_KB)
    client = FakeShapeClient(grounded)

    answer, chunks = answer_question("vacation days", retriever=retriever, client=client)

    assert answer.insufficient_context is False
    assert len(chunks) == 1


def test_entire_raw_corpus_chunks_cleanly():
    """Every source doc in data/raw parses, chunks, and carries valid metadata.

    Post-pivot the corpus is the pre-boarding PDFs (each with a sibling
    <name>.meta.yaml), parsed through the same path scripts/ingest.py uses —
    not the retired synthetic Markdown set.
    """
    from scripts.ingest import SUPPORTED_SUFFIXES, add_source_file, normalize_text, parse_raw_file

    raw_files = sorted(p for p in config.RAW_DIR.glob("*") if p.suffix in SUPPORTED_SUFFIXES)
    assert raw_files, "no source documents in data/raw"
    for path in raw_files:
        markdown = add_source_file(normalize_text(parse_raw_file(path)), path.name)
        chunks = chunk_document(markdown)
        assert chunks, f"{path.name} produced no chunks"
        for chunk in chunks:
            assert chunk.category in config.CATEGORIES
            assert chunk.section_path
            assert chunk.text.startswith(chunk.title)
            assert chunk.token_count > 0
