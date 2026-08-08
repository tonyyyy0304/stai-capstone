from pathlib import Path

import pytest

from src import config
from src.rag.answerer import answer_question, detect_stated_class, no_answer, verify_citations
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


def test_verify_citations_stamps_page_from_chunk():
    """The citation's page comes from chunk metadata, not the model — even if the
    model returned page=0 (or a wrong value), verify_citations overwrites it."""
    chunk = RetrievedChunk(
        chunk_id="faculty-manual-2021#050", text="dress code text", similarity=0.8,
        doc_id="faculty-manual-2021", title="Faculty Manual 2021",
        section_path="Attire and Grooming", category="faculty_manual", page_start=131,
    )
    answer = GroundedAnswer(
        answer="Dress neatly.",
        citations=[Citation(chunk_id="faculty-manual-2021#050", title="Faculty Manual 2021",
                            section_path="Attire and Grooming", page=0)],
    )
    verified = verify_citations(answer, [chunk])
    assert verified.citations[0].page == 131


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


# --- Faculty-class disambiguation (Phase 2) ---------------------------------

def _class_chunk(chunk_id, faculty_class, similarity=0.7):
    return RetrievedChunk(
        chunk_id=chunk_id, text="Faculty Manual 2021 > x\n\nleave text", similarity=similarity,
        doc_id="faculty-manual-2021", title="Faculty Manual 2021",
        section_path="Leaves", category="faculty_manual", faculty_class=faculty_class,
    )


class RaisingClient:
    """Any generate_content call fails — proves the disambiguation branch answers
    without invoking the LLM."""
    class _Models:
        def generate_content(self, *a, **k):
            raise AssertionError("LLM must not be called on the clarification path")
    def __init__(self):
        self.models = self._Models()


@pytest.mark.parametrize("msg,expected", [
    ("how many vacation days do I get?", None),
    ("I'm part-time, how many vacation days?", "part_time_academic"),
    ("as a full-time faculty, what leave do I get?", "full_time_academic"),
    ("ASF vacation leave?", "academic_service"),
    ("am I full-time or part-time eligible?", None),  # contradictory → None
])
def test_detect_stated_class(msg, expected):
    assert detect_stated_class(msg) == expected


def test_multi_class_evidence_triggers_clarification_without_llm():
    retriever = FakeRetriever([
        _class_chunk("m#1", "full_time_academic"),
        _class_chunk("m#2", "academic_service"),
    ])
    answer, chunks = answer_question(
        "how many vacation leave days?", retriever=retriever, client=RaisingClient()
    )
    assert answer.requires_clarification is True
    assert chunks == []
    assert "faculty class" in answer.clarifying_question.lower()


def test_second_class_below_top_n_does_not_clarify(monkeypatch):
    """A class-split only among lower-ranked chunks (lexical noise) must NOT
    trigger clarification — the strong (top-N) evidence is a single class."""
    monkeypatch.setattr(config, "DISAMBIG_TOP_N", 3)
    retriever = FakeRetriever([
        _class_chunk("m#1", "full_time_academic", similarity=0.70),
        _class_chunk("m#2", "", similarity=0.69),
        _class_chunk("m#3", "full_time_academic", similarity=0.68),
        _class_chunk("m#4", "academic_service", similarity=0.60),  # rank 4, below top-3
    ])
    grounded = GroundedAnswer(answer="answer.", source=AnswerSource.INTERNAL_KB)
    answer, _ = answer_question("some question", retriever=retriever, client=FakeShapeClient(grounded))
    assert answer.requires_clarification is False


def test_stated_class_does_not_clarify_and_reranks_to_top():
    retriever = FakeRetriever([
        _class_chunk("m#ft", "full_time_academic", similarity=0.70),
        _class_chunk("m#asf", "academic_service", similarity=0.72),
    ])
    grounded = GroundedAnswer(answer="one term.", source=AnswerSource.INTERNAL_KB)
    answer, chunks = answer_question(
        "as an ASF, how much vacation leave?", retriever=retriever, client=FakeShapeClient(grounded)
    )
    assert answer.requires_clarification is False
    assert chunks[0].faculty_class == "academic_service"  # boosted above the FT chunk


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
