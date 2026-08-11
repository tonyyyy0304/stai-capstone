"""Unit tests for the hybrid dense+BM25 RRF retriever (PLAN.md §3.4).

These are hermetic — no Gemini calls, no persistent Chroma. The RRF math and the
FTS5 BM25 index are exercised directly with fakes and a temp SQLite file.
"""

from src.rag.hybrid import (
    HybridRetriever,
    _fts_match_query,
    build_bm25_index,
    reciprocal_rank_fusion,
)
from src.rag.retriever import RetrievedChunk


# --- Pure RRF fusion ---------------------------------------------------------

def test_rrf_rewards_agreement_across_retrievers():
    """A chunk both retrievers rank highly beats one only a single list ranks #1."""
    dense = ["a", "b", "c"]
    bm25 = ["b", "a", "d"]
    scores = reciprocal_rank_fusion([dense, bm25], k=60)
    # 'b' is 2nd + 1st, 'a' is 1st + 2nd — both appear twice; 'c'/'d' once each.
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["d"]
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert set(ranked[:2]) == {"a", "b"}


def test_rrf_k_dampens_rank_weight():
    """Larger k flattens the gap between rank 1 and rank 2 contributions."""
    small_k = reciprocal_rank_fusion([["x", "y"]], k=1)
    large_k = reciprocal_rank_fusion([["x", "y"]], k=1000)
    assert (small_k["x"] - small_k["y"]) > (large_k["x"] - large_k["y"])


def test_rrf_empty_lists_yield_no_scores():
    assert reciprocal_rank_fusion([[], []]) == {}


# --- FTS5 query sanitization -------------------------------------------------

def test_fts_match_query_strips_syntax_characters():
    # Quotes/parens would be FTS5 syntax errors if passed raw.
    q = _fts_match_query('What is BIR Form 1902 (for "new" hires)?')
    assert q == '"What" OR "is" OR "BIR" OR "Form" OR "1902" OR "for" OR "new" OR "hires"'


def test_fts_match_query_empty_when_no_word_chars():
    assert _fts_match_query("?!...") == ""


# --- BM25 index build + ranking (real FTS5, no API) --------------------------

class FakeCollection:
    """Minimal stand-in for a Chroma collection for BM25 build/get."""

    def __init__(self, rows):
        # rows: list of (chunk_id, category, text, embedding)
        self._rows = rows

    def get(self, ids=None, include=None):
        rows = self._rows if ids is None else [r for r in self._rows if r[0] in ids]
        out = {"ids": [r[0] for r in rows], "documents": [r[2] for r in rows],
               "metadatas": [{"category": r[1], "doc_id": "d", "title": "t",
                              "section_path": "s"} for r in rows]}
        if include and "embeddings" in include:
            out["embeddings"] = [r[3] for r in rows]
        return out


class FakeEmbedder:
    def __init__(self, vector):
        self._vector = vector

    def embed_query(self, text):
        return self._vector

    def embed_documents(self, texts):
        return [self._vector for _ in texts]


def _rows():
    return [
        ("c1", "onboarding", "NBI Clearance validity is one year from date of issue", [1.0, 0.0]),
        ("c2", "onboarding", "BIR Form 1902 is filed by first-time employees", [0.0, 1.0]),
        ("c3", "leave", "Vacation leave accrues at fifteen days", [0.0, -1.0]),
    ]


def test_build_bm25_index_and_lexical_ranking(tmp_path):
    db = tmp_path / "bm25.sqlite"
    col = FakeCollection(_rows())
    n = build_bm25_index(collection=col, db_path=db)
    assert n == 3

    hr = HybridRetriever(embedder=FakeEmbedder([0.0, 1.0]), collection=col, db_path=db)
    ranked = hr._bm25_ranking("BIR Form 1902", top_k=5, category=None)
    assert ranked[0] == "c2"  # the only lexical match for the exact form number


def test_bm25_ranking_is_category_agnostic(tmp_path):
    """Category is a soft signal now (PLAN.md §4.1): BM25 must NOT hard-filter,
    so a lexical match in a different category still surfaces — otherwise the
    multi-topic Faculty Manual would be excluded by a mismatched query category."""
    db = tmp_path / "bm25.sqlite"
    col = FakeCollection(_rows())
    build_bm25_index(collection=col, db_path=db)
    hr = HybridRetriever(embedder=FakeEmbedder([0.0, 1.0]), collection=col, db_path=db)

    # "leave" text exists only in the c3 (category=leave) chunk; passing a
    # mismatched category must not filter it out.
    ranked = hr._bm25_ranking("vacation leave", top_k=5, category="onboarding")
    assert "c3" in ranked


def test_hybrid_retrieve_fuses_dense_and_bm25(tmp_path, monkeypatch):
    """End-to-end fusion with fakes: dense favors c1, BM25 favors c2; both surface."""
    db = tmp_path / "bm25.sqlite"
    col = FakeCollection(_rows())
    build_bm25_index(collection=col, db_path=db)

    # Fake the dense side: query vector [1,0] makes c1 the top cosine match.
    def fake_dense_retrieve(self, query, top_k=8, category=None):
        return [RetrievedChunk(chunk_id="c1", text=_rows()[0][2], similarity=0.99,
                               doc_id="d", title="t", section_path="s", category="onboarding")]

    monkeypatch.setattr("src.rag.retriever.Retriever.retrieve", fake_dense_retrieve)

    hr = HybridRetriever(embedder=FakeEmbedder([0.0, 1.0]), collection=col, db_path=db)
    results = hr.retrieve("BIR Form 1902 clearance", top_k=5)
    ids = {c.chunk_id for c in results}
    # c1 from dense, c2 from BM25 lexical match — fusion must include both.
    assert {"c1", "c2"} <= ids
    # Every returned chunk carries a real cosine similarity (not an RRF score).
    assert all(-1.0 <= c.similarity <= 1.0 for c in results)
