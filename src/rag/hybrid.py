"""Hybrid retrieval: dense (Chroma) + BM25 (SQLite FTS5) fused with RRF.

Advanced RAG component (PLAN.md §3.4). Dense cosine retrieval blurs the exact
identifiers this corpus is dense with — "BIR Form 1902", "Assistant Professor",
"p.24", "15 working days". A lexical BM25 index catches those, and Reciprocal
Rank Fusion combines the two rankings at **zero extra LLM calls at query time**:

    score(chunk) = Σ_r  1 / (RRF_K + rank_r(chunk))

where r ranges over the retrievers that returned the chunk and rank_r is its
1-based position in that retriever's list. `RRF_K` is in src/config.py.

The BM25 side lives in a small FTS5 table (data/bm25.sqlite) built from the same
chunks that are in Chroma; `build_bm25_index()` is called by scripts/ingest.py so
the two indexes never drift. HybridRetriever exposes the exact same
`.retrieve(query, top_k, category)` surface as Retriever, so answerer.py and the
eval harness need no changes — they select it via `get_retriever()`.

Each returned RetrievedChunk keeps its real dense cosine `similarity` (recomputed
locally from the stored embedding for chunks that only BM25 surfaced), so the
similarity floor / "I don't know" logic in answerer.py behaves identically to
dense mode. Ordering, however, follows the fused RRF score.
"""

import math
import re
import sqlite3

from src import config
from src.rag.embeddings import Embedder
from src.rag.retriever import RetrievedChunk, Retriever, get_collection

_FTS_TABLE = "chunks_fts"
# FTS5 MATCH treats many punctuation characters as syntax; we tokenize the query
# into bare word terms and OR them, so a raw user question can never be a syntax
# error and any lexical overlap contributes.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


# --- BM25 index construction (called from scripts/ingest.py) ---

def _connect(path=None) -> sqlite3.Connection:
    return sqlite3.connect(str(path or config.BM25_SQLITE_PATH))


def build_bm25_index(collection=None, db_path=None) -> int:
    """(Re)build the FTS5 BM25 table from every chunk currently in Chroma.

    Idempotent: drops and recreates the table so it always mirrors the vector
    index exactly. Returns the number of chunks indexed. Stores only what BM25
    needs — chunk_id, category (for filtered retrieval), and the searchable
    text; all other fields are read back from Chroma at query time.
    """
    collection = collection if collection is not None else get_collection()
    data = collection.get(include=["documents", "metadatas"])
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]

    conn = _connect(db_path)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {_FTS_TABLE}")
        conn.execute(
            f"CREATE VIRTUAL TABLE {_FTS_TABLE} USING fts5("
            "chunk_id UNINDEXED, category UNINDEXED, text)"
        )
        conn.executemany(
            f"INSERT INTO {_FTS_TABLE} (chunk_id, category, text) VALUES (?, ?, ?)",
            [(cid, (m or {}).get("category", ""), doc) for cid, m, doc in zip(ids, metas, docs)],
        )
        conn.commit()
    finally:
        conn.close()
    return len(ids)


def _fts_match_query(query: str) -> str:
    """Turn a free-text question into a safe FTS5 MATCH expression."""
    terms = _WORD_RE.findall(query)
    return " OR ".join(f'"{t}"' for t in terms)


# --- Fusion ---

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = config.RRF_K
) -> dict[str, float]:
    """Fuse several ranked id lists into one {chunk_id: rrf_score} map.

    Pure and dependency-free so it is unit-testable in isolation. `ranked_lists`
    is a list of rankings, each a list of chunk_ids ordered best-first.
    """
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class HybridRetriever:
    """Dense + BM25 retriever fused with RRF, drop-in for Retriever."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        collection=None,
        db_path=None,
        rrf_k: int = config.RRF_K,
    ):
        self.dense = Retriever(embedder=embedder, collection=collection)
        self.collection = self.dense.collection
        self.embedder = self.dense.embedder
        self.db_path = db_path or config.BM25_SQLITE_PATH
        self.rrf_k = rrf_k

    def _bm25_ranking(self, query: str, top_k: int, category: str | None) -> list[str]:
        match = _fts_match_query(query)
        if not match:
            return []
        conn = _connect(self.db_path)
        try:
            sql = f"SELECT chunk_id FROM {_FTS_TABLE} WHERE {_FTS_TABLE} MATCH ?"
            params: list = [match]
            if category:
                sql += " AND category = ?"
                params.append(category)
            sql += " ORDER BY bm25(%s) LIMIT ?" % _FTS_TABLE
            params.append(top_k)
            return [row[0] for row in conn.execute(sql, params).fetchall()]
        except sqlite3.OperationalError:
            # No BM25 table yet (index not built) — degrade to dense-only.
            return []
        finally:
            conn.close()

    def _hydrate(self, chunk_ids: list[str], query_vector: list[float]) -> dict[str, RetrievedChunk]:
        """Pull text/metadata/embeddings for a set of ids and build chunks with
        real cosine similarities against the query."""
        if not chunk_ids:
            return {}
        got = self.collection.get(
            ids=chunk_ids, include=["documents", "metadatas", "embeddings"]
        )
        out: dict[str, RetrievedChunk] = {}
        for cid, doc, meta, emb in zip(
            got["ids"], got["documents"], got["metadatas"], got["embeddings"]
        ):
            meta = meta or {}
            out[cid] = RetrievedChunk(
                chunk_id=cid,
                text=doc,
                similarity=_cosine(query_vector, list(emb)),
                doc_id=meta.get("doc_id", ""),
                title=meta.get("title", ""),
                section_path=meta.get("section_path", ""),
                category=meta.get("category", ""),
                effective_date=meta.get("effective_date", ""),
                version=meta.get("version", ""),
            )
        return out

    def retrieve(
        self, query: str, top_k: int = config.TOP_K, category: str | None = None
    ) -> list[RetrievedChunk]:
        """Top-k chunks ordered by fused RRF score. Same signature/semantics as
        Retriever.retrieve — no similarity floor applied here (callers use
        apply_floor), but each chunk carries its true dense cosine similarity."""
        pool = max(top_k, config.RRF_CANDIDATE_POOL)

        dense_chunks = self.dense.retrieve(query, top_k=pool, category=category)
        dense_ranking = [c.chunk_id for c in dense_chunks]
        query_vector = self.embedder.embed_query(query)

        bm25_ranking = self._bm25_ranking(query, top_k=pool, category=category)

        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=self.rrf_k)
        if not fused:
            return []

        # Reuse similarities we already have from the dense pass; only hydrate the
        # ids that BM25 surfaced but dense didn't (avoids a redundant Chroma read).
        chunks: dict[str, RetrievedChunk] = {c.chunk_id: c for c in dense_chunks}
        missing = [cid for cid in fused if cid not in chunks]
        chunks.update(self._hydrate(missing, query_vector))

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [chunks[cid] for cid, _ in ordered[:top_k] if cid in chunks]
