"""Query-time retrieval (PLAN.md §3.2, Stage 5).

Embeds the query with RETRIEVAL_QUERY, pulls top-k from Chroma by cosine
similarity, optionally filtered by category, and applies the similarity floor:
if nothing passes, the caller must answer "I don't know" — never from memory.
"""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from src import config
from src.rag.embeddings import Embedder


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    similarity: float
    doc_id: str
    title: str
    section_path: str
    category: str
    effective_date: str = ""
    version: str = ""
    audience_class: str = ""  # "" = class-agnostic (Phase 2)
    page_start: int = 0  # printed page number for citations (Phase 3); 0 = unknown


def get_collection():
    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _to_chunks(result: dict) -> list[RetrievedChunk]:
    chunks = []
    for chunk_id, doc, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=doc,
                similarity=1.0 - distance,  # cosine distance -> cosine similarity
                doc_id=meta.get("doc_id", ""),
                title=meta.get("title", ""),
                section_path=meta.get("section_path", ""),
                category=meta.get("category", ""),
                effective_date=meta.get("effective_date", ""),
                version=meta.get("version", ""),
                audience_class=meta.get("audience_class", ""),
                page_start=int(meta.get("page_start", 0) or 0),
            )
        )
    return chunks


def apply_floor(
    chunks: list[RetrievedChunk], floor: float = config.SIMILARITY_FLOOR
) -> list[RetrievedChunk]:
    return [c for c in chunks if c.similarity >= floor]


def rerank_by_category(
    chunks: list[RetrievedChunk],
    category: str | None,
    boost: float = config.CATEGORY_BOOST,
) -> list[RetrievedChunk]:
    """Stable re-rank that nudges category-matching chunks up by `boost` on the
    ordering score only. `similarity` on each chunk is left untouched (the floor
    must still see true cosine). With no category the input order is preserved,
    so plain similarity ordering is unchanged. Never drops a chunk — this is the
    soft replacement for the old hard `$eq` category filter (PLAN.md §4.1)."""
    if not category:
        return chunks
    return sorted(
        chunks,
        key=lambda c: c.similarity + (boost if c.category == category else 0.0),
        reverse=True,
    )


def rerank_by_audience(
    chunks: list[RetrievedChunk],
    audience_class: str | None,
    boost: float = config.AUDIENCE_CLASS_BOOST,
) -> list[RetrievedChunk]:
    """Soft re-rank toward the reader's stated faculty class (Phase 2). Same
    ordering-only mechanism as rerank_by_category — the stored similarity (and
    thus the floor) is untouched, and no chunk is dropped. Class-agnostic chunks
    (audience_class == "") are never boosted, so shared content (dress code, table
    of offenses) still competes on pure relevance."""
    if not audience_class:
        return chunks
    return sorted(
        chunks,
        key=lambda c: c.similarity + (boost if c.audience_class == audience_class else 0.0),
        reverse=True,
    )


def get_retriever(embedder: Embedder | None = None, collection=None):
    """Return the retriever selected by config.RETRIEVER_MODE.

    "dense" (default) -> Retriever; "hybrid" -> HybridRetriever (dense + BM25
    RRF, PLAN.md §3.4). Both expose the same .retrieve(query, top_k, category)
    surface, so callers (answerer.py, the eval harness) are agnostic. Hybrid is
    imported lazily to avoid a circular import (hybrid.py imports from here).
    """
    if config.RETRIEVER_MODE == "hybrid":
        from src.rag.hybrid import HybridRetriever

        return HybridRetriever(embedder=embedder, collection=collection)
    if config.RETRIEVER_MODE == "dense":
        return Retriever(embedder=embedder, collection=collection)
    raise RuntimeError(
        f"Unknown RETRIEVER_MODE={config.RETRIEVER_MODE!r}; expected 'dense' or 'hybrid'."
    )


class Retriever:
    def __init__(self, embedder: Embedder | None = None, collection=None):
        self.embedder = embedder or config.get_embedder()
        self.collection = collection if collection is not None else get_collection()

    def retrieve(
        self, query: str, top_k: int = config.TOP_K, category: str | None = None
    ) -> list[RetrievedChunk]:
        """Top-k chunks, most relevant first. No floor applied — callers use
        apply_floor() so evals can sweep thresholds.

        `category` is a *soft* signal: retrieval spans all categories (so the
        multi-topic Faculty Manual is always reachable), and a chunk whose
        category matches gets CATEGORY_BOOST added to its ordering score only.
        The stored `similarity` stays true cosine, so the floor is unaffected.
        With no category, this is plain top-k by similarity.
        """
        query_vector = self.embedder.embed_query(query)
        # Pull a wider pool than top_k so a relevant chunk that a category boost
        # would lift into the top_k isn't missed by an early cosine cutoff.
        pool = max(top_k, config.RRF_CANDIDATE_POOL)
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=pool,
            include=["documents", "metadatas", "distances"],
        )
        chunks = _to_chunks(result)
        return rerank_by_category(chunks, category)[:top_k]
