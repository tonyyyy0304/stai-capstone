"""Query-time retrieval (PLAN.md §3.2, Stage 5).

Embeds the query with RETRIEVAL_QUERY, pulls top-k from Chroma by cosine
similarity, optionally filtered by category, and applies the similarity floor:
if nothing passes, the caller must answer "I don't know" — never from memory.
"""

from dataclasses import dataclass

import chromadb

from src import config
from src.rag.embeddings import GeminiEmbedder


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


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
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
            )
        )
    return chunks


def apply_floor(
    chunks: list[RetrievedChunk], floor: float = config.SIMILARITY_FLOOR
) -> list[RetrievedChunk]:
    return [c for c in chunks if c.similarity >= floor]


class Retriever:
    def __init__(self, embedder: GeminiEmbedder | None = None, collection=None):
        self.embedder = embedder or GeminiEmbedder()
        self.collection = collection if collection is not None else get_collection()

    def retrieve(
        self, query: str, top_k: int = config.TOP_K, category: str | None = None
    ) -> list[RetrievedChunk]:
        """Top-k chunks by similarity, most similar first. No floor applied —
        callers use apply_floor() so evals can sweep thresholds."""
        query_vector = self.embedder.embed_query(query)
        where = {"category": {"$eq": category}} if category else None
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _to_chunks(result)
