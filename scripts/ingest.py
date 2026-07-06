"""Ingestion pipeline: data/raw -> data/processed -> ChromaDB (PLAN.md §3).

The only writer to data/processed/ and the Chroma index. Idempotent: raw files
are hashed against data/index_manifest.json, so re-running on unchanged docs is
a no-op; changed docs are re-chunked and re-embedded; deleted docs have their
chunks removed from the index.

Usage:
    python scripts/ingest.py            # incremental
    python scripts/ingest.py --force    # rebuild everything
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.rag.chunking import Chunk, chunk_document
from src.rag.embeddings import GeminiEmbedder
from src.rag.retriever import get_collection

SUPPORTED_SUFFIXES = (".md", ".pdf", ".docx")


# --- Stage 1: parse & normalize ---

def parse_raw_file(path: Path) -> str:
    """Extract text from a raw document. Markdown passes through; PDF/DOCX are
    converted (their frontmatter must be provided as a sibling <name>.meta.yaml)."""
    if path.suffix == ".md":
        return path.read_text(encoding="utf-8")
    if path.suffix == ".pdf":
        from src.rag.pdf_to_md import convert_pdf_to_markdown

        text = convert_pdf_to_markdown(path)
    elif path.suffix == ".docx":
        import docx

        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    meta_path = path.with_suffix(".meta.yaml")
    if not meta_path.exists():
        raise ValueError(f"{path.name} needs a sibling {meta_path.name} with frontmatter fields")
    return f"---\n{meta_path.read_text(encoding='utf-8').strip()}\n---\n\n{text}"


def normalize_text(text: str) -> str:
    """Strip page artifacts, fix hyphenated line breaks, normalize unicode/whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # hyphenated line breaks
    text = re.sub(r"^\s*Page \d+( of \d+)?\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def add_source_file(markdown: str, source_name: str) -> str:
    """Record the raw filename in the processed doc's frontmatter."""
    end = markdown.index("\n---", 3)
    return markdown[:end] + f"\nsource_file: {source_name}" + markdown[end:]


# --- Idempotency: manifest + change planning ---

def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_source(path: Path) -> str:
    """Hash a raw document plus its .meta.yaml sidecar (if any), so editing
    either one triggers re-ingestion."""
    digest = hashlib.sha256(path.read_bytes())
    meta_path = path.with_suffix(".meta.yaml")
    if meta_path.exists():
        digest.update(meta_path.read_bytes())
    return digest.hexdigest()


def load_manifest(path: Path = config.MANIFEST_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}}


@dataclass
class ChangeSet:
    to_ingest: list[str] = field(default_factory=list)  # new or changed raw filenames
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)  # in manifest but no longer in data/raw


def plan_changes(current_hashes: dict[str, str], manifest: dict, force: bool = False) -> ChangeSet:
    """Pure diff between the raw corpus and the manifest — unit-tested directly."""
    plan = ChangeSet()
    recorded = manifest.get("files", {})
    for name, digest in sorted(current_hashes.items()):
        if not force and recorded.get(name, {}).get("sha256") == digest:
            plan.unchanged.append(name)
        else:
            plan.to_ingest.append(name)
    plan.deleted = sorted(set(recorded) - set(current_hashes))
    return plan


# --- Stage 4: embed & index ---

def index_chunks(collection, embedder: GeminiEmbedder, doc_id: str, chunks: list[Chunk]) -> None:
    """Replace a document's chunks in Chroma (delete old, upsert new)."""
    collection.delete(where={"doc_id": {"$eq": doc_id}})
    if not chunks:
        return
    vectors = embedder.embed_documents([c.text for c in chunks])
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[c.metadata() for c in chunks],
    )


def run_ingestion(force: bool = False) -> dict:
    raw_files = sorted(
        p for p in config.RAW_DIR.glob("*") if p.suffix in SUPPORTED_SUFFIXES
    )
    if not raw_files:
        raise SystemExit(f"No source documents found in {config.RAW_DIR}")

    manifest = load_manifest()
    current_hashes = {p.name: hash_source(p) for p in raw_files}
    plan = plan_changes(current_hashes, manifest, force=force)
    print(
        f"Ingestion plan: {len(plan.to_ingest)} to (re)ingest, "
        f"{len(plan.unchanged)} unchanged, {len(plan.deleted)} deleted"
    )

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    collection = get_collection()
    embedder = GeminiEmbedder()
    files_entry = dict(manifest.get("files", {}))

    for name in plan.deleted:
        doc_id = files_entry[name].get("doc_id", "")
        if doc_id:
            collection.delete(where={"doc_id": {"$eq": doc_id}})
        processed = config.PROCESSED_DIR / files_entry[name].get("processed_file", "")
        if processed.name and processed.exists():
            processed.unlink()
        del files_entry[name]
        print(f"  removed: {name}")

    total_new_chunks = 0
    for name in plan.to_ingest:
        raw_path = config.RAW_DIR / name
        markdown = normalize_text(parse_raw_file(raw_path))
        markdown = add_source_file(markdown, name)
        chunks = chunk_document(markdown)
        if not chunks:
            raise ValueError(f"{name} produced no chunks")
        doc_id = chunks[0].doc_id

        processed_name = f"{doc_id}.md"
        (config.PROCESSED_DIR / processed_name).write_text(markdown, encoding="utf-8")
        index_chunks(collection, embedder, doc_id, chunks)

        files_entry[name] = {
            "sha256": current_hashes[name],
            "doc_id": doc_id,
            "processed_file": processed_name,
            "chunk_count": len(chunks),
        }
        total_new_chunks += len(chunks)
        print(f"  ingested: {name} -> {len(chunks)} chunks (doc_id={doc_id})")

    new_manifest = {
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIM,
        "chunk_target_tokens": config.CHUNK_TARGET_TOKENS,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "total_chunks": sum(f["chunk_count"] for f in files_entry.values()),
        "files": files_entry,
    }
    config.MANIFEST_PATH.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    print(
        f"Done. Index has {new_manifest['total_chunks']} chunks across "
        f"{len(files_entry)} documents ({total_new_chunks} newly embedded)."
    )
    return new_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the HR knowledge base index")
    parser.add_argument("--force", action="store_true", help="re-ingest all documents")
    args = parser.parse_args()
    run_ingestion(force=args.force)
