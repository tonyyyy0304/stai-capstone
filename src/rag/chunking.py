"""Structure-aware chunking (PLAN.md §3.2, Stages 2–3).

Splits processed Markdown on headings first (a chunk never spans two policy
sections), then packs paragraphs to ~CHUNK_TARGET_TOKENS with
~CHUNK_OVERLAP_TOKENS of overlap. Tiny sections are merged into their parent,
tables are kept whole, and every chunk gets a "{title} > {section path}"
context header before embedding.
"""

import re
from dataclasses import dataclass

import yaml

from src import config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars/token for English prose)."""
    return max(1, round(len(text) / 4))


def parse_frontmatter(markdown: str) -> tuple[dict, str]:
    """Split a processed document into (frontmatter dict, body)."""
    match = FRONTMATTER_RE.match(markdown)
    if not match:
        raise ValueError("Document is missing YAML frontmatter")
    meta = yaml.safe_load(match.group(1)) or {}
    for required in ("doc_id", "title", "category"):
        if required not in meta:
            raise ValueError(f"Frontmatter missing required field: {required}")
    if meta["category"] not in config.CATEGORIES:
        raise ValueError(f"Unknown category '{meta['category']}' in doc '{meta['doc_id']}'")
    return meta, markdown[match.end():]


@dataclass
class Section:
    path: list[str]  # heading stack below the document title, e.g. ["Sick Leave", "Documentation"]
    text: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    section_path: str
    text: str  # includes the context header; this is what gets embedded
    token_count: int
    effective_date: str = ""
    version: str = ""

    def metadata(self) -> dict:
        """Flat metadata for Chroma (str/int/float/bool values only)."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "category": self.category,
            "section_path": self.section_path,
            "effective_date": self.effective_date,
            "version": self.version,
            "token_count": self.token_count,
        }


def split_sections(body: str, doc_title: str) -> list[Section]:
    """Split the body on headings, tracking the heading stack as the section path.

    A single leading `#` heading equal to the document title is treated as the
    title line and excluded from paths; all other headings are path components.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading text)
    current = Section(path=[])

    def flush() -> None:
        if current.text.strip():
            current.text = current.text.strip()
            sections.append(current)

    def level_path() -> list[str]:
        return [h for _, h in stack]

    for line in body.splitlines():
        m = HEADING_RE.match(line)
        if not m:
            current.text += line + "\n"
            continue
        level, heading = len(m.group(1)), m.group(2).strip()
        if level == 1 and not stack and not sections and heading.lower() == doc_title.lower():
            continue  # document title line, not a section
        flush()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        current = Section(path=level_path())
    flush()
    return sections


def merge_tiny_sections(sections: list[Section], min_tokens: int) -> list[Section]:
    """Merge sections under min_tokens into their parent (or the previous section)."""
    merged: list[Section] = []
    for sec in sections:
        if merged and estimate_tokens(sec.text) < min_tokens:
            target = None
            for prev in reversed(merged):
                if sec.path[: len(prev.path)] == prev.path:  # prev is an ancestor
                    target = prev
                    break
            target = target or merged[-1]
            label = sec.path[-1] if sec.path else ""
            addition = f"**{label}:** {sec.text}" if label else sec.text
            target.text = f"{target.text}\n\n{addition}"
        else:
            merged.append(sec)
    return merged


def _blocks(text: str) -> list[str]:
    """Split into paragraph blocks; contiguous Markdown table lines stay one block."""
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            blocks.append(para)
    return blocks


def is_table(block: str) -> bool:
    lines = block.splitlines()
    return len(lines) >= 2 and all(line.lstrip().startswith("|") for line in lines)


def _split_oversized(block: str, target: int) -> list[str]:
    """Sentence-split a single non-table block that exceeds the target size."""
    sentences = re.split(r"(?<=[.!?])\s+", block)
    parts, buf = [], ""
    for sent in sentences:
        if buf and estimate_tokens(buf + " " + sent) > target:
            parts.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        parts.append(buf)
    return parts


def _pack_blocks(blocks: list[str], target: int, overlap: int) -> list[str]:
    """Greedily pack blocks into windows of ~target tokens with ~overlap carryover."""
    windows: list[list[str]] = []
    buf: list[str] = []
    buf_tokens = 0
    for block in blocks:
        block_tokens = estimate_tokens(block)
        if buf and buf_tokens + block_tokens > target:
            windows.append(buf)
            # Carry trailing blocks up to `overlap` tokens into the next window.
            carry: list[str] = []
            carry_tokens = 0
            for prev in reversed(buf):
                t = estimate_tokens(prev)
                if carry_tokens + t > overlap:
                    break
                carry.insert(0, prev)
                carry_tokens += t
            buf, buf_tokens = list(carry), carry_tokens
        buf.append(block)
        buf_tokens += block_tokens
    if buf:
        windows.append(buf)
    return ["\n\n".join(w) for w in windows]


def split_section_text(text: str, target: int, overlap: int) -> list[str]:
    """Split one section's text into chunk-sized pieces; tables are never split."""
    blocks: list[str] = []
    for block in _blocks(text):
        if not is_table(block) and estimate_tokens(block) > target:
            blocks.extend(_split_oversized(block, target))
        else:
            blocks.append(block)
    return _pack_blocks(blocks, target, overlap)


def chunk_document(markdown: str) -> list[Chunk]:
    """Full Stage 2–3: frontmatter → sections → merge tiny → split → context headers."""
    meta, body = parse_frontmatter(markdown)
    title = str(meta["title"])
    sections = split_sections(body, title)
    sections = merge_tiny_sections(sections, config.CHUNK_MIN_TOKENS)

    chunks: list[Chunk] = []
    for sec in sections:
        section_path = " > ".join(sec.path) if sec.path else "General"
        header = f"{title} > {section_path}"
        for piece in split_section_text(
            sec.text, config.CHUNK_TARGET_TOKENS, config.CHUNK_OVERLAP_TOKENS
        ):
            text = f"{header}\n\n{piece}"
            chunks.append(
                Chunk(
                    chunk_id=f"{meta['doc_id']}#{len(chunks):03d}",
                    doc_id=str(meta["doc_id"]),
                    title=title,
                    category=str(meta["category"]),
                    section_path=section_path,
                    text=text,
                    token_count=estimate_tokens(text),
                    effective_date=str(meta.get("effective_date", "")),
                    version=str(meta.get("version", "")),
                )
            )
    return chunks
