import pytest

from src import config
from src.rag.chunking import (
    Section,
    chunk_document,
    estimate_tokens,
    is_table,
    merge_tiny_sections,
    parse_frontmatter,
    split_section_text,
    split_sections,
)

SAMPLE_DOC = """---
doc_id: test-policy
title: Test Policy
category: leave
effective_date: 2026-01-01
version: v1
---

# Test Policy

## Vacation Leave

### Accrual

{accrual}

### Carry-Over

{carry}

## Sick Leave

{sick}

| Tenure | Days |
| --- | --- |
| 0-2 years | 15 |
| 3-5 years | 18 |
"""


def make_doc(accrual="Employees accrue leave monthly. " * 20,
             carry="Up to five days carry over each year. " * 20,
             sick="Sick leave requires notification before 10 AM. " * 20):
    return SAMPLE_DOC.format(accrual=accrual, carry=carry, sick=sick)


def test_parse_frontmatter():
    meta, body = parse_frontmatter(make_doc())
    assert meta["doc_id"] == "test-policy"
    assert meta["category"] == "leave"
    assert body.lstrip().startswith("# Test Policy")


def test_parse_frontmatter_rejects_missing_fields():
    with pytest.raises(ValueError, match="frontmatter"):
        parse_frontmatter("no frontmatter here")
    bad = "---\ndoc_id: x\ntitle: X\ncategory: not-a-category\n---\nbody"
    with pytest.raises(ValueError, match="category"):
        parse_frontmatter(bad)


def test_split_sections_tracks_heading_paths():
    _, body = parse_frontmatter(make_doc())
    sections = split_sections(body, "Test Policy")
    paths = [" > ".join(s.path) for s in sections]
    assert "Vacation Leave > Accrual" in paths
    assert "Vacation Leave > Carry-Over" in paths
    assert "Sick Leave" in paths
    # The title heading itself is not a section
    assert "Test Policy" not in paths


def test_chunk_never_spans_two_sections():
    chunks = chunk_document(make_doc())
    for chunk in chunks:
        # each chunk carries exactly one section_path
        assert chunk.section_path in {
            "Vacation Leave > Accrual",
            "Vacation Leave > Carry-Over",
            "Sick Leave",
        }


def test_context_header_prepended():
    chunks = chunk_document(make_doc())
    accrual = [c for c in chunks if c.section_path == "Vacation Leave > Accrual"]
    assert accrual
    for chunk in accrual:
        assert chunk.text.startswith("Test Policy > Vacation Leave > Accrual")


def test_chunk_ids_are_sequential_and_unique():
    chunks = chunk_document(make_doc())
    ids = [c.chunk_id for c in chunks]
    assert ids == [f"test-policy#{i:03d}" for i in range(len(chunks))]
    assert len(set(ids)) == len(ids)


def test_long_sections_split_with_overlap():
    long_text = "This is a filler sentence about leave policy details. " * 120
    chunks = chunk_document(make_doc(accrual=long_text))
    accrual = [c for c in chunks if c.section_path == "Vacation Leave > Accrual"]
    assert len(accrual) >= 2
    # generous ceiling: target + overlap + header slack
    for chunk in accrual:
        assert chunk.token_count <= config.CHUNK_TARGET_TOKENS + config.CHUNK_OVERLAP_TOKENS + 120


def test_tiny_sections_merged_into_parent():
    tiny = "Short note."
    sections = [
        Section(path=["Vacation Leave"], text="Parent section content. " * 30),
        Section(path=["Vacation Leave", "Tiny"], text=tiny),
    ]
    merged = merge_tiny_sections(sections, min_tokens=config.CHUNK_MIN_TOKENS)
    assert len(merged) == 1
    assert "Short note." in merged[0].text
    assert "**Tiny:**" in merged[0].text


def test_tables_kept_whole():
    table = "\n".join(["| Col A | Col B |", "| --- | --- |"] + [f"| row {i} | value {i} |" for i in range(80)])
    assert is_table(table)
    pieces = split_section_text(table, target=50, overlap=10)
    assert len(pieces) == 1  # never split, even though it exceeds the target


def test_estimate_tokens_positive():
    assert estimate_tokens("") == 1
    assert estimate_tokens("word " * 100) > 50


def test_metadata_is_chroma_flat():
    chunk = chunk_document(make_doc())[0]
    for value in chunk.metadata().values():
        assert isinstance(value, (str, int, float, bool))
