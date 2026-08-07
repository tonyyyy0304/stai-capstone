"""PDF -> Markdown conversion, exercised against a real corpus PDF
(data/raw/dlsu-faculty-preemployment-requirements.pdf, built by
scripts/build_preboarding_corpus.py with three heading tiers set above the body
font, ruled tables, and a running head/footer that embeds the page number)."""

import pytest

from scripts.ingest import parse_raw_file
from src import config
from src.rag.chunking import chunk_document
from src.rag.pdf_to_md import _table_to_markdown, convert_pdf_to_markdown

PDF_PATH = config.RAW_DIR / "dlsu-faculty-preemployment-requirements.pdf"


@pytest.fixture(scope="module")
def markdown():
    assert PDF_PATH.exists(), (
        "corpus PDF missing; run `python scripts/build_preboarding_corpus.py`"
    )
    return convert_pdf_to_markdown(PDF_PATH)


def test_headings_reconstructed_from_font_sizes(markdown):
    # H1 title, numbered H2 sections, and a nested H3 all reconstruct from the
    # three font-size tiers the corpus builder uses.
    assert "# DLSU Faculty Pre-employment Requirements Checklist" in markdown
    for section in ("1. Core Document Set -- All Faculty Classes",
                    "3. Requirements by Faculty Class",
                    "4. Document Format Rules"):
        assert f"## {section}" in markdown
    assert "### 3.2 Part-time Academic Faculty" in markdown


def test_page_chrome_removed(markdown):
    # The running head/footer carries the page number on the same extracted line
    # ("... -- 2025  3"), so it differs per page; it must still be dropped.
    assert "DLSU Faculty Pre-employment Requirements Checklist -- 2025" not in markdown
    # No stranded bare page-number lines either.
    assert not any(line.strip().isdigit() for line in markdown.splitlines())


def test_table_rendered_as_markdown(markdown):
    assert "| Document | Purpose | Format requirement |" in markdown
    assert (
        "| Diploma (highest relevant degree) | Confirms degree conferral. "
        "| Original or notarized true copy. |"
    ) in markdown
    # table cell text must not leak into prose as duplicate plain lines
    assert markdown.count("Confirms degree conferral.") == 1


def test_wrapped_paragraphs_rejoined(markdown):
    # this sentence wraps across several physical lines in the PDF
    assert (
        "format specifications HRMO uses to determine whether a submitted "
        "document is acceptable"
    ) in markdown


def test_pdf_flows_through_pipeline_to_chunks():
    doc = parse_raw_file(PDF_PATH)  # stitches in the .meta.yaml sidecar
    chunks = chunk_document(doc)
    assert chunks
    assert all(c.doc_id == "dlsu-faculty-preemployment-requirements" for c in chunks)
    assert all(c.category == "onboarding" for c in chunks)
    paths = {c.section_path for c in chunks}
    assert any("Requirements by Faculty Class" in p for p in paths)
    # the core document-set table survives inside a chunk
    assert any("| Document | Purpose | Format requirement |" in c.text for c in chunks)


def test_table_to_markdown_handles_ragged_rows():
    rows = [["A", "B"], ["1", None, "extra"], [None, None]]
    out = _table_to_markdown(rows)
    assert out.splitlines()[0].startswith("| A | B |")  # padded to 3 columns
    assert "| 1 |  | extra |" in out
    assert len(out.splitlines()) == 3  # empty row dropped
