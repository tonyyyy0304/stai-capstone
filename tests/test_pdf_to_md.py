"""PDF -> Markdown conversion, exercised against the real corpus PDF
(data/raw/data-privacy-policy.pdf, generated with distinct font sizes, a ruled
table, and repeated page headers/footers)."""

import pytest

from scripts.ingest import parse_raw_file
from src import config
from src.rag.chunking import chunk_document
from src.rag.pdf_to_md import _table_to_markdown, convert_pdf_to_markdown

PDF_PATH = config.RAW_DIR / "data-privacy-policy.pdf"


@pytest.fixture(scope="module")
def markdown():
    assert PDF_PATH.exists(), "corpus PDF missing; regenerate per data/raw"
    return convert_pdf_to_markdown(PDF_PATH)


def test_headings_reconstructed_from_font_sizes(markdown):
    assert "# Data Privacy Policy" in markdown
    for section in ("Purpose and Scope", "Employee Rights", "Breach Reporting",
                    "Data Protection Officer"):
        assert f"## {section}" in markdown


def test_page_chrome_removed(markdown):
    assert "Internal Use Only" not in markdown  # repeated page header
    assert "Page 1" not in markdown             # footer page numbers
    assert "Page 2" not in markdown


def test_table_rendered_as_markdown(markdown):
    assert "| Data Category | Examples | Retention Period |" in markdown
    assert "| Payroll records | Payslips, tax withholding, loans | 10 years |" in markdown
    # table cell text must not leak into prose as duplicate plain lines
    assert markdown.count("Payroll records") == 1


def test_wrapped_paragraphs_rejoined(markdown):
    # this sentence wraps across lines (and the section spans the page break) in the PDF
    assert "applicant data on personal devices or personal cloud accounts" in markdown


def test_pdf_flows_through_pipeline_to_chunks():
    doc = parse_raw_file(PDF_PATH)  # stitches in the .meta.yaml sidecar
    chunks = chunk_document(doc)
    assert chunks
    assert all(c.doc_id == "data-privacy-policy" for c in chunks)
    assert all(c.category == "conduct" for c in chunks)
    paths = {c.section_path for c in chunks}
    assert any("Breach Reporting" in p for p in paths)
    # the retention table survives inside a chunk
    assert any("| Payroll records |" in c.text for c in chunks)


def test_table_to_markdown_handles_ragged_rows():
    rows = [["A", "B"], ["1", None, "extra"], [None, None]]
    out = _table_to_markdown(rows)
    assert out.splitlines()[0].startswith("| A | B |")  # padded to 3 columns
    assert "| 1 |  | extra |" in out
    assert len(out.splitlines()) == 3  # empty row dropped
