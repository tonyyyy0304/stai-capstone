"""PDF -> structured Markdown conversion (Stage 1 of the ingestion pipeline).

Plain-text extraction flattens PDFs, which destroys the heading hierarchy the
chunker and citations depend on. This module reconstructs structure from layout
cues via pdfplumber:

- Headings: lines set larger than the body font (or whole-line bold) become
  `#`/`##`/`###` by font-size tier.
- Tables: ruled tables are detected and rendered as Markdown tables, kept in
  reading order, so the "tables stay whole" chunking rule applies to PDFs too.
- Page chrome: lines repeated in the top/bottom margins across pages and bare
  page numbers are dropped.
- Paragraphs: wrapped lines are re-joined; vertical gaps mark paragraph breaks;
  end-of-line hyphenation is repaired.

Single-column documents only — good enough for policy documents; multi-column
layouts would need a proper layout model (e.g. docling).
"""

import re
from collections import Counter
from dataclasses import dataclass

# Layout heuristics (points, fractions of page height, characters).
MARGIN_FRACTION = 0.10        # top/bottom zone scanned for repeated page chrome
REPEAT_FRACTION = 0.6         # line must appear on this share of pages to be chrome
HEADING_SIZE_DELTA = 0.9      # points above body size to qualify as a heading
MAX_HEADING_CHARS = 90
MAX_HEADING_LEVEL = 3
PARAGRAPH_GAP_MIN = 4.0       # vertical gap (pt) that forces a paragraph break

PAGE_NUMBER_RE = re.compile(r"^(page\s+)?\d+(\s+of\s+\d+)?$", re.IGNORECASE)


@dataclass
class _Line:
    text: str
    top: float
    bottom: float
    size: float      # dominant font size, rounded to 0.5pt
    bold: bool
    page: int


def _make_line(raw: dict, page: int) -> _Line:
    chars = raw.get("chars", [])
    sizes = [c["size"] for c in chars if c.get("size")]
    size = round(2 * sum(sizes) / len(sizes)) / 2 if sizes else 0.0
    bold_count = sum("bold" in (c.get("fontname") or "").lower() for c in chars)
    return _Line(
        text=raw["text"].strip(),
        top=raw["top"],
        bottom=raw["bottom"],
        size=size,
        bold=bool(chars) and bold_count >= len(chars) / 2,
        page=page,
    )


def _inside(raw_line: dict, bbox: tuple) -> bool:
    _, top, _, bottom = bbox
    center = (raw_line["top"] + raw_line["bottom"]) / 2
    return top - 1 <= center <= bottom + 1


def _table_to_markdown(rows: list[list]) -> str:
    clean = [
        [(cell or "").replace("\n", " ").strip() for cell in row]
        for row in rows
        if any(cell and cell.strip() for cell in row)
    ]
    if not clean:
        return ""
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    header, *data = clean
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(row) + " |" for row in data]
    return "\n".join(lines)


def _in_margin(line: _Line, page_height: float) -> bool:
    zone = page_height * MARGIN_FRACTION
    return line.bottom <= zone or line.top >= page_height - zone


def _drop_page_chrome(pages: list[dict]) -> None:
    """Remove headers/footers repeated across pages, and bare page numbers."""
    counts: Counter[str] = Counter()
    for page in pages:
        for line in page["lines"]:
            if _in_margin(line, page["height"]):
                counts[line.text.casefold()] += 1
    threshold = max(2, REPEAT_FRACTION * len(pages))
    repeated = {text for text, n in counts.items() if n >= threshold}
    for page in pages:
        page["lines"] = [
            line
            for line in page["lines"]
            if not (
                _in_margin(line, page["height"])
                and (line.text.casefold() in repeated or PAGE_NUMBER_RE.match(line.text))
            )
        ]


def _body_font_size(pages: list[dict]) -> float:
    """The document's body size = most common line size, weighted by text length."""
    counts: Counter[float] = Counter()
    for page in pages:
        for line in page["lines"]:
            counts[line.size] += len(line.text)
    return counts.most_common(1)[0][0] if counts else 0.0


def _heading_levels(pages: list[dict], body_size: float) -> dict[float, int]:
    """Map each font size larger than the body to a heading level (biggest = #)."""
    sizes = sorted(
        {
            line.size
            for page in pages
            for line in page["lines"]
            if line.size >= body_size + HEADING_SIZE_DELTA
        },
        reverse=True,
    )
    return {size: min(rank + 1, MAX_HEADING_LEVEL) for rank, size in enumerate(sizes)}


def _heading_level(line: _Line, body_size: float, levels: dict[float, int]) -> int | None:
    if len(line.text) > MAX_HEADING_CHARS or line.text.endswith("."):
        return None
    if line.size in levels:
        return levels[line.size]
    if line.bold and abs(line.size - body_size) < HEADING_SIZE_DELTA:
        # Whole-line bold at body size: a run-in heading; put it below sized tiers.
        return min(max(levels.values(), default=1) + 1, MAX_HEADING_LEVEL)
    return None


def _render(pages: list[dict], body_size: float, levels: dict[float, int]) -> str:
    blocks: list[str] = []
    paragraph = ""
    prev: _Line | None = None

    def flush() -> None:
        nonlocal paragraph
        if paragraph.strip():
            blocks.append(paragraph.strip())
        paragraph = ""

    for page_index, page in enumerate(pages):
        items = [("line", line.top, line) for line in page["lines"]]
        items += [("table", top, markdown) for top, markdown in page["tables"] if markdown]
        for kind, _, item in sorted(items, key=lambda entry: entry[1]):
            if kind == "table":
                flush()
                blocks.append(item)
                prev = None
                continue
            line: _Line = item
            level = _heading_level(line, body_size, levels)
            if level is not None:
                flush()
                blocks.append(f"{'#' * level} {line.text}")
                prev = None
                continue
            gap_break = (
                prev is not None
                and line.top - prev.bottom > max(PARAGRAPH_GAP_MIN, 0.5 * line.size)
            )
            if prev is None and page_index > 0 or gap_break:
                flush()
            if paragraph.endswith("-") and line.text[:1].islower():
                paragraph = paragraph[:-1] + line.text  # repair hyphenated wrap
            else:
                paragraph = f"{paragraph} {line.text}".strip()
            prev = line
        prev = None  # page boundary
    flush()
    return "\n\n".join(blocks) + "\n"


def convert_pdf_to_markdown(path) -> str:
    """Convert a single-column PDF to Markdown with reconstructed headings/tables."""
    import pdfplumber

    pages: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages):
            tables = page.find_tables()
            table_items = [(t.bbox[1], _table_to_markdown(t.extract())) for t in tables]
            lines = []
            for raw in page.extract_text_lines(return_chars=True):
                if any(_inside(raw, t.bbox) for t in tables):
                    continue  # table text is rendered via the table, not as prose
                line = _make_line(raw, index)
                if line.text:
                    lines.append(line)
            pages.append({"height": page.height, "lines": lines, "tables": table_items})

    _drop_page_chrome(pages)
    body_size = _body_font_size(pages)
    levels = _heading_levels(pages, body_size)
    return _render(pages, body_size, levels)
