"""Build the supplementary pre-boarding corpus PDFs into data/raw/.

The Faculty Manual 2021 covers hiring *criteria* and a one-page hiring
procedure, but it does not describe the pre-boarding workflow a new hire
actually walks through, and it never mentions the national statutory documents
(verified: "NBI" appears zero times in the Manual). These documents fill that
gap so T1 questions have real, citable corpus text.

Output (each PDF gets a sibling <name>.meta.yaml, which scripts/ingest.py
requires for non-Markdown sources):

    data/raw/dlsu-faculty-preboarding-process.pdf
    data/raw/dlsu-faculty-preemployment-requirements.pdf
    data/raw/ph-statutory-preemployment.pdf

Layout is deliberately tuned for src/rag/pdf_to_md.py: single column, three
heading tiers set well above the body font size, ruled tables, and running
page chrome that the parser drops.

Usage:
    python scripts/build_preboarding_corpus.py
    python scripts/build_preboarding_corpus.py --out data/raw --only statutory
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from scripts.corpus_content import doc_process, doc_requirements, doc_statutory

# --- Typography ---------------------------------------------------------------
# Body 10.5pt; heading tiers at 17/13.5/11.5pt. pdf_to_md.py treats any line
# >= body + 0.9pt as a heading and maps the three largest sizes to # / ## / ###,
# so these three tiers reconstruct exactly. Body text is never fully bold —
# whole-line bold at body size would also register as a heading.
BODY_SIZE = 10.5
H1_SIZE, H2_SIZE, H3_SIZE = 17.0, 13.5, 11.5
INK = colors.HexColor("#111111")
RULE = colors.HexColor("#9aa0a6")
BAND = colors.HexColor("#eef1f5")

STYLES = {
    "h1": ParagraphStyle(
        "h1", fontName="Helvetica-Bold", fontSize=H1_SIZE, leading=H1_SIZE * 1.25,
        spaceBefore=16, spaceAfter=8, textColor=INK,
    ),
    "h2": ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=H2_SIZE, leading=H2_SIZE * 1.25,
        spaceBefore=13, spaceAfter=6, textColor=INK,
    ),
    "h3": ParagraphStyle(
        "h3", fontName="Helvetica-Bold", fontSize=H3_SIZE, leading=H3_SIZE * 1.3,
        spaceBefore=10, spaceAfter=4, textColor=INK,
    ),
    "p": ParagraphStyle(
        "p", fontName="Helvetica", fontSize=BODY_SIZE, leading=BODY_SIZE * 1.45,
        spaceAfter=6, alignment=TA_JUSTIFY, textColor=INK,
    ),
    "li": ParagraphStyle(
        "li", fontName="Helvetica", fontSize=BODY_SIZE, leading=BODY_SIZE * 1.4,
        spaceAfter=3, textColor=INK,
    ),
    "cell": ParagraphStyle(
        "cell", fontName="Helvetica", fontSize=9.0, leading=11.5, textColor=INK,
    ),
    "cellhead": ParagraphStyle(
        "cellhead", fontName="Helvetica-Bold", fontSize=9.0, leading=11.5, textColor=INK,
    ),
    "note": ParagraphStyle(
        "note", fontName="Helvetica-Oblique", fontSize=BODY_SIZE, leading=BODY_SIZE * 1.4,
        leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=8, textColor=INK,
    ),
    "cover_title": ParagraphStyle(
        "cover_title", fontName="Helvetica-Bold", fontSize=22, leading=27,
        spaceAfter=10, textColor=INK,
    ),
    "cover_sub": ParagraphStyle(
        "cover_sub", fontName="Helvetica", fontSize=12.5, leading=17,
        spaceAfter=18, textColor=INK,
    ),
}


class _Doc(BaseDocTemplate):
    """Single-column template with a repeated footer (dropped as page chrome)."""

    def __init__(self, path: str, running_head: str, **kw):
        super().__init__(path, pagesize=A4, **kw)
        self.running_head = running_head
        frame = Frame(
            22 * mm, 20 * mm,
            A4[0] - 44 * mm, A4[1] - 42 * mm,
            id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._chrome)])

    def _chrome(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(RULE)
        canvas.drawString(22 * mm, 12 * mm, self.running_head)
        canvas.drawRightString(A4[0] - 22 * mm, 12 * mm, f"{doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(22 * mm, 15.5 * mm, A4[0] - 22 * mm, 15.5 * mm)
        canvas.restoreState()


def _table(rows, widths=None):
    total = A4[0] - 44 * mm
    ncols = max(len(r) for r in rows)
    if widths:
        col_widths = [total * w / sum(widths) for w in widths]
    else:
        col_widths = [total / ncols] * ncols
    data = [
        [Paragraph(str(c), STYLES["cellhead" if i == 0 else "cell"]) for c in row]
        for i, row in enumerate(rows)
    ]
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [table, Spacer(1, 9)]


def _list(items, bullet):
    return [
        ListFlowable(
            [ListItem(Paragraph(t, STYLES["li"]), leftIndent=16) for t in items],
            bulletType=bullet, start="1" if bullet == "1" else None,
            leftIndent=14, bulletFontSize=BODY_SIZE, bulletFontName="Helvetica",
        ),
        Spacer(1, 6),
    ]


def render(blocks) -> list:
    """Turn the content DSL into reportlab flowables."""
    flow: list = []
    for kind, value in blocks:
        if kind in ("h1", "h2", "h3"):
            # Keep a heading with the block that follows it.
            flow.append(Paragraph(value, STYLES[kind]))
        elif kind == "p":
            flow.append(Paragraph(value, STYLES["p"]))
        elif kind == "note":
            flow.append(KeepTogether(Paragraph(value, STYLES["note"])))
        elif kind == "bullets":
            flow += _list(value, "bullet")
        elif kind == "numbers":
            flow += _list(value, "1")
        elif kind == "table":
            rows, widths = value if isinstance(value, tuple) else (value, None)
            flow += _table(rows, widths)
        elif kind == "cover_title":
            flow.append(Paragraph(value, STYLES["cover_title"]))
        elif kind == "cover_sub":
            flow.append(Paragraph(value, STYLES["cover_sub"]))
        elif kind == "space":
            flow.append(Spacer(1, value))
        elif kind == "pagebreak":
            flow.append(PageBreak())
        else:
            raise ValueError(f"Unknown block kind: {kind}")
    return flow


def build(module, out_dir: Path) -> Path:
    meta = module.DOC
    pdf_path = out_dir / f"{meta['doc_id']}.pdf"
    doc = _Doc(str(pdf_path), running_head=meta["running_head"], title=meta["title"],
               author=meta["author"], subject=meta["title"])
    doc.build(render(module.BLOCKS))

    sidecar = pdf_path.with_suffix(".meta.yaml")
    sidecar.write_text(
        "\n".join(
            f"{key}: {meta[key]}"
            for key in ("doc_id", "title", "category", "effective_date", "version")
        ) + "\n",
        encoding="utf-8",
    )
    return pdf_path


MODULES = {
    "process": doc_process,
    "requirements": doc_requirements,
    "statutory": doc_statutory,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--only", choices=sorted(MODULES), help="build a single document")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [MODULES[args.only]] if args.only else list(MODULES.values())
    for module in selected:
        path = build(module, out_dir)
        print(f"wrote {path} (+ {path.with_suffix('.meta.yaml').name})")


if __name__ == "__main__":
    main()
