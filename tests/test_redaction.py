import json
import os

import pytest

import pdfplumber
from pdfplumber.redaction import (
    RedactedCell,
    Redactor,
    coerce_redaction_rect,
    coerce_redaction_rects,
)
from pdfplumber.utils import (
    get_bbox_intersection_area,
    rects_to_union,
    union_area,
)

HERE = os.path.abspath(os.path.dirname(__file__))
ISSUE_140 = os.path.join(HERE, "pdfs/issue-140-example.pdf")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def build_table_pdf(path, rows=None):
    """A deterministic 400x300 PDF: a 3x3 grid whose cells read
    AA1/BB1/CC1, AA2/BB2/CC2, AA3/BB3/CC3 by default."""
    if rows is None:
        rows = [["AA1", "BB1", "CC1"], ["AA2", "BB2", "CC2"], ["AA3", "BB3", "CC3"]]
    objs = []

    def add(body):
        objs.append(body)
        return len(objs)

    lines = []
    for x in [50, 150, 250, 350]:
        lines.append(f"0.8 w {x} 50 m {x} 250 l S")
    for y in [50, 116.667, 183.333, 250]:
        lines.append(f"0.8 w 50 {y} m 350 {y} l S")
    for r, row in enumerate(rows):
        ybase = 220 - r * 66.667
        for c, txt in enumerate(row):
            if not txt:
                continue
            lines.append(f"BT /F1 11 Tf {60 + c * 100} {ybase} Td ({txt}) Tj ET")

    stream = "\n".join(lines).encode("latin-1")
    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 300] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
    )
    add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    with open(path, "wb") as f:
        f.write(bytes(out))


@pytest.fixture
def table_pdf(tmp_path):
    path = str(tmp_path / "table.pdf")
    build_table_pdf(path)
    with pdfplumber.open(path) as pdf:
        yield pdf


@pytest.fixture
def issue_140_pdf():
    with pdfplumber.open(ISSUE_140) as pdf:
        yield pdf


# ---------------------------------------------------------------------------
# Geometry: union of overlapping rectangles
# ---------------------------------------------------------------------------


def test_union_area_counts_overlap_once():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    # 100 + 100 - 25 overlap
    assert union_area([a, b]) == 175
    assert union_area([a, a, a]) == 100
    assert union_area([]) == 0


def test_rects_to_union_is_disjoint_and_area_preserving():
    rects = [(0, 0, 10, 10), (5, 5, 15, 15), (2, 2, 8, 8), (20, 0, 30, 5)]
    merged = rects_to_union(rects)
    # No two output rectangles may overlap.
    for i, r1 in enumerate(merged):
        for r2 in merged[i + 1 :]:
            assert get_bbox_intersection_area(r1, r2) == 0
    assert union_area(merged) == union_area(rects)


def test_rects_to_union_coalesces_adjacent_rects():
    assert rects_to_union([(0, 0, 5, 5), (5, 0, 10, 5)]) == [(0, 0, 10, 5)]
    assert rects_to_union([(0, 0, 10, 5), (0, 5, 10, 10)]) == [(0, 0, 10, 10)]


def test_coerce_redaction_rect_normalizes_and_flips():
    # Sort corners.
    assert coerce_redaction_rect((10, 20, 0, 5)) == (0.0, 5.0, 10.0, 20.0)
    # Top-left-origin dict.
    assert coerce_redaction_rect({"x0": 1, "top": 2, "x1": 3, "bottom": 4}) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )
    # PDF-native dict needs page height: y1=280,y0=260 on a 300pt page
    assert coerce_redaction_rect(
        {"x0": 0, "y0": 260, "x1": 10, "y1": 280}, page_height=300
    ) == (0.0, 20.0, 10.0, 40.0)
    # Degenerate rectangles are dropped.
    assert coerce_redaction_rect((0, 0, 0, 10)) is None
    assert coerce_redaction_rect((0, 0, 10, 0)) is None
    with pytest.raises(ValueError):
        coerce_redaction_rect({"x0": 0, "y0": 0, "x1": 1, "y1": 1})
    with pytest.raises(ValueError):
        coerce_redaction_rect((1, 2, 3))
    with pytest.raises(ValueError):
        coerce_redaction_rect(("a", "b", "c", "d"))


def test_coerce_redaction_rects_clips_to_page():
    page_bbox = (0, 0, 100, 100)
    assert coerce_redaction_rects([(-10, -10, 50, 50)], page_bbox=page_bbox) == [
        (0.0, 0.0, 50.0, 50.0)
    ]
    # Fully outside the page -> dropped.
    assert coerce_redaction_rects([(200, 200, 300, 300)], page_bbox=page_bbox) == []


# ---------------------------------------------------------------------------
# Per-glyph occlusion semantics
# ---------------------------------------------------------------------------


def test_occlusion_requires_positive_area():
    red = Redactor([])
    glyph = {"x0": 0, "top": 0, "x1": 10, "bottom": 10}
    # Edges merely touching: no shared area, so not occluded.
    red_touch = Redactor([(10, 0, 20, 10)])
    assert red_touch.occlusion(glyph).occluded is False
    assert red.occlusion(glyph).occluded is False


def test_occlusion_partial_vs_full():
    glyph = {"x0": 0, "top": 0, "x1": 10, "bottom": 10}
    partial = Redactor([(0, 0, 5, 10)]).occlusion(glyph)
    assert partial.partially_occluded and not partial.fully_occluded
    assert partial.occluded and partial.coverage == pytest.approx(0.5)

    full = Redactor([(-5, -5, 15, 15)]).occlusion(glyph)
    assert full.fully_occluded and not full.partially_occluded
    assert full.coverage == pytest.approx(1.0)


def test_occlusion_uses_union_not_sum_of_rects():
    # Two rectangles each cover the left third and right third; the middle
    # third is not covered. Summed overlap would falsely total >100%.
    glyph = {"x0": 0, "top": 0, "x1": 30, "bottom": 10}
    red = Redactor([(0, 0, 10, 10), (0, 0, 10, 10), (20, 0, 30, 10)])
    occ = red.occlusion(glyph)
    assert occ.covered_area == pytest.approx(20 * 10)
    assert occ.partially_occluded and not occ.fully_occluded


def test_fully_covered_glyph_is_withheld_when_split_across_rects():
    # The glyph is fully inside the *union* of two abutting rectangles.
    glyph = {"x0": 0, "top": 0, "x1": 10, "bottom": 10}
    red = Redactor([(0, 0, 5, 10), (5, 0, 10, 10)])
    assert red.occlusion(glyph).fully_occluded is True
    assert red.is_occluded(glyph) is True


def test_include_partial_false_keeps_partial_glyphs():
    glyph = {"x0": 0, "top": 0, "x1": 10, "bottom": 10}
    red = Redactor([(0, 0, 5, 10)], include_partial=False)
    assert red.is_occluded(glyph) is False
    full = Redactor([(-5, -5, 15, 15)], include_partial=False)
    assert full.is_occluded(glyph) is True


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def test_extract_text_redacted_removes_covered_glyphs(issue_140_pdf):
    page = issue_140_pdf.pages[0]
    original = page.extract_text()
    # Cover the whole UPC column.
    redacted = page.extract_text_redacted([(70, 30, 165, 115)])
    assert "0085648100305" not in redacted
    assert "[REDACTED]" in redacted
    # Untouched text is still present.
    assert "CENTRAL KMA" in redacted
    # The unredacted API is unaffected.
    assert page.extract_text() == original


def test_extract_text_redacted_does_not_leak_partial_glyph(issue_140_pdf):
    page = issue_140_pdf.pages[0]
    first = page.chars[0]
    # Marginally clip the glyph's right edge: it is partially occluded, so
    # none of its text may survive.
    rect = [(first["x0"], first["top"], first["x1"] - 0.5, first["bottom"])]
    redacted = page.extract_text_redacted(rect)
    assert first["text"] not in redacted.split("[REDACTED]")[0]


def test_consecutive_occluded_glyphs_collapse_to_one_placeholder():
    chars = [
        {"text": "A", "x0": 0, "top": 0, "x1": 10, "bottom": 10, "doctop": 0},
        {"text": "B", "x0": 10, "top": 0, "x1": 20, "bottom": 10, "doctop": 0},
        {"text": "C", "x0": 20, "top": 0, "x1": 30, "bottom": 10, "doctop": 0},
        {"text": "D", "x0": 30, "top": 0, "x1": 40, "bottom": 10, "doctop": 0},
    ]
    red = Redactor([(10, 0, 30, 10)], placeholder="###")
    out = red.redacted_chars(chars)
    texts = [c["text"] for c in out]
    assert texts == ["A", "###", "D"]
    placeholder = out[1]
    # The placeholder spans the whole removed run's original bounds.
    assert (placeholder["x0"], placeholder["x1"]) == (10, 30)
    assert placeholder["redacted_glyph_count"] == 2


def test_separated_runs_get_separate_placeholders():
    chars = [
        {"text": "A", "x0": 0, "top": 0, "x1": 10, "bottom": 10, "doctop": 0},
        {"text": "B", "x0": 10, "top": 0, "x1": 20, "bottom": 10, "doctop": 0},
        {"text": "C", "x0": 20, "top": 0, "x1": 30, "bottom": 10, "doctop": 0},
        {"text": "D", "x0": 30, "top": 0, "x1": 40, "bottom": 10, "doctop": 0},
    ]
    red = Redactor([(0, 0, 10, 10), (30, 0, 40, 10)])
    texts = [c["text"] for c in red.redacted_chars(chars)]
    assert texts == [red.placeholder, "B", "C", red.placeholder]


def test_placeholder_boundaries_preserve_gaps(issue_140_pdf):
    page = issue_140_pdf.pages[0]
    # Redact one interior word, leaving text on both sides.
    words = page.extract_words()
    target = words[1]
    rect = [(target["x0"], target["top"], target["x1"], target["bottom"])]
    redacted = page.extract_text_redacted(rect)
    assert words[0]["text"] in redacted
    assert "[REDACTED]" in redacted
    assert target["text"] not in redacted


def test_overlapping_redactions_do_not_double_count(issue_140_pdf):
    page = issue_140_pdf.pages[0]
    single = page.redaction_report([(80, 30, 200, 115)])
    overlapping = page.redaction_report(
        [(80, 30, 200, 115), (80, 30, 200, 115), (150, 30, 260, 115)]
    )
    # Glyph counts depend on the union, never on how many rectangles overlap.
    doubled = page.redaction_report([(80, 30, 200, 115)] * 5)
    assert doubled.glyphs.redacted == single.glyphs.redacted
    # And the union is exactly the larger rectangle.
    assert overlapping.positions == [(80.0, 30.0, 260.0, 115.0)]
    assert overlapping.union_area == pytest.approx(180 * 85)


def test_empty_and_out_of_page_redactions(issue_140_pdf):
    page = issue_140_pdf.pages[0]
    assert page.extract_text_redacted([]) == page.extract_text()
    assert page.extract_text_redacted([(9999, 9999, 10000, 10000)]) == (
        page.extract_text()
    )


def test_layout_and_kwargs_are_supported(issue_140_pdf):
    page = issue_140_pdf.pages[0]
    out = page.extract_text_redacted([(80, 30, 200, 115)], layout=True)
    assert "[REDACTED]" in out
    assert "\n" in out


# ---------------------------------------------------------------------------
# Redaction report: no source text
# ---------------------------------------------------------------------------


def test_report_contains_only_positions_and_counts(issue_140_pdf):
    page = issue_140_pdf.pages[0]
    report = page.redaction_report([(70, 30, 165, 115)])
    d = report.to_dict()
    assert d["page_number"] == 1
    assert d["positions"] == [[70.0, 30.0, 165.0, 115.0]]
    assert d["counts"]["redacted"] > 0
    assert d["counts"]["redacted"] == (
        d["counts"]["fully_occluded"] + d["counts"]["partially_occluded"]
    )
    # No redacted source text may appear anywhere in the report. Check the
    # words whose glyphs are actually occluded by this redaction.
    redactor = Redactor([(70, 30, 165, 115)], page_bbox=page.bbox)
    hidden_words = [w for w in page.extract_words() if redactor.is_occluded(w)]
    assert hidden_words
    blob = json.dumps(d)
    for word in hidden_words:
        assert word["text"] not in blob


def test_report_cell_positions_and_counts_are_text_free(table_pdf):
    page = table_pdf.pages[0]
    # Cover the top-left cell (AA1) entirely; it sits at roughly x 50-150.
    report = page.redaction_report([(55, 55, 145, 110)])
    d = report.to_dict()
    assert len(d["tables"]) == 1
    cells = d["tables"][0]["cells"]
    assert len(cells) == 1
    cell = cells[0]
    assert (cell["row_index"], cell["column_index"]) == (0, 0)
    assert cell["counts"]["redacted"] > 0
    blob = json.dumps(d)
    assert "AA1" not in blob


def test_multi_page_report(table_pdf):
    report = table_pdf.redaction_report({1: [(55, 55, 145, 110)]})
    assert len(report.pages) == 1
    assert report.to_dict()["counts"]["pages"] == 1


# ---------------------------------------------------------------------------
# Tables: grid preserved, cells marked
# ---------------------------------------------------------------------------


def test_table_grid_is_preserved(table_pdf):
    page = table_pdf.pages[0]
    plain = page.extract_tables()
    redacted = page.extract_tables_redacted([(55, 55, 145, 110)])
    assert len(plain) == len(redacted)
    for p_table, r_table in zip(plain, redacted):
        assert len(p_table) == len(r_table)
        for p_row, r_row in zip(p_table, r_table):
            assert len(p_row) == len(r_row)


def test_redacted_cell_never_looks_empty(table_pdf):
    page = table_pdf.pages[0]
    tables = page.extract_tables_redacted([(55, 55, 145, 110)])
    cell = tables[0][0][0]
    assert isinstance(cell, RedactedCell)
    assert isinstance(cell, str)
    assert cell != ""
    assert cell.redaction is not None
    assert cell.redaction.row_index == 0
    assert cell.redaction.column_index == 0
    # Untouched cells remain plain strings.
    assert type(tables[0][0][1]) is str
    assert tables[0][0][1] == "BB1"


def test_cell_empty_before_redaction_stays_unmarked(tmp_path):
    # Build a table whose middle cell (row 1, col 1) has no text.
    path = str(tmp_path / "table-empty.pdf")
    build_table_pdf(
        path, rows=[["AA1", "BB1", "CC1"], ["AA2", "", "CC2"], ["AA3", "BB3", "CC3"]]
    )
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        # Redact a different cell entirely.
        tables = page.extract_tables_redacted([(55, 55, 145, 110)])
        empty_cell = tables[0][1][1]
        # Emptiness is preserved exactly: plain "", not a marked cell.
        assert empty_cell == ""
        assert not isinstance(empty_cell, RedactedCell)


def test_partially_covered_cell_text_is_fully_withheld(table_pdf):
    page = table_pdf.pages[0]
    # Clip only the right part of "AA1" (the "1"), leaving "AA".
    cell = page.find_tables()[0].cells[0]
    rect = [(cell[0] + 10, cell[1], cell[0] + 12, cell[3])]
    tables = page.extract_tables_redacted(rect)
    value = tables[0][0][0]
    assert "AA1" not in value
    assert isinstance(value, RedactedCell)


def test_extract_table_redacted_returns_largest_table(table_pdf):
    page = table_pdf.pages[0]
    table = page.extract_table_redacted([(55, 55, 145, 110)])
    assert table is not None
    assert len(table) == 3
    assert all(len(row) == 3 for row in table)


def test_table_redaction_does_not_affect_plain_extract(table_pdf):
    page = table_pdf.pages[0]
    before = page.extract_tables()
    page.extract_tables_redacted([(0, 0, 400, 300)])
    assert page.extract_tables() == before


def test_original_api_unchanged(issue_140_pdf):
    page = issue_140_pdf.pages[0]
    text = page.extract_text()
    tables = page.extract_tables()
    page.extract_redacted([(0, 0, 800, 500)])
    assert page.extract_text() == text
    assert page.extract_tables() == tables


# ---------------------------------------------------------------------------
# One-pass extraction
# ---------------------------------------------------------------------------


def test_extract_redacted_returns_text_tables_report(table_pdf):
    page = table_pdf.pages[0]
    result = page.extract_redacted([(55, 55, 145, 110)])
    assert isinstance(result.text, str)
    assert "[REDACTED]" in result.text
    assert len(result.tables) == 1
    assert result.report.page_number == 1
    assert "AA1" not in json.dumps(result.report.to_dict())
    # Tables in the result also carry marked cells.
    assert isinstance(result.tables[0][0][0], RedactedCell)


def test_custom_placeholder(table_pdf):
    page = table_pdf.pages[0]
    out = page.extract_text_redacted([(55, 55, 145, 110)], placeholder="<X>")
    assert "<X>" in out
    assert "[REDACTED]" not in out


def test_placeholder_must_be_nonempty():
    with pytest.raises(ValueError):
        Redactor([], placeholder="")


def test_redactor_is_reusable_and_stateless(table_pdf):
    page = table_pdf.pages[0]
    red = Redactor([(55, 55, 145, 110)], page_bbox=page.bbox)
    first = red.extract_text(page.chars)
    second = red.extract_text(page.chars)
    assert first == second
    assert red.counts(page.chars).total == len(page.chars)
