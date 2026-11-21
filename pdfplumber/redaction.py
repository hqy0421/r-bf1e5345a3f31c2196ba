"""
Helpers for redaction-aware ("sanitized") extraction.

A *redaction* is a rectangle, in the same top-left-origin coordinate
system used by ``Page.bbox`` (``x0``, ``top``, ``x1``, ``bottom``), that an
external review process has confirmed should never appear in exported
text. Redaction rectangles may overlap with one another.

Everything here is computed against the *union* of the redaction
rectangles, so overlapping rectangles never double-count an area. Two
kinds of occlusion are distinguished:

- **fully occluded**: the glyph's entire area lies within the union.
- **partially occluded**: the glyph overlaps the union over a *positive*
  area, but is not entirely inside it.

Either kind of occlusion removes the glyph from exported text: if any
part of a glyph is hidden, the whole glyph is withheld. Removed glyphs
that are consecutive in reading order are collapsed into one placeholder,
whose bounding box spans exactly the run's original extent, so the
remaining text keeps its original boundaries.

Nothing in this module modifies the underlying PDF or the page's ordinary
``objects``/``chars``; redacted extraction is opt-in.
"""

import itertools
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ._typing import T_bbox, T_num, T_obj, T_obj_list
from .utils import chars_to_textmap, extract_text, objects_to_bbox
from .utils.geometry import (
    calculate_area,
    get_union_intersection_area,
    rects_to_union,
)
from .utils.text import DEFAULT_X_TOLERANCE, DEFAULT_Y_TOLERANCE, WordExtractor

if TYPE_CHECKING:  # pragma: nocover
    from .table import Table

DEFAULT_PLACEHOLDER = "[REDACTED]"

# Occlusion is triggered by *any* positive covered area. This relative
# epsilon only guards against floating-point noise when deciding whether a
# glyph is *fully* covered; it never turns a positive overlap into none.
FULL_COVERAGE_EPSILON = 1e-9

T_rect_like = Union[T_bbox, T_obj, Sequence[T_num]]


def coerce_redaction_rect(
    rect: T_rect_like, page_height: Optional[T_num] = None
) -> Optional[T_bbox]:
    """
    Normalize a user-supplied redaction rectangle to
    ``(x0, top, x1, bottom)``, with corners sorted. Degenerate rectangles
    (zero width or height) return ``None``.

    Accepted forms:

    - a 4-item sequence of ``(x0, top, x1, bottom)``;
    - a dict with ``x0``/``top``/``x1``/``bottom`` (already in pdfplumber's
      top-left coordinates);
    - a dict with ``x0``/``y0``/``x1``/``y1`` (PDF-native bottom-left
      coordinates), in which case ``page_height`` is required to flip them.
    """
    if isinstance(rect, dict):
        if all(k in rect for k in ("x0", "top", "x1", "bottom")):
            x0, top, x1, bottom = rect["x0"], rect["top"], rect["x1"], rect["bottom"]
        elif all(k in rect for k in ("x0", "y0", "x1", "y1")):
            if page_height is None:
                raise ValueError(
                    "page_height is required to interpret rectangles given as"
                    " x0/y0/x1/y1 (PDF-native) coordinates."
                )
            x0, x1 = rect["x0"], rect["x1"]
            top = page_height - rect["y1"]
            bottom = page_height - rect["y0"]
        else:
            raise ValueError(f"Uninterpretable redaction rectangle: {rect!r}")
    else:
        try:
            x0, top, x1, bottom = rect
        except (TypeError, ValueError):
            raise ValueError(f"Uninterpretable redaction rectangle: {rect!r}")

    values = [x0, top, x1, bottom]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        raise ValueError(f"Redaction rectangle has non-numeric values: {rect!r}")

    x0, x1 = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(top), float(bottom)))
    if x1 <= x0 or bottom <= top:
        return None
    return (x0, top, x1, bottom)


def coerce_redaction_rects(
    rects: Iterable[T_rect_like],
    page_bbox: Optional[T_bbox] = None,
    page_height: Optional[T_num] = None,
) -> List[T_bbox]:
    """
    Normalize a collection of redaction rectangles, optionally clipping
    each to ``page_bbox``. Invalid or degenerate rectangles are skipped by
    this convenience wrapper; use :func:`coerce_redaction_rect` directly
    for strict validation.
    """
    out: List[T_bbox] = []
    for rect in rects:
        coerced = coerce_redaction_rect(rect, page_height=page_height)
        if coerced is None:
            continue
        if page_bbox is not None:
            clipped = (
                max(coerced[0], page_bbox[0]),
                max(coerced[1], page_bbox[1]),
                min(coerced[2], page_bbox[2]),
                min(coerced[3], page_bbox[3]),
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            coerced = clipped
        out.append(coerced)
    return out


@dataclass(frozen=True)
class GlyphOcclusion:
    """How a single glyph's bounding box is covered by the redaction union."""

    bbox: T_bbox
    glyph_area: T_num
    covered_area: T_num
    fully_occluded: bool
    partially_occluded: bool

    @property
    def occluded(self) -> bool:
        return self.fully_occluded or self.partially_occluded

    @property
    def coverage(self) -> T_num:
        return 0.0 if self.glyph_area <= 0 else self.covered_area / self.glyph_area


@dataclass(frozen=True)
class OcclusionCounts:
    """Counts of glyphs, by occlusion status. Contains no glyph text."""

    total: int = 0
    redacted: int = 0
    fully_occluded: int = 0
    partially_occluded: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "total": self.total,
            "redacted": self.redacted,
            "fully_occluded": self.fully_occluded,
            "partially_occluded": self.partially_occluded,
        }


class Redactor:
    """
    Applies a union of redaction rectangles to character objects.

    ``Redactor.occlusion(obj)`` reports how a glyph's bbox is covered;
    ``Redactor.redacted_chars(chars)`` returns the char objects to be fed
    into text extraction; ``Redactor.report(...)`` returns a
    position-and-count report that never includes redacted text.
    """

    def __init__(
        self,
        redactions: Iterable[T_rect_like],
        page_bbox: Optional[T_bbox] = None,
        page_height: Optional[T_num] = None,
        placeholder: str = DEFAULT_PLACEHOLDER,
        include_partial: bool = True,
    ) -> None:
        if not isinstance(placeholder, str) or not placeholder:
            raise ValueError("`placeholder` must be a non-empty string.")

        raw = list(redactions)
        self.raw_redaction_count = len(raw)
        self.page_bbox = page_bbox
        self.placeholder = placeholder
        self.include_partial = include_partial

        self.redactions = coerce_redaction_rects(
            raw, page_bbox=page_bbox, page_height=page_height
        )
        # Pairwise-disjoint rectangles covering exactly the same area as
        # the input rectangles. Every measurement below uses these, so
        # overlapping rectangles can never be counted twice.
        self.union = rects_to_union(self.redactions)

    @property
    def union_area(self) -> T_num:
        return sum(calculate_area(r) for r in self.union)

    def occlusion(self, obj_or_bbox: Union[T_obj, T_bbox]) -> GlyphOcclusion:
        if isinstance(obj_or_bbox, dict):
            bbox: T_bbox = (
                obj_or_bbox["x0"],
                obj_or_bbox["top"],
                obj_or_bbox["x1"],
                obj_or_bbox["bottom"],
            )
        else:
            bbox = (
                obj_or_bbox[0],
                obj_or_bbox[1],
                obj_or_bbox[2],
                obj_or_bbox[3],
            )
        glyph_area = calculate_area(bbox)
        covered = get_union_intersection_area(bbox, self.union)

        eps = FULL_COVERAGE_EPSILON * max(1.0, glyph_area)
        if covered <= eps:
            return GlyphOcclusion(bbox, glyph_area, 0.0, False, False)
        if glyph_area > 0 and covered >= glyph_area - eps:
            return GlyphOcclusion(bbox, glyph_area, covered, True, False)
        return GlyphOcclusion(bbox, glyph_area, covered, False, True)

    def is_occluded(self, obj: T_obj) -> bool:
        occ = self.occlusion(obj)
        return occ.fully_occluded or (self.include_partial and occ.partially_occluded)

    def visible_chars(self, chars: Iterable[T_obj]) -> T_obj_list:
        """The original char objects that are not occluded at all."""
        return [c for c in chars if not self.is_occluded(c)]

    def counts(self, chars: Iterable[T_obj]) -> OcclusionCounts:
        total = redacted = fully = partially = 0
        for char in chars:
            total += 1
            occ = self.occlusion(char)
            if occ.fully_occluded:
                fully += 1
                redacted += 1
            elif occ.partially_occluded:
                partially += 1
                if self.include_partial:
                    redacted += 1
        return OcclusionCounts(total, redacted, fully, partially)

    def _placeholder_char(self, run: T_obj_list) -> T_obj:
        base = dict(run[0])
        x0, top, x1, bottom = objects_to_bbox(run)
        counts = self.counts(run)
        base.update(
            {
                "text": self.placeholder,
                "x0": x0,
                "top": top,
                "x1": x1,
                "bottom": bottom,
                "width": x1 - x0,
                "height": bottom - top,
                "object_type": "char",
                "redaction_placeholder": True,
                "redacted_glyph_count": counts.redacted,
                "fully_occluded_glyph_count": counts.fully_occluded,
                "partially_occluded_glyph_count": counts.partially_occluded,
            }
        )
        if "doctop" in base:
            base["doctop"] = top + (base["doctop"] - run[0]["top"])
        return base

    def redacted_chars(
        self,
        chars: Iterable[T_obj],
        x_tolerance: T_num = DEFAULT_X_TOLERANCE,
        y_tolerance: T_num = DEFAULT_Y_TOLERANCE,
    ) -> T_obj_list:
        """
        Return char objects suitable for text extraction: all surviving
        glyphs plus a single placeholder per maximal run of consecutive
        occluded glyphs. The occluded glyphs' original text is discarded.
        """
        chars_list = list(chars)
        if not chars_list:
            return []

        # ``iter_chars_to_lines`` reads "upright" directly, so supply a
        # default for callers that pass minimal char-like dicts.
        chars_list = [
            (
                c
                if ("upright" in c and isinstance(c.get("text"), str) and c["text"])
                else {
                    **c,
                    "upright": c.get("upright", True),
                    "text": c.get("text") or " ",
                }
            )
            for c in chars_list
        ]

        occluded_ids = {id(c) for c in chars_list if self.is_occluded(c)}
        if not occluded_ids:
            return chars_list

        extractor = WordExtractor(
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
            keep_blank_chars=True,
        )

        order = {id(c): i for i, c in enumerate(chars_list)}
        keyed: List[Tuple[int, T_obj]] = []

        def flush(run: T_obj_list) -> None:
            if run:
                keyed.append(
                    (min(order[id(c)] for c in run), self._placeholder_char(run))
                )
                run.clear()

        # Group upright and rotated text separately. Within each group,
        # identify maximal runs of occluded glyphs in reading order.
        grouped = itertools.groupby(
            sorted(chars_list, key=lambda c: bool(c.get("upright", True))),
            key=lambda c: bool(c.get("upright", True)),
        )
        for _, group in grouped:
            for line_chars, _direction in extractor.iter_chars_to_lines(list(group)):
                run: T_obj_list = []
                for char in line_chars:
                    if id(char) in occluded_ids:
                        run.append(char)
                    else:
                        flush(run)
                        keyed.append((order[id(char)], char))
                flush(run)

        # Restore approximate document order; text-map construction
        # re-sorts by position unless use_text_flow=True.
        keyed.sort(key=lambda pair: pair[0])
        return [obj for _, obj in keyed]

    def extract_text(self, chars: Iterable[T_obj], **kwargs: Any) -> str:
        x_tolerance = kwargs.get("x_tolerance", DEFAULT_X_TOLERANCE)
        y_tolerance = kwargs.get("y_tolerance", DEFAULT_Y_TOLERANCE)
        synthetic = self.redacted_chars(
            chars, x_tolerance=x_tolerance, y_tolerance=y_tolerance
        )
        return extract_text(synthetic, **kwargs)

    def to_textmap(self, chars: Iterable[T_obj], **kwargs: Any) -> Any:
        x_tolerance = kwargs.get("x_tolerance", DEFAULT_X_TOLERANCE)
        y_tolerance = kwargs.get("y_tolerance", DEFAULT_Y_TOLERANCE)
        synthetic = self.redacted_chars(
            chars, x_tolerance=x_tolerance, y_tolerance=y_tolerance
        )
        return chars_to_textmap(synthetic, **kwargs)


@dataclass(frozen=True)
class TableCellRedaction:
    """Redaction metadata for one table cell. No cell text is included."""

    page_number: int
    row_index: int
    column_index: int
    bbox: T_bbox
    counts: OcclusionCounts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "row_index": self.row_index,
            "column_index": self.column_index,
            "position": list(self.bbox),
            "counts": self.counts.as_dict(),
        }


@dataclass(frozen=True)
class TableRedaction:
    """Redaction metadata for one table. No cell text is included."""

    page_number: int
    bbox: T_bbox
    cells: Tuple[TableCellRedaction, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "position": list(self.bbox),
            "cells": [c.to_dict() for c in self.cells],
        }


class RedactedCell(str):
    """
    A table-cell string that also records that the cell contained
    occluded glyphs.

    It is a genuine ``str`` (so callers that only need text keep working),
    but its ``redaction`` attribute carries the cell's position and
    occlusion counts. A cell that contains occluded glyphs is never plain
    ``""``, so it cannot be confused with a cell that was empty in the
    source document.
    """

    redaction: TableCellRedaction

    def __new__(cls, value: str, *, redaction: TableCellRedaction) -> "RedactedCell":
        obj = super().__new__(cls, value)
        obj.redaction = redaction
        return obj


@dataclass
class PageRedactionReport:
    """
    Redaction report for a single page: page number, positions, and
    counts only. It never contains text from the source document.
    """

    page_number: int
    redaction_rectangles: int
    merged_rectangles: int
    positions: List[T_bbox]
    union_area: T_num
    glyphs: OcclusionCounts
    tables: List[TableRedaction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "redaction_rectangles": self.redaction_rectangles,
            "merged_rectangles": self.merged_rectangles,
            "positions": [list(p) for p in self.positions],
            "union_area": self.union_area,
            "counts": self.glyphs.as_dict(),
            "tables": [t.to_dict() for t in self.tables],
        }


def build_page_report(
    redactor: Redactor,
    page_number: int,
    chars: Iterable[T_obj],
    tables: Optional[List["Table"]] = None,
    text_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[PageRedactionReport, Dict[int, List[List[Any]]]]:
    """
    Build a page report, plus a ``{table_index: matrix}`` mapping of every
    table's redacted cells.

    The returned report contains only positions and counts.
    """
    counts = redactor.counts(chars)
    table_reports: List[TableRedaction] = []
    cell_matrices: Dict[int, List[List[Any]]] = {}

    for i, table in enumerate(tables or []):
        matrix, cell_reports = redact_table_cells(
            table, redactor, text_settings=text_settings
        )
        cell_matrices[i] = matrix
        if cell_reports:
            table_reports.append(
                TableRedaction(
                    page_number=page_number,
                    bbox=table.bbox,
                    cells=tuple(cell_reports),
                )
            )

    report = PageRedactionReport(
        page_number=page_number,
        redaction_rectangles=redactor.raw_redaction_count,
        merged_rectangles=len(redactor.union),
        positions=list(redactor.union),
        union_area=redactor.union_area,
        glyphs=counts,
        tables=table_reports,
    )
    return report, cell_matrices


def redact_table_cells(
    table: "Table",
    redactor: Redactor,
    text_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[List[List[Any]], List[TableCellRedaction]]:
    """
    Extract a table's cells with redaction applied.

    Returns ``(matrix, cell_reports)``. Every returned cell is a
    :class:`RedactedCell` whose ``redaction`` attribute is populated for
    cells that contain occluded glyphs and is ``None`` otherwise.
    ``cell_reports`` contains only the marked cells.
    """
    kwargs = dict(text_settings or {})
    matrix = table.extract_redacted(redactor, cell_factory=RedactedCell, **kwargs)
    cell_reports: List[TableCellRedaction] = []
    for row in matrix:
        for cell in row:
            if cell is None:
                continue
            meta = getattr(cell, "redaction", None)
            if meta is not None:
                cell_reports.append(meta)
    return matrix, cell_reports


@dataclass
class RedactedExtraction:
    """
    The result of a one-pass, redacted extraction.

    ``text`` is a shareable string, ``tables`` is a list of
    ``table -> row -> cell`` matrices, and ``report`` is a
    :class:`PageRedactionReport` containing only positions and counts.
    """

    text: str
    tables: List[List[List[Any]]]
    report: PageRedactionReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tables": [
                [
                    [
                        (
                            cell.redaction.to_dict()
                            if isinstance(cell, RedactedCell)
                            else None
                        )
                        for cell in row
                    ]
                    for row in table
                ]
                for table in self.tables
            ],
            "report": self.report.to_dict(),
        }


@dataclass
class RedactionReport:
    """A multi-page redaction report. Contains no redacted text."""

    pages: List[PageRedactionReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        pages = [p.to_dict() for p in self.pages]
        return {
            "pages": pages,
            "counts": {
                "pages": len(pages),
                "redaction_rectangles": sum(p["redaction_rectangles"] for p in pages),
                "redacted_glyphs": sum(p["counts"]["redacted"] for p in pages),
                "fully_occluded_glyphs": sum(
                    p["counts"]["fully_occluded"] for p in pages
                ),
                "partially_occluded_glyphs": sum(
                    p["counts"]["partially_occluded"] for p in pages
                ),
                "occluded_cells": sum(
                    len(t["cells"]) for p in pages for t in p["tables"]
                ),
            },
        }


def build_redaction_report(
    pages: Sequence[Any],
    redactions_by_page: Mapping[int, Iterable[T_rect_like]],
    table_settings: Optional[Any] = None,
    include_partial: bool = True,
) -> RedactionReport:
    """
    Convenience function for assembling a multi-page report from a
    mapping of ``{page_number: redaction_rectangles}``.

    The report contains only page numbers, positions, and counts.
    """
    report = RedactionReport()
    for page in pages:
        rects = redactions_by_page.get(page.page_number, [])
        report.pages.append(
            page.redaction_report(
                rects,
                include_partial=include_partial,
                table_settings=table_settings,
            )
        )
    return report
