"""
Reads a cleaned image and packages what it finds into our Intermediate
Representation (the IR).

It calls Mathpix to read the handwriting, then maps Mathpix's raw response into
our own consistent structure: one segment per line of writing, each with its
text, its maths notation, its shape on the image, and its position as a
character range inside one continuous answer string. Everything downstream reads
from the IR, so if the OCR provider ever changes, only this layer changes.

Two things are decided here, from the research plan:
  - Every line of writing is kept, printed or handwritten. Telling the student's
    answer apart from a question copied onto the sheet is the evaluator's job: it
    is given the question as well as the text, and excludes the question itself.
    We do not try to guess it here, because the printed/handwritten flag is not
    reliable (clean or digital handwriting is often reported as printed).
  - Drawn figures are set aside on their own track, because they are annotated
    as whole figures, not read as text.
"""

import re
from typing import List, Optional

from config import settings
from contracts.schema import IR, ImageMeta, Point, Quad, Segment, SegmentType
from ocr.mathpix import read_image

# Mathpix line types we treat as a drawn figure rather than a line of writing.
_DIAGRAM_TYPES = {"diagram", "chart"}

# Pulls the LaTeX out of a maths line's Mathpix Markdown, e.g. "\( 2x = 6 \)".
_MATH_DELIMITERS = re.compile(r"\\\(|\\\)|\\\[|\\\]|\$\$|\$")


def _quad_from_contour(contour: List[List[float]]) -> Quad:
    """
    Turn a Mathpix contour into our four-corner quad.

    Mathpix outlines each line with a contour of many points. We take the box
    that just contains it, in our corner order: top-left, TR, BR, bottom-left.
    On a cleaned (straightened) image this box sits squarely on the writing.
    """

    xs = [point[0] for point in contour]
    ys = [point[1] for point in contour]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    return [
        Point(x=left, y=top),
        Point(x=right, y=top),
        Point(x=right, y=bottom),
        Point(x=left, y=bottom),
    ]


def _latex_of(text: str) -> Optional[str]:
    """Strip the maths delimiters from a maths line, leaving the LaTeX."""

    return _MATH_DELIMITERS.sub("", text).strip()


def _segment_type(mathpix_type: Optional[str]) -> SegmentType:
    """Map a Mathpix line type onto our simpler set."""

    if mathpix_type == "math":
        return SegmentType.MATH
    if mathpix_type in _DIAGRAM_TYPES:
        return SegmentType.DIAGRAM
    return SegmentType.TEXT


def parse_ir(response: dict, image_width: int, image_height: int) -> IR:
    """
    Turn a raw Mathpix response into the IR.

    The image size is passed in (measured from the cleaned image) rather than
    read from the response, so the annotations always match the picture the
    student sees. Kept apart from the network call so it can be tested on saved
    responses.
    """

    line_data = response.get("line_data", []) or []
    image_meta = ImageMeta(width=image_width, height=image_height)

    segments: List[Segment] = []
    diagrams: List[Segment] = []
    texts: List[str] = []
    offset = 0

    for line_number, line in enumerate(line_data):
        contour = line.get("cnt")
        if not contour:
            continue

        seg_type = _segment_type(line.get("type"))
        text = (line.get("text") or "").strip()
        quad = _quad_from_contour(contour)
        confidence = line.get("confidence")
        if confidence is None:
            confidence = line.get("confidence_rate")
        if confidence is None:
            confidence = 0.0

        # A drawn figure goes on its own track and never joins the text.
        if seg_type == SegmentType.DIAGRAM:
            diagrams.append(
                Segment(
                    id=line.get("id") or f"diagram-{line_number}",
                    line=line_number,
                    type=seg_type,
                    text=text,
                    latex=None,
                    quad=quad,
                    char_start=0,
                    char_end=0,
                    confidence=confidence,
                    is_handwritten=bool(line.get("is_handwritten", False)),
                )
            )
            continue

        char_start = offset
        char_end = char_start + len(text)
        offset = char_end + 1  # +1 for the line break that will join the lines
        texts.append(text)

        segments.append(
            Segment(
                id=line.get("id") or f"seg-{line_number}",
                line=line_number,
                type=seg_type,
                text=text,
                latex=_latex_of(text) if seg_type == SegmentType.MATH else None,
                quad=quad,
                char_start=char_start,
                char_end=char_end,
                confidence=confidence,
                is_handwritten=bool(line.get("is_handwritten", False)),
            )
        )

    return IR(
        segments=segments,
        flat_text="\n".join(texts),
        image=image_meta,
        diagrams=diagrams,
    )


def area_weighted_confidence(ir: IR) -> float:
    """
    How sure the reading is overall, weighting bigger lines more heavily.

    A large, confidently-read line counts for more than a tiny uncertain one, so
    a couple of small shaky lines don't sink an otherwise clear page. Returns 0
    when there is nothing to measure.
    """

    total_area = 0.0
    weighted = 0.0
    for segment in ir.segments:
        xs = [p.x for p in segment.quad]
        ys = [p.y for p in segment.quad]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        total_area += area
        weighted += area * segment.confidence

    return weighted / total_area if total_area else 0.0


def is_readable(ir: IR) -> bool:
    """
    Whether the page was read clearly enough to annotate.

    Below the threshold, the student is asked to retake the photo rather than
    shown annotations placed on an unreliable reading.
    """

    return area_weighted_confidence(ir) >= settings.confidence_threshold


def build_ir(cleaned_image: bytes) -> IR:
    """
    Read a cleaned image with Mathpix and package it into the IR.

    Uses both readings — line and word — so grouped maths is split back into its
    rows and every line carries its word shapes (see ocr/hybrid.py).
    """

    from io import BytesIO

    from PIL import Image

    from ocr.hybrid import reconstruct
    from ocr.mathpix import read_all

    width, height = Image.open(BytesIO(cleaned_image)).size
    both = read_all(cleaned_image)
    return reconstruct(both["line"], both["word"], width, height)
