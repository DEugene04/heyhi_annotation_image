"""
Builds our Intermediate Representation (the IR) from what Mathpix reads.

Mathpix reads the page two ways, and each is good at a different thing:
  - The line reading tells us where the lines are, in reading order, and which
    are text, maths, or figures. It is the skeleton.
  - The word reading gives the precise shape of every individual word.

This layer reconciles the two — dropping each word onto the line that contains
it, and splitting a maths block Mathpix folded into one box back into its rows —
and then assembles the result into the IR: one segment per line of writing, each
with its text, its maths notation, its shape on the image, and its position as a
character range inside one continuous answer string. It also produces that
continuous string with line breaks preserved.

Everything downstream reads from the IR, not from Mathpix directly, so if the OCR
provider ever changes, this is the layer that changes.

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
from contracts.schema import IR, ImageMeta, Point, Quad, Segment, SegmentType, Word

# Mathpix line types we treat as a drawn figure rather than a line of writing.
_DIAGRAM_TYPES = {"diagram", "chart"}

# Pulls the LaTeX out of a maths line's Mathpix Markdown, e.g. "\( 2x = 6 \)".
_MATH_DELIMITERS = re.compile(r"\\\(|\\\)|\\\[|\\\]|\$\$|\$")

# Splits a grouped maths block into its rows, on the LaTeX row separator "\\".
_ROW_SEPARATOR = re.compile(r"\\\\")

# Matches \begin{array}{l} and \end{array}, including the optional column spec.
_ENVIRONMENT = re.compile(r"\\(begin|end)\{[^}]*\}(\{[^}]*\})?")


# --- Shared helpers: reading Mathpix's shapes and maths notation -------------


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


def _bounds(contour: List[List[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    return min(xs), min(ys), max(xs), max(ys)


def _y_center(contour: List[List[float]]) -> float:
    ys = [p[1] for p in contour]
    return (min(ys) + max(ys)) / 2


def _height(contour: List[List[float]]) -> float:
    ys = [p[1] for p in contour]
    return max(ys) - min(ys)


def _quad(left: float, top: float, right: float, bottom: float) -> Quad:
    return [
        Point(x=left, y=top),
        Point(x=right, y=top),
        Point(x=right, y=bottom),
        Point(x=left, y=bottom),
    ]


def _union_quad(words: List[dict]) -> Quad:
    """The box that just contains a group of words."""

    boxes = [_bounds(w["cnt"]) for w in words]
    return _quad(
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _word_text(word: dict) -> str:
    return (word.get("text") or word.get("latex") or "").strip()


def _words_of(words: List[dict]) -> List[Word]:
    return [Word(text=_word_text(w), quad=_quad_from_contour(w["cnt"])) for w in words]


# --- Reconciliation: line reading + word reading together -------------------


def _drop_oversized(words: List[dict]) -> List[dict]:
    """
    Remove word entries far taller than a normal word.

    Mathpix occasionally returns one entry whose box covers several rows at once.
    Left in, it would blow a tight row box up to cover half the page, so we drop
    these outliers and keep the real per-word entries.
    """

    if not words:
        return words
    heights = sorted(_height(w["cnt"]) for w in words)
    median = heights[len(heights) // 2]
    return [w for w in words if _height(w["cnt"]) <= 4 * median]


def _rescale(line_data: List[dict], words: List[dict], line_response: dict,
             image_width: int, image_height: int) -> None:
    """
    Scale Mathpix's shape coordinates onto the image we display.

    Mathpix returns shapes in the pixels of the image it processed, which may be
    a shrunk copy of ours. This multiplies every point back to our image's size.
    Does nothing when the sizes already match (the common case).
    """

    processed_w = line_response.get("image_width") or image_width
    processed_h = line_response.get("image_height") or image_height
    scale_x = image_width / processed_w
    scale_y = image_height / processed_h
    if scale_x == 1 and scale_y == 1:
        return

    for item in [*line_data, *words]:
        if item.get("cnt"):
            item["cnt"] = [[x * scale_x, y * scale_y] for x, y in item["cnt"]]


def _assign_words_to_lines(line_data: List[dict], words: List[dict]) -> None:
    """
    Record on each word which line it belongs to.

    A word belongs to the line whose box contains its centre. A word that falls
    in no line's box (a stray near an edge) is given to the nearest line by
    height, so nothing is lost.
    """

    boxes = [(_bounds(l["cnt"]) if l.get("cnt") else None) for l in line_data]

    for word in words:
        wb = _bounds(word["cnt"])
        cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2

        word["_line"] = None
        nearest, best = None, float("inf")
        for i, box in enumerate(boxes):
            if box is None:
                continue
            left, top, right, bottom = box
            if left <= cx <= right and top <= cy <= bottom:
                word["_line"] = i
                break
            distance = abs((top + bottom) / 2 - cy)
            if distance < best:
                best, nearest = distance, i
        if word["_line"] is None:
            word["_line"] = nearest


# --- Splitting a grouped maths block back into its rows ----------------------


def _is_grouped(text: str) -> bool:
    """Whether a maths line holds several rows folded into one."""

    return "\\begin{" in text or bool(_ROW_SEPARATOR.search(text))


def _latex_rows(text: str) -> List[str]:
    """Split a grouped maths block into its individual row expressions."""

    inner = _ENVIRONMENT.sub("", text)
    inner = _MATH_DELIMITERS.sub("", inner)
    return [row.strip() for row in _ROW_SEPARATOR.split(inner) if row.strip()]


def _split_into_rows(words: List[dict], row_count: int) -> List[List[dict]]:
    """
    Group a block's words into rows, top to bottom.

    We already know how many rows there should be, so we sort the words by
    height and cut at the biggest vertical gaps between them.
    """

    if row_count <= 1 or len(words) <= 1:
        return [words] if words else []

    ordered = sorted(words, key=lambda w: _y_center(w["cnt"]))
    centers = [_y_center(w["cnt"]) for w in ordered]
    gaps = sorted(range(1, len(ordered)), key=lambda i: centers[i] - centers[i - 1], reverse=True)
    cuts = sorted(gaps[: row_count - 1])

    rows, previous = [], 0
    for cut in cuts:
        rows.append(ordered[previous:cut])
        previous = cut
    rows.append(ordered[previous:])
    return rows


# --- Assembly: reconciled lines into the IR ---------------------------------


def reconstruct(line_response: dict, word_data: List[dict],
                image_width: int, image_height: int) -> IR:
    """Build the IR from the line reading and the word reading together."""

    line_data = line_response.get("line_data", []) or []
    words = _drop_oversized([w for w in word_data if w.get("cnt")])

    # Large images are shrunk before being sent to Mathpix, so Mathpix reports
    # its shapes in the shrunk image's pixels. Scale them back up to the image we
    # actually draw on, using the size Mathpix says it processed.
    _rescale(line_data, words, line_response, image_width, image_height)

    _assign_words_to_lines(line_data, words)

    segments: List[Segment] = []
    diagrams: List[Segment] = []
    texts: List[str] = []
    offset = 0
    line_no = 0

    def add(id, seg_type, text, latex, quad, confidence, is_handwritten, words):
        """Add one segment in reading order, recording its character range."""
        nonlocal offset, line_no
        char_start = offset
        char_end = char_start + len(text)
        texts.append(text)
        segments.append(
            Segment(
                id=id, line=line_no, type=seg_type, text=text, latex=latex, quad=quad,
                char_start=char_start, char_end=char_end, confidence=confidence,
                is_handwritten=is_handwritten, words=words,
            )
        )
        offset = char_end + 1
        line_no += 1

    for index, line in enumerate(line_data):
        if not line.get("cnt"):
            continue

        seg_type = _segment_type(line.get("type"))
        confidence = line.get("confidence")
        if confidence is None:
            confidence = line.get("confidence_rate")
        if confidence is None:
            confidence = 0.0
        is_hw = bool(line.get("is_handwritten", False))
        line_words = [w for w in words if w.get("_line") == index]

        if seg_type == SegmentType.DIAGRAM:
            diagrams.append(
                Segment(
                    id=line.get("id") or f"diagram-{index}", line=line_no,
                    type=seg_type, text=(line.get("text") or "").strip(), latex=None,
                    quad=_quad_from_contour(line["cnt"]), char_start=0, char_end=0,
                    confidence=confidence, is_handwritten=is_hw,
                )
            )
            line_no += 1
            continue

        text = (line.get("text") or "").strip()

        # A grouped maths block is split back into its rows.
        if seg_type == SegmentType.MATH and _is_grouped(text):
            rows = _latex_rows(text)
            word_rows = _split_into_rows(line_words, len(rows))
            block = _bounds(line["cnt"])
            for i, row_text in enumerate(rows):
                row_words = word_rows[i] if i < len(word_rows) else []
                if row_words:
                    quad = _union_quad(row_words)
                else:  # no words landed here: fall back to an even slice of the block
                    top = block[1] + (block[3] - block[1]) * i / len(rows)
                    bottom = block[1] + (block[3] - block[1]) * (i + 1) / len(rows)
                    quad = _quad(block[0], top, block[2], bottom)
                add(f"{line.get('id') or index}-r{i}", SegmentType.MATH, row_text,
                    _latex_of(row_text), quad, confidence, is_hw, _words_of(row_words))
            continue

        # An ordinary line: keep Mathpix's line, attach its words.
        add(line.get("id") or f"seg-{index}", seg_type, text,
            _latex_of(text) if seg_type == SegmentType.MATH else None,
            _quad_from_contour(line["cnt"]), confidence, is_hw, _words_of(line_words))

    return IR(
        segments=segments,
        flat_text="\n".join(texts),
        image=ImageMeta(width=image_width, height=image_height),
        diagrams=diagrams,
    )


# --- Reading quality and the orchestrator -----------------------------------


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


def build_ir(image_bytes: bytes) -> IR:
    """
    Read a photo with Mathpix and package it into the IR.

    Uses both readings — line and word — so grouped maths is split back into its
    rows and every line carries its word shapes.
    """

    from io import BytesIO

    from PIL import Image

    from ocr.mathpix import read_all

    width, height = Image.open(BytesIO(image_bytes)).size
    both = read_all(image_bytes)
    return reconstruct(both["line"], both["word"], width, height)
