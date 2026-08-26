"""
Builds the IR from Mathpix's line reading *and* word reading together.

Mathpix's line reading is good at telling where the lines of a page are — but it
folds a whole maths derivation into one box. Its word reading gives the precise
shape of every word, but on its own we would have to guess where the lines are.
So we use each for what it is good at:

  - The line reading is the skeleton: it decides the lines, in reading order,
    and which are text, maths, or figures.
  - Each word from the word reading is dropped into the line whose box contains
    it, giving every line its precise word shapes.
  - A maths line that Mathpix grouped into several rows is split back into those
    rows. We know exactly how many rows to expect, because Mathpix's own LaTeX
    separates them with "\\\\"; the words then give each row its shape.

The result is one segment per real line (or per real maths row), each carrying
the shapes of its words — accurate geometry at whatever size the feedback needs.
"""

import re
from typing import List, Optional

from contracts.schema import IR, ImageMeta, Point, Quad, Segment, SegmentType, Word
from ocr.ir_builder import (
    _DIAGRAM_TYPES,
    _latex_of,
    _quad_from_contour,
    _segment_type,
)

# Splits a grouped maths block into its rows, on the LaTeX row separator "\\".
_ROW_SEPARATOR = re.compile(r"\\\\")
# Matches \begin{array}{l} and \end{array}, including the optional column spec.
_ENVIRONMENT = re.compile(r"\\(begin|end)\{[^}]*\}(\{[^}]*\})?")
_MATH_DELIMS = re.compile(r"\\\(|\\\)|\\\[|\\\]|\$\$|\$")


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


def _is_grouped(text: str) -> bool:
    """Whether a maths line holds several rows folded into one."""

    return "\\begin{" in text or bool(_ROW_SEPARATOR.search(text))


def _latex_rows(text: str) -> List[str]:
    """Split a grouped maths block into its individual row expressions."""

    inner = _ENVIRONMENT.sub("", text)
    inner = _MATH_DELIMS.sub("", inner)
    return [row.strip() for row in _ROW_SEPARATOR.split(inner) if row.strip()]


def _assign_words_to_lines(line_data: List[dict], words: List[dict]) -> None:
    """
    Record on each word which line it belongs to.

    A word belongs to the line whose box contains its centre. A word that falls
    in no line's box (a stray near an edge) is given to the nearest line by
    height, so nothing is lost.
    """

    boxes = [(_bounds(l["cnt"]) if l.get("cnt") else None) for l in line_data]

    for word in words:
        cx, cy = 0.0, 0.0
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


def _words_of(words: List[dict]) -> List[Word]:
    return [Word(text=_word_text(w), quad=_quad_from_contour(w["cnt"])) for w in words]


def reconstruct(line_response: dict, word_data: List[dict],
                image_width: int, image_height: int) -> IR:
    """Build the IR from the line reading and the word reading together."""

    line_data = line_response.get("line_data", []) or []
    words = _drop_oversized([w for w in word_data if w.get("cnt")])
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
