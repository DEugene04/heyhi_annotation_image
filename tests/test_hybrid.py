"""
Tests for the hybrid reconstruction: line reading + word reading combined.

The readings below are shaped like Mathpix replies — a text line, then a maths
block that Mathpix folded into three rows — with word entries positioned to fall
inside each. We check the block is split back into its rows, each row gets the
shape of its words, and ordinary lines keep their words too.
"""

from contracts.schema import SegmentType
from ocr.hybrid import reconstruct


def _cnt(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


LINE_RESPONSE = {
    "line_data": [
        {"id": "l0", "type": "text", "is_handwritten": True, "confidence": 0.98,
         "text": "Working:", "cnt": _cnt(0, 0, 120, 20)},
        {"id": "l1", "type": "math", "is_handwritten": True, "confidence": 0.95,
         "text": r"\[ \begin{array}{l} a=1 \\ b=2 \\ c=3 \end{array} \]",
         "cnt": _cnt(0, 30, 120, 120)},
    ]
}

WORD_DATA = [
    {"type": "text", "text": "Working:", "cnt": _cnt(0, 0, 80, 20)},
    {"type": "math", "text": "a=1", "cnt": _cnt(0, 35, 40, 55)},
    {"type": "math", "text": "b=2", "cnt": _cnt(0, 65, 40, 85)},
    {"type": "math", "text": "c=3", "cnt": _cnt(0, 95, 40, 115)},
]


def _ir():
    return reconstruct(LINE_RESPONSE, WORD_DATA, image_width=200, image_height=200)


def test_grouped_block_is_split_into_its_rows():
    ir = _ir()
    # One text line + three maths rows = four segments.
    assert [s.text for s in ir.segments] == ["Working:", "a=1", "b=2", "c=3"]
    assert [s.type for s in ir.segments[1:]] == [SegmentType.MATH] * 3


def test_each_row_takes_the_shape_of_its_words():
    ir = _ir()
    row_a = ir.segments[1]  # "a=1"
    # Its box is the word's box, not the whole block.
    assert (row_a.quad[0].x, row_a.quad[0].y) == (0, 35)
    assert (row_a.quad[2].x, row_a.quad[2].y) == (40, 55)


def test_rows_are_numbered_in_reading_order():
    ir = _ir()
    assert [s.line for s in ir.segments] == [0, 1, 2, 3]


def test_ordinary_line_keeps_its_words():
    ir = _ir()
    working = ir.segments[0]
    assert [w.text for w in working.words] == ["Working:"]


def test_char_ranges_index_into_flat_text():
    ir = _ir()
    for segment in ir.segments:
        assert ir.flat_text[segment.char_start : segment.char_end] == segment.text
    assert ir.flat_text == "Working:\na=1\nb=2\nc=3"
