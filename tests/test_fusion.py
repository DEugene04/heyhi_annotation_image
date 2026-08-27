"""
Tests for laying the VLM's text onto Mathpix's shapes.

The IR below stands in for a Mathpix reading — two lines with shapes but with
misread text. Fusing in the VLM's lines should keep the shapes and fix the text,
and keep the character ranges and flat text consistent with the new text.
"""

import pytest

from contracts.schema import IR, ImageMeta, Point, Segment, SegmentType
from ocr.fusion import AlignmentError, alignment_ok, fuse


def _segment(id, line, text, x2):
    quad = [Point(x=0, y=line * 10), Point(x=x2, y=line * 10),
            Point(x=x2, y=line * 10 + 8), Point(x=0, y=line * 10 + 8)]
    return Segment(id=id, line=line, type=SegmentType.TEXT, text=text, quad=quad,
                   char_start=0, char_end=len(text), confidence=0.9, is_handwritten=True)


def _ir():
    return IR(
        image=ImageMeta(width=100, height=100),
        flat_text="teh cat\nsat down",
        segments=[_segment("s0", 0, "teh cat", 70), _segment("s1", 1, "sat down", 80)],
    )


def test_fuse_replaces_text_but_keeps_shapes():
    ir = _ir()
    fused = fuse(ir, ["the cat", "sat down"])
    assert [s.text for s in fused.segments] == ["the cat", "sat down"]
    # Shapes are untouched.
    assert fused.segments[0].quad == ir.segments[0].quad


def test_fuse_rebuilds_char_ranges_and_flat_text():
    fused = fuse(_ir(), ["the cat", "sat down"])
    assert fused.flat_text == "the cat\nsat down"
    for segment in fused.segments:
        assert fused.flat_text[segment.char_start : segment.char_end] == segment.text


def test_fuse_refuses_when_line_counts_differ():
    # Readers disagreeing on line count would slide text onto the wrong shapes,
    # so the page is refused rather than annotated wrongly.
    with pytest.raises(AlignmentError) as exc:
        fuse(_ir(), ["the cat"])  # two shapes, one VLM line
    assert exc.value.mathpix_lines == 2
    assert exc.value.vlm_lines == 1
    assert exc.value.warning  # a student-facing message to show instead


def test_alignment_flag():
    ir = _ir()
    assert alignment_ok(ir, ["a", "b"]) is True
    assert alignment_ok(ir, ["a", "b", "c"]) is False
