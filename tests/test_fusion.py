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


def test_fuse_redistributes_when_counts_differ():
    # The VLM split the same words into a different number of lines. They are laid
    # back onto Mathpix's two lines, so each shape keeps its own text.
    fused = fuse(_ir(), ["the", "cat sat", "down"])  # 3 VLM lines, 2 Mathpix lines
    assert len(fused.segments) == 2
    assert [s.text for s in fused.segments] == ["the cat", "sat down"]


def test_fuse_refuses_when_agreement_is_too_low():
    # Different line count AND unrelated words: the readers disagree on the page,
    # so it is refused rather than annotated wrongly.
    with pytest.raises(AlignmentError) as exc:
        fuse(_ir(), ["xxxx yyyy zzzz"])
    assert exc.value.agreement < 0.3
    assert exc.value.warning  # a student-facing message to show instead


def test_alignment_flag():
    ir = _ir()
    assert alignment_ok(ir, ["a", "b"]) is True          # same count — always ok
    assert alignment_ok(ir, ["xxxx yyyy zzzz"]) is False  # count differs, no agreement
