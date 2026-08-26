"""
Tests for the span-to-geometry core, driven by hand-made fixture offsets.

The IR below stands in for a real reading: three lines of writing at known
positions on the page, with known character ranges in the flat text. Each test
hands the resolver a character range and checks the shapes it draws — no live
evaluator involved.
"""

from contracts.schema import IR, Feedback, ImageMeta, Point, Segment, SegmentType
from geometry.resolver import resolve_payload, resolve_regions


def _quad(x1, y1, x2, y2):
    """A four-corner box in the agreed order: TL, TR, BR, BL."""

    return [
        Point(x=x1, y=y1),
        Point(x=x2, y=y1),
        Point(x=x2, y=y2),
        Point(x=x1, y=y2),
    ]


def _ir():
    """
    Three lines. flat_text is:

        "line one\nline two\nline four"
         0-8       9-17      18-27

    Line numbers are 1, 2, and 4 — there is a gap at line 3 on purpose, so we can
    test that non-adjacent lines split into separate regions.
    """

    return IR(
        image=ImageMeta(width=1000, height=1400),
        flat_text="line one\nline two\nline four",
        segments=[
            Segment(
                id="s1", line=1, type=SegmentType.TEXT, text="line one",
                quad=_quad(100, 100, 500, 140),
                char_start=0, char_end=8, confidence=0.99, is_handwritten=True,
            ),
            Segment(
                id="s2", line=2, type=SegmentType.TEXT, text="line two",
                quad=_quad(100, 150, 500, 190),
                char_start=9, char_end=17, confidence=0.99, is_handwritten=True,
            ),
            Segment(
                id="s4", line=4, type=SegmentType.TEXT, text="line four",
                quad=_quad(100, 300, 520, 340),
                char_start=18, char_end=27, confidence=0.99, is_handwritten=True,
            ),
        ],
    )


def test_single_line_gives_one_region_matching_the_line():
    regions = resolve_regions(_ir(), 0, 8)
    assert len(regions) == 1
    assert regions[0].quad == _quad(100, 100, 500, 140)


def test_adjacent_lines_merge_into_one_region():
    # A span covering line 1 and line 2 (which are next to each other).
    regions = resolve_regions(_ir(), 0, 17)
    assert len(regions) == 1
    # One outline tracing both lines: two points per line, both edges = 8 points.
    assert len(regions[0].quad) == 8


def test_non_adjacent_lines_split_into_separate_regions():
    # A span covering line 2 and line 4, with the line-3 gap between them.
    regions = resolve_regions(_ir(), 9, 27)
    assert len(regions) == 2
    assert all(len(r.quad) == 4 for r in regions)


def test_span_matching_no_writing_gives_no_region():
    # A range past the end of the text.
    assert resolve_regions(_ir(), 500, 600) == []


def test_empty_range_gives_no_region():
    assert resolve_regions(_ir(), 5, 5) == []


def test_partial_overlap_still_selects_the_line():
    # A range landing in the middle of line one still highlights that line.
    regions = resolve_regions(_ir(), 2, 4)
    assert len(regions) == 1
    assert regions[0].quad == _quad(100, 100, 500, 140)


def test_span_less_feedback_becomes_panel_only_item():
    payload = resolve_payload(
        _ir(),
        [Feedback(comment="You skipped a step.", category="info")],
    )
    assert len(payload.items) == 1
    assert payload.items[0].regions == []
    assert payload.items[0].comment == "You skipped a step."


def test_unresolvable_span_keeps_feedback_but_drops_region():
    payload = resolve_payload(
        _ir(),
        [Feedback(comment="Somewhere off the page.", category="error",
                  char_start=500, char_end=600)],
    )
    assert len(payload.items) == 1
    assert payload.items[0].regions == []  # region dropped, feedback kept


def test_multi_region_item_from_a_gapped_span():
    payload = resolve_payload(
        _ir(),
        [Feedback(comment="Carried down to line four.", category="carried",
                  char_start=9, char_end=27)],
    )
    assert len(payload.items) == 1
    assert len(payload.items[0].regions) == 2  # one item owning two shapes


def test_payload_carries_the_image_metadata():
    payload = resolve_payload(_ir(), [])
    assert payload.image.width == 1000
    assert payload.image.height == 1400
