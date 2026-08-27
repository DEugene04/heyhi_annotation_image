"""
Stand-in data used while the evaluator is on hold.

The evaluator layer is paused, but the rest of the chain still needs something to
work against. This fixture provides the two inputs the geometry layer expects — a
reading of a page (an IR) and a handful of feedback items — and then runs them
through the real span-to-geometry resolver to produce the payload the frontend
draws. So the payload here is not hand-built: it is what the actual resolver
returns for this fixture, exactly as it will for the real evaluator later.

The feedback covers the cases that matter: a single-line remark, one that spans
adjacent lines, one that lands on two non-adjacent lines (two regions, one item),
and one with no place on the page at all (panel-only).
"""

from contracts.schema import (
    IR,
    Category,
    Feedback,
    ImageMeta,
    Point,
    Segment,
    SegmentType,
)
from geometry.resolver import resolve_payload


def _quad(x1: float, y1: float, x2: float, y2: float) -> list[Point]:
    """A four-corner box in the agreed order: top-left, TR, BR, bottom-left."""

    return [
        Point(x=x1, y=y1),
        Point(x=x2, y=y1),
        Point(x=x2, y=y2),
        Point(x=x1, y=y2),
    ]


# A small reading of a maths answer. Line 3 is a sketch the student drew, which
# is handled on a separate track and so has no text line-shape here — its text
# still sits in flat_text, leaving a gap between the line-shapes above and below
# it. That gap is what lets one feedback item produce two separate regions.
#   flat_text = "2x + 4 = 10\n2x = 6\n(sketch)\nx = 3"
#                0-11         12-18   19-27    28-33
SAMPLE_IR = IR(
    image=ImageMeta(width=1000, height=1400),
    flat_text="2x + 4 = 10\n2x = 6\n(sketch)\nx = 3",
    segments=[
        Segment(
            id="s1", line=1, type=SegmentType.MATH, text="2x + 4 = 10",
            latex="2x + 4 = 10", quad=_quad(120, 200, 560, 250),
            char_start=0, char_end=11, confidence=0.98, is_handwritten=True,
        ),
        Segment(
            id="s2", line=2, type=SegmentType.MATH, text="2x = 6",
            latex="2x = 6", quad=_quad(120, 300, 400, 350),
            char_start=12, char_end=18, confidence=0.97, is_handwritten=True,
        ),
        # Line 3 (the sketch) is intentionally absent from the line-shapes.
        Segment(
            id="s4", line=4, type=SegmentType.MATH, text="x = 3",
            latex="x = 3", quad=_quad(120, 470, 360, 520),
            char_start=28, char_end=33, confidence=0.95, is_handwritten=True,
        ),
    ],
)


# Feedback as the evaluator would return it, standing in with fixtures for now.
SAMPLE_FEEDBACK = [
    # A correct single line.
    Feedback(
        comment="Right — the equation is set up correctly.",
        category=Category.CORRECT,
        char_start=0, char_end=11,
    ),
    # A remark on a single line.
    Feedback(
        comment="Good — you subtracted 4 from both sides correctly.",
        category=Category.ERROR,
        char_start=12, char_end=18,
    ),
    # A remark spanning line 2 and line 4, with the sketch line between them:
    # two separate regions owned by one feedback item.
    Feedback(
        comment="The final value here doesn't follow from the working on line 2.",
        category=Category.CARRIED,
        char_start=12, char_end=33,
    ),
    # Feedback with no place on the page: panel-only.
    Feedback(
        comment="Show the division step between lines 2 and 3.",
        category=Category.INFO,
    ),
]


# The payload the frontend consumes, produced by the real resolver.
SAMPLE_PAYLOAD = resolve_payload(SAMPLE_IR, SAMPLE_FEEDBACK)


def stand_in_feedback(ir: IR) -> list[Feedback]:
    """
    A few feedback items anchored to real spans of a reading.

    The evaluator is on hold, so while the pipeline runs end-to-end we stand in
    for it: a mark on the first line, a remark on the third if there is one, and
    one panel-only note with no place on the page. Because the spans are taken
    from the reading passed in, they land on that page's real shapes.
    """

    segments = ir.segments
    feedback: list[Feedback] = []
    if segments:
        first = segments[0]
        feedback.append(Feedback(
            comment="Clear, confident opening line.",
            category=Category.CORRECT,
            char_start=first.char_start, char_end=first.char_end))
    if len(segments) >= 3:
        third = segments[2]
        feedback.append(Feedback(
            comment="This line needs another look.",
            category=Category.ERROR,
            char_start=third.char_start, char_end=third.char_end))
    feedback.append(Feedback(
        comment="Overall: a coherent, well-structured answer.",
        category=Category.INFO))
    return feedback
