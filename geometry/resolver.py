"""
Turns feedback positions into the shapes to draw on the image.

This layer is deterministic — it never guesses. Given a character range within
the answer text, it finds which line-shapes from the IR fall inside that range
and combines them into the region(s) to draw:

  - Lines that sit next to each other are joined into a single region whose
    outline follows all of them at once.
  - A gap between lines splits the feedback into separate regions, so one piece
    of feedback can own several shapes (for example, an error that lands on two
    lines that are not next to each other).
  - A range that matches no writing draws nothing, rather than a box in the
    wrong place. The feedback still appears, listed in the panel on its own.

This core — position in, shapes out — stays deterministic. The part that turns
the real evaluator's verbatim-text spans into positions lives in
evaluator/adapter.py, which locates each target in flat_text before this layer
draws it (see contracts/evaluator_contract.md).
"""

import logging
from typing import List, Optional

from config import settings
from contracts.schema import (
    IR,
    AnnotationItem,
    AnnotationPayload,
    CriterionScore,
    Feedback,
    Point,
    Quad,
    Region,
    Segment,
)

logger = logging.getLogger(__name__)


def _corners(quad: Quad) -> tuple[Point, Point, Point, Point]:
    """Read a line-shape's four corners, in their agreed order."""

    top_left, top_right, bottom_right, bottom_left = quad[0], quad[1], quad[2], quad[3]
    return top_left, top_right, bottom_right, bottom_left


def _overlaps(segment: Segment, char_start: int, char_end: int) -> bool:
    """True when a line's character range touches the feedback's range."""

    return segment.char_start < char_end and segment.char_end > char_start


def _lerp(a: Point, b: Point, t: float) -> Point:
    """The point a fraction `t` of the way from `a` to `b`."""

    return Point(x=a.x + (b.x - a.x) * t, y=a.y + (b.y - a.y) * t)


def _fractions_in_segment(
    segment: Segment, char_start: int, char_end: int
) -> tuple[float, float]:
    """
    Where in a line, left to right, the feedback span falls — as two fractions.

    The span [char_start, char_end) is clipped to this line's character range and
    turned into the fraction of the line it covers, assuming characters are spread
    evenly across the line's width. A span covering the whole line gives (0, 1); a
    span over the first half gives (0, 0.5). This is what lets the drawn box hug
    just the words the feedback points at instead of the whole line.

    Falls back to the full line (0, 1) whenever the maths would be degenerate (an
    empty line, or a clipped span with no width), so a box is never lost.
    """

    length = segment.char_end - segment.char_start
    if length <= 0:
        return 0.0, 1.0
    local_start = max(char_start, segment.char_start) - segment.char_start
    local_end = min(char_end, segment.char_end) - segment.char_start
    f0 = max(0.0, min(1.0, local_start / length))
    f1 = max(0.0, min(1.0, local_end / length))
    if f1 <= f0:
        return 0.0, 1.0
    return f0, f1


def _trim_quad(quad: Quad, f0: float, f1: float) -> Quad:
    """
    Narrow a line-shape horizontally to the fraction [f0, f1] of its width.

    The top edge (top-left → top-right) and the bottom edge (bottom-left →
    bottom-right) are each cut at f0 and f1, so the box stays true to the line
    even if it is slightly skewed. With (0, 1) this returns the original quad
    unchanged, so the full-line behaviour is a special case of the same path.
    """

    top_left, top_right, bottom_right, bottom_left = _corners(quad)
    return [
        _lerp(top_left, top_right, f0),
        _lerp(top_left, top_right, f1),
        _lerp(bottom_left, bottom_right, f1),
        _lerp(bottom_left, bottom_right, f0),
    ]


def _group_into_runs(segments: List[Segment]) -> List[List[Segment]]:
    """
    Split lines into runs that sit next to each other on the page.

    Lines whose numbers follow on from one another (2 then 3) belong to the same
    run; a jump (3 then 5) starts a new one. Each run becomes one region.
    """

    ordered = sorted(segments, key=lambda s: s.line)
    runs: List[List[Segment]] = []
    for segment in ordered:
        if runs and segment.line == runs[-1][-1].line + 1:
            runs[-1].append(segment)
        else:
            runs.append([segment])
    return runs


def _region_for_run(run: List[Segment], char_start: int, char_end: int) -> Region:
    """
    Build one region whose outline follows every line in a run.

    Each line is first trimmed to the part of it the feedback span actually
    covers (see _trim_quad): the first line starts where the span starts, the
    last line ends where it ends, and any lines between are covered in full. When
    tight boxes are turned off, every line is trimmed to (0, 1) — its full width —
    reproducing the old whole-line behaviour.

    A single line keeps its own (trimmed) shape. Several lines are traced as one
    outline: down the left edge of each line from top to bottom, then back up the
    right edge from bottom to top, so the shape hugs the writing.
    """

    def trimmed(segment: Segment) -> Quad:
        if settings.tight_feedback_boxes:
            f0, f1 = _fractions_in_segment(segment, char_start, char_end)
        else:
            f0, f1 = 0.0, 1.0
        return _trim_quad(segment.quad, f0, f1)

    if len(run) == 1:
        return Region(quad=trimmed(run[0]))

    quads = [trimmed(segment) for segment in run]
    points: List[Point] = []
    for quad in quads:  # top to bottom, down the left edge
        top_left, _, _, bottom_left = _corners(quad)
        points.append(top_left)
        points.append(bottom_left)
    for quad in reversed(quads):  # bottom to top, up the right edge
        _, top_right, bottom_right, _ = _corners(quad)
        points.append(bottom_right)
        points.append(top_right)
    return Region(quad=points)


def resolve_regions(ir: IR, char_start: int, char_end: int) -> List[Region]:
    """
    Find the shapes to draw for one feedback span.

    Returns one region per run of lines the span covers, in reading order.
    Returns an empty list when the span matches no writing.
    """

    if char_start >= char_end:
        return []

    touched = [seg for seg in ir.segments if _overlaps(seg, char_start, char_end)]
    if not touched:
        return []

    return [
        _region_for_run(run, char_start, char_end)
        for run in _group_into_runs(touched)
    ]


def resolve_payload(
    ir: IR,
    feedback: List[Feedback],
    scorecard: Optional[List[CriterionScore]] = None,
) -> AnnotationPayload:
    """
    Turn the IR and a list of feedback into the payload the frontend draws.

    Each feedback item becomes one annotation item. Feedback with no span, and
    feedback whose span matches no writing, becomes a panel-only item with no
    region — the feedback is never dropped, only its (missing or wrong) shape is.

    `scorecard` is the essay path's per-criterion rubric results, shown in the
    panel; it is passed straight through. The short-answer path leaves it empty.
    """

    items: List[AnnotationItem] = []
    for index, fb in enumerate(feedback):
        item_id = f"item-{index}"

        if fb.char_start is None or fb.char_end is None:
            regions: List[Region] = []
        else:
            regions = resolve_regions(ir, fb.char_start, fb.char_end)
            if not regions:
                logger.warning(
                    "Feedback %s span [%s, %s) matched no writing; "
                    "showing it in the panel with no region.",
                    item_id,
                    fb.char_start,
                    fb.char_end,
                )

        items.append(
            AnnotationItem(
                id=item_id,
                category=fb.category,
                comment=fb.comment,
                regions=regions,
            )
        )

    return AnnotationPayload(image=ir.image, items=items, scorecard=scorecard or [])
