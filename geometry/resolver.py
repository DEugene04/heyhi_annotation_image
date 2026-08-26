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

Only this core — position in, shapes out — is built. The part that turns the
real evaluator's output into a position is on hold until we know how the
evaluator points at spans (see contracts/evaluator_contract.md).
"""

import logging
from typing import List

from contracts.schema import (
    IR,
    AnnotationItem,
    AnnotationPayload,
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


def _region_for_run(run: List[Segment]) -> Region:
    """
    Build one region whose outline follows every line in a run.

    A single line keeps its own shape. Several lines are traced as one outline:
    down the left edge of each line from top to bottom, then back up the right
    edge from bottom to top, so the shape hugs the writing instead of boxing it.
    """

    if len(run) == 1:
        return Region(quad=list(run[0].quad))

    points: List[Point] = []
    for segment in run:  # top to bottom, down the left edge
        top_left, _, _, bottom_left = _corners(segment.quad)
        points.append(top_left)
        points.append(bottom_left)
    for segment in reversed(run):  # bottom to top, up the right edge
        _, top_right, bottom_right, _ = _corners(segment.quad)
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

    return [_region_for_run(run) for run in _group_into_runs(touched)]


def resolve_payload(ir: IR, feedback: List[Feedback]) -> AnnotationPayload:
    """
    Turn the IR and a list of feedback into the payload the frontend draws.

    Each feedback item becomes one annotation item. Feedback with no span, and
    feedback whose span matches no writing, becomes a panel-only item with no
    region — the feedback is never dropped, only its (missing or wrong) shape is.
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

    return AnnotationPayload(image=ir.image, items=items)
