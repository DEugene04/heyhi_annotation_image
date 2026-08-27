"""
Combines the VLM's accurate text with Mathpix's accurate shapes.

Mathpix gives us one segment per line (or maths row), each with a shape on the
image. The VLM gives us the text of each line, read far more accurately. This
lays the VLM's text onto Mathpix's shapes: each Mathpix segment keeps its shape
and position but takes the VLM's reading as its text. The character ranges and
the continuous text are rebuilt from the new text, so feedback that points at a
span still lands on the right shape.

The two lists are matched line by line, in reading order, so they must describe
the same number of lines. When they disagree, the readers have split the page
differently and a line-by-line match would slide the text onto the wrong shapes
from the mismatch onward. This is an annotation system, so a wrong box is worse
than no result: rather than fuse a page the readers disagree on, we refuse it and
ask the student for a clearer photo.
"""

from typing import List

from contracts.schema import IR, Segment

# Shown to the student when the page cannot be annotated reliably.
RETAKE_WARNING = (
    "We couldn't read this page clearly. Please retake a straight, "
    "well-lit photo of the whole answer."
)


class AlignmentError(Exception):
    """
    Raised when the two readings describe a different number of lines.

    Carries the student-facing warning to show, plus the two line counts for
    logging so a page that keeps failing can be looked at.
    """

    def __init__(self, mathpix_lines: int, vlm_lines: int):
        self.mathpix_lines = mathpix_lines
        self.vlm_lines = vlm_lines
        self.warning = RETAKE_WARNING
        super().__init__(
            f"{RETAKE_WARNING} (readers disagree on line count: "
            f"Mathpix {mathpix_lines}, VLM {vlm_lines})"
        )


def alignment_ok(ir: IR, vlm_lines: List[str]) -> bool:
    """
    Whether the two readings line up one-to-one.

    They are matched line by line, so they must split the page into the same
    number of lines. When they do not, we cannot trust which text belongs on
    which shape. Check this before fusing to decide the page's outcome.
    """

    return len(ir.segments) == len(vlm_lines)


def fuse(ir: IR, vlm_lines: List[str]) -> IR:
    """
    Return a new IR with Mathpix's shapes carrying the VLM's text.

    Requires the two readings to describe the same number of lines. If they do
    not, raises AlignmentError instead of sliding text onto the wrong shapes, so
    the page is sent back for a clearer photo rather than annotated wrongly.
    """

    if not alignment_ok(ir, vlm_lines):
        raise AlignmentError(len(ir.segments), len(vlm_lines))

    segments: List[Segment] = []
    offset = 0
    for segment, text in zip(ir.segments, vlm_lines):
        char_start = offset
        char_end = char_start + len(text)
        offset = char_end + 1
        segments.append(
            segment.model_copy(
                update={"text": text, "char_start": char_start, "char_end": char_end}
            )
        )

    return ir.model_copy(
        update={"segments": segments, "flat_text": "\n".join(s.text for s in segments)}
    )
