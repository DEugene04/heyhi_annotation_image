"""
Combines the VLM's accurate text with Mathpix's accurate shapes.

Mathpix gives us one segment per line (or maths row), each with a shape on the
image. The VLM gives us the text of each line, read far more accurately. This
lays the VLM's text onto Mathpix's shapes: each Mathpix segment keeps its shape
and position but takes the VLM's reading as its text.

Both lists are in reading order, so they are matched line by line. The character
ranges and the continuous text are rebuilt from the new text, so feedback that
points at a span still lands on the right shape.
"""

from typing import List

from contracts.schema import IR, Segment


def fuse(ir: IR, vlm_lines: List[str]) -> IR:
    """Return a new IR with Mathpix's shapes carrying the VLM's text."""

    segments: List[Segment] = []
    offset = 0
    for index, segment in enumerate(ir.segments):
        text = vlm_lines[index] if index < len(vlm_lines) else segment.text
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


def alignment_ok(ir: IR, vlm_lines: List[str]) -> bool:
    """
    Whether the two readings line up one-to-one.

    When Mathpix and the VLM split the page into a different number of lines, a
    plain line-by-line match will drift. This flags that case so we can fall back
    or refine, rather than silently mislabel shapes.
    """

    return len(ir.segments) == len(vlm_lines)
