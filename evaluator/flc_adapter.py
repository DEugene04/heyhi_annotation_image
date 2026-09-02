"""
Turn FLC findings into positioned feedback the geometry layer can draw.

Each FLC finding already carries character offsets (`start`/`end`) into the exact
text we sent it -- which is `flat_text` -- so in the common case we use them
directly, and the existing geometry layer (`resolve_regions`) turns the char
range into polygons. No new polygon logic is needed.

We defend against drift: if `flat_text[start:end]` does not match the finding's
`target` (e.g. FLC normalised whitespace before indexing), we fall back to the
short-answer adapter's whitespace-tolerant locator to find the target instead. A
target we cannot place at all becomes a panel-only item, never dropped.

All findings map to ERROR (red): spelling and grammar are objective corrections.
"""

import logging
from typing import List

from contracts.schema import Category, Feedback
from evaluator.flc import FlcFinding
from evaluator.short_answer.adapter import _locate

logger = logging.getLogger(__name__)


def to_feedback(flat_text: str, findings: List[FlcFinding]) -> List[Feedback]:
    """Convert FLC `checking[]` findings into red `Feedback` items."""

    feedback: List[Feedback] = []
    for finding in findings:
        target = finding.get("target", "")
        start = finding.get("start")
        end = finding.get("end")

        # Trust FLC's offsets only when the slice they point at actually matches
        # the target; otherwise re-locate the target in flat_text.
        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(flat_text)
            and flat_text[start:end] == target
        ):
            start, end = _locate(flat_text, target, 0)

        feedback.append(
            Feedback(
                comment=finding.get("suggestion", ""),
                category=Category.ERROR,
                char_start=start,
                char_end=end,
            )
        )
    return feedback
