"""
Translate the evaluator's verbatim-text highlights into positioned feedback.

This is the adapter that contracts/evaluator_contract.md calls "the open
question", and that geometry/resolver.py notes is on hold: the evaluator points
at a span by its exact text (`target`), but the geometry layer needs a character
range into the reading's `flat_text`. This module bridges the two by locating
each target in `flat_text`.

Matching is whitespace-tolerant. `flat_text` joins the reading's lines with
newlines, but the evaluator returns each sentence with those breaks flattened to
spaces, so an exact substring search misses almost every sentence that wraps
across a line. We match the target's words separated by any run of whitespace,
which lets a sentence span line breaks and still resolve to the lines it covers.

Two known hazards, both handled by degrading gracefully rather than guessing:

  - The same text can appear more than once. Highlights arrive in reading order,
    so we walk a cursor forward through `flat_text`; repeated targets map to
    successive occurrences instead of all landing on the first.
  - For maths the evaluator rewrites formatting, so a target may not appear in
    `flat_text` even up to whitespace. A target we cannot find becomes a
    panel-only item (no span), which the resolver shows in the panel with nothing
    drawn on the image.
"""

import re
from typing import List, Optional, Tuple

from contracts.schema import Category, Feedback

# The evaluator's per-span status vocabulary mapped onto our feedback categories.
# "partially_correct" has no exact category; it is neither right nor plainly
# wrong, so it maps to the neutral INFO note.
_STATUS_TO_CATEGORY = {
    "correct": Category.CORRECT,
    "incorrect": Category.ERROR,
    "partially_correct": Category.INFO,
}


def _locate(
    flat_text: str, target: str, cursor: int
) -> Tuple[Optional[int], Optional[int]]:
    """Find `target` in `flat_text` at or after `cursor`, tolerating whitespace.

    The target's words must appear in order, separated by any run of whitespace,
    so a sentence that wraps across newlines in `flat_text` still matches and
    spans those lines. Searches from `cursor` first, then falls back to a search
    from the start (a highlight may legitimately point behind the cursor), and
    returns (None, None) when the words are not present at all -- the signal for a
    panel-only item.
    """

    words = target.split()
    if not words:
        return None, None

    pattern = re.compile(r"\s+".join(re.escape(word) for word in words))
    match = pattern.search(flat_text, cursor) or pattern.search(flat_text)
    if match is None:
        return None, None
    return match.start(), match.end()


def to_feedback(flat_text: str, evaluation: dict) -> List[Feedback]:
    """Convert an `evaluate()` result into positioned `Feedback` items.

    Each `mark_highlight` becomes one feedback item anchored to its span; the
    overall `remarks` become a final panel-only note carrying the summary.
    """

    feedback: List[Feedback] = []

    cursor = 0
    for highlight in evaluation.get("mark_highlights", []):
        target = highlight.get("target", "")
        start, end = _locate(flat_text, target, cursor)
        if end is not None:
            cursor = end
        feedback.append(
            Feedback(
                comment=highlight.get("remark", ""),
                category=_STATUS_TO_CATEGORY.get(
                    highlight.get("status", ""), Category.INFO
                ),
                char_start=start,
                char_end=end,
            )
        )

    overall = evaluation.get("remarks")
    if overall:
        feedback.append(Feedback(comment=overall, category=Category.INFO))

    return feedback
