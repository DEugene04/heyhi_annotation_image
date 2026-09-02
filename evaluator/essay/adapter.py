"""
Translate the essay evaluator's output into the pipeline's payload shapes.

The essay (compo) marker produces two things the annotation pipeline can use:

  - a set of *strength* excerpts (from `generate_good_points`), each an exact
    quote from the essay with feedback -- these become on-image highlights, the
    first-cut equivalent of the short-answer path's `mark_highlights`; and
  - per-criterion rubric results (Content, Organization, Language ...), which are
    holistic (not tied to a span) and become the panel scorecard.

First cut: only strengths are drawn on the page (category CORRECT). Weaknesses
and the per-criterion feedback live in the scorecard panel until we validate.

Excerpts are located back into `flat_text` with the same whitespace-tolerant
matcher the short-answer adapter uses, so a quote that wraps across lines still
resolves; an excerpt that cannot be found becomes a panel-only item (no span),
never dropped.
"""

from typing import List

from contracts.schema import Category, CriterionScore, Feedback
from evaluator.short_answer.adapter import _locate


def strengths_to_feedback(flat_text: str, good_points: List[dict]) -> List[Feedback]:
    """Turn strength excerpts into positioned CORRECT feedback items.

    Each good point is `{"excerpt": <exact quote>, "feedback": <why it's good>}`.
    We walk a cursor forward through `flat_text` so repeated excerpts map to
    successive occurrences rather than all landing on the first.
    """

    feedback: List[Feedback] = []
    cursor = 0
    for point in good_points:
        excerpt = point.get("excerpt", "")
        start, end = _locate(flat_text, excerpt, cursor)
        if end is not None:
            cursor = end
        feedback.append(
            Feedback(
                comment=point.get("feedback", ""),
                category=Category.CORRECT,
                char_start=start,
                char_end=end,
            )
        )
    return feedback


def score_detail_to_scorecard(
    score_detail: List[dict], rubric_table: List[dict]
) -> List[CriterionScore]:
    """Turn the marker's per-criterion results into panel scorecard entries.

    The marker's `score_detail` item carries the component name, score, and
    feedback but not the criterion's maximum, so we look the max up from the
    rubric table by name.
    """

    max_by_name = {c["name"]: c["max_score"] for c in rubric_table}
    scorecard: List[CriterionScore] = []
    for detail in score_detail:
        if detail is None:
            continue
        name = detail.get("component", "")
        scorecard.append(
            CriterionScore(
                name=name,
                score=detail.get("score", 0),
                max_score=max_by_name.get(name, 0),
                feedback=detail.get("feedback", ""),
                feedback_detailed=detail.get("feedback_detailed", "") or "",
            )
        )
    return scorecard
