"""
Simplified entry point for the essay (compo) evaluator.

Mirrors evaluator/short_answer/evaluate.py: it hides the compo marking chain's
wide, structured interface behind a small call the pipeline can make with the
same inputs it already has (a question and the read-back answer text), and
returns one dict the essay adapter can turn into feedback + a scorecard.

Two compo-specific inputs are handled here so callers don't have to:

  - the rubric. Compo *requires* a structured rubric table (criteria with scoring
    bands). In production the frontend sends the teacher's chosen template; for
    now this accepts an optional `rubric_table` dict and falls back to a
    hardcoded default when none is given.
  - `model_composition`. Compo's signature requires it, but it only feeds a word
    count that the marking never uses (it does not influence the score), so we
    pass an empty string.
"""

from typing import List, Optional

import config
from evaluator.essay.compo_marking import compo_marking_dependent_rubric_switching

# Default argumentative-essay rubric, used when the caller supplies none. Each
# criterion has an id, a name, a max score, a score_type, and banded breakdowns
# (each band needs an id). Replaced at request time by the chosen frontend
# template once that is wired up.
DEFAULT_RUBRIC_TABLE: List[dict] = [
    {
        "id": 1,
        "name": "Content & Ideas",
        "max_score": 10,
        "score_type": "range",
        "breakdown": [
            {"id": 11, "from_score": 0, "to_score": 3,
             "description": "Ideas are minimal, off-topic, or unsupported; the position is unclear."},
            {"id": 12, "from_score": 4, "to_score": 6,
             "description": "A clear position with some relevant reasons/examples, but ideas are thin or underdeveloped."},
            {"id": 13, "from_score": 7, "to_score": 10,
             "description": "A clear, well-argued position with developed, well-supported ideas and balanced reasoning."},
        ],
    },
    {
        "id": 2,
        "name": "Organization",
        "max_score": 5,
        "score_type": "range",
        "breakdown": [
            {"id": 21, "from_score": 0, "to_score": 2,
             "description": "Little structure; ideas are not grouped or linked; no clear opening or conclusion."},
            {"id": 22, "from_score": 3, "to_score": 4,
             "description": "Generally organised into paragraphs with some linking, but transitions or a conclusion may be weak or missing."},
            {"id": 23, "from_score": 5, "to_score": 5,
             "description": "Clear, logical structure with an introduction, well-linked body paragraphs, and a conclusion."},
        ],
    },
    {
        "id": 3,
        "name": "Language",
        "max_score": 5,
        "score_type": "range",
        "breakdown": [
            {"id": 31, "from_score": 0, "to_score": 2,
             "description": "Frequent errors in spelling, grammar, or word choice that impede meaning."},
            {"id": 32, "from_score": 3, "to_score": 4,
             "description": "Generally accurate with some errors in spelling, grammar, or word choice that do not impede meaning."},
            {"id": 33, "from_score": 5, "to_score": 5,
             "description": "Accurate, varied, and fluent language with very few errors."},
        ],
    },
]


async def evaluate_essay(
    question: str,
    student_answer: str,
    rubric_table: Optional[List[dict]] = None,
    *,
    student_class: str = "Secondary 2",
    language: str = "English",
    api_key: Optional[str] = None,
) -> dict:
    """Mark one essay and return the marking plus strength excerpts.

    Args:
        question: The essay question / prompt.
        student_answer: The read-back essay text (the pipeline's flat_text).
        rubric_table: The structured rubric to mark against. When None, the
            hardcoded DEFAULT_RUBRIC_TABLE is used.
        student_class / language: Extra context the compo chain needs.
        api_key: OpenAI key; defaults to the environment via config.

    Returns:
        A dict with `mark`, `feedback`, `feedback_detailed`, `score_detail`
        (per-criterion results), the `rubric_table` used, and usage totals. The
        on-page highlights come from FLC, so no strength excerpts are produced.
    """

    api_key = api_key or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"]
    rubric_table = rubric_table or DEFAULT_RUBRIC_TABLE

    # 1) Mark against the rubric. model_composition is required by the signature
    #    but does not affect the score, so it is left empty.
    marking = await compo_marking_dependent_rubric_switching(
        question_statement=question,
        rubric_table=rubric_table,
        student_composition=student_answer,
        model_composition="",
        student_class=student_class,
        language=language,
        api_key=api_key,
    )
    feedback = marking["feedback"]

    # The on-page highlights for essays come from FLC (spelling/grammar), not from
    # the compo marker's "good points" strengths, so we no longer request them --
    # this also saves an LLM call per essay.
    return {
        "mark": feedback["mark"],
        "feedback": feedback["feedback"],
        "feedback_detailed": feedback["feedback_detailed"],
        "score_detail": feedback["score_detail"],
        "rubric_table": rubric_table,
        "total_tokens": marking["total_tokens"],
        "total_cost": marking["total_cost"],
        "models_used": sorted(set(marking["models"])),
    }
