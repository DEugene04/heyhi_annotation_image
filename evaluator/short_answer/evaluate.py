"""
Simplified entry point for the automarking evaluator.

The original evaluator (from the smartjen `Modules.automarking` service) accepted
a full worksheet JSON and pulled the correct answer, marks, rubric, images and
solution out of it. In this project the evaluator is an internal pipeline stage
that judges a single handwritten answer, so it only needs three things:

  - the question text,
  - the student's answer (already extracted to plain text upstream), and
  - an optional answer key -- a ground-truth answer, marking note, or rubric hint
    -- to guide scoring.

The LLM is responsible for producing the score; the answer key, when given, only
helps it. The OpenAI API key is read from the environment via `config.settings`
(the same source the rest of the pipeline uses), so callers don't pass it.

This bypasses `QuestionExtractor.constructQuestionContext`, which only knows how
to parse the smartjen worksheet JSON, and builds the marking context directly.
"""

from typing import List, Optional, Union

import config
from evaluator.short_answer.chain import AutoMarkingChain
from evaluator.short_answer.fitb import evaluate_fitb, maybe_parse_fitb
from evaluator.short_answer.logic import (
    AnswerContext,
    MarkingResultContext,
    QuestionContext,
    QuestionMarkContext,
    RubricContext,
)

# The marking chain always needs a maximum score to frame the prompt; the LLM
# allocates the actual mark within it. Callers who know the real maximum can
# override this, but it is not part of the core three-input contract.
DEFAULT_FULL_MARK = 1.0


def _build_question_context(
    question: str,
    student_answer: Union[str, List[str]],
    answer_key: Optional[str],
    full_mark: float,
    subject: str,
    language: str,
    marking_note: str,
    rubric: str,
) -> QuestionContext:
    """Assemble the context the marking chain expects from the simplified inputs.

    An open-ended, non-FITB question: `type_fitb` is False and `fitb_index` is an
    int (not a list), which routes the chain down its single-answer path. A null
    answer key becomes an empty correct answer -- the chain then leans on the LLM
    to judge the answer on its own merits.
    """

    return QuestionContext(
        question=question,
        string_urls=[],
        incorrect_flags=[],
        marking_note=marking_note,
        subject=subject,
        language=language,
        mark_context=QuestionMarkContext(
            full_mark=full_mark,
            step="0.5",
            type_fitb=False,
            fitb_index=0,          # int (not list) -> single-answer marking path
            num_blanks=0,
            unique_blanks=0,
        ),
        answer_context=AnswerContext(
            correct_answer=answer_key or "",
            original_correct_answer="",
            answer_options=answer_key or "",
            pas_answer="",
            student_answer=student_answer,
            student_image_urls=None,
            solution="",
        ),
        # The prompt injects `string_rubric` verbatim, so a plain-text rubric
        # works directly -- no JSON parsing (unlike the smartjen path).
        rubric_context=RubricContext(
            rubric=rubric, rubric_json=[], string_rubric=rubric
        ),
    )


async def evaluate(
    question: str,
    student_answer: Union[str, List[str]],
    answer_key: Optional[str] = None,
    *,
    full_mark: float = DEFAULT_FULL_MARK,
    subject: str = "",
    language: str = "English",
    marking_note: str = "",
    rubric: str = "",
    api_key: Optional[str] = None,
    debug_mode: bool = False,
) -> dict:
    """Judge one student's answer and return the mark plus a feedback breakdown.

    Args:
        question: The question text.
        student_answer: The extracted student answer (a string, or a list of
            strings for a multi-part answer).
        answer_key: Optional ground-truth answer to guide scoring. When None, the
            LLM judges the answer without a reference.
        full_mark: Maximum score used to frame the prompt (default 1.0). The LLM
            allocates the actual mark within it.
        subject / language: Optional extra guidance passed through to the prompt.
        marking_note: Optional question-specific marking instructions/exceptions.
        rubric: Optional plain-text rubric (scoring criteria) to follow.
        api_key: OpenAI key. Defaults to the environment (`config.settings`).
        debug_mode: When True, the exact LLM message lists for both stages are
            included in the result under `messages_automarking` (the marking call)
            and `messages_breakdown` (the highlight/rephrase call), for logging.

    Returns:
        A dict with `mark`, `reason`, `remarks`, `remarks_detailed`,
        `mark_highlights`, and usage totals (`total_tokens`, `total_cost`,
        `models_used`). With `debug_mode`, also `messages_automarking` and
        `messages_breakdown` -- the verbatim prompts sent to each stage.
    """

    api_key = (
        api_key
        or config.settings.openai_api_key
        or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"]
    )

    # Multi-blank fill-in-the-blank: split the one OCR blob into a per-blank
    # answer and mark each blank on this same single-answer path, then sum. A
    # single-blank FITB (or a non-FITB question) parses to None or one blank and
    # falls straight through to the unchanged path below.
    if isinstance(student_answer, str):
        fitb_spec = await maybe_parse_fitb(question, answer_key, full_mark)
        if fitb_spec is not None and fitb_spec.num_blanks >= 2:
            return await evaluate_fitb(
                fitb_spec,
                student_answer,
                subject=subject,
                language=language,
                marking_note=marking_note,
                rubric=rubric,
                api_key=api_key,
                debug_mode=debug_mode,
            )

    question_context = _build_question_context(
        question=question,
        student_answer=student_answer,
        answer_key=answer_key,
        full_mark=full_mark,
        subject=subject,
        language=language,
        marking_note=marking_note,
        rubric=rubric,
    )

    # 1) Numeric mark + developer-facing reasoning. `reqs` is only consulted on the
    #    multi-answer FITB path, which this single-answer context never takes.
    result, mark_tokens, mark_cost, mark_models = await AutoMarkingChain.get_result(
        question_context=question_context,
        reqs=None,
        debug_mode=debug_mode,
        auto_fail=False,
        api_key=api_key,
    )

    # 2) Student-facing remarks + per-span correct/incorrect highlights.
    marking_result_context = MarkingResultContext(
        ai_mark=result["mark"],
        reason=result["reason"],
    )

    highlights, hl_tokens, hl_cost, hl_model = await AutoMarkingChain.get_automarking_breakdown(
        question_context=question_context,
        marking_result_context=marking_result_context,
        debug_mode=debug_mode,
        auto_fail=False,
        api_key=api_key,
    )

    models = set(mark_models)
    models.add(hl_model)

    output = {
        "mark": result["mark"],
        "reason": result["reason"],
        "remarks": highlights["remarks_simplified"],
        "remarks_detailed": highlights["remarks"],
        "mark_highlights": highlights["mark_highlights"],
        "total_tokens": mark_tokens + hl_tokens,
        "total_cost": mark_cost + hl_cost,
        "models_used": sorted(models),
    }

    if debug_mode:
        # The verbatim prompts each stage sent, surfaced for logging so the exact
        # text (and spelling) the evaluator saw can be inspected.
        output["messages_automarking"] = result.get("messages_automarking")
        output["messages_breakdown"] = highlights.get("messages_breakdown")

    return output
