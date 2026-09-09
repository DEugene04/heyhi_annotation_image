"""
Tests for the multi-blank FITB orchestration (evaluator/short_answer/fitb.py).

Both LLM-backed steps are faked (`_split`, `_mark_blank`) so these run offline:
we exercise the payload parsing, the per-blank fan-out, and the aggregation --
the parts that carry logic -- without touching OpenAI.

No pytest-asyncio in this project, so async calls are driven with asyncio.run.
"""

import asyncio
import json

from evaluator.short_answer.fitb import (
    FitbSpec,
    _blank_answers,
    _isolated_question,
    evaluate_fitb,
    maybe_parse_fitb,
)


def _segment(text):
    return {"type": "text", "text": text}


def _two_blank_payload():
    """A minimal two-blank FITB payload in the same shape as test.json.

    Blank markup is <span class="ans-div">...<ans></ans>...</span>; the correct
    answer segment is canonical `blank N :` with an alternative on blank 2.
    """
    question_html = (
        '<strong>Fill in the blanks.</strong><br />'
        'The capital of France is '
        '<span class="ans-div" id="0"><label class="answer-tags"><ans></ans></label></span>'
        ' and the revolution began in '
        '<span class="ans-div" id="1"><label class="answer-tags"><ans></ans></label></span>.'
    )
    segments = [
        _segment("question type: FITB Without Option (Vocab Cloze)"),
        _segment("question :"),
        _segment(question_html),
        _segment("correct answer:"),
        _segment("blank 1 : Paris\nblank 2 : 1789 / blank 2 : seventeen eighty-nine"),
        _segment("mark: 1"),
        _segment("Solution:"),
        _segment("Paris; 1789."),
    ]
    return json.dumps(segments)


# --- _blank_answers --------------------------------------------------------

def test_blank_answers_repeated_prefix_alternatives():
    raw = "blank 1 : Paris\nblank 2 : 1789 / blank 2 : seventeen eighty-nine"
    assert _blank_answers(raw) == ["Paris", "1789 / seventeen eighty-nine"]


def test_blank_answers_single_prefix_alternatives():
    raw = "blank 1 : Water / H2O"
    assert _blank_answers(raw) == ["Water / H2O"]


def test_blank_answers_space_separated_blanks():
    raw = "blank 1 : a blank 2 : b"
    assert _blank_answers(raw) == ["a", "b"]


def test_blank_answers_orders_by_blank_number():
    raw = "blank 2 : b\nblank 1 : a"
    assert _blank_answers(raw) == ["a", "b"]


# --- _isolated_question ----------------------------------------------------

def test_isolated_question_fills_other_blanks_keeps_target():
    q = "The capital is (blank 1) and the year was (blank 2)."
    answers = ["Paris", "1789 / seventeen eighty-nine"]
    # Marking blank 1: blank 2 filled with its first alternative, blank 1 kept.
    assert _isolated_question(q, answers, 0) == "The capital is (blank 1) and the year was 1789."
    # Marking blank 2: blank 1 filled, blank 2 kept.
    assert _isolated_question(q, answers, 1) == "The capital is Paris and the year was (blank 2)."


# --- maybe_parse_fitb ------------------------------------------------------

def test_parse_plain_text_question_is_not_fitb():
    assert asyncio.run(maybe_parse_fitb("Just a normal question?", None, 1.0)) is None


def test_parse_two_blank_payload():
    spec = asyncio.run(maybe_parse_fitb(_two_blank_payload(), None, 1.0))
    assert spec is not None
    assert spec.num_blanks == 2
    assert spec.per_blank_answers == ["Paris", "1789 / seventeen eighty-nine"]
    assert spec.blank_full_mark == 1.0
    # The answer spans are rewritten to position markers, and the answers do NOT
    # leak into the question text handed to the splitter.
    assert "(blank 1)" in spec.question and "(blank 2)" in spec.question
    assert "Paris" not in spec.question


def test_parse_real_single_blank_payload_stays_single(tmp_path):
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "test.json")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    spec = asyncio.run(maybe_parse_fitb(payload["question"], None, 1.0))
    # It IS FITB, but with a single blank -- evaluate() will not delegate.
    assert spec is not None
    assert spec.num_blanks == 1


def _span_with_answer(answer):
    """An ans-div span carrying its correct answer inside, like the real payload."""
    return (
        '<span class="ans-div"><label class="answer-tags"><ans></label>'
        f"{answer}"
        '<label class="answer-tags"></ans></label></span>'
    )


def test_parse_leaner_entry_takes_answers_from_spans():
    """The entry-point shape: <ans> spans carry the answers, and there is NO
    `correct answer:` segment and NO `Solution:` segment. Answers must come from
    the span contents, and the answers must not leak into the question text."""
    question_html = (
        "The capital of France is " + _span_with_answer("Paris") +
        " and the revolution began in " + _span_with_answer("1789") + "."
    )
    segments = [
        _segment("question type: FITB Without Option (Vocab Cloze)"),
        _segment("question :"),
        _segment(question_html),
        _segment("mark: 1"),
    ]
    spec = asyncio.run(maybe_parse_fitb(json.dumps(segments), None, 1.0))
    assert spec is not None
    assert spec.num_blanks == 2
    assert spec.per_blank_answers == ["Paris", "1789"]
    assert spec.blank_full_mark == 1.0
    assert "(blank 1)" in spec.question and "(blank 2)" in spec.question
    # The answers came from the spans but were stripped out of the question text.
    assert "Paris" not in spec.question and "1789" not in spec.question


def test_parse_spans_without_type_label_still_fitb():
    """Detection falls back to the presence of answer spans when the
    `question type:` label is absent."""
    question_html = "Fill: " + _span_with_answer("Paris") + "."
    segments = [_segment("question :"), _segment(question_html), _segment("mark: 1")]
    spec = asyncio.run(maybe_parse_fitb(json.dumps(segments), None, 1.0))
    assert spec is not None and spec.num_blanks == 1
    assert spec.per_blank_answers == ["Paris"]


def test_parse_payload_without_solution_segment_still_fitb():
    """A payload with a `correct answer:` segment but NO `Solution:` segment must
    still parse (the old extract_question_obj path crashed here and the error was
    silently swallowed, mis-marking it as non-FITB)."""
    segments = [
        _segment("question type: FITB Without Option (Vocab Cloze)"),
        _segment("question :"),
        _segment("Answer: " + '<span class="ans-div"><ans></ans></span>'),
        _segment("correct answer:"),
        _segment("blank 1 : Bridge closures and travel delays"),
        _segment("mark: 1"),
    ]
    spec = asyncio.run(maybe_parse_fitb(json.dumps(segments), None, 1.0))
    assert spec is not None
    assert spec.num_blanks == 1
    assert spec.per_blank_answers == ["Bridge closures and travel delays"]


def _photosynthesis_payload():
    """The real 2-blank science payload (senior's API-automarking shape).

    Spans carry the answer inside; a `correct answer:` segment restates them; each
    blank is worth 2 (`mark: 2.0`, matching `answer_pas` "... (1) ... (1)"), so the
    question total is 4; and the `Solution:` segment is empty.
    """
    q = (
        "During photosynthesis, green leaves use light energy to take in "
        '<span class="ans-div" id="0"><label class="answer-tags"><ans></label>'
        "carbon dioxide and water"
        '<label class="answer-tags"></ans></label></span>'
        " to make "
        '<span class="ans-div" id="1"><label class="answer-tags"><ans></label>'
        "sugar and oxygen"
        '<label class="answer-tags"></ans></label></span> as a product.'
    )
    segments = [
        _segment("question type: FITB Without Option (Vocab Cloze)"),
        _segment("question header:"),
        _segment("question :"),
        _segment(q),
        _segment("answer option:"),
        _segment("correct answer:"),
        _segment("blank 1 : carbon dioxide and water"),
        _segment("blank 2 : sugar and oxygen"),
        _segment("difficulty_level: Easy"),
        _segment("mark: 2.0"),
        _segment("Solution:"),
    ]
    return json.dumps(segments)


def test_parse_real_two_blank_science_payload():
    spec = asyncio.run(maybe_parse_fitb(_photosynthesis_payload(), None, 2.0))
    assert spec is not None
    assert spec.num_blanks == 2
    assert spec.per_blank_answers == ["carbon dioxide and water", "sugar and oxygen"]
    # `mark: 2.0` is the PER-BLANK max (each blank worth 2); the question total is
    # the sum across blanks (4). The answers are stripped out of the question text.
    assert spec.blank_full_mark == 2.0
    assert "(blank 1)" in spec.question and "(blank 2)" in spec.question
    assert "carbon dioxide" not in spec.question and "sugar" not in spec.question


def test_answer_key_overrides_payload_correct_answer():
    spec = asyncio.run(
        maybe_parse_fitb(
            _two_blank_payload(),
            "blank 1 : Lyon\nblank 2 : 1800",
            1.0,
        )
    )
    assert spec.per_blank_answers == ["Lyon", "1800"]


# --- evaluate_fitb ---------------------------------------------------------

def _fake_split(pieces):
    async def _split(question, flat_ocr_text, num_blanks, *, api_key=None):
        return pieces

    return _split


def _recording_marker(per_blank_result):
    """A fake evaluate() that records its per-blank calls and returns canned marks."""
    calls = []

    async def _mark(question, student_answer, answer_key, **kwargs):
        calls.append(
            {"student_answer": student_answer, "answer_key": answer_key, "kwargs": kwargs}
        )
        return per_blank_result(student_answer, answer_key, kwargs)

    return _mark, calls


def test_evaluate_fitb_sums_marks_and_concatenates_highlights():
    spec = FitbSpec(
        question="The capital of France is (blank 1) and the revolution began in (blank 2).",
        per_blank_answers=["Paris", "1789"],
        num_blanks=2,
        blank_full_mark=1.0,
    )

    def per_blank(student_answer, answer_key, kwargs):
        # blank 1 right (1.0), blank 2 wrong (0.0)
        mark = 1.0 if student_answer == "Paris" else 0.0
        status = "correct" if mark else "incorrect"
        return {
            "mark": mark,
            "reason": f"reason for {student_answer}",
            "remarks": f"remark for {student_answer}",
            "remarks_detailed": f"detailed for {student_answer}",
            "mark_highlights": [
                {"target": student_answer, "status": status, "remark": "r"}
            ],
            "total_tokens": 10,
            "total_cost": 0.001,
            "models_used": ["gpt-4.1-mini"],
        }

    marker, calls = _recording_marker(per_blank)
    result = asyncio.run(
        evaluate_fitb(
            spec,
            "Paris 1690",
            _split=_fake_split(["Paris", "1690"]),
            _mark_blank=marker,
        )
    )

    # One marking call per blank, each with its own answer key and student piece.
    assert [c["student_answer"] for c in calls] == ["Paris", "1690"]
    assert [c["answer_key"] for c in calls] == ["Paris", "1789"]
    # Each blank framed by its own per-blank full mark.
    assert all(c["kwargs"]["full_mark"] == 1.0 for c in calls)

    # Marks summed; highlights concatenated in blank order.
    assert result["mark"] == 1.0
    assert [h["target"] for h in result["mark_highlights"]] == ["Paris", "1690"]
    assert result["total_tokens"] == 20
    assert "Blank 1:" in result["remarks"] and "Blank 2:" in result["remarks"]


def test_evaluate_fitb_split_error_propagates():
    from evaluator.short_answer.fitb_split import BlankSplitError

    spec = FitbSpec(
        question="(blank 1) (blank 2)",
        per_blank_answers=["a", "b"],
        num_blanks=2,
        blank_full_mark=1.0,
    )

    async def _bad_split(question, flat_ocr_text, num_blanks, *, api_key=None):
        raise BlankSplitError("could not split")

    async def _never(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("marking must not run when the split fails")

    try:
        asyncio.run(evaluate_fitb(spec, "blob", _split=_bad_split, _mark_blank=_never))
        assert False, "expected BlankSplitError"
    except BlankSplitError:
        pass
