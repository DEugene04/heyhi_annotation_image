"""
Tests for the multi-blank FITB splitter (evaluator/short_answer/fitb_split.py).

The LLM call is replaced with a fake (`_complete`) so these run offline and
free. They cover the parts that actually carry logic: counting/marking blanks
from the answer spans, the single-blank no-op, verbatim pass-through, and -- the
important one -- refusing (not truncating) when the split count is wrong.

No pytest-asyncio in this project, so async calls are driven with asyncio.run.
"""

import asyncio
import json
import os

import pytest

from evaluator.short_answer.fitb_split import (
    BlankSplitError,
    count_blanks,
    mark_blanks,
    split_student_blanks,
)

# A minimal two-blank question in the same shape the real payload uses: each
# blank is a <span class="ans-div"> ... <ans> ... </span>.
TWO_BLANK_HTML = (
    '<p>The capital is '
    '<span class="ans-div" id="0"><label class="answer-tags"><ans></ans></label></span>'
    ' and the year was '
    '<span class="ans-div" id="1"><label class="answer-tags"><ans></ans></label></span>.</p>'
)


def _canned(answers):
    """An async _complete stand-in that returns a fixed answers list."""

    async def _complete(messages, api_key, model):
        return json.dumps({"answers": answers})

    return _complete


def _raising_if_called():
    """An async _complete stand-in that fails if the splitter calls the LLM."""

    async def _complete(messages, api_key, model):  # pragma: no cover - must not run
        raise AssertionError("LLM should not be called for a single blank")

    return _complete


# --- count_blanks / mark_blanks --------------------------------------------

def test_count_blanks_counts_answer_spans():
    assert count_blanks(TWO_BLANK_HTML) == 2


def test_count_blanks_on_real_payload_is_one():
    """The senior's reference payload (test.json) has exactly one blank.

    Documents the caller's boundary responsibility: the request's `question` is
    doubly-encoded -- a JSON string whose content is a JSON array of {type, text}
    segments -- so it must be decoded to clean HTML before the splitter sees it
    (the marking chain does the same via `json.loads(reqs.question)`). Handing the
    raw, still-escaped string to BeautifulSoup would see `class=\\"ans-div\\"` and
    silently find zero blanks.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "test.json")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    segments = json.loads(payload["question"])
    question_html = "\n".join(seg["text"] for seg in segments)
    assert count_blanks(question_html) == 1


def test_mark_blanks_numbers_spans_in_document_order():
    marked = mark_blanks(TWO_BLANK_HTML)
    assert "(blank 1)" in marked
    assert "(blank 2)" in marked
    # Order is preserved: blank 1 appears before blank 2 in the text.
    assert marked.index("(blank 1)") < marked.index("(blank 2)")
    # The answer spans are gone, replaced by the markers.
    assert "ans-div" not in marked


# --- split_student_blanks --------------------------------------------------

def test_single_blank_returns_whole_blob_without_llm():
    result = asyncio.run(
        split_student_blanks(
            "<p>irrelevant</p>",
            "sudden closure",
            num_blanks=1,
            _complete=_raising_if_called(),
        )
    )
    assert result == ["sudden closure"]


def test_multi_blank_returns_verbatim_answers():
    answers = ["Paris", "1789"]
    result = asyncio.run(
        split_student_blanks(
            TWO_BLANK_HTML,
            "Paris ... 1789",
            num_blanks=2,
            _complete=_canned(answers),
        )
    )
    assert result == answers


def test_empty_string_allowed_for_unanswered_blank():
    result = asyncio.run(
        split_student_blanks(
            TWO_BLANK_HTML,
            "Paris",
            num_blanks=2,
            _complete=_canned(["Paris", ""]),
        )
    )
    assert result == ["Paris", ""]


def test_count_mismatch_raises_rather_than_truncating():
    with pytest.raises(BlankSplitError):
        asyncio.run(
            split_student_blanks(
                TWO_BLANK_HTML,
                "Paris 1789",
                num_blanks=2,
                _complete=_canned(["Paris"]),  # only one -> must refuse
            )
        )


def test_unparseable_output_raises():
    async def _garbage(messages, api_key, model):
        return "not json at all"

    with pytest.raises(BlankSplitError):
        asyncio.run(
            split_student_blanks(
                TWO_BLANK_HTML, "x y", num_blanks=2, _complete=_garbage
            )
        )


def test_non_string_answers_raise():
    with pytest.raises(BlankSplitError):
        asyncio.run(
            split_student_blanks(
                TWO_BLANK_HTML,
                "x y",
                num_blanks=2,
                _complete=_canned(["Paris", 1789]),  # 1789 is an int
            )
        )


def test_zero_blanks_is_a_caller_bug():
    with pytest.raises(ValueError):
        asyncio.run(
            split_student_blanks(TWO_BLANK_HTML, "x", num_blanks=0)
        )
