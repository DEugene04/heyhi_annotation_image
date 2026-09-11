"""
Tests for the /annotate route's wiring.

The two readers (Mathpix and GPT-5) and the evaluator are replaced with fakes so
the route can be exercised without calling the live services. We check the three
outcomes: a page the readers agree on is annotated, a page they disagree on is
refused with the retake message, and a page read too unclearly is refused too.
"""

import pytest

import main
from contracts.schema import IR, ImageMeta, Point, Segment, SegmentType
from fastapi.testclient import TestClient
from ocr.fusion import RETAKE_WARNING

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _isolate_logs(tmp_path, monkeypatch):
    """Send each test's /annotate log file to a temp dir instead of the real logs/.

    `_annotate_log_file` opens a FileHandler under `main.LOG_DIR` on every request,
    so a plain test run would otherwise litter logs/short_answer/ with 0-byte files
    (pytest captures the log records, so they never reach the handler). Redirecting
    LOG_DIR per test keeps the real logs/ clean.
    """

    monkeypatch.setattr(main, "LOG_DIR", tmp_path)


def _segment(id, line, text):
    y = line * 10
    quad = [Point(x=0, y=y), Point(x=70, y=y), Point(x=70, y=y + 8), Point(x=0, y=y + 8)]
    return Segment(id=id, line=line, type=SegmentType.TEXT, text=text, quad=quad,
                   char_start=0, char_end=len(text), confidence=0.9, is_handwritten=True)


def _fake_ir():
    return IR(
        image=ImageMeta(width=100, height=100),
        flat_text="teh cat\nsat down",
        segments=[_segment("s0", 0, "teh cat"), _segment("s1", 1, "sat down")],
    )


def _post():
    return client.post(
        "/annotate",
        files={"photo": ("a.png", b"bytes", "image/png")},
        data={"question": "Spell the sentence."},
    )


async def _fake_evaluate(question, student_answer, answer_key=None, **kwargs):
    """Stand in for the evaluator: a correct highlight plus an overall remark."""

    return {
        "mark": 1.0,
        "mark_highlights": [
            {"target": "the cat", "status": "correct", "remark": "Right."},
        ],
        "remarks": "Overall: good.",
    }


async def _fake_flc_check(student_composition, question_statement, **kwargs):
    """Stand in for the FLC service so tests never make a real network call."""

    return []


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_annotate_returns_a_payload_when_the_readers_agree(monkeypatch):
    monkeypatch.setattr(main, "build_ir", lambda _: _fake_ir())
    monkeypatch.setattr(main, "transcribe_lines", lambda *_: ["the cat", "sat down"])
    monkeypatch.setattr(main, "evaluate", _fake_evaluate)
    monkeypatch.setattr(main.flc, "check", _fake_flc_check)
    response = _post()
    assert response.status_code == 200
    body = response.json()
    assert body["image"] == {"width": 100, "height": 100}
    assert len(body["items"]) >= 1


def test_annotate_refuses_when_the_readers_disagree(monkeypatch):
    monkeypatch.setattr(main, "build_ir", lambda _: _fake_ir())
    monkeypatch.setattr(main, "transcribe_lines", lambda *_: ["only one line"])
    response = _post()
    assert response.status_code == main.RETAKE_STATUS
    assert response.json()["detail"] == RETAKE_WARNING


def test_annotate_refuses_when_the_reading_is_unclear(monkeypatch):
    monkeypatch.setattr(main, "build_ir", lambda _: _fake_ir())
    monkeypatch.setattr(main, "transcribe_lines", lambda *_: ["the cat", "sat down"])
    monkeypatch.setattr(main, "is_readable", lambda _: False)
    response = _post()
    assert response.status_code == main.RETAKE_STATUS
    assert response.json()["detail"] == RETAKE_WARNING