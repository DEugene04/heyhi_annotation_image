"""
The data shapes every layer of the pipeline reads and writes.

Two shapes live here, because both are our own internal structures:

  - The Intermediate Representation (IR): what the OCR layer produces after
    reading a cleaned image. Everything downstream reads from the IR, never
    from the OCR provider directly.

  - The Annotation Payload: what the backend hands the frontend to draw. It
    holds the regions to draw on the image plus the feedback each region carries.

The evaluator's own contract is kept separately (see evaluator_contract.md),
because that one is a specification we hand to the organization, not code we run.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class Point(BaseModel):
    """A single (x, y) location on the image, measured in pixels."""

    x: float
    y: float


# A quad is the four-corner shape that follows a piece of writing on the page.
#
# For a line-shape in the IR it is exactly four points, in this order:
#   top-left, top-right, bottom-right, bottom-left (clockwise from the top-left).
# Keeping this order consistent is what lets the geometry layer find a shape's
# edges without guessing; the IR builder is responsible for emitting quads this
# way. A region drawn for a run of several lines may hold more than four points,
# because it traces the outline of all those lines at once.
Quad = List[Point]


class SegmentType(str, Enum):
    """What kind of content a line of writing holds."""

    TEXT = "text"        # ordinary words
    MATH = "math"        # an equation or maths expression
    DIAGRAM = "diagram"  # a drawn figure, handled as a whole


# ---------------------------------------------------------------------------
# Intermediate Representation (IR) — produced by the OCR + IR layer
# ---------------------------------------------------------------------------

class Word(BaseModel):
    """
    A single word (or, in maths, a single sub-expression) within a line.

    Each word carries its own shape on the image. Keeping these lets feedback
    be drawn tightly around exactly the part it refers to — a single wrong word
    in a sentence, rather than the whole line.
    """

    text: str
    quad: Quad


class Segment(BaseModel):
    """
    One line of writing that the OCR read from the page.

    Alongside the text itself, each segment records exactly where it sits on
    the image (the quad) and where it sits inside the continuous answer text
    (the character range). That character range is what lets a piece of
    feedback, which points at a span of the answer text, be traced back to a
    shape on the image. The words within the line are kept too, so feedback can
    be drawn around just part of the line when it needs to be.
    """

    id: str                         # our own stable identifier for this line
    line: int                       # line number on the page, from the top
    type: SegmentType               # words, maths, or a diagram
    text: str                       # the plain-text reading of the line
    latex: Optional[str] = None     # maths notation, when the line is maths
    quad: Quad                      # the shape on the image this line occupies
    char_start: int                 # index in flat_text where this line begins
    char_end: int                   # index just past this line's last character
    confidence: float               # how sure the OCR was, from 0 to 1
    is_handwritten: bool            # student's own writing, vs printed question
    words: List[Word] = Field(default_factory=list)  # the words within the line


class ImageMeta(BaseModel):
    """Facts about the cleaned image the annotations are drawn on."""

    width: int
    height: int


class IR(BaseModel):
    """
    The complete structured reading of one answer page.

    `flat_text` is every line joined into one continuous string with the line
    breaks kept, so feedback can point at a character range across the whole
    answer. Each segment's char_start / char_end index into this same string.

    `diagrams` holds any drawn figures found on the page. They are kept apart
    from the written lines because they are not text and are annotated as whole
    figures, not read character by character.
    """

    segments: List[Segment]
    flat_text: str
    image: ImageMeta
    diagrams: List[Segment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Feedback categories — shared by the evaluator output and the payload
# ---------------------------------------------------------------------------

class Category(str, Enum):
    """The kind of remark a piece of feedback makes about the work."""

    CORRECT = "correct"  # this part is right
    ERROR = "error"      # this part is wrong
    INFO = "info"        # a neutral note (for example, "you skipped a step")
    CARRIED = "carried"  # a mistake carried forward from an earlier line


# ---------------------------------------------------------------------------
# Feedback — one remark as it arrives from the evaluator, before it has a shape
# ---------------------------------------------------------------------------

class Feedback(BaseModel):
    """
    A single piece of feedback as the evaluator produces it.

    It carries the comment, its category, and the span of the answer it refers
    to, given as a character range in flat_text. The range is optional: feedback
    that refers to no particular place (for example, "you skipped a step") leaves
    both ends empty and is shown in the panel on its own.

    This is the shape the geometry layer reads. While the evaluator is on hold,
    these come from fixtures instead.
    """

    comment: str
    category: Category
    char_start: Optional[int] = None
    char_end: Optional[int] = None


# ---------------------------------------------------------------------------
# Annotation Payload — produced by the geometry layer, consumed by the frontend
# ---------------------------------------------------------------------------

class Region(BaseModel):
    """
    One shape to draw on the image for a piece of feedback.

    A single feedback item may own several regions (for example, an error that
    runs across two separate lines), so regions are always owned by an item.
    """

    quad: Quad


class AnnotationItem(BaseModel):
    """
    One piece of feedback as the frontend receives it.

    It carries the comment to show and the category to tint by. It may own
    several regions, one region, or none at all. An item with no regions is
    valid: it is feedback that could not be tied to a place on the page (for
    example, "you skipped a step") and is listed in the panel on its own.
    """

    id: str
    category: Category
    comment: str
    regions: List[Region] = Field(default_factory=list)


class CriterionScore(BaseModel):
    """
    One rubric criterion's result, shown in the side panel (not on the image).

    Essay marking scores against a rubric of criteria (for example Content,
    Organization, Language), each with its own mark out of a maximum and its own
    feedback. Unlike a Feedback item, this is not tied to a span on the page — it
    is a holistic judgement about the whole answer against that criterion, so it
    lives in the panel as a scorecard. The short-answer path produces none of
    these; only the essay path fills the scorecard.
    """

    name: str                       # the criterion, e.g. "Content & Ideas"
    score: float                    # the mark awarded for this criterion
    max_score: float                # the highest mark the criterion can earn
    feedback: str                   # the student-facing feedback (concise)
    feedback_detailed: str = ""     # the fuller explanation, when available


class AnnotationPayload(BaseModel):
    """
    Everything the frontend needs to render one annotated answer:
    the image to draw on, the feedback items with their regions, and (for essays)
    the per-criterion scorecard shown in the panel.
    """

    image: ImageMeta
    items: List[AnnotationItem]
    scorecard: List[CriterionScore] = Field(default_factory=list)
